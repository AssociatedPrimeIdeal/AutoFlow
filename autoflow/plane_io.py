import json
import os

import h5py
import numpy as np

from .core.models import PlaneData


def _normalize(v):
    arr = np.asarray(v, dtype=float).reshape(3)
    n = np.linalg.norm(arr)
    if n <= 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return arr / n


def _path_cumdist(path):
    pts = np.asarray(path, dtype=float).reshape(-1, 3)
    if len(pts) <= 1:
        return np.zeros(len(pts), dtype=float)
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])


def _json_safe(value):
    return json.loads(json.dumps(value, ensure_ascii=False))


def _generic_label_name(label_value):
    return f"Label {int(label_value)}"


def _is_generic_label_name(name, label_value):
    text = str(name or "").strip()
    if not text:
        return True
    label_value = int(label_value)
    return text in {
        _generic_label_name(label_value),
        f"label_{label_value}",
        f"label {label_value}",
    }


def _configured_label_name(ws, label_value):
    label_value = int(label_value)
    for attr_name in ("label_params", "skeleton_params"):
        params = getattr(ws, attr_name, None)
        label_map = getattr(params, "label_map", {}) if params is not None else {}
        for name, value in dict(label_map or {}).items():
            try:
                if int(value) == label_value:
                    return str(name)
            except Exception:
                continue
    return ""


def _segmentation_label_name(ws, label_value):
    seg = getattr(ws, "segmentation", None)
    if seg is None:
        return ""
    label_names = getattr(seg, "label_names", {}) or {}
    return str(label_names.get(str(int(label_value)), "") or "")


def _majority_vote_labels_3d(labels):
    seg = np.asarray(labels, dtype=np.int16)
    if seg.ndim == 3:
        return seg
    if seg.ndim != 4:
        return None
    values = [int(x) for x in np.unique(seg)]
    counts = np.stack([(seg == int(value)).sum(axis=3) for value in values], axis=-1)
    winners = np.argmax(counts, axis=-1)
    out = np.zeros(seg.shape[:3], dtype=np.int16)
    for idx, value in enumerate(values):
        out[winners == idx] = int(value)
    return out


def _segmentation_labels_3d(ws):
    labels_3d = getattr(ws, "segmask_labels_3d", None)
    if labels_3d is not None:
        return np.asarray(labels_3d, dtype=np.int16)
    labels = getattr(ws, "segmask_labels", None)
    if labels is None:
        labels = getattr(ws, "segmask_raw", None)
    if labels is None:
        return None
    return _majority_vote_labels_3d(labels)


def _group_labels_for_plane(ws, plane):
    group_name = str(getattr(plane, "group_name", "") or "")
    groups = getattr(ws, "multilabel_groups", {}) or {}
    group_state = groups.get(group_name, {}) if group_name else {}
    labels = []
    if isinstance(group_state, dict):
        for value in list(group_state.get("labels", []) or []):
            try:
                labels.append(int(value))
            except Exception:
                continue
    return labels


def _sample_label_neighborhood(label_volume, ijk, allowed_labels, radius):
    shape = np.asarray(label_volume.shape, dtype=int)
    ijk = np.clip(np.asarray(ijk, dtype=int).reshape(3), 0, shape - 1)
    lo = np.maximum(ijk - int(radius), 0)
    hi = np.minimum(ijk + int(radius) + 1, shape)
    patch = np.asarray(
        label_volume[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]],
        dtype=np.int16,
    ).reshape(-1)
    patch = patch[patch > 0]
    if patch.size == 0:
        return 0
    if allowed_labels:
        patch = patch[np.isin(patch, np.asarray(list(allowed_labels), dtype=np.int16))]
    if patch.size == 0:
        return 0
    values, counts = np.unique(patch, return_counts=True)
    return int(values[np.argmax(counts)])


def _plane_source_label_value(ws, plane, label_volume=None):
    group_labels = tuple(_group_labels_for_plane(ws, plane))
    if label_volume is None:
        label_volume = _segmentation_labels_3d(ws)
    if label_volume is not None:
        resolution = np.asarray(getattr(ws, "resolution", [1.0, 1.0, 1.0]), dtype=float).reshape(3)
        center = np.asarray(getattr(plane, "center", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
        for radius in (0, 1, 2):
            ijk = np.rint(center / (resolution + 1e-12)).astype(int)
            label_value = _sample_label_neighborhood(label_volume, ijk, group_labels, radius)
            if label_value > 0:
                return int(label_value)
    if len(group_labels) == 1:
        return int(group_labels[0])
    plane_label = int(getattr(plane, "label", 0) or 0)
    if _configured_label_name(ws, plane_label):
        return plane_label
    return 0


def _plane_label_name(ws, plane, label_volume=None):
    label_value = _plane_source_label_value(ws, plane, label_volume=label_volume)
    if label_value <= 0:
        return ""
    configured_name = _configured_label_name(ws, label_value)
    if configured_name:
        return configured_name
    seg_name = _segmentation_label_name(ws, label_value)
    if seg_name and not _is_generic_label_name(seg_name, label_value):
        return seg_name
    if seg_name:
        return seg_name
    return f"label_{label_value}"


def _plane_record(ws, plane_index, label_volume=None, path_info=None):
    origin = np.asarray(ws.origin, dtype=float).reshape(3)
    plane = ws.planes[int(plane_index)]
    center_local = np.asarray(plane.center, dtype=float).reshape(3)
    record = {
        "plane_index": int(plane_index),
        "center": center_local.tolist(),
        "center_world": (center_local + origin).tolist(),
        "normal": _normalize(plane.normal).tolist(),
        "label": int(plane.label),
        "label_name": _plane_label_name(ws, plane, label_volume=label_volume),
        "path_index": int(plane.path_index),
        "distance": float(plane.distance),
        "group_name": str(getattr(plane, "group_name", "") or ""),
        "placement_mode": "manual" if int(plane.path_index) < 0 else "path",
    }
    if plane.metrics:
        record.update(_json_safe(plane.metrics))
    if path_info is None:
        path_info = list(getattr(ws, "path_info", []) or [])
    if 0 <= int(plane.path_index) < len(path_info):
        record["path_info"] = _json_safe(path_info[int(plane.path_index)])
    return record


def build_plane_records(ws):
    label_volume = _segmentation_labels_3d(ws)
    path_info = list(getattr(ws, "path_info", []) or [])
    return [
        _plane_record(ws, idx, label_volume=label_volume, path_info=path_info)
        for idx in range(len(ws.planes))
    ]


def _make_plane_payload(ws, source_path=""):
    origin = np.asarray(ws.origin, dtype=float).reshape(3)
    payload = {
        "source": source_path,
        "origin": origin.tolist(),
        "resolution": np.asarray(ws.resolution, dtype=float).reshape(3).tolist(),
        "planes": [],
    }
    for i, plane in enumerate(ws.planes):
        center_local = np.asarray(plane.center, dtype=float).reshape(3)
        payload["planes"].append(
            {
                "plane_index": int(i),
                "center": center_local.tolist(),
                "center_world": (center_local + origin).tolist(),
                "normal": _normalize(plane.normal).tolist(),
                "label": int(plane.label),
                "path_index": int(plane.path_index),
                "distance": float(plane.distance),
                "placement_mode": "manual" if int(plane.path_index) < 0 else "path",
            }
        )
    return payload


def save_plane_positions(ws, out_path, source_path=""):
    payload = _make_plane_payload(ws, source_path=source_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def save_planes_json(ws, out_path, source_path="", payload=None):
    if payload is None:
        payload = build_plane_records(ws)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def _write_h5_string(group, name, value):
    return group.create_dataset(name, data=np.asarray(str(value), dtype=h5py.string_dtype(encoding="utf-8")))


def _write_h5_json(group, name, value):
    ds = _write_h5_string(group, name, json.dumps(value, ensure_ascii=False))
    ds.attrs["encoding"] = "json"
    return ds


def _write_h5_sequence(group, name, value):
    items = list(value)
    if not items:
        return _write_h5_json(group, name, items)
    if all(isinstance(item, dict) for item in items):
        sub = group.create_group(name)
        sub.attrs["container"] = "list[dict]"
        for idx, item in enumerate(items):
            child = sub.create_group(f"item_{idx:04d}")
            for key, child_value in sorted(item.items(), key=lambda kv: str(kv[0])):
                _write_h5_value(child, str(key), child_value)
        return sub
    try:
        arr = np.asarray(items)
    except Exception:
        return _write_h5_json(group, name, items)
    if arr.dtype.kind in {"i", "u", "f", "b"}:
        return group.create_dataset(name, data=arr)
    if arr.dtype.kind in {"U", "S"}:
        return group.create_dataset(name, data=np.asarray(items, dtype=h5py.string_dtype(encoding="utf-8")))
    return _write_h5_json(group, name, items)


def _write_h5_value(group, name, value):
    if isinstance(value, dict):
        sub = group.create_group(name)
        sub.attrs["container"] = "dict"
        for key, child_value in sorted(value.items(), key=lambda kv: str(kv[0])):
            _write_h5_value(sub, str(key), child_value)
        return sub
    if isinstance(value, np.ndarray):
        arr = np.asarray(value)
        if arr.dtype.kind in {"U", "S"}:
            return group.create_dataset(name, data=np.asarray(arr, dtype=h5py.string_dtype(encoding="utf-8")))
        if arr.dtype.kind in {"i", "u", "f", "b"}:
            return group.create_dataset(name, data=arr)
        return _write_h5_json(group, name, arr.tolist())
    if isinstance(value, (list, tuple)):
        return _write_h5_sequence(group, name, value)
    if value is None:
        return _write_h5_json(group, name, None)
    if isinstance(value, (str, bytes, np.str_)):
        return _write_h5_string(group, name, value.decode("utf-8") if isinstance(value, bytes) else value)
    if isinstance(value, (bool, np.bool_)):
        return group.create_dataset(name, data=np.asarray(bool(value), dtype=np.bool_))
    if isinstance(value, (int, np.integer)):
        return group.create_dataset(name, data=np.asarray(int(value), dtype=np.int64))
    if isinstance(value, (float, np.floating)):
        return group.create_dataset(name, data=np.asarray(float(value), dtype=np.float64))
    return _write_h5_json(group, name, value)


def save_planes_h5(ws, out_path, source_path="", payload=None):
    if payload is None:
        payload = build_plane_records(ws)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as h5:
        h5.attrs["schema"] = "autoflow.planes.v1"
        h5.attrs["source"] = str(source_path or "")
        h5.attrs["source_format"] = str(getattr(getattr(ws, "input_state", None), "source_format", "") or "")
        h5.attrs["origin"] = np.asarray(ws.origin, dtype=float).reshape(3)
        h5.attrs["resolution"] = np.asarray(ws.resolution, dtype=float).reshape(3)
        h5.attrs["rr_ms"] = float(getattr(ws, "rr", 0.0) or 0.0)
        h5.attrs["plane_count"] = int(len(payload))
        for record in payload:
            plane_index = int(record.get("plane_index", 0))
            group = h5.create_group(f"plane_{plane_index:04d}")
            group.attrs["plane_index"] = plane_index
            _write_h5_json(group, "payload_json", record)
            for key, value in sorted(record.items(), key=lambda kv: str(kv[0])):
                _write_h5_value(group, str(key), value)
    return out_path



def build_pwv_payload(results, source_path="", source_format="", source_group=None):
    return {
        "results": _json_safe(list(results or [])),
        "source": str(source_path or ""),
        "source_format": str(source_format or ""),
        "source_group": None if source_group is None else str(source_group),
    }


def save_pwv_h5(results, out_path, source_path="", source_format="", source_group=None):
    payload = build_pwv_payload(results, source_path=source_path, source_format=source_format, source_group=source_group)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as h5:
        h5.attrs["schema"] = "autoflow.pwv.v1"
        h5.attrs["source"] = str(source_path or "")
        h5.attrs["source_format"] = str(source_format or "")
        if source_group is not None:
            h5.attrs["source_group"] = str(source_group)
        results_list = list(payload.get("results", []) or [])
        h5.attrs["group_count"] = int(len(results_list))
        _write_h5_json(h5, "payload_json", payload)
        for idx, record in enumerate(results_list):
            name = str(record.get("name", f"group_{idx:04d}") or f"group_{idx:04d}")
            group = h5.create_group(f"group_{idx:04d}")
            group.attrs["group_index"] = int(idx)
            group.attrs["name"] = name
            _write_h5_json(group, "payload_json", record)
            for key, value in sorted(record.items(), key=lambda kv: str(kv[0])):
                _write_h5_value(group, str(key), value)
    return out_path

def load_plane_positions(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "planes" in payload:
        return payload["planes"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Invalid plane position file: {path}")


def _nearest_path_info(center_world, paths_world):
    best_dist = np.inf
    best_path_idx = -1
    best_point_idx = -1
    best_distance = 0.0
    for path_idx, path in enumerate(paths_world):
        pts = np.asarray(path, dtype=float).reshape(-1, 3)
        if len(pts) == 0:
            continue
        d = np.linalg.norm(pts - center_world.reshape(1, 3), axis=1)
        point_idx = int(np.argmin(d))
        dist = float(d[point_idx])
        if dist < best_dist:
            cum = _path_cumdist(pts)
            best_dist = dist
            best_path_idx = int(path_idx)
            best_point_idx = point_idx
            best_distance = float(cum[point_idx]) if len(cum) > point_idx else 0.0
    return best_path_idx, best_point_idx, best_dist, best_distance


def _path_tangent(path_world, point_idx):
    pts = np.asarray(path_world, dtype=float).reshape(-1, 3)
    if len(pts) == 0:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    i0 = max(0, int(point_idx) - 1)
    i1 = min(len(pts) - 1, int(point_idx) + 1)
    if i1 == i0:
        i1 = min(len(pts) - 1, i0 + 1)
    tangent = pts[i1] - pts[i0]
    return _normalize(tangent)


def project_planes_to_workspace(plane_items, ws):
    origin = np.asarray(ws.origin, dtype=float).reshape(3)
    paths_local = ws.centerline_paths_smooth if len(ws.centerline_paths_smooth) > 0 else ws.centerline_paths
    paths_world = [np.asarray(path, dtype=float).reshape(-1, 3) + origin.reshape(1, 3) for path in paths_local]
    planes = []
    for item in plane_items:
        if "center_world" in item:
            center_world = np.asarray(item["center_world"], dtype=float).reshape(3)
        elif "center" in item:
            center_world = np.asarray(item["center"], dtype=float).reshape(3)
        else:
            continue
        normal = _normalize(item.get("normal", [1.0, 0.0, 0.0]))
        path_index = int(item.get("path_index", -1))
        distance = float(item.get("distance", 0.0))
        manual_placement = str(item.get("placement_mode", "")).strip().lower() == "manual"
        if paths_world and not manual_placement:
            nearest_path_idx, nearest_point_idx, _, nearest_distance = _nearest_path_info(center_world, paths_world)
            if nearest_path_idx >= 0:
                path_index = int(nearest_path_idx)
                distance = float(nearest_distance)
                if np.linalg.norm(normal) <= 1e-12:
                    normal = _path_tangent(paths_world[path_index], nearest_point_idx)
        if path_index < 0 and not manual_placement:
            path_index = 0
        planes.append(
            PlaneData(
                center=center_world - origin,
                normal=_normalize(normal),
                label=int(item.get("label", int(path_index) + 1)),
                path_index=int(path_index),
                distance=float(distance),
                group_name=str(item.get("group_name", "") or ""),
            )
        )
    return planes


def resolve_reuse_plane_file(reuse_spec, case_name):
    if not reuse_spec:
        return ""
    if os.path.isfile(reuse_spec):
        return reuse_spec
    if os.path.isdir(reuse_spec):
        candidates = [
            os.path.join(reuse_spec, case_name, "plane_positions.json"),
            os.path.join(reuse_spec, case_name, "planes.json"),
            os.path.join(reuse_spec, "plane_positions.json"),
            os.path.join(reuse_spec, "planes.json"),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
    return reuse_spec
