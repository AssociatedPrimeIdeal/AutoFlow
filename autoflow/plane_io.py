import json
import os

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
            }
        )
    return payload


def save_plane_positions(ws, out_path, source_path=""):
    payload = _make_plane_payload(ws, source_path=source_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
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
        if paths_world:
            nearest_path_idx, nearest_point_idx, _, nearest_distance = _nearest_path_info(center_world, paths_world)
            if nearest_path_idx >= 0:
                path_index = int(nearest_path_idx)
                distance = float(nearest_distance)
                if np.linalg.norm(normal) <= 1e-12:
                    normal = _path_tangent(paths_world[path_index], nearest_point_idx)
        if path_index < 0:
            path_index = 0
        planes.append(
            PlaneData(
                center=center_world - origin,
                normal=_normalize(normal),
                label=int(path_index) + 1,
                path_index=int(path_index),
                distance=float(distance),
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
