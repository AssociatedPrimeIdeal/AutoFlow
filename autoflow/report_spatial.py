"""Bounded, read-only spatial evidence for standalone case reports.

Coordinates match Workspace paths/planes: local physical millimetres.  The
segmentation layer is sampled from real boundary voxels, never a new analysis.
"""
from collections import Counter
import copy
import hashlib
import json

import numpy as np

from .core.plane_state import metric_validity, selected_metric
from .qc_locations import iter_issues, qc_status
from .task_control import check_cancelled

DEFAULT_LIMITS = {
    "max_segmentation_points": 8000, "max_skeleton_points": 4000,
    "max_path_points": 12000, "max_paths": 256, "max_markers": 2048,
    "max_planes": 1024, "max_labels": 128,
}
STATUSES = ("pass", "warn", "fail", "missing", "stale", "not_run", "unsupported", "unknown")


def _point(value):
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    return arr.tolist() if arr.size == 3 and np.all(np.isfinite(arr)) else None


def _integers(values, *, positive=False):
    result = []
    for value in values if isinstance(values, (list, tuple, np.ndarray)) else []:
        try:
            number = int(value)
            if isinstance(value, (float, np.floating)) and (not np.isfinite(value) or number != value):
                continue
        except (TypeError, ValueError, OverflowError):
            continue
        if number >= (1 if positive else 0) and number not in result:
            result.append(number)
    return result


def _paths(ws):
    smooth = getattr(ws, "centerline_paths_smooth", [])
    return (smooth, "centerline_paths_smooth") if len(smooth) else (getattr(ws, "centerline_paths", []), "centerline_paths")


def _path_labels(ws, path_id):
    labels, sources = [], []
    info = getattr(ws, "path_info", [])
    if 0 <= path_id < len(info):
        for key in ("owner_label", "segmentation_label"):
            found = _integers([info[path_id].get(key)], positive=True)
            if found:
                labels.extend(found)
                sources.append("path_info." + key)
        group_name = str(info[path_id].get("group_name", "") or "")
    else:
        group_name = ""
    for plane in ws.planes:
        if int(getattr(plane, "path_index", -1)) == path_id:
            found = _integers([getattr(plane, "segmentation_label", 0)], positive=True)
            if found:
                labels.extend(found)
                sources.append("plane.segmentation_label")
    group = (getattr(ws, "multilabel_groups", {}) or {}).get(group_name, {})
    found = _integers(group.get("labels", []), positive=True)
    if found:
        labels.extend(found)
        sources.append("multilabel_groups.labels")
    return sorted(set(labels)), group_name, sorted(set(sources))


def _label_definitions(ws, observed_ids):
    candidates = {}
    # Explicit user names take precedence; aliases remain visible with sources.
    custom = getattr(getattr(ws, "segmentation", None), "label_names", {}) or {}
    for value, name in custom.items():
        ids = _integers([value], positive=True)
        text = str(name or "").strip()
        if ids and text and text.casefold() != f"label {ids[0]}".casefold():
            candidates.setdefault(ids[0], []).append((text, "segmentation.label_names"))
    for field in ("label_params", "skeleton_params"):
        mapping = getattr(getattr(ws, field, None), "label_map", {}) or {}
        for name, value in mapping.items():
            ids = _integers([value], positive=True)
            if ids and str(name).strip():
                candidates.setdefault(ids[0], []).append((str(name), field + ".label_map"))
    rows = []
    for label_id in sorted(set(observed_ids)):
        choices = candidates.get(label_id, [])
        name, source = choices[0] if choices else (
            ("Unlabeled foreground", "binary_foreground") if label_id == 0
            else (f"Label {label_id}", "numeric_fallback"))
        rows.append({"label_id": label_id, "name": name, "name_source": source,
                     "aliases": sorted({text for text, _source in choices if text != name}),
                     "configured_names": [{"name": text, "source": origin} for text, origin in choices]})
    return rows


def _local_evaluations(ws, signature, selected, max_planes):
    """Explicit small-record evaluations; never change the quality report."""
    result = []
    paths, _ = _paths(ws)
    spacing = _point(ws.resolution)
    offset_limit = max(2., 2 * float(np.mean(spacing))) if spacing and min(spacing) > 0 else None
    phase_count = int(ws.time_count())

    def record(plane, index, family, status, summary, observed, validity="current", phases=()):
        geometry = family == "planes.geometry"
        phase_scope = "static_geometry" if geometry else ("phases" if phases else "whole_cycle")
        scope = {"scope_kind": "plane", "plane_ids": [str(plane.plane_id)],
                 "path_ids": _integers([getattr(plane, "path_index", -1)]),
                 "label_ids": _integers([getattr(plane, "segmentation_label", 0)], positive=True),
                 "phase_scope": phase_scope, "phase_indices": list(phases), "location_method": "plane_center"}
        identity = (f"report.local:{family}:{plane.plane_id}", scope, signature)
        threshold_status = status
        status = qc_status({"status": status})
        result.append({
            "threshold_status": threshold_status, **scope, "id": identity[0], "issue_id": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20],
            "rule_id": family, "source_rule_id": family, "rule_source": "autoflow.qc_locations.annotate_checks",
            "rule_version": 2, "record_type": "report_local_evaluation",
            "dependency_signature": signature, "validity": validity, "status": status,
            "title": f"Plane {index} geometry" if geometry else f"Plane {index} sampling validity",
            "summary": summary, "observed": observed, "unit": "deg / mm" if geometry else "mm2 / ratio",
            "threshold": {"angle_warn_deg": 20., "angle_fail_deg": 45., "offset_warn_mm": offset_limit}
                         if geometry else {"all_samples_valid": True, "area_cv_warn": .2},
            "evidence_ref": f"planes/{plane.plane_id}/" + ("geometry" if geometry else "metrics"),
            "action": "Review the plane and its source evidence." if status != "pass" else "",
            "required": False,
        })

    included = 0
    for index, plane in enumerate(ws.planes):
        if selected is not None and plane.plane_id not in selected:
            continue
        if included >= max_planes:
            break
        included += 1
        check_cancelled()
        center, normal = _point(plane.center), _point(plane.normal)
        draft = any((getattr(plane, "placement", {}) or {}).get(k) for k in ("_draft", "_roi_draft"))
        normal_length = float(np.linalg.norm(normal)) if normal is not None else np.nan
        normal_ok = bool(np.isfinite(normal_length) and normal_length > 1e-12)
        observed = {"center_finite": center is not None, "normal_finite_nonzero": bool(normal_ok)}
        status, summary = "missing", "No usable centerline reference for the local alignment check."
        if draft:
            status, summary = "stale", "Uncommitted plane or ROI draft; previous evidence is not current."
        elif center is None or not normal_ok:
            status, summary = "fail", "Non-finite center or invalid plane normal."
        elif offset_limit is None:
            status, summary = "missing", "Finite positive spacing is unavailable for the offset check."
        elif 0 <= int(plane.path_index) < len(paths):
            path = np.asarray(paths[int(plane.path_index)], dtype=float)
            if path.ndim == 2 and path.shape[1] == 3 and len(path) > 1 and np.all(np.isfinite(path)):
                center_arr, normal_arr = np.asarray(center), np.asarray(normal)
                nearest = int(np.argmin(np.linalg.norm(path - center_arr, axis=1)))
                tangent = path[min(nearest + 1, len(path) - 1)] - path[max(0, nearest - 1)]
                length = float(np.linalg.norm(tangent))
                if np.isfinite(length) and length > 1e-12:
                    angle = float(np.degrees(np.arccos(np.clip(abs(np.dot(normal_arr, tangent)
                                                 / np.linalg.norm(normal_arr) / length), 0., 1.))))
                    offset = float(np.linalg.norm(path[nearest] - center_arr))
                    status = ("fail" if angle > 45. else ("warn" if angle > 20. or offset > offset_limit else "pass")) if np.isfinite(angle) and np.isfinite(offset) else "missing"
                    observed.update(angle_deg=angle, offset_mm=offset, reference_path_id=int(plane.path_index))
                    summary = f"Evaluated normal angle {angle:.2f} deg and centerline offset {offset:.2f} mm."
        record(plane, index, "planes.geometry", status, summary, observed,
               validity="stale" if draft else ("missing" if status == "missing" else "current"))

        raw = selected_metric(ws, index, current_only=False)
        state = metric_validity(plane, raw)
        if state != "current":
            record(plane, index, "hemodynamics.plane_metrics", state,
                   "Previous result / needs recalculation." if state == "stale" else "Plane sampling has not been calculated.",
                   {"metric_validity": state}, validity=state)
            continue
        metric = selected_metric(ws, index) or {}
        inferred = [t for t, mode in enumerate(metric.get('roi_resolution_t', []))
                    if mode in {'interpolated', 'single_source_hold'}]
        if inferred:
            record(plane, index, 'hemodynamics.temporal_roi', 'warn',
                   'Missing observed contours use interpolation or a single available shape; review source phases.',
                   {'phases': inferred, 'source_frames': metric.get('roi_source_frames_t', []),
                    'resolution': metric.get('roi_resolution_t', [])}, phases=inferred)
        try:
            areas = np.asarray(metric.get("area_mm2", []), dtype=float)
            if areas.ndim != 1 or areas.size == 0 or areas.size != phase_count:
                raise ValueError("Area time series does not match the current phase count.")
            raw_valid = metric.get("sample_valid_t")
            valid = np.isfinite(areas) & (areas > 0)
            if raw_valid is not None:
                flags = np.asarray(raw_valid, dtype=bool)
                if flags.shape != areas.shape:
                    raise ValueError("Sample validity and area counts differ.")
                valid &= flags
            invalid = np.flatnonzero(~valid).tolist()
            cv = float(np.std(areas) / np.mean(areas)) if np.all(valid) else None
            if cv is not None and not np.isfinite(cv):
                raise ValueError("Area variability could not be evaluated as a finite value.")
            status = "fail" if invalid else ("warn" if cv > .2 else "pass")
            record(plane, index, "hemodynamics.plane_metrics", status,
                   f"Evaluated {len(areas)} phases; invalid phases {invalid}; area CV {cv if cv is not None else 'unavailable'}.",
                   {"invalid_phases": invalid, "area_cv": cv, "phase_count": len(areas),
                    "validity_source": "sample_valid_t_and_area" if raw_valid is not None else "finite_positive_area"},
                   phases=invalid)
        except (TypeError, ValueError) as exc:
            record(plane, index, "hemodynamics.plane_metrics", "missing", str(exc),
                   {"metric_validity": state, "phase_count": phase_count}, validity="missing")
    return result


def _boundary_slabs(volume, labelled, digest):
    """Read one 2D slab at a time; never materialize a 4D mask or coordinate grid."""
    for x in range(volume.shape[0]):
        check_cancelled()
        values = np.asarray(volume[x])
        digest.update(np.ascontiguousarray(values).tobytes())
        positive = np.isfinite(values) & (values > 0)
        interior = positive.copy()
        if x == 0 or x == volume.shape[0] - 1:
            interior[:] = False
        else:
            previous, following = np.asarray(volume[x - 1]), np.asarray(volume[x + 1])
            interior &= np.isfinite(previous) & (previous > 0)
            interior &= np.isfinite(following) & (following > 0)
        interior[0, :] = False
        interior[-1, :] = False
        interior[:, 0] = False
        interior[:, -1] = False
        interior[1:, :] &= positive[:-1, :]
        interior[:-1, :] &= positive[1:, :]
        interior[:, 1:] &= positive[:, :-1]
        interior[:, :-1] &= positive[:, 1:]
        yz = np.argwhere(positive & ~interior)
        points = np.column_stack((np.full(len(yz), x, dtype=np.int64), yz))
        if labelled and len(yz):
            labels = values[yz[:, 0], yz[:, 1]]
            valid_ids = (labels == np.rint(labels)) & (labels <= np.iinfo(np.int32).max)
            labels = np.where(valid_ids, labels, 0).astype(np.int64)
        else:
            labels = np.zeros(len(yz), dtype=np.int64)
        yield points, labels, int(np.count_nonzero(positive))


def _segmentation_context(ws, spacing, limits):
    volume, field, phase_scope, phase_indices, labelled = None, None, "static_aggregate", [], False
    for name in ("segmentation_analysis_labels_3d", "segmentation_analysis_mask", "segmask_labels_3d", "segmask_3d"):
        value = getattr(ws, name, None)
        if value is not None and np.asarray(value).ndim == 3:
            volume, field, labelled = np.asarray(value), name, name in {"segmentation_analysis_labels_3d", "segmask_labels_3d"}
            if name.startswith("segmentation_analysis"):
                spacing = np.ones(3)
            break
    if volume is None:
        for name in ("segmask_labels", "segmask_binary", "segmask_raw"):
            value = getattr(ws, name, None)
            if value is None:
                continue
            array = np.asarray(value)
            if array.ndim == 4 and array.shape[3]:
                phase = min(max(int(getattr(ws, "current_t", 0)), 0), array.shape[3] - 1)
                volume, field, phase_scope, phase_indices = array[..., phase], name, "phase", [phase]
                labelled = name != "segmask_binary"
                break
            if array.ndim == 3:
                volume, field, phase_scope = array, name, "static_source"
                labelled = name != "segmask_binary"
                break
    result = {"status": "missing", "representation": "boundary_voxel_centers", "points_mm": [], "label_ids": [],
              "observed_label_ids": [], "source": field, "phase_scope": phase_scope, "phase_indices": phase_indices,
              "display_sampled": False, "boundary_point_count": 0, "foreground_voxel_count": 0}
    if volume is None:
        result["note"] = "No existing segmentation array is available."
        return result
    if volume.dtype.kind not in "biuf" or any(size == 0 for size in volume.shape):
        result.update(status="unsupported", note="Unsupported or empty segmentation array layout.")
        return result
    first_digest, second_digest = hashlib.sha256(), hashlib.sha256()
    count, foreground, extrema, label_counts = 0, 0, {}, Counter()
    labels_limited = False
    for points, labels, occupied in _boundary_slabs(volume, labelled, first_digest):
        count += len(points)
        foreground += occupied
        for value in np.unique(labels):
            label_id = int(value)
            if label_id not in label_counts and len(label_counts) >= limits["max_labels"]:
                labels_limited = True
                continue
            chosen = points[labels == value]
            label_counts[label_id] += len(chosen)
            anchors = [chosen[int(np.argmin(chosen[:, axis]))] for axis in range(3)]
            anchors += [chosen[int(np.argmax(chosen[:, axis]))] for axis in range(3)]
            if label_id not in extrema:
                extrema[label_id] = [p.copy() for p in anchors]
            else:
                for axis in range(3):
                    if anchors[axis][axis] < extrema[label_id][axis][axis]:
                        extrema[label_id][axis] = anchors[axis].copy()
                    if anchors[axis + 3][axis] > extrema[label_id][axis + 3][axis]:
                        extrema[label_id][axis + 3] = anchors[axis + 3].copy()
    selected = {}
    for label_id in sorted(extrema):
        for point in extrema[label_id]:
            if len(selected) >= limits["max_segmentation_points"]:
                break
            selected[tuple(map(int, point))] = label_id
    budget = max(0, limits["max_segmentation_points"] - len(selected))
    targets = np.linspace(0, max(0, count - 1), min(budget, count), dtype=np.int64) if budget and count else np.array([], dtype=np.int64)
    offset = 0
    for points, labels, _occupied in _boundary_slabs(volume, labelled, second_digest):
        lo, hi = np.searchsorted(targets, (offset, offset + len(points)))
        for target in targets[lo:hi]:
            index = int(target - offset)
            selected[tuple(map(int, points[index]))] = int(labels[index])
        offset += len(points)
    if offset != count or first_digest.digest() != second_digest.digest():
        raise RuntimeError("Segmentation changed while sampling the report display.")
    ijk = np.asarray(list(selected), dtype=float).reshape(-1, 3)
    displayed_ids = list(selected.values())
    physical_points = ijk * spacing
    if not np.all(np.isfinite(physical_points)):
        raise ValueError("Segmentation display coordinates overflowed finite millimetres.")
    result.update(status="available" if count else "empty", points_mm=physical_points.tolist(), label_ids=displayed_ids,
                  observed_label_ids=sorted(label_counts), label_boundary_counts=dict(label_counts),
                  display_sampled=len(selected) < count, boundary_point_count=count, foreground_voxel_count=foreground,
                  source_shape=list(volume.shape), source_dtype=str(volume.dtype), source_data_sha256=first_digest.hexdigest(),
                  labels_limited=labels_limited,
                  sampling_method="real six-neighbor boundary voxel centers; per-label extrema plus uniform traversal samples",
                  note="Display sampling only; no segmentation or measurement arrays are changed.")
    return result


def _status(check):
    validity = check.get("validity")
    if validity in {"stale", "missing"}:
        return validity
    value = qc_status(check)
    return value if value in STATUSES else "unknown"


def _aggregate_status(checks):
    values = {_status(item) for item in checks}
    for status in ("fail", "warn", "stale", "missing", "unsupported", "not_run", "unknown"):
        if status in values:
            return status
    return "pass" if checks and values == {"pass"} else "unknown"


def unavailable_spatial_context(message, *, error_type=None):
    return {"schema": "autoflow.report.spatial.v1", "coordinate_system": "local_mm",
            "availability": "unavailable", "paths": [], "surfaces": [], "planes": [], "markers": [],
            "skeleton": {"points_mm": [], "status": "unavailable"},
            "segmentation": {"points_mm": [], "label_ids": [], "status": "unavailable"},
            "global_issue_ids": [], "unlocated_issue_ids": [], "unlocated_checks": [],
            "local_evaluations": [], "label_definitions": [], "bounds_mm": None,
            "notes": [str(message)], "error_type": error_type, "counts": {}, "limits": dict(DEFAULT_LIMITS)}


def build_spatial_context(ws, quality_report, *, selected_plane_ids=None, limits=None):
    """Build bounded JSON-sized geometry without VTK, rendering, or analysis."""
    resolved = dict(DEFAULT_LIMITS)
    if limits:
        for key, maximum in DEFAULT_LIMITS.items():
            if key in limits:
                resolved[key] = max(1, min(int(limits[key]), maximum))
    origin, spacing = _point(ws.origin), _point(ws.resolution)
    if origin is None or spacing is None or min(spacing) <= 0:
        return unavailable_spatial_context("Finite origin and positive voxel spacing are required for spatial evidence.")
    spacing_array = np.asarray(spacing)
    selected = None if selected_plane_ids is None else set(selected_plane_ids)
    context = unavailable_spatial_context("")
    context.update(availability="available", origin_mm=origin, spacing_mm=spacing, limits=resolved, notes=[],
                   dependency_signature=quality_report["dependency_signature"],
                   coordinate_note="Workspace local physical mm; scene coordinates = local_mm + origin_mm. No additional scanner-axis reorientation.",
                   phase_count=int(ws.time_count()), geometry_is_static=True)
    segmentation = _segmentation_context(ws, spacing_array, resolved)
    context["segmentation"] = segmentation
    observed_labels = set(segmentation["observed_label_ids"])
    paths, path_source = _paths(ws)
    path_rows, valid_segments = [], []
    invalid_path_points = 0
    for path_id, raw in enumerate(paths[:resolved["max_paths"]]):
        check_cancelled()
        array = np.asarray(raw, dtype=float)
        if array.ndim != 2 or array.shape[1] != 3:
            invalid_path_points += int(array.size > 0)
            continue
        finite = np.all(np.isfinite(array), axis=1)
        invalid_path_points += int(np.count_nonzero(~finite))
        edges = np.flatnonzero(np.diff(np.r_[False, finite, False].astype(np.int8)))
        label_ids, group, label_sources = _path_labels(ws, path_id)
        observed_labels.update(label_ids)
        for segment_index, (start, stop) in enumerate(zip(edges[::2], edges[1::2])):
            if stop - start:
                valid_segments.append((path_id, segment_index, array[start:stop], label_ids, group, label_sources))
    total_path_points = sum(len(item[2]) for item in valid_segments)
    budget = resolved["max_path_points"]
    for segment_no, (path_id, segment_index, array, label_ids, group, label_sources) in enumerate(valid_segments):
        remaining = len(valid_segments) - segment_no
        take = min(len(array), max(1, budget // max(1, remaining)))
        if budget <= 0:
            break
        indices = np.linspace(0, len(array) - 1, take, dtype=np.int64)
        path_rows.append({"path_id": path_id, "segment_index": segment_index, "points_mm": array[indices].tolist(),
                          "label_ids": label_ids, "group_name": group, "label_sources": label_sources, "source": path_source,
                          "original_point_count": len(array), "display_sampled": take < len(array)})
        budget -= take
    context["paths"] = path_rows
    skeleton_raw = getattr(ws, "skeleton_points", None)
    skeleton_source = "skeleton_points"
    if skeleton_raw is None or np.asarray(skeleton_raw).size == 0:
        skeleton_raw = getattr(getattr(ws, "graph", None), "points", [])
        skeleton_source = "graph.points"
    skeleton = np.asarray(skeleton_raw, dtype=float)
    if skeleton.ndim == 2 and skeleton.shape[1] == 3:
        skeleton = skeleton[np.all(np.isfinite(skeleton), axis=1)]
    else:
        skeleton = np.empty((0, 3))
    take = min(len(skeleton), resolved["max_skeleton_points"])
    sample = skeleton[np.linspace(0, len(skeleton) - 1, take, dtype=np.int64)] if take else skeleton
    context["skeleton"] = {"points_mm": sample.tolist(), "source": skeleton_source,
                           "status": "available" if take else "missing", "original_point_count": len(skeleton),
                           "display_sampled": take < len(skeleton)}
    planes = {str(p.plane_id): p for p in ws.planes}
    context["planes"] = [{"plane_id": str(p.plane_id), "center_mm": _point(p.center), "normal": _point(p.normal),
                           "path_id": int(p.path_index), "label_ids": _integers([p.segmentation_label], positive=True)}
                          for p in ws.planes if (selected is None or p.plane_id in selected) and _point(p.center) is not None][:resolved["max_planes"]]
    local = _local_evaluations(ws, quality_report["dependency_signature"], selected, resolved["max_planes"])
    context["local_evaluations"] = local
    all_checks = [copy.deepcopy(item) for item in iter_issues(quality_report)] + local
    marker_map, unlocated, global_ids, unlocated_ids = {}, [], [], []
    graph_points = np.asarray(getattr(getattr(ws, "graph", None), "points", []), dtype=float)
    phase_count = int(ws.time_count())
    for check in all_checks:
        check_cancelled()
        issue_id = str(check.get("issue_id") or check.get("id") or "")
        kind = str(check.get("scope_kind", "global"))
        if check.get("dependency_signature") not in (None, quality_report["dependency_signature"]):
            check["validity"] = "stale"
        check["status"] = _status(check)
        labels = _integers(check.get("label_ids", []), positive=True)
        phases = _integers(check.get("phase_indices", []))
        observed = check.get("observed")
        if not phases and isinstance(observed, dict):
            phases = _integers(observed.get("invalid_phases", []))
        phases = [phase for phase in phases if phase < phase_count]
        check["phase_indices"] = phases
        if phases and check.get("phase_scope") != "static_geometry":
            check["phase_scope"] = "phases"
        anchors = []
        if check.get("location_method") == "graph_cycle_center":
            point = _point(check.get("center_local_mm"))
            if point is not None:
                anchors.append((str(check.get("cycle_id") or issue_id), point, "graph_cycle_center", True))
        elif kind == "plane":
            for uid in check.get("plane_ids", []):
                plane = planes.get(str(uid))
                if plane is None or (selected is not None and plane.plane_id not in selected):
                    continue
                labels.extend(_integers([getattr(plane, "segmentation_label", 0)], positive=True))
                point = _point(plane.center)
                if point is not None:
                    anchors.append((str(uid), point, "plane_center", False))
        elif kind == "path":
            for path_id in _integers(check.get("path_ids", [])):
                if path_id >= len(paths):
                    continue
                array = np.asarray(paths[path_id], dtype=float)
                if array.ndim != 2 or array.shape[1] != 3 or not len(array) or not np.all(np.isfinite(array)):
                    continue
                arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(array, axis=0), axis=1))]
                point = [float(np.interp(arc[-1] / 2., arc, array[:, axis])) for axis in range(3)]
                labels.extend(_path_labels(ws, path_id)[0])
                if _point(point) is not None:
                    anchors.append((str(path_id), point, "path_arc_midpoint", True))
        elif kind == "fork":
            nodes = _integers([check.get("node_id")])
            if nodes and graph_points.ndim == 2 and graph_points.shape[1] == 3 and nodes[0] < len(graph_points):
                point = _point(graph_points[nodes[0]])
                if point is not None:
                    anchors.append((str(nodes[0]), point, "fork_node", False))
        check["label_ids"] = sorted(set(labels))
        observed_labels.update(labels)
        if not anchors:
            if check["status"] != "pass":
                reason = "global_scope" if kind == "global" else "No current finite anchor exists for this typed scope."
                unlocated.append({**check, "unlocated_reason": reason})
                (global_ids if kind == "global" else unlocated_ids).append(issue_id)
            continue
        for entity, point, method, representative in anchors:
            key = (kind, entity, tuple(point))
            marker = marker_map.setdefault(key, {
                "marker_id": hashlib.sha256(json.dumps(key).encode()).hexdigest()[:20],
                "point_mm": point, "position_mm": point, "scope_kind": kind, "kind": kind,
                "location_method": method, "location_is_representative": representative,
                "issue_ids": [], "checks": [], "plane_ids": [], "path_ids": [], "label_ids": [],
            })
            if issue_id not in marker["issue_ids"]:
                marker["issue_ids"].append(issue_id)
                marker["checks"].append(check)
            marker["plane_ids"] = sorted(set(marker["plane_ids"] + list(check.get("plane_ids", []))))
            marker["path_ids"] = sorted(set(marker["path_ids"] + _integers(check.get("path_ids", []))))
            marker["label_ids"] = sorted(set(marker["label_ids"] + labels))
    markers = list(marker_map.values())
    rank = {"fail": 0, "warn": 1, "stale": 2, "missing": 3, "unsupported": 4, "not_run": 5, "unknown": 6, "pass": 7}
    for marker in markers:
        marker["status"] = _aggregate_status(marker["checks"])
        marker["title"] = f"{marker['scope_kind'].title()} local checks"
        marker["summary"] = " | ".join(item.get("summary", "") for item in marker["checks"])
        marker["phase_indices"] = sorted({phase for item in marker["checks"] for phase in item.get("phase_indices", [])})
        scopes = {item.get("phase_scope", "whole_cycle") for item in marker["checks"]}
        marker["phase_scope"] = next(iter(scopes)) if len(scopes) == 1 else "mixed"
    markers.sort(key=lambda item: (rank[item["status"]], item["marker_id"]))
    omitted_markers = max(0, len(markers) - resolved["max_markers"])
    context.update(markers=markers[:resolved["max_markers"]], unlocated_checks=unlocated,
                   global_issue_ids=list(dict.fromkeys(global_ids)), unlocated_issue_ids=list(dict.fromkeys(unlocated_ids)),
                   label_definitions=_label_definitions(ws, observed_labels))
    collections = [np.asarray(segmentation["points_mm"], dtype=float).reshape(-1, 3), sample]
    collections += [np.asarray(row["points_mm"]) for row in path_rows]
    collections += [np.asarray(row["point_mm"]).reshape(1, 3) for row in context["markers"]]
    finite_collections = [array for array in collections if len(array)]
    if finite_collections:
        bounds = np.vstack([np.vstack((array.min(axis=0), array.max(axis=0))) for array in finite_collections])
        context["bounds_mm"] = [bounds.min(axis=0).tolist(), bounds.max(axis=0).tolist()]
    else:
        context["availability"] = "unavailable"
        context["notes"].append("No finite existing geometry or localized checks are available.")
    context["counts"] = {
        "paths_total": len(paths), "path_segments_displayed": len(path_rows), "path_points_total": total_path_points,
        "path_points_displayed": sum(len(row["points_mm"]) for row in path_rows), "invalid_path_points_omitted": invalid_path_points,
        "skeleton_points_displayed": take, "segmentation_points_displayed": len(segmentation["points_mm"]),
        "markers_total": len(markers), "markers_displayed": len(context["markers"]), "markers_omitted": omitted_markers,
        "planes_total": len(ws.planes), "planes_displayed": len(context["planes"]),
        "marker_statuses": dict(Counter(row["status"] for row in context["markers"])),
        "global_checks_unlocated": len(global_ids), "local_checks_unlocated": len(unlocated_ids),
    }
    context["notes"].append("Green means an explicit local check passed, not whole-vessel or clinical validation.")
    context["notes"].append("Paths and segmentation are existing static spatial context; phase selection does not synthesize moving anatomy.")
    if len(paths) > resolved["max_paths"] or omitted_markers or invalid_path_points:
        context["notes"].append("Display limits or invalid geometry omitted some visual records; original results are unchanged.")
    return context

