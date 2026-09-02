import json
import os
from datetime import datetime, timezone

import numpy as np
from scipy.ndimage import label as ndi_label


QUALITY_REPORT_SCHEMA = "autoflow.quality.v1"


def _json_value(value):
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _check(check_id, stage, status, title, summary, *, value=None, threshold=None, action="", required=True):
    return {
        "id": str(check_id),
        "stage": str(stage),
        "status": str(status),
        "title": str(title),
        "summary": str(summary),
        "value": _json_value(value),
        "threshold": threshold,
        "action": str(action or ""),
        "required": bool(required),
    }


def _sample_flow(flow, max_spatial_samples=250000):
    arr = np.asarray(flow)
    spatial_count = int(np.prod(arr.shape[:3]))
    stride = max(1, int(np.ceil((spatial_count / max(1, int(max_spatial_samples))) ** (1.0 / 3.0))))
    return arr[::stride, ::stride, ::stride, ...], stride


def _graph_topology(points, edges):
    point_count = int(len(points))
    adjacency = [set() for _ in range(point_count)]
    valid_edges = 0
    for edge in np.asarray(edges, dtype=int).reshape(-1, 2) if len(edges) else []:
        a, b = int(edge[0]), int(edge[1])
        if a == b or not (0 <= a < point_count and 0 <= b < point_count):
            continue
        adjacency[a].add(b)
        adjacency[b].add(a)
        valid_edges += 1
    seen = set()
    components = 0
    for node in range(point_count):
        if node in seen:
            continue
        components += 1
        stack = [node]
        seen.add(node)
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
    degrees = np.asarray([len(neighbors) for neighbors in adjacency], dtype=int)
    cycle_rank = max(0, int(valid_edges - point_count + components)) if point_count else 0
    return {
        "nodes": point_count,
        "edges": int(valid_edges),
        "components": int(components),
        "endpoints": int(np.count_nonzero(degrees == 1)),
        "branch_nodes": int(np.count_nonzero(degrees >= 3)),
        "isolated_nodes": int(np.count_nonzero(degrees == 0)),
        "cycle_rank": int(cycle_rank),
    }


def _path_tangent(path, point_index):
    points = np.asarray(path, dtype=float).reshape(-1, 3)
    lo = max(0, int(point_index) - 1)
    hi = min(len(points) - 1, int(point_index) + 1)
    tangent = points[hi] - points[lo]
    norm = float(np.linalg.norm(tangent))
    return tangent / norm if norm > 1e-12 else np.zeros(3, dtype=float)


def _plane_geometry_stats(workspace):
    paths = workspace.centerline_paths_smooth if len(workspace.centerline_paths_smooth) > 0 else workspace.centerline_paths
    angles = []
    offsets = []
    invalid = []
    for plane_index, plane in enumerate(workspace.planes):
        center = np.asarray(plane.center, dtype=float).reshape(3)
        normal = np.asarray(plane.normal, dtype=float).reshape(3)
        normal_norm = float(np.linalg.norm(normal))
        if not np.all(np.isfinite(center)) or not np.all(np.isfinite(normal)) or normal_norm <= 1e-12:
            invalid.append(int(plane_index))
            continue
        path_index = int(getattr(plane, "path_index", -1))
        if not (0 <= path_index < len(paths)):
            continue
        path = np.asarray(paths[path_index], dtype=float).reshape(-1, 3)
        if len(path) < 2:
            continue
        distances = np.linalg.norm(path - center.reshape(1, 3), axis=1)
        nearest_index = int(np.argmin(distances))
        tangent = _path_tangent(path, nearest_index)
        if np.linalg.norm(tangent) <= 1e-12:
            continue
        cosine = float(np.clip(abs(np.dot(normal / normal_norm, tangent)), 0.0, 1.0))
        angles.append(float(np.degrees(np.arccos(cosine))))
        offsets.append(float(distances[nearest_index]))
    return {
        "invalid_plane_indices": invalid,
        "evaluated_generated_planes": int(len(angles)),
        "max_normal_angle_deg": float(max(angles)) if angles else None,
        "median_normal_angle_deg": float(np.median(angles)) if angles else None,
        "max_centerline_offset_mm": float(max(offsets)) if offsets else None,
        "median_centerline_offset_mm": float(np.median(offsets)) if offsets else None,
    }


def build_quality_report(workspace, source_path="", run_context=None):
    ws = workspace
    checks = []
    flow = getattr(ws, "flow_raw", None)
    mag = getattr(ws, "mag_raw", None)
    resolution = np.asarray(getattr(ws, "resolution", [np.nan] * 3), dtype=float).reshape(-1)[:3]
    geometry_ok = resolution.size == 3 and np.all(np.isfinite(resolution)) and np.all(resolution > 0.0)
    checks.append(_check(
        "input.geometry",
        "Input & QC",
        "pass" if geometry_ok else "fail",
        "Physical geometry",
        f"spacing={resolution.tolist()} mm" if geometry_ok else "Resolution is missing, non-finite, or non-positive.",
        value={"resolution_mm": resolution.tolist(), "origin_mm": np.asarray(getattr(ws, "origin", [0, 0, 0]), dtype=float).reshape(-1)[:3].tolist()},
        threshold="three finite positive spacing values",
        action="Correct resolution and orientation metadata before analysis." if not geometry_ok else "",
    ))

    if flow is None:
        checks.append(_check(
            "input.flow",
            "Input & QC",
            "fail",
            "Velocity data",
            "No velocity field is loaded.",
            action="Load a supported 4D Flow H5 or DICOM case.",
        ))
    else:
        sampled_flow, stride = _sample_flow(flow)
        finite_fraction = float(np.count_nonzero(np.isfinite(sampled_flow)) / max(1, sampled_flow.size))
        finite_status = "pass" if finite_fraction >= 0.999 else ("warn" if finite_fraction >= 0.95 else "fail")
        checks.append(_check(
            "input.flow",
            "Input & QC",
            finite_status,
            "Velocity data",
            f"shape={list(np.asarray(flow).shape)}, finite={finite_fraction:.3%}",
            value={"shape": list(np.asarray(flow).shape), "finite_fraction": finite_fraction, "sample_stride": int(stride)},
            threshold="pass >=99.9%; warn >=95%",
            action="Inspect the loader and exclude or repair non-finite voxels." if finite_status != "pass" else "",
        ))

        venc = np.asarray(getattr(ws, "venc", []), dtype=float).reshape(-1)
        if venc.size >= 3 and np.all(np.isfinite(venc[:3])) and np.all(venc[:3] > 0.0):
            sample = np.asarray(sampled_flow, dtype=float)
            mask = getattr(ws, "segmask_3d", None)
            if mask is not None:
                sampled_mask = np.asarray(mask, dtype=bool)[::stride, ::stride, ::stride]
                sample = sample[sampled_mask]
            ratios = np.abs(sample) / venc[:3].reshape((1,) * (sample.ndim - 1) + (3,))
            finite_ratios = ratios[np.isfinite(ratios)]
            saturation_fraction = float(np.mean(finite_ratios >= 0.95)) if finite_ratios.size else 0.0
            saturation_status = "pass" if saturation_fraction < 0.001 else ("warn" if saturation_fraction < 0.01 else "fail")
            checks.append(_check(
                "input.venc_saturation",
                "Input & QC",
                saturation_status,
                "VENC saturation screening",
                f"{saturation_fraction:.3%} of sampled component values reach at least 95% of VENC.",
                value={"fraction": saturation_fraction, "venc_cm_s": venc[:3].tolist(), "sample_stride": int(stride)},
                threshold="pass <0.1%; warn <1%; screening threshold only",
                action="Inspect phase aliasing and consider anti-alias correction or a higher-VENC acquisition." if saturation_status != "pass" else "",
            ))
        else:
            checks.append(_check(
                "input.venc_saturation",
                "Input & QC",
                "not_run",
                "VENC saturation screening",
                "VENC metadata is unavailable.",
                action="Provide finite positive VENC values to enable saturation screening.",
            ))

    correction = dict((getattr(getattr(ws, "input_state", None), "metadata", {}) or {}).get("background_phase_correction", {}) or {})
    correction_enabled = bool(getattr(getattr(getattr(ws, "loader_params", None), "background_phase_correction", None), "enabled", False))
    checks.append(_check(
        "input.background_phase",
        "Input & QC",
        "pass" if correction and correction_enabled else "not_run",
        "Background phase provenance",
        "Correction metadata is recorded." if correction else "No applied background-phase correction metadata is recorded.",
        value=correction,
        action="Enable and review background-phase correction when stationary-tissue offsets are material." if not correction else "",
        required=False,
    ))

    segmentation = getattr(ws, "segmask_labels_3d", None)
    if segmentation is None:
        segmentation = getattr(ws, "segmask_3d", None)
    if segmentation is None:
        checks.append(_check(
            "segmentation.availability",
            "Segmentation",
            "not_run",
            "Segmentation",
            "No active segmentation is available.",
            action="Import, generate, or review a segmentation before geometry-dependent analysis.",
        ))
    else:
        seg = np.asarray(segmentation)
        binary = seg > 0
        component_labels, component_count = ndi_label(binary)
        component_sizes = np.bincount(component_labels.reshape(-1))[1:]
        retained_components = int(np.count_nonzero(component_sizes > 0))
        seg_status = "fail" if not np.any(binary) else ("warn" if retained_components > 8 else "pass")
        label_values = [int(x) for x in np.unique(seg) if int(x) > 0]
        volume_mm3 = float(np.count_nonzero(binary) * np.prod(resolution)) if geometry_ok else None
        checks.append(_check(
            "segmentation.availability",
            "Segmentation",
            seg_status,
            "Segmentation content",
            f"labels={len(label_values)}, components={retained_components}, volume={volume_mm3:.1f} mm^3" if volume_mm3 is not None else f"labels={len(label_values)}, components={retained_components}",
            value={"labels": label_values, "component_count": retained_components, "volume_mm3": volume_mm3},
            threshold="non-empty; warn when >8 disconnected foreground components",
            action="Review disconnected components and label grouping." if seg_status != "pass" else "",
        ))

        seg4d = getattr(ws, "segmask_labels", None)
        if seg4d is None:
            seg4d = getattr(ws, "segmask_binary", None)
        if seg4d is not None and np.asarray(seg4d).ndim == 4 and np.asarray(seg4d).shape[3] > 1:
            volumes = np.count_nonzero(np.asarray(seg4d) > 0, axis=(0, 1, 2)).astype(float)
            mean_volume = float(np.mean(volumes))
            volume_cv = float(np.std(volumes) / mean_volume) if mean_volume > 0.0 else np.inf
            temporal_status = "pass" if volume_cv <= 0.05 else ("warn" if volume_cv <= 0.15 else "fail")
            checks.append(_check(
                "segmentation.temporal_stability",
                "Segmentation",
                temporal_status,
                "Temporal segmentation stability",
                f"foreground-volume CV={volume_cv:.2%}",
                value={"volume_cv": volume_cv, "voxel_counts": volumes.tolist()},
                threshold="pass <=5%; warn <=15%",
                action="Review time phases with abrupt mask-volume changes." if temporal_status != "pass" else "",
            ))

    graph = getattr(ws, "graph", None)
    graph_points = np.asarray(getattr(graph, "points", []), dtype=float).reshape(-1, 3)
    graph_edges = np.asarray(getattr(graph, "edges", []), dtype=int).reshape(-1, 2)
    if len(graph_points) == 0:
        checks.append(_check(
            "centerline.topology",
            "Centerline & Planes",
            "not_run",
            "Centerline topology",
            "No centerline graph is available.",
            action="Generate and review the skeleton and graph.",
        ))
    else:
        topology = _graph_topology(graph_points, graph_edges)
        topology_status = "warn" if topology["components"] > 1 or topology["cycle_rank"] > 0 or topology["isolated_nodes"] > 0 else "pass"
        checks.append(_check(
            "centerline.topology",
            "Centerline & Planes",
            topology_status,
            "Centerline topology",
            f"components={topology['components']}, endpoints={topology['endpoints']}, branches={topology['branch_nodes']}, cycles={topology['cycle_rank']}",
            value=topology,
            threshold="one connected component, no isolated nodes or graph cycles",
            action="Review short branches, disconnected segments, and loops before trusting downstream planes." if topology_status != "pass" else "",
        ))

    if not list(getattr(ws, "planes", []) or []):
        checks.append(_check(
            "planes.geometry",
            "Centerline & Planes",
            "not_run",
            "Plane geometry",
            "No analysis planes are available.",
            action="Generate or import planes before plane-based analysis.",
        ))
    else:
        plane_stats = _plane_geometry_stats(ws)
        max_angle = plane_stats["max_normal_angle_deg"]
        max_offset = plane_stats["max_centerline_offset_mm"]
        offset_limit = max(2.0, float(np.mean(resolution)) * 2.0) if geometry_ok else 2.0
        if plane_stats["invalid_plane_indices"] or (max_angle is not None and max_angle > 45.0):
            plane_status = "fail"
        elif (max_angle is not None and max_angle > 20.0) or (max_offset is not None and max_offset > offset_limit):
            plane_status = "warn"
        else:
            plane_status = "pass"
        checks.append(_check(
            "planes.geometry",
            "Centerline & Planes",
            plane_status,
            "Plane geometry",
            f"planes={len(ws.planes)}, max normal angle={max_angle if max_angle is not None else 'n/a'} deg, max path offset={max_offset if max_offset is not None else 'n/a'} mm",
            value=plane_stats,
            threshold=f"normal angle <=20 deg; generated-plane center offset <={offset_limit:.3g} mm",
            action="Open the flagged planes, verify the centerline, and adjust center or orientation." if plane_status != "pass" else "",
        ))

    metrics = list(getattr(getattr(ws, "derived", None), "plane_metrics", []) or [])
    if not metrics:
        checks.append(_check(
            "hemodynamics.plane_metrics",
            "Hemodynamics",
            "not_run",
            "Plane metrics",
            "Plane metrics have not been calculated.",
            action="Calculate plane metrics after reviewing plane geometry.",
        ))
    else:
        empty_planes = []
        area_cvs = []
        for plane_index, metric in enumerate(metrics):
            areas = np.asarray(metric.get("area_mm2", []), dtype=float)
            finite_areas = areas[np.isfinite(areas)]
            if finite_areas.size == 0 or float(np.max(finite_areas)) <= 0.0:
                empty_planes.append(int(plane_index))
            elif float(np.mean(finite_areas)) > 0.0:
                area_cvs.append(float(np.std(finite_areas) / np.mean(finite_areas)))
        metric_status = "fail" if empty_planes else ("warn" if area_cvs and max(area_cvs) > 0.20 else "pass")
        checks.append(_check(
            "hemodynamics.plane_metrics",
            "Hemodynamics",
            metric_status,
            "Plane sampling",
            f"metrics={len(metrics)}, empty planes={len(empty_planes)}, max area CV={max(area_cvs) if area_cvs else 0.0:.2%}",
            value={"metric_count": len(metrics), "empty_plane_indices": empty_planes, "max_area_cv": max(area_cvs) if area_cvs else 0.0},
            threshold="no empty plane slices; warn when temporal area CV >20%",
            action="Inspect empty slices and phases with abrupt cross-sectional area changes." if metric_status != "pass" else "",
        ))

        plane_qc = dict(getattr(getattr(ws, "derived", None), "plane_qc", {}) or {})
        ic_values = [float(value) for value in dict(plane_qc.get("path_ic", {}) or {}).values()]
        ic_values.extend(float(value) for value in dict(plane_qc.get("segmentation_label_ic", {}) or {}).values())
        ic_values.extend(float(value) for value in dict(plane_qc.get("fork_ic", {}) or {}).values())
        if ic_values:
            min_ic = float(min(ic_values))
            ic_status = "pass" if min_ic >= 0.8 else ("warn" if min_ic >= 0.6 else "fail")
            checks.append(_check(
                "hemodynamics.flow_consistency",
                "Hemodynamics",
                ic_status,
                "Flow internal consistency",
                f"minimum path/fork consistency={min_ic:.3f}",
                value={"minimum_ic": min_ic, "path_ic": plane_qc.get("path_ic", {}),
                       "segmentation_label_ic": plane_qc.get("segmentation_label_ic", {}),
                       "fork_ic": plane_qc.get("fork_ic", {})},
                threshold="pass >=0.8; warn >=0.6",
                action="Review segmentation, plane ownership, orientation, and temporal flow curves." if ic_status != "pass" else "",
            ))

    pwv_enabled = bool(getattr(getattr(ws, "pwv_params", None), "enabled", False))
    pwv_results = list(getattr(getattr(ws, "derived", None), "pwv_results", []) or [])
    if pwv_enabled:
        if not pwv_results:
            checks.append(_check(
                "hemodynamics.pwv",
                "Hemodynamics",
                "not_run",
                "PWV fit",
                "PWV is enabled but no fit result is available.",
                action="Compute PWV and inspect the arrival-time fit.",
            ))
        else:
            ok_results = [item for item in pwv_results if str(item.get("status", "")) == "ok"]
            r2_values = [float(item["fit_r2"]) for item in ok_results if item.get("fit_r2") is not None]
            min_r2 = min(r2_values) if r2_values else None
            pwv_status = "fail" if not ok_results else ("warn" if len(ok_results) < len(pwv_results) or (min_r2 is not None and min_r2 < 0.8) else "pass")
            checks.append(_check(
                "hemodynamics.pwv",
                "Hemodynamics",
                pwv_status,
                "PWV fit",
                f"successful groups={len(ok_results)}/{len(pwv_results)}, minimum R2={min_r2 if min_r2 is not None else 'n/a'}",
                value={"successful_groups": len(ok_results), "group_count": len(pwv_results), "minimum_fit_r2": min_r2},
                threshold="all requested groups succeed and fit R2 >=0.8",
                action="Review waveform quality, valid planes, cycle wrapping, and fit outliers." if pwv_status != "pass" else "",
            ))

    status_counts = {status: sum(1 for item in checks if item["status"] == status) for status in ("pass", "warn", "fail", "not_run")}
    required_checks = [item for item in checks if item.get("required", True)]
    if any(item["status"] == "fail" for item in required_checks):
        overall_status = "not_ready"
    elif any(item["status"] == "warn" for item in required_checks):
        overall_status = "needs_review"
    elif any(item["status"] == "not_run" for item in required_checks):
        overall_status = "incomplete"
    else:
        overall_status = "ready"

    return {
        "schema": QUALITY_REPORT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source_path or getattr(getattr(ws, "paths", None), "flow_path", "") or ""),
        "source_format": str(getattr(getattr(ws, "input_state", None), "source_format", "") or ""),
        "source_group": getattr(getattr(ws, "input_state", None), "source_group", None),
        "overall_status": overall_status,
        "status_counts": status_counts,
        "checks": checks,
        "run_context": _json_value(dict(run_context or {})),
        "interpretation": "Automated engineering QC for review prioritization; it is not clinical validation or a diagnostic conclusion.",
    }


def save_quality_report(workspace, out_path, source_path="", run_context=None, report=None):
    payload = report if report is not None else build_quality_report(workspace, source_path=source_path, run_context=run_context)
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(_json_value(payload), handle, ensure_ascii=False, indent=2)
    return out_path
