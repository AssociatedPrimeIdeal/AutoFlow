"""Stable plane identity and dependency-aware result validity.

Index remains the legacy display/export order. It is never a persistent result
identity. Signatures contain small geometry/ROI/config payloads, not full 4D
array copies, and job completion also checks the workspace generation.
"""

import copy
import hashlib
import json
import uuid

import numpy as np


def ensure_plane_identity(plane):
    if not getattr(plane, "plane_id", None):
        plane.plane_id = str(uuid.uuid4())
    return {"plane_id": str(plane.plane_id), "geometry_revision": int(getattr(plane, "geometry_revision", 0)),
            "roi_revision": int(getattr(plane, "roi_revision", 0))}


def plane_signature(plane):
    payload = {
        "plane_id": plane.plane_id,
        "center": np.asarray(plane.center).tolist(),
        "normal": np.asarray(plane.normal).tolist(),
        "label": int(plane.label), "segmentation_label": int(plane.segmentation_label),
        "path_index": int(plane.path_index), "distance": float(plane.distance),
        "roi": plane.roi_polygon_uv_mm, "operations": plane.roi_edit_operations,
        "geometry_revision": int(plane.geometry_revision), "roi_revision": int(plane.roi_revision),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     default=lambda x: np.asarray(x).tolist()).encode()).hexdigest()


def workspace_signature(ws, include_analysis=True):
    def token(value):
        if value is None:
            return None
        array = np.asarray(value)
        return (id(value), array.shape, array.dtype.str)

    from ..algorithms.geometry_grid import QUANTITATIVE_POLICY
    payload = (QUANTITATIVE_POLICY, id(ws), int(getattr(ws, "analysis_revision", 0)) if include_analysis else None,
               int(ws.input_state.metadata.get("metric_parameter_revision", 0)),
               tuple((plane_signature(p), metric_validity(p)) for p in ws.planes) if include_analysis else None,
               int(ws.segmentation.revision), float(ws.rr),
               token(ws.flow_raw), token(ws.segmask_binary), token(ws.segmask_labels),
               token(ws.segmentation_analysis_binary), token(ws.segmentation_analysis_labels),
               tuple(np.asarray(ws.resolution).reshape(-1)), tuple(np.asarray(ws.origin).reshape(-1)),
               tuple(sorted((key, value) for key, value in ws.derived.artifact_signatures.items()
                            if key not in {"plane_metrics", "pwv"})))
    return hashlib.sha256(repr(payload).encode()).hexdigest()


def stamp_metric(plane, metric, index=None):
    result = dict(metric or {})
    if not result:
        return result
    result.update(plane_id=plane.plane_id, geometry_revision=int(plane.geometry_revision),
                  roi_revision=int(plane.roi_revision), dependency_signature=plane_signature(plane),
                  validity="current")
    result.pop("stale_reason", None)
    if index is not None:
        result["plane_index"] = int(index)
    return result


def metric_validity(plane, metric=None):
    if any((getattr(plane, "placement", {}) or {}).get(key) for key in ("_draft", "_roi_draft")):
        return "stale"
    value = plane.metrics if metric is None else metric
    if not value:
        return "missing"
    if value.get("validity") == "stale":
        return "stale"
    if value.get("plane_id") not in (None, "", plane.plane_id):
        return "stale"
    signature = value.get("dependency_signature")
    if signature and signature != plane_signature(plane):
        return "stale"
    # Legacy records without identity must at least match their saved geometry.
    for key in ("center", "normal"):
        if key in value and not np.allclose(value[key], getattr(plane, key), rtol=0, atol=1e-7):
            return "stale"
    return "current"


def selected_metric(ws, index, current_only=True):
    if not 0 <= int(index) < len(ws.planes):
        return None
    plane = ws.planes[int(index)]
    value = plane.metrics or next((m for m in ws.derived.plane_metrics
                                  if m.get("plane_id") == plane.plane_id), {})
    if not value and int(index) < len(ws.derived.plane_metrics):
        legacy = ws.derived.plane_metrics[int(index)]
        if not legacy.get("plane_id"):
            value = legacy
    if value and value.get("rr_ms") is not None and not np.isclose(float(value["rr_ms"]), float(ws.rr), rtol=0, atol=1e-9):
        value = dict(value, validity="stale", stale_reason="Cardiac-cycle timing changed")
    if not value or (current_only and metric_validity(plane, value) != "current"):
        return None
    result = dict(value)
    if current_only:
        for family, state in result.get("field_validity", {}).items():
            if state == "stale":
                for key in list(result):
                    if metric_family(key) == family:
                        result[key] = None
    return result


def metric_family(key):
    if key.startswith("wss_"):
        return "wss"
    if key.startswith("tke_"):
        return "tke"
    if key.startswith(("pressure_", "relative_pressure_")):
        return "pressure"
    if key.startswith(("vorticity_", "q_criterion_", "swirling_")):
        return "vortex"
    return None


def invalidate_metric_families(ws, families):
    families = set(families)
    if not families:
        return
    ws.analysis_revision += 1
    metadata = ws.input_state.metadata
    metadata["metric_parameter_revision"] = int(metadata.get("metric_parameter_revision", 0))+1
    metadata["stale_metric_families"] = sorted(set(metadata.get("stale_metric_families", [])) | families)
    for i, plane in enumerate(ws.planes):
        value = selected_metric(ws, i, current_only=False)
        if value:
            flags = dict(value.get("field_validity", {}))
            flags.update({family: "stale" for family in families})
            plane.metrics = dict(value, field_validity=flags)
    ws.derived.plane_metrics = [dict(p.metrics or {}) for p in ws.planes]
    for family in families:
        ws.derived.artifact_signatures.pop(family, None)
    ws.derived.plane_pixelwise_file = ""


def stale_metric_families(ws):
    families = set(ws.input_state.metadata.get("stale_metric_families", []))
    for i in range(len(ws.planes)):
        value = selected_metric(ws, i, current_only=False) or {}
        families.update(key for key, state in value.get("field_validity", {}).items() if state == "stale")
    return sorted(families)


def current_plane_metrics(ws):
    return [stamp_metric(ws.planes[i], metric, i) for i in range(len(ws.planes))
            if (metric := selected_metric(ws, i))]


def refresh_plane_consistency(ws):
    from ..algorithms.metrics.consistency import apply_internal_consistency_to_metrics
    values = current_plane_metrics(ws)
    incomplete_paths = {plane.path_index for i, plane in enumerate(ws.planes) if selected_metric(ws, i) is None}
    values, qc = apply_internal_consistency_to_metrics(values, path_info=ws.path_info, forks=ws.forks)
    by_id = {metric.get("plane_id"): metric for metric in values}
    for path in incomplete_paths:
        qc.get("path_ic", {}).pop(str(path), None)
        qc.get("path_ic", {}).pop(path, None)
    for i, plane in enumerate(ws.planes):
        if plane.plane_id in by_id:
            metric = by_id[plane.plane_id]
            if plane.path_index in incomplete_paths:
                metric["path_ic"] = None
            metric["plane_index"] = i
            plane.metrics = metric
    ws.derived.plane_metrics = [dict(p.metrics or {}) for p in ws.planes]
    if incomplete_paths:
        qc.update(validity="partial", incomplete_path_indices=sorted(incomplete_paths))
        qc["fork_ic"] = {}
    ws.derived.plane_qc = qc
    return qc


def invalidate_planes(ws, plane_ids, reason, *, geometry=False, roi=False):
    selected = set(plane_ids)
    ws.analysis_revision = int(getattr(ws, "analysis_revision", 0)) + 1
    values = list(ws.derived.plane_metrics)
    for index, plane in enumerate(ws.planes):
        if plane.plane_id not in selected:
            continue
        plane.geometry_revision += int(bool(geometry))
        plane.roi_revision += int(bool(roi))
        previous = plane.metrics or (values[index] if index < len(values) else {})
        if previous:
            plane.metrics = dict(previous, validity="stale", stale_reason=str(reason))
            if index < len(values):
                values[index] = plane.metrics
    ws.derived.plane_metrics = values
    ws.derived.plane_qc = {"validity": "stale", "stale_reason": str(reason)}
    ws.derived.plane_pixelwise_file = ""
    affected_paths = {p.path_index for p in ws.planes if p.plane_id in selected}
    for result in ws.derived.pwv_results:
        if result.get("dependency_scope") == "segmentation_centerline_generated_planes":
            continue
        # Older PWV outputs do not carry stable plane references. Mark their
        # status conservatively; retain values as an explicit previous result.
        result["validity"] = "stale"
        result["stale_reason"] = str(reason)
    ws.derived.artifact_signatures.pop("plane_metrics", None)
    ws.derived.artifact_signatures.pop("pwv", None)
    for index, plane in enumerate(ws.planes):
        if plane.plane_id in selected:
            ws.pathline_cache.pop(index, None)
            getattr(ws, "pathline_seed_cache", {}).pop(index, None)
    return affected_paths


def remove_planes(ws, plane_ids):
    """Remove by stable ID and keep surviving legacy indexes synchronized."""
    removed = set(plane_ids)
    old_planes = list(ws.planes)
    by_id = {p.plane_id: dict(p.metrics or {}) for p in old_planes}
    for index, metric in enumerate(ws.derived.plane_metrics):
        if not isinstance(metric, dict):
            continue
        key = metric.get("plane_id") or (old_planes[index].plane_id if index < len(old_planes) else None)
        if key and not by_id.get(key):
            by_id[key] = dict(metric)
    index_map = {}
    ws.planes = [plane for index, plane in enumerate(old_planes) if plane.plane_id not in removed]
    for new_index, plane in enumerate(ws.planes):
        old_index = next(i for i, old in enumerate(old_planes) if old.plane_id == plane.plane_id)
        index_map[old_index] = new_index
        metric = by_id.get(plane.plane_id, {})
        if metric:
            metric.update(plane_id=plane.plane_id, plane_index=new_index)
            # Preserve computed values but remove topology-dependent checks.
            for key in tuple(metric):
                if key.endswith("_ic") or key in {"IC", "path_IC", "fork_IC"}:
                    metric.pop(key, None)
        plane.metrics = metric
    ws.derived.plane_metrics = [dict(p.metrics) for p in ws.planes]
    ws.analysis_revision = int(getattr(ws, "analysis_revision", 0)) + 1
    ws.derived.plane_qc = {"validity": "stale", "stale_reason": "Plane collection changed"}
    ws.derived.plane_pixelwise_file = ""
    for name in ("pathline_cache", "pathline_seed_cache", "pathline_colors"):
        original = getattr(ws, name, {})
        setattr(ws, name, {index_map[i]: value for i, value in original.items() if i in index_map})
    ws.active_pathline_plane_indices = [index_map[i] for i in ws.active_pathline_plane_indices if i in index_map]
    for result in ws.derived.pwv_results:
        if result.get("dependency_scope") == "segmentation_centerline_generated_planes":
            continue
        result.update(validity="stale", stale_reason="Plane collection changed")
    # Group snapshots may retain their own plane list. Remove the same IDs.
    for group in ws.multilabel_groups.values():
        group["planes"] = [p for p in group.get("planes", []) if p.plane_id not in removed]
    return index_map


def mutation_snapshot(ws):
    """Small undo snapshot: plane/ROI/results and references to existing arrays."""
    return {"planes": copy.deepcopy(ws.planes), "metrics": copy.deepcopy(ws.derived.plane_metrics),
            "case_identity": (id(ws), str(ws.paths.flow_path), ws.input_state.source_group),
            "qc": copy.deepcopy(ws.derived.plane_qc), "pwv": copy.deepcopy(ws.derived.pwv_results),
            "pixelwise": ws.derived.plane_pixelwise_file,
            "input_signature": workspace_signature(ws, include_analysis=False),
            "pathlines": dict(ws.pathline_cache), "pathline_seeds": dict(ws.pathline_seed_cache),
            "pathline_colors": dict(ws.pathline_colors), "active_pathlines": list(ws.active_pathline_plane_indices),
            "group_plane_ids": {key: [p.plane_id for p in group.get("planes", [])]
                                for key, group in ws.multilabel_groups.items()}}
