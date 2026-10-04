"""Basic plane flow, area and velocity metrics with isolated-process dispatch."""

import os
import numpy as np
from ...task_control import check_cancelled, task_scope
from ..paths import _determine_plane_forward, _vector_orientation_text
from ..surfaces import _build_branch_grid

from ._common import _ensure_flow5d
from ._parallel import (
    _PlaneProcessToken,
    _mark_plane_process_progress,
    _wait_plane_processes,
)
from .consistency import apply_internal_consistency_to_metrics
from .sampling import (
    _add_labeled_plane_support_meshes,
    _build_mask_phase_lookup,
    _build_plane_support_mesh_cache,
    _get_cached_plane_slice_spec,
    _sample_field_from_slice_spec,
    _target_label_for_plane,
)


def compute_plane_metrics(flow_xyzt3, segmask_binary_4d, spacing, origin, planes, RR=1000.0,
                          branch_labels_3d=None, path_info=None, forks=None,
                          paths=None, return_qc=False, segmentation_labels_3d=None,
                          segmentation_labels_4d=None, skip_support_mesh=False,
                          progress_callback=None):
    flow = _ensure_flow5d(flow_xyzt3)
    mask = np.asarray(segmask_binary_4d, dtype=bool)
    spacing = np.asarray(spacing, dtype=float).reshape(-1)[:3]
    origin = np.asarray(origin, dtype=float).reshape(-1)[:3]
    if flow.ndim != 5 or flow.shape[-1] != 3:
        raise ValueError(f"flow must be XYZTV, got {flow.shape}")
    if mask.ndim == 3:
        mask = np.repeat(mask[..., np.newaxis], flow.shape[3], axis=3)
    elif mask.ndim == 4 and mask.shape[3] == 1 and flow.shape[3] > 1:
        mask = np.repeat(mask, flow.shape[3], axis=3)
    if mask.shape[3] != flow.shape[3]:
        raise ValueError(f"mask time dimension {mask.shape[3]} does not match flow {flow.shape[3]}")
    Nt = int(flow.shape[3])
    # A replacement ROI is already an explicit local cut; branch ownership is
    # not sampled for that path. The frame-only contour update can therefore
    # skip constructing the full-volume branch grid as well.
    branch_grid = None if bool(skip_support_mesh) else _build_branch_grid(branch_labels_3d, spacing, origin)
    mask_phase_lookup = _build_mask_phase_lookup(mask)
    # ROI replacement edits build a small local support mesh inside
    # ``_build_plane_slice_region``. Avoid eagerly meshing the entire 3-D mask
    # for the common one-frame contour-edit case.
    needs_support_mesh = not bool(skip_support_mesh)
    support_mesh_cache = (
        _build_plane_support_mesh_cache(mask, mask_phase_lookup, spacing, origin)
        if needs_support_mesh else None
    )
    if support_mesh_cache is not None:
        _add_labeled_plane_support_meshes(
            support_mesh_cache, mask,
            segmentation_labels_4d if segmentation_labels_4d is not None else segmentation_labels_3d,
            planes, spacing, origin
        )
    mask_static = all(int(rep_t) == 0 for rep_t in mask_phase_lookup)
    mask_template = mask[..., 0] if mask_static else None

    paths_lookup = None
    if paths is not None:
        paths_lookup = [np.asarray(p, dtype=float).reshape(-1, 3) for p in paths]

    if len(planes) == 0:
        empty_qc = {"path_ic": {}, "segmentation_label_ic": {}, "fork_ic": {}, "forks": []}
        if return_qc:
            return [], empty_qc
        return []

    results = []
    total_planes = len(planes)
    for plane_index, plane in enumerate(planes, start=1):
        check_cancelled()
        target_label = None
        if branch_labels_3d is not None:
            target_label = int(getattr(plane, "label", 0) or 0)
            if target_label <= 0:
                target_label = _target_label_for_plane(plane, branch_labels_3d, spacing, origin)
        pp = None
        if paths_lookup is not None:
            pi = int(getattr(plane, "path_index", -1))
            if 0 <= pi < len(paths_lookup):
                pp = paths_lookup[pi]
        results.append(_compute_single_plane_metric(
            (flow, mask, spacing, origin, plane, Nt, RR,
             branch_grid, target_label, path_info, pp, mask_template, mask_phase_lookup,
             support_mesh_cache, segmentation_labels_3d, segmentation_labels_4d)
        ))
        if progress_callback is not None:
            progress_callback({
                "stage": "plane_metric",
                "current": int(plane_index),
                "total": int(total_planes),
                "message": f"Calculated plane metrics ({plane_index}/{total_planes})",
            })

    results, qc = apply_internal_consistency_to_metrics(results, path_info=path_info, forks=forks)
    for plane_index, metric in enumerate(results):
        if isinstance(metric, dict):
            metric["plane_index"] = int(plane_index)
    if return_qc:
        return results, qc
    return results


def _compute_single_plane_metric(args):
    (flow, mask, spacing, origin, plane, Nt, RR, branch_grid, target_label,
     path_info, path_points, mask_template, mask_phase_lookup,
     support_mesh_cache, segmentation_labels_3d, segmentation_labels_4d) = args
    normal = np.asarray(plane.normal, dtype=float).reshape(3)
    normal = normal / (np.linalg.norm(normal) + 1e-12)

    forward_sign, forward_sign_source, local_tangent, ntc = _determine_plane_forward(
        plane, path_points, path_info, eps=0.1,
    )

    peakv_abs = 0.0
    peakv_fwd = 0.0
    peakv_rev = 0.0
    flowrate = []        
    flowrate_fwd = []     
    flowrate_rev = []     
    area = []
    meanv_t = []         
    meanv_fwd_t = []     
    meanv_rev_t = []     
    slice_cache = {}
    labels4d_arr = None if segmentation_labels_4d is None else np.asarray(segmentation_labels_4d)
    labels3d_arr = None if segmentation_labels_3d is None else np.asarray(segmentation_labels_3d)

    for t in range(Nt):
        check_cancelled()
        rep_t = int(mask_phase_lookup[t]) if mask_phase_lookup else int(t)
        mask_t = mask_template if mask_template is not None else mask[..., rep_t]
        plane_seg_label = int(getattr(plane, "segmentation_label", 0) or 0)
        mask_t = np.asarray(mask_t, dtype=bool)
        label_support = None
        if plane_seg_label > 0 and support_mesh_cache is not None:
            label_support = support_mesh_cache.get(
                ("seg4d", int(t), plane_seg_label) if segmentation_labels_4d is not None
                else ("seg3d", rep_t, plane_seg_label)
            )
        if plane_seg_label > 0:
            # Prefer phase-specific labels when available, while retaining
            # the legacy 3-D label behavior for older workspaces and H5
            # inputs that do not carry a 4-D label sequence.
            if labels4d_arr is not None and labels4d_arr.ndim == 4 and labels4d_arr.shape[:3] == mask_t.shape[:3]:
                if label_support is None:
                    mask_t = mask_t & (labels4d_arr[..., int(t)] == plane_seg_label)
            elif labels3d_arr is not None:
                labels3d = labels3d_arr[..., 0] if labels3d_arr.ndim == 4 else labels3d_arr
                if labels3d.shape == mask_t.shape[:3] and label_support is None:
                    mask_t = mask_t & (labels3d == plane_seg_label)
        frame_specific_roi = bool(getattr(plane, "roi_edit_operations", {}) or {})
        frame_specific_labels = plane_seg_label > 0 and segmentation_labels_4d is not None
        label_phase = (
            support_mesh_cache.get(("seg_phase", int(t), plane_seg_label), t)
            if support_mesh_cache is not None else t
        )
        slice_cache_key = (rep_t, t) if frame_specific_roi else (
            (rep_t, label_phase) if frame_specific_labels else rep_t
        )
        slice_spec = _get_cached_plane_slice_spec(
            slice_cache,
            slice_cache_key,
            mask_t,
            plane,
            spacing,
            origin,
            branch_grid=branch_grid,
            target_label=target_label,
            select_connected=True,
            support_mesh=(
                label_support
                if plane_seg_label > 0 and support_mesh_cache is not None
                else (support_mesh_cache.get(rep_t) if support_mesh_cache is not None else None)
            ),
            frame_index=t,
        )
        if slice_spec is None:
            flowrate.append(0.0); flowrate_fwd.append(0.0); flowrate_rev.append(0.0)
            meanv_t.append(0.0); meanv_fwd_t.append(0.0); meanv_rev_t.append(0.0)
            area.append(0.0)
            continue
        vec = np.asarray(
            _sample_field_from_slice_spec(flow[..., t, :], mask.shape[:3], "flow", slice_spec),
            dtype=float,
        )
        ca = np.asarray(slice_spec["areas"], dtype=float).reshape(-1)
        if vec.ndim != 2 or vec.shape[1] != 3 or len(vec) != len(ca):
            flowrate.append(0.0); flowrate_fwd.append(0.0); flowrate_rev.append(0.0)
            meanv_t.append(0.0); meanv_fwd_t.append(0.0); meanv_rev_t.append(0.0)
            area.append(float(np.sum(ca)) if len(ca) else 0.0)
            continue

        proj = np.dot(vec, normal)
        proj_fwd = proj * float(forward_sign)
        pos_mask = proj_fwd > 0.0
        neg_mask = proj_fwd < 0.0

        area_t = float(np.sum(ca)) if len(ca) else 0.0

        fr = float(np.sum(proj * ca) / 100.0) if area_t > 0.0 else 0.0
        if area_t > 0.0:
            fr_fwd = float(np.sum(proj_fwd[pos_mask] * ca[pos_mask]) / 100.0)
            fr_rev = float(-np.sum(proj_fwd[neg_mask] * ca[neg_mask]) / 100.0)
            mv = float(np.sum(proj * ca) / area_t)
            mv_fwd = float(np.sum(np.where(pos_mask, proj_fwd, 0.0) * ca) / area_t)
            mv_rev = float(-np.sum(np.where(neg_mask, proj_fwd, 0.0) * ca) / area_t)
        else:
            fr_fwd = fr_rev = 0.0
            mv = mv_fwd = mv_rev = 0.0

        if len(proj):
            peakv_abs = max(peakv_abs, float(np.max(np.abs(proj))))
        if len(proj_fwd):
            if np.any(pos_mask):
                peakv_fwd = max(peakv_fwd, float(np.max(proj_fwd[pos_mask])))
            if np.any(neg_mask):
                peakv_rev = max(peakv_rev, float(-np.min(proj_fwd[neg_mask])))

        flowrate.append(fr)
        flowrate_fwd.append(fr_fwd)
        flowrate_rev.append(fr_rev)
        meanv_t.append(mv)
        meanv_fwd_t.append(mv_fwd)
        meanv_rev_t.append(mv_rev)
        area.append(area_t)

    netflow_mag = float(abs(np.mean(flowrate)) * RR / 1000.0) if len(flowrate) else 0.0
    netflow_fwd = float(np.mean(flowrate_fwd) * RR / 1000.0) if len(flowrate_fwd) else 0.0
    netflow_rev = float(np.mean(flowrate_rev) * RR / 1000.0) if len(flowrate_rev) else 0.0
    net_signed = float(netflow_fwd - netflow_rev)
    reflux_frac = float(netflow_rev / netflow_fwd) if netflow_fwd > 1e-12 else 0.0
    meanv_fwd = float(np.mean(meanv_fwd_t)) if meanv_fwd_t else 0.0
    meanv_rev = float(np.mean(meanv_rev_t)) if meanv_rev_t else 0.0

    sgn = float(forward_sign)
    flowrate_signed = [float(x) * sgn for x in flowrate]
    meanv_signed_t = [float(x) * sgn for x in meanv_t]
    meanv_signed = float(np.mean(meanv_signed_t)) if meanv_signed_t else 0.0

    metric = {
        "center": np.asarray(plane.center, dtype=float).tolist(),
        "normal": normal.tolist(),
        "label": int(plane.label),
        "segmentation_label": int(getattr(plane, "segmentation_label", 0) or 0),
        "path_index": int(plane.path_index),
        "distance": float(plane.distance),
        "target_branch_label": int(target_label) if target_label is not None else 0,
        "peakv_cm_s": float(peakv_abs),
        "flowrate_mL_s": [float(x) for x in flowrate],
        "netflow_mL_beat": netflow_mag,
        "meanv_cm_s": float(np.mean(meanv_t)) if meanv_t else 0.0,
        "meanv_cm_s_t": [float(x) for x in meanv_t],
        "area_mm2": [float(x) for x in area],
        "flowrate_forward_mL_s": [float(x) for x in flowrate_fwd],
        "flowrate_reverse_mL_s": [float(x) for x in flowrate_rev],
        "meanv_forward_cm_s_t": [float(x) for x in meanv_fwd_t],
        "meanv_reverse_cm_s_t": [float(x) for x in meanv_rev_t],
        "flowrate_signed_mL_s": flowrate_signed,
        "meanv_signed_cm_s_t": meanv_signed_t,
        "meanv_signed_cm_s": meanv_signed,
        "netflow_forward_mL_beat": netflow_fwd,
        "netflow_reverse_mL_beat": netflow_rev,
        "net_netflow_signed_mL_beat": net_signed,
        "reflux_fraction": reflux_frac,
        "peakv_forward_cm_s": float(peakv_fwd),
        "peakv_reverse_cm_s": float(peakv_rev),
        "meanv_forward_cm_s": meanv_fwd,
        "meanv_reverse_cm_s": meanv_rev,
        "forward_sign": int(forward_sign),
        "forward_sign_source": str(forward_sign_source),
        "local_path_tangent": [float(x) for x in np.asarray(local_tangent, dtype=float).tolist()],
        "local_path_direction": _vector_orientation_text(local_tangent),
        "normal_tangent_cos": float(ntc),
        "roi_mode": "manual_polygon" if len(getattr(plane, "roi_polygon_uv_mm", []) or []) >= 3 else "automatic",
        "roi_polygon_uv_mm": [
            [float(point[0]), float(point[1])]
            for point in (getattr(plane, "roi_polygon_uv_mm", []) or [])
            if len(point) >= 2
        ],
    }
    if path_info is not None and 0 <= int(plane.path_index) < len(path_info):
        info = path_info[int(plane.path_index)]
        metric["path_direction"] = info.get("direction_text", "")
        metric["path_start_point"] = info.get("start_point", [0.0, 0.0, 0.0])
        metric["path_end_point"] = info.get("end_point", [0.0, 0.0, 0.0])
        metric["path_fork_ids"] = info.get("fork_ids", [])
        metric["path_fork_roles"] = info.get("fork_roles", [])
    return metric


def _compute_plane_metrics_process_chunk(
    indexed_planes, flow, mask, spacing, origin, RR, branch_labels_3d,
    path_info, paths, segmentation_labels_3d, segmentation_labels_4d,
    progress_path,
):
    indices = [int(index) for index, _plane in indexed_planes]
    planes = [plane for _index, plane in indexed_planes]

    def _mark_progress(_payload):
        _mark_plane_process_progress(progress_path)

    with task_scope(_PlaneProcessToken(progress_path)):
        metrics = compute_plane_metrics(
            flow, mask, spacing, origin, planes,
            RR=RR,
            branch_labels_3d=branch_labels_3d,
            path_info=path_info,
            forks=None,
            paths=paths,
            return_qc=False,
            segmentation_labels_3d=segmentation_labels_3d,
            segmentation_labels_4d=segmentation_labels_4d,
            progress_callback=_mark_progress if progress_path else None,
        )
    return list(zip(indices, metrics))


def _run_plane_metric_processes(
    flow, mask, spacing, origin, planes, RR, branch_labels_3d, path_info,
    paths, segmentation_labels_3d, segmentation_labels_4d, max_workers,
    progress_callback,
):
    import tempfile
    from joblib import Parallel, delayed
    worker_count = max(1, min(int(max_workers), len(planes)))
    indexed = list(enumerate(planes))
    chunks = [indexed[offset::worker_count] for offset in range(worker_count)]
    with tempfile.TemporaryDirectory(prefix="autoflow_plane_metrics_") as progress_dir:
        progress_path = os.path.join(progress_dir, "progress.log")
        open(progress_path, "ab").close()
        def calculate():
            return Parallel(n_jobs=worker_count, backend="loky", max_nbytes="10M", mmap_mode="r")(
                delayed(_compute_plane_metrics_process_chunk)(
                    chunk, flow, mask, spacing, origin, RR, branch_labels_3d,
                    path_info, paths, segmentation_labels_3d, segmentation_labels_4d, progress_path,
                ) for chunk in chunks if chunk
            )
        chunk_results = _wait_plane_processes(calculate, progress_path, len(planes), progress_callback)
    results = [None] * len(planes)
    for chunk in chunk_results or []:
        for plane_index, metric in chunk:
            results[int(plane_index)] = metric
    return results


def compute_plane_metrics_multithread(flow_xyzt3, segmask_binary_4d, spacing, origin, planes, RR=1000.0,
                                       branch_labels_3d=None, path_info=None, forks=None,
                                       paths=None, return_qc=False, max_workers=None,
                                       segmentation_labels_3d=None, segmentation_labels_4d=None,
                                       progress_callback=None):
    flow = _ensure_flow5d(flow_xyzt3)
    mask = np.asarray(segmask_binary_4d, dtype=bool)
    spacing = np.asarray(spacing, dtype=float).reshape(-1)[:3]
    origin = np.asarray(origin, dtype=float).reshape(-1)[:3]
    if flow.ndim != 5 or flow.shape[-1] != 3:
        raise ValueError(f"flow must be XYZTV, got {flow.shape}")
    if mask.ndim == 3:
        mask = np.repeat(mask[..., np.newaxis], flow.shape[3], axis=3)
    elif mask.ndim == 4 and mask.shape[3] == 1 and flow.shape[3] > 1:
        mask = np.repeat(mask, flow.shape[3], axis=3)
    if len(planes) == 0:
        empty_qc = {"path_ic": {}, "segmentation_label_ic": {}, "fork_ic": {}, "forks": []}
        if return_qc:
            return [], empty_qc
        return []

    if max_workers is None:
        import os as _os
        # Separate processes avoid concurrent access to shared VTK datasets.
        # Temporal workloads can be expensive even below 128 planes: the
        # controlled 96-plane/20-phase case benefits from four workers.
        # Keep short/small jobs serial to avoid startup and memmap overhead.
        plane_phase_work = len(planes) * int(flow.shape[3])
        if len(planes) < 128 and plane_phase_work < 1920:
            max_workers = 1
        else:
            worker_limit = 4 if len(planes) < 128 else 8
            max_workers = min(len(planes), worker_limit, max(1, _os.cpu_count() or 4))
    if int(max_workers) <= 1:
        result = compute_plane_metrics(
            flow, mask, spacing, origin, planes, RR=RR,
            branch_labels_3d=branch_labels_3d, path_info=path_info,
            forks=forks, paths=paths, return_qc=return_qc,
            segmentation_labels_3d=segmentation_labels_3d,
            segmentation_labels_4d=segmentation_labels_4d,
            progress_callback=progress_callback,
        )
        return result
    else:
        results = _run_plane_metric_processes(
            flow, mask, spacing, origin, planes, RR, branch_labels_3d,
            path_info, paths, segmentation_labels_3d, segmentation_labels_4d,
            max_workers, progress_callback,
        )
    results, qc = apply_internal_consistency_to_metrics(results, path_info=path_info, forks=forks)
    for plane_index, metric in enumerate(results):
        if isinstance(metric, dict):
            metric["plane_index"] = int(plane_index)
    if return_qc:
        return results, qc
    return results
