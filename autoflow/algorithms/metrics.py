import os

import h5py
import numpy as np
import pyvista as pv
from ..task_control import TaskCancelled, check_cancelled, current_cancellation_token, report_progress, task_scope
from scipy.ndimage import binary_erosion, gaussian_filter, generate_binary_structure, label
from scipy.sparse import bmat, csc_matrix, csr_matrix, diags
from scipy.sparse.linalg import LinearOperator, cg, factorized, minres, spsolve

try:
    import pyamg
except ImportError:  # Keep source checkouts usable before dependencies are refreshed.
    pyamg = None

from .paths import _determine_plane_forward, _vector_orientation_text
from .surfaces import (
    _extract_surface,
    _build_branch_grid,
    _select_connected_region,
    create_uniform_field_grid,
    create_uniform_grid,
    create_uniform_vector,
)


# Replacement contours usually stay within the same local voxel box across
# successive edits. Cache those small support meshes so each edit does not
# rebuild an identical VTK threshold grid. The cache is intentionally bounded
# because meshes retain VTK-owned memory.
_LOCAL_SUPPORT_MESH_CACHE = {}
_LOCAL_SUPPORT_MESH_CACHE_MAX = 16


def extract_vectors(polydata):
    return np.stack((polydata["u"], polydata["v"], polydata["w"]), axis=-1)


def get_orthogonal_vectors(vectors, point_normals):
    c = np.sum(vectors * point_normals, axis=1)
    return c[:, None] * point_normals, vectors - c[:, None] * point_normals


def get_vector_magnitude(vectors):
    return np.sqrt(np.sum(vectors * vectors, axis=1))


def align_tangential_samples(reference_vectors, target_vectors):
    reference = np.asarray(reference_vectors, dtype=float)
    target = np.asarray(target_vectors, dtype=float)
    ref_mag = get_vector_magnitude(reference)
    tgt_mag = get_vector_magnitude(target)
    denom = np.maximum(ref_mag * tgt_mag, 1e-12)
    direction = np.sum(reference * target, axis=1) / denom
    return np.clip(direction, -1.0, 1.0) * tgt_mag


def resolve_wss_inward_distance(spacing, inward_distance=None):
    spacing_mm = np.asarray(spacing, dtype=float).reshape(3)
    if not np.all(np.isfinite(spacing_mm)) or np.any(spacing_mm <= 0):
        raise ValueError("WSS spacing must be finite and positive (mm)")
    if inward_distance is None:
        inward_distance = float(np.min(spacing_mm))
    distance = float(inward_distance)
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError("WSS inward distance must be finite and positive (mm)")
    return distance


def calculate_gradient(pc0_tangent_mag, pc1_tangent_mag, pc2_tangent_mag, inward_distance, use_parabolic=True):
    """Derivative at zero of equidistant scalar or vector wall samples."""
    distance = float(inward_distance)
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError("WSS inward distance must be finite and positive (mm)")
    v0, v1, v2 = (np.asarray(v, dtype=float) for v in
                  (pc0_tangent_mag, pc1_tangent_mag, pc2_tangent_mag))
    if use_parabolic:
        return (-3.0 * v0 + 4.0 * v1 - v2) / (2.0 * distance)
    return (v1 - v0) / distance


def cal_wss_from_surf(surf, velocity, viscosity=4.0, inward_distance=0.6,
                      parabolic_fitting=True, no_slip_condition=True, support_grid=None):
    """Compute tangential WSS vectors (Pa) using inward-normal derivatives.

    Cell-centred velocity is first interpolated to points for continuous probing.
    Invalid probes or wall-normal segments leaving the lumen produce NaNs.
    """
    distance = float(inward_distance)
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError("WSS inward distance must be finite and positive (mm)")
    viscosity = float(viscosity)
    if not np.isfinite(viscosity) or viscosity < 0:
        raise ValueError("WSS viscosity must be finite and nonnegative (mPa s)")
    if not all(key in velocity.point_data for key in ("u", "v", "w")):
        velocity = velocity.cell_data_to_point_data(pass_cell_data=False)
    surf.compute_normals(point_normals=True, cell_normals=True, inplace=True,
                         consistent_normals=True, auto_orient_normals=(surf.n_open_edges == 0),
                         flip_normals=True)
    normals = np.asarray(surf.point_normals, dtype=float)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    pc0 = pv.PolyData(surf.points).sample(velocity)
    pc1 = pv.PolyData(pc0.points + distance * normals).sample(velocity)
    pc2 = pv.PolyData(pc0.points + 2.0 * distance * normals).sample(velocity)

    if no_slip_condition:
        tang0 = np.zeros((len(pc0.points), 3), dtype=float)
    else:
        _, tang0 = get_orthogonal_vectors(extract_vectors(pc0), normals)

    _, tang1 = get_orthogonal_vectors(extract_vectors(pc1), normals)
    _, tang2 = get_orthogonal_vectors(extract_vectors(pc2), normals)
    required = [pc1] + ([pc2] if parabolic_fitting else [])
    if not no_slip_condition:
        required.append(pc0)
    valid = np.all(np.isfinite(normals), axis=1) & (np.linalg.norm(normals, axis=1) > 0.5)
    for probe in required:
        valid &= np.asarray(probe["vtkValidPointMask"], dtype=bool)
        valid &= np.all(np.isfinite(extract_vectors(probe)), axis=1)
    if support_grid is not None:
        # Check intermediate locations too: endpoints alone can miss a crossing
        # through background into another branch or the opposite vessel wall.
        fractions = (0.25, 0.5, 1.0, 1.5, 2.0) if parabolic_fitting else (0.25, 0.5, 1.0)
        for fraction in fractions:
            probe = pv.PolyData(pc0.points + fraction * distance * normals).sample(support_grid)
            valid &= np.asarray(probe["vtkValidPointMask"], dtype=bool)
            valid &= np.asarray(probe["wss_lumen"], dtype=float) > 0.5
    # mPa s * (m/s)/mm is numerically Pa; keep signed vector components.
    vectors = calculate_gradient(tang0, tang1, tang2, distance,
                                 use_parabolic=parabolic_fitting) * viscosity
    vectors[~valid] = np.nan
    surf["wss_vectors"] = vectors
    surf["wss"] = get_vector_magnitude(vectors)
    surf["wss_valid"] = valid.astype(np.uint8)
    return surf


def summarize_internal_consistency(plane_metrics, path_info=None, forks=None):
    by_path = {}
    for metric in plane_metrics:
        pidx = int(metric.get("path_index", -1))
        by_path.setdefault(pidx, []).append(abs(float(metric.get("netflow_mL_beat", 0.0))))
    path_ic = {}
    by_path_mean = {}
    for pidx, values in by_path.items():
        arr = np.abs(np.asarray(values, dtype=float))
        mu = float(np.mean(arr)) if len(arr) else 0.0
        by_path_mean[pidx] = mu
        if len(arr) <= 1:
            # A path with zero or one usable plane has no consistency
            # comparison to make.  Report it as undefined instead of
            # presenting a vacuous perfect score.
            ic = None
        elif mu <= 1e-12:
            ic = 1.0 if float(np.max(arr)) <= 1e-12 else 0.0
        else:
            ic = 1.0 - float(np.mean(np.abs(arr - mu)) / mu)
        path_ic[str(int(pidx))] = None if ic is None else float(np.clip(ic, 0.0, 1.0))
    # Keep topology paths with no usable plane metrics visible in QC as
    # undefined.  They must not silently disappear or be interpreted as zero
    # flow by fork consistency calculations.
    if path_info is not None:
        for pidx in range(len(path_info)):
            path_ic.setdefault(str(int(pidx)), None)
    # When segmentation filtering is active, expose an additional consistency
    # view keyed by the numeric segmentation label.  Path consistency remains
    # available for topology QC and backwards compatibility.
    by_seg = {}
    for metric in plane_metrics:
        label = int(metric.get("segmentation_label", 0) or 0)
        if label > 0:
            by_seg.setdefault(label, []).append(abs(float(metric.get("netflow_mL_beat", 0.0))))
    segmentation_ic = {}
    for label, values in by_seg.items():
        arr = np.asarray(values, dtype=float)
        mu = float(np.mean(arr)) if len(arr) else 0.0
        if len(arr) <= 1 or mu <= 1e-12:
            ic = 1.0
        else:
            ic = 1.0 - float(np.mean(np.abs(arr - mu)) / mu)
        segmentation_ic[str(int(label))] = float(np.clip(ic, 0.0, 1.0))
    fork_items = []
    fork_ic = {}
    for fork_id, fork in enumerate(forks or []):
        left = [int(x) for x in fork.get("left", [])]
        right = [int(x) for x in fork.get("right", [])]
        sum_left = float(np.sum([abs(by_path_mean.get(x, 0.0)) for x in left]))
        sum_right = float(np.sum([abs(by_path_mean.get(x, 0.0)) for x in right]))
        # A topology fork with no incoming or no outgoing path has no
        # meaningful conservation comparison.  This can happen when local
        # flow orientation is ambiguous; report it as undefined instead of
        # manufacturing an IC of zero (or one for an empty fork).
        missing_left = [x for x in left if x not in by_path_mean]
        missing_right = [x for x in right if x not in by_path_mean]
        if not left or not right:
            ic = None
            status = "one_sided_topology"
        elif missing_left or missing_right:
            # A missing path means no plane supplied a measurable flow for
            # that side.  Treating it as numerical zero would bias the fork
            # conservation score, so report the comparison as incomplete.
            ic = None
            status = "missing_path_metrics"
        else:
            denom = sum_left + sum_right
            if denom <= 1e-12:
                ic = 1.0
            else:
                ic = 1.0 - 2.0 * abs(sum_left - sum_right) / denom
            ic = float(np.clip(ic, 0.0, 1.0))
            status = "ok"
        fork_ic[str(int(fork_id))] = ic
        item = {
            "fork_id": int(fork_id),
            "left": left,
            "right": right,
            "crosspoint": fork.get("crosspoint", [0.0, 0.0, 0.0]),
            "node": int(fork.get("node", -1)),
            "ic": ic,
            "status": status,
        }
        if path_info is not None:
            item["left_dirs"] = [path_info[x].get("direction_text", "") for x in left if 0 <= x < len(path_info)]
            item["right_dirs"] = [path_info[x].get("direction_text", "") for x in right if 0 <= x < len(path_info)]
        fork_items.append(item)
    return {"path_ic": path_ic, "segmentation_label_ic": segmentation_ic,
            "fork_ic": fork_ic, "forks": fork_items}


def apply_internal_consistency_to_metrics(plane_metrics, path_info=None, forks=None):
    metrics = [dict(metric) for metric in plane_metrics]
    qc = summarize_internal_consistency(metrics, path_info=path_info, forks=forks)
    for metric in metrics:
        pidx = str(int(metric.get("path_index", -1)))
        path_ic = qc["path_ic"].get(pidx, None)
        metric["path_ic"] = None if path_ic is None else float(path_ic)
        seg_label = str(int(metric.get("segmentation_label", 0) or 0))
        metric["segmentation_label_ic"] = float(qc.get("segmentation_label_ic", {}).get(seg_label, 1.0))
        rel = []
        for fork in qc.get("forks", []):
            pid = int(metric.get("path_index", -1))
            if pid in fork.get("left", []) or pid in fork.get("right", []):
                role = "incoming" if pid in fork.get("left", []) else "outgoing"
                fork_ic_value = fork.get("ic", 1.0)
                rel.append({
                    "fork_id": int(fork.get("fork_id", -1)),
                    "role": role,
                    "ic": None if fork_ic_value is None else float(fork_ic_value),
                })
        metric["fork_ic"] = rel
    return metrics, qc


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


def _ensure_mask4d(mask4d):
    mask4d = np.asarray(mask4d, dtype=bool)
    if mask4d.ndim == 3:
        mask4d = mask4d[..., np.newaxis]
    if mask4d.ndim != 4:
        raise ValueError(f"mask4d must be XYZ or XYZT, got {mask4d.shape}")
    return mask4d


def _ensure_flow5d(flow):
    flow = np.asarray(flow, dtype=np.float32)
    if flow.ndim != 5 or flow.shape[-1] != 3:
        raise ValueError(f"flow must be XYZTV, got {flow.shape}")
    return flow


def _target_label_for_plane(plane, branch_labels_3d, spacing, origin):
    if branch_labels_3d is None:
        return None
    target_label = int(getattr(plane, "label", 0) or 0)
    if target_label > 0:
        return target_label
    # PlaneData.center is local physical space.  The origin is only applied
    # when crossing into world-space VTK geometry.
    ijk = np.rint(np.asarray(plane.center, dtype=float).reshape(3) / (spacing + 1e-12)).astype(int)
    ijk = np.clip(ijk, 0, np.array(np.asarray(branch_labels_3d).shape) - 1)
    return int(np.asarray(branch_labels_3d)[ijk[0], ijk[1], ijk[2]])


def _build_mask_phase_lookup(mask4d):
    mask4d = _ensure_mask4d(mask4d)
    representatives = {}
    lookup = []
    for tidx in range(int(mask4d.shape[3])):
        mask_t = mask4d[..., tidx]
        mask_key = mask_t.tobytes()
        rep_t = representatives.get(mask_key)
        if rep_t is None:
            rep_t = int(tidx)
            representatives[mask_key] = rep_t
        lookup.append(rep_t)
    return lookup


def _build_plane_support_mesh(mask_xyz, spacing, origin):
    """Build the thresholded mask mesh once for all planes using this mask."""
    mask_xyz = np.asarray(mask_xyz, dtype=bool)
    if not np.any(mask_xyz):
        return None
    # Only occupied cells are retained by thresholding. Build their bounding
    # box directly while preserving the full volume's voxel ids and origin.
    bounds = [
        np.flatnonzero(np.any(mask_xyz, axis=tuple(other for other in range(3) if other != axis)))
        for axis in range(3)
    ]
    lower = np.asarray([int(values[0]) for values in bounds])
    upper = np.asarray([int(values[-1]) + 1 for values in bounds])
    local_mask = mask_xyz[tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))]
    local_origin = np.asarray(origin, dtype=float) + lower * np.asarray(spacing, dtype=float)
    grid = create_uniform_field_grid(local_mask.astype(np.uint8), spacing, origin=local_origin, name="mask")
    x, y, z = [np.arange(int(lo), int(hi), dtype=np.int32) for lo, hi in zip(lower, upper)]
    cell_ids = x[:, None, None] + mask_xyz.shape[0] * (
        y[None, :, None] + mask_xyz.shape[1] * z[None, None, :]
    )
    grid.cell_data["_cell_id"] = cell_ids.ravel(order="F")
    mesh = grid.threshold(0.1, scalars="mask")
    if mesh is None or mesh.n_cells == 0:
        return None
    return mesh


def _build_plane_support_mesh_cache(mask4d, mask_phase_lookup, spacing, origin):
    mask4d = _ensure_mask4d(mask4d)
    cache = {}
    for rep_t in sorted({int(x) for x in mask_phase_lookup}):
        cache[rep_t] = _build_plane_support_mesh(mask4d[..., rep_t], spacing, origin)
    return cache


def _add_labeled_plane_support_meshes(cache, mask4d, labels4d, planes, spacing, origin):
    """Prebuild support meshes for labeled planes so planes share VTK geometry."""
    if labels4d is None:
        return
    labels = np.asarray(labels4d)
    mask = _ensure_mask4d(mask4d)
    labels_was_3d = labels.ndim == 3
    if labels.ndim == 3:
        labels = labels[..., np.newaxis]
    if labels.ndim != 4 or labels.shape[:3] != mask.shape[:3]:
        return
    requested = sorted({
        int(getattr(plane, "segmentation_label", 0) or 0)
        for plane in planes
        if int(getattr(plane, "segmentation_label", 0) or 0) > 0
    })
    for label in requested:
        representatives = {}
        for tidx in range(int(mask.shape[3])):
            label_t = 0 if labels_was_3d else min(int(tidx), int(labels.shape[3]) - 1)
            label_mask = mask[..., tidx] & (labels[..., label_t] == int(label))
            token = label_mask.tobytes()
            rep_t = representatives.setdefault(token, int(tidx))
            mesh_key = ("seg4d", rep_t, int(label))
            if mesh_key not in cache:
                cache[mesh_key] = _build_plane_support_mesh(label_mask, spacing, origin)
            cache[("seg4d", int(tidx), int(label))] = cache[mesh_key]
            cache[("seg3d", int(tidx), int(label))] = cache[("seg4d", int(tidx), int(label))]
            cache[("seg_phase", int(tidx), int(label))] = rep_t


def filter_planes_by_branch_support(
    planes,
    mask4d,
    spacing,
    origin,
    branch_labels_3d=None,
    segmentation_labels_3d=None,
    *,
    branch_label_offset=0,
    support_cache=None,
):
    """Drop generated planes whose requested branch has no slice support.

    A zero numerical flow value is not sufficient evidence that a plane is
    invalid: a real vessel can have low or cancelling flow.  This helper only
    rejects a plane when its branch-filtered cross-section has no cells in any
    representative mask phase.  The returned QC keeps the original local
    plane index so callers can report stable diagnostics without renumbering
    graph paths.
    """
    planes = list(planes or [])
    mask = _ensure_mask4d(mask4d)
    if not planes:
        return [], []
    # Without a branch volume there is no safe way to decide ownership.  Keep
    # all planes and make the reason explicit for callers that want to expose
    # QC details.
    branch_array = None if branch_labels_3d is None else np.asarray(branch_labels_3d)
    if branch_array is None or branch_array.shape != mask.shape[:3] or not np.any(branch_array > 0):
        reason = "branch_labels_unavailable" if branch_array is None else "branch_labels_empty_or_mismatched"
        return list(planes), [
            {
                "plane_index": int(i),
                "path_index": int(getattr(plane, "path_index", -1)),
                "valid": True,
                "reason": reason,
                "supported_phase_count": 0,
                "max_cell_count": 0,
            }
            for i, plane in enumerate(planes)
        ]

    cache = support_cache if isinstance(support_cache, dict) else {}
    branch_grid = cache.get("branch_grid")
    phase_lookup = cache.get("phase_lookup")
    support_mesh_cache = cache.get("support_mesh_cache")
    if branch_grid is None:
        branch_grid = _build_branch_grid(branch_array, spacing, origin)
        cache["branch_grid"] = branch_grid
    if phase_lookup is None:
        phase_lookup = _build_mask_phase_lookup(mask)
        cache["phase_lookup"] = phase_lookup
    # Plane eligibility only needs to know whether the branch is supported in
    # at least one cardiac phase.  A single union probe mask is sufficient and
    # avoids constructing one VTK mesh per temporal phase during generation;
    # the metric stage still performs the exact per-phase sampling later.
    probe_mask = cache.get("probe_mask")
    if probe_mask is None:
        probe_mask = np.any(mask, axis=3)
        cache["probe_mask"] = np.asarray(probe_mask, dtype=bool)
    probe_mesh = cache.get("probe_mesh")
    if probe_mesh is None:
        probe_mesh = _build_plane_support_mesh(probe_mask, spacing, origin)
        cache["probe_mesh"] = probe_mesh
    segmentation_mesh_cache = cache.setdefault("segmentation_mesh_cache", {})
    seg3d = None
    if segmentation_labels_3d is not None:
        seg3d = np.asarray(segmentation_labels_3d)
        if seg3d.ndim == 4:
            seg3d = seg3d[..., 0]
        if seg3d.shape != mask.shape[:3]:
            seg3d = None

    valid_planes = []
    qc = []
    for plane_index, plane in enumerate(planes):
        path_index = int(getattr(plane, "path_index", -1))
        target_label = int(getattr(plane, "label", 0) or 0)
        if int(branch_label_offset) and path_index >= 0:
            target_label = path_index + 1 + int(branch_label_offset)
        plane_seg_label = int(getattr(plane, "segmentation_label", 0) or 0)
        supported_phase_count = 0
        max_cell_count = 0
        mask_t = np.asarray(probe_mask, dtype=bool)
        if plane_seg_label > 0 and seg3d is not None:
            mask_t = mask_t & (seg3d == plane_seg_label)
            mesh_key = int(plane_seg_label)
            if mesh_key not in segmentation_mesh_cache:
                segmentation_mesh_cache[mesh_key] = _build_plane_support_mesh(
                    mask_t, spacing, origin
                )
            support_mesh = segmentation_mesh_cache.get(mesh_key)
        else:
            support_mesh = probe_mesh
        spec = _build_plane_slice_spec(
            mask_t,
            plane,
            spacing,
            origin,
            branch_grid=branch_grid,
            target_label=target_label,
            select_connected=True,
            support_mesh=support_mesh,
        )
        cell_count = int(len(spec.get("cell_ids", []))) if spec is not None else 0
        max_cell_count = cell_count
        supported_phase_count = 1 if cell_count > 0 else 0
        valid = supported_phase_count > 0
        record = {
            "plane_index": int(plane_index),
            "path_index": int(path_index),
            "branch_label": int(target_label),
            "segmentation_label": int(plane_seg_label),
            "valid": bool(valid),
            "reason": "" if valid else "no_branch_support",
            "supported_phase_count": int(supported_phase_count),
            "phase_count": int(len(set(phase_lookup))),
            "max_cell_count": int(max_cell_count),
        }
        qc.append(record)
        if valid:
            valid_planes.append(plane)
    return valid_planes, qc


def _build_plane_slice_region(
    mask_xyz,
    plane,
    spacing,
    origin,
    branch_grid=None,
    target_label=None,
    *,
    select_connected=False,
    support_mesh=None,
    frame_index=0,
):
    mask_xyz = np.asarray(mask_xyz, dtype=bool)
    if support_mesh is None and not np.any(mask_xyz):
        return None
    operations = list((getattr(plane, "roi_edit_operations", {}) or {}).get(str(int(frame_index)), []) or [])
    replacement_polygon = None
    for operation in operations:
        if str(operation.get("mode", "")) != "replace":
            continue
        candidate = np.asarray(operation.get("polygon", []), dtype=float)
        if candidate.ndim == 2 and candidate.shape[1] == 2 and len(candidate) >= 3:
            replacement_polygon = candidate

    mesh = support_mesh
    if replacement_polygon is not None:
        normal = np.asarray(plane.normal, dtype=float).reshape(3)
        normal /= np.linalg.norm(normal) + 1e-12
        reference = np.eye(3, dtype=float)[int(np.argmin(np.abs(normal)))]
        axis_u = np.cross(reference, normal); axis_u /= np.linalg.norm(axis_u) + 1e-12
        axis_v = np.cross(normal, axis_u); axis_v /= np.linalg.norm(axis_v) + 1e-12
        center_local = np.asarray(plane.center, dtype=float).reshape(3)
        polygon_xyz = (
            center_local.reshape(1, 3)
            + replacement_polygon[:, :1] * axis_u.reshape(1, 3)
            + replacement_polygon[:, 1:2] * axis_v.reshape(1, 3)
        )
        voxel_spacing = np.asarray(spacing, dtype=float).reshape(3)
        # Keep a generous local halo around the polygon. This makes a support
        # mesh prewarmed for the current contour reusable for nearby outward
        # edits without changing the selected cells (the polygon filter below
        # still defines the exact ROI).
        margin = max(float(np.linalg.norm(voxel_spacing)), 4.0 * float(np.min(voxel_spacing)))
        lower = np.floor((np.min(polygon_xyz, axis=0) - margin) / spacing).astype(int)
        upper = np.ceil((np.max(polygon_xyz, axis=0) + margin) / spacing).astype(int) + 1
        lower = np.clip(lower, 0, np.asarray(mask_xyz.shape) - 1)
        upper = np.clip(upper, lower + 1, np.asarray(mask_xyz.shape))
        expanded_mask = np.zeros_like(mask_xyz, dtype=bool)
        expanded_mask[
            lower[0]:upper[0],
            lower[1]:upper[1],
            lower[2]:upper[2],
        ] = True
        cache_key = (
            tuple(int(x) for x in mask_xyz.shape),
            tuple(int(x) for x in lower),
            tuple(int(x) for x in upper),
            tuple(np.round(np.asarray(spacing, dtype=float).reshape(3), 6)),
            tuple(np.round(np.asarray(origin, dtype=float).reshape(3), 6)),
        )
        mesh = _LOCAL_SUPPORT_MESH_CACHE.get(cache_key)
        if mesh is None:
            # Reuse a cached mesh whose local box contains this edit's box.
            # The mesh carries global cell ids, so extracting the narrower
            # polygon region remains exact.
            for candidate_key, candidate_mesh in _LOCAL_SUPPORT_MESH_CACHE.items():
                if candidate_key[0] != cache_key[0] or candidate_key[3:] != cache_key[3:]:
                    continue
                candidate_lower = np.asarray(candidate_key[1], dtype=int)
                candidate_upper = np.asarray(candidate_key[2], dtype=int)
                if np.all(candidate_lower <= lower) and np.all(candidate_upper >= upper):
                    mesh = candidate_mesh
                    break
        if mesh is None:
            mesh = _build_plane_support_mesh(expanded_mask, spacing, origin)
            if mesh is not None:
                if len(_LOCAL_SUPPORT_MESH_CACHE) >= _LOCAL_SUPPORT_MESH_CACHE_MAX:
                    _LOCAL_SUPPORT_MESH_CACHE.pop(next(iter(_LOCAL_SUPPORT_MESH_CACHE)))
                _LOCAL_SUPPORT_MESH_CACHE[cache_key] = mesh
    if mesh is None:
        mesh = _build_plane_support_mesh(mask_xyz, spacing, origin)
    if mesh is None or mesh.n_cells == 0:
        return None
    plane_center_world = (
        np.asarray(plane.center, dtype=float).reshape(3)
        + np.asarray(origin, dtype=float).reshape(3)
    )
    pg = mesh.slice(normal=np.asarray(plane.normal, dtype=float), origin=plane_center_world)
    if pg is None or pg.n_cells == 0:
        return None
    pg = pg.compute_cell_sizes(area=True)
    if replacement_polygon is None and branch_grid is not None and target_label is not None and int(target_label) > 0:
        centers = pg.cell_centers().sample(branch_grid)
        bid = np.asarray(centers.point_data.get("branch_id", []))
        if len(bid) == 0:
            return None
        keep = np.where(bid == int(target_label))[0]
        if len(keep) == 0:
            return None
        pg = pg.extract_cells(keep)
        if pg is None or pg.n_cells == 0:
            return None
        pg = pg.compute_cell_sizes(area=True)
    # Lasso edits may only operate on the same branch-supported cut cells as
    # the automatic metric ROI; the later polygon limit narrows that base ROI.
    operation_pg = pg
    roi_polygon = np.asarray(getattr(plane, "roi_polygon_uv_mm", []) or [], dtype=float)
    if roi_polygon.size and replacement_polygon is None:
        if roi_polygon.ndim != 2 or roi_polygon.shape[1] != 2 or len(roi_polygon) < 3:
            return None
        normal = np.asarray(plane.normal, dtype=float).reshape(3)
        normal = normal / (np.linalg.norm(normal) + 1e-12)
        reference = np.eye(3, dtype=float)[int(np.argmin(np.abs(normal)))]
        axis_u = np.cross(reference, normal)
        axis_u = axis_u / (np.linalg.norm(axis_u) + 1e-12)
        axis_v = np.cross(normal, axis_u)
        axis_v = axis_v / (np.linalg.norm(axis_v) + 1e-12)
        centers = np.asarray(pg.cell_centers().points, dtype=float)
        relative = centers - plane_center_world.reshape(1, 3)
        points_uv = np.column_stack((np.dot(relative, axis_u), np.dot(relative, axis_v)))
        inside = _points_in_polygon(points_uv, roi_polygon)
        keep = np.where(inside)[0]
        if len(keep) == 0:
            return None
        pg = pg.extract_cells(keep)
        if pg is None or pg.n_cells == 0:
            return None
        pg = pg.compute_cell_sizes(area=True)
    if select_connected and not operations:
        pg = _select_connected_region(pg, ref_point=plane_center_world)
        if pg is None or pg.n_cells == 0:
            return None
    if operations:
        normal = np.asarray(plane.normal, dtype=float).reshape(3)
        normal /= np.linalg.norm(normal) + 1e-12
        reference = np.eye(3, dtype=float)[int(np.argmin(np.abs(normal)))]
        axis_u = np.cross(reference, normal); axis_u /= np.linalg.norm(axis_u) + 1e-12
        axis_v = np.cross(normal, axis_u); axis_v /= np.linalg.norm(axis_v) + 1e-12
        base_cell_ids = np.asarray(pg.cell_data.get("_cell_id", []), dtype=np.int64).reshape(-1)
        all_centers = np.asarray(operation_pg.cell_centers().points, dtype=float)
        all_relative = all_centers - plane_center_world.reshape(1, 3)
        all_points_uv = np.column_stack((all_relative @ axis_u, all_relative @ axis_v))
        selected_ids = set(int(x) for x in base_cell_ids.tolist())
        for operation in operations:
            polygon = np.asarray(operation.get("polygon", []), dtype=float)
            if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
                continue
            hit = _points_in_polygon(all_points_uv, polygon)
            hit_ids = set(int(x) for x in np.asarray(operation_pg.cell_data.get("_cell_id", []), dtype=np.int64).reshape(-1)[hit].tolist())
            operation_mode = str(operation.get("mode", "remove"))
            if operation_mode == "replace":
                selected_ids = hit_ids
            elif operation_mode == "add":
                selected_ids |= hit_ids
            else:
                selected_ids -= hit_ids
        all_ids = np.asarray(operation_pg.cell_data.get("_cell_id", []), dtype=np.int64).reshape(-1)
        pg = operation_pg.extract_cells(np.where(np.isin(all_ids, list(selected_ids)))[0])
        if pg is None or pg.n_cells == 0:
            return None
        pg = pg.compute_cell_sizes(area=True)
    return pg


def _points_in_polygon(points_xy, polygon_xy):
    points = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    polygon = np.asarray(polygon_xy, dtype=float).reshape(-1, 2)
    if len(points) == 0 or len(polygon) < 3:
        return np.zeros(len(points), dtype=bool)
    point_x = points[:, 0]
    point_y = points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    previous = polygon[-1]
    for current in polygon:
        x0, y0 = float(previous[0]), float(previous[1])
        x1, y1 = float(current[0]), float(current[1])
        crosses = (y0 > point_y) != (y1 > point_y)
        x_cross = (x1 - x0) * (point_y - y0) / (y1 - y0 + 1e-15) + x0
        inside ^= crosses & (point_x < x_cross)
        previous = current
    return inside


def _plane_roi_point_mask(points_world, plane, origin):
    points = np.asarray(points_world, dtype=float).reshape(-1, 3)
    roi_polygon = np.asarray(getattr(plane, "roi_polygon_uv_mm", []) or [], dtype=float)
    if len(points) == 0 or roi_polygon.size == 0:
        return np.ones(len(points), dtype=bool)
    if roi_polygon.ndim != 2 or roi_polygon.shape[1] != 2 or len(roi_polygon) < 3:
        return np.zeros(len(points), dtype=bool)
    normal = np.asarray(plane.normal, dtype=float).reshape(3)
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    reference = np.eye(3, dtype=float)[int(np.argmin(np.abs(normal)))]
    axis_u = np.cross(reference, normal)
    axis_u = axis_u / (np.linalg.norm(axis_u) + 1e-12)
    axis_v = np.cross(normal, axis_u)
    axis_v = axis_v / (np.linalg.norm(axis_v) + 1e-12)
    center_world = np.asarray(plane.center, dtype=float).reshape(3) + np.asarray(origin, dtype=float).reshape(3)
    relative = points - center_world.reshape(1, 3)
    points_uv = np.column_stack((np.dot(relative, axis_u), np.dot(relative, axis_v)))
    return _points_in_polygon(points_uv, roi_polygon)


def _build_plane_slice_spec(
    mask_xyz,
    plane,
    spacing,
    origin,
    branch_grid=None,
    target_label=None,
    *,
    select_connected=False,
    support_mesh=None,
    frame_index=0,
):
    pg = _build_plane_slice_region(
        mask_xyz,
        plane,
        spacing,
        origin,
        branch_grid=branch_grid,
        target_label=target_label,
        select_connected=select_connected,
        support_mesh=support_mesh,
        frame_index=frame_index,
    )
    if pg is None or pg.n_cells == 0:
        return None
    cell_ids = np.asarray(pg.cell_data.get("_cell_id", []), dtype=np.int64).reshape(-1)
    if cell_ids.size != int(pg.n_cells):
        return None
    areas = np.asarray(pg.cell_data.get("Area", np.ones(pg.n_cells, dtype=float)), dtype=float).reshape(-1)
    if areas.size != int(pg.n_cells):
        areas = np.ones(int(pg.n_cells), dtype=float)
    return {
        "cell_ids": cell_ids,
        "voxel_indices": np.unravel_index(cell_ids, mask_xyz.shape, order="F"),
        "areas": areas,
    }
def _get_cached_plane_slice_spec(slice_cache, cache_key, mask_xyz, plane, spacing, origin,
                                 branch_grid=None, target_label=None, *, select_connected=False,
                                 support_mesh=None, frame_index=0):
    if cache_key not in slice_cache:
        slice_cache[cache_key] = _build_plane_slice_spec(
            mask_xyz,
            plane,
            spacing,
            origin,
            branch_grid=branch_grid,
            target_label=target_label,
            select_connected=select_connected,
            support_mesh=support_mesh,
            frame_index=frame_index,
        )
    return slice_cache[cache_key]


def _sample_field_from_slice_spec(field_t, mask_shape, field_name, slice_spec):
    if slice_spec is None:
        return None
    field_arr = np.asarray(field_t)
    if tuple(field_arr.shape[:3]) != tuple(mask_shape):
        raise ValueError(f"{field_name} spatial shape {field_arr.shape[:3]} does not match mask {mask_shape}")
    voxel_indices = slice_spec.get("voxel_indices")
    if voxel_indices is None:
        cell_ids = np.asarray(slice_spec["cell_ids"], dtype=np.int64).reshape(-1)
        voxel_indices = np.unravel_index(cell_ids, mask_shape, order="F")
    if field_arr.ndim == 3:
        return field_arr[voxel_indices]
    if field_arr.ndim == 4 and field_arr.shape[-1] in (1, 3):
        payload = field_arr if field_arr.shape[-1] != 1 else field_arr[..., 0]
        if payload.ndim == 3:
            return payload[voxel_indices]
        return payload[voxel_indices]
    raise ValueError(f"{field_name} must be XYZ, XYZT-slice scalar, or XYZV, got {field_arr.shape}")


def _extract_plane_field_region(mask_xyz, field_t, plane, spacing, origin, field_name, branch_grid=None, target_label=None):
    mask_xyz = np.asarray(mask_xyz, dtype=bool)
    if not np.any(mask_xyz):
        return None
    grid = create_uniform_field_grid(mask_xyz.astype(np.uint8), spacing, origin=origin, name="mask")
    field_arr = np.asarray(field_t)
    if field_arr.shape[:3] != mask_xyz.shape:
        raise ValueError(f"{field_name} spatial shape {field_arr.shape[:3]} does not match mask {mask_xyz.shape}")
    if field_arr.ndim == 3:
        grid.cell_data[field_name] = field_arr.reshape(-1, order="F")
    elif field_arr.ndim == 4 and field_arr.shape[-1] in (1, 3):
        payload = field_arr if field_arr.shape[-1] != 1 else field_arr[..., 0]
        if payload.ndim == 3:
            grid.cell_data[field_name] = payload.reshape(-1, order="F")
        else:
            grid.cell_data[field_name] = payload.reshape(-1, payload.shape[-1], order="F")
    else:
        raise ValueError(f"{field_name} must be XYZ, XYZT-slice scalar, or XYZV, got {field_arr.shape}")
    mesh = grid.threshold(0.1, scalars="mask")
    if mesh is None or mesh.n_cells == 0:
        return None
    plane_center_world = (
        np.asarray(plane.center, dtype=float).reshape(3)
        + np.asarray(origin, dtype=float).reshape(3)
    )
    pg = mesh.slice(normal=np.asarray(plane.normal, dtype=float), origin=plane_center_world)
    if pg is None or pg.n_cells == 0:
        return None
    pg = pg.compute_cell_sizes(area=True)
    if branch_grid is not None and target_label is not None and int(target_label) > 0:
        centers = pg.cell_centers().sample(branch_grid)
        bid = np.asarray(centers.point_data.get("branch_id", []))
        if len(bid) == 0:
            return None
        keep = np.where(bid == int(target_label))[0]
        if len(keep) == 0:
            return None
        pg = pg.extract_cells(keep)
        if pg is None or pg.n_cells == 0:
            return None
        pg = pg.compute_cell_sizes(area=True)
    return pg


def _nanpercentile_safe(values, q):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0
    return float(np.nanpercentile(finite, q))


def _weighted_mean(values, weights):
    vals = np.asarray(values, dtype=float).reshape(-1)
    wts = np.asarray(weights, dtype=float).reshape(-1)
    if vals.size == 0 or wts.size != vals.size:
        return 0.0
    finite = np.isfinite(vals) & np.isfinite(wts) & (wts > 0)
    if not np.any(finite):
        return 0.0
    vals = vals[finite]
    wts = wts[finite]
    denom = float(np.sum(wts))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(vals * wts) / denom)


def _append_summary(metric, prefix, series):
    arr = np.asarray(series, dtype=float).reshape(-1)
    metric[f"{prefix}_t"] = [float(x) for x in arr.tolist()]
    metric[prefix] = float(np.mean(arr)) if arr.size else 0.0


def summarize_plane_derived_metrics(plane, mask4d, spacing, origin, branch_labels_3d=None,
                                    tke_array=None, pressure_gradient_array=None,
                                    relative_pressure_array=None, wss_surfaces=None,
                                    branch_grid=None, mask_phase_lookup=None,
                                    support_mesh_cache=None):
    mask4d = _ensure_mask4d(mask4d)
    spacing = np.asarray(spacing, dtype=float).reshape(3)
    origin = np.asarray(origin, dtype=float).reshape(3)
    Nt = int(mask4d.shape[3])
    branch_grid = branch_grid if branch_grid is not None else _build_branch_grid(branch_labels_3d, spacing, origin)
    target_label = _target_label_for_plane(plane, branch_labels_3d, spacing, origin)
    normal = np.asarray(plane.normal, dtype=float).reshape(3)
    normal = normal / (np.linalg.norm(normal) + 1e-12)

    summary = {}
    pixelwise = {"timepoints": []}

    has_tke = tke_array is not None
    has_pressure_gradient = pressure_gradient_array is not None
    has_relative_pressure = relative_pressure_array is not None
    has_wss = bool(wss_surfaces)
    needs_volume_slice = has_tke or has_pressure_gradient or has_relative_pressure

    if has_tke:
        tke_array = _prepare_tke_array(mask4d, tke_array=tke_array, sigma=None)
    if has_pressure_gradient:
        pressure_gradient_array = np.asarray(pressure_gradient_array, dtype=np.float32)
        if pressure_gradient_array.ndim != 5 or pressure_gradient_array.shape[-1] != 3:
            raise ValueError(f"pressure_gradient_array must be XYZTV, got {pressure_gradient_array.shape}")
    if has_relative_pressure:
        relative_pressure_array = np.asarray(relative_pressure_array, dtype=np.float32)
        if relative_pressure_array.ndim != 4:
            raise ValueError(f"relative_pressure_array must be XYZT, got {relative_pressure_array.shape}")
    if mask_phase_lookup is None:
        mask_phase_lookup = _build_mask_phase_lookup(mask4d) if needs_volume_slice else []
    if support_mesh_cache is None and needs_volume_slice:
        support_mesh_cache = _build_plane_support_mesh_cache(mask4d, mask_phase_lookup, spacing, origin)
    slice_cache = {}

    tke_mean_t = []
    tke_peak_t = []
    tke_p95_t = []
    pg_mag_mean_t = []
    pg_mag_peak_t = []
    pg_mag_p95_t = []
    pg_normal_mean_t = []
    pg_normal_peak_t = []
    pg_normal_p95_t = []
    rp_mean_t = []
    rp_peak_t = []
    rp_p95_t = []
    wss_mean_t = []
    wss_peak_t = []
    wss_p95_t = []

    for tidx in range(Nt):
        entry = {"time_index": int(tidx)}

        tke_series_vals = np.array([], dtype=float)
        pg_mag_vals = np.array([], dtype=float)
        pg_normal_vals = np.array([], dtype=float)
        rp_vals = np.array([], dtype=float)
        areas = np.array([], dtype=float)
        slice_spec = None
        if needs_volume_slice:
            rep_t = int(mask_phase_lookup[tidx])
            slice_spec = _get_cached_plane_slice_spec(
                slice_cache,
                (rep_t, tidx) if getattr(plane, "roi_edit_operations", {}) else rep_t,
                mask4d[..., rep_t],
                plane,
                spacing,
                origin,
                branch_grid=branch_grid,
                target_label=target_label,
                select_connected=False,
                support_mesh=support_mesh_cache.get(rep_t) if support_mesh_cache is not None else None,
                frame_index=tidx,
            )

        if has_tke and slice_spec is not None:
            tke_series_vals = np.asarray(
                _sample_field_from_slice_spec(tke_array[..., tidx], mask4d.shape[:3], "tke", slice_spec),
                dtype=float,
            ).reshape(-1)
            if tke_series_vals.size:
                areas = np.asarray(slice_spec["areas"], dtype=float).reshape(-1)
                entry["cell_area_mm2"] = areas.astype(np.float32)
                entry["tke_J_m3"] = tke_series_vals.astype(np.float32)
                entry["lumen_mask"] = np.ones_like(tke_series_vals, dtype=np.uint8)

        if has_pressure_gradient and slice_spec is not None:
            vec = np.asarray(
                _sample_field_from_slice_spec(
                    pressure_gradient_array[..., tidx, :],
                    mask4d.shape[:3],
                    "pressure_gradient",
                    slice_spec,
                ),
                dtype=float,
            )
            if vec.ndim == 2 and vec.shape[1] == 3:
                if areas.size == 0:
                    areas = np.asarray(slice_spec["areas"], dtype=float).reshape(-1)
                    entry.setdefault("cell_area_mm2", areas.astype(np.float32))
                    entry.setdefault("lumen_mask", np.ones(len(areas), dtype=np.uint8))
                pg_mag_vals = np.linalg.norm(vec, axis=1)
                pg_normal_vals = np.dot(vec, normal)
                entry["pressure_gradient_mag_Pa_m"] = pg_mag_vals.astype(np.float32)
                entry["pressure_gradient_normal_Pa_m"] = pg_normal_vals.astype(np.float32)
                entry["pressure_gradient_vec_Pa_m"] = vec.astype(np.float32)

        if has_relative_pressure and slice_spec is not None:
            vals = np.asarray(
                _sample_field_from_slice_spec(
                    relative_pressure_array[..., tidx],
                    mask4d.shape[:3],
                    "relative_pressure",
                    slice_spec,
                ),
                dtype=float,
            ).reshape(-1)
            if vals.size:
                if areas.size == 0:
                    areas = np.asarray(slice_spec["areas"], dtype=float).reshape(-1)
                    entry.setdefault("cell_area_mm2", areas.astype(np.float32))
                    entry.setdefault("lumen_mask", np.ones(len(areas), dtype=np.uint8))
                rp_vals = vals
                entry["relative_pressure_Pa"] = rp_vals.astype(np.float32)

        if tke_series_vals.size:
            tke_mean_t.append(_weighted_mean(tke_series_vals, areas))
            tke_peak_t.append(float(np.max(tke_series_vals)))
            tke_p95_t.append(_nanpercentile_safe(tke_series_vals, 95.0))
        else:
            tke_mean_t.append(0.0)
            tke_peak_t.append(0.0)
            tke_p95_t.append(0.0)

        if pg_mag_vals.size:
            pg_mag_mean_t.append(_weighted_mean(pg_mag_vals, areas))
            pg_mag_peak_t.append(float(np.max(pg_mag_vals)))
            pg_mag_p95_t.append(_nanpercentile_safe(pg_mag_vals, 95.0))
            pg_normal_mean_t.append(_weighted_mean(pg_normal_vals, areas))
            pg_normal_peak_t.append(float(np.max(np.abs(pg_normal_vals))))
            pg_normal_p95_t.append(_nanpercentile_safe(np.abs(pg_normal_vals), 95.0))
        else:
            pg_mag_mean_t.append(0.0)
            pg_mag_peak_t.append(0.0)
            pg_mag_p95_t.append(0.0)
            pg_normal_mean_t.append(0.0)
            pg_normal_peak_t.append(0.0)
            pg_normal_p95_t.append(0.0)

        if rp_vals.size:
            rp_mean_t.append(_weighted_mean(rp_vals, areas))
            rp_peak_t.append(float(np.max(np.abs(rp_vals))))
            rp_p95_t.append(_nanpercentile_safe(np.abs(rp_vals), 95.0))
        else:
            rp_mean_t.append(0.0)
            rp_peak_t.append(0.0)
            rp_p95_t.append(0.0)

        wss_vals = np.array([], dtype=float)
        surf = None if not has_wss or tidx >= len(wss_surfaces) else wss_surfaces[tidx]
        if surf is not None and getattr(surf, "n_points", 0) > 0:
            try:
                wall = surf.slice(
                    normal=normal,
                    origin=np.asarray(plane.center, dtype=float).reshape(3) + origin,
                )
            except Exception:
                wall = None
            if wall is not None and getattr(wall, "n_points", 0) > 0:
                if "wss" not in wall.point_data and "wss" in wall.cell_data:
                    vals = np.asarray(wall.cell_data.get("wss", []), dtype=float).reshape(-1)
                    sample_points = np.asarray(wall.cell_centers().points, dtype=float)
                else:
                    vals = np.asarray(wall.point_data.get("wss", []), dtype=float).reshape(-1)
                    sample_points = np.asarray(wall.points, dtype=float)
                if len(vals) == len(sample_points):
                    vals = vals[_plane_roi_point_mask(sample_points, plane, origin)]
                wss_vals = vals[np.isfinite(vals)]
        if wss_vals.size:
            wss_mean_t.append(float(np.mean(wss_vals)))
            wss_peak_t.append(float(np.max(wss_vals)))
            wss_p95_t.append(_nanpercentile_safe(wss_vals, 95.0))
        else:
            wss_mean_t.append(0.0)
            wss_peak_t.append(0.0)
            wss_p95_t.append(0.0)

        pixelwise["timepoints"].append(entry)

    if has_tke:
        _append_summary(summary, "tke_mean_J_m3", tke_mean_t)
        _append_summary(summary, "tke_peak_J_m3", tke_peak_t)
        _append_summary(summary, "tke_p95_J_m3", tke_p95_t)
    if has_pressure_gradient:
        _append_summary(summary, "pressure_gradient_mag_mean_Pa_m", pg_mag_mean_t)
        _append_summary(summary, "pressure_gradient_mag_peak_Pa_m", pg_mag_peak_t)
        _append_summary(summary, "pressure_gradient_mag_p95_Pa_m", pg_mag_p95_t)
        _append_summary(summary, "pressure_gradient_normal_mean_Pa_m", pg_normal_mean_t)
        _append_summary(summary, "pressure_gradient_normal_peak_Pa_m", pg_normal_peak_t)
        _append_summary(summary, "pressure_gradient_normal_p95_Pa_m", pg_normal_p95_t)
    if has_relative_pressure:
        _append_summary(summary, "relative_pressure_mean_Pa", rp_mean_t)
        _append_summary(summary, "relative_pressure_peak_Pa", rp_peak_t)
        _append_summary(summary, "relative_pressure_p95_Pa", rp_p95_t)
    if has_wss:
        _append_summary(summary, "wss_wall_mean_Pa", wss_mean_t)
        _append_summary(summary, "wss_wall_peak_Pa", wss_peak_t)
        _append_summary(summary, "wss_wall_p95_Pa", wss_p95_t)
    return summary, pixelwise


def _augment_plane_metrics_serial(plane_metrics, planes, mask4d, spacing, origin, branch_labels_3d=None,
                                 tke_array=None, pressure_gradient_array=None,
                                 relative_pressure_array=None, wss_surfaces=None, progress_callback=None):
    mask4d = _ensure_mask4d(mask4d)
    shared_branch_grid = _build_branch_grid(branch_labels_3d, spacing, origin)
    mask_phase_lookup = _build_mask_phase_lookup(mask4d)
    support_mesh_cache = _build_plane_support_mesh_cache(mask4d, mask_phase_lookup, spacing, origin)
    metrics = [dict(m) for m in plane_metrics]
    pixelwise = []
    for idx, metric in enumerate(metrics):
        check_cancelled()
        if idx >= len(planes):
            pixelwise.append({"plane_index": int(idx), "timepoints": []})
            continue
        summary, payload = summarize_plane_derived_metrics(
            planes[idx], mask4d, spacing, origin, branch_labels_3d=branch_labels_3d,
            tke_array=tke_array, pressure_gradient_array=pressure_gradient_array,
            relative_pressure_array=relative_pressure_array,
            wss_surfaces=wss_surfaces,
            branch_grid=shared_branch_grid,
            mask_phase_lookup=mask_phase_lookup,
            support_mesh_cache=support_mesh_cache,
        )
        metric.update(summary)
        payload["plane_index"] = int(idx)
        payload["center"] = np.asarray(planes[idx].center, dtype=float).reshape(3).tolist()
        payload["normal"] = np.asarray(planes[idx].normal, dtype=float).reshape(3).tolist()
        payload["label"] = int(getattr(planes[idx], "label", 0) or 0)
        payload["path_index"] = int(getattr(planes[idx], "path_index", -1))
        pixelwise.append(payload)
        if progress_callback is not None:
            progress_callback({"stage": "plane_derived", "current": idx + 1, "total": len(metrics),
                               "message": f"Sampled derived plane metrics ({idx + 1}/{len(metrics)})"})
    return metrics, pixelwise


def _augment_plane_metrics_chunk(indexed_metrics, planes, mask4d, spacing, origin, branch_labels_3d,
                                 tke_array, pressure_gradient_array, relative_pressure_array, wss_surfaces,
                                 progress_path):
    indices = [index for index, _metric in indexed_metrics]
    local_planes = [planes[index] for index in indices]
    with task_scope(_PlaneProcessToken(progress_path)):
        metrics, payloads = _augment_plane_metrics_serial(
            [metric for _index, metric in indexed_metrics], local_planes, mask4d, spacing, origin,
            branch_labels_3d, tke_array, pressure_gradient_array, relative_pressure_array, wss_surfaces,
            progress_callback=lambda _payload: _mark_plane_process_progress(progress_path),
        )
    for index, payload in zip(indices, payloads):
        payload["plane_index"] = int(index)
    return list(zip(indices, metrics, payloads))


def augment_plane_metrics_with_derived(plane_metrics, planes, mask4d, spacing, origin, branch_labels_3d=None,
                                       tke_array=None, pressure_gradient_array=None,
                                       relative_pressure_array=None, wss_surfaces=None, *,
                                       use_multithread=False, max_workers=None, progress_callback=None):
    mask4d = _ensure_mask4d(mask4d)
    count = min(len(plane_metrics), len(planes))
    if max_workers is None:
        max_workers = min(4, count, max(1, os.cpu_count() or 1)) if use_multithread and count * mask4d.shape[3] >= 1920 else 1
    if int(max_workers) <= 1 or count != len(plane_metrics):
        return _augment_plane_metrics_serial(
            plane_metrics, planes, mask4d, spacing, origin, branch_labels_3d,
            tke_array, pressure_gradient_array, relative_pressure_array, wss_surfaces, progress_callback,
        )
    import tempfile
    from joblib import Parallel, delayed
    workers = min(int(max_workers), count)
    indexed = list(enumerate(plane_metrics))
    with tempfile.TemporaryDirectory(prefix="autoflow_plane_derived_") as progress_dir:
        progress_path = os.path.join(progress_dir, "progress.log")
        open(progress_path, "ab").close()
        def calculate():
            return Parallel(n_jobs=workers, backend="loky", max_nbytes="10M", mmap_mode="r")(
                delayed(_augment_plane_metrics_chunk)(
                    indexed[offset::workers], planes, mask4d, spacing, origin, branch_labels_3d,
                    tke_array, pressure_gradient_array, relative_pressure_array, wss_surfaces, progress_path,
                ) for offset in range(workers)
            )
        chunks = _wait_plane_processes(calculate, progress_path, count, progress_callback, stage="plane_derived")
    metrics = [None] * count
    payloads = [None] * count
    for chunk in chunks:
        for index, metric, payload in chunk:
            metrics[index] = metric
            payloads[index] = payload
    return metrics, payloads


def save_plane_pixelwise_h5(path, plane_payloads, rr_ms=None, source_format=""):
    """Publish a complete file, preserving the previous export on cancellation."""
    import uuid
    import stat
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temporary = os.path.join(directory, f".autoflow_plane_{uuid.uuid4().hex}.h5")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    os.close(fd)
    try:
        _write_plane_pixelwise_h5(temporary, plane_payloads, rr_ms, source_format)
        check_cancelled()
        if os.path.isfile(path):
            os.chmod(temporary, stat.S_IMODE(os.stat(path).st_mode))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _write_plane_pixelwise_h5(path, plane_payloads, rr_ms=None, source_format=""):
    with h5py.File(path, "w") as h5:
        meta = h5.create_group("meta")
        meta.create_dataset("version", data=np.bytes_("1.0"))
        meta.create_dataset("source_format", data=np.bytes_(str(source_format or "")))
        if rr_ms is not None:
            meta.create_dataset("rr_ms", data=float(rr_ms))
        planes_group = h5.create_group("planes")
        for export_index, payload in enumerate(plane_payloads):
            report_progress({"stage": "plane_pixelwise_export", "current": export_index,
                             "total": len(plane_payloads) if hasattr(plane_payloads, "__len__") else 0,
                             "message": f"Writing pixelwise plane {export_index + 1}"})
            plane_idx = int(payload.get("plane_index", len(planes_group)))
            grp = planes_group.create_group(f"{plane_idx:03d}")
            grp.create_dataset("center_xyz", data=np.asarray(payload.get("center", [0.0, 0.0, 0.0]), dtype=np.float32))
            grp.create_dataset("normal_xyz", data=np.asarray(payload.get("normal", [1.0, 0.0, 0.0]), dtype=np.float32))
            grp.attrs["label"] = int(payload.get("label", 0))
            grp.attrs["path_index"] = int(payload.get("path_index", -1))
            timepoints = payload.get("timepoints", [])
            times_grp = grp.create_group("timepoints")
            for entry in timepoints:
                tidx = int(entry.get("time_index", len(times_grp)))
                tgrp = times_grp.create_group(f"{tidx:03d}")
                for key, value in entry.items():
                    if key == "time_index" or value is None:
                        continue
                    arr = np.asarray(value)
                    if arr.dtype.kind in {"U", "O"}:
                        continue
                    tgrp.create_dataset(key, data=arr, compression="gzip")


def compute_tke_array_from_sigma(sigma, rho=1060.0):
    return (
        0.5
        * float(rho)
        * np.sum((np.asarray(sigma, dtype=float) / 100.0) ** 2, axis=-1)
    ).astype(np.float32)


def _prepare_tke_array(mask4d, tke_array=None, sigma=None, rho=1060.0):
    mask4d = _ensure_mask4d(mask4d)
    Nt = int(mask4d.shape[-1])

    if tke_array is None:
        if sigma is None:
            raise ValueError("tke_array or sigma is required")
        tke_array = compute_tke_array_from_sigma(sigma, rho=rho)

    tke_array = np.asarray(tke_array, dtype=np.float32)
    if tke_array.ndim == 3:
        tke_array = np.repeat(tke_array[..., None], Nt, axis=3)
    elif tke_array.ndim == 4 and tke_array.shape[3] == 1 and Nt > 1:
        tke_array = np.repeat(tke_array, Nt, axis=3)
    elif tke_array.ndim != 4:
        raise ValueError(f"tke_array must be XYZ or XYZT, got {tke_array.shape}")
    if tke_array.shape[3] != Nt:
        raise ValueError(f"tke_array time dimension {tke_array.shape[3]} does not match mask {Nt}")

    tke_array = tke_array * mask4d.astype(np.float32)
    return tke_array


def compute_tke_metrics(mask4d, spacing, origin=(0, 0, 0), tke_array=None, sigma=None, rho=1060.0):
    mask4d = _ensure_mask4d(mask4d)
    tke_array = _prepare_tke_array(mask4d, tke_array=tke_array, sigma=sigma, rho=rho)
    tke_peak = np.max(tke_array, axis=3)

    TKE = create_uniform_grid(tke_peak, spacing, origin=origin, name="TKE")
    mesh_union = create_uniform_grid(np.max(mask4d > 0, axis=-1), spacing, origin=origin)
    mesh_union = mesh_union.threshold(0.1)
    TKE = mesh_union.sample(TKE)

    return {
        "tke_volume": TKE,
        "tke_array": tke_array,
        "tke_peak": np.asarray(tke_peak, dtype=np.float32),
    }


def compute_wss_metrics(mask4d, flow, spacing, origin=(0, 0, 0),
                        smoothing_iteration=200, viscosity=4.0,
                        inward_distance=None, parabolic_fitting=True,
                        no_slip_condition=True):
    mask4d = _ensure_mask4d(mask4d)
    flow = np.asarray(flow)
    if flow.ndim != 5 or flow.shape[-1] != 3:
        raise ValueError(f"flow must be XYZTV, got {flow.shape}")
    if flow.shape[:4] != mask4d.shape:
        raise ValueError(f"flow shape {flow.shape[:4]} does not match mask {mask4d.shape}")

    spacing = np.asarray(spacing, dtype=float).reshape(3)
    inward_distance = resolve_wss_inward_distance(spacing, inward_distance)
    origin = np.asarray(origin, dtype=float).reshape(3)
    wss_volume = np.zeros(mask4d.shape, dtype=np.float32)
    mask_phase_lookup = _build_mask_phase_lookup(mask4d)
    surface_cache = {}
    support_cache = {}
    surfs = []
    for showt in range(int(mask4d.shape[-1])):
        report_progress({"stage": "wss", "current": showt, "total": mask4d.shape[3],
                         "message": f"Computing WSS phase {showt + 1}/{mask4d.shape[3]}"})
        rep_t = int(mask_phase_lookup[showt])
        if rep_t not in surface_cache:
            support_cache[rep_t] = create_uniform_grid(
                (mask4d[..., rep_t] > 0).astype(np.uint8), spacing, origin=origin, name="wss_lumen")
            mesh = support_cache[rep_t]
            mesh = mesh.threshold(0.1)
            if mesh is None or mesh.n_cells == 0:
                surface_cache[rep_t] = None
                surfs.append(None)
                continue
            base_surface = _extract_surface(mesh)
            if int(smoothing_iteration) > 0:
                base_surface = base_surface.smooth_taubin(n_iter=int(smoothing_iteration), pass_band=0.1)
            surface_cache[rep_t] = base_surface
        base_surface = surface_cache[rep_t]
        if base_surface is None:
            surfs.append(None)
            continue
        # Preserve double-precision sampling without duplicating the whole
        # multi-phase velocity field just to process one phase at a time.
        flow_t = np.asarray(flow[..., showt, :], dtype=float)
        velocity = create_uniform_vector(
            flow_t[..., 0] / 100.0, flow_t[..., 1] / 100.0,
            flow_t[..., 2] / 100.0, spacing, origin=origin)
        # cal_wss_from_surf writes phase-specific point data, so each phase
        # gets a cheap geometry copy while the expensive surface preparation
        # remains shared for identical masks.
        surf = base_surface.copy(deep=True)
        surf = cal_wss_from_surf(surf, velocity, viscosity=viscosity,
                                 inward_distance=inward_distance,
                                  parabolic_fitting=parabolic_fitting,
                                  no_slip_condition=no_slip_condition,
                                  support_grid=support_cache[rep_t])
        surfs.append(surf)
        if surf.n_points > 0 and "wss" in surf.point_data:
            pts = np.asarray(surf.points, dtype=float)
            vals = np.asarray(surf.point_data["wss"], dtype=np.float32)
            vox = np.rint((pts - origin.reshape(1, 3)) / (spacing.reshape(1, 3) + 1e-12)).astype(int)
            for k in range(3):
                vox[:, k] = np.clip(vox[:, k], 0, mask4d.shape[k] - 1)
            flat = np.ravel_multi_index((vox[:, 0], vox[:, 1], vox[:, 2]), mask4d.shape[:3])
            tgt = wss_volume[..., showt].reshape(-1)
            valid = np.isfinite(vals)
            tgt[flat[~valid]] = np.nan
            np.fmax.at(tgt, flat[valid], vals[valid])

    return {
        "wss_surfaces": surfs,
        "wss_volume": wss_volume,
    }


def _periodic_central_difference(values, dt_s):
    """Central temporal derivative with the cardiac cycle treated as periodic."""
    arr = np.asarray(values)
    if arr.ndim < 2 or arr.shape[-2] <= 1:
        return np.zeros_like(arr, dtype=np.result_type(arr, np.float32))
    dt = max(float(dt_s), 1e-12)
    return (np.roll(arr, -1, axis=-2) - np.roll(arr, 1, axis=-2)) / (2.0 * dt)


def _finite_percentile_abs(values, q, default=0.0):
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float(default)
    return float(np.percentile(np.abs(finite), float(q)))


def _normalize_pressure_method(method):
    token = str(method or "ppe").strip().lower()
    if token in {
        "ls", "least_squares", "least-squares", "least squares",
        "ppe", "poisson", "poisson_pressure_equation",
    }:
        return "ppe"
    if token in {"ste", "stokes", "stokes_estimator", "stokes-estimator"}:
        return "ste"
    raise ValueError(f"unsupported pressure reconstruction method: {method}")


def _neighbor_shifts():
    return (
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    )


def _solve_reconstruction_system(system, rhs, *, tol=1e-5, max_iter=2000):
    matrix = system["matrix"]
    n = int(matrix.shape[0])
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    rhs = np.asarray(rhs, dtype=np.float64).reshape(n).copy()
    anchors = np.asarray(system.get("anchor_indices", [system.get("anchor_index", 0)]), dtype=int)
    rhs[anchors] = 0.0
    if n == 1:
        return np.zeros(1, dtype=np.float32)

    if "preconditioner" not in system:
        preconditioner = None
        if pyamg is not None and n >= 5000:
            try:
                hierarchy = pyamg.smoothed_aggregation_solver(matrix, max_coarse=50)
                system["amg_hierarchy"] = hierarchy
                preconditioner = hierarchy.aspreconditioner(cycle="V")
                system["preconditioner_kind"] = "amg"
            except Exception:
                preconditioner = None
        if preconditioner is None:
            diag = np.asarray(matrix.diagonal(), dtype=np.float64)
            inv_diag = np.zeros_like(diag)
            nz = np.abs(diag) > 1e-12
            inv_diag[nz] = 1.0 / diag[nz]
            preconditioner = diags(inv_diag, 0, format="csr")
            system["preconditioner_kind"] = "jacobi"
        system["preconditioner"] = preconditioner

    x0 = system.get("last_solution")
    sol, info = cg(
        matrix,
        rhs,
        x0=x0,
        M=system["preconditioner"],
        maxiter=int(max_iter),
        rtol=float(tol),
        atol=0.0,
    )
    if info != 0:
        try:
            solver = system.get("factorized_solver")
            if solver is None:
                solver = factorized(csc_matrix(matrix))
                system["factorized_solver"] = solver
            sol = np.asarray(solver(rhs), dtype=np.float64).reshape(n)
        except Exception:
            raise RuntimeError(f"sparse pressure solve failed to converge: info={info}")
    else:
        sol = np.asarray(sol, dtype=np.float64).reshape(n)
    system["last_solution"] = sol

    sol[anchors] = 0.0
    return sol.astype(np.float32)


def _build_pressure_reconstruction_system(mask_t, spacing_m):
    coords = np.argwhere(mask_t)
    n = int(len(coords))
    if n == 0:
        return {
            "coords": coords,
            "index_map": -np.ones(mask_t.shape, dtype=np.int32),
            "matrix": csr_matrix((0, 0), dtype=np.float64),
            "edge_pairs": np.zeros((0, 2), dtype=np.int32),
            "edge_axes": np.zeros(0, dtype=np.int8),
            "edge_scales": np.zeros(0, dtype=np.float64),
            "anchor_index": 0,
        }

    index_map = -np.ones(mask_t.shape, dtype=np.int32)
    index_map[mask_t] = np.arange(n, dtype=np.int32)
    edge_pairs = []
    edge_axes = []
    edge_scales = []
    dx, dy, dz = [float(v) for v in spacing_m]
    for axis, step in enumerate((dx, dy, dz)):
        left = [slice(None), slice(None), slice(None)]
        right = [slice(None), slice(None), slice(None)]
        left[axis] = slice(0, -1)
        right[axis] = slice(1, None)
        left = tuple(left)
        right = tuple(right)
        adjacent = mask_t[left] & mask_t[right]
        src = np.asarray(index_map[left][adjacent], dtype=np.int32)
        dst = np.asarray(index_map[right][adjacent], dtype=np.int32)
        if src.size:
            edge_pairs.append(np.column_stack((src, dst)))
            edge_axes.append(np.full(src.size, axis, dtype=np.int8))
            edge_scales.append(np.full(src.size, 1.0 / step, dtype=np.float64))

    pairs = np.vstack(edge_pairs).astype(np.int32, copy=False) if edge_pairs else np.zeros((0, 2), dtype=np.int32)
    axes = np.concatenate(edge_axes) if edge_axes else np.zeros(0, dtype=np.int8)
    scales = np.concatenate(edge_scales) if edge_scales else np.zeros(0, dtype=np.float64)
    rows = np.repeat(np.arange(len(pairs), dtype=np.int32), 2)
    cols = pairs.reshape(-1)
    data = np.column_stack((-scales, scales)).reshape(-1)
    incidence = csr_matrix((data, (rows, cols)), shape=(len(pairs), n), dtype=np.float64)
    matrix = (incidence.T @ incidence).tocsr()
    # Fix one pressure gauge in EACH connected component. Eliminate both rows
    # and columns to retain the symmetric positive-definite system needed by CG.
    components, count = label(mask_t, structure=generate_binary_structure(3, 1))
    component_ids = components[mask_t] - 1
    anchors = np.full(count, n, dtype=np.int32)
    np.minimum.at(anchors, component_ids, np.arange(n, dtype=np.int32))
    keep = np.ones(n, dtype=np.float64)
    keep[anchors] = 0.0
    gauge_scale = max(float(np.max(matrix.diagonal())), 1.0)
    selector = diags(keep, format="csr")
    matrix = selector @ matrix @ selector + diags((1.0 - keep) * gauge_scale, format="csr")
    return {
        "coords": coords,
        "index_map": index_map,
        "matrix": matrix.tocsr(),
        "rhs_operator": incidence.T.tocsr(),
        "edge_pairs": pairs,
        "edge_axes": axes,
        "edge_scales": scales,
        "anchor_index": 0,
        "anchor_indices": anchors,
    }


def _pressure_reconstruction_rhs(grad_t, system):
    edge_pairs = system["edge_pairs"]
    if edge_pairs.size == 0:
        return np.zeros(0, dtype=np.float64)
    edge_axes = system["edge_axes"]
    coords = system["coords"]
    src = coords[edge_pairs[:, 0]]
    dst = coords[edge_pairs[:, 1]]
    # D already contains 1/h. Its target is the face-averaged gradient in Pa/m,
    # with no additional scale or minus sign. Averaging integrates linear PG
    # exactly and avoids a one-sided pressure bias for varying gradients.
    return 0.5 * (np.asarray(grad_t[src[:, 0], src[:, 1], src[:, 2], edge_axes], dtype=np.float64)
                  + np.asarray(grad_t[dst[:, 0], dst[:, 1], dst[:, 2], edge_axes], dtype=np.float64))


def _build_least_squares_system(mask_t, spacing_m):
    """Backward-compatible name for the merged Cartesian PPE system."""
    return _build_pressure_reconstruction_system(mask_t, spacing_m)


def _build_ppe_system(mask_t, spacing_m):
    """Backward-compatible name for the merged Cartesian PPE system."""
    system = _build_pressure_reconstruction_system(mask_t, spacing_m)
    system.pop("rhs_operator", None)
    return system


def _least_squares_rhs(grad_t, system):
    """Backward-compatible name for the merged pressure RHS."""
    return _pressure_reconstruction_rhs(grad_t, system)


def _ppe_rhs(grad_t, mask_t, spacing_m, system):
    """Backward-compatible name for the merged pressure RHS."""
    rhs = np.zeros(system["matrix"].shape[0], dtype=np.float64)
    pairs = system["edge_pairs"]
    if pairs.size:
        flux = _pressure_reconstruction_rhs(grad_t, system) * system["edge_scales"]
        np.add.at(rhs, pairs[:, 0], -flux)
        np.add.at(rhs, pairs[:, 1], flux)
    return rhs


def _build_ste_mac_system(mask_t, spacing_m):
    """Build a staggered-grid Stokes estimator system.

    The pressure is cell-centred and the auxiliary velocity is face-centred.
    Interior faces carry unknown velocities; faces on the support boundary
    are zero (no-slip). This MAC layout avoids the checkerboard pressure
    modes that occur when all variables are collocated on voxel centres.
    """
    mask_t = np.asarray(mask_t, dtype=bool)
    coords = np.argwhere(mask_t)
    n_cells = int(len(coords))
    if n_cells == 0:
        return {
            "coords": coords,
            "matrix": csr_matrix((0, 0), dtype=np.float64),
            "face_count": 0,
            "pressure_indices": np.zeros(0, dtype=np.int32),
            "pressure_count": 0,
            "fallback": True,
        }

    spacing_m = tuple(float(value) for value in spacing_m)
    cell_index = -np.ones(mask_t.shape, dtype=np.int32)
    cell_index[mask_t] = np.arange(n_cells, dtype=np.int32)
    face_maps = []
    face_low = []
    face_high = []
    face_axis = []
    face_offsets = []
    face_count = 0
    for axis in range(3):
        low_slices = [slice(None), slice(None), slice(None)]
        high_slices = [slice(None), slice(None), slice(None)]
        low_slices[axis] = slice(0, -1)
        high_slices[axis] = slice(1, None)
        low_slices = tuple(low_slices)
        high_slices = tuple(high_slices)
        adjacent = mask_t[low_slices] & mask_t[high_slices]
        low_coords = np.argwhere(adjacent)
        local_count = int(len(low_coords))
        face_map = -np.ones(mask_t.shape, dtype=np.int32)
        if local_count:
            face_map[tuple(low_coords.T)] = np.arange(local_count, dtype=np.int32)
            high_coords = low_coords.copy()
            high_coords[:, axis] += 1
            face_low.append(low_coords)
            face_high.append(high_coords)
            face_axis.append(np.full(local_count, axis, dtype=np.int8))
        else:
            face_low.append(np.zeros((0, 3), dtype=np.int32))
            face_high.append(np.zeros((0, 3), dtype=np.int32))
            face_axis.append(np.zeros(0, dtype=np.int8))
        face_maps.append(face_map)
        face_offsets.append(face_count)
        face_count += local_count

    if face_count == 0:
        return {
            "coords": coords,
            "matrix": csr_matrix((0, 0), dtype=np.float64),
            "face_count": 0,
            "pressure_indices": np.zeros(0, dtype=np.int32),
            "pressure_count": 0,
            "fallback": True,
        }

    all_low = np.concatenate(face_low, axis=0)
    all_high = np.concatenate(face_high, axis=0)
    all_axis = np.concatenate(face_axis, axis=0)
    matrix_rows = []
    matrix_cols = []
    matrix_data = []
    diagonal = 2.0 * sum(1.0 / (step * step) for step in spacing_m)
    for axis in range(3):
        low_coords = face_low[axis]
        offset = int(face_offsets[axis])
        face_map = face_maps[axis]
        for local_id, low_coord in enumerate(low_coords):
            row = offset + int(local_id)
            matrix_rows.append(row)
            matrix_cols.append(row)
            matrix_data.append(diagonal)
            for direction, step in enumerate(spacing_m):
                for sign in (-1, 1):
                    neighbour = low_coord.copy()
                    neighbour[direction] += sign
                    if np.any(neighbour < 0) or np.any(neighbour >= np.asarray(mask_t.shape)):
                        continue
                    neighbour_id = int(face_map[tuple(neighbour)])
                    if neighbour_id >= 0:
                        matrix_rows.append(row)
                        matrix_cols.append(offset + neighbour_id)
                        matrix_data.append(-1.0 / (step * step))
    velocity_laplacian = csr_matrix(
        (matrix_data, (matrix_rows, matrix_cols)),
        shape=(face_count, face_count),
        dtype=np.float64,
    )

    div_rows = []
    div_cols = []
    div_data = []
    for face_id, (low_coord, high_coord, axis) in enumerate(zip(all_low, all_high, all_axis)):
        scale = 1.0 / spacing_m[int(axis)]
        low_id = int(cell_index[tuple(low_coord)])
        high_id = int(cell_index[tuple(high_coord)])
        div_rows.extend((low_id, high_id))
        div_cols.extend((face_id, face_id))
        div_data.extend((scale, -scale))
    divergence = csr_matrix(
        (div_data, (div_rows, div_cols)),
        shape=(n_cells, face_count),
        dtype=np.float64,
    )

    components, count = label(mask_t, structure=generate_binary_structure(3, 1))
    component_ids = components[mask_t] - 1
    anchors = np.full(count, n_cells, dtype=np.int32)
    np.minimum.at(anchors, component_ids, np.arange(n_cells, dtype=np.int32))
    pressure_keep = np.ones(n_cells, dtype=bool)
    pressure_keep[anchors] = False
    reduced_divergence = divergence[pressure_keep, :].tocsr()
    matrix = bmat(
        [[velocity_laplacian, reduced_divergence.T],
         [reduced_divergence, None]],
        format="csr",
    )
    diagonal_values = np.abs(np.asarray(matrix.diagonal(), dtype=np.float64))
    diagonal_values[diagonal_values < 1e-8] = 1.0
    return {
        "coords": coords,
        "matrix": matrix,
        "preconditioner": diags(1.0 / diagonal_values, format="csr"),
        "face_count": face_count,
        "pressure_indices": np.flatnonzero(pressure_keep).astype(np.int32),
        "pressure_count": int(np.sum(pressure_keep)),
        "anchors": anchors,
        "face_low": all_low,
        "face_high": all_high,
        "face_axis": all_axis,
        "fallback": False,
    }


def _solve_ste_mac_pressure(grad_t, system):
    if system.get("fallback", False):
        return np.zeros(len(system.get("coords", [])), dtype=np.float32)
    face_low = system["face_low"]
    face_high = system["face_high"]
    face_axis = system["face_axis"]
    rhs = np.zeros(system["matrix"].shape[0], dtype=np.float64)
    if len(face_low):
        low_values = np.asarray(grad_t[tuple(face_low.T)], dtype=np.float64)
        high_values = np.asarray(grad_t[tuple(face_high.T)], dtype=np.float64)
        rhs[: system["face_count"]] = 0.5 * (
            low_values[np.arange(len(face_axis)), face_axis]
            + high_values[np.arange(len(face_axis)), face_axis]
        )

    matrix = system["matrix"]
    solution = None
    if matrix.shape[0] <= 12000:
        try:
            candidate = np.asarray(spsolve(matrix.tocsc(), rhs), dtype=np.float64)
            if np.all(np.isfinite(candidate)):
                solution = candidate
        except Exception:
            solution = None
    if solution is None:
        preconditioner = system.get("iterative_preconditioner")
        if preconditioner is None:
            preconditioner = system["preconditioner"]
            if pyamg is not None and system["face_count"] > 0:
                try:
                    velocity_matrix = matrix[: system["face_count"], : system["face_count"]].tocsr()
                    hierarchy = pyamg.smoothed_aggregation_solver(velocity_matrix, max_coarse=50)

                    def apply_preconditioner(vector):
                        result = np.zeros_like(vector, dtype=np.float64)
                        result[: system["face_count"]] = hierarchy.solve(
                            vector[: system["face_count"]],
                            tol=1e-6,
                            maxiter=5,
                            cycle="V",
                        )
                        result[system["face_count"]:] = vector[system["face_count"]:]
                        return result

                    preconditioner = LinearOperator(
                        matrix.shape,
                        matvec=apply_preconditioner,
                        dtype=np.float64,
                    )
                except Exception:
                    preconditioner = system["preconditioner"]
            system["iterative_preconditioner"] = preconditioner
        solution, info = minres(
            matrix,
            rhs,
            M=preconditioner,
            maxiter=10000,
            rtol=1e-8,
        )
        solution = np.asarray(solution, dtype=np.float64)
        residual = np.linalg.norm(matrix @ solution - rhs) / max(np.linalg.norm(rhs), 1.0)
        if info != 0 or not np.all(np.isfinite(solution)) or not np.isfinite(residual) or residual > 1e-4:
            raise RuntimeError(f"Stokes pressure solve failed to converge: info={info}, residual={residual:.3g}")

    pressure = np.zeros(len(system["coords"]), dtype=np.float32)
    pressure[system["pressure_indices"]] = -solution[system["face_count"]:].astype(np.float32)
    return pressure


def reconstruct_relative_pressure_map(pressure_gradient_array, support_mask, spacing, *, method="ppe"):
    method = _normalize_pressure_method(method)
    grad = np.asarray(pressure_gradient_array, dtype=np.float32)
    if grad.ndim != 5 or grad.shape[-1] != 3:
        raise ValueError(f"pressure_gradient_array must be XYZTV, got {grad.shape}")
    support = np.asarray(support_mask, dtype=bool)
    if support.shape != grad.shape[:4]:
        raise ValueError(f"support_mask shape {support.shape} does not match {grad.shape[:4]}")

    spacing_m = np.asarray(spacing, dtype=float).reshape(3) / 1000.0
    if not np.all(np.isfinite(spacing_m)) or np.any(spacing_m <= 0):
        raise ValueError("Pressure spacing must be finite and positive (mm)")
    if not np.all(np.isfinite(grad[support])):
        raise ValueError("Pressure gradients must be finite inside the support mask")
    dx, dy, dz = spacing_m.tolist()
    nt = grad.shape[3]
    pressure = np.zeros(grad.shape[:4], dtype=np.float32)

    system_cache = {}
    for tidx in range(nt):
        report_progress({"stage": "relative_pressure", "current": tidx, "total": nt,
                         "message": f"Reconstructing relative pressure phase {tidx + 1}/{nt}"})
        mask_t = support[..., tidx]
        if not np.any(mask_t):
            continue
        cache_key = mask_t.tobytes()
        if method == "ste":
            system = system_cache.get(cache_key)
            if system is None:
                system = _build_ste_mac_system(mask_t, (dx, dy, dz))
                system_cache[cache_key] = system
            sol = _solve_ste_mac_pressure(grad[..., tidx, :], system)
        else:
            system = system_cache.get(cache_key)
            if system is None:
                system = _build_pressure_reconstruction_system(mask_t, (dx, dy, dz))
                system_cache[cache_key] = system
            rhs_rows = _pressure_reconstruction_rhs(grad[..., tidx, :], system)
            rhs = system["rhs_operator"] @ rhs_rows
            sol = _solve_reconstruction_system(system, rhs)

        pressure_t = np.zeros(mask_t.shape, dtype=np.float32)
        coords = system["coords"]
        if len(coords):
            pressure_t[coords[:, 0], coords[:, 1], coords[:, 2]] = sol
        pressure[..., tidx] = pressure_t

    peak = np.max(np.abs(pressure), axis=3).astype(np.float32) if pressure.shape[3] > 0 else np.zeros(pressure.shape[:3], dtype=np.float32)
    finite = pressure[support & np.isfinite(pressure)]
    upper = _finite_percentile_abs(finite, 99.0, default=1.0)
    upper = upper if upper > 0.0 else 1.0
    return {
        "relative_pressure_array": pressure,
        "relative_pressure_peak": peak,
        "relative_pressure_display_clim": (-upper, upper),
        "pressure_method": method,
    }


def _points_as_voxels(points_xyz, spacing, origin, shape, coordinate_space="world"):
    pts = np.asarray(points_xyz, dtype=float).reshape(-1, 3)
    spacing = np.asarray(spacing, dtype=float).reshape(3)
    origin = np.asarray(origin, dtype=float).reshape(3)
    shape = np.asarray(shape, dtype=int).reshape(3)
    if len(pts) == 0:
        return pts, False
    if str(coordinate_space).lower() == "local":
        return pts / (spacing.reshape(1, 3) + 1e-12), True
    if str(coordinate_space).lower() == "voxel":
        return pts, True
    looks_like_voxels = (
        np.all(np.isfinite(pts))
        and np.all(pts >= -0.5)
        and np.all(pts <= (shape.reshape(1, 3) - 0.5))
    )
    if looks_like_voxels:
        return pts, True
    vox = (pts - origin.reshape(1, 3)) / (spacing.reshape(1, 3) + 1e-12)
    return vox, False


def _sample_volume_at_points(volume_xyz, points_xyz, spacing, origin, coordinate_space="world"):
    vol = np.asarray(volume_xyz, dtype=np.float32)
    pts = np.asarray(points_xyz, dtype=float).reshape(-1, 3)
    if vol.ndim != 3 or len(pts) == 0:
        return np.zeros(len(pts), dtype=np.float32)
    spacing = np.asarray(spacing, dtype=float).reshape(3)
    origin = np.asarray(origin, dtype=float).reshape(3)
    shape = np.array(vol.shape, dtype=int)
    vox, _ = _points_as_voxels(pts, spacing, origin, shape, coordinate_space=coordinate_space)
    out = np.zeros(len(pts), dtype=np.float32)
    for idx, coord in enumerate(vox):
        if np.any(coord < 0.0) or np.any(coord > (shape - 1)):
            continue
        base = np.floor(coord).astype(int)
        frac = coord - base
        upper = np.minimum(base + 1, shape - 1)
        x0, y0, z0 = base.tolist()
        x1, y1, z1 = upper.tolist()
        c000 = float(vol[x0, y0, z0])
        c100 = float(vol[x1, y0, z0])
        c010 = float(vol[x0, y1, z0])
        c110 = float(vol[x1, y1, z0])
        c001 = float(vol[x0, y0, z1])
        c101 = float(vol[x1, y0, z1])
        c011 = float(vol[x0, y1, z1])
        c111 = float(vol[x1, y1, z1])
        xd, yd, zd = frac.tolist()
        c00 = c000 * (1.0 - xd) + c100 * xd
        c10 = c010 * (1.0 - xd) + c110 * xd
        c01 = c001 * (1.0 - xd) + c101 * xd
        c11 = c011 * (1.0 - xd) + c111 * xd
        c0 = c00 * (1.0 - yd) + c10 * yd
        c1 = c01 * (1.0 - yd) + c11 * yd
        out[idx] = c0 * (1.0 - zd) + c1 * zd
    return out


def compute_centerline_pressure_profiles(relative_pressure_array, centerline_paths, spacing, origin):
    pressure = np.asarray(relative_pressure_array, dtype=np.float32)
    if pressure.ndim != 4:
        raise ValueError(f"relative_pressure_array must be XYZT, got {pressure.shape}")
    spacing = np.asarray(spacing, dtype=float).reshape(3)
    origin = np.asarray(origin, dtype=float).reshape(3)
    profiles = []
    nt = pressure.shape[3]
    for path_idx, path in enumerate(list(centerline_paths or [])):
        pts = np.asarray(path, dtype=float).reshape(-1, 3)
        if len(pts) == 0:
            profiles.append({
                "path_index": int(path_idx),
                "distances_mm": [],
                "relative_pressure_Pa_t": [],
                "pressure_drop_Pa_t": [],
                "pressure_drop_mean_Pa": 0.0,
                "pressure_drop_peak_Pa": 0.0,
            })
            continue
        # Centerline paths are stored in local physical millimetres.  Convert
        # explicitly instead of guessing from coordinate magnitudes.
        pts_vox, _ = _points_as_voxels(
            pts, spacing, origin, pressure.shape[:3], coordinate_space="local"
        )
        if len(pts_vox) > 1:
            diffs = np.diff(pts_vox, axis=0)
            seg = np.linalg.norm(diffs * spacing.reshape(1, 3), axis=1)
        else:
            seg = np.zeros(0, dtype=float)
        dist = np.concatenate([[0.0], np.cumsum(seg)]) if len(pts) > 0 else np.zeros(0, dtype=float)
        sample_pts = pts_vox
        samples_t = []
        drop_t = []
        for tidx in range(nt):
            vals = _sample_volume_at_points(
                pressure[..., tidx], sample_pts, spacing, origin, coordinate_space="voxel"
            )
            samples_t.append(vals.astype(np.float32).tolist())
            drop_t.append(float(vals[0] - vals[-1]) if len(vals) else 0.0)
        profiles.append({
            "path_index": int(path_idx),
            "distances_mm": dist.astype(np.float32).tolist(),
            "relative_pressure_Pa_t": samples_t,
            "pressure_drop_Pa_t": [float(x) for x in drop_t],
            "pressure_drop_mean_Pa": float(np.mean(drop_t)) if drop_t else 0.0,
            "pressure_drop_peak_Pa": float(np.max(np.abs(drop_t))) if drop_t else 0.0,
        })
    return profiles


def compute_vortex_metrics(mask4d, flow, spacing, *, smoothing_sigma=0.0,
                           support_erosion_iters=1):
    """Compute velocity-gradient vortex descriptors inside an eroded lumen mask.

    ``flow`` is the normalized XYZTV velocity field in cm/s and ``spacing`` is
    in mm.  Calculations are carried out in SI units so vorticity and swirling
    strength are returned in s^-1 and Q is returned in s^-2.  The public
    arrays retain the input shape, while ``vortex_support_mask`` records the
    voxels for which all results are considered valid.
    """
    mask4d = _ensure_mask4d(mask4d)
    flow = _ensure_flow5d(flow)
    if flow.shape[:3] != mask4d.shape[:3]:
        raise ValueError(
            f"flow spatial shape {flow.shape[:3]} does not match mask {mask4d.shape[:3]}"
        )
    if flow.shape[3] != mask4d.shape[3]:
        raise ValueError(f"flow time dimension {flow.shape[3]} does not match mask {mask4d.shape[3]}")

    spatial_shape = tuple(int(value) for value in flow.shape[:3])
    output_shape = flow.shape[:4]
    vorticity = np.zeros(flow.shape, dtype=np.float32)
    vorticity_magnitude = np.zeros(output_shape, dtype=np.float32)
    q_criterion = np.zeros(output_shape, dtype=np.float32)
    swirling_strength = np.zeros(output_shape, dtype=np.float32)
    support = np.zeros(output_shape, dtype=bool)

    # A one-voxel halo is required for the central spatial-difference stencil.
    # Crop to the vessel extent for predictable memory use on large images.
    union_mask = np.any(mask4d, axis=3)
    occupied = np.where(union_mask)
    if occupied[0].size == 0:
        return {
            "vorticity_array": vorticity,
            "vorticity_magnitude": vorticity_magnitude,
            "vorticity_magnitude_peak": np.zeros(spatial_shape, dtype=np.float32),
            "q_criterion_array": q_criterion,
            "q_criterion_peak": np.zeros(spatial_shape, dtype=np.float32),
            "swirling_strength_array": swirling_strength,
            "swirling_strength_peak": np.zeros(spatial_shape, dtype=np.float32),
            "vortex_support_mask": support.astype(np.uint8),
        }
    spatial_slices = tuple(
        slice(max(int(np.min(axis_values)) - 1, 0), min(int(np.max(axis_values)) + 2, spatial_shape[axis]))
        for axis, axis_values in enumerate(occupied)
    )
    work_mask = np.asarray(mask4d[spatial_slices + (slice(None),)], dtype=bool)
    work_flow = np.asarray(flow[spatial_slices + (slice(None), slice(None))], dtype=np.float32)
    if any(size < 3 for size in work_flow.shape[:3]):
        return {
            "vorticity_array": vorticity,
            "vorticity_magnitude": vorticity_magnitude,
            "vorticity_magnitude_peak": np.zeros(spatial_shape, dtype=np.float32),
            "q_criterion_array": q_criterion,
            "q_criterion_peak": np.zeros(spatial_shape, dtype=np.float32),
            "swirling_strength_array": swirling_strength,
            "swirling_strength_peak": np.zeros(spatial_shape, dtype=np.float32),
            "vortex_support_mask": support.astype(np.uint8),
        }

    # Convert cm/s to m/s before differentiating over meter spacing.
    velocity = work_flow / 100.0
    mask_float = work_mask.astype(np.float32)
    velocity = velocity * mask_float[..., None]
    sigma = max(float(smoothing_sigma), 0.0)
    if sigma > 0.0:
        # Normalize the filtered field by filtered mask weights so a zero
        # background does not artificially damp values near the valid support.
        weights = gaussian_filter(mask_float, sigma=(sigma, sigma, sigma, 0.0), mode="nearest")
        weights = np.maximum(weights, np.float32(1e-6))
        for component in range(3):
            velocity[..., component] = (
                gaussian_filter(velocity[..., component], sigma=(sigma, sigma, sigma, 0.0), mode="nearest")
                / weights
            )
        velocity *= mask_float[..., None]

    support_work = work_mask.copy()
    erosion_iters = max(int(support_erosion_iters), 0)
    if erosion_iters > 0:
        structure = np.ones((3, 3, 3), dtype=bool)
        for tidx in range(work_mask.shape[3]):
            support_work[..., tidx] = binary_erosion(
                work_mask[..., tidx],
                structure=structure,
                iterations=erosion_iters,
                border_value=0,
            )
    support_inner = support_work[1:-1, 1:-1, 1:-1, :]

    spacing_m = np.asarray(spacing, dtype=float).reshape(3) / 1000.0
    dx, dy, dz = [float(max(value, 1e-12)) for value in spacing_m]
    # The descriptors outside support are zero by definition. Gather the
    # same central-difference neighbours only for valid voxels so background
    # tensors, eigensolves, and their large intermediate arrays are avoided.
    support_indices = np.nonzero(support_inner)
    centers = tuple(index + 1 for index in support_indices[:3]) + (support_indices[3],)
    jacobian = np.empty((len(support_indices[0]), 3, 3), dtype=np.float32)
    spacings = (dx, dy, dz)
    for axis, step in enumerate(spacings):
        before = list(centers)
        after = list(centers)
        before[axis] = before[axis] - 1
        after[axis] = after[axis] + 1
        for component in range(3):
            jacobian[..., component, axis] = (
                velocity[tuple(after) + (component,)]
                - velocity[tuple(before) + (component,)]
            ) / (2.0 * step)

    vort_inner = np.empty(jacobian.shape[:-2] + (3,), dtype=np.float32)
    vort_inner[..., 0] = jacobian[..., 2, 1] - jacobian[..., 1, 2]
    vort_inner[..., 1] = jacobian[..., 0, 2] - jacobian[..., 2, 0]
    vort_inner[..., 2] = jacobian[..., 1, 0] - jacobian[..., 0, 1]
    vortmag_inner = np.sqrt(np.sum(np.square(vort_inner, dtype=np.float32), axis=-1)).astype(np.float32)

    strain = 0.5 * (jacobian + np.swapaxes(jacobian, -1, -2))
    rotation = 0.5 * (jacobian - np.swapaxes(jacobian, -1, -2))
    q_inner = 0.5 * (
        np.sum(np.square(rotation, dtype=np.float32), axis=(-2, -1))
        - np.sum(np.square(strain, dtype=np.float32), axis=(-2, -1))
    )

    # λci is the positive imaginary part of the complex-conjugate eigenvalue
    # pair of the local velocity-gradient tensor.  It is zero for pure shear.
    eigvals = np.linalg.eigvals(jacobian)
    lambda_ci_inner = np.max(np.abs(np.imag(eigvals)), axis=1).astype(np.float32)

    output_slices = tuple(slice(part.start + 1, part.stop - 1) for part in spatial_slices) + (slice(None),)
    vorticity[output_slices][support_inner] = vort_inner
    vorticity_magnitude[output_slices][support_inner] = vortmag_inner
    q_criterion[output_slices][support_inner] = q_inner
    swirling_strength[output_slices][support_inner] = lambda_ci_inner
    support[output_slices] = support_inner
    return {
        "vorticity_array": vorticity,
        "vorticity_magnitude": vorticity_magnitude,
        "vorticity_magnitude_peak": np.max(vorticity_magnitude, axis=3).astype(np.float32),
        "q_criterion_array": q_criterion,
        "q_criterion_peak": np.max(q_criterion, axis=3).astype(np.float32),
        "swirling_strength_array": swirling_strength,
        "swirling_strength_peak": np.max(swirling_strength, axis=3).astype(np.float32),
        "vortex_support_mask": support.astype(np.uint8),
    }


def compute_pressure_gradient_metrics(mask4d, flow, spacing, rr, rho=1060.0, viscosity=4.0,
                                      smoothing_sigma=0.0, support_erosion_iters=1, use_convective_acceleration=True,
                                      pressure_method="ppe", centerline_paths=None,
                                      origin=(0, 0, 0)):
    mask4d = _ensure_mask4d(mask4d)
    flow = np.asarray(flow, dtype=np.float32)
    if flow.ndim != 5 or flow.shape[-1] != 3:
        raise ValueError(f"flow must be XYZTV, got {flow.shape}")
    if flow.shape[:4] != mask4d.shape:
        raise ValueError(f"flow shape {flow.shape[:4]} does not match mask {mask4d.shape}")

    spacing_mm = np.asarray(spacing, dtype=float).reshape(3)
    if not np.all(np.isfinite(spacing_mm)) or np.any(spacing_mm <= 0):
        raise ValueError("Pressure spacing must be finite and positive (mm)")
    if flow.shape[3] == 0 or not np.isfinite(rr) or float(rr) <= 0:
        raise ValueError("Pressure analysis requires at least one phase and a positive RR (ms)")
    spacing_m = spacing_mm / 1000.0
    dt_s = float(rr) / 1000.0 / float(flow.shape[3])
    rho = float(rho)
    mu_pa_s = float(viscosity) / 1000.0
    if not np.isfinite(rho) or rho <= 0 or not np.isfinite(mu_pa_s) or mu_pa_s < 0:
        raise ValueError("Pressure density must be positive and viscosity nonnegative and finite")
    if not np.isfinite(smoothing_sigma):
        raise ValueError("Pressure smoothing sigma must be finite")
    sigma = max(float(smoothing_sigma), 0.0)

    # Spatial derivatives only need the vessel extent plus one stencil voxel.
    # Keep the public result arrays full-sized, but avoid applying every large
    # finite-difference temporary to the mostly empty image volume.
    union_mask = np.any(mask4d, axis=3)
    occupied = np.where(union_mask)
    if occupied[0].size:
        lo = [max(int(np.min(axis_values)) - 1, 0) for axis_values in occupied]
        hi = [
            min(int(np.max(axis_values)) + 2, int(mask4d.shape[axis]))
            for axis, axis_values in enumerate(occupied)
        ]
        spatial_slices = tuple(slice(lo[axis], hi[axis]) for axis in range(3))
    else:
        spatial_slices = tuple(slice(0, int(mask4d.shape[axis])) for axis in range(3))

    work_mask = mask4d[spatial_slices + (slice(None),)]
    work_flow = flow[spatial_slices + (slice(None), slice(None))]
    finite_velocity = np.all(np.isfinite(work_flow), axis=-1)
    valid_samples = work_mask & finite_velocity
    velocity = np.where(finite_velocity[..., None], work_flow, 0.0).astype(np.float32) / 100.0
    if sigma > 0.0:
        weights = valid_samples.astype(np.float32)
        denominator = gaussian_filter(weights, sigma=(sigma, sigma, sigma, 0.0), mode='constant')
        for comp in range(3):
            numerator = gaussian_filter(
                velocity[..., comp] * weights,
                sigma=(sigma, sigma, sigma, 0.0),
                mode='constant',
            )
            velocity[..., comp] = np.divide(
                numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)

    dx, dy, dz = [float(max(s, 1e-12)) for s in spacing_m]
    vc = velocity[1:-1, 1:-1, 1:-1, :, :]
    du_dt = _periodic_central_difference(velocity, dt_s)[1:-1, 1:-1, 1:-1, :, :]

    conv = np.zeros_like(vc, dtype=np.float32)
    lap = np.zeros_like(vc, dtype=np.float32)
    for comp in range(3):
        du_dx = (
            velocity[2:, 1:-1, 1:-1, :, comp]
            - velocity[:-2, 1:-1, 1:-1, :, comp]
        ) / (2.0 * dx)
        du_dy = (
            velocity[1:-1, 2:, 1:-1, :, comp]
            - velocity[1:-1, :-2, 1:-1, :, comp]
        ) / (2.0 * dy)
        du_dz = (
            velocity[1:-1, 1:-1, 2:, :, comp]
            - velocity[1:-1, 1:-1, :-2, :, comp]
        ) / (2.0 * dz)
        d2u_dx2 = (
            velocity[2:, 1:-1, 1:-1, :, comp]
            - 2.0 * velocity[1:-1, 1:-1, 1:-1, :, comp]
            + velocity[:-2, 1:-1, 1:-1, :, comp]
        ) / (dx * dx)
        d2u_dy2 = (
            velocity[1:-1, 2:, 1:-1, :, comp]
            - 2.0 * velocity[1:-1, 1:-1, 1:-1, :, comp]
            + velocity[1:-1, :-2, 1:-1, :, comp]
        ) / (dy * dy)
        d2u_dz2 = (
            velocity[1:-1, 1:-1, 2:, :, comp]
            - 2.0 * velocity[1:-1, 1:-1, 1:-1, :, comp]
            + velocity[1:-1, 1:-1, :-2, :, comp]
        ) / (dz * dz)
        if use_convective_acceleration:
            conv[..., comp] = vc[..., 0] * du_dx + vc[..., 1] * du_dy + vc[..., 2] * du_dz
        lap[..., comp] = d2u_dx2 + d2u_dy2 + d2u_dz2

    grad_inner = -rho * (du_dt + conv) + mu_pa_s * lap
    # Always require the actual six-neighbour spatial derivative stencil to be
    # measured inside the lumen, even when optional extra erosion is disabled.
    # Also require both temporal neighbours; segmentation flicker must not be
    # interpreted as acceleration from setting a phase's velocity to zero.
    support_work = np.zeros(work_mask.shape, dtype=bool)
    stencil = generate_binary_structure(3, 1)
    for tidx in range(work_mask.shape[3]):
        support_work[..., tidx] = binary_erosion(valid_samples[..., tidx], structure=stencil, border_value=0)
    if work_mask.shape[3] > 1:
        support_work &= np.roll(valid_samples, -1, axis=3) & np.roll(valid_samples, 1, axis=3)
    erosion_iters = max(int(support_erosion_iters), 0)
    if erosion_iters > 0:
        structure = np.ones((3, 3, 3), dtype=bool)
        for tidx in range(work_mask.shape[3]):
            support_work[..., tidx] &= binary_erosion(
                work_mask[..., tidx],
                structure=structure,
                iterations=erosion_iters,
                border_value=0,
            )
    grad_work = np.zeros(work_flow.shape, dtype=np.float32)
    grad_work[1:-1, 1:-1, 1:-1, :, :] = grad_inner.astype(np.float32)
    grad_work *= support_work.astype(np.float32)[..., None]
    grad_mag_work = np.sqrt(np.sum(np.square(grad_work, dtype=np.float32), axis=-1)).astype(np.float32)

    display_upper = _finite_percentile_abs(grad_mag_work[support_work], 99.0, default=1.0)
    display_upper = display_upper if display_upper > 0 else 1.0

    pressure_result_work = reconstruct_relative_pressure_map(
        grad_work,
        support_work,
        spacing_mm,
        method=pressure_method,
    )

    grad = np.zeros(flow.shape, dtype=np.float32)
    grad[spatial_slices + (slice(None), slice(None))] = grad_work
    grad_mag = np.zeros(mask4d.shape, dtype=np.float32)
    grad_mag[spatial_slices + (slice(None),)] = grad_mag_work
    grad_peak = np.zeros(mask4d.shape[:3], dtype=np.float32)
    grad_peak[spatial_slices] = np.max(grad_mag_work, axis=3).astype(np.float32)
    support_mask = np.zeros(mask4d.shape, dtype=bool)
    support_mask[spatial_slices + (slice(None),)] = support_work
    relative_pressure = np.zeros(mask4d.shape, dtype=np.float32)
    relative_pressure[spatial_slices + (slice(None),)] = pressure_result_work["relative_pressure_array"]
    relative_pressure_peak = np.zeros(mask4d.shape[:3], dtype=np.float32)
    relative_pressure_peak[spatial_slices] = pressure_result_work["relative_pressure_peak"]

    centerline_profiles = compute_centerline_pressure_profiles(
        relative_pressure,
        centerline_paths or [],
        spacing_mm,
        origin,
    )

    return {
        'pressure_gradient_array': grad,
        'pressure_gradient_magnitude': grad_mag,
        'pressure_gradient_peak': grad_peak,
        'pressure_gradient_dt_s': float(dt_s),
        'pressure_gradient_temporal_scheme': 'periodic_central_difference',
        'pressure_gradient_support_mask': support_mask.astype(np.uint8),
        'pressure_gradient_display_clim': (0.0, display_upper),
        'relative_pressure_array': relative_pressure,
        'relative_pressure_peak': relative_pressure_peak,
        'relative_pressure_display_clim': pressure_result_work['relative_pressure_display_clim'],
        'pressure_method': pressure_result_work['pressure_method'],
        'centerline_pressure_profiles': centerline_profiles,
    }


def compute_derived_metrics(mask4d, flow, spacing, origin=(0, 0, 0),
                            smoothing_iteration=200, viscosity=4.0,
                            inward_distance=None, parabolic_fitting=True,
                            no_slip_condition=True, step_size=5,
                            tube_radius=0.1, rho=1060.0,
                            save_pixelwise=False, tke_array=None, sigma=None,
                            rr=1000.0, pressure_gradient_smoothing_sigma=0.0,
                            pressure_gradient_support_erosion_iters=1,
                            pressure_gradient_use_convective_acceleration=True,
                            compute_wss=True, compute_tke=True,
                            compute_pressure_gradient=True,
                            compute_vortex=False,
                            wss_smoothing_iteration=None,
                            wss_viscosity=None,
                            wss_inward_distance=None,
                            wss_parabolic_fitting=None,
                            wss_no_slip_condition=None,
                            tke_rho=None,
                            pressure_gradient_rho=None,
                            pressure_gradient_viscosity=None,
                            vortex_smoothing_sigma=0.0,
                            vortex_support_erosion_iters=1,
                            pressure_method="ppe",
                            centerline_paths=None):
    mask4d = _ensure_mask4d(mask4d)
    wss_smoothing_iteration = smoothing_iteration if wss_smoothing_iteration is None else wss_smoothing_iteration
    wss_viscosity = viscosity if wss_viscosity is None else wss_viscosity
    wss_inward_distance = inward_distance if wss_inward_distance is None else wss_inward_distance
    wss_parabolic_fitting = parabolic_fitting if wss_parabolic_fitting is None else wss_parabolic_fitting
    wss_no_slip_condition = no_slip_condition if wss_no_slip_condition is None else wss_no_slip_condition
    tke_rho = rho if tke_rho is None else tke_rho
    pressure_gradient_rho = rho if pressure_gradient_rho is None else pressure_gradient_rho
    pressure_gradient_viscosity = viscosity if pressure_gradient_viscosity is None else pressure_gradient_viscosity
    wss = None
    if compute_wss:
        wss = compute_wss_metrics(
            mask4d, flow, spacing, origin=origin,
            smoothing_iteration=wss_smoothing_iteration,
            viscosity=wss_viscosity,
            inward_distance=wss_inward_distance,
            parabolic_fitting=wss_parabolic_fitting,
            no_slip_condition=wss_no_slip_condition,
        )
    tke = None
    if compute_tke and (tke_array is not None or sigma is not None):
        tke = compute_tke_metrics(
            mask4d, spacing, origin=origin, tke_array=tke_array, sigma=sigma, rho=tke_rho,
        )
    pressure_gradient = None
    if compute_pressure_gradient:
        pressure_gradient = compute_pressure_gradient_metrics(
            mask4d,
            flow,
            spacing,
            rr=rr,
            rho=pressure_gradient_rho,
            viscosity=pressure_gradient_viscosity,
            smoothing_sigma=pressure_gradient_smoothing_sigma,
            support_erosion_iters=pressure_gradient_support_erosion_iters,
            use_convective_acceleration=pressure_gradient_use_convective_acceleration,
            pressure_method=pressure_method,
            centerline_paths=centerline_paths,
            origin=origin,
        )
    vortex = None
    if compute_vortex:
        vortex = compute_vortex_metrics(
            mask4d,
            flow,
            spacing,
            smoothing_sigma=vortex_smoothing_sigma,
            support_erosion_iters=vortex_support_erosion_iters,
        )

    spacing = np.asarray(spacing, dtype=float).reshape(3)
    origin = np.asarray(origin, dtype=float).reshape(3)
    result = {
        "wss_surfaces": [] if wss is None else wss["wss_surfaces"],
        "wss_volume": None if wss is None else wss["wss_volume"],
        "tke_volume": None if tke is None else tke["tke_volume"],
        "tke_array": None if tke is None else tke["tke_array"],
        "tke_peak": None if tke is None else tke["tke_peak"],
        "pressure_gradient_array": None if pressure_gradient is None else pressure_gradient["pressure_gradient_array"],
        "pressure_gradient_magnitude": None if pressure_gradient is None else pressure_gradient["pressure_gradient_magnitude"],
        "pressure_gradient_peak": None if pressure_gradient is None else pressure_gradient["pressure_gradient_peak"],
        "pressure_gradient_dt_s": None if pressure_gradient is None else pressure_gradient["pressure_gradient_dt_s"],
        "pressure_gradient_temporal_scheme": (
            None if pressure_gradient is None else pressure_gradient["pressure_gradient_temporal_scheme"]
        ),
        "pressure_gradient_support_mask": None if pressure_gradient is None else pressure_gradient["pressure_gradient_support_mask"],
        "pressure_gradient_display_clim": None if pressure_gradient is None else pressure_gradient["pressure_gradient_display_clim"],
        "relative_pressure_array": None if pressure_gradient is None else pressure_gradient["relative_pressure_array"],
        "relative_pressure_peak": None if pressure_gradient is None else pressure_gradient["relative_pressure_peak"],
        "relative_pressure_display_clim": None if pressure_gradient is None else pressure_gradient["relative_pressure_display_clim"],
        "centerline_pressure_profiles": [] if pressure_gradient is None else pressure_gradient["centerline_pressure_profiles"],
        "pressure_method": None if pressure_gradient is None else pressure_gradient["pressure_method"],
        "vorticity_array": None if vortex is None else vortex["vorticity_array"],
        "vorticity_magnitude": None if vortex is None else vortex["vorticity_magnitude"],
        "vorticity_magnitude_peak": None if vortex is None else vortex["vorticity_magnitude_peak"],
        "q_criterion_array": None if vortex is None else vortex["q_criterion_array"],
        "q_criterion_peak": None if vortex is None else vortex["q_criterion_peak"],
        "swirling_strength_array": None if vortex is None else vortex["swirling_strength_array"],
        "swirling_strength_peak": None if vortex is None else vortex["swirling_strength_peak"],
        "vortex_support_mask": None if vortex is None else vortex["vortex_support_mask"],
        "streamlines": [],
        "tube_radius": float(tube_radius),
    }
    if save_pixelwise:
        pixelwise_export = {
            "spacing": np.asarray(spacing, dtype=np.float32),
            "origin": np.asarray(origin, dtype=np.float32),
        }
        if wss is not None:
            pixelwise_export["wss"] = np.asarray(wss["wss_volume"], dtype=np.float32)
        if pressure_gradient is not None:
            pixelwise_export["pressure_gradient"] = np.asarray(pressure_gradient["pressure_gradient_array"], dtype=np.float32)
            pixelwise_export["pressure_gradient_mag"] = np.asarray(pressure_gradient["pressure_gradient_magnitude"], dtype=np.float32)
            pixelwise_export["pressure_gradient_peak"] = np.asarray(pressure_gradient["pressure_gradient_peak"], dtype=np.float32)
            pixelwise_export["pressure_gradient_support_mask"] = np.asarray(pressure_gradient["pressure_gradient_support_mask"], dtype=np.uint8)
            pixelwise_export["relative_pressure"] = np.asarray(pressure_gradient["relative_pressure_array"], dtype=np.float32)
            pixelwise_export["relative_pressure_peak"] = np.asarray(pressure_gradient["relative_pressure_peak"], dtype=np.float32)
        if tke is not None:
            pixelwise_export["tke"] = np.asarray(tke["tke_peak"], dtype=np.float32)
            pixelwise_export["tke_time"] = np.asarray(tke["tke_array"], dtype=np.float32)
        if vortex is not None:
            pixelwise_export["vorticity"] = np.asarray(vortex["vorticity_array"], dtype=np.float32)
            pixelwise_export["vorticity_magnitude"] = np.asarray(vortex["vorticity_magnitude"], dtype=np.float32)
            pixelwise_export["vorticity_magnitude_peak"] = np.asarray(vortex["vorticity_magnitude_peak"], dtype=np.float32)
            pixelwise_export["q_criterion"] = np.asarray(vortex["q_criterion_array"], dtype=np.float32)
            pixelwise_export["q_criterion_peak"] = np.asarray(vortex["q_criterion_peak"], dtype=np.float32)
            pixelwise_export["swirling_strength"] = np.asarray(vortex["swirling_strength_array"], dtype=np.float32)
            pixelwise_export["swirling_strength_peak"] = np.asarray(vortex["swirling_strength_peak"], dtype=np.float32)
            pixelwise_export["vortex_support_mask"] = np.asarray(vortex["vortex_support_mask"], dtype=np.uint8)
        result["pixelwise_export"] = pixelwise_export
    else:
        result["pixelwise_export"] = {}
    return result


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


class _PlaneProcessToken:
    def __init__(self, progress_path):
        self.path = f"{progress_path}.cancel" if progress_path else ""

    def check(self):
        if self.path and os.path.exists(self.path):
            raise TaskCancelled("Cancelled by user")


def _mark_plane_process_progress(progress_path):
    check_cancelled()
    if progress_path:
        fd = os.open(progress_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, b"1\n")
        finally:
            os.close(fd)


def _wait_plane_processes(calculate, progress_path, total, progress_callback, *, stage="plane_metric"):
    if progress_callback is None and current_cancellation_token() is None:
        return calculate()
    import threading
    state = {"result": None, "error": None}
    def run():
        try:
            state["result"] = calculate()
        except BaseException as exc:
            state["error"] = exc
    runner = threading.Thread(target=run, name="autoflow-plane-processes")
    runner.start()
    offset = completed = 0
    def update():
        nonlocal offset, completed
        with open(progress_path, "rb") as handle:
            handle.seek(offset)
            events = handle.readlines()
            offset = handle.tell()
        if events:
            completed += len(events)
            if progress_callback is not None:
                progress_callback({"stage": stage, "current": completed, "total": total,
                                   "message": f"{'Sampled derived' if stage == 'plane_derived' else 'Calculated'} plane metrics ({completed}/{total})"})
    try:
        while runner.is_alive():
            check_cancelled()
            update()
            runner.join(timeout=0.05)
        check_cancelled()
        update()
        if state["error"] is not None:
            raise state["error"]
        return state["result"]
    except BaseException:
        # Workers observe this marker at phase/plane boundaries. Join before
        # deleting shared files or unlocking the workspace.
        open(f"{progress_path}.cancel", "ab").close()
        runner.join()
        raise


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


def load_metrics_as_table(metrics_json_path, qc_json_path=None):
    import json as _json
    with open(metrics_json_path, "r", encoding="utf-8") as f:
        metrics = _json.load(f)
    scalar_keys = [
        "label", "segmentation_label", "path_index", "distance", "target_branch_label",
        "peakv_cm_s", "netflow_mL_beat", "meanv_cm_s", "path_ic", "segmentation_label_ic",
        "path_direction",
        "peakv_forward_cm_s", "peakv_reverse_cm_s",
        "meanv_forward_cm_s", "meanv_reverse_cm_s",
        "netflow_forward_mL_beat", "netflow_reverse_mL_beat",
        "net_netflow_signed_mL_beat", "reflux_fraction",
        "meanv_signed_cm_s",
        "forward_sign", "forward_sign_source",
        "local_path_direction", "normal_tangent_cos",
        "tke_mean_J_m3", "tke_peak_J_m3", "tke_p95_J_m3",
        "pressure_gradient_mag_mean_Pa_m", "pressure_gradient_mag_peak_Pa_m", "pressure_gradient_mag_p95_Pa_m",
        "pressure_gradient_normal_mean_Pa_m", "pressure_gradient_normal_peak_Pa_m", "pressure_gradient_normal_p95_Pa_m",
        "relative_pressure_mean_Pa", "relative_pressure_peak_Pa", "relative_pressure_p95_Pa",
        "wss_wall_mean_Pa", "wss_wall_peak_Pa", "wss_wall_p95_Pa",
    ]
    table_rows = []
    for i, m in enumerate(metrics):
        row = {"plane_index": int(m.get("plane_index", i))}
        for k in scalar_keys:
            if k in m:
                row[k] = m[k]
        row["center_x"] = m["center"][0] if "center" in m else None
        row["center_y"] = m["center"][1] if "center" in m else None
        row["center_z"] = m["center"][2] if "center" in m else None
        Nt = len(m.get("flowrate_mL_s", []))
        for t in range(Nt):
            row[f"flowrate_t{t}"] = m["flowrate_mL_s"][t]
        for t in range(len(m.get("area_mm2", []))):
            row[f"area_t{t}"] = m["area_mm2"][t]
        for t in range(len(m.get("meanv_cm_s_t", []))):
            row[f"meanv_t{t}"] = m["meanv_cm_s_t"][t]
        for t in range(len(m.get("flowrate_forward_mL_s", []))):
            row[f"flowrate_fwd_t{t}"] = m["flowrate_forward_mL_s"][t]
        for t in range(len(m.get("flowrate_reverse_mL_s", []))):
            row[f"flowrate_rev_t{t}"] = m["flowrate_reverse_mL_s"][t]
        for t in range(len(m.get("meanv_forward_cm_s_t", []))):
            row[f"meanv_fwd_t{t}"] = m["meanv_forward_cm_s_t"][t]
        for t in range(len(m.get("meanv_reverse_cm_s_t", []))):
            row[f"meanv_rev_t{t}"] = m["meanv_reverse_cm_s_t"][t]
        for t in range(len(m.get("flowrate_signed_mL_s", []))):
            row[f"flowrate_signed_t{t}"] = m["flowrate_signed_mL_s"][t]
        for t in range(len(m.get("meanv_signed_cm_s_t", []))):
            row[f"meanv_signed_t{t}"] = m["meanv_signed_cm_s_t"][t]
        fork_ic = m.get("fork_ic", [])
        for fi, fic in enumerate(fork_ic):
            row[f"fork{fi}_id"] = fic.get("fork_id", -1)
            row[f"fork{fi}_role"] = fic.get("role", "")
            row[f"fork{fi}_ic"] = fic.get("ic", 1.0)
        table_rows.append(row)
    qc_data = None
    if qc_json_path is not None:
        with open(qc_json_path, "r", encoding="utf-8") as f:
            qc_data = _json.load(f)
    return table_rows, metrics, qc_data
