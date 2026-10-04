"""Segmentation support, plane geometry, ROI selection and field sampling."""

import numpy as np
from ..surfaces import _build_branch_grid, _select_connected_region, create_uniform_field_grid

from ._common import _ensure_mask4d

# Replacement contours usually stay within the same local voxel box across
# successive edits. Cache those small support meshes so each edit does not
# rebuild an identical VTK threshold grid. The cache is intentionally bounded
# because meshes retain VTK-owned memory.
_LOCAL_SUPPORT_MESH_CACHE = {}
_LOCAL_SUPPORT_MESH_CACHE_MAX = 16


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
