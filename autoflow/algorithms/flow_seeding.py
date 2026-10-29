"""Physical seed grids and explicit domains shared by GUI and batch tracing."""

import copy
import numpy as np

from ..task_control import check_cancelled
from .metrics.sampling import (
    _build_plane_slice_region, _resolve_plane_roi_frame, _plane_slice_point_mask,
    _points_in_polygon, _target_label_for_plane,
)
from .surfaces import _build_branch_grid


def streamline_inputs_available(workspace):
    """Check prerequisites for the selected export domain without allocating it."""
    if workspace.flow_raw is None:
        return False
    region = workspace.streamline_params.seed_region
    if region == "fov":
        return True
    if region == "segmentation":
        return workspace.segmask_binary is not None or workspace.segmask_3d is not None
    if region == "noise_free":
        return workspace.pcmra_render_mask is not None
    return False


def _spatial(spacing, origin):
    spacing, origin = np.asarray(spacing, float).reshape(3), np.asarray(origin, float).reshape(3)
    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0) or not np.all(np.isfinite(origin)):
        raise ValueError("Spacing must be positive and finite, and origin must be finite")
    return spacing, origin


def phase_mask(mask, shape, phase=0, *, name="Mask", time_count=None):
    if mask is None:
        raise ValueError(f"{name} is unavailable")
    array = np.asarray(mask)
    if array.ndim == 4:
        if array.shape[3] < 1 or (time_count is not None and array.shape[3] not in (1, time_count)):
            raise ValueError(f"{name} has a mismatched number of phases")
        array = array[..., min(int(phase), array.shape[3] - 1)]
    if array.ndim != 3 or tuple(array.shape) != tuple(shape):
        raise ValueError(f"{name} must match the velocity image spatial shape")
    return np.asarray(array, dtype=bool)


def volume_seed_domain(flow, region="segmentation", *, segmentation=None, noise_keep=None, phase=0):
    flow = np.asarray(flow)
    if flow.ndim != 5 or flow.shape[-1] != 3 or not 0 <= int(phase) < flow.shape[3]:
        raise ValueError("Velocity must be XYZT3 with a valid phase")
    finite = np.all(np.isfinite(flow[..., int(phase), :]), axis=-1)
    if region == "fov":
        return finite
    if region == "segmentation":
        mask = phase_mask(segmentation, flow.shape[:3], phase, name="Segmentation", time_count=flow.shape[3])
    elif region == "noise_free":
        # PC-MRA/noise removal stores True for retained, non-noise voxels.
        mask = phase_mask(noise_keep, flow.shape[:3], phase, name="Noise-removal mask", time_count=flow.shape[3])
    else:
        raise ValueError(f"Unknown seed region: {region}")
    return mask & finite


def points_in_volume(points, mask, spacing, origin):
    spacing, origin = _spatial(spacing, origin)
    points = np.asarray(points, float).reshape(-1, 3)
    coordinates = (points - origin) / spacing
    finite = np.all(np.isfinite(coordinates), axis=1)
    indices = np.floor(np.where(np.isfinite(coordinates), coordinates, -1)).astype(np.int64)
    inside = finite & np.all((indices >= 0) & (indices < np.asarray(mask.shape)), axis=1)
    rows = np.flatnonzero(inside)
    inside[rows] &= np.asarray(mask, bool)[tuple(indices[rows].T)]
    return inside


def _grid_chunks(axes):
    sizes = tuple(len(axis) for axis in axes)
    count = int(np.prod(sizes, dtype=np.int64))
    if count > 20_000_000:
        raise ValueError(f"Grid has {count:,} candidate seeds; increase Grid spacing (mm)")
    for start in range(0, count, 131072):
        check_cancelled()
        flat = np.arange(start, min(start + 131072, count), dtype=np.int64)
        indices = np.unravel_index(flat, sizes)
        yield np.column_stack([axis[index] for axis, index in zip(axes, indices)])


def volume_seed_points(mask, spacing, origin, *, distribution="random", grid_spacing_mm=0.,
                       mode="ratio", count=1000, ratio=.1, min_seeds=50, rng_seed=0):
    mask = np.asarray(mask, bool)
    spacing, origin = _spatial(spacing, origin)
    if mask.ndim != 3:
        raise ValueError("Seed domain must be 3D")
    if not np.any(mask):
        return np.empty((0, 3), float)
    if distribution == "random":
        indices = np.argwhere(mask)
        requested = int(count) if mode == "fixed" else max(int(min_seeds), int(round(len(indices) * float(ratio))))
        requested = max(0, min(len(indices), requested))
        picked = np.random.default_rng(int(rng_seed)).choice(len(indices), requested, replace=False)
        # Keep legacy random seed coordinates for old saved workspaces.
        return origin + indices[picked].astype(float) * spacing
    if distribution != "uniform":
        raise ValueError(f"Unknown seed distribution: {distribution}")
    requested_step = float(grid_spacing_mm)
    if not np.isfinite(requested_step) or requested_step < 0:
        raise ValueError("Grid spacing (mm) must be nonnegative and finite; 0 selects 3 × voxel spacing")
    steps = 3.0 * spacing if requested_step == 0 else np.full(3, requested_step)
    axes = []
    for axis in range(3):
        step = steps[axis]
        occupied = np.flatnonzero(np.any(mask, axis=tuple(i for i in range(3) if i != axis)))
        first = max(0, int(np.ceil(occupied[0] * spacing[axis] / step - .5)))
        last = int(np.ceil((occupied[-1] + 1) * spacing[axis] / step - .5))
        axes.append(origin[axis] + (np.arange(first, last, dtype=float) + .5) * step)
    selected = [points[points_in_volume(points, mask, spacing, origin)] for points in _grid_chunks(axes)]
    return np.vstack(selected) if selected else np.empty((0, 3), float)


def plane_basis(plane):
    normal = np.asarray(plane.normal, float).reshape(3).copy()
    length = np.linalg.norm(normal)
    if not np.isfinite(length) or length < 1e-12:
        raise ValueError("Plane normal must be finite and nonzero")
    normal /= length
    reference = np.eye(3)[int(np.argmin(np.abs(normal)))]
    horizontal = np.cross(reference, normal); horizontal /= np.linalg.norm(horizontal)
    return horizontal, np.cross(normal, horizontal)


def plane_seed_points(flow, plane, spacing, origin, *, phase=0, region="segmentation",
                      segmentation=None, segmentation_labels=None, branch_labels=None,
                      distribution="random", grid_spacing_mm=1., mode="fixed", count=250,
                      ratio=.2, min_seeds=50, rng_seed=0):
    """Resolve the metric ROI, including hand edits and missing-frame carry."""
    flow = np.asarray(flow)
    spacing, origin = _spatial(spacing, origin)
    finite = volume_seed_domain(flow, "fov", phase=phase)
    shape = flow.shape[:3]
    if region not in {"segmentation", "contour"}:
        raise ValueError(f"Unknown plane seed region: {region}")
    if region == "segmentation" and segmentation is None:
        raise ValueError("Segmentation is unavailable")
    selected_plane = copy.copy(plane)
    if region == "segmentation":
        selected_plane.roi_polygon_uv_mm = []
        selected_plane.roi_edit_operations = {}
    empty = np.zeros(shape, bool) if segmentation is None else None
    branch_grid = _build_branch_grid(branch_labels, spacing, origin)
    target = _target_label_for_plane(selected_plane, branch_labels, spacing, origin)
    label = int(getattr(selected_plane, "segmentation_label", 0) or 0)
    def build(t):
        mask = empty if segmentation is None else phase_mask(segmentation, shape, t, name="Segmentation", time_count=flow.shape[3])
        if label > 0 and segmentation_labels is not None:
            labels = np.asarray(segmentation_labels)
            if labels.ndim == 4:
                labels = labels[..., min(t, labels.shape[3] - 1)]
            if labels.shape == mask.shape:
                mask = mask & (labels == label)
        return _build_plane_slice_region(mask, selected_plane, spacing, origin,
            branch_grid=branch_grid, target_label=target, select_connected=True, frame_index=t)
    if region == "contour":
        from .roi_time import interpolate_plane_regions
        history = {}
        cut, source = _resolve_plane_roi_frame(history, selected_plane, int(phase), build,
            time_count=flow.shape[3], interpolate=lambda a, b, w: interpolate_plane_regions(
                a, b, w, selected_plane, shape, spacing, origin))
        if history['_roi_frame_provenance'][int(phase)]['mode'] == 'interpolated':
            source = None
    else:
        cut, source = build(int(phase)), int(phase)
    if cut is None or not cut.n_cells:
        return np.empty((0, 3), float), source
    if distribution == "random":
        points = np.asarray(cut.cell_centers().points, float)
        requested = int(count) if mode == "fixed" else min(int(count), max(int(min_seeds), int(np.ceil(len(points) * float(ratio)))))
        requested = max(0, min(requested, len(points)))
        rng = np.random.default_rng(int(rng_seed) + 104729 * int(getattr(plane, "path_index", 0)) + int(phase))
        if len(points) > requested:
            points = points[np.sort(rng.choice(len(points), requested, replace=False))]
    elif distribution == "uniform":
        step = float(grid_spacing_mm)
        if not np.isfinite(step) or step <= 0:
            raise ValueError("Grid spacing (mm) must be positive and finite")
        horizontal, vertical = plane_basis(plane)
        center = np.asarray(plane.center, float).reshape(3) + origin
        uv = (np.asarray(cut.points) - center) @ np.column_stack((horizontal, vertical))
        axes = [np.arange(np.ceil(uv[:, i].min()/step), np.floor(uv[:, i].max()/step)+1) * step for i in range(2)]
        ids = np.asarray(cut.cell_data.get("_cell_id", []), np.int64)
        operations = (getattr(selected_plane, "roi_edit_operations", {}) or {}).get(str(source), [])
        legacy_polygon = getattr(selected_plane, "roi_polygon_uv_mm", [])
        selected = []
        for coordinates in _grid_chunks(axes):
            points = center + coordinates[:, :1] * horizontal + coordinates[:, 1:] * vertical
            keep = _plane_slice_point_mask(points, {"cell_ids": ids}, shape, spacing, origin)
            if operations:
                edited = np.ones(len(coordinates), bool)
                for operation in operations:
                    polygon = operation.get("polygon", [])
                    if len(polygon) < 3:
                        continue
                    hit = _points_in_polygon(coordinates, polygon)
                    mode = operation.get("mode", "remove")
                    if mode == "replace": edited = hit
                    elif mode == "add": edited |= hit
                    else: edited &= ~hit
                keep &= edited
            elif len(legacy_polygon) >= 3:
                keep &= _points_in_polygon(coordinates, legacy_polygon)
            selected.append(points[keep])
        points = np.vstack(selected) if selected else np.empty((0, 3), float)
    else:
        raise ValueError(f"Unknown seed distribution: {distribution}")
    return points[points_in_volume(points, finite, spacing, origin)], source
