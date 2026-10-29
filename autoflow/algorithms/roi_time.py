"""Periodic, order-independent resolution of missing plane ROI geometry."""

import copy
import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates

from ..task_control import check_cancelled


def roi_pixel_metadata(sources=(), interpolated=False, weight=None, resolution='missing'):
    pair = list(sources)[:2]
    pair += [-1] * (2 - len(pair))
    modes = {'observed': 0, 'interpolated': 1, 'single_source_hold': 2,
             'explicit_empty': 3, 'missing': 4, 'interpolation_empty': 5}
    return {'roi_source_frames': np.asarray(pair, dtype=np.int32),
            'roi_interpolated': np.asarray([interpolated], dtype=np.uint8),
            'roi_interpolation_weight': np.asarray([np.nan if weight is None else weight], dtype=float),
            'roi_resolution_code': np.asarray([modes.get(resolution, 4)], dtype=np.uint8)}


def resolve_roi_frame(cache, plane, frame, count, build, interpolate):
    count = int(count)
    t = int(frame)
    if count < 1 or not 0 <= t < count:
        raise ValueError("ROI phase is outside the cardiac cycle")
    raw = cache.setdefault("_observed_roi_frames", {})
    resolved = cache.setdefault("_resolved_roi_frames", {})
    provenance = cache.setdefault("_roi_frame_provenance", {})
    if t in resolved:
        return resolved[t]
    edits = getattr(plane, "roi_edit_operations", {}) or {}

    def observed(i):
        if i not in raw:
            check_cancelled()
            raw[i] = build(i)
        return raw[i]

    def save(region, source, mode, sources=(), weight=None):
        resolved[t] = (region, source)
        provenance[t] = {"mode": mode, "source_frames": list(sources), "weight": weight}
        return resolved[t]

    current = observed(t)
    if current is not None:
        return save(current, t, "observed", [t, t], 0.0)
    if edits.get(str(t)):
        return save(None, None, "explicit_empty")

    def neighbor(direction):
        for distance in range(1, count):
            i = (t + direction * distance) % count
            value = observed(i)
            if value is not None:
                return i, distance, value
            # Do not interpolate or hold a shape through a deliberate deletion.
            if edits.get(str(i)):
                break
        return None

    before, after = neighbor(-1), neighbor(1)
    if before is not None and after is not None and before[0] != after[0]:
        weight = before[1] / float(before[1] + after[1])
        region = interpolate(before[2], after[2], weight)
        if region is None:
            return save(None, None, "interpolation_empty", [before[0], after[0]], weight)
        return save(region, before[0], "interpolated", [before[0], after[0]], weight)
    available = before or after
    if available is not None:
        return save(available[2], available[0], "single_source_hold", [available[0], available[0]], 0.0)
    return save(None, None, "missing")


def interpolate_plane_regions(first, second, weight, plane, shape, spacing, origin, *, voxel_centered=False):
    """Blend centroid-aligned signed distances on a common physical plane grid.

    Observed endpoint meshes are never changed. The interpolated region retains
    source voxel ids/areas so all metrics sample fields from the requested phase.
    """
    from matplotlib.path import Path as Polygon
    from .metrics.sampling import _build_plane_slice_region

    spacing = np.asarray(spacing, float)
    origin = np.asarray(origin, float)
    normal = np.asarray(plane.normal, float).copy()
    normal /= np.linalg.norm(normal)
    u = np.cross(np.eye(3)[np.argmin(np.abs(normal))], normal)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    basis = np.column_stack((u, v))
    center = np.asarray(plane.center, float) + origin
    meshes = [region.extract_surface().triangulate() for region in (first, second)]
    coordinates = [(np.asarray(mesh.points) - center) @ basis for mesh in meshes]
    centroids = []
    for mesh in meshes:
        area = np.asarray(mesh.compute_cell_sizes(area=True).cell_data['Area'])
        centroids.append(np.average((np.asarray(mesh.cell_centers().points) - center) @ basis, axis=0, weights=area))
    destination = (1 - weight) * centroids[0] + weight * centroids[1]
    aligned = [points - centroid for points, centroid in zip(coordinates, centroids)]
    step = float(np.min(spacing)) / 2
    lower = np.floor(np.min(np.vstack(aligned), axis=0) / step) * step - 3 * step
    upper = np.ceil(np.max(np.vstack(aligned), axis=0) / step) * step + 3 * step
    grid_origin = lower + step / 2
    axes = [np.arange(grid_origin[i], upper[i], step) for i in range(2)]
    if len(axes[0]) * len(axes[1]) > 4_000_000:
        raise ValueError("Temporal ROI interpolation exceeds the local plane-grid limit")
    xx, yy = np.meshgrid(*axes, indexing='ij')
    samples = np.column_stack((xx.ravel(), yy.ravel()))
    distances = []
    for mesh, points in zip(meshes, aligned):
        inside = np.zeros(len(samples), bool)
        faces = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
        # Restrict each triangle query to its local box rather than scanning
        # every raster point for every cell in a large vessel section.
        for triangle in faces:
            polygon = points[triangle]
            selected = np.all((samples >= polygon.min(axis=0) - 1e-9) & (samples <= polygon.max(axis=0) + 1e-9), axis=1)
            ids = np.flatnonzero(selected)
            if len(ids):
                boundary = Polygon(polygon)
                inside[ids] |= (boundary.contains_points(samples[ids], radius=step * 1e-6)
                                | boundary.contains_points(samples[ids], radius=-step * 1e-6))
        inside = inside.reshape(xx.shape)
        if not inside.any():
            return None
        distances.append((distance_transform_edt(inside) - distance_transform_edt(~inside)) * step)
    signed = (1 - weight) * distances[0] + weight * distances[1]
    rectangle = np.array([[lower[0], lower[1]], [upper[0], lower[1]], [upper[0], upper[1]], [lower[0], upper[1]]]) + destination
    target = copy.copy(plane)
    target.roi_polygon_uv_mm = []
    target.roi_edit_operations = {'0': [{'mode': 'replace', 'polygon': rectangle.tolist()}]}
    region = _build_plane_slice_region(np.zeros(shape, bool), target, spacing, origin, frame_index=0, voxel_centered=voxel_centered)
    if region is None:
        return None
    uv = (np.asarray(region.cell_centers().points) - center) @ basis - destination
    values = map_coordinates(signed, ((uv - grid_origin) / step).T, order=1, mode='constant', cval=-step, prefilter=False)
    selected = np.flatnonzero(values >= 0)
    if not len(selected):
        return None
    return region.extract_cells(selected).compute_cell_sizes(area=True)
