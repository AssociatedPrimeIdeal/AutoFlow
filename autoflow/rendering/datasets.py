"""Metric display datasets shared by the GUI and movie renderers."""

import numpy as np
import pyvista as pv

from ..algorithms import create_uniform_grid
from ..algorithms.surfaces import _extract_surface


def tke_display_mesh(workspace, t):
    if workspace.derived.tke_array is None:
        legacy = workspace.derived.tke_volume
        if legacy is not None and not isinstance(legacy, pv.ImageData):
            legacy = legacy.cast_to_unstructured_grid().triangulate()
        return legacy
    array = np.asarray(workspace.derived.tke_array, dtype=np.float32)
    phase = min(max(0, int(t)), array.shape[3] - 1) if array.ndim == 4 else 0
    volume = array[..., phase] if array.ndim == 4 else array
    mask = workspace.segmask_binary
    if mask is not None and mask.ndim == 4:
        mask = mask[..., min(max(0, int(t)), mask.shape[3] - 1)]
    if mask is None:
        mask = workspace.segmask_3d
    if mask is None:
        mask = np.ones(volume.shape, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    # Place the measured energy at voxel centres, with a transparent padded
    # background. Volume rendering reveals interior energy instead of only
    # painting the vessel's outermost cells.
    spacing = np.asarray(workspace.resolution, dtype=float)
    grid = pv.ImageData(dimensions=np.array(volume.shape) + 2, spacing=spacing,
                        origin=np.asarray(workspace.origin) - (1.0 if workspace.segmentation_labels_1mm is not None else 0.5) * spacing)
    grid.point_data['TKE'] = np.pad(np.where(mask & np.isfinite(volume), volume, 0.0), 1).ravel(order='F')
    return grid


def display_support_surface(mask, spacing, origin, smooth_iter=80):
    """Closed display geometry; scientific volume arrays remain unchanged."""
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return None
    spacing = np.asarray(spacing, dtype=float)
    grid = pv.ImageData(dimensions=np.array(mask.shape) + 2, spacing=spacing,
                        origin=np.asarray(origin) - 0.5 * spacing)
    grid.point_data['support'] = np.pad(mask.astype(np.float32), 1).ravel(order='F')
    surface = grid.contour([0.5], scalars='support')
    surface.clear_data()
    if smooth_iter > 0:
        surface = surface.smooth_taubin(n_iter=int(smooth_iter), pass_band=0.1)
    return surface


def sample_display_field(volume, mask, surface, spacing, origin, name):
    """Interpolate using finite supported values, without zero-background dilution."""
    if surface is None or not surface.n_points:
        return None
    volume = np.asarray(volume, dtype=np.float32)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(volume)
    grid = create_uniform_grid(np.where(valid, volume, 0.0), spacing, origin=origin, name='numerator')
    grid.cell_data['weight'] = valid.astype(np.float32).ravel(order='F')
    sampled = surface.sample(grid.cell_data_to_point_data(pass_cell_data=False))
    weights = np.asarray(sampled['weight'])
    available = np.asarray(sampled['vtkValidPointMask'], dtype=bool) & (weights > 1e-6)
    values = np.full(sampled.n_points, np.nan, dtype=np.float32)
    np.divide(sampled['numerator'], weights, out=values, where=available)
    sampled.clear_data()
    sampled.point_data[name] = values
    return sampled


def active_wrap_mesh(result, t, spacing, origin, *, counts=False):
    field = result.get('wrap_count' if counts else 'wrap_mask')
    if field is None:
        return None
    arr = np.asarray(field)
    if arr.ndim == 5:
        arr = arr[..., min(max(0, int(t)), arr.shape[3] - 1), :]
    if arr.ndim == 4:
        if counts:
            # Show the signed count of the component with largest |k|. This
            # includes unwrap in any velocity direction, with x/y/z tie order.
            index = np.argmax(np.abs(arr), axis=-1)
            arr = np.take_along_axis(arr, index[..., None], axis=-1)[..., 0]
        else:
            arr = np.any(arr, axis=-1)
    name = 'wrap_count' if counts else 'wrap_mask'
    active = np.isfinite(arr) & (arr != 0)
    if not np.any(active):
        return None
    grid = create_uniform_grid(np.asarray(arr, dtype=np.float32), spacing, origin=origin, name=name)
    grid.cell_data['active'] = active.astype(np.uint8).ravel(order='F')
    return _extract_surface(grid.threshold(0.5, scalars='active', preference='cell'))


def noise_display_points(workspace, max_points=40000):
    """Sparse rejected-voxel centres; zero-signal padding is omitted in 3D."""
    if workspace.pcmra_render_mask is None:
        return None
    mask = np.asarray(workspace.pcmra_render_mask, dtype=bool)
    rejected = ~mask if mask.ndim == 3 else ~np.any(mask, axis=3)
    values = rejected.astype(np.float32)
    if workspace.mag_raw is not None:
        magnitude = np.asarray(workspace.mag_raw)
        mean = np.mean(np.nan_to_num(magnitude, nan=0.0, posinf=0.0, neginf=0.0), axis=3, dtype=np.float64) if magnitude.ndim == 4 else magnitude
        if mean.shape == rejected.shape:
            threshold = float((workspace.noise_removal_result or {}).get('magnitude_threshold', 0.0))
            if threshold <= 0:
                threshold = 0.05 * float(np.max(mean))
            if threshold > 0:
                values *= np.clip(mean / threshold, 0.0, 1.0).astype(np.float32) ** 2
            else:
                values *= mean > 0
    indices = np.flatnonzero(values > 0)
    if not indices.size:
        return None
    if indices.size > max_points:
        indices = np.sort(np.random.default_rng(0).choice(indices, max_points, replace=False))
    voxels = np.column_stack(np.unravel_index(indices, rejected.shape))
    offset = 0.0 if workspace.segmentation_labels_1mm is not None else 0.5
    points = np.asarray(workspace.origin) + (voxels + offset) * np.asarray(workspace.resolution)
    cloud = pv.PolyData(points)
    cloud.point_data['Noise mask'] = values.reshape(-1)[indices]
    return cloud


_METRIC_FIELDS = {
    "tke_volume": ("tke_array", "TKE", None),
    "pressure_gradient_volume": ("pressure_gradient_magnitude", "PressureGradient", "pressure_gradient_support_mask"),
    "relative_pressure_volume": ("relative_pressure_array", "RelativePressure", "pressure_gradient_support_mask"),
    "vorticity_magnitude_volume": ("vorticity_magnitude", "Vorticity Magnitude", "vortex_support_mask"),
    "q_criterion_volume": ("q_criterion_array", "Q-Criterion", "vortex_support_mask"),
    "swirling_strength_volume": ("swirling_strength_array", "Swirling Strength", "vortex_support_mask"),
}


def _display_phase(array, t):
    array = np.asarray(array)
    return array[..., min(max(0, int(t)), array.shape[3]-1)] if array.ndim == 4 else array


def _smooth_display_values(mesh, values, iterations):
    """Smooth finite display samples, without making unavailable samples valid."""
    from scipy.sparse import csr_matrix
    values = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(values)
    if not iterations or not np.any(valid):
        return values
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    row = np.r_[faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 1], faces[:, 2], faces[:, 0]]
    col = np.r_[faces[:, 1], faces[:, 2], faces[:, 0], faces[:, 0], faces[:, 1], faces[:, 2]]
    adjacency = csr_matrix((np.ones(len(row)), (row, col)), shape=(mesh.n_points, mesh.n_points))
    degree = np.maximum(np.asarray(adjacency.sum(axis=1)).ravel(), 1.)
    numerator, weight = np.where(valid, values, 0.).astype(float), valid.astype(float)
    for _ in range(int(iterations)):
        numerator = .7*numerator + .3*(adjacency@numerator)/degree
        weight = .7*weight + .3*(adjacency@weight)/degree
    result = np.full(mesh.n_points, np.nan, np.float32)
    np.divide(numerator, weight, out=result, where=valid & (weight > 1e-6))
    return result


def metric_display_mesh(workspace, data_key, t):
    """Phase-specific values on a cleaned lumen surface, with unavailable values transparent.

    Gaussian filtering, local projection and surface interpolation are display
    operations only. Quantitative arrays and their validity masks are untouched.
    """
    from scipy.ndimage import gaussian_filter
    from scipy.spatial import cKDTree
    from ..algorithms.surfaces import build_cell_mask_surface
    from ..config import DEFAULT_RENDER_STYLE_CFG
    from .style import render_style_settings, wss_display_surface

    cfg = {**DEFAULT_RENDER_STYLE_CFG["metric_surface"],
           **render_style_settings(workspace).get("metric_surface", {})}
    spacing = np.asarray(workspace.resolution, float)
    origin = np.asarray(workspace.origin, float)
    is_fine = getattr(workspace, "segmentation_labels_1mm", None) is not None
    offset = 0. if is_fine else .5
    native_mask = getattr(workspace, "segmask_binary", None)
    native_mask = None if native_mask is None else _display_phase(native_mask, t).astype(bool)
    native_labels = getattr(workspace, "segmask_labels", None)
    native_labels = None if native_labels is None else _display_phase(native_labels, t)

    if data_key == "wss_surface_live":
        surfaces = workspace.derived.wss_surfaces
        if not surfaces:
            return None
        source_wall = wss_display_surface(surfaces[min(max(int(t), 0), len(surfaces)-1)])
        if source_wall is None:
            return None
        volume, support, scalar = None, native_mask, "wss"
    else:
        field, scalar, support_field = _METRIC_FIELDS[data_key]
        source = getattr(workspace.derived, field, None)
        if source is None:
            return getattr(workspace.derived, "tke_volume", None) if data_key == "tke_volume" else None
        volume = _display_phase(source, t)
        support_source = getattr(workspace.derived, support_field, None) if support_field else None
        support = _display_phase(support_source, t).astype(bool) if support_source is not None else native_mask
        if support is None:
            support = np.isfinite(volume)
        support = np.asarray(support, bool) & np.isfinite(volume)
        if native_mask is not None:
            support &= native_mask

    geometry_labels = getattr(workspace, "segmentation_analysis_labels_3d", None)
    fine_geometry = geometry_labels is not None
    if geometry_labels is None:
        geometry_labels = getattr(workspace, "segmask_labels_3d", None)
    if geometry_labels is None:
        geometry_labels = getattr(workspace, "segmask_3d", None)
    temporal_geometry = geometry_labels is None
    if geometry_labels is None:
        geometry_labels = native_mask if native_mask is not None else support
    if geometry_labels is None or not np.any(geometry_labels):
        return None
    geometry_spacing = np.asarray(workspace.segmentation_spacing if fine_geometry else spacing, float)
    geometry_origin = origin-.5*geometry_spacing if fine_geometry else origin
    revision = int(getattr(getattr(workspace, "segmentation", None), "revision", 0))
    key = (id(geometry_labels), int(t) if temporal_geometry else -1, revision, tuple(geometry_spacing), tuple(origin),
           int(cfg["surface_smoothing_iterations"]), int(cfg["surface_taubin_iterations"]))
    cache = getattr(workspace, "_metric_display_geometry_cache", None)
    if cache is None or cache[0] != key:
        surface = build_cell_mask_surface(np.asarray(geometry_labels) > 0, geometry_spacing,
                  geometry_origin, smooth_iter=cfg["surface_smoothing_iterations"])
        if surface is None:
            return None
        surface = surface.triangulate()
        if cfg["surface_taubin_iterations"]:
            surface = surface.smooth_taubin(n_iter=cfg["surface_taubin_iterations"], pass_band=.08)
        surface.clear_data()
        workspace._metric_display_geometry_cache = (key, surface)
    else:
        surface = cache[1]
    values = np.full(surface.n_points, np.nan, np.float32)
    max_distance = float(cfg["max_projection_distance_voxels"])*float(np.max(spacing))

    if volume is None:
        raw = np.asarray(source_wall["wss"])
        if source_wall.n_points:
            k = min(4, source_wall.n_points)
            dist, indices = cKDTree(source_wall.points).query(surface.points, k=k)
            if k == 1:
                dist, indices = dist[:, None], indices[:, None]
            samples = raw[indices]
            weights = np.where(np.isfinite(samples), 1./np.maximum(dist, .03)**2, 0.)
            denominator = weights.sum(axis=1)
            available = np.isfinite(samples[:, 0]) & (dist[:, 0] <= max_distance) & (denominator > 0)
            np.divide((np.nan_to_num(samples)*weights).sum(axis=1), denominator,
                      out=values, where=available)
    elif np.any(support):
        sigma = float(cfg["smoothing_sigma"])
        smooth_weight = gaussian_filter(support.astype(np.float32), sigma)
        filtered = gaussian_filter(np.where(support, volume, 0.).astype(np.float32), sigma)
        filtered /= np.maximum(smooth_weight, 1e-6)
        lumen = native_mask if native_mask is not None else support
        if native_labels is None or native_labels.shape != volume.shape:
            native_labels = np.ones(volume.shape, np.uint8)
        occupied = np.argwhere(lumen)
        if len(occupied):
            _, nearest = cKDTree((occupied+offset)*spacing+origin).query(surface.points)
            surface_labels = native_labels[tuple(occupied[nearest].T)]
            for label in np.unique(surface_labels):
                target = np.flatnonzero(surface_labels == label)
                coordinates = np.argwhere(support & (native_labels == label))
                if not len(coordinates):
                    continue
                k = min(4, len(coordinates))
                dist, indices = cKDTree((coordinates+offset)*spacing+origin).query(surface.points[target], k=k)
                if k == 1:
                    dist, indices = dist[:, None], indices[:, None]
                weights = 1./np.maximum(dist, .03)**2
                samples = filtered[tuple(coordinates[indices].transpose(2, 0, 1))]
                available = dist[:, 0] <= max_distance
                values[target[available]] = ((samples*weights).sum(axis=1)/weights.sum(axis=1))[available]
    result = surface.copy(deep=True)
    result.point_data[scalar] = _smooth_display_values(result, values, cfg["scalar_smoothing_iterations"])
    result.point_data["display_valid"] = np.isfinite(result.point_data[scalar]).astype(np.uint8)
    return result
