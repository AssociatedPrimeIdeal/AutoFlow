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
                        origin=np.asarray(workspace.origin) - 0.5 * spacing)
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
    points = np.asarray(workspace.origin) + (voxels + 0.5) * np.asarray(workspace.resolution)
    cloud = pv.PolyData(points)
    cloud.point_data['Noise mask'] = values.reshape(-1)[indices]
    return cloud
