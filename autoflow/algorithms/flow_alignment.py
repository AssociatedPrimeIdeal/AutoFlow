"""Compute only axis and phase alignment on existing physical graph arms."""

import numpy as np
from scipy.ndimage import map_coordinates

from ..task_control import check_cancelled


def branch_flow_evidence(points, flow, *, spacing=(1, 1, 1), origin=(0, 0, 0),
                         mask=None, mask_spacing=None, sample_step_mm=.5):
    """Compute axis agreement and phase agreement, without other metrics.

    Sample acquired velocity by linear interpolation. Tangents span one
    acquired voxel on each side, reducing digital stair-step sensitivity.
    Use the same velocity sampling and signal selection as earlier trials.
    """
    empty = dict(status='unavailable', axis_alignment=None, phase_alignment=None)
    if not np.isfinite(sample_step_mm) or sample_step_mm<=0:
        raise ValueError('sample_step_mm must be positive and finite')
    if flow is None:
        return dict(empty, reason='flow_unavailable')
    field = np.asarray(flow)
    if field.ndim == 4 and field.shape[-1] == 3:
        field = field[..., None, :]
    if field.ndim != 5 or field.shape[-1] != 3:
        raise ValueError('Branch velocity must have shape X,Y,Z,T,3')
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(points) < 2 or not np.all(np.isfinite(points)):
        return dict(empty, reason='empty_branch')
    spacing = np.asarray(spacing, dtype=float)
    origin = np.asarray(origin, dtype=float)
    if np.any(spacing <= 0) or not np.all(np.isfinite(spacing)):
        raise ValueError('Branch velocity spacing must be positive and finite')
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    keep = np.r_[True, np.diff(arc) > 1e-10]
    points, arc = points[keep], arc[keep]
    if len(points) < 2 or arc[-1] <= 1e-8:
        return dict(empty, reason='empty_branch')
    count = max(1, int(np.ceil(arc[-1]/float(sample_step_mm))))
    boundaries = np.linspace(0., arc[-1], count+1)
    sample = .5*(boundaries[:-1]+boundaries[1:])
    weights = np.diff(boundaries)
    interpolate = lambda s: np.column_stack([np.interp(s, arc, points[:, a]) for a in range(3)])
    centers = interpolate(sample)
    half_window = max(float(spacing.min()), float(sample_step_mm))
    tangent = interpolate(np.minimum(sample+half_window, arc[-1]))-interpolate(np.maximum(0., sample-half_window))
    norm = np.linalg.norm(tangent, axis=1)
    tangent /= np.maximum(norm[:, None], 1e-12)
    xyz = (centers-origin)/spacing
    nt = field.shape[3]
    coordinates = np.vstack((np.repeat(xyz.T, nt, axis=1), np.tile(np.arange(nt), count)))
    check_cancelled()
    vectors = np.stack([map_coordinates(field[..., a], coordinates, order=1,
                        mode='constant', cval=np.nan, prefilter=False).reshape(count, nt)
                        for a in range(3)], axis=-1)
    valid = np.all(np.isfinite(vectors), axis=-1) & (norm[:, None] > 1e-10)
    valid &= np.all((xyz >= 0) & (xyz <= np.asarray(field.shape[:3])-1), axis=1)[:, None]
    if mask is not None:
        mask = np.asarray(mask)
        mask_xyz = (centers-origin)/np.asarray(spacing if mask_spacing is None else mask_spacing)
        if mask.ndim == 3:
            occupied = map_coordinates(mask, mask_xyz.T, order=0,
                                       mode='constant', cval=0, prefilter=False)[:, None]
        elif mask.ndim == 4 and mask.shape[3] in (1, nt):
            frames = np.zeros(nt, int) if mask.shape[3] == 1 else np.arange(nt)
            coords = np.vstack((np.repeat(mask_xyz.T, nt, axis=1), np.tile(frames, count)))
            occupied = map_coordinates(mask, coords, order=0,
                                       mode='constant', cval=0, prefilter=False).reshape(count, nt)
        else:
            raise ValueError('Branch mask must have three spatial axes and matching phases')
        valid &= occupied > 0
    speed = np.linalg.norm(np.where(np.isfinite(vectors), vectors, 0.), axis=-1)
    scale = float(np.percentile(speed[valid], 90)) if np.any(valid) else 0.
    active = valid & (speed > max(1e-12, .1*scale))
    dot = np.sum(np.where(np.isfinite(vectors), vectors, 0.)*tangent[:, None, :], axis=-1)
    weight = weights[:, None]*active
    denominator = np.sum(weight*speed, axis=0)
    numerator = np.sum(weight*dot, axis=0)
    total = float(denominator.sum())
    if total <= 1e-12:
        return dict(empty, reason='no_valid_velocity_signal')
    return dict(status='available', reason='', axis_alignment=float(np.sum(weight*np.abs(dot))/total),
                phase_alignment=float(np.abs(numerator).sum()/total))
