"""Physical, radius-adaptive centerline fits with bounded lumen displacement."""

import os
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter1d, map_coordinates

from ..task_control import check_cancelled


from ..plane_rules import SMOOTHING_DEFAULTS, smoothing_options


class PathSmoothingContext:
    """Lazily prepare one cropped physical radius field per vessel group.

    Paths and ``origin`` must use the same coordinate system. Pipeline paths
    are local physical coordinates, so their context origin is zero even
    when the acquired image has a nonzero world origin.
    """

    def __init__(self, mask, spacing=(1, 1, 1), origin=(0, 0, 0)):
        mask = np.asarray(mask, dtype=bool)
        self.spacing = np.asarray(spacing, dtype=float).reshape(3)
        self.origin = np.asarray(origin, dtype=float).reshape(3)
        if mask.ndim != 3 or np.any(~np.isfinite(self.spacing)) or np.any(self.spacing <= 0):
            raise ValueError('Path smoothing needs a 3-D mask and positive physical spacing')
        bounds = [np.flatnonzero(np.any(mask, axis=tuple(k for k in range(3) if k != axis)))
                  for axis in range(3)]
        if not all(len(axis) for axis in bounds):
            self.mask = np.zeros((3, 3, 3), dtype=np.uint8)
        else:
            lower = np.array([axis[0] for axis in bounds])
            upper = np.array([axis[-1]+1 for axis in bounds])
            local = mask[tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))]
            self.mask = np.pad(local.astype(np.uint8), 1)
            self.origin = self.origin+(lower-1)*self.spacing
        self._radius = None

    def radii(self, points):
        if self._radius is None:
            check_cancelled()
            self._radius = distance_transform_edt(self.mask, sampling=self.spacing)
        coordinates = ((np.asarray(points)-self.origin)/self.spacing).T
        return map_coordinates(self._radius, coordinates, order=1, mode='constant', cval=0., prefilter=False)

    def contains(self, points):
        coordinates = ((np.asarray(points)-self.origin)/self.spacing).T
        return map_coordinates(self.mask, coordinates, order=0, mode='constant', cval=0, prefilter=False) > 0

    def constrain(self, source, delta):
        """Keep the complete move inside the original group, not just its end."""
        count = max(10, int(np.ceil(np.max(np.linalg.norm(delta, axis=1), initial=0.) /
                                    (0.25*float(np.min(self.spacing))))))
        alive = self.contains(source)
        fraction = np.zeros(len(source), dtype=float)
        for index in range(1, count+1):
            alive &= self.contains(source+(index/count)*delta)
            fraction[alive] = index/count
        return source+fraction[:, None]*delta, fraction


def _uniform_path(points, step):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(points) < 2:
        return points.copy(), np.zeros(len(points))
    if not np.all(np.isfinite(points)):
        raise ValueError('Centerline coordinates must be finite')
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    keep = np.r_[True, np.diff(arc) > 1e-10]
    points, arc = points[keep], arc[keep]
    if len(points) < 2:
        return points.copy(), np.zeros(len(points))
    distances = np.linspace(0., arc[-1], int(np.ceil(arc[-1]/step))+1)
    return np.column_stack([np.interp(distances, arc, points[:, k]) for k in range(3)]), distances


def _fit_batch(points, arc, radii, radius_factor, backend):
    """Solve local quadratic fits in bounded batches rather than Python points."""
    # Real centerlines are small and already reside on the CPU. Avoid a CUDA
    # import/context startup in auto mode; an explicit GPU remains measurable.
    xp, selected, fallback = np, 'cpu', None
    if backend == 'gpu':
        try:
            if os.name == 'nt':
                import torch  # Initialize the packaged CUDA runtime first.
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError('No CUDA device')
            xp, selected = cp, 'gpu'
        except Exception as error:
            fallback = f'{type(error).__name__}: {error}'
    half = np.maximum(radius_factor*radii, 1e-8)
    sample_step = np.minimum(.5, np.maximum(radii/5., 1e-4))
    sizes = np.maximum(5, (2*half/sample_step).astype(int)+1)
    delta = np.zeros_like(points)
    xp_arc, xp_points = xp.asarray(arc), xp.asarray(points)
    try:
        for count in np.unique(sizes):
            indices = np.flatnonzero(sizes == count)
            batch_size = max(1, 262144//int(count))
            for first in range(0, len(indices), batch_size):
                check_cancelled()
                ids = indices[first:first+batch_size]
                target, width = xp.asarray(arc[ids]), xp.asarray(half[ids])
                lower, upper = xp.maximum(0., target-width), xp.minimum(arc[-1], target+width)
                samples = lower[:, None]+(upper-lower)[:, None]*xp.linspace(0., 1., int(count))
                offset = samples-target[:, None]
                u = offset/width[:, None]
                weights = xp.maximum(1.-xp.abs(u)**3, 1e-12)**3
                # Rescaling only the polynomial basis keeps short paths and
                # large radii well conditioned without changing the intercept.
                v = offset/xp.minimum(width, arc[-1])[:, None]
                design = xp.stack((xp.ones_like(v), v, v*v), axis=-1)
                locations = xp.stack([xp.interp(samples, xp_arc, xp_points[:, k]) for k in range(3)], axis=-1)
                locations -= xp.asarray(points[ids])[:, None, :]
                normal = xp.einsum('nmi,nmj,nm->nij', design, design, weights)
                right = xp.einsum('nmi,nmc,nm->nic', design, locations, weights)
                coefficients = xp.linalg.solve(normal, right)
                output = coefficients[:, 0, :]
                delta[ids] = output if xp is np else xp.asnumpy(output)
        if not np.all(np.isfinite(delta)):
            raise FloatingPointError('Nonfinite local quadratic fit')
    except Exception as error:
        if selected != 'gpu':
            raise
        # CPU fallback repeats the same fit, never a different smoothing rule.
        delta, selected, _ = _fit_batch(points, arc, radii, radius_factor, 'cpu')
        fallback = f'{type(error).__name__}: {error}'
    return delta, selected, fallback


def prepare_centerline(points, *, context=None, smoothing_window=15,
                       smoothing_polyorder=2, inter_time=10, **options):
    """Return matched source/placement samples plus a serializable receipt.

    Radius-adaptive fitting uses 0.5-mm physical arc samples by default, a
    tricube-weighted quadratic over a half-window of 2 local EDT radii, a
    0.15-radius movement cap, and continuous lumen containment. Every path
    endpoint stays fixed; graph junctions are endpoints of the input paths.
    """
    options = smoothing_options(options)
    method = options['smoothing_method']
    report = dict(options, endpoint_displacement_mm=0., constrained_samples=0)
    if method != 'radius_adaptive':
        from .paths import inter_points, smooth_path_savgol
        source = inter_points(points, time=inter_time)
        smooth = (smooth_path_savgol(source, smoothing_window, smoothing_polyorder)
                  if method == 'savgol' else source.copy())
        report.update(backend='cpu', samples=len(source))
        return source, smooth, report
    source, arc = _uniform_path(points, options['smoothing_sample_step_mm'])
    report['samples'] = len(source)
    if len(source) < 3 or arc[-1] <= 1e-10:
        report['backend'] = 'identity'
        return source, source.copy(), report
    if context is None:
        report.update(backend='identity', lumen_guard=False, reason='segmentation_unavailable')
        return source, source.copy(), report
    radii = np.maximum(context.radii(source), .5)
    report['outside_source_samples'] = int(np.count_nonzero(~context.contains(source)))
    sigma = options['smoothing_radius_sigma_mm']
    if sigma > 0:
        radii = gaussian_filter1d(radii, sigma/max(arc[1]-arc[0], 1e-8), mode='nearest')
    delta, selected, fallback = _fit_batch(source, arc, radii, options['smoothing_radius_factor'], options['smoothing_backend'])
    proposed = np.linalg.norm(delta, axis=1)
    cap = options['smoothing_max_displacement_ratio']*radii
    delta *= np.minimum(1., cap/np.maximum(proposed, 1e-12))[:, None]
    delta[[0, -1]] = 0.
    if context is not None:
        smooth, fraction = context.constrain(source, delta)
        report['constrained_samples'] = int(np.count_nonzero(fraction < 1.))
    else:
        smooth = source+delta
    smooth[[0, -1]] = source[[0, -1]]
    report.update(backend=selected, lumen_guard=context is not None,
                  outside_fitted_samples=int(np.count_nonzero(~context.contains(smooth))),
                  p95_displacement_mm=float(np.percentile(np.linalg.norm(smooth-source, axis=1), 95)),
                  max_displacement_mm=float(np.max(np.linalg.norm(smooth-source, axis=1), initial=0.)))
    if fallback:
        report['backend_fallback'] = fallback
    return source, smooth, report
