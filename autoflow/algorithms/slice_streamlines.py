"""Display geometry for instantaneous planar and slab-projected streamlines."""

import numpy as np
from scipy.ndimage import map_coordinates


def limit_polyline_length(points, maximum):
    """Retain an exact physical arc length, interpolating the final endpoint."""
    points = np.asarray(points, dtype=float)
    if maximum <= 0 or len(points) < 2:
        return points
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if distance[-1] <= maximum:
        return points
    index = min(int(np.searchsorted(distance, maximum, side="right")), len(points) - 1)
    fraction = (maximum - distance[index - 1]) / max(distance[index] - distance[index - 1], 1e-12)
    return np.vstack((points[:index], points[index - 1] + fraction * (points[index] - points[index - 1])))


def trace_planar_streamlines(vectors, spacing, *, mask=None, offset=(0.0, 0.0),
                             density=1.0, terminal_speed=1.0, max_steps=400):
    """Trace a horizontal/vertical cm/s field with a physical-space midpoint step.

    Arrays use horizontal-first order, matching SliceView. Returned mm polylines
    run upstream to downstream; the discarded normal velocity never contributes
    to integration or the terminal-speed threshold. No time integration occurs.
    """
    field = np.asarray(vectors, dtype=float)
    spacing = np.asarray(spacing, dtype=float).reshape(2)
    if field.ndim != 3 or field.shape[-1] != 2:
        raise ValueError("planar vectors must have shape (horizontal, vertical, 2)")
    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("planar spacing must be positive and finite")
    support = np.all(np.isfinite(field), axis=-1)
    if mask is not None:
        support &= np.asarray(mask, dtype=bool)
    threshold = max(float(terminal_speed), 1e-8)
    support &= np.linalg.norm(field, axis=-1) >= threshold
    occupied = np.argwhere(support)
    if not len(occupied) or min(field.shape[:2]) < 2:
        return []
    field = np.where(support[..., None], field, 0.0)
    support_bytes = support.astype(np.uint8)
    density = float(np.clip(density, 0.25, 3.0))
    seed_step = 4.0 * float(np.min(spacing)) / density
    bins = np.floor(occupied * spacing / seed_step).astype(int)
    _, picks = np.unique(bins, axis=0, return_index=True)
    picks = picks[np.linspace(0, len(picks) - 1, min(len(picks), 512), dtype=int)]
    seeds = occupied[picks].astype(float) * spacing
    step = 0.45 * float(np.min(spacing))
    limit = max(1, min(int(max_steps), 800))

    def direction(points):
        coords = (points / spacing).T
        sampled = np.column_stack([
            map_coordinates(field[..., component], coords, order=1, mode="constant", cval=0.0)
            for component in range(2)
        ])
        speed = np.linalg.norm(sampled, axis=1)
        inside = map_coordinates(support_bytes, coords, order=0, mode="constant", cval=0)
        valid = (inside > 0) & np.isfinite(speed) & (speed >= threshold)
        return sampled / np.maximum(speed[:, None], 1e-12), valid

    def integrate(sign):
        points = seeds.copy()
        alive = np.ones(len(seeds), dtype=bool)
        history = [points.copy()]
        for index in range(limit):
            active = np.flatnonzero(alive)
            selected = points[active]
            unit, valid = direction(selected)
            midpoint = selected + sign * 0.5 * step * unit
            middle_unit, middle_valid = direction(midpoint)
            following = selected + sign * step * middle_unit
            _, end_valid = direction(following)
            survived = valid & middle_valid & end_valid
            if index > 8:
                survived &= np.linalg.norm(following - seeds[active], axis=1) > 0.75 * step
            alive[active] = survived
            if not np.any(alive):
                break
            points[active[survived]] = following[survived]
            history.append(np.where(alive[:, None], points, np.nan))
        return np.asarray(history)

    backward, forward = integrate(-1.0), integrate(1.0)
    # Suppress duplicate trajectories through the same seed-density cells.
    used = set()
    curves = []
    for index, seed in enumerate(seeds):
        seed_bin = tuple(np.floor(seed / seed_step).astype(int))
        if seed_bin in used:
            continue
        before, after = backward[:, index], forward[1:, index]
        before = before[np.all(np.isfinite(before), axis=1)][::-1]
        after = after[np.all(np.isfinite(after), axis=1)]
        curve = np.vstack((before, after))
        if len(curve) < 3:
            continue
        used.update(map(tuple, np.floor(curve / seed_step).astype(int)))
        curves.append(curve + np.asarray(offset, dtype=float))
    return curves


def project_streamline_slab(mesh, center, horizontal, vertical, thickness, *, return_vectors=False, fraction=1.0):
    """Clip VTK polylines to a full-width mm slab, then project onto its basis.

    Segment clipping retains crossings even when both original endpoints are
    outside the slab. Separate lines and slab exits are never joined together.
    Input center and mesh points are in the same world coordinate system.
    With return_vectors=True, also return matching clipped tracer vectors
    (m/s). Fraction selects a deterministic subset of original trajectories.
    """
    empty = ([], []) if return_vectors else []
    if mesh is None or not mesh.n_points:
        return empty
    horizontal = np.asarray(horizontal, dtype=float).reshape(3)
    vertical = np.asarray(vertical, dtype=float).reshape(3)
    normal = np.cross(horizontal, vertical)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    points = np.asarray(mesh.points, dtype=float) - np.asarray(center, dtype=float).reshape(3)
    packed = np.asarray(mesh.lines, dtype=int)
    starts, ends, breaks = [], [], []
    cursor = 0
    line_index = 0
    fraction = float(np.clip(fraction, 0.0, 1.0))
    while cursor < len(packed):
        count = int(packed[cursor])
        indices = packed[cursor + 1:cursor + count + 1]
        # Stable integer hashing thins original trajectories before clipping,
        # preserving the subset when the slice moves or slab thickness changes.
        keep_line = ((line_index * 2654435761) % 4294967296) / 4294967296.0 < fraction
        if count >= 2 and keep_line:
            starts.extend(indices[:-1])
            ends.extend(indices[1:])
            breaks.extend([True] + [False] * (count - 2))
        cursor += count + 1
        line_index += 1
    if not starts:
        return empty
    a, b = points[starts], points[ends]
    da, db = a @ normal, b @ normal
    delta = db - da
    half = max(float(thickness), 1e-6) / 2.0
    parallel = np.abs(delta) < 1e-12
    safe_delta = np.where(parallel, 1.0, delta)
    first, second = (-half - da) / safe_delta, (half - da) / safe_delta
    lo = np.where(parallel, 0.0, np.maximum(0.0, np.minimum(first, second)))
    hi = np.where(parallel, 1.0, np.minimum(1.0, np.maximum(first, second)))
    keep = (hi >= lo) & (~parallel | (np.abs(da) <= half))
    keep &= np.all(np.isfinite(a), axis=1) & np.all(np.isfinite(b), axis=1)
    basis = np.column_stack((horizontal, vertical))
    p = (a + lo[:, None] * (b - a)) @ basis
    q = (a + hi[:, None] * (b - a)) @ basis
    keep &= np.linalg.norm(q - p, axis=1) > 1e-8
    selected = np.flatnonzero(keep)
    if not len(selected):
        return empty
    p, q = p[selected], q[selected]
    split = np.asarray(breaks, dtype=bool)[selected[1:]]
    split |= np.diff(selected) != 1
    split |= np.linalg.norm(p[1:] - q[:-1], axis=1) > 1e-7
    groups = np.split(np.arange(len(selected)), np.flatnonzero(split) + 1)
    # VTK's backward traces are stored seed-to-upstream. Orient their display
    # by the sampled velocity rather than assuming point order means flow.
    velocity = mesh.point_data.get("vector")
    alignment = None
    if velocity is not None and np.asarray(velocity).shape == (len(points), 3):
        velocity = np.asarray(velocity, dtype=float)
        alignment = np.einsum("ij,ij->i", b - a, velocity[starts] + velocity[ends])
    curves, curve_vectors = [], []
    if alignment is not None:
        velocity_a, velocity_b = velocity[starts], velocity[ends]
        vp = (velocity_a + lo[:, None] * (velocity_b - velocity_a))[selected]
        vq = (velocity_a + hi[:, None] * (velocity_b - velocity_a))[selected]
    for group in groups:
        curve = np.vstack((p[group], q[group[-1]]))
        vectors = np.vstack((vp[group], vq[group[-1]])) if alignment is not None else np.zeros((len(curve), 3))
        if alignment is not None and np.nanmean(alignment[selected[group]]) < 0.0:
            curve = curve[::-1].copy()
            vectors = vectors[::-1].copy()
        curves.append(curve)
        curve_vectors.append(vectors)
    return (curves, curve_vectors) if return_vectors else curves
