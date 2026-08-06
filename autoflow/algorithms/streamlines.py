import numpy as np
import pyvista as pv
from scipy.ndimage import map_coordinates

from .surfaces import (
    _build_branch_grid,
    create_uniform_grid,
    create_uniform_vector,
    extract_plane_cross_section,
)


_AUTOMATIC_CLIM_PERCENTILE = 99.0


def automatic_streamline_clim(flow_xyzt3, mask=None):
    """Return a robust, all-phase velocity range in m/s for streamline colors."""
    if flow_xyzt3 is None:
        return (0.0, 1.0)
    flow = np.asarray(flow_xyzt3, dtype=np.float32)
    if flow.ndim != 5 or flow.shape[-1] != 3:
        return (0.0, 1.0)

    vectors = None
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape == flow.shape[:4]:
            vectors = flow[mask_arr]
        elif mask_arr.shape == flow.shape[:3]:
            vectors = flow[mask_arr].reshape(-1, 3)
        elif mask_arr.ndim == 4 and mask_arr.shape[:3] == flow.shape[:3] and mask_arr.shape[3] == 1:
            vectors = flow[np.broadcast_to(mask_arr, flow.shape[:4])]
    if vectors is None:
        vectors = flow.reshape(-1, 3)
    vectors = np.asarray(vectors, dtype=np.float32).reshape(-1, 3)
    if vectors.size == 0:
        return (0.0, 1.0)
    vectors = vectors[np.all(np.isfinite(vectors), axis=1)]
    if vectors.size == 0:
        return (0.0, 1.0)

    speed_sq = np.einsum("ij,ij->i", vectors, vectors, optimize=True)
    speed_cm_s = np.sqrt(speed_sq)
    speed_limit = float(
        np.percentile(speed_cm_s, _AUTOMATIC_CLIM_PERCENTILE)
    ) / 100.0
    if speed_limit <= 0.0 and np.any(speed_cm_s > 0.0):
        speed_limit = float(np.max(speed_cm_s)) / 100.0
    if not np.isfinite(speed_limit) or speed_limit <= 0.0:
        speed_limit = 1.0
    return (0.0, max(speed_limit, 1e-6))


def _set_streamline_velocity_scalar(streamlines):
    """Color streamlines by the magnitude of the vector used by the tracer."""
    if streamlines is None or "vector" not in streamlines.point_data:
        return streamlines
    vectors = np.asarray(streamlines.point_data["vector"], dtype=np.float32)
    streamlines.point_data["Velocity"] = np.linalg.norm(
        vectors, axis=1
    ).astype(np.float32, copy=False)
    streamlines.set_active_scalars("Velocity")
    return streamlines


def generate_seed_points(mask_3d, spacing, origin, ratio=0.02, rng_seed=0,
                         min_seeds=50):
    idx = np.argwhere(np.asarray(mask_3d, dtype=bool))
    K = len(idx)
    if K == 0:
        return np.empty((0, 3), dtype=float)
    n = int(min(K, max(int(min_seeds), int(round(K * float(ratio))))))
    rng = np.random.default_rng(int(rng_seed))
    pick = idx[rng.choice(K, size=n, replace=False)]
    sp = np.asarray(spacing, dtype=float).reshape(1, 3)
    org = np.asarray(origin, dtype=float).reshape(1, 3)
    return (org + pick.astype(float) * sp).astype(float)


def _crop_vector_domain_to_mask(flow_t, mask_3d, spacing, origin, padding=1):
    mask = np.asarray(mask_3d, dtype=bool)
    occupied = np.where(mask)
    if not occupied[0].size:
        return np.asarray(flow_t), mask, np.asarray(origin, dtype=float).reshape(3)
    pad = max(int(padding), 0)
    lo = np.array([max(int(np.min(axis_values)) - pad, 0) for axis_values in occupied], dtype=int)
    hi = np.array([
        min(int(np.max(axis_values)) + pad + 1, int(mask.shape[axis]))
        for axis, axis_values in enumerate(occupied)
    ], dtype=int)
    spatial_slices = tuple(slice(int(lo[axis]), int(hi[axis])) for axis in range(3))
    cropped_origin = (
        np.asarray(origin, dtype=float).reshape(3)
        + lo.astype(float) * np.asarray(spacing, dtype=float).reshape(3)
    )
    return np.asarray(flow_t)[spatial_slices], mask[spatial_slices], cropped_origin


def generate_streamlines_at_t(flow_xyzt3, t, seeds, spacing, origin, mask_3d=None,
                              max_steps=2000, terminal_speed=0.01,
                              seed_ratio=0.02, min_seeds=50, rng_seed=0):
    if mask_3d is None:
        return None
    flow_t, mask_work, work_origin = _crop_vector_domain_to_mask(
        flow_xyzt3[..., int(t), :],
        mask_3d,
        spacing,
        origin,
    )
    flow_t = np.asarray(flow_t, dtype=np.float32)
    velocity = create_uniform_vector(
        flow_t[..., 0] / 100.0,
        flow_t[..., 1] / 100.0,
        flow_t[..., 2] / 100.0,
        spacing, origin=work_origin)
    mesh = create_uniform_grid(mask_work, spacing, origin=work_origin)
    mesh = mesh.threshold(0.1)
    if mesh.n_points == 0:
        return None
    volume = mesh.sample(velocity)
    volume.set_active_scalars('Velocity')
    if volume.points.shape[0] == 0:
        return None
    if seeds is None or len(seeds) == 0:
        seeds = generate_seed_points(
            mask_3d,
            spacing,
            origin,
            ratio=seed_ratio,
            rng_seed=int(rng_seed) + int(t),
            min_seeds=min_seeds,
        )
    source = np.asarray(seeds, dtype=float)
    if source.size == 0:
        return None
    sl = volume.streamlines_from_source(
        vectors='vector',
        source=source,
        integrator_type=4,
        max_steps=int(max_steps),
        terminal_speed=float(terminal_speed),
        compute_vorticity=False,
    )
    if sl is None or sl.n_points == 0:
        return None
    return _set_streamline_velocity_scalar(sl)


def _plane_seeds(mask_3d, plane, spacing, origin, seed_ratio, min_seeds,
                 rng_seed, branch_labels_3d=None, t=0):
    if mask_3d is None:
        return None
    mesh = create_uniform_grid(np.asarray(mask_3d) > 0, spacing, origin=origin)
    mesh = mesh.threshold(0.1)
    if mesh.n_points == 0:
        return None
    branch_grid = _build_branch_grid(branch_labels_3d, spacing, origin)
    target_label = None
    if branch_labels_3d is not None:
        target_label = int(getattr(plane, 'label', 0) or 0)
        if target_label <= 0:
            ijk = np.rint((np.asarray(plane.center, dtype=float) - np.asarray(origin, dtype=float).reshape(3)) / (np.asarray(spacing, dtype=float).reshape(3) + 1e-12)).astype(int)
            ijk = np.clip(ijk, 0, np.array(np.asarray(branch_labels_3d).shape) - 1)
            target_label = int(np.asarray(branch_labels_3d)[ijk[0], ijk[1], ijk[2]])
    pg = extract_plane_cross_section(mask_3d, plane, spacing, origin, branch_grid=branch_grid, target_label=target_label)
    if pg is None or pg.n_cells == 0:
        return None
    seeds = pg.cell_centers().points
    if len(seeds) == 0:
        return None
    n_seeds = int(max(int(min_seeds), int(np.ceil(len(seeds) * float(seed_ratio)))))
    n_seeds = max(1, min(int(n_seeds), len(seeds)))
    if len(seeds) > n_seeds:
        rng = np.random.default_rng(int(rng_seed) + 104729 * int(getattr(plane, 'path_index', 0)) + int(t))
        idx = rng.choice(len(seeds), size=n_seeds, replace=False)
        seeds = seeds[np.sort(idx)]
    return np.asarray(seeds, dtype=float)


def generate_streamlines_from_plane_at_t(flow_xyzt3, t, plane, spacing, origin,
                                         mask_3d=None, max_steps=2000,
                                         terminal_speed=0.01,
                                         seed_ratio=0.02, min_seeds=50,
                                         rng_seed=0,
                                         branch_labels_3d=None):
    flow_t = np.asarray(flow_xyzt3[..., int(t), :], dtype=np.float32)
    velocity = create_uniform_vector(
        flow_t[..., 0] / 100.0,
        flow_t[..., 1] / 100.0,
        flow_t[..., 2] / 100.0,
        spacing, origin=origin)
    if mask_3d is None:
        return None
    mesh = create_uniform_grid(np.asarray(mask_3d) > 0, spacing, origin=origin)
    mesh = mesh.threshold(0.1)
    if mesh.n_points == 0:
        return None
    volume = mesh.sample(velocity)
    volume.set_active_scalars('Velocity')
    if volume.points.shape[0] == 0:
        return None
    seeds = _plane_seeds(
        mask_3d,
        plane,
        spacing,
        origin,
        seed_ratio=seed_ratio,
        min_seeds=min_seeds,
        rng_seed=rng_seed,
        branch_labels_3d=branch_labels_3d,
        t=t,
    )
    if seeds is None or len(seeds) == 0:
        return None
    sl = volume.streamlines_from_source(
        vectors='vector',
        source=seeds,
        integrator_type=4,
        max_steps=int(max_steps),
        terminal_speed=float(terminal_speed),
        compute_vorticity=False,
    )
    if sl is None or sl.n_points == 0:
        return None
    return _set_streamline_velocity_scalar(sl)


def _world_to_index(points_xyz, spacing, origin):
    pts = np.asarray(points_xyz, dtype=float).reshape(-1, 3)
    sp = np.asarray(spacing, dtype=float).reshape(1, 3)
    org = np.asarray(origin, dtype=float).reshape(1, 3)
    return (pts - org) / (sp + 1e-12)


def _sample_scalar_xyz(volume_xyz, points_xyz, spacing, origin, order=1, cval=0.0):
    coords = _world_to_index(points_xyz, spacing, origin).T
    return map_coordinates(
        np.asarray(volume_xyz, dtype=np.float32),
        coords,
        order=int(order),
        mode='constant',
        cval=float(cval),
    )


def _sample_vector_xyz(vector_xyz3, points_xyz, spacing, origin):
    pts = np.asarray(points_xyz, dtype=float).reshape(-1, 3)
    field = np.asarray(vector_xyz3, dtype=np.float32)
    out = np.empty((pts.shape[0], 3), dtype=np.float32)
    for comp in range(3):
        out[:, comp] = _sample_scalar_xyz(field[..., comp], pts, spacing, origin, order=1, cval=0.0)
    return out


def _wrap_time_position(time_pos, n_time):
    if n_time <= 1:
        return 0.0, 0, 0
    tau = float(time_pos) % float(n_time)
    t0 = int(np.floor(tau)) % int(n_time)
    t1 = (t0 + 1) % int(n_time)
    return tau - np.floor(tau), t0, t1


def _sample_velocity_periodic(flow_xyzt3, points_xyz, time_pos, spacing, origin):
    flow = np.asarray(flow_xyzt3, dtype=np.float32)
    n_time = int(flow.shape[3]) if flow.ndim == 5 else 1
    alpha, t0, t1 = _wrap_time_position(time_pos, n_time)
    vel0_cm_s = _sample_vector_xyz(flow[..., t0, :], points_xyz, spacing, origin)
    if n_time <= 1 or alpha <= 1e-12:
        vel_cm_s = vel0_cm_s
    else:
        vel1_cm_s = _sample_vector_xyz(flow[..., t1, :], points_xyz, spacing, origin)
        vel_cm_s = (1.0 - float(alpha)) * vel0_cm_s + float(alpha) * vel1_cm_s
    speed_m_s = np.linalg.norm(vel_cm_s, axis=1) / 100.0
    vel_mm_s = vel_cm_s * 10.0
    return vel_mm_s.astype(np.float32), speed_m_s.astype(np.float32)


def _sample_mask_periodic(mask_4d, points_xyz, time_pos, spacing, origin):
    pts = np.asarray(points_xyz, dtype=float).reshape(-1, 3)
    if mask_4d is None:
        return np.ones(pts.shape[0], dtype=bool)
    mask_arr = np.asarray(mask_4d, dtype=np.float32)
    if mask_arr.ndim == 3:
        values = _sample_scalar_xyz(mask_arr, pts, spacing, origin, order=1, cval=0.0)
        return values > 0.5
    n_time = int(mask_arr.shape[3])
    if n_time <= 1:
        values = _sample_scalar_xyz(mask_arr[..., 0], pts, spacing, origin, order=1, cval=0.0)
        return values > 0.5
    _, t0, _ = _wrap_time_position(time_pos, n_time)
    values = _sample_scalar_xyz(mask_arr[..., t0], pts, spacing, origin, order=1, cval=0.0)
    return values > 0.5


def _polydata_from_tracks(tracks, speeds):
    keep = []
    for track, speed in zip(tracks, speeds):
        track_arr = np.asarray(track, dtype=float).reshape(-1, 3)
        speed_arr = np.asarray(speed, dtype=np.float32).reshape(-1)
        if track_arr.shape[0] >= 2 and speed_arr.shape[0] >= 2:
            keep.append((track_arr, speed_arr[:track_arr.shape[0]]))
    if not keep:
        return None
    points = []
    lines = []
    scalars = []
    offset = 0
    for track_arr, speed_arr in keep:
        n_pts = int(track_arr.shape[0])
        points.append(track_arr)
        lines.append(np.concatenate(([n_pts], np.arange(offset, offset + n_pts, dtype=np.int64))))
        scalars.append(speed_arr)
        offset += n_pts
    poly = pv.PolyData(np.vstack(points))
    poly.lines = np.concatenate(lines).astype(np.int64)
    poly.point_data['Velocity'] = np.concatenate(scalars).astype(np.float32)
    return poly


def generate_pathlines_from_plane_at_t(flow_xyzt3, t, plane, spacing, origin,
                                       mask_4d=None, mask_3d=None, max_steps=2000,
                                       terminal_speed=0.01,
                                       seed_ratio=0.02, min_seeds=50,
                                       rng_seed=0, rr=1000.0,
                                       branch_labels_3d=None):
    flow = np.asarray(flow_xyzt3, dtype=np.float32)
    if flow.ndim != 5 or flow.shape[-1] != 3:
        raise ValueError(f'flow must be XYZT3, got {flow.shape}')
    n_time = int(flow.shape[3])
    if mask_3d is None and mask_4d is not None:
        mask_arr = np.asarray(mask_4d)
        if mask_arr.ndim == 4:
            mask_3d = mask_arr[..., min(max(0, int(t)), mask_arr.shape[3] - 1)]
        else:
            mask_3d = mask_arr
    seeds = _plane_seeds(
        mask_3d,
        plane,
        spacing,
        origin,
        seed_ratio=seed_ratio,
        min_seeds=min_seeds,
        rng_seed=rng_seed,
        branch_labels_3d=branch_labels_3d,
        t=t,
    )
    if seeds is None or len(seeds) == 0:
        return None
    steps = max(1, int(max_steps))
    rr_s = max(float(rr) / 1000.0, 1e-6)
    dt_s = rr_s / float(steps)
    dt_t = float(n_time) / float(steps) if n_time > 0 else 0.0
    start_t = float(int(t) % max(n_time, 1))
    mask_src = mask_4d if mask_4d is not None else mask_3d

    seed_count = int(seeds.shape[0])
    tracks = [[seeds[i].astype(float)] for i in range(seed_count)]
    initial_speed = _sample_velocity_periodic(flow, seeds, start_t, spacing, origin)[1]
    speeds = [[float(initial_speed[i])] for i in range(seed_count)]
    positions = np.asarray(seeds, dtype=float).copy()
    active = _sample_mask_periodic(mask_src, positions, start_t, spacing, origin)
    active &= initial_speed >= float(terminal_speed)

    for step_idx in range(steps):
        if not np.any(active):
            break
        tau = start_t + float(step_idx) * dt_t
        active_idx = np.flatnonzero(active)
        pos_active = positions[active_idx]
        vel1_mm_s, speed1_m_s = _sample_velocity_periodic(flow, pos_active, tau, spacing, origin)
        keep = speed1_m_s >= float(terminal_speed)
        if not np.any(keep):
            active[active_idx] = False
            continue
        idx_keep = active_idx[keep]
        pos_keep = pos_active[keep]
        vel_keep = vel1_mm_s[keep]
        mid_pos = pos_keep + 0.5 * dt_s * vel_keep
        tau_mid = tau + 0.5 * dt_t
        inside_mid = _sample_mask_periodic(mask_src, mid_pos, tau_mid, spacing, origin)
        if not np.any(inside_mid):
            active[idx_keep] = False
            continue
        idx_mid = idx_keep[inside_mid]
        pos_mid = mid_pos[inside_mid]
        pos_start = pos_keep[inside_mid]
        vel2_mm_s, speed2_m_s = _sample_velocity_periodic(flow, pos_mid, tau_mid, spacing, origin)
        new_pos = pos_start + dt_s * vel2_mm_s
        tau_next = tau + dt_t
        inside_next = _sample_mask_periodic(mask_src, new_pos, tau_next, spacing, origin)
        keep_next = inside_next & (speed2_m_s >= float(terminal_speed))
        active[idx_keep] = False
        if not np.any(keep_next):
            continue
        idx_next = idx_mid[keep_next]
        new_pos = new_pos[keep_next]
        speed_next = speed2_m_s[keep_next]
        positions[idx_next] = new_pos
        active[idx_next] = True
        for local_i, seed_i in enumerate(idx_next.tolist()):
            tracks[seed_i].append(new_pos[local_i].astype(float))
            speeds[seed_i].append(float(speed_next[local_i]))

    return _polydata_from_tracks(tracks, speeds)
