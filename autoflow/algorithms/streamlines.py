import numpy as np

from .surfaces import (
    _build_branch_grid,
    create_uniform_grid,
    create_uniform_vector,
    extract_plane_cross_section,
)


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


def generate_streamlines_at_t(flow_xyzt3, t, seeds, spacing, origin, mask_3d=None,
                              max_steps=2000, terminal_speed=0.01,
                              seed_ratio=0.02, min_seeds=50, rng_seed=0):
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
    volume.set_active_scalars("Velocity")
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
        vectors="vector",
        source=source,
        integrator_type=4,
        max_steps=int(max_steps),
        terminal_speed=float(terminal_speed),
        compute_vorticity=False,
    )
    if sl is None or sl.n_points == 0:
        return None
    return sl


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
    volume.set_active_scalars("Velocity")
    if volume.points.shape[0] == 0:
        return None
    branch_grid = _build_branch_grid(branch_labels_3d, spacing, origin)
    target_label = None
    if branch_labels_3d is not None:
        target_label = int(getattr(plane, "label", 0) or 0)
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
        rng = np.random.default_rng(int(rng_seed) + 104729 * int(getattr(plane, "path_index", 0)) + int(t))
        idx = rng.choice(len(seeds), size=n_seeds, replace=False)
        seeds = seeds[np.sort(idx)]
    sl = volume.streamlines_from_source(
        vectors="vector",
        source=seeds,
        integrator_type=4,
        max_steps=int(max_steps),
        terminal_speed=float(terminal_speed),
        compute_vorticity=False,
    )
    if sl is None or sl.n_points == 0:
        return None
    return sl
