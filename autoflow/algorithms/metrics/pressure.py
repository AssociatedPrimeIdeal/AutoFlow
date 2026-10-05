"""Pressure gradients, relative-pressure reconstruction and centreline profiles."""

import numpy as np
from ...task_control import report_progress
from scipy.ndimage import binary_erosion, gaussian_filter, generate_binary_structure, label
from scipy.sparse import bmat, csc_matrix, csr_matrix, diags
from scipy.sparse.linalg import LinearOperator, cg, factorized, minres, spsolve

try:
    import pyamg
except ImportError:  # Keep source checkouts usable before dependencies are refreshed.
    pyamg = None

from ._common import _ensure_mask4d


def _periodic_central_difference(values, dt_s):
    """Central temporal derivative with the cardiac cycle treated as periodic."""
    arr = np.asarray(values)
    if arr.ndim < 2 or arr.shape[-2] <= 1:
        return np.zeros_like(arr, dtype=np.result_type(arr, np.float32))
    dt = max(float(dt_s), 1e-12)
    return (np.roll(arr, -1, axis=-2) - np.roll(arr, 1, axis=-2)) / (2.0 * dt)


def _finite_percentile_abs(values, q, default=0.0):
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float(default)
    return float(np.percentile(np.abs(finite), float(q)))


def _normalize_pressure_method(method):
    token = str(method or "ppe").strip().lower()
    if token in {
        "ls", "least_squares", "least-squares", "least squares",
        "ppe", "poisson", "poisson_pressure_equation",
    }:
        return "ppe"
    if token in {"ste", "stokes", "stokes_estimator", "stokes-estimator"}:
        return "ste"
    raise ValueError(f"unsupported pressure reconstruction method: {method}")


def _neighbor_shifts():
    return (
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    )


def _solve_reconstruction_system(system, rhs, *, tol=1e-5, max_iter=2000):
    matrix = system["matrix"]
    n = int(matrix.shape[0])
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    rhs = np.asarray(rhs, dtype=np.float64).reshape(n).copy()
    anchors = np.asarray(system.get("anchor_indices", [system.get("anchor_index", 0)]), dtype=int)
    rhs[anchors] = 0.0
    if n == 1:
        return np.zeros(1, dtype=np.float32)

    if "preconditioner" not in system:
        preconditioner = None
        if pyamg is not None and n >= 5000:
            try:
                hierarchy = pyamg.smoothed_aggregation_solver(matrix, max_coarse=50)
                system["amg_hierarchy"] = hierarchy
                preconditioner = hierarchy.aspreconditioner(cycle="V")
                system["preconditioner_kind"] = "amg"
            except Exception:
                preconditioner = None
        if preconditioner is None:
            diag = np.asarray(matrix.diagonal(), dtype=np.float64)
            inv_diag = np.zeros_like(diag)
            nz = np.abs(diag) > 1e-12
            inv_diag[nz] = 1.0 / diag[nz]
            preconditioner = diags(inv_diag, 0, format="csr")
            system["preconditioner_kind"] = "jacobi"
        system["preconditioner"] = preconditioner

    x0 = system.get("last_solution")
    sol, info = cg(
        matrix,
        rhs,
        x0=x0,
        M=system["preconditioner"],
        maxiter=int(max_iter),
        rtol=float(tol),
        atol=0.0,
    )
    if info != 0:
        try:
            solver = system.get("factorized_solver")
            if solver is None:
                solver = factorized(csc_matrix(matrix))
                system["factorized_solver"] = solver
            sol = np.asarray(solver(rhs), dtype=np.float64).reshape(n)
        except Exception:
            raise RuntimeError(f"sparse pressure solve failed to converge: info={info}")
    else:
        sol = np.asarray(sol, dtype=np.float64).reshape(n)
    system["last_solution"] = sol

    sol[anchors] = 0.0
    return sol.astype(np.float32)


def _build_pressure_reconstruction_system(mask_t, spacing_m):
    coords = np.argwhere(mask_t)
    n = int(len(coords))
    if n == 0:
        return {
            "coords": coords,
            "index_map": -np.ones(mask_t.shape, dtype=np.int32),
            "matrix": csr_matrix((0, 0), dtype=np.float64),
            "edge_pairs": np.zeros((0, 2), dtype=np.int32),
            "edge_axes": np.zeros(0, dtype=np.int8),
            "edge_scales": np.zeros(0, dtype=np.float64),
            "anchor_index": 0,
        }

    index_map = -np.ones(mask_t.shape, dtype=np.int32)
    index_map[mask_t] = np.arange(n, dtype=np.int32)
    edge_pairs = []
    edge_axes = []
    edge_scales = []
    dx, dy, dz = [float(v) for v in spacing_m]
    for axis, step in enumerate((dx, dy, dz)):
        left = [slice(None), slice(None), slice(None)]
        right = [slice(None), slice(None), slice(None)]
        left[axis] = slice(0, -1)
        right[axis] = slice(1, None)
        left = tuple(left)
        right = tuple(right)
        adjacent = mask_t[left] & mask_t[right]
        src = np.asarray(index_map[left][adjacent], dtype=np.int32)
        dst = np.asarray(index_map[right][adjacent], dtype=np.int32)
        if src.size:
            edge_pairs.append(np.column_stack((src, dst)))
            edge_axes.append(np.full(src.size, axis, dtype=np.int8))
            edge_scales.append(np.full(src.size, 1.0 / step, dtype=np.float64))

    pairs = np.vstack(edge_pairs).astype(np.int32, copy=False) if edge_pairs else np.zeros((0, 2), dtype=np.int32)
    axes = np.concatenate(edge_axes) if edge_axes else np.zeros(0, dtype=np.int8)
    scales = np.concatenate(edge_scales) if edge_scales else np.zeros(0, dtype=np.float64)
    rows = np.repeat(np.arange(len(pairs), dtype=np.int32), 2)
    cols = pairs.reshape(-1)
    data = np.column_stack((-scales, scales)).reshape(-1)
    incidence = csr_matrix((data, (rows, cols)), shape=(len(pairs), n), dtype=np.float64)
    matrix = (incidence.T @ incidence).tocsr()
    # Fix one pressure gauge in EACH connected component. Eliminate both rows
    # and columns to retain the symmetric positive-definite system needed by CG.
    components, count = label(mask_t, structure=generate_binary_structure(3, 1))
    component_ids = components[mask_t] - 1
    anchors = np.full(count, n, dtype=np.int32)
    np.minimum.at(anchors, component_ids, np.arange(n, dtype=np.int32))
    keep = np.ones(n, dtype=np.float64)
    keep[anchors] = 0.0
    gauge_scale = max(float(np.max(matrix.diagonal())), 1.0)
    selector = diags(keep, format="csr")
    matrix = selector @ matrix @ selector + diags((1.0 - keep) * gauge_scale, format="csr")
    return {
        "coords": coords,
        "index_map": index_map,
        "matrix": matrix.tocsr(),
        "rhs_operator": incidence.T.tocsr(),
        "edge_pairs": pairs,
        "edge_axes": axes,
        "edge_scales": scales,
        "anchor_index": 0,
        "anchor_indices": anchors,
    }


def _pressure_reconstruction_rhs(grad_t, system):
    edge_pairs = system["edge_pairs"]
    if edge_pairs.size == 0:
        return np.zeros(0, dtype=np.float64)
    edge_axes = system["edge_axes"]
    coords = system["coords"]
    src = coords[edge_pairs[:, 0]]
    dst = coords[edge_pairs[:, 1]]
    # D already contains 1/h. Its target is the face-averaged gradient in Pa/m,
    # with no additional scale or minus sign. Averaging integrates linear PG
    # exactly and avoids a one-sided pressure bias for varying gradients.
    return 0.5 * (np.asarray(grad_t[src[:, 0], src[:, 1], src[:, 2], edge_axes], dtype=np.float64)
                  + np.asarray(grad_t[dst[:, 0], dst[:, 1], dst[:, 2], edge_axes], dtype=np.float64))


def _build_least_squares_system(mask_t, spacing_m):
    """Backward-compatible name for the merged Cartesian PPE system."""
    return _build_pressure_reconstruction_system(mask_t, spacing_m)


def _build_ppe_system(mask_t, spacing_m):
    """Backward-compatible name for the merged Cartesian PPE system."""
    system = _build_pressure_reconstruction_system(mask_t, spacing_m)
    system.pop("rhs_operator", None)
    return system


def _least_squares_rhs(grad_t, system):
    """Backward-compatible name for the merged pressure RHS."""
    return _pressure_reconstruction_rhs(grad_t, system)


def _ppe_rhs(grad_t, mask_t, spacing_m, system):
    """Backward-compatible name for the merged pressure RHS."""
    rhs = np.zeros(system["matrix"].shape[0], dtype=np.float64)
    pairs = system["edge_pairs"]
    if pairs.size:
        flux = _pressure_reconstruction_rhs(grad_t, system) * system["edge_scales"]
        np.add.at(rhs, pairs[:, 0], -flux)
        np.add.at(rhs, pairs[:, 1], flux)
    return rhs


def _build_ste_mac_system(mask_t, spacing_m):
    """Build a staggered-grid Stokes estimator system.

    The pressure is cell-centred and the auxiliary velocity is face-centred.
    Interior faces carry unknown velocities; faces on the support boundary
    are zero (no-slip). This MAC layout avoids the checkerboard pressure
    modes that occur when all variables are collocated on voxel centres.
    """
    mask_t = np.asarray(mask_t, dtype=bool)
    coords = np.argwhere(mask_t)
    n_cells = int(len(coords))
    if n_cells == 0:
        return {
            "coords": coords,
            "matrix": csr_matrix((0, 0), dtype=np.float64),
            "face_count": 0,
            "pressure_indices": np.zeros(0, dtype=np.int32),
            "pressure_count": 0,
            "fallback": True,
        }

    spacing_m = tuple(float(value) for value in spacing_m)
    cell_index = -np.ones(mask_t.shape, dtype=np.int32)
    cell_index[mask_t] = np.arange(n_cells, dtype=np.int32)
    face_maps = []
    face_low = []
    face_high = []
    face_axis = []
    face_offsets = []
    face_count = 0
    for axis in range(3):
        low_slices = [slice(None), slice(None), slice(None)]
        high_slices = [slice(None), slice(None), slice(None)]
        low_slices[axis] = slice(0, -1)
        high_slices[axis] = slice(1, None)
        low_slices = tuple(low_slices)
        high_slices = tuple(high_slices)
        adjacent = mask_t[low_slices] & mask_t[high_slices]
        low_coords = np.argwhere(adjacent)
        local_count = int(len(low_coords))
        face_map = -np.ones(mask_t.shape, dtype=np.int32)
        if local_count:
            face_map[tuple(low_coords.T)] = np.arange(local_count, dtype=np.int32)
            high_coords = low_coords.copy()
            high_coords[:, axis] += 1
            face_low.append(low_coords)
            face_high.append(high_coords)
            face_axis.append(np.full(local_count, axis, dtype=np.int8))
        else:
            face_low.append(np.zeros((0, 3), dtype=np.int32))
            face_high.append(np.zeros((0, 3), dtype=np.int32))
            face_axis.append(np.zeros(0, dtype=np.int8))
        face_maps.append(face_map)
        face_offsets.append(face_count)
        face_count += local_count

    if face_count == 0:
        return {
            "coords": coords,
            "matrix": csr_matrix((0, 0), dtype=np.float64),
            "face_count": 0,
            "pressure_indices": np.zeros(0, dtype=np.int32),
            "pressure_count": 0,
            "fallback": True,
        }

    all_low = np.concatenate(face_low, axis=0)
    all_high = np.concatenate(face_high, axis=0)
    all_axis = np.concatenate(face_axis, axis=0)
    matrix_rows = []
    matrix_cols = []
    matrix_data = []
    diagonal = 2.0 * sum(1.0 / (step * step) for step in spacing_m)
    for axis in range(3):
        low_coords = face_low[axis]
        offset = int(face_offsets[axis])
        face_map = face_maps[axis]
        for local_id, low_coord in enumerate(low_coords):
            row = offset + int(local_id)
            matrix_rows.append(row)
            matrix_cols.append(row)
            matrix_data.append(diagonal)
            for direction, step in enumerate(spacing_m):
                for sign in (-1, 1):
                    neighbour = low_coord.copy()
                    neighbour[direction] += sign
                    if np.any(neighbour < 0) or np.any(neighbour >= np.asarray(mask_t.shape)):
                        continue
                    neighbour_id = int(face_map[tuple(neighbour)])
                    if neighbour_id >= 0:
                        matrix_rows.append(row)
                        matrix_cols.append(offset + neighbour_id)
                        matrix_data.append(-1.0 / (step * step))
    velocity_laplacian = csr_matrix(
        (matrix_data, (matrix_rows, matrix_cols)),
        shape=(face_count, face_count),
        dtype=np.float64,
    )

    div_rows = []
    div_cols = []
    div_data = []
    for face_id, (low_coord, high_coord, axis) in enumerate(zip(all_low, all_high, all_axis)):
        scale = 1.0 / spacing_m[int(axis)]
        low_id = int(cell_index[tuple(low_coord)])
        high_id = int(cell_index[tuple(high_coord)])
        div_rows.extend((low_id, high_id))
        div_cols.extend((face_id, face_id))
        div_data.extend((scale, -scale))
    divergence = csr_matrix(
        (div_data, (div_rows, div_cols)),
        shape=(n_cells, face_count),
        dtype=np.float64,
    )

    components, count = label(mask_t, structure=generate_binary_structure(3, 1))
    component_ids = components[mask_t] - 1
    anchors = np.full(count, n_cells, dtype=np.int32)
    np.minimum.at(anchors, component_ids, np.arange(n_cells, dtype=np.int32))
    pressure_keep = np.ones(n_cells, dtype=bool)
    pressure_keep[anchors] = False
    reduced_divergence = divergence[pressure_keep, :].tocsr()
    matrix = bmat(
        [[velocity_laplacian, reduced_divergence.T],
         [reduced_divergence, None]],
        format="csr",
    )
    diagonal_values = np.abs(np.asarray(matrix.diagonal(), dtype=np.float64))
    diagonal_values[diagonal_values < 1e-8] = 1.0
    return {
        "coords": coords,
        "matrix": matrix,
        "preconditioner": diags(1.0 / diagonal_values, format="csr"),
        "face_count": face_count,
        "pressure_indices": np.flatnonzero(pressure_keep).astype(np.int32),
        "pressure_count": int(np.sum(pressure_keep)),
        "anchors": anchors,
        "face_low": all_low,
        "face_high": all_high,
        "face_axis": all_axis,
        "fallback": False,
    }


def _solve_ste_mac_pressure(grad_t, system):
    if system.get("fallback", False):
        return np.zeros(len(system.get("coords", [])), dtype=np.float32)
    face_low = system["face_low"]
    face_high = system["face_high"]
    face_axis = system["face_axis"]
    rhs = np.zeros(system["matrix"].shape[0], dtype=np.float64)
    if len(face_low):
        low_values = np.asarray(grad_t[tuple(face_low.T)], dtype=np.float64)
        high_values = np.asarray(grad_t[tuple(face_high.T)], dtype=np.float64)
        rhs[: system["face_count"]] = 0.5 * (
            low_values[np.arange(len(face_axis)), face_axis]
            + high_values[np.arange(len(face_axis)), face_axis]
        )

    matrix = system["matrix"]
    solution = None
    if matrix.shape[0] <= 12000:
        try:
            candidate = np.asarray(spsolve(matrix.tocsc(), rhs), dtype=np.float64)
            if np.all(np.isfinite(candidate)):
                solution = candidate
        except Exception:
            solution = None
    if solution is None:
        preconditioner = system.get("iterative_preconditioner")
        if preconditioner is None:
            preconditioner = system["preconditioner"]
            if pyamg is not None and system["face_count"] > 0:
                try:
                    velocity_matrix = matrix[: system["face_count"], : system["face_count"]].tocsr()
                    hierarchy = pyamg.smoothed_aggregation_solver(velocity_matrix, max_coarse=50)

                    def apply_preconditioner(vector):
                        result = np.zeros_like(vector, dtype=np.float64)
                        result[: system["face_count"]] = hierarchy.solve(
                            vector[: system["face_count"]],
                            tol=1e-6,
                            maxiter=5,
                            cycle="V",
                        )
                        result[system["face_count"]:] = vector[system["face_count"]:]
                        return result

                    preconditioner = LinearOperator(
                        matrix.shape,
                        matvec=apply_preconditioner,
                        dtype=np.float64,
                    )
                except Exception:
                    preconditioner = system["preconditioner"]
            system["iterative_preconditioner"] = preconditioner
        solution, info = minres(
            matrix,
            rhs,
            M=preconditioner,
            maxiter=10000,
            rtol=1e-8,
        )
        solution = np.asarray(solution, dtype=np.float64)
        residual = np.linalg.norm(matrix @ solution - rhs) / max(np.linalg.norm(rhs), 1.0)
        if info != 0 or not np.all(np.isfinite(solution)) or not np.isfinite(residual) or residual > 1e-4:
            raise RuntimeError(f"Stokes pressure solve failed to converge: info={info}, residual={residual:.3g}")

    pressure = np.zeros(len(system["coords"]), dtype=np.float32)
    pressure[system["pressure_indices"]] = -solution[system["face_count"]:].astype(np.float32)
    return pressure


def reconstruct_relative_pressure_map(pressure_gradient_array, support_mask, spacing, *, method="ppe"):
    method = _normalize_pressure_method(method)
    grad = np.asarray(pressure_gradient_array, dtype=np.float32)
    if grad.ndim != 5 or grad.shape[-1] != 3:
        raise ValueError(f"pressure_gradient_array must be XYZTV, got {grad.shape}")
    support = np.asarray(support_mask, dtype=bool)
    if support.shape != grad.shape[:4]:
        raise ValueError(f"support_mask shape {support.shape} does not match {grad.shape[:4]}")

    spacing_m = np.asarray(spacing, dtype=float).reshape(3) / 1000.0
    if not np.all(np.isfinite(spacing_m)) or np.any(spacing_m <= 0):
        raise ValueError("Pressure spacing must be finite and positive (mm)")
    if not np.all(np.isfinite(grad[support])):
        raise ValueError("Pressure gradients must be finite inside the support mask")
    dx, dy, dz = spacing_m.tolist()
    nt = grad.shape[3]
    pressure = np.zeros(grad.shape[:4], dtype=np.float32)

    system_cache = {}
    for tidx in range(nt):
        report_progress({"stage": "relative_pressure", "current": tidx, "total": nt,
                         "message": f"Reconstructing relative pressure phase {tidx + 1}/{nt}"})
        mask_t = support[..., tidx]
        if not np.any(mask_t):
            continue
        cache_key = mask_t.tobytes()
        if method == "ste":
            system = system_cache.get(cache_key)
            if system is None:
                system = _build_ste_mac_system(mask_t, (dx, dy, dz))
                system_cache[cache_key] = system
            sol = _solve_ste_mac_pressure(grad[..., tidx, :], system)
        else:
            system = system_cache.get(cache_key)
            if system is None:
                system = _build_pressure_reconstruction_system(mask_t, (dx, dy, dz))
                system_cache[cache_key] = system
            rhs_rows = _pressure_reconstruction_rhs(grad[..., tidx, :], system)
            rhs = system["rhs_operator"] @ rhs_rows
            sol = _solve_reconstruction_system(system, rhs)

        pressure_t = np.zeros(mask_t.shape, dtype=np.float32)
        coords = system["coords"]
        if len(coords):
            pressure_t[coords[:, 0], coords[:, 1], coords[:, 2]] = sol
        pressure[..., tidx] = pressure_t

    peak = np.max(np.abs(pressure), axis=3).astype(np.float32) if pressure.shape[3] > 0 else np.zeros(pressure.shape[:3], dtype=np.float32)
    finite = pressure[support & np.isfinite(pressure)]
    upper = _finite_percentile_abs(finite, 99.0, default=1.0)
    upper = upper if upper > 0.0 else 1.0
    return {
        "relative_pressure_array": pressure,
        "relative_pressure_peak": peak,
        "relative_pressure_display_clim": (-upper, upper),
        "pressure_method": method,
    }


def _points_as_voxels(points_xyz, spacing, origin, shape, coordinate_space="world"):
    pts = np.asarray(points_xyz, dtype=float).reshape(-1, 3)
    spacing = np.asarray(spacing, dtype=float).reshape(3)
    origin = np.asarray(origin, dtype=float).reshape(3)
    shape = np.asarray(shape, dtype=int).reshape(3)
    if len(pts) == 0:
        return pts, False
    if str(coordinate_space).lower() == "local":
        return pts / (spacing.reshape(1, 3) + 1e-12), True
    if str(coordinate_space).lower() == "voxel":
        return pts, True
    looks_like_voxels = (
        np.all(np.isfinite(pts))
        and np.all(pts >= -0.5)
        and np.all(pts <= (shape.reshape(1, 3) - 0.5))
    )
    if looks_like_voxels:
        return pts, True
    vox = (pts - origin.reshape(1, 3)) / (spacing.reshape(1, 3) + 1e-12)
    return vox, False


def _sample_volume_at_points(volume_xyz, points_xyz, spacing, origin, coordinate_space="world"):
    vol = np.asarray(volume_xyz, dtype=np.float32)
    pts = np.asarray(points_xyz, dtype=float).reshape(-1, 3)
    if vol.ndim != 3 or len(pts) == 0:
        return np.zeros(len(pts), dtype=np.float32)
    spacing = np.asarray(spacing, dtype=float).reshape(3)
    origin = np.asarray(origin, dtype=float).reshape(3)
    shape = np.array(vol.shape, dtype=int)
    vox, _ = _points_as_voxels(pts, spacing, origin, shape, coordinate_space=coordinate_space)
    out = np.zeros(len(pts), dtype=np.float32)
    for idx, coord in enumerate(vox):
        if np.any(coord < 0.0) or np.any(coord > (shape - 1)):
            continue
        base = np.floor(coord).astype(int)
        frac = coord - base
        upper = np.minimum(base + 1, shape - 1)
        x0, y0, z0 = base.tolist()
        x1, y1, z1 = upper.tolist()
        c000 = float(vol[x0, y0, z0])
        c100 = float(vol[x1, y0, z0])
        c010 = float(vol[x0, y1, z0])
        c110 = float(vol[x1, y1, z0])
        c001 = float(vol[x0, y0, z1])
        c101 = float(vol[x1, y0, z1])
        c011 = float(vol[x0, y1, z1])
        c111 = float(vol[x1, y1, z1])
        xd, yd, zd = frac.tolist()
        c00 = c000 * (1.0 - xd) + c100 * xd
        c10 = c010 * (1.0 - xd) + c110 * xd
        c01 = c001 * (1.0 - xd) + c101 * xd
        c11 = c011 * (1.0 - xd) + c111 * xd
        c0 = c00 * (1.0 - yd) + c10 * yd
        c1 = c01 * (1.0 - yd) + c11 * yd
        out[idx] = c0 * (1.0 - zd) + c1 * zd
    return out


def compute_centerline_pressure_profiles(relative_pressure_array, centerline_paths, spacing, origin, *, support_mask=None):
    pressure = np.asarray(relative_pressure_array, dtype=np.float32)
    if pressure.ndim != 4:
        raise ValueError(f"relative_pressure_array must be XYZT, got {pressure.shape}")
    spacing = np.asarray(spacing, dtype=float).reshape(3)
    origin = np.asarray(origin, dtype=float).reshape(3)
    if support_mask is None:
        support = np.isfinite(pressure)
    else:
        support = np.asarray(support_mask, dtype=bool)
        if support.shape != pressure.shape:
            raise ValueError('Centerline pressure support must match the pressure array')
        support = support & np.isfinite(pressure)
    profiles = []
    nt = pressure.shape[3]
    for path_idx, path in enumerate(list(centerline_paths or [])):
        pts = np.asarray(path, dtype=float).reshape(-1, 3)
        if len(pts) == 0:
            profiles.append({
                "path_index": int(path_idx),
                "distances_mm": [],
                "relative_pressure_Pa_t": [],
                "pressure_drop_Pa_t": [],
                "pressure_drop_mean_Pa": None,
                "pressure_drop_peak_Pa": None,
                "valid_sample_t": [],
                "pressure_drop_valid_t": [],
            })
            continue
        # Centerline paths are stored in local physical millimetres.  Convert
        # explicitly instead of guessing from coordinate magnitudes.
        pts_vox, _ = _points_as_voxels(
            pts, spacing, origin, pressure.shape[:3], coordinate_space="local"
        )
        if len(pts_vox) > 1:
            diffs = np.diff(pts_vox, axis=0)
            seg = np.linalg.norm(diffs * spacing.reshape(1, 3), axis=1)
        else:
            seg = np.zeros(0, dtype=float)
        dist = np.concatenate([[0.0], np.cumsum(seg)]) if len(pts) > 0 else np.zeros(0, dtype=float)
        sample_pts = pts_vox
        samples_t = []
        drop_t = []
        valid_sample_t = []
        drop_valid_t = []
        in_grid = np.all(np.isfinite(sample_pts) & (sample_pts >= 0)
                         & (sample_pts <= np.array(pressure.shape[:3]) - 1), axis=1)
        for tidx in range(nt):
            vals = _sample_volume_at_points(
                pressure[..., tidx], sample_pts, spacing, origin, coordinate_space="voxel"
            )
            weights = _sample_volume_at_points(
                support[..., tidx].astype(np.float32), sample_pts, spacing, origin, coordinate_space='voxel')
            valid = in_grid & (weights >= 1.0 - 1e-6) & np.isfinite(vals)
            components, _ = label(support[..., tidx], structure=generate_binary_structure(3, 1))
            rounded = np.clip(np.rint(np.nan_to_num(sample_pts, nan=0.0, posinf=0.0, neginf=0.0)).astype(int), 0, np.array(pressure.shape[:3]) - 1)
            ids = components[tuple(rounded.T)]
            # Gauges differ between disconnected components. A zero-filled
            # unsupported endpoint is not a measured zero-pressure sample.
            drop_valid = bool(len(vals) > 1 and valid[0] and valid[-1]
                              and ids[0] > 0 and ids[0] == ids[-1])
            samples_t.append([float(v) if ok else None for v, ok in zip(vals, valid)])
            valid_sample_t.append(valid.tolist())
            drop_valid_t.append(drop_valid)
            drop_t.append(float(vals[0] - vals[-1]) if drop_valid else None)
        profiles.append({
            "path_index": int(path_idx),
            "distances_mm": dist.astype(np.float32).tolist(),
            "relative_pressure_Pa_t": samples_t,
            "pressure_drop_Pa_t": drop_t,
            "pressure_drop_mean_Pa": float(np.mean([v for v in drop_t if v is not None])) if any(drop_valid_t) else None,
            "pressure_drop_peak_Pa": float(np.max(np.abs([v for v in drop_t if v is not None]))) if any(drop_valid_t) else None,
            "valid_sample_t": valid_sample_t,
            "pressure_drop_valid_t": drop_valid_t,
        })
    return profiles


def compute_pressure_gradient_metrics(mask4d, flow, spacing, rr, rho=1060.0, viscosity=4.0,
                                      smoothing_sigma=0.0, support_erosion_iters=1, use_convective_acceleration=True,
                                      pressure_method="ppe", centerline_paths=None,
                                      origin=(0, 0, 0)):
    mask4d = _ensure_mask4d(mask4d)
    flow = np.asarray(flow, dtype=np.float32)
    if flow.ndim != 5 or flow.shape[-1] != 3:
        raise ValueError(f"flow must be XYZTV, got {flow.shape}")
    if flow.shape[:4] != mask4d.shape:
        raise ValueError(f"flow shape {flow.shape[:4]} does not match mask {mask4d.shape}")

    spacing_mm = np.asarray(spacing, dtype=float).reshape(3)
    if not np.all(np.isfinite(spacing_mm)) or np.any(spacing_mm <= 0):
        raise ValueError("Pressure spacing must be finite and positive (mm)")
    if flow.shape[3] == 0 or not np.isfinite(rr) or float(rr) <= 0:
        raise ValueError("Pressure analysis requires at least one phase and a positive RR (ms)")
    spacing_m = spacing_mm / 1000.0
    dt_s = float(rr) / 1000.0 / float(flow.shape[3])
    rho = float(rho)
    mu_pa_s = float(viscosity) / 1000.0
    if not np.isfinite(rho) or rho <= 0 or not np.isfinite(mu_pa_s) or mu_pa_s < 0:
        raise ValueError("Pressure density must be positive and viscosity nonnegative and finite")
    if not np.isfinite(smoothing_sigma):
        raise ValueError("Pressure smoothing sigma must be finite")
    sigma = max(float(smoothing_sigma), 0.0)

    # Spatial derivatives only need the vessel extent plus one stencil voxel.
    # Keep the public result arrays full-sized, but avoid applying every large
    # finite-difference temporary to the mostly empty image volume.
    union_mask = np.any(mask4d, axis=3)
    occupied = np.where(union_mask)
    if occupied[0].size:
        lo = [max(int(np.min(axis_values)) - 1, 0) for axis_values in occupied]
        hi = [
            min(int(np.max(axis_values)) + 2, int(mask4d.shape[axis]))
            for axis, axis_values in enumerate(occupied)
        ]
        spatial_slices = tuple(slice(lo[axis], hi[axis]) for axis in range(3))
    else:
        spatial_slices = tuple(slice(0, int(mask4d.shape[axis])) for axis in range(3))

    work_mask = mask4d[spatial_slices + (slice(None),)]
    work_flow = flow[spatial_slices + (slice(None), slice(None))]
    finite_velocity = np.all(np.isfinite(work_flow), axis=-1)
    valid_samples = work_mask & finite_velocity
    velocity = np.where(finite_velocity[..., None], work_flow, 0.0).astype(np.float32) / 100.0
    if sigma > 0.0:
        weights = valid_samples.astype(np.float32)
        denominator = gaussian_filter(weights, sigma=(sigma, sigma, sigma, 0.0), mode='constant')
        for comp in range(3):
            numerator = gaussian_filter(
                velocity[..., comp] * weights,
                sigma=(sigma, sigma, sigma, 0.0),
                mode='constant',
            )
            velocity[..., comp] = np.divide(
                numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)

    dx, dy, dz = [float(max(s, 1e-12)) for s in spacing_m]
    vc = velocity[1:-1, 1:-1, 1:-1, :, :]
    du_dt = _periodic_central_difference(velocity, dt_s)[1:-1, 1:-1, 1:-1, :, :]

    conv = np.zeros_like(vc, dtype=np.float32)
    lap = np.zeros_like(vc, dtype=np.float32)
    for comp in range(3):
        du_dx = (
            velocity[2:, 1:-1, 1:-1, :, comp]
            - velocity[:-2, 1:-1, 1:-1, :, comp]
        ) / (2.0 * dx)
        du_dy = (
            velocity[1:-1, 2:, 1:-1, :, comp]
            - velocity[1:-1, :-2, 1:-1, :, comp]
        ) / (2.0 * dy)
        du_dz = (
            velocity[1:-1, 1:-1, 2:, :, comp]
            - velocity[1:-1, 1:-1, :-2, :, comp]
        ) / (2.0 * dz)
        d2u_dx2 = (
            velocity[2:, 1:-1, 1:-1, :, comp]
            - 2.0 * velocity[1:-1, 1:-1, 1:-1, :, comp]
            + velocity[:-2, 1:-1, 1:-1, :, comp]
        ) / (dx * dx)
        d2u_dy2 = (
            velocity[1:-1, 2:, 1:-1, :, comp]
            - 2.0 * velocity[1:-1, 1:-1, 1:-1, :, comp]
            + velocity[1:-1, :-2, 1:-1, :, comp]
        ) / (dy * dy)
        d2u_dz2 = (
            velocity[1:-1, 1:-1, 2:, :, comp]
            - 2.0 * velocity[1:-1, 1:-1, 1:-1, :, comp]
            + velocity[1:-1, 1:-1, :-2, :, comp]
        ) / (dz * dz)
        if use_convective_acceleration:
            conv[..., comp] = vc[..., 0] * du_dx + vc[..., 1] * du_dy + vc[..., 2] * du_dz
        lap[..., comp] = d2u_dx2 + d2u_dy2 + d2u_dz2

    grad_inner = -rho * (du_dt + conv) + mu_pa_s * lap
    # Always require the actual six-neighbour spatial derivative stencil to be
    # measured inside the lumen, even when optional extra erosion is disabled.
    # Also require both temporal neighbours; segmentation flicker must not be
    # interpreted as acceleration from setting a phase's velocity to zero.
    support_work = np.zeros(work_mask.shape, dtype=bool)
    stencil = generate_binary_structure(3, 1)
    for tidx in range(work_mask.shape[3]):
        support_work[..., tidx] = binary_erosion(valid_samples[..., tidx], structure=stencil, border_value=0)
    if work_mask.shape[3] > 1:
        support_work &= np.roll(valid_samples, -1, axis=3) & np.roll(valid_samples, 1, axis=3)
    erosion_iters = max(int(support_erosion_iters), 0)
    if erosion_iters > 0:
        structure = np.ones((3, 3, 3), dtype=bool)
        for tidx in range(work_mask.shape[3]):
            support_work[..., tidx] &= binary_erosion(
                work_mask[..., tidx],
                structure=structure,
                iterations=erosion_iters,
                border_value=0,
            )
    grad_work = np.zeros(work_flow.shape, dtype=np.float32)
    grad_work[1:-1, 1:-1, 1:-1, :, :] = grad_inner.astype(np.float32)
    grad_work *= support_work.astype(np.float32)[..., None]
    grad_mag_work = np.sqrt(np.sum(np.square(grad_work, dtype=np.float32), axis=-1)).astype(np.float32)

    display_upper = _finite_percentile_abs(grad_mag_work[support_work], 99.0, default=1.0)
    display_upper = display_upper if display_upper > 0 else 1.0

    pressure_result_work = reconstruct_relative_pressure_map(
        grad_work,
        support_work,
        spacing_mm,
        method=pressure_method,
    )

    grad = np.zeros(flow.shape, dtype=np.float32)
    grad[spatial_slices + (slice(None), slice(None))] = grad_work
    grad_mag = np.zeros(mask4d.shape, dtype=np.float32)
    grad_mag[spatial_slices + (slice(None),)] = grad_mag_work
    grad_peak = np.zeros(mask4d.shape[:3], dtype=np.float32)
    grad_peak[spatial_slices] = np.max(grad_mag_work, axis=3).astype(np.float32)
    support_mask = np.zeros(mask4d.shape, dtype=bool)
    support_mask[spatial_slices + (slice(None),)] = support_work
    relative_pressure = np.zeros(mask4d.shape, dtype=np.float32)
    relative_pressure[spatial_slices + (slice(None),)] = pressure_result_work["relative_pressure_array"]
    relative_pressure_peak = np.zeros(mask4d.shape[:3], dtype=np.float32)
    relative_pressure_peak[spatial_slices] = pressure_result_work["relative_pressure_peak"]

    centerline_profiles = compute_centerline_pressure_profiles(
        relative_pressure,
        centerline_paths or [],
        spacing_mm,
        origin,
        support_mask=support_mask,
    )

    return {
        'pressure_gradient_array': grad,
        'pressure_gradient_magnitude': grad_mag,
        'pressure_gradient_peak': grad_peak,
        'pressure_gradient_dt_s': float(dt_s),
        'pressure_gradient_temporal_scheme': 'periodic_central_difference',
        'pressure_gradient_support_mask': support_mask.astype(np.uint8),
        'pressure_gradient_display_clim': (0.0, display_upper),
        'relative_pressure_array': relative_pressure,
        'relative_pressure_peak': relative_pressure_peak,
        'relative_pressure_display_clim': pressure_result_work['relative_pressure_display_clim'],
        'pressure_method': pressure_result_work['pressure_method'],
        'centerline_pressure_profiles': centerline_profiles,
    }
