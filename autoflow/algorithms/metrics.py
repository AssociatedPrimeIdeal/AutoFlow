import h5py
import numpy as np
import pyvista as pv
from scipy.ndimage import binary_erosion, gaussian_filter
from scipy.sparse import csc_matrix, csr_matrix, diags
from scipy.sparse.linalg import cg, factorized

from .paths import _determine_plane_forward, _vector_orientation_text
from .surfaces import (
    _build_branch_grid,
    _select_connected_region,
    create_uniform_field_grid,
    create_uniform_grid,
    create_uniform_vector,
)


def extract_vectors(polydata):
    return np.stack((polydata["u"], polydata["v"], polydata["w"]), axis=-1)


def get_orthogonal_vectors(vectors, point_normals):
    c = np.sum(vectors * point_normals, axis=1)
    return c[:, None] * point_normals, vectors - c[:, None] * point_normals


def get_vector_magnitude(vectors):
    return np.sqrt(np.sum(vectors * vectors, axis=1))


def align_tangential_samples(reference_vectors, target_vectors):
    reference = np.asarray(reference_vectors, dtype=float)
    target = np.asarray(target_vectors, dtype=float)
    ref_mag = get_vector_magnitude(reference)
    tgt_mag = get_vector_magnitude(target)
    denom = np.maximum(ref_mag * tgt_mag, 1e-12)
    direction = np.sum(reference * target, axis=1) / denom
    return np.clip(direction, -1.0, 1.0) * tgt_mag


def resolve_wss_inward_distance(spacing, inward_distance=None):
    spacing_mm = np.asarray(spacing, dtype=float).reshape(3)
    if inward_distance is None:
        inward_distance = float(np.min(spacing_mm))
    return max(float(inward_distance), 0.01)


def calculate_gradient(pc0_tangent_mag, pc1_tangent_mag, pc2_tangent_mag, inward_distance, use_parabolic=True):
    x = np.array([0, 1, 2], dtype=float) * float(inward_distance)
    y = np.stack((pc0_tangent_mag, pc1_tangent_mag, pc2_tangent_mag), axis=1).T
    z = np.polynomial.polynomial.polyfit(x, y, len(x) - 1)
    x_new = np.linspace(x[0], x[-1], len(x) * 5) if use_parabolic else x
    y_new = np.polynomial.polynomial.polyval(x_new, z)
    return np.gradient(y_new, x_new, axis=1)[:, 0]


def cal_wss_from_surf(surf, velocity, viscosity=4.0, inward_distance=0.6,
                      parabolic_fitting=True, no_slip_condition=True):
    surf.compute_normals(point_normals=True, cell_normals=True, inplace=True, flip_normals=True)
    pc0 = pv.PolyData(surf.points).sample(velocity)
    pc1 = pv.PolyData(pc0.points + float(inward_distance) * surf.point_normals).sample(velocity)
    pc2 = pv.PolyData(pc0.points + 2.0 * float(inward_distance) * surf.point_normals).sample(velocity)

    if no_slip_condition:
        t0 = np.zeros(len(pc0.points))
    else:
        _, tang0 = get_orthogonal_vectors(extract_vectors(pc0), surf.point_normals)
        t0 = get_vector_magnitude(tang0)

    _, tang1 = get_orthogonal_vectors(extract_vectors(pc1), surf.point_normals)
    t1 = get_vector_magnitude(tang1)
    _, tang2 = get_orthogonal_vectors(extract_vectors(pc2), surf.point_normals)
    t2 = align_tangential_samples(tang1, tang2)
    surf["wss"] = calculate_gradient(t0, t1, t2, inward_distance, use_parabolic=parabolic_fitting) * float(viscosity)
    surf["wss_vectors"] = tang1
    return surf


def summarize_internal_consistency(plane_metrics, path_info=None, forks=None):
    by_path = {}
    for metric in plane_metrics:
        pidx = int(metric.get("path_index", -1))
        by_path.setdefault(pidx, []).append(abs(float(metric.get("netflow_mL_beat", 0.0))))
    path_ic = {}
    by_path_mean = {}
    for pidx, values in by_path.items():
        arr = np.abs(np.asarray(values, dtype=float))
        mu = float(np.mean(arr)) if len(arr) else 0.0
        by_path_mean[pidx] = mu
        if len(arr) <= 1:
            ic = 1.0
        elif mu <= 1e-12:
            ic = 1.0 if float(np.max(arr)) <= 1e-12 else 0.0
        else:
            ic = 1.0 - float(np.mean(np.abs(arr - mu)) / mu)
        path_ic[str(int(pidx))] = float(np.clip(ic, 0.0, 1.0))
    fork_items = []
    fork_ic = {}
    for fork_id, fork in enumerate(forks or []):
        left = [int(x) for x in fork.get("left", [])]
        right = [int(x) for x in fork.get("right", [])]
        sum_left = float(np.sum([abs(by_path_mean.get(x, 0.0)) for x in left]))
        sum_right = float(np.sum([abs(by_path_mean.get(x, 0.0)) for x in right]))
        denom = sum_left + sum_right
        if denom <= 1e-12:
            ic = 1.0
        else:
            ic = 1.0 - 2.0 * abs(sum_left - sum_right) / denom
        ic = float(np.clip(ic, 0.0, 1.0))
        fork_ic[str(int(fork_id))] = ic
        item = {
            "fork_id": int(fork_id),
            "left": left,
            "right": right,
            "crosspoint": fork.get("crosspoint", [0.0, 0.0, 0.0]),
            "node": int(fork.get("node", -1)),
            "ic": ic,
        }
        if path_info is not None:
            item["left_dirs"] = [path_info[x].get("direction_text", "") for x in left if 0 <= x < len(path_info)]
            item["right_dirs"] = [path_info[x].get("direction_text", "") for x in right if 0 <= x < len(path_info)]
        fork_items.append(item)
    return {"path_ic": path_ic, "fork_ic": fork_ic, "forks": fork_items}


def apply_internal_consistency_to_metrics(plane_metrics, path_info=None, forks=None):
    metrics = [dict(metric) for metric in plane_metrics]
    qc = summarize_internal_consistency(metrics, path_info=path_info, forks=forks)
    for metric in metrics:
        pidx = str(int(metric.get("path_index", -1)))
        metric["path_ic"] = float(qc["path_ic"].get(pidx, 1.0))
        rel = []
        for fork in qc.get("forks", []):
            pid = int(metric.get("path_index", -1))
            if pid in fork.get("left", []) or pid in fork.get("right", []):
                role = "incoming" if pid in fork.get("left", []) else "outgoing"
                rel.append({"fork_id": int(fork.get("fork_id", -1)), "role": role, "ic": float(fork.get("ic", 1.0))})
        metric["fork_ic"] = rel
    return metrics, qc


def compute_plane_metrics(flow_xyzt3, segmask_binary_4d, spacing, origin, planes, RR=1000.0,
                          branch_labels_3d=None, path_info=None, forks=None,
                          paths=None, return_qc=False):
    flow = _ensure_flow5d(flow_xyzt3)
    mask = np.asarray(segmask_binary_4d, dtype=bool)
    spacing = np.asarray(spacing, dtype=float).reshape(-1)[:3]
    origin = np.asarray(origin, dtype=float).reshape(-1)[:3]
    if flow.ndim != 5 or flow.shape[-1] != 3:
        raise ValueError(f"flow must be XYZTV, got {flow.shape}")
    if mask.ndim == 3:
        mask = np.repeat(mask[..., np.newaxis], flow.shape[3], axis=3)
    elif mask.ndim == 4 and mask.shape[3] == 1 and flow.shape[3] > 1:
        mask = np.repeat(mask, flow.shape[3], axis=3)
    if mask.shape[3] != flow.shape[3]:
        raise ValueError(f"mask time dimension {mask.shape[3]} does not match flow {flow.shape[3]}")
    Nt = int(flow.shape[3])
    branch_grid = _build_branch_grid(branch_labels_3d, spacing, origin)
    mask_phase_lookup = _build_mask_phase_lookup(mask)
    mask_static = all(int(rep_t) == 0 for rep_t in mask_phase_lookup)
    mask_template = mask[..., 0] if mask_static else None

    paths_lookup = None
    if paths is not None:
        paths_lookup = [np.asarray(p, dtype=float).reshape(-1, 3) for p in paths]

    if len(planes) == 0:
        empty_qc = {"path_ic": {}, "fork_ic": {}, "forks": []}
        if return_qc:
            return [], empty_qc
        return []

    results = []
    for plane in planes:
        target_label = None
        if branch_labels_3d is not None:
            target_label = int(getattr(plane, "label", 0) or 0)
            if target_label <= 0:
                ijk = np.rint((np.asarray(plane.center, dtype=float).reshape(3) - origin) / (spacing + 1e-12)).astype(int)
                ijk = np.clip(ijk, 0, np.array(np.asarray(branch_labels_3d).shape) - 1)
                target_label = int(np.asarray(branch_labels_3d)[ijk[0], ijk[1], ijk[2]])
        pp = None
        if paths_lookup is not None:
            pi = int(getattr(plane, "path_index", -1))
            if 0 <= pi < len(paths_lookup):
                pp = paths_lookup[pi]
        results.append(_compute_single_plane_metric(
            (flow, mask, spacing, origin, plane, Nt, RR,
             branch_grid, target_label, path_info, pp, mask_template, mask_phase_lookup)
        ))

    results, qc = apply_internal_consistency_to_metrics(results, path_info=path_info, forks=forks)
    if return_qc:
        return results, qc
    return results


def _ensure_mask4d(mask4d):
    mask4d = np.asarray(mask4d, dtype=bool)
    if mask4d.ndim == 3:
        mask4d = mask4d[..., np.newaxis]
    if mask4d.ndim != 4:
        raise ValueError(f"mask4d must be XYZ or XYZT, got {mask4d.shape}")
    return mask4d


def _ensure_flow5d(flow):
    flow = np.asarray(flow, dtype=np.float32)
    if flow.ndim != 5 or flow.shape[-1] != 3:
        raise ValueError(f"flow must be XYZTV, got {flow.shape}")
    return flow


def _target_label_for_plane(plane, branch_labels_3d, spacing, origin):
    if branch_labels_3d is None:
        return None
    target_label = int(getattr(plane, "label", 0) or 0)
    if target_label > 0:
        return target_label
    ijk = np.rint((np.asarray(plane.center, dtype=float).reshape(3) - origin) / (spacing + 1e-12)).astype(int)
    ijk = np.clip(ijk, 0, np.array(np.asarray(branch_labels_3d).shape) - 1)
    return int(np.asarray(branch_labels_3d)[ijk[0], ijk[1], ijk[2]])


def _build_mask_phase_lookup(mask4d):
    mask4d = _ensure_mask4d(mask4d)
    representatives = {}
    lookup = []
    for tidx in range(int(mask4d.shape[3])):
        mask_t = mask4d[..., tidx]
        rep_t = representatives.get(mask_t.tobytes())
        if rep_t is None:
            rep_t = int(tidx)
            representatives[mask_t.tobytes()] = rep_t
        lookup.append(rep_t)
    return lookup


def _build_plane_slice_spec(mask_xyz, plane, spacing, origin, branch_grid=None, target_label=None, *, select_connected=False):
    mask_xyz = np.asarray(mask_xyz, dtype=bool)
    if not np.any(mask_xyz):
        return None
    grid = create_uniform_field_grid(mask_xyz.astype(np.uint8), spacing, origin=origin, name="mask")
    grid.cell_data["_cell_id"] = np.arange(mask_xyz.size, dtype=np.int32)
    mesh = grid.threshold(0.1, scalars="mask")
    if mesh is None or mesh.n_cells == 0:
        return None
    pg = mesh.slice(normal=np.asarray(plane.normal, dtype=float), origin=np.asarray(plane.center, dtype=float))
    if pg is None or pg.n_cells == 0:
        return None
    pg = pg.compute_cell_sizes(area=True)
    if branch_grid is not None and target_label is not None and int(target_label) > 0:
        centers = pg.cell_centers().sample(branch_grid)
        bid = np.asarray(centers.point_data.get("branch_id", []))
        if len(bid) == 0:
            return None
        keep = np.where(bid == int(target_label))[0]
        if len(keep) == 0:
            return None
        pg = pg.extract_cells(keep)
        if pg is None or pg.n_cells == 0:
            return None
        pg = pg.compute_cell_sizes(area=True)
    if select_connected:
        pg = _select_connected_region(pg, ref_point=np.asarray(plane.center, dtype=float))
        if pg is None or pg.n_cells == 0:
            return None
    cell_ids = np.asarray(pg.cell_data.get("_cell_id", []), dtype=np.int64).reshape(-1)
    if cell_ids.size != int(pg.n_cells):
        return None
    areas = np.asarray(pg.cell_data.get("Area", np.ones(pg.n_cells, dtype=float)), dtype=float).reshape(-1)
    if areas.size != int(pg.n_cells):
        areas = np.ones(int(pg.n_cells), dtype=float)
    return {
        "cell_ids": cell_ids,
        "areas": areas,
    }


def _get_cached_plane_slice_spec(slice_cache, cache_key, mask_xyz, plane, spacing, origin,
                                 branch_grid=None, target_label=None, *, select_connected=False):
    if cache_key not in slice_cache:
        slice_cache[cache_key] = _build_plane_slice_spec(
            mask_xyz,
            plane,
            spacing,
            origin,
            branch_grid=branch_grid,
            target_label=target_label,
            select_connected=select_connected,
        )
    return slice_cache[cache_key]


def _sample_field_from_slice_spec(field_t, mask_shape, field_name, slice_spec):
    if slice_spec is None:
        return None
    field_arr = np.asarray(field_t)
    if tuple(field_arr.shape[:3]) != tuple(mask_shape):
        raise ValueError(f"{field_name} spatial shape {field_arr.shape[:3]} does not match mask {mask_shape}")
    cell_ids = np.asarray(slice_spec["cell_ids"], dtype=np.int64).reshape(-1)
    if field_arr.ndim == 3:
        return field_arr.reshape(-1, order="F")[cell_ids]
    if field_arr.ndim == 4 and field_arr.shape[-1] in (1, 3):
        payload = field_arr if field_arr.shape[-1] != 1 else field_arr[..., 0]
        if payload.ndim == 3:
            return payload.reshape(-1, order="F")[cell_ids]
        return payload.reshape(-1, payload.shape[-1], order="F")[cell_ids]
    raise ValueError(f"{field_name} must be XYZ, XYZT-slice scalar, or XYZV, got {field_arr.shape}")


def _extract_plane_field_region(mask_xyz, field_t, plane, spacing, origin, field_name, branch_grid=None, target_label=None):
    mask_xyz = np.asarray(mask_xyz, dtype=bool)
    if not np.any(mask_xyz):
        return None
    grid = create_uniform_field_grid(mask_xyz.astype(np.uint8), spacing, origin=origin, name="mask")
    field_arr = np.asarray(field_t)
    if field_arr.shape[:3] != mask_xyz.shape:
        raise ValueError(f"{field_name} spatial shape {field_arr.shape[:3]} does not match mask {mask_xyz.shape}")
    if field_arr.ndim == 3:
        grid.cell_data[field_name] = field_arr.reshape(-1, order="F")
    elif field_arr.ndim == 4 and field_arr.shape[-1] in (1, 3):
        payload = field_arr if field_arr.shape[-1] != 1 else field_arr[..., 0]
        if payload.ndim == 3:
            grid.cell_data[field_name] = payload.reshape(-1, order="F")
        else:
            grid.cell_data[field_name] = payload.reshape(-1, payload.shape[-1], order="F")
    else:
        raise ValueError(f"{field_name} must be XYZ, XYZT-slice scalar, or XYZV, got {field_arr.shape}")
    mesh = grid.threshold(0.1, scalars="mask")
    if mesh is None or mesh.n_cells == 0:
        return None
    pg = mesh.slice(normal=np.asarray(plane.normal, dtype=float), origin=np.asarray(plane.center, dtype=float))
    if pg is None or pg.n_cells == 0:
        return None
    pg = pg.compute_cell_sizes(area=True)
    if branch_grid is not None and target_label is not None and int(target_label) > 0:
        centers = pg.cell_centers().sample(branch_grid)
        bid = np.asarray(centers.point_data.get("branch_id", []))
        if len(bid) == 0:
            return None
        keep = np.where(bid == int(target_label))[0]
        if len(keep) == 0:
            return None
        pg = pg.extract_cells(keep)
        if pg is None or pg.n_cells == 0:
            return None
        pg = pg.compute_cell_sizes(area=True)
    return pg


def _nanpercentile_safe(values, q):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0
    return float(np.nanpercentile(finite, q))


def _weighted_mean(values, weights):
    vals = np.asarray(values, dtype=float).reshape(-1)
    wts = np.asarray(weights, dtype=float).reshape(-1)
    if vals.size == 0 or wts.size != vals.size:
        return 0.0
    finite = np.isfinite(vals) & np.isfinite(wts) & (wts > 0)
    if not np.any(finite):
        return 0.0
    vals = vals[finite]
    wts = wts[finite]
    denom = float(np.sum(wts))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(vals * wts) / denom)


def _append_summary(metric, prefix, series):
    arr = np.asarray(series, dtype=float).reshape(-1)
    metric[f"{prefix}_t"] = [float(x) for x in arr.tolist()]
    metric[prefix] = float(np.mean(arr)) if arr.size else 0.0


def summarize_plane_derived_metrics(plane, mask4d, spacing, origin, branch_labels_3d=None,
                                    tke_array=None, pressure_gradient_array=None,
                                    relative_pressure_array=None, wss_surfaces=None,
                                    branch_grid=None, mask_phase_lookup=None):
    mask4d = _ensure_mask4d(mask4d)
    spacing = np.asarray(spacing, dtype=float).reshape(3)
    origin = np.asarray(origin, dtype=float).reshape(3)
    Nt = int(mask4d.shape[3])
    branch_grid = branch_grid if branch_grid is not None else _build_branch_grid(branch_labels_3d, spacing, origin)
    target_label = _target_label_for_plane(plane, branch_labels_3d, spacing, origin)
    normal = np.asarray(plane.normal, dtype=float).reshape(3)
    normal = normal / (np.linalg.norm(normal) + 1e-12)

    summary = {}
    pixelwise = {"timepoints": []}

    has_tke = tke_array is not None
    has_pressure_gradient = pressure_gradient_array is not None
    has_relative_pressure = relative_pressure_array is not None
    has_wss = wss_surfaces is not None
    needs_volume_slice = has_tke or has_pressure_gradient or has_relative_pressure

    if has_tke:
        tke_array = _prepare_tke_array(mask4d, tke_array=tke_array, sigma=None)
    if has_pressure_gradient:
        pressure_gradient_array = np.asarray(pressure_gradient_array, dtype=np.float32)
        if pressure_gradient_array.ndim != 5 or pressure_gradient_array.shape[-1] != 3:
            raise ValueError(f"pressure_gradient_array must be XYZTV, got {pressure_gradient_array.shape}")
    if has_relative_pressure:
        relative_pressure_array = np.asarray(relative_pressure_array, dtype=np.float32)
        if relative_pressure_array.ndim != 4:
            raise ValueError(f"relative_pressure_array must be XYZT, got {relative_pressure_array.shape}")
    if mask_phase_lookup is None:
        mask_phase_lookup = _build_mask_phase_lookup(mask4d) if needs_volume_slice else []
    slice_cache = {}

    tke_mean_t = []
    tke_peak_t = []
    tke_p95_t = []
    pg_mag_mean_t = []
    pg_mag_peak_t = []
    pg_mag_p95_t = []
    pg_normal_mean_t = []
    pg_normal_peak_t = []
    pg_normal_p95_t = []
    rp_mean_t = []
    rp_peak_t = []
    rp_p95_t = []
    wss_mean_t = []
    wss_peak_t = []
    wss_p95_t = []

    for tidx in range(Nt):
        entry = {"time_index": int(tidx)}

        tke_series_vals = np.array([], dtype=float)
        pg_mag_vals = np.array([], dtype=float)
        pg_normal_vals = np.array([], dtype=float)
        rp_vals = np.array([], dtype=float)
        areas = np.array([], dtype=float)
        slice_spec = None
        if needs_volume_slice:
            rep_t = int(mask_phase_lookup[tidx])
            slice_spec = _get_cached_plane_slice_spec(
                slice_cache,
                rep_t,
                mask4d[..., rep_t],
                plane,
                spacing,
                origin,
                branch_grid=branch_grid,
                target_label=target_label,
                select_connected=False,
            )

        if has_tke and slice_spec is not None:
            tke_series_vals = np.asarray(
                _sample_field_from_slice_spec(tke_array[..., tidx], mask4d.shape[:3], "tke", slice_spec),
                dtype=float,
            ).reshape(-1)
            if tke_series_vals.size:
                areas = np.asarray(slice_spec["areas"], dtype=float).reshape(-1)
                entry["cell_area_mm2"] = areas.astype(np.float32)
                entry["tke_J_m3"] = tke_series_vals.astype(np.float32)
                entry["lumen_mask"] = np.ones_like(tke_series_vals, dtype=np.uint8)

        if has_pressure_gradient and slice_spec is not None:
            vec = np.asarray(
                _sample_field_from_slice_spec(
                    pressure_gradient_array[..., tidx, :],
                    mask4d.shape[:3],
                    "pressure_gradient",
                    slice_spec,
                ),
                dtype=float,
            )
            if vec.ndim == 2 and vec.shape[1] == 3:
                if areas.size == 0:
                    areas = np.asarray(slice_spec["areas"], dtype=float).reshape(-1)
                    entry.setdefault("cell_area_mm2", areas.astype(np.float32))
                    entry.setdefault("lumen_mask", np.ones(len(areas), dtype=np.uint8))
                pg_mag_vals = np.linalg.norm(vec, axis=1)
                pg_normal_vals = np.dot(vec, normal)
                entry["pressure_gradient_mag_Pa_m"] = pg_mag_vals.astype(np.float32)
                entry["pressure_gradient_normal_Pa_m"] = pg_normal_vals.astype(np.float32)
                entry["pressure_gradient_vec_Pa_m"] = vec.astype(np.float32)

        if has_relative_pressure and slice_spec is not None:
            vals = np.asarray(
                _sample_field_from_slice_spec(
                    relative_pressure_array[..., tidx],
                    mask4d.shape[:3],
                    "relative_pressure",
                    slice_spec,
                ),
                dtype=float,
            ).reshape(-1)
            if vals.size:
                if areas.size == 0:
                    areas = np.asarray(slice_spec["areas"], dtype=float).reshape(-1)
                    entry.setdefault("cell_area_mm2", areas.astype(np.float32))
                    entry.setdefault("lumen_mask", np.ones(len(areas), dtype=np.uint8))
                rp_vals = vals
                entry["relative_pressure_Pa"] = rp_vals.astype(np.float32)

        if tke_series_vals.size:
            tke_mean_t.append(_weighted_mean(tke_series_vals, areas))
            tke_peak_t.append(float(np.max(tke_series_vals)))
            tke_p95_t.append(_nanpercentile_safe(tke_series_vals, 95.0))
        else:
            tke_mean_t.append(0.0)
            tke_peak_t.append(0.0)
            tke_p95_t.append(0.0)

        if pg_mag_vals.size:
            pg_mag_mean_t.append(_weighted_mean(pg_mag_vals, areas))
            pg_mag_peak_t.append(float(np.max(pg_mag_vals)))
            pg_mag_p95_t.append(_nanpercentile_safe(pg_mag_vals, 95.0))
            pg_normal_mean_t.append(_weighted_mean(pg_normal_vals, areas))
            pg_normal_peak_t.append(float(np.max(np.abs(pg_normal_vals))))
            pg_normal_p95_t.append(_nanpercentile_safe(np.abs(pg_normal_vals), 95.0))
        else:
            pg_mag_mean_t.append(0.0)
            pg_mag_peak_t.append(0.0)
            pg_mag_p95_t.append(0.0)
            pg_normal_mean_t.append(0.0)
            pg_normal_peak_t.append(0.0)
            pg_normal_p95_t.append(0.0)

        if rp_vals.size:
            rp_mean_t.append(_weighted_mean(rp_vals, areas))
            rp_peak_t.append(float(np.max(np.abs(rp_vals))))
            rp_p95_t.append(_nanpercentile_safe(np.abs(rp_vals), 95.0))
        else:
            rp_mean_t.append(0.0)
            rp_peak_t.append(0.0)
            rp_p95_t.append(0.0)

        wss_vals = np.array([], dtype=float)
        surf = None if not has_wss or tidx >= len(wss_surfaces) else wss_surfaces[tidx]
        if surf is not None and getattr(surf, "n_points", 0) > 0:
            try:
                wall = surf.slice(normal=normal, origin=np.asarray(plane.center, dtype=float))
            except Exception:
                wall = None
            if wall is not None and getattr(wall, "n_points", 0) > 0:
                if "wss" not in wall.point_data and "wss" in wall.cell_data:
                    vals = np.asarray(wall.cell_data.get("wss", []), dtype=float).reshape(-1)
                else:
                    vals = np.asarray(wall.point_data.get("wss", []), dtype=float).reshape(-1)
                wss_vals = vals[np.isfinite(vals)]
        if wss_vals.size:
            wss_mean_t.append(float(np.mean(wss_vals)))
            wss_peak_t.append(float(np.max(wss_vals)))
            wss_p95_t.append(_nanpercentile_safe(wss_vals, 95.0))
        else:
            wss_mean_t.append(0.0)
            wss_peak_t.append(0.0)
            wss_p95_t.append(0.0)

        pixelwise["timepoints"].append(entry)

    if has_tke:
        _append_summary(summary, "tke_mean_J_m3", tke_mean_t)
        _append_summary(summary, "tke_peak_J_m3", tke_peak_t)
        _append_summary(summary, "tke_p95_J_m3", tke_p95_t)
    if has_pressure_gradient:
        _append_summary(summary, "pressure_gradient_mag_mean_Pa_m", pg_mag_mean_t)
        _append_summary(summary, "pressure_gradient_mag_peak_Pa_m", pg_mag_peak_t)
        _append_summary(summary, "pressure_gradient_mag_p95_Pa_m", pg_mag_p95_t)
        _append_summary(summary, "pressure_gradient_normal_mean_Pa_m", pg_normal_mean_t)
        _append_summary(summary, "pressure_gradient_normal_peak_Pa_m", pg_normal_peak_t)
        _append_summary(summary, "pressure_gradient_normal_p95_Pa_m", pg_normal_p95_t)
    if has_relative_pressure:
        _append_summary(summary, "relative_pressure_mean_Pa", rp_mean_t)
        _append_summary(summary, "relative_pressure_peak_Pa", rp_peak_t)
        _append_summary(summary, "relative_pressure_p95_Pa", rp_p95_t)
    if has_wss:
        _append_summary(summary, "wss_wall_mean_Pa", wss_mean_t)
        _append_summary(summary, "wss_wall_peak_Pa", wss_peak_t)
        _append_summary(summary, "wss_wall_p95_Pa", wss_p95_t)
    return summary, pixelwise


def augment_plane_metrics_with_derived(plane_metrics, planes, mask4d, spacing, origin, branch_labels_3d=None,
                                       tke_array=None, pressure_gradient_array=None,
                                       relative_pressure_array=None, wss_surfaces=None):
    mask4d = _ensure_mask4d(mask4d)
    shared_branch_grid = _build_branch_grid(branch_labels_3d, spacing, origin)
    mask_phase_lookup = _build_mask_phase_lookup(mask4d)
    metrics = [dict(m) for m in plane_metrics]
    pixelwise = []
    for idx, metric in enumerate(metrics):
        if idx >= len(planes):
            pixelwise.append({"plane_index": int(idx), "timepoints": []})
            continue
        summary, payload = summarize_plane_derived_metrics(
            planes[idx], mask4d, spacing, origin, branch_labels_3d=branch_labels_3d,
            tke_array=tke_array, pressure_gradient_array=pressure_gradient_array,
            relative_pressure_array=relative_pressure_array,
            wss_surfaces=wss_surfaces,
            branch_grid=shared_branch_grid,
            mask_phase_lookup=mask_phase_lookup,
        )
        metric.update(summary)
        payload["plane_index"] = int(idx)
        payload["center"] = np.asarray(planes[idx].center, dtype=float).reshape(3).tolist()
        payload["normal"] = np.asarray(planes[idx].normal, dtype=float).reshape(3).tolist()
        payload["label"] = int(getattr(planes[idx], "label", 0) or 0)
        payload["path_index"] = int(getattr(planes[idx], "path_index", -1))
        pixelwise.append(payload)
    return metrics, pixelwise


def save_plane_pixelwise_h5(path, plane_payloads, rr_ms=None, source_format=""):
    with h5py.File(path, "w") as h5:
        meta = h5.create_group("meta")
        meta.create_dataset("version", data=np.bytes_("1.0"))
        meta.create_dataset("source_format", data=np.bytes_(str(source_format or "")))
        if rr_ms is not None:
            meta.create_dataset("rr_ms", data=float(rr_ms))
        planes_group = h5.create_group("planes")
        for payload in plane_payloads:
            plane_idx = int(payload.get("plane_index", len(planes_group)))
            grp = planes_group.create_group(f"{plane_idx:03d}")
            grp.create_dataset("center_xyz", data=np.asarray(payload.get("center", [0.0, 0.0, 0.0]), dtype=np.float32))
            grp.create_dataset("normal_xyz", data=np.asarray(payload.get("normal", [1.0, 0.0, 0.0]), dtype=np.float32))
            grp.attrs["label"] = int(payload.get("label", 0))
            grp.attrs["path_index"] = int(payload.get("path_index", -1))
            timepoints = payload.get("timepoints", [])
            times_grp = grp.create_group("timepoints")
            for entry in timepoints:
                tidx = int(entry.get("time_index", len(times_grp)))
                tgrp = times_grp.create_group(f"{tidx:03d}")
                for key, value in entry.items():
                    if key == "time_index" or value is None:
                        continue
                    arr = np.asarray(value)
                    if arr.dtype.kind in {"U", "O"}:
                        continue
                    tgrp.create_dataset(key, data=arr, compression="gzip")


def compute_tke_array_from_sigma(sigma, rho=1060.0):
    return (
        0.5
        * float(rho)
        * np.sum((np.asarray(sigma, dtype=float) / 100.0) ** 2, axis=-1)
    ).astype(np.float32)


def _prepare_tke_array(mask4d, tke_array=None, sigma=None, rho=1060.0):
    mask4d = _ensure_mask4d(mask4d)
    Nt = int(mask4d.shape[-1])

    if tke_array is None:
        if sigma is None:
            raise ValueError("tke_array or sigma is required")
        tke_array = compute_tke_array_from_sigma(sigma, rho=rho)

    tke_array = np.asarray(tke_array, dtype=np.float32)
    if tke_array.ndim == 3:
        tke_array = np.repeat(tke_array[..., None], Nt, axis=3)
    elif tke_array.ndim == 4 and tke_array.shape[3] == 1 and Nt > 1:
        tke_array = np.repeat(tke_array, Nt, axis=3)
    elif tke_array.ndim != 4:
        raise ValueError(f"tke_array must be XYZ or XYZT, got {tke_array.shape}")
    if tke_array.shape[3] != Nt:
        raise ValueError(f"tke_array time dimension {tke_array.shape[3]} does not match mask {Nt}")

    tke_array = tke_array * mask4d.astype(np.float32)
    return tke_array


def compute_tke_metrics(mask4d, spacing, origin=(0, 0, 0), tke_array=None, sigma=None, rho=1060.0):
    mask4d = _ensure_mask4d(mask4d)
    tke_array = _prepare_tke_array(mask4d, tke_array=tke_array, sigma=sigma, rho=rho)
    tke_peak = np.max(tke_array, axis=3)

    TKE = create_uniform_grid(tke_peak, spacing, origin=origin, name="TKE")
    mesh_union = create_uniform_grid(np.max(mask4d > 0, axis=-1), spacing, origin=origin)
    mesh_union = mesh_union.threshold(0.1)
    TKE = mesh_union.sample(TKE)

    return {
        "tke_volume": TKE,
        "tke_array": tke_array,
        "tke_peak": np.asarray(tke_peak, dtype=np.float32),
    }


def compute_wss_metrics(mask4d, flow, spacing, origin=(0, 0, 0),
                        smoothing_iteration=200, viscosity=4.0,
                        inward_distance=None, parabolic_fitting=True,
                        no_slip_condition=False):
    mask4d = _ensure_mask4d(mask4d)
    flow = np.asarray(flow, dtype=float)
    if flow.ndim != 5 or flow.shape[-1] != 3:
        raise ValueError(f"flow must be XYZTV, got {flow.shape}")
    if flow.shape[3] != mask4d.shape[3]:
        raise ValueError(f"flow time dimension {flow.shape[3]} does not match mask {mask4d.shape[3]}")

    spacing = np.asarray(spacing, dtype=float).reshape(3)
    inward_distance = resolve_wss_inward_distance(spacing, inward_distance)
    origin = np.asarray(origin, dtype=float).reshape(3)
    wss_volume = np.zeros(mask4d.shape, dtype=np.float32)
    surfs = []
    for showt in range(int(mask4d.shape[-1])):
        velocity = create_uniform_vector(
            flow[..., showt, 0] / 100.0, flow[..., showt, 1] / 100.0,
            flow[..., showt, 2] / 100.0, spacing, origin=origin)
        mesh = create_uniform_grid(mask4d[..., showt] > 0, spacing, origin=origin)
        mesh = mesh.threshold(0.1)
        surf = mesh.extract_surface().smooth(n_iter=int(smoothing_iteration))
        surf = cal_wss_from_surf(surf, velocity, viscosity=viscosity,
                                 inward_distance=inward_distance,
                                 parabolic_fitting=parabolic_fitting,
                                 no_slip_condition=no_slip_condition)
        surfs.append(surf)
        if surf.n_points > 0 and "wss" in surf.point_data:
            pts = np.asarray(surf.points, dtype=float)
            vals = np.asarray(surf.point_data["wss"], dtype=np.float32)
            vox = np.rint((pts - origin.reshape(1, 3)) / (spacing.reshape(1, 3) + 1e-12)).astype(int)
            for k in range(3):
                vox[:, k] = np.clip(vox[:, k], 0, mask4d.shape[k] - 1)
            flat = np.ravel_multi_index((vox[:, 0], vox[:, 1], vox[:, 2]), mask4d.shape[:3])
            tgt = wss_volume[..., showt].reshape(-1)
            np.maximum.at(tgt, flat, vals)

    return {
        "wss_surfaces": surfs,
        "wss_volume": wss_volume,
    }


def _finite_percentile_abs(values, q, default=0.0):
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float(default)
    return float(np.percentile(np.abs(finite), float(q)))


def _normalize_pressure_method(method):
    token = str(method or "least_squares").strip().lower()
    if token in {"ls", "least_squares", "least-squares", "least squares"}:
        return "least_squares"
    if token in {"ppe", "poisson", "poisson_pressure_equation"}:
        return "ppe"
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
    rhs = np.asarray(rhs, dtype=np.float64).reshape(n)
    if n == 1:
        return np.zeros(1, dtype=np.float32)

    if "preconditioner" not in system:
        diag = np.asarray(matrix.diagonal(), dtype=np.float64)
        inv_diag = np.zeros_like(diag)
        nz = np.abs(diag) > 1e-12
        inv_diag[nz] = 1.0 / diag[nz]
        system["preconditioner"] = diags(inv_diag, 0, format="csr")

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

    anchor_index = int(system.get("anchor_index", 0))
    if 0 <= anchor_index < n:
        sol -= float(sol[anchor_index])
    else:
        sol -= float(np.mean(sol))
    return sol.astype(np.float32)


def _build_least_squares_system(mask_t, spacing_m):
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
    rows = []
    cols = []
    data = []
    edge_pairs = []
    edge_axes = []
    edge_scales = []
    rhs_rows = []
    rhs_cols = []
    rhs_data = []
    row_idx = 0
    dx, dy, dz = [float(v) for v in spacing_m]
    for voxel_idx, (ix, iy, iz) in enumerate(coords):
        if voxel_idx == 0:
            rows.append(row_idx)
            cols.append(voxel_idx)
            data.append(1.0)
            rhs_rows.append(voxel_idx)
            rhs_cols.append(row_idx)
            rhs_data.append(1.0)
            row_idx += 1
        if ix + 1 < mask_t.shape[0] and mask_t[ix + 1, iy, iz]:
            nbr = int(index_map[ix + 1, iy, iz])
            scale = 1.0 / dx
            rows.extend((row_idx, row_idx))
            cols.extend((voxel_idx, nbr))
            data.extend((-scale, scale))
            edge_pairs.append((voxel_idx, nbr))
            edge_axes.append(0)
            edge_scales.append(scale)
            rhs_rows.extend((voxel_idx, nbr))
            rhs_cols.extend((row_idx, row_idx))
            rhs_data.extend((-scale, scale))
            row_idx += 1
        if iy + 1 < mask_t.shape[1] and mask_t[ix, iy + 1, iz]:
            nbr = int(index_map[ix, iy + 1, iz])
            scale = 1.0 / dy
            rows.extend((row_idx, row_idx))
            cols.extend((voxel_idx, nbr))
            data.extend((-scale, scale))
            edge_pairs.append((voxel_idx, nbr))
            edge_axes.append(1)
            edge_scales.append(scale)
            rhs_rows.extend((voxel_idx, nbr))
            rhs_cols.extend((row_idx, row_idx))
            rhs_data.extend((-scale, scale))
            row_idx += 1
        if iz + 1 < mask_t.shape[2] and mask_t[ix, iy, iz + 1]:
            nbr = int(index_map[ix, iy, iz + 1])
            scale = 1.0 / dz
            rows.extend((row_idx, row_idx))
            cols.extend((voxel_idx, nbr))
            data.extend((-scale, scale))
            edge_pairs.append((voxel_idx, nbr))
            edge_axes.append(2)
            edge_scales.append(scale)
            rhs_rows.extend((voxel_idx, nbr))
            rhs_cols.extend((row_idx, row_idx))
            rhs_data.extend((-scale, scale))
            row_idx += 1

    if row_idx == 0:
        matrix = diags([1.0], [0], shape=(n, n), dtype=np.float64).tocsr()
    else:
        incidence = csr_matrix((np.asarray(data, dtype=np.float64), (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))), shape=(row_idx, n), dtype=np.float64)
        matrix = (incidence.T @ incidence).tocsr()
    matrix = matrix + diags([1e-6], [0], shape=(n, n), dtype=np.float64)
    return {
        "coords": coords,
        "index_map": index_map,
        "matrix": matrix.tocsr(),
        "rhs_operator": csr_matrix((np.asarray(rhs_data, dtype=np.float64), (np.asarray(rhs_rows, dtype=np.int32), np.asarray(rhs_cols, dtype=np.int32))), shape=(n, row_idx), dtype=np.float64),
        "edge_pairs": np.asarray(edge_pairs, dtype=np.int32).reshape(-1, 2),
        "edge_axes": np.asarray(edge_axes, dtype=np.int8),
        "edge_scales": np.asarray(edge_scales, dtype=np.float64),
        "anchor_index": 0,
    }


def _build_ppe_system(mask_t, spacing_m):
    coords = np.argwhere(mask_t)
    n = int(len(coords))
    if n == 0:
        return {
            "coords": coords,
            "index_map": -np.ones(mask_t.shape, dtype=np.int32),
            "matrix": csr_matrix((0, 0), dtype=np.float64),
            "anchor_index": 0,
        }

    index_map = -np.ones(mask_t.shape, dtype=np.int32)
    index_map[mask_t] = np.arange(n, dtype=np.int32)
    rows = []
    cols = []
    data = []
    dx, dy, dz = [float(v) for v in spacing_m]
    for voxel_idx, (ix, iy, iz) in enumerate(coords):
        if voxel_idx == 0:
            rows.append(voxel_idx)
            cols.append(voxel_idx)
            data.append(1.0)
            continue
        diag = 0.0
        for sx, sy, sz in _neighbor_shifts():
            jx, jy, jz = ix + sx, iy + sy, iz + sz
            if not (0 <= jx < mask_t.shape[0] and 0 <= jy < mask_t.shape[1] and 0 <= jz < mask_t.shape[2]):
                continue
            if not mask_t[jx, jy, jz]:
                continue
            step = dx if sx != 0 else dy if sy != 0 else dz
            weight = 1.0 / (step * step)
            rows.append(voxel_idx)
            cols.append(int(index_map[jx, jy, jz]))
            data.append(-weight)
            diag += weight
        rows.append(voxel_idx)
        cols.append(voxel_idx)
        data.append(diag if diag > 0.0 else 1.0)

    matrix = csr_matrix((np.asarray(data, dtype=np.float64), (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))), shape=(n, n), dtype=np.float64)
    return {
        "coords": coords,
        "index_map": index_map,
        "matrix": matrix.tocsr(),
        "anchor_index": 0,
    }


def _least_squares_rhs(grad_t, system):
    edge_pairs = system["edge_pairs"]
    if edge_pairs.size == 0:
        rhs = np.zeros(system["matrix"].shape[0], dtype=np.float64)
        if rhs.size:
            rhs[0] = 0.0
        return rhs
    edge_axes = system["edge_axes"]
    coords = system["coords"]
    edge_scales = system["edge_scales"]
    rhs_rows = np.zeros(int(edge_pairs.shape[0]) + 1, dtype=np.float64)
    rhs_rows[1:] = -np.asarray(grad_t[coords[edge_pairs[:, 0], 0], coords[edge_pairs[:, 0], 1], coords[edge_pairs[:, 0], 2], edge_axes], dtype=np.float64) * edge_scales
    return rhs_rows


def _ppe_rhs(grad_t, mask_t, spacing_m, system):
    dx, dy, dz = [float(v) for v in spacing_m]
    divergence = np.zeros(mask_t.shape, dtype=np.float64)
    gx_field = np.asarray(grad_t[..., 0], dtype=np.float64)
    gy_field = np.asarray(grad_t[..., 1], dtype=np.float64)
    gz_field = np.asarray(grad_t[..., 2], dtype=np.float64)
    divergence[1:-1, :, :] += (gx_field[2:, :, :] - gx_field[:-2, :, :]) / (2.0 * dx)
    divergence[:, 1:-1, :] += (gy_field[:, 2:, :] - gy_field[:, :-2, :]) / (2.0 * dy)
    divergence[:, :, 1:-1] += (gz_field[:, :, 2:] - gz_field[:, :, :-2]) / (2.0 * dz)
    rhs = np.zeros(system["matrix"].shape[0], dtype=np.float64)
    coords = system["coords"]
    if len(coords):
        rhs[:] = divergence[coords[:, 0], coords[:, 1], coords[:, 2]]
        rhs[0] = 0.0
    return rhs


def reconstruct_relative_pressure_map(pressure_gradient_array, support_mask, spacing, *, method="least_squares"):
    method = _normalize_pressure_method(method)
    grad = np.asarray(pressure_gradient_array, dtype=np.float32)
    if grad.ndim != 5 or grad.shape[-1] != 3:
        raise ValueError(f"pressure_gradient_array must be XYZTV, got {grad.shape}")
    support = np.asarray(support_mask, dtype=bool)
    if support.shape != grad.shape[:4]:
        raise ValueError(f"support_mask shape {support.shape} does not match {grad.shape[:4]}")

    spacing_m = np.asarray(spacing, dtype=float).reshape(3) / 1000.0
    dx, dy, dz = [float(max(v, 1e-12)) for v in spacing_m]
    nt = grad.shape[3]
    pressure = np.zeros(grad.shape[:4], dtype=np.float32)

    system_cache = {}
    for tidx in range(nt):
        mask_t = support[..., tidx]
        if not np.any(mask_t):
            continue
        cache_key = mask_t.tobytes()
        if method == "least_squares":
            system = system_cache.get(cache_key)
            if system is None:
                system = _build_least_squares_system(mask_t, (dx, dy, dz))
                system_cache[cache_key] = system
            rhs_rows = _least_squares_rhs(grad[..., tidx, :], system)
            rhs = system["rhs_operator"] @ rhs_rows
            sol = _solve_reconstruction_system(system, rhs)
        else:
            system = system_cache.get(cache_key)
            if system is None:
                system = _build_ppe_system(mask_t, (dx, dy, dz))
                system_cache[cache_key] = system
            rhs = _ppe_rhs(grad[..., tidx, :], mask_t, (dx, dy, dz), system)
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


def _points_as_voxels(points_xyz, spacing, origin, shape):
    pts = np.asarray(points_xyz, dtype=float).reshape(-1, 3)
    spacing = np.asarray(spacing, dtype=float).reshape(3)
    origin = np.asarray(origin, dtype=float).reshape(3)
    shape = np.asarray(shape, dtype=int).reshape(3)
    if len(pts) == 0:
        return pts, False
    looks_like_voxels = (
        np.all(np.isfinite(pts))
        and np.all(pts >= -0.5)
        and np.all(pts <= (shape.reshape(1, 3) - 0.5))
    )
    if looks_like_voxels:
        return pts, True
    vox = (pts - origin.reshape(1, 3)) / (spacing.reshape(1, 3) + 1e-12)
    return vox, False


def _sample_volume_at_points(volume_xyz, points_xyz, spacing, origin):
    vol = np.asarray(volume_xyz, dtype=np.float32)
    pts = np.asarray(points_xyz, dtype=float).reshape(-1, 3)
    if vol.ndim != 3 or len(pts) == 0:
        return np.zeros(len(pts), dtype=np.float32)
    spacing = np.asarray(spacing, dtype=float).reshape(3)
    origin = np.asarray(origin, dtype=float).reshape(3)
    shape = np.array(vol.shape, dtype=int)
    vox, _ = _points_as_voxels(pts, spacing, origin, shape)
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


def compute_centerline_pressure_profiles(relative_pressure_array, centerline_paths, spacing, origin):
    pressure = np.asarray(relative_pressure_array, dtype=np.float32)
    if pressure.ndim != 4:
        raise ValueError(f"relative_pressure_array must be XYZT, got {pressure.shape}")
    spacing = np.asarray(spacing, dtype=float).reshape(3)
    origin = np.asarray(origin, dtype=float).reshape(3)
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
                "pressure_drop_mean_Pa": 0.0,
                "pressure_drop_peak_Pa": 0.0,
            })
            continue
        pts_vox, are_voxels = _points_as_voxels(pts, spacing, origin, pressure.shape[:3])
        if len(pts_vox) > 1:
            diffs = np.diff(pts_vox, axis=0)
            seg = np.linalg.norm(diffs * spacing.reshape(1, 3), axis=1)
        else:
            seg = np.zeros(0, dtype=float)
        dist = np.concatenate([[0.0], np.cumsum(seg)]) if len(pts) > 0 else np.zeros(0, dtype=float)
        sample_pts = pts_vox if are_voxels else pts
        samples_t = []
        drop_t = []
        for tidx in range(nt):
            vals = _sample_volume_at_points(pressure[..., tidx], sample_pts, spacing, origin)
            samples_t.append(vals.astype(np.float32).tolist())
            drop_t.append(float(vals[0] - vals[-1]) if len(vals) else 0.0)
        profiles.append({
            "path_index": int(path_idx),
            "distances_mm": dist.astype(np.float32).tolist(),
            "relative_pressure_Pa_t": samples_t,
            "pressure_drop_Pa_t": [float(x) for x in drop_t],
            "pressure_drop_mean_Pa": float(np.mean(drop_t)) if drop_t else 0.0,
            "pressure_drop_peak_Pa": float(np.max(np.abs(drop_t))) if drop_t else 0.0,
        })
    return profiles


def compute_pressure_gradient_metrics(mask4d, flow, spacing, rr, rho=1060.0, viscosity=4.0,
                                      smoothing_sigma=0.0, support_erosion_iters=1, use_convective_acceleration=True,
                                      pressure_method="least_squares", centerline_paths=None,
                                      origin=(0, 0, 0)):
    mask4d = _ensure_mask4d(mask4d)
    flow = np.asarray(flow, dtype=np.float32)
    if flow.ndim != 5 or flow.shape[-1] != 3:
        raise ValueError(f"flow must be XYZTV, got {flow.shape}")
    if flow.shape[3] != mask4d.shape[3]:
        raise ValueError(f"flow time dimension {flow.shape[3]} does not match mask {mask4d.shape[3]}")

    spacing_mm = np.asarray(spacing, dtype=float).reshape(3)
    spacing_m = spacing_mm / 1000.0
    dt_s = float(rr) / 1000.0 / float(flow.shape[3])
    rho = float(rho)
    mu_pa_s = float(viscosity) / 1000.0
    sigma = max(float(smoothing_sigma), 0.0)

    velocity = np.asarray(flow, dtype=np.float32) / 100.0
    mask_float = mask4d.astype(np.float32)
    velocity = velocity * mask_float[..., None]
    if sigma > 0.0:
        for comp in range(3):
            velocity[..., comp] = gaussian_filter(
                velocity[..., comp],
                sigma=(sigma, sigma, sigma, 0.0),
                mode='nearest',
            )
        velocity = velocity * mask_float[..., None]

    dx, dy, dz = [float(max(s, 1e-12)) for s in spacing_m]
    vc = velocity[1:-1, 1:-1, 1:-1, 1:-1, :]
    du_dt = (
        velocity[1:-1, 1:-1, 1:-1, 2:, :]
        - velocity[1:-1, 1:-1, 1:-1, :-2, :]
    ) / (2.0 * max(dt_s, 1e-12))

    conv = np.zeros_like(vc, dtype=np.float32)
    lap = np.zeros_like(vc, dtype=np.float32)
    for comp in range(3):
        du_dx = (
            velocity[2:, 1:-1, 1:-1, 1:-1, comp]
            - velocity[:-2, 1:-1, 1:-1, 1:-1, comp]
        ) / (2.0 * dx)
        du_dy = (
            velocity[1:-1, 2:, 1:-1, 1:-1, comp]
            - velocity[1:-1, :-2, 1:-1, 1:-1, comp]
        ) / (2.0 * dy)
        du_dz = (
            velocity[1:-1, 1:-1, 2:, 1:-1, comp]
            - velocity[1:-1, 1:-1, :-2, 1:-1, comp]
        ) / (2.0 * dz)
        d2u_dx2 = (
            velocity[2:, 1:-1, 1:-1, 1:-1, comp]
            - 2.0 * velocity[1:-1, 1:-1, 1:-1, 1:-1, comp]
            + velocity[:-2, 1:-1, 1:-1, 1:-1, comp]
        ) / (dx * dx)
        d2u_dy2 = (
            velocity[1:-1, 2:, 1:-1, 1:-1, comp]
            - 2.0 * velocity[1:-1, 1:-1, 1:-1, 1:-1, comp]
            + velocity[1:-1, :-2, 1:-1, 1:-1, comp]
        ) / (dy * dy)
        d2u_dz2 = (
            velocity[1:-1, 1:-1, 2:, 1:-1, comp]
            - 2.0 * velocity[1:-1, 1:-1, 1:-1, 1:-1, comp]
            + velocity[1:-1, 1:-1, :-2, 1:-1, comp]
        ) / (dz * dz)
        if use_convective_acceleration:
            conv[..., comp] = vc[..., 0] * du_dx + vc[..., 1] * du_dy + vc[..., 2] * du_dz
        lap[..., comp] = d2u_dx2 + d2u_dy2 + d2u_dz2

    grad_inner = -rho * (du_dt + conv) + mu_pa_s * lap
    support_mask = np.asarray(mask4d, dtype=bool).copy()
    erosion_iters = max(int(support_erosion_iters), 0)
    if erosion_iters > 0:
        structure = np.ones((3, 3, 3), dtype=bool)
        for tidx in range(mask4d.shape[3]):
            support_mask[..., tidx] = binary_erosion(mask4d[..., tidx], structure=structure, iterations=erosion_iters, border_value=0)
    support_inner = support_mask[1:-1, 1:-1, 1:-1, 1:-1]

    grad = np.zeros(flow.shape, dtype=np.float32)
    grad[1:-1, 1:-1, 1:-1, 1:-1, :] = grad_inner.astype(np.float32)
    grad *= support_mask.astype(np.float32)[..., None]
    grad_mag = np.sqrt(np.sum(np.square(grad, dtype=np.float32), axis=-1)).astype(np.float32)
    grad_peak = np.max(grad_mag, axis=3).astype(np.float32)

    finite_inner = grad_inner[np.isfinite(grad_inner) & support_inner[..., None]]
    display_upper = _finite_percentile_abs(finite_inner, 99.0, default=1.0)
    display_upper = display_upper if display_upper > 0 else 1.0

    pressure_result = reconstruct_relative_pressure_map(
        grad,
        support_mask.astype(bool),
        spacing_mm,
        method=pressure_method,
    )
    centerline_profiles = compute_centerline_pressure_profiles(
        pressure_result["relative_pressure_array"],
        centerline_paths or [],
        spacing_mm,
        origin,
    )

    return {
        'pressure_gradient_array': grad,
        'pressure_gradient_magnitude': grad_mag,
        'pressure_gradient_peak': grad_peak,
        'pressure_gradient_dt_s': float(dt_s),
        'pressure_gradient_support_mask': support_mask.astype(np.uint8),
        'pressure_gradient_display_clim': (0.0, display_upper),
        'relative_pressure_array': pressure_result['relative_pressure_array'],
        'relative_pressure_peak': pressure_result['relative_pressure_peak'],
        'relative_pressure_display_clim': pressure_result['relative_pressure_display_clim'],
        'pressure_method': pressure_result['pressure_method'],
        'centerline_pressure_profiles': centerline_profiles,
    }


def compute_derived_metrics(mask4d, flow, spacing, origin=(0, 0, 0),
                            smoothing_iteration=200, viscosity=4.0,
                            inward_distance=None, parabolic_fitting=True,
                            no_slip_condition=False, step_size=5,
                            tube_radius=0.1, rho=1060.0,
                            save_pixelwise=False, tke_array=None, sigma=None,
                            rr=1000.0, pressure_gradient_smoothing_sigma=0.0,
                            pressure_gradient_support_erosion_iters=1,
                            pressure_gradient_use_convective_acceleration=True,
                            compute_wss=True, compute_tke=True,
                            compute_pressure_gradient=True,
                            wss_smoothing_iteration=None,
                            wss_viscosity=None,
                            wss_inward_distance=None,
                            wss_parabolic_fitting=None,
                            wss_no_slip_condition=None,
                            tke_rho=None,
                            pressure_gradient_rho=None,
                            pressure_gradient_viscosity=None,
                            pressure_method="least_squares",
                            centerline_paths=None):
    mask4d = _ensure_mask4d(mask4d)
    wss_smoothing_iteration = smoothing_iteration if wss_smoothing_iteration is None else wss_smoothing_iteration
    wss_viscosity = viscosity if wss_viscosity is None else wss_viscosity
    wss_inward_distance = inward_distance if wss_inward_distance is None else wss_inward_distance
    wss_parabolic_fitting = parabolic_fitting if wss_parabolic_fitting is None else wss_parabolic_fitting
    wss_no_slip_condition = no_slip_condition if wss_no_slip_condition is None else wss_no_slip_condition
    tke_rho = rho if tke_rho is None else tke_rho
    pressure_gradient_rho = rho if pressure_gradient_rho is None else pressure_gradient_rho
    pressure_gradient_viscosity = viscosity if pressure_gradient_viscosity is None else pressure_gradient_viscosity
    wss = None
    if compute_wss:
        wss = compute_wss_metrics(
            mask4d, flow, spacing, origin=origin,
            smoothing_iteration=wss_smoothing_iteration,
            viscosity=wss_viscosity,
            inward_distance=wss_inward_distance,
            parabolic_fitting=wss_parabolic_fitting,
            no_slip_condition=wss_no_slip_condition,
        )
    tke = None
    if compute_tke and (tke_array is not None or sigma is not None):
        tke = compute_tke_metrics(
            mask4d, spacing, origin=origin, tke_array=tke_array, sigma=sigma, rho=tke_rho,
        )
    pressure_gradient = None
    if compute_pressure_gradient:
        pressure_gradient = compute_pressure_gradient_metrics(
            mask4d,
            flow,
            spacing,
            rr=rr,
            rho=pressure_gradient_rho,
            viscosity=pressure_gradient_viscosity,
            smoothing_sigma=pressure_gradient_smoothing_sigma,
            support_erosion_iters=pressure_gradient_support_erosion_iters,
            use_convective_acceleration=pressure_gradient_use_convective_acceleration,
            pressure_method=pressure_method,
            centerline_paths=centerline_paths,
            origin=origin,
        )

    spacing = np.asarray(spacing, dtype=float).reshape(3)
    origin = np.asarray(origin, dtype=float).reshape(3)
    result = {
        "wss_surfaces": [] if wss is None else wss["wss_surfaces"],
        "wss_volume": None if wss is None else wss["wss_volume"],
        "tke_volume": None if tke is None else tke["tke_volume"],
        "tke_array": None if tke is None else tke["tke_array"],
        "tke_peak": None if tke is None else tke["tke_peak"],
        "pressure_gradient_array": None if pressure_gradient is None else pressure_gradient["pressure_gradient_array"],
        "pressure_gradient_magnitude": None if pressure_gradient is None else pressure_gradient["pressure_gradient_magnitude"],
        "pressure_gradient_peak": None if pressure_gradient is None else pressure_gradient["pressure_gradient_peak"],
        "pressure_gradient_dt_s": None if pressure_gradient is None else pressure_gradient["pressure_gradient_dt_s"],
        "pressure_gradient_support_mask": None if pressure_gradient is None else pressure_gradient["pressure_gradient_support_mask"],
        "pressure_gradient_display_clim": None if pressure_gradient is None else pressure_gradient["pressure_gradient_display_clim"],
        "relative_pressure_array": None if pressure_gradient is None else pressure_gradient["relative_pressure_array"],
        "relative_pressure_peak": None if pressure_gradient is None else pressure_gradient["relative_pressure_peak"],
        "relative_pressure_display_clim": None if pressure_gradient is None else pressure_gradient["relative_pressure_display_clim"],
        "centerline_pressure_profiles": [] if pressure_gradient is None else pressure_gradient["centerline_pressure_profiles"],
        "pressure_method": None if pressure_gradient is None else pressure_gradient["pressure_method"],
        "streamlines": [],
        "tube_radius": float(tube_radius),
    }
    if save_pixelwise:
        pixelwise_export = {
            "spacing": np.asarray(spacing, dtype=np.float32),
            "origin": np.asarray(origin, dtype=np.float32),
        }
        if wss is not None:
            pixelwise_export["wss"] = np.asarray(wss["wss_volume"], dtype=np.float32)
        if pressure_gradient is not None:
            pixelwise_export["pressure_gradient"] = np.asarray(pressure_gradient["pressure_gradient_array"], dtype=np.float32)
            pixelwise_export["pressure_gradient_mag"] = np.asarray(pressure_gradient["pressure_gradient_magnitude"], dtype=np.float32)
            pixelwise_export["pressure_gradient_peak"] = np.asarray(pressure_gradient["pressure_gradient_peak"], dtype=np.float32)
            pixelwise_export["pressure_gradient_support_mask"] = np.asarray(pressure_gradient["pressure_gradient_support_mask"], dtype=np.uint8)
            pixelwise_export["relative_pressure"] = np.asarray(pressure_gradient["relative_pressure_array"], dtype=np.float32)
            pixelwise_export["relative_pressure_peak"] = np.asarray(pressure_gradient["relative_pressure_peak"], dtype=np.float32)
        if tke is not None:
            pixelwise_export["tke"] = np.asarray(tke["tke_peak"], dtype=np.float32)
            pixelwise_export["tke_time"] = np.asarray(tke["tke_array"], dtype=np.float32)
        result["pixelwise_export"] = pixelwise_export
    else:
        result["pixelwise_export"] = {}
    return result


def _compute_single_plane_metric(args):
    (flow, mask, spacing, origin, plane, Nt, RR, branch_grid, target_label,
     path_info, path_points, mask_template, mask_phase_lookup) = args
    normal = np.asarray(plane.normal, dtype=float).reshape(3)
    normal = normal / (np.linalg.norm(normal) + 1e-12)

    forward_sign, forward_sign_source, local_tangent, ntc = _determine_plane_forward(
        plane, path_points, path_info, eps=0.1,
    )

    peakv_abs = 0.0
    peakv_fwd = 0.0
    peakv_rev = 0.0
    flowrate = []        
    flowrate_fwd = []     
    flowrate_rev = []     
    area = []
    meanv_t = []         
    meanv_fwd_t = []     
    meanv_rev_t = []     
    slice_cache = {}

    for t in range(Nt):
        rep_t = int(mask_phase_lookup[t]) if mask_phase_lookup else int(t)
        mask_t = mask_template if mask_template is not None else mask[..., rep_t]
        slice_spec = _get_cached_plane_slice_spec(
            slice_cache,
            rep_t,
            mask_t,
            plane,
            spacing,
            origin,
            branch_grid=branch_grid,
            target_label=target_label,
            select_connected=True,
        )
        if slice_spec is None:
            flowrate.append(0.0); flowrate_fwd.append(0.0); flowrate_rev.append(0.0)
            meanv_t.append(0.0); meanv_fwd_t.append(0.0); meanv_rev_t.append(0.0)
            area.append(0.0)
            continue
        vec = np.asarray(
            _sample_field_from_slice_spec(flow[..., t, :], mask.shape[:3], "flow", slice_spec),
            dtype=float,
        )
        ca = np.asarray(slice_spec["areas"], dtype=float).reshape(-1)
        if vec.ndim != 2 or vec.shape[1] != 3 or len(vec) != len(ca):
            flowrate.append(0.0); flowrate_fwd.append(0.0); flowrate_rev.append(0.0)
            meanv_t.append(0.0); meanv_fwd_t.append(0.0); meanv_rev_t.append(0.0)
            area.append(float(np.sum(ca)) if len(ca) else 0.0)
            continue

        proj = np.dot(vec, normal)
        proj_fwd = proj * float(forward_sign)
        pos_mask = proj_fwd > 0.0
        neg_mask = proj_fwd < 0.0

        area_t = float(np.sum(ca)) if len(ca) else 0.0

        fr = float(np.sum(proj * ca) / 100.0) if area_t > 0.0 else 0.0
        if area_t > 0.0:
            fr_fwd = float(np.sum(proj_fwd[pos_mask] * ca[pos_mask]) / 100.0)
            fr_rev = float(-np.sum(proj_fwd[neg_mask] * ca[neg_mask]) / 100.0)
            mv = float(np.sum(proj * ca) / area_t)
            mv_fwd = float(np.sum(np.where(pos_mask, proj_fwd, 0.0) * ca) / area_t)
            mv_rev = float(-np.sum(np.where(neg_mask, proj_fwd, 0.0) * ca) / area_t)
        else:
            fr_fwd = fr_rev = 0.0
            mv = mv_fwd = mv_rev = 0.0

        if len(proj):
            peakv_abs = max(peakv_abs, float(np.max(np.abs(proj))))
        if len(proj_fwd):
            if np.any(pos_mask):
                peakv_fwd = max(peakv_fwd, float(np.max(proj_fwd[pos_mask])))
            if np.any(neg_mask):
                peakv_rev = max(peakv_rev, float(-np.min(proj_fwd[neg_mask])))

        flowrate.append(fr)
        flowrate_fwd.append(fr_fwd)
        flowrate_rev.append(fr_rev)
        meanv_t.append(mv)
        meanv_fwd_t.append(mv_fwd)
        meanv_rev_t.append(mv_rev)
        area.append(area_t)

    netflow_mag = float(abs(np.mean(flowrate)) * RR / 1000.0) if len(flowrate) else 0.0
    netflow_fwd = float(np.mean(flowrate_fwd) * RR / 1000.0) if len(flowrate_fwd) else 0.0
    netflow_rev = float(np.mean(flowrate_rev) * RR / 1000.0) if len(flowrate_rev) else 0.0
    net_signed = float(netflow_fwd - netflow_rev)
    reflux_frac = float(netflow_rev / netflow_fwd) if netflow_fwd > 1e-12 else 0.0
    meanv_fwd = float(np.mean(meanv_fwd_t)) if meanv_fwd_t else 0.0
    meanv_rev = float(np.mean(meanv_rev_t)) if meanv_rev_t else 0.0

    sgn = float(forward_sign)
    flowrate_signed = [float(x) * sgn for x in flowrate]
    meanv_signed_t = [float(x) * sgn for x in meanv_t]
    meanv_signed = float(np.mean(meanv_signed_t)) if meanv_signed_t else 0.0

    metric = {
        "center": np.asarray(plane.center, dtype=float).tolist(),
        "normal": normal.tolist(),
        "label": int(plane.label),
        "path_index": int(plane.path_index),
        "distance": float(plane.distance),
        "target_branch_label": int(target_label) if target_label is not None else 0,
        "peakv_cm_s": float(peakv_abs),
        "flowrate_mL_s": [float(x) for x in flowrate],
        "netflow_mL_beat": netflow_mag,
        "meanv_cm_s": float(np.mean(meanv_t)) if meanv_t else 0.0,
        "meanv_cm_s_t": [float(x) for x in meanv_t],
        "area_mm2": [float(x) for x in area],
        "flowrate_forward_mL_s": [float(x) for x in flowrate_fwd],
        "flowrate_reverse_mL_s": [float(x) for x in flowrate_rev],
        "meanv_forward_cm_s_t": [float(x) for x in meanv_fwd_t],
        "meanv_reverse_cm_s_t": [float(x) for x in meanv_rev_t],
        "flowrate_signed_mL_s": flowrate_signed,
        "meanv_signed_cm_s_t": meanv_signed_t,
        "meanv_signed_cm_s": meanv_signed,
        "netflow_forward_mL_beat": netflow_fwd,
        "netflow_reverse_mL_beat": netflow_rev,
        "net_netflow_signed_mL_beat": net_signed,
        "reflux_fraction": reflux_frac,
        "peakv_forward_cm_s": float(peakv_fwd),
        "peakv_reverse_cm_s": float(peakv_rev),
        "meanv_forward_cm_s": meanv_fwd,
        "meanv_reverse_cm_s": meanv_rev,
        "forward_sign": int(forward_sign),
        "forward_sign_source": str(forward_sign_source),
        "local_path_tangent": [float(x) for x in np.asarray(local_tangent, dtype=float).tolist()],
        "local_path_direction": _vector_orientation_text(local_tangent),
        "normal_tangent_cos": float(ntc),
    }
    if path_info is not None and 0 <= int(plane.path_index) < len(path_info):
        info = path_info[int(plane.path_index)]
        metric["path_direction"] = info.get("direction_text", "")
        metric["path_start_point"] = info.get("start_point", [0.0, 0.0, 0.0])
        metric["path_end_point"] = info.get("end_point", [0.0, 0.0, 0.0])
        metric["path_fork_ids"] = info.get("fork_ids", [])
        metric["path_fork_roles"] = info.get("fork_roles", [])
    return metric


def compute_plane_metrics_multithread(flow_xyzt3, segmask_binary_4d, spacing, origin, planes, RR=1000.0,
                                       branch_labels_3d=None, path_info=None, forks=None,
                                       paths=None, return_qc=False, max_workers=None):
    from concurrent.futures import ThreadPoolExecutor
    flow = _ensure_flow5d(flow_xyzt3)
    mask = np.asarray(segmask_binary_4d, dtype=bool)
    spacing = np.asarray(spacing, dtype=float).reshape(-1)[:3]
    origin = np.asarray(origin, dtype=float).reshape(-1)[:3]
    if flow.ndim != 5 or flow.shape[-1] != 3:
        raise ValueError(f"flow must be XYZTV, got {flow.shape}")
    if mask.ndim == 3:
        mask = np.repeat(mask[..., np.newaxis], flow.shape[3], axis=3)
    elif mask.ndim == 4 and mask.shape[3] == 1 and flow.shape[3] > 1:
        mask = np.repeat(mask, flow.shape[3], axis=3)
    Nt = int(flow.shape[3])
    mask_phase_lookup = _build_mask_phase_lookup(mask)
    mask_static = all(int(rep_t) == 0 for rep_t in mask_phase_lookup)
    mask_template = mask[..., 0] if mask_static else None

    branch_grid = _build_branch_grid(branch_labels_3d, spacing, origin)
    paths_lookup = None
    if paths is not None:
        paths_lookup = [np.asarray(p, dtype=float).reshape(-1, 3) for p in paths]

    if len(planes) == 0:
        empty_qc = {"path_ic": {}, "fork_ic": {}, "forks": []}
        if return_qc:
            return [], empty_qc
        return []

    args_list = []
    for plane in planes:
        target_label = None
        if branch_labels_3d is not None:
            target_label = int(getattr(plane, "label", 0) or 0)
            if target_label <= 0:
                ijk = np.rint((np.asarray(plane.center, dtype=float).reshape(3) - origin) / (spacing + 1e-12)).astype(int)
                ijk = np.clip(ijk, 0, np.array(np.asarray(branch_labels_3d).shape) - 1)
                target_label = int(np.asarray(branch_labels_3d)[ijk[0], ijk[1], ijk[2]])
        pp = None
        if paths_lookup is not None:
            pi = int(getattr(plane, "path_index", -1))
            if 0 <= pi < len(paths_lookup):
                pp = paths_lookup[pi]
        args_list.append((flow, mask, spacing, origin, plane, Nt, RR,
                          branch_grid, target_label, path_info, pp, mask_template, mask_phase_lookup))
    if max_workers is None:
        import os as _os
        max_workers = min(len(planes), max(1, _os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(_compute_single_plane_metric, args_list))
    results, qc = apply_internal_consistency_to_metrics(results, path_info=path_info, forks=forks)
    if return_qc:
        return results, qc
    return results


def load_metrics_as_table(metrics_json_path, qc_json_path=None):
    import json as _json
    with open(metrics_json_path, "r", encoding="utf-8") as f:
        metrics = _json.load(f)
    scalar_keys = [
        "label", "path_index", "distance", "target_branch_label",
        "peakv_cm_s", "netflow_mL_beat", "meanv_cm_s", "path_ic",
        "path_direction",
        "peakv_forward_cm_s", "peakv_reverse_cm_s",
        "meanv_forward_cm_s", "meanv_reverse_cm_s",
        "netflow_forward_mL_beat", "netflow_reverse_mL_beat",
        "net_netflow_signed_mL_beat", "reflux_fraction",
        "meanv_signed_cm_s",
        "forward_sign", "forward_sign_source",
        "local_path_direction", "normal_tangent_cos",
        "tke_mean_J_m3", "tke_peak_J_m3", "tke_p95_J_m3",
        "pressure_gradient_mag_mean_Pa_m", "pressure_gradient_mag_peak_Pa_m", "pressure_gradient_mag_p95_Pa_m",
        "pressure_gradient_normal_mean_Pa_m", "pressure_gradient_normal_peak_Pa_m", "pressure_gradient_normal_p95_Pa_m",
        "relative_pressure_mean_Pa", "relative_pressure_peak_Pa", "relative_pressure_p95_Pa",
        "wss_wall_mean_Pa", "wss_wall_peak_Pa", "wss_wall_p95_Pa",
    ]
    table_rows = []
    for i, m in enumerate(metrics):
        row = {"plane_index": i}
        for k in scalar_keys:
            if k in m:
                row[k] = m[k]
        row["center_x"] = m["center"][0] if "center" in m else None
        row["center_y"] = m["center"][1] if "center" in m else None
        row["center_z"] = m["center"][2] if "center" in m else None
        Nt = len(m.get("flowrate_mL_s", []))
        for t in range(Nt):
            row[f"flowrate_t{t}"] = m["flowrate_mL_s"][t]
        for t in range(len(m.get("area_mm2", []))):
            row[f"area_t{t}"] = m["area_mm2"][t]
        for t in range(len(m.get("meanv_cm_s_t", []))):
            row[f"meanv_t{t}"] = m["meanv_cm_s_t"][t]
        for t in range(len(m.get("flowrate_forward_mL_s", []))):
            row[f"flowrate_fwd_t{t}"] = m["flowrate_forward_mL_s"][t]
        for t in range(len(m.get("flowrate_reverse_mL_s", []))):
            row[f"flowrate_rev_t{t}"] = m["flowrate_reverse_mL_s"][t]
        for t in range(len(m.get("meanv_forward_cm_s_t", []))):
            row[f"meanv_fwd_t{t}"] = m["meanv_forward_cm_s_t"][t]
        for t in range(len(m.get("meanv_reverse_cm_s_t", []))):
            row[f"meanv_rev_t{t}"] = m["meanv_reverse_cm_s_t"][t]
        for t in range(len(m.get("flowrate_signed_mL_s", []))):
            row[f"flowrate_signed_t{t}"] = m["flowrate_signed_mL_s"][t]
        for t in range(len(m.get("meanv_signed_cm_s_t", []))):
            row[f"meanv_signed_t{t}"] = m["meanv_signed_cm_s_t"][t]
        fork_ic = m.get("fork_ic", [])
        for fi, fic in enumerate(fork_ic):
            row[f"fork{fi}_id"] = fic.get("fork_id", -1)
            row[f"fork{fi}_role"] = fic.get("role", "")
            row[f"fork{fi}_ic"] = fic.get("ic", 1.0)
        table_rows.append(row)
    qc_data = None
    if qc_json_path is not None:
        with open(qc_json_path, "r", encoding="utf-8") as f:
            qc_data = _json.load(f)
    return table_rows, metrics, qc_data
