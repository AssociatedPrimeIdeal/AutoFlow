import h5py
import numpy as np
import pyvista as pv
from scipy.ndimage import binary_erosion, gaussian_filter

from .paths import _determine_plane_forward, _vector_orientation_text
from .surfaces import (
    _build_branch_grid,
    _extract_plane_flow_region,
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
    mask_static = all(np.array_equal(mask[..., 0], mask[..., t]) for t in range(1, Nt))
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
             branch_grid, target_label, path_info, pp, mask_template)
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
                                    tke_array=None, pressure_gradient_array=None, wss_surfaces=None):
    mask4d = _ensure_mask4d(mask4d)
    spacing = np.asarray(spacing, dtype=float).reshape(3)
    origin = np.asarray(origin, dtype=float).reshape(3)
    Nt = int(mask4d.shape[3])
    branch_grid = _build_branch_grid(branch_labels_3d, spacing, origin)
    target_label = _target_label_for_plane(plane, branch_labels_3d, spacing, origin)
    normal = np.asarray(plane.normal, dtype=float).reshape(3)
    normal = normal / (np.linalg.norm(normal) + 1e-12)

    summary = {}
    pixelwise = {"timepoints": []}

    has_tke = tke_array is not None
    has_pressure_gradient = pressure_gradient_array is not None
    has_wss = wss_surfaces is not None

    if has_tke:
        tke_array = _prepare_tke_array(mask4d, tke_array=tke_array, sigma=None)
    if has_pressure_gradient:
        pressure_gradient_array = np.asarray(pressure_gradient_array, dtype=np.float32)
        if pressure_gradient_array.ndim != 5 or pressure_gradient_array.shape[-1] != 3:
            raise ValueError(f"pressure_gradient_array must be XYZTV, got {pressure_gradient_array.shape}")

    tke_mean_t = []
    tke_peak_t = []
    tke_p95_t = []
    pg_mag_mean_t = []
    pg_mag_peak_t = []
    pg_mag_p95_t = []
    pg_normal_mean_t = []
    pg_normal_peak_t = []
    pg_normal_p95_t = []
    wss_mean_t = []
    wss_peak_t = []
    wss_p95_t = []

    for tidx in range(Nt):
        mask_t = mask4d[..., tidx]
        entry = {"time_index": int(tidx)}

        tke_series_vals = np.array([], dtype=float)
        pg_mag_vals = np.array([], dtype=float)
        pg_normal_vals = np.array([], dtype=float)
        areas = np.array([], dtype=float)

        if has_tke:
            pg_tke = _extract_plane_field_region(
                mask_t, tke_array[..., tidx], plane, spacing, origin, "tke",
                branch_grid=branch_grid, target_label=target_label,
            )
            if pg_tke is not None and pg_tke.n_cells > 0:
                if "tke" not in pg_tke.cell_data and "tke" in pg_tke.point_data:
                    pg_tke = pg_tke.point_data_to_cell_data(pass_point_data=True)
                tke_series_vals = np.asarray(pg_tke.cell_data.get("tke", []), dtype=float).reshape(-1)
                areas = np.asarray(pg_tke.cell_data.get("Area", np.ones(pg_tke.n_cells, dtype=float)), dtype=float).reshape(-1)
                entry["cell_area_mm2"] = areas.astype(np.float32)
                entry["tke_J_m3"] = tke_series_vals.astype(np.float32)
                entry["lumen_mask"] = np.ones_like(tke_series_vals, dtype=np.uint8)

        if has_pressure_gradient:
            pg_pg = _extract_plane_field_region(
                mask_t, pressure_gradient_array[..., tidx, :], plane, spacing, origin, "pressure_gradient",
                branch_grid=branch_grid, target_label=target_label,
            )
            if pg_pg is not None and pg_pg.n_cells > 0:
                if "pressure_gradient" not in pg_pg.cell_data and "pressure_gradient" in pg_pg.point_data:
                    pg_pg = pg_pg.point_data_to_cell_data(pass_point_data=True)
                vec = np.asarray(pg_pg.cell_data.get("pressure_gradient", []), dtype=float)
                if vec.ndim == 2 and vec.shape[1] == 3 and len(vec) == pg_pg.n_cells:
                    if areas.size == 0:
                        areas = np.asarray(pg_pg.cell_data.get("Area", np.ones(pg_pg.n_cells, dtype=float)), dtype=float).reshape(-1)
                        entry.setdefault("cell_area_mm2", areas.astype(np.float32))
                        entry.setdefault("lumen_mask", np.ones(len(areas), dtype=np.uint8))
                    pg_mag_vals = np.linalg.norm(vec, axis=1)
                    pg_normal_vals = np.dot(vec, normal)
                    entry["pressure_gradient_mag_Pa_m"] = pg_mag_vals.astype(np.float32)
                    entry["pressure_gradient_normal_Pa_m"] = pg_normal_vals.astype(np.float32)
                    entry["pressure_gradient_vec_Pa_m"] = vec.astype(np.float32)

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
    if has_wss:
        _append_summary(summary, "wss_wall_mean_Pa", wss_mean_t)
        _append_summary(summary, "wss_wall_peak_Pa", wss_peak_t)
        _append_summary(summary, "wss_wall_p95_Pa", wss_p95_t)
    return summary, pixelwise


def augment_plane_metrics_with_derived(plane_metrics, planes, mask4d, spacing, origin, branch_labels_3d=None,
                                       tke_array=None, pressure_gradient_array=None, wss_surfaces=None):
    metrics = [dict(m) for m in plane_metrics]
    pixelwise = []
    for idx, metric in enumerate(metrics):
        if idx >= len(planes):
            pixelwise.append({"plane_index": int(idx), "timepoints": []})
            continue
        summary, payload = summarize_plane_derived_metrics(
            planes[idx], mask4d, spacing, origin, branch_labels_3d=branch_labels_3d,
            tke_array=tke_array, pressure_gradient_array=pressure_gradient_array,
            wss_surfaces=wss_surfaces,
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


def compute_pressure_gradient_metrics(mask4d, flow, spacing, rr, rho=1060.0, viscosity=4.0,
                                      smoothing_sigma=0.0, use_convective_acceleration=True):
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
    support_mask = np.zeros(mask4d.shape, dtype=bool)
    for tidx in range(mask4d.shape[3]):
        support_mask[..., tidx] = binary_erosion(mask4d[..., tidx], structure=np.ones((3, 3, 3), dtype=bool), border_value=0)
    support_inner = support_mask[1:-1, 1:-1, 1:-1, 1:-1]

    grad = np.zeros(flow.shape, dtype=np.float32)
    grad[1:-1, 1:-1, 1:-1, 1:-1, :] = grad_inner.astype(np.float32)
    grad *= support_mask.astype(np.float32)[..., None]
    grad_mag = np.sqrt(np.sum(np.square(grad, dtype=np.float32), axis=-1)).astype(np.float32)
    grad_peak = np.max(grad_mag, axis=3).astype(np.float32)

    finite_inner = grad_inner[np.isfinite(grad_inner) & support_inner[..., None]]
    display_upper = float(np.percentile(np.abs(finite_inner), 99.0)) if finite_inner.size else 0.0

    return {
        'pressure_gradient_array': grad,
        'pressure_gradient_magnitude': grad_mag,
        'pressure_gradient_peak': grad_peak,
        'pressure_gradient_dt_s': float(dt_s),
        'pressure_gradient_support_mask': support_mask.astype(np.uint8),
        'pressure_gradient_display_clim': (0.0, display_upper if display_upper > 0 else 1.0),
    }


def compute_derived_metrics(mask4d, flow, spacing, origin=(0, 0, 0),
                            smoothing_iteration=200, viscosity=4.0,
                            inward_distance=None, parabolic_fitting=True,
                            no_slip_condition=False, step_size=5,
                            tube_radius=0.1, rho=1060.0,
                            save_pixelwise=False, tke_array=None, sigma=None,
                            rr=1000.0, pressure_gradient_smoothing_sigma=0.0,
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
                            pressure_gradient_viscosity=None):
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
            use_convective_acceleration=pressure_gradient_use_convective_acceleration,
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
        if tke is not None:
            pixelwise_export["tke"] = np.asarray(tke["tke_peak"], dtype=np.float32)
            pixelwise_export["tke_time"] = np.asarray(tke["tke_array"], dtype=np.float32)
        result["pixelwise_export"] = pixelwise_export
    else:
        result["pixelwise_export"] = {}
    return result


def _compute_single_plane_metric(args):
    (flow, mask, spacing, origin, plane, Nt, RR, branch_grid, target_label,
     path_info, path_points, mask_template) = args
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

    for t in range(Nt):
        mask_t = mask_template if mask_template is not None else mask[..., t]
        pg = _extract_plane_flow_region(
            mask_t, flow[..., t, :], plane, spacing, origin,
            branch_grid=branch_grid, target_label=target_label,
        )
        if pg is None or pg.n_cells == 0:
            flowrate.append(0.0); flowrate_fwd.append(0.0); flowrate_rev.append(0.0)
            meanv_t.append(0.0); meanv_fwd_t.append(0.0); meanv_rev_t.append(0.0)
            area.append(0.0)
            continue
        if "flow" not in pg.cell_data and "flow" in pg.point_data:
            pg = pg.point_data_to_cell_data(pass_point_data=True)
        vec = np.asarray(pg.cell_data.get("flow", []), dtype=float)
        if vec.ndim != 2 or vec.shape[1] != 3 or len(vec) != pg.n_cells:
            ca0 = np.asarray(pg.cell_data.get("Area", []), dtype=float)
            flowrate.append(0.0); flowrate_fwd.append(0.0); flowrate_rev.append(0.0)
            meanv_t.append(0.0); meanv_fwd_t.append(0.0); meanv_rev_t.append(0.0)
            area.append(float(np.sum(ca0)) if len(ca0) else 0.0)
            continue
        ca = np.asarray(pg.cell_data.get("Area", np.ones(pg.n_cells, dtype=float)), dtype=float).reshape(-1)
        if len(ca) != len(vec):
            ca = np.ones(len(vec), dtype=float)

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
    mask_static = all(np.array_equal(mask[..., 0], mask[..., t]) for t in range(1, Nt))
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
                          branch_grid, target_label, path_info, pp, mask_template))
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
