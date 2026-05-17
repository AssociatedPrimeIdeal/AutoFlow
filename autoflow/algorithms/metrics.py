import numpy as np
import pyvista as pv

from .paths import _determine_plane_forward, _vector_orientation_text
from .surfaces import (
    _build_branch_grid,
    _extract_plane_flow_region,
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
    t2 = get_vector_magnitude(tang2)
    c = np.sum(tang1 * tang2, axis=1).clip(-1, 1)
    t2 = c * t2

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
    flow = np.asarray(flow_xyzt3, dtype=float)
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
                        inward_distance=0.6, parabolic_fitting=True,
                        no_slip_condition=True):
    mask4d = _ensure_mask4d(mask4d)
    flow = np.asarray(flow, dtype=float)
    if flow.ndim != 5 or flow.shape[-1] != 3:
        raise ValueError(f"flow must be XYZTV, got {flow.shape}")
    if flow.shape[3] != mask4d.shape[3]:
        raise ValueError(f"flow time dimension {flow.shape[3]} does not match mask {mask4d.shape[3]}")

    spacing = np.asarray(spacing, dtype=float).reshape(3)
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


def compute_derived_metrics(mask4d, flow, spacing, origin=(0, 0, 0),
                            smoothing_iteration=200, viscosity=4.0,
                            inward_distance=0.6, parabolic_fitting=True,
                            no_slip_condition=True, step_size=5,
                            tube_radius=0.1, rho=1060.0,
                            save_pixelwise=False, tke_array=None, sigma=None):
    mask4d = _ensure_mask4d(mask4d)
    wss = compute_wss_metrics(
        mask4d, flow, spacing, origin=origin,
        smoothing_iteration=smoothing_iteration,
        viscosity=viscosity,
        inward_distance=inward_distance,
        parabolic_fitting=parabolic_fitting,
        no_slip_condition=no_slip_condition,
    )
    tke = None
    if tke_array is not None or sigma is not None:
        tke = compute_tke_metrics(
            mask4d, spacing, origin=origin, tke_array=tke_array, sigma=sigma, rho=rho,
        )

    spacing = np.asarray(spacing, dtype=float).reshape(3)
    origin = np.asarray(origin, dtype=float).reshape(3)
    result = {
        "wss_surfaces": wss["wss_surfaces"],
        "wss_volume": wss["wss_volume"],
        "tke_volume": None if tke is None else tke["tke_volume"],
        "tke_array": None if tke is None else tke["tke_array"],
        "tke_peak": None if tke is None else tke["tke_peak"],
        "streamlines": [],
        "tube_radius": float(tube_radius),
    }
    if save_pixelwise:
        pixelwise_export = {
            "wss": np.asarray(wss["wss_volume"], dtype=np.float32),
            "spacing": np.asarray(spacing, dtype=np.float32),
            "origin": np.asarray(origin, dtype=np.float32),
        }
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
    flow = np.asarray(flow_xyzt3, dtype=float)
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
