from typing import Any, Dict, Optional, Tuple

import numpy as np
from skimage import filters

from ..case_types import BackgroundPhaseCorrectionConfig


def _emit_progress(progress_callback, stage, current=None, total=None, message=""):
    if progress_callback is None:
        return
    payload = {
        "stage": str(stage),
        "current": None if current is None else int(current),
        "total": None if total is None else int(total),
        "message": str(message or ""),
    }
    try:
        progress_callback(payload)
    except Exception:
        pass


def coerce_background_phase_correction_config(config=None):
    if config is None:
        return BackgroundPhaseCorrectionConfig()
    if isinstance(config, BackgroundPhaseCorrectionConfig):
        return BackgroundPhaseCorrectionConfig.from_dict(config.to_dict())
    if isinstance(config, dict):
        return BackgroundPhaseCorrectionConfig.from_dict(config)
    raise TypeError(f"unsupported background phase correction config: {type(config)!r}")


def background_phase_report_for_metadata(report):
    payload = dict(report or {})
    payload.pop("stationary_voxels", None)
    payload.pop("corr", None)
    return payload


def _min_samples_for_order(run_4d, order):
    if run_4d:
        return {0: 1, 1: 4, 2: 10, 3: 20}.get(int(order), 4)
    return {0: 1, 1: 3, 2: 6, 3: 10}.get(int(order), 3)


def _magnitude_threshold(magnitude):
    values = np.asarray(magnitude, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    if np.max(values) <= 0:
        return float(np.max(values))
    unique_count = int(np.unique(np.round(values, decimals=6)).size)
    try:
        classes = min(5, max(2, unique_count))
        return float(filters.threshold_multiotsu(values, classes=classes)[0])
    except Exception:
        try:
            return float(filters.threshold_otsu(values))
        except Exception:
            return float(np.quantile(values, 0.6))


def _ensure_mag_flow_time(flow, mag):
    flow = np.asarray(flow, dtype=np.float32)
    mag = np.asarray(mag, dtype=np.float32)
    if flow.ndim == 4 and flow.shape[-1] == 3:
        flow = flow[..., np.newaxis, :]
    if flow.ndim != 5 or flow.shape[-1] != 3:
        raise ValueError(f"flow must be XYZTV or XYZV with 3 components, got {flow.shape}")
    nt = int(flow.shape[3])
    if mag.ndim == 3:
        mag = np.repeat(mag[..., np.newaxis], nt, axis=3)
    elif mag.ndim == 4 and mag.shape[3] == 1 and nt > 1:
        mag = np.repeat(mag, nt, axis=3)
    elif mag.ndim != 4:
        raise ValueError(f"mag must be XYZT or XYZ, got {mag.shape}")
    if mag.shape[3] != nt:
        if mag.shape[3] == 1:
            mag = np.repeat(mag, nt, axis=3)
        else:
            raise ValueError(f"mag time dimension {mag.shape[3]} does not match flow {nt}")
    return (
        np.ascontiguousarray(flow, dtype=np.float32),
        np.ascontiguousarray(mag, dtype=np.float32),
    )


def _zero_corr_nvtzyx(shape):
    if len(shape) != 5:
        return np.zeros((3, 1, 1, 1, 1), dtype=np.float32)
    return np.zeros((3, int(shape[1]), int(shape[2]), int(shape[3]), int(shape[4])), dtype=np.float32)


def apply_background_phase_correction_to_complex(
    img_complex,
    config=None,
    progress_callback=None,
    source_mode="complex_source",
    cached_corr=None,
):
    cfg = coerce_background_phase_correction_config(config)
    arr = np.asarray(img_complex)
    report = {
        "enabled": bool(cfg.enabled),
        "applied": False,
        "source_mode": str(source_mode),
        "corr_fit_order": int(cfg.corr_fit_order),
        "threshold": float(cfg.threshold),
        "stationary_voxels": 0,
        "skipped_reason": "",
        "cache_hit": False,
        "cache_reason": "missing",
    }
    if not cfg.enabled:
        report["skipped_reason"] = "disabled"
        return arr.copy(), None, report
    if arr.ndim != 5 or arr.shape[-1] < 4:
        report["skipped_reason"] = f"expected complex XYZT4+ input, got shape={arr.shape}"
        return arr.copy(), None, report

    corr_xyzt3 = None
    stationary_mask = None
    if cached_corr is not None:
        try:
            corr_xyzt3 = np.asarray(cached_corr, dtype=np.float32)
            expected_shape = tuple(arr.shape[:-1]) + (3,)
            if corr_xyzt3.ndim == len(expected_shape) - 1 and corr_xyzt3.shape[-1] == 3:
                corr_xyzt3 = corr_xyzt3[..., np.newaxis, :]
            singleton_time_match = (
                corr_xyzt3.ndim == 5
                and corr_xyzt3.shape[:3] == expected_shape[:3]
                and corr_xyzt3.shape[3] == 1
                and corr_xyzt3.shape[4] == expected_shape[4]
            )
            if corr_xyzt3.shape != expected_shape and not singleton_time_match:
                raise ValueError(f"cached corr shape {corr_xyzt3.shape} does not match expected {expected_shape}")
            report["cache_hit"] = True
            report["cache_reason"] = "hit"
            report["corr_source"] = "cache"
        except Exception as exc:
            corr_xyzt3 = None
            report["cache_hit"] = False
            report["cache_reason"] = f"invalid_cache:{type(exc).__name__}: {exc}"

    if corr_xyzt3 is None:
        _emit_progress(progress_callback, "background_phase_start", message="Running background phase correction")
        try:
            corr_nvtzyx, stationary_mask, diag = execute_msac(
                np.transpose(np.asarray(arr[..., :4], dtype=np.complex64), (4, 3, 2, 1, 0)),
                corr_fit_order=int(cfg.corr_fit_order),
                th=float(cfg.threshold),
            )
        except Exception as exc:
            report["skipped_reason"] = f"{type(exc).__name__}: {exc}"
            return arr.copy(), None, report

        report.update(diag)
        report["corr_source"] = "computed"
        if not bool(diag.get("applied", False)):
            return arr.copy(), None, report

        corr_xyzt3 = np.transpose(corr_nvtzyx, (4, 3, 2, 1, 0))
        report["stationary_voxels"] = int(np.sum(stationary_mask))
    else:
        _emit_progress(progress_callback, "background_phase_start", message="Reusing cached background phase correction")

    corrected = np.asarray(arr, dtype=np.complex64).copy()
    corrected[..., 1:4] *= np.exp(-1j * np.asarray(corr_xyzt3, dtype=np.float32))
    corrected = corrected.astype(arr.dtype, copy=False)
    report["applied"] = True
    report["corr_algorithm"] = "msac"
    report["corr_version"] = 1
    report["corr_components"] = int(corr_xyzt3.shape[-1])
    report["corr"] = np.asarray(corr_xyzt3, dtype=np.float32)
    _emit_progress(
        progress_callback,
        "background_phase_done",
        message="Background phase correction reused from cache" if report.get("cache_hit", False) else "Background phase correction applied",
    )
    return corrected, (stationary_mask.astype(bool) if stationary_mask is not None else None), report


def apply_background_phase_correction_to_mag_flow(
    mag,
    flow,
    venc,
    config=None,
    progress_callback=None,
    cached_corr=None,
):
    cfg = coerce_background_phase_correction_config(config)
    flow_xyzt3, mag_xyzt = _ensure_mag_flow_time(flow, mag)
    venc_arr = np.asarray(venc, dtype=np.float32).reshape(-1)
    if venc_arr.size == 1:
        venc_arr = np.repeat(venc_arr, 3)
    if venc_arr.size != 3:
        raise ValueError(f"venc must have 1 or 3 entries, got {venc_arr.shape}")

    img_complex = np.zeros(flow_xyzt3.shape[:4] + (4,), dtype=np.complex64)
    ref = np.asarray(mag_xyzt, dtype=np.float32)
    img_complex[..., 0] = ref.astype(np.complex64)
    phase = np.pi * flow_xyzt3 / venc_arr.reshape((1, 1, 1, 1, 3))
    img_complex[..., 1:4] = ref[..., None] * np.exp(1j * phase)

    corrected_complex, stationary_mask, report = apply_background_phase_correction_to_complex(
        img_complex,
        config=cfg,
        progress_callback=progress_callback,
        source_mode="synthetic_complex",
        cached_corr=cached_corr,
    )
    if not bool(report.get("applied", False)):
        return flow_xyzt3.copy(), stationary_mask, report

    corrected_phase = np.angle(corrected_complex[..., 1:4] * np.conj(corrected_complex[..., 0:1]))
    corrected_flow = (corrected_phase / np.pi) * venc_arr.reshape((1, 1, 1, 1, 3))
    return np.asarray(corrected_flow, dtype=np.float32), stationary_mask, report


def execute_msac(im, corr_fit_order=3, th=0.1):
    np.random.seed(274612)
    raw = np.asarray(im)
    zero_corr = _zero_corr_nvtzyx(raw.shape)
    if raw.ndim != 5:
        diag = {
            "applied": False,
            "corr_fit_order": int(corr_fit_order),
            "threshold": float(th),
            "stationary_voxels": 0,
            "skipped_reason": f"expected NVTZYX with 5 dims, got shape={raw.shape}",
        }
        return zero_corr, np.zeros((1, 1, 1), dtype=bool), diag
    im = np.transpose(raw, (4, 3, 2, 1, 0))
    diag = {
        "applied": False,
        "corr_fit_order": int(corr_fit_order),
        "threshold": float(th),
        "stationary_voxels": 0,
        "skipped_reason": "",
    }

    if im.ndim != 5 or im.shape[-1] < 4:
        diag["skipped_reason"] = f"expected NVTZYX with at least 4 encodes, got shape={im.shape}"
        return zero_corr, np.zeros(im.shape[:3], dtype=bool), diag

    im = im[..., 1:4] * np.conj(im[..., 0:1]) / (np.abs(im[..., 0:1]) + 1e-9)
    phase_im_t = np.angle(im) / np.pi
    magnitude_im_t = np.abs(im)
    magnitude_im_t = np.mean(magnitude_im_t, axis=-1)
    mag_max = float(np.max(magnitude_im_t)) if magnitude_im_t.size else 0.0
    if not np.isfinite(mag_max) or mag_max <= 1e-12:
        diag["skipped_reason"] = "empty magnitude image"
        return zero_corr, np.zeros(phase_im_t.shape[:3], dtype=bool), diag
    magnitude_im_t = magnitude_im_t / mag_max

    magnitude = np.mean(magnitude_im_t, axis=3)
    phase = np.mean(phase_im_t, axis=3)
    threshold = _magnitude_threshold(magnitude)
    mask_mgn = magnitude > threshold
    m, n, l, _t, d = phase_im_t.shape
    run_4d = d == 3
    stationary_mask = np.zeros((m, n, l), dtype=bool)

    if not np.any(mask_mgn):
        diag["skipped_reason"] = "no magnitude mask candidates"
        return zero_corr, stationary_mask, diag

    if run_4d:
        allm = np.ones((m, n, l), dtype=bool)
        aa = np.argwhere(allm)
        ma = np.argwhere(mask_mgn)

        ap = np.zeros((aa.shape[0], 6), dtype=np.float32)
        mp = np.zeros((ma.shape[0], 6), dtype=np.float32)
        ap[:, 3:6] = aa
        mp[:, 3:6] = ma
        for direction in np.arange(d):
            pdir = phase[:, :, :, direction]
            ap[:, direction] = pdir[allm]
            mp[:, direction] = pdir[mask_mgn]
    else:
        allm = np.ones((m, n), dtype=bool)
        aa = np.argwhere(allm)
        ma = np.argwhere(mask_mgn[:, :, 0])

        ap = np.zeros((aa.shape[0], 3), dtype=np.float32)
        mp = np.zeros((ma.shape[0], 3), dtype=np.float32)
        ap[:, 1:3] = aa
        mp[:, 1:3] = ma
        ap[:, 0] = np.squeeze(phase[allm])
        mp[:, 0] = np.squeeze(phase[mask_mgn[:, :, 0]])

    min_samples = _min_samples_for_order(run_4d, 1)
    if mp.shape[0] < min_samples:
        diag["skipped_reason"] = f"not enough stationary candidates: {mp.shape[0]} < {min_samples}"
        return zero_corr, stationary_mask, diag

    parameters = {
        "msac_thresh": float(th),
        "samples": int(min(10, mp.shape[0])),
        "trials": 100,
        "msac_fit_order": 1,
        "n_enc": int(im.shape[-1]),
    }

    mfunc, cfunc, _fmfunc, fcfunc, dfunc = get_functions(run_4d, parameters["msac_fit_order"], corr_fit_order)
    functions = {
        "msac_fit": mfunc,
        "msac_dist": dfunc,
    }
    _cost, inlier_idx = msac(mp, parameters, functions)

    model = cfunc(mp, **{"inlierIndx": inlier_idx})
    est, _xyz = fcfunc(model, ap)

    if run_4d:
        keep = np.all(np.asarray(inlier_idx, dtype=bool), axis=1)
        if np.any(keep):
            stationary_mask[ma[keep, 0], ma[keep, 1], ma[keep, 2]] = True
    else:
        keep = np.asarray(inlier_idx, dtype=bool)[:, 0]
        if np.any(keep):
            stationary_mask[ma[keep, 0], ma[keep, 1], 0] = True

    bgr_msac = np.zeros((m, n, l, d), dtype=np.float32)
    aw = np.where(allm)
    if run_4d:
        bgr_msac[aw[0], aw[1], aw[2], :] = est
    else:
        bgr_msac[aw[0], aw[1], 0, :] = est[:, :, np.newaxis]

    corr_t = np.expand_dims(bgr_msac, -1)
    corr_t = np.swapaxes(corr_t, 3, 4)
    corr_t = np.transpose(corr_t, (4, 3, 2, 1, 0))

    diag["applied"] = True
    diag["stationary_voxels"] = int(np.sum(stationary_mask))
    return np.asarray(corr_t * np.pi, dtype=np.float32), stationary_mask, diag


def get_functions(run_4d, fitorder_msac, fitorder_corr):
    if run_4d:
        mfunc = lambda points, **kwargs: fit4d(fitorder_msac, points, **kwargs)
        cfunc = lambda points, **kwargs: fit4d(fitorder_corr, points, **kwargs)
        fmfunc = lambda coeffs, points: eval4d(fitorder_msac, coeffs, points)
        fcfunc = lambda coeffs, points: eval4d(fitorder_corr, coeffs, points)
        dfunc = lambda coeffs, points: dist4d(fitorder_msac, coeffs, points)
    else:
        mfunc = lambda points, **kwargs: fit2d(fitorder_msac, points, **kwargs)
        cfunc = lambda points, **kwargs: fit2d(fitorder_corr, points, **kwargs)
        fmfunc = lambda coeffs, points: eval2d(fitorder_msac, coeffs, points)
        fcfunc = lambda coeffs, points: eval2d(fitorder_corr, coeffs, points)
        dfunc = lambda coeffs, points: dist2d(fitorder_msac, coeffs, points)
    return mfunc, cfunc, fmfunc, fcfunc, dfunc


def get_in_out_4d(order, points):
    xyz = points[:, 0:3]
    p1 = points[:, 3]
    p2 = points[:, 4]
    p3 = points[:, 5]
    no_p = points.shape[0]
    if order == 0:
        a = np.ones((no_p, 1), dtype=np.float32)
        no = 1
    elif order == 1:
        a = np.ones((no_p, 4), dtype=np.float32)
        a[:, 1] = p1
        a[:, 2] = p2
        a[:, 3] = p3
        no = 4
    elif order == 2:
        a = np.ones((no_p, 10), dtype=np.float32)
        a[:, 1] = p1
        a[:, 2] = p2
        a[:, 3] = p3
        a[:, 4] = p1 ** 2
        a[:, 5] = p1 * p2
        a[:, 6] = p1 * p3
        a[:, 7] = p2 ** 2
        a[:, 8] = p2 * p3
        a[:, 9] = p3 ** 2
        no = 10
    else:
        a = np.ones((no_p, 20), dtype=np.float32)
        a[:, 1] = p1
        a[:, 2] = p2
        a[:, 3] = p3
        a[:, 4] = p1 ** 2
        a[:, 5] = p1 * p2
        a[:, 6] = p1 * p3
        a[:, 7] = p2 ** 2
        a[:, 8] = p2 * p3
        a[:, 9] = p3 ** 2
        a[:, 10] = p1 ** 3
        a[:, 11] = (p1 ** 2) * p2
        a[:, 12] = (p1 ** 2) * p3
        a[:, 13] = p2 ** 3
        a[:, 14] = (p2 ** 2) * p1
        a[:, 15] = (p2 ** 2) * p3
        a[:, 16] = p3 ** 3
        a[:, 17] = (p3 ** 2) * p1
        a[:, 18] = (p3 ** 2) * p2
        a[:, 19] = p1 * p2 * p3
        no = 20
    return xyz, a, no


def fit4d(order, points, **kwargs):
    inlier_idx = np.ones((points.shape[0], 3), dtype=bool)
    if kwargs:
        inlier_idx = kwargs["inlierIndx"]

    indx = inlier_idx == 1
    xyz, a, no = get_in_out_4d(order, points)
    coeffs = np.zeros((no, 3), dtype=np.float32)
    for direction in np.arange(3):
        if order == 0:
            coeffs[0, direction] = np.mean(xyz[indx[:, direction], direction], 0)
        else:
            inlier = indx[:, direction]
            coeffs[:, direction] = np.linalg.lstsq(a[inlier, :], xyz[inlier, direction], rcond=None)[0]
    return coeffs


def dist4d(order, coeffs, points):
    est, xyz = eval4d(order, coeffs, points)
    return np.abs((xyz - est) / 2)


def eval4d(order, coeffs, points):
    xyz, a, _no = get_in_out_4d(order, points)
    est = a @ coeffs
    return est, xyz


def get_in_out_2d(order, points):
    xyz = points[:, 0:1]
    p1 = points[:, 1]
    p2 = points[:, 2]
    no_p = points.shape[0]
    if order == 0:
        a = np.ones((no_p, 1), dtype=np.float32)
        no = 1
    elif order == 1:
        a = np.ones((no_p, 3), dtype=np.float32)
        a[:, 1] = p1
        a[:, 2] = p2
        no = 3
    elif order == 2:
        a = np.ones((no_p, 6), dtype=np.float32)
        a[:, 1] = p1
        a[:, 2] = p2
        a[:, 3] = p1 ** 2
        a[:, 4] = p1 * p2
        a[:, 5] = p2 ** 2
        no = 6
    else:
        a = np.ones((no_p, 10), dtype=np.float32)
        a[:, 1] = p1
        a[:, 2] = p2
        a[:, 3] = p1 ** 2
        a[:, 4] = p1 * p2
        a[:, 5] = p2 ** 2
        a[:, 6] = p1 ** 3
        a[:, 7] = (p1 ** 2) * p2
        a[:, 8] = p2 ** 3
        a[:, 9] = (p2 ** 2) * p1
        no = 10
    return xyz, a, no


def fit2d(order, points, **kwargs):
    inlier_idx = np.ones((points.shape[0], 1), dtype=bool)
    if kwargs:
        inlier_idx = kwargs["inlierIndx"]

    indx = (inlier_idx == 1)[:, 0]
    xyz, a, no = get_in_out_2d(order, points)
    coeffs = np.zeros((no, 1), dtype=np.float32)
    if order == 0:
        coeffs[0, 0] = np.mean(xyz[indx, 0], 0)
    else:
        coeffs[:, 0] = np.linalg.lstsq(a[indx, :], xyz[indx, 0], rcond=None)[0]
    return coeffs


def dist2d(order, coeffs, points):
    est, xyz = eval2d(order, coeffs, points)
    return np.abs((xyz - est) / 2)


def eval2d(order, coeffs, points):
    xyz, a, _no = get_in_out_2d(order, points)
    est = a @ coeffs
    return est, xyz


def msac(points, parameters, functions):
    samples = parameters["samples"]
    threshold = parameters["msac_thresh"]
    trials = parameters["trials"]
    n_enc = parameters["n_enc"]

    msac_fit = functions["msac_fit"]
    msac_dist = functions["msac_dist"]

    no_p = points.shape[0]
    best_cost = np.ones(n_enc, dtype=np.float32) * threshold * no_p
    best_inliers = np.zeros((no_p, n_enc), dtype=bool)

    for _ in np.arange(trials):
        indx = np.random.permutation(no_p)[0:samples]
        sample = points[indx, :]

        coeffs = msac_fit(sample)
        residuals = np.asarray(msac_dist(coeffs, points), dtype=np.float32)

        residuals[residuals > threshold] = threshold
        inliers = residuals < threshold
        cost = np.sum(residuals, 0)
        comp_cost = best_cost > cost

        best_cost[comp_cost] = cost[comp_cost]
        best_inliers[:, comp_cost] = inliers[:, comp_cost]

    return best_cost, best_inliers
