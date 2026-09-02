import threading
from typing import Any, Dict, Optional, Tuple

import numpy as np
from skimage import exposure, filters

from ..case_types import BackgroundPhaseCorrectionConfig


_CORRECTION_ALGORITHM_VERSION = {"msac": 1, "wrls_arto": 1}
_WRLS_BASIS_CACHE = {}
_WRLS_BASIS_CACHE_LOCK = threading.Lock()
_TORCH_STATE = {"checked": False, "module": None, "reason": ""}
_TORCH_STATE_LOCK = threading.Lock()


def _available_torch_cuda():
    with _TORCH_STATE_LOCK:
        if not _TORCH_STATE["checked"]:
            try:
                import torch

                if not bool(torch.cuda.is_available()):
                    raise RuntimeError("no CUDA device")
                probe = torch.arange(1, dtype=torch.float32, device="cuda") + 1.0
                probe.cpu()
                _TORCH_STATE.update({"checked": True, "module": torch, "reason": ""})
            except Exception as exc:
                _TORCH_STATE.update({
                    "checked": True,
                    "module": None,
                    "reason": f"{type(exc).__name__}: {exc}",
                })
        return _TORCH_STATE["module"], str(_TORCH_STATE["reason"])


def _normalize_correction_method(method):
    value = str(method or "wrls_arto").strip().lower().replace("-", "_").replace("+", "_")
    aliases = {"wrls": "wrls_arto", "arto": "wrls_arto", "wrlsarto": "wrls_arto"}
    value = aliases.get(value, value)
    if value not in _CORRECTION_ALGORITHM_VERSION:
        raise ValueError(f"unsupported background phase correction method: {method!r}")
    return value


def background_phase_correction_cache_metadata(config=None):
    cfg = coerce_background_phase_correction_config(config)
    method = _normalize_correction_method(cfg.method)
    metadata = {
        "corr_algorithm": method,
        "corr_version": int(_CORRECTION_ALGORITHM_VERSION[method]),
        "corr_fit_order": int(cfg.corr_fit_order),
    }
    if method == "msac":
        metadata["corr_threshold"] = float(cfg.threshold)
    else:
        metadata.update({
            "corr_wrls_lambda": float(cfg.wrls_lambda),
            "corr_wrls_magnitude_threshold": float(cfg.wrls_magnitude_threshold),
            "corr_wrls_mid_fov_fraction": float(cfg.wrls_mid_fov_fraction),
            "corr_wrls_mid_slice_fraction": float(cfg.wrls_mid_slice_fraction),
            "corr_wrls_arto_iterations": int(cfg.wrls_arto_iterations),
            "corr_wrls_tau": float(cfg.wrls_tau),
            "corr_wrls_delta": float(cfg.wrls_delta),
            "corr_wrls_central_probability": float(cfg.wrls_central_probability),
            "corr_wrls_fista_iterations": int(cfg.wrls_fista_iterations),
            "corr_wrls_gmm_iterations": int(cfg.wrls_gmm_iterations),
        })
    return metadata


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
        if classes <= 2:
            return float(filters.threshold_otsu(values))
        probability, bin_centers = exposure.histogram(
            values,
            nbins=256,
            source_range="image",
            normalize=True,
        )
        probability = np.asarray(probability, dtype=np.float32)
        nonzero_count = int(np.count_nonzero(probability))
        classes = min(classes, nonzero_count)
        if classes <= 2:
            return float(filters.threshold_otsu(values))

        level_count = int(probability.size)
        levels = np.arange(level_count, dtype=np.float32)
        cumulative_probability = np.concatenate([
            np.zeros(1, dtype=np.float32),
            np.cumsum(probability, dtype=np.float32),
        ])
        cumulative_moment = np.concatenate([
            np.zeros(1, dtype=np.float32),
            np.cumsum(probability * levels, dtype=np.float32),
        ])
        class_score = np.zeros((level_count, level_count), dtype=np.float32)
        for start in range(level_count):
            mass = cumulative_probability[start + 1:] - cumulative_probability[start]
            moment = cumulative_moment[start + 1:] - cumulative_moment[start]
            class_score[start, start:] = np.divide(
                moment * moment,
                mass,
                out=np.zeros_like(mass),
                where=mass > 0.0,
            )

        score = np.full((classes + 1, level_count), -np.inf, dtype=np.float32)
        split = np.full((classes + 1, level_count), -1, dtype=np.int16)
        score[1] = class_score[0]
        for class_count in range(2, classes + 1):
            for end in range(class_count - 1, level_count):
                starts = np.arange(class_count - 1, end + 1)
                candidates = score[class_count - 1, starts - 1] + class_score[starts, end]
                best = int(np.argmax(candidates))
                score[class_count, end] = candidates[best]
                split[class_count, end] = int(starts[best])

        threshold_indices = []
        end = level_count - 1
        for class_count in range(classes, 1, -1):
            start = int(split[class_count, end])
            threshold_indices.append(start - 1)
            end = start - 1
        threshold_indices.reverse()
        return float(np.asarray(bin_centers)[threshold_indices[0]])
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
    method = _normalize_correction_method(cfg.method)
    arr = np.asarray(img_complex)
    report = {
        "enabled": bool(cfg.enabled),
        "applied": False,
        "source_mode": str(source_mode),
        "corr_algorithm": method,
        "corr_version": int(_CORRECTION_ALGORITHM_VERSION[method]),
        "corr_fit_order": int(cfg.corr_fit_order),
        "threshold": float(cfg.threshold),
        "stationary_voxels": 0,
        "skipped_reason": "",
        "cache_hit": False,
        "cache_reason": "missing",
    }
    if not cfg.enabled:
        report["skipped_reason"] = "disabled"
        return arr, None, report
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
        _emit_progress(
            progress_callback,
            "background_phase_start",
            message=f"Running {method} background phase correction",
        )
        try:
            algorithm_input = np.transpose(
                np.asarray(arr[..., :4], dtype=np.complex64),
                (4, 3, 2, 1, 0),
            )
            if method == "msac":
                corr_nvtzyx, stationary_mask, diag = execute_msac(
                    algorithm_input,
                    corr_fit_order=int(cfg.corr_fit_order),
                    th=float(cfg.threshold),
                    progress_callback=progress_callback,
                )
            else:
                corr_nvtzyx, stationary_mask, diag = execute_wrls_arto(
                    algorithm_input,
                    corr_fit_order=int(cfg.corr_fit_order),
                    lam=float(cfg.wrls_lambda),
                    magnitude_threshold=float(cfg.wrls_magnitude_threshold),
                    mid_fov_fraction=float(cfg.wrls_mid_fov_fraction),
                    mid_slice_fraction=float(cfg.wrls_mid_slice_fraction),
                    arto_iterations=int(cfg.wrls_arto_iterations),
                    tau=float(cfg.wrls_tau),
                    delta=float(cfg.wrls_delta),
                    central_probability=float(cfg.wrls_central_probability),
                    fista_iterations=int(cfg.wrls_fista_iterations),
                    gmm_iterations=int(cfg.wrls_gmm_iterations),
                    progress_callback=progress_callback,
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
    report.update(background_phase_correction_cache_metadata(cfg))
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
    if not cfg.enabled:
        metadata = background_phase_correction_cache_metadata(cfg)
        return flow_xyzt3, None, {
            "enabled": False,
            "applied": False,
            "source_mode": "synthetic_complex",
            **metadata,
            "corr_fit_order": int(cfg.corr_fit_order),
            "threshold": float(cfg.threshold),
            "stationary_voxels": 0,
            "skipped_reason": "disabled",
            "cache_hit": False,
            "cache_reason": "missing",
        }
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


def _polynomial_exponents_3d(order):
    order = int(order)
    if order < 0 or order > 4:
        raise ValueError(f"WRLS+ARTO supports 3D polynomial orders 0 through 4, got {order}")
    exponents = []
    for degree in range(order + 1):
        for z_power in range(degree + 1):
            for y_power in range(degree - z_power + 1):
                x_power = degree - y_power - z_power
                exponents.append((x_power, y_power, z_power))
    return tuple(exponents)


def _build_normalized_polynomial_basis(shape, order):
    shape = tuple(int(value) for value in shape)
    requested_order = max(1, int(order))
    with _WRLS_BASIS_CACHE_LOCK:
        for (cached_shape, cached_order), cached_basis in _WRLS_BASIS_CACHE.items():
            if cached_shape == shape and cached_order >= requested_order:
                count = len(_polynomial_exponents_3d(requested_order))
                return cached_basis[:count]

        x = np.arange(shape[0], dtype=np.float64) - np.floor(shape[0] / 2.0)
        y = np.arange(shape[1], dtype=np.float64) - np.floor(shape[1] / 2.0)
        z = np.arange(shape[2], dtype=np.float64) - np.floor(shape[2] / 2.0)
        exponents = _polynomial_exponents_3d(requested_order)
        basis = np.empty((len(exponents), int(np.prod(shape))), dtype=np.float64)
        for column, (x_power, y_power, z_power) in enumerate(exponents):
            x_term = x ** x_power
            y_term = y ** y_power
            z_term = z ** z_power
            norm = np.sqrt(
                np.sum(x_term * x_term)
                * np.sum(y_term * y_term)
                * np.sum(z_term * z_term)
            )
            term = (
                x_term[:, np.newaxis, np.newaxis]
                * y_term[np.newaxis, :, np.newaxis]
                * z_term[np.newaxis, np.newaxis, :]
            )
            basis[column] = np.ravel(term / norm, order="C")
        _WRLS_BASIS_CACHE.clear()
        _WRLS_BASIS_CACHE[(shape, requested_order)] = basis
        return basis


def _fista_l1_normal_equations(gram, rhs, initial, lam, iterations):
    gram = np.asarray(gram, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    x = np.asarray(initial, dtype=np.float64).copy()
    y = x.copy()
    t_value = 1.0
    largest_eigenvalue = float(np.max(np.linalg.eigvalsh(gram))) if gram.size else 0.0
    lipschitz = 2.05 * largest_eigenvalue
    if not np.isfinite(lipschitz) or lipschitz <= np.finfo(np.float64).eps:
        return x
    shrink_threshold = float(lam) / lipschitz
    for _ in range(max(0, int(iterations))):
        alpha = y - (2.0 / lipschitz) * (gram @ y - rhs)
        x_new = np.sign(alpha) * np.maximum(np.abs(alpha) - shrink_threshold, 0.0)
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_value * t_value)) / 2.0
        y = x_new + ((t_value - 1.0) / t_new) * (x_new - x)
        x = x_new
        t_value = t_new
    return x


def _wrls_fit(phi, sigma, fit_mask, basis, order, lam, fista_iterations):
    exponent_count = len(_polynomial_exponents_3d(order))
    fit_basis = basis[:exponent_count]
    flat_indices = np.flatnonzero(np.asarray(fit_mask, dtype=bool).ravel(order="C"))
    if flat_indices.size < exponent_count:
        raise ValueError(
            f"not enough WRLS candidates for order {order}: {flat_indices.size} < {exponent_count}"
        )

    phi_flat = np.asarray(phi, dtype=np.float64).ravel(order="C")
    sigma_flat = np.asarray(sigma, dtype=np.float64).ravel(order="C")
    gram = np.zeros((exponent_count, exponent_count), dtype=np.float64)
    rhs = np.zeros(exponent_count, dtype=np.float64)
    chunk_size = 131072
    for start in range(0, flat_indices.size, chunk_size):
        index = flat_indices[start:start + chunk_size]
        design = fit_basis[:, index].T
        inverse_sigma = 1.0 / sigma_flat[index]
        weighted_design = design * inverse_sigma[:, np.newaxis]
        weighted_phi = phi_flat[index] * inverse_sigma
        gram += weighted_design.T @ weighted_design
        rhs += weighted_design.T @ weighted_phi

    try:
        initial = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        initial = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    coefficients = _fista_l1_normal_equations(
        gram,
        rhs,
        initial,
        lam=float(lam),
        iterations=int(fista_iterations),
    )
    correction = np.asarray(coefficients @ fit_basis, dtype=np.float64).reshape(phi.shape, order="C")
    return np.asarray(phi, dtype=np.float64) - correction, correction


def _middle_fov_mask(shape, mid_fov_fraction, mid_slice_fraction):
    mask = np.zeros(tuple(int(value) for value in shape), dtype=bool)
    center_y = int(np.floor(shape[1] / 2.0))
    center_z = int(np.floor(shape[2] / 2.0))
    extent_y = int(np.floor((shape[1] * float(mid_fov_fraction)) / 2.0))
    extent_z = int(np.floor((shape[2] * float(mid_slice_fraction)) / 2.0))
    y0 = max(0, center_y - extent_y)
    y1 = min(shape[1], center_y + extent_y + 1)
    z0 = max(0, center_z - extent_z)
    z1 = min(shape[2], center_z + extent_z + 1)
    mask[:, y0:y1, z0:z1] = True
    return mask


def _gmm_arto_cpu(
    epsilon,
    delta=2.0,
    central_probability=0.5,
    max_iterations=1000,
    initial_means=None,
):
    values = np.sort(np.asarray(epsilon, dtype=np.float64).reshape(-1))
    values = values[np.isfinite(values)]
    if values.size < 3:
        raise ValueError(f"not enough finite ARTO residuals: {values.size}")
    variance = float(np.var(values, ddof=1))
    overall_std = float(np.sqrt(max(variance, 0.0)))
    if not np.isfinite(overall_std) or overall_std <= np.finfo(np.float64).eps:
        return 0.0, max(overall_std, np.finfo(np.float64).eps), np.array([-1.0, 0.0, 1.0]), np.ones(3), np.array([0.25, 0.5, 0.25]), 0, "cpu"

    means = (
        np.array([-float(delta) * overall_std, 0.0, float(delta) * overall_std], dtype=np.float64)
        if initial_means is None
        else np.asarray(initial_means, dtype=np.float64).reshape(3).copy()
    )
    means[int(np.argmin(means))] = max(float(np.min(means)), float(values[0]))
    means[int(np.argmax(means))] = min(float(np.max(means)), float(values[-1]))
    gamma = np.full(3, overall_std / 2.0, dtype=np.float64)
    probability = np.array(
        [(1.0 - float(central_probability)) / 2.0, float(central_probability), (1.0 - float(central_probability)) / 2.0],
        dtype=np.float64,
    )
    tolerance = variance * 1e-4
    gamma_floor = overall_std / 50.0
    central_index = 1
    values_squared = values * values

    for iteration in range(max(0, int(max_iterations))):
        previous_means = means.copy()
        gamma = np.maximum(gamma, gamma_floor)
        scaled = (values[:, np.newaxis] - means[np.newaxis, :]) / gamma[np.newaxis, :]
        weighted_pdf = (
            np.exp(-0.5 * scaled * scaled)
            / (gamma[np.newaxis, :] * np.sqrt(2.0 * np.pi))
        ) * probability[np.newaxis, :]
        denominator = np.sum(weighted_pdf, axis=1)
        positive = denominator > 0.0
        if not np.all(positive):
            replacement = float(np.min(denominator[positive])) if np.any(positive) else np.finfo(np.float64).tiny
            denominator[~positive] = replacement
        weights = np.empty(3, dtype=np.float64)
        means = np.empty(3, dtype=np.float64)
        second_moments = np.empty(3, dtype=np.float64)
        for component in range(3):
            responsibility = weighted_pdf[:, component] / denominator
            weights[component] = np.sum(responsibility)
            means[component] = (responsibility @ values) / weights[component]
            second_moments[component] = (responsibility @ values_squared) / weights[component]
        probability = weights / float(values.size)
        gamma = np.sqrt(np.maximum(second_moments - means * means, 0.0))

        left_index = int(np.argmin(means))
        right_index = int(np.argmax(means))
        central_index = int(({0, 1, 2} - {left_index, right_index}).pop())
        means[left_index] = min(means[left_index], -gamma[central_index] * float(delta))
        means[right_index] = max(means[right_index], gamma[central_index] * float(delta))
        probability[central_index] = max(probability[central_index], float(central_probability))
        if np.all(np.abs(means - previous_means) < tolerance):
            return means[central_index], gamma[central_index], means, gamma, probability, iteration + 1, "cpu"

    return means[central_index], gamma[central_index], means, gamma, probability, int(max_iterations), "cpu"


def _gmm_arto_torch(
    torch,
    epsilon,
    delta=2.0,
    central_probability=0.5,
    max_iterations=1000,
    initial_means=None,
):
    if isinstance(epsilon, torch.Tensor):
        values = epsilon.to(device="cuda", dtype=torch.float64).reshape(-1)
    else:
        values = torch.as_tensor(
            np.asarray(epsilon, dtype=np.float64).reshape(-1),
            dtype=torch.float64,
            device="cuda",
        )
    # EM does not require sorted samples.  Sorting the full residual vector
    # was an O(N log N) pass and dominated large-volume CUDA runs.
    values = values[torch.isfinite(values)]
    value_count = int(values.numel())
    if value_count < 3:
        raise ValueError(f"not enough finite ARTO residuals: {value_count}")
    variance = float(torch.var(values, correction=1).item())
    overall_std = float(np.sqrt(max(variance, 0.0)))
    if not np.isfinite(overall_std) or overall_std <= np.finfo(np.float64).eps:
        return 0.0, max(overall_std, np.finfo(np.float64).eps), np.array([-1.0, 0.0, 1.0]), np.ones(3), np.array([0.25, 0.5, 0.25]), 0, "cuda"

    means = torch.as_tensor(
        ([-float(delta) * overall_std, 0.0, float(delta) * overall_std]
         if initial_means is None else np.asarray(initial_means, dtype=np.float64).reshape(3)),
        dtype=torch.float64, device="cuda",
    ).clone()
    min_value = torch.amin(values)
    max_value = torch.amax(values)
    left = torch.argmin(means)
    right = torch.argmax(means)
    means[left] = torch.maximum(means[left], min_value)
    means[right] = torch.minimum(means[right], max_value)
    gamma = torch.full((3,), overall_std / 2.0, dtype=torch.float64, device="cuda")
    probability = torch.as_tensor(
        [(1.0 - float(central_probability)) / 2.0, float(central_probability),
         (1.0 - float(central_probability)) / 2.0],
        dtype=torch.float64, device="cuda",
    )
    tolerance = variance * 1e-4
    gamma_floor = overall_std / 50.0
    values_squared = values * values
    central_index = 1

    for iteration in range(max(0, int(max_iterations))):
        previous_means = means.clone()
        gamma = torch.clamp(gamma, min=gamma_floor)
        scaled = (values[:, None] - means[None, :]) / gamma[None, :]
        weighted_pdf = (
            torch.exp(-0.5 * scaled * scaled)
            / (gamma[None, :] * np.sqrt(2.0 * np.pi))
        ) * probability[None, :]
        denominator = torch.sum(weighted_pdf, dim=1)
        positive = denominator > 0.0
        replacement = torch.min(
            torch.where(
                positive,
                denominator,
                torch.full_like(denominator, float("inf")),
            )
        )
        replacement = torch.where(
            torch.isfinite(replacement),
            replacement,
            torch.full_like(replacement, np.finfo(np.float64).tiny),
        )
        safe_denominator = torch.where(positive, denominator, replacement)
        weights = []
        updated_means = []
        second_moments = []
        for component in range(3):
            responsibility = weighted_pdf[:, component] / safe_denominator
            weight = torch.sum(responsibility)
            weights.append(weight)
            updated_means.append((responsibility @ values) / weight)
            second_moments.append((responsibility @ values_squared) / weight)
        weights = torch.stack(weights)
        updated_means = torch.stack(updated_means)
        second_moments = torch.stack(second_moments)
        updated_gamma = torch.sqrt(torch.clamp(second_moments - updated_means * updated_means, min=0.0))
        means = updated_means
        gamma = updated_gamma
        probability = weights / float(value_count)

        left_index = int(torch.argmin(means).item())
        right_index = int(torch.argmax(means).item())
        central_index = int(({0, 1, 2} - {left_index, right_index}).pop())
        means[left_index] = torch.minimum(means[left_index], -gamma[central_index] * float(delta))
        means[right_index] = torch.maximum(means[right_index], gamma[central_index] * float(delta))
        probability[central_index] = torch.maximum(
            probability[central_index], torch.as_tensor(float(central_probability), device="cuda", dtype=torch.float64)
        )
        if bool(torch.all(torch.abs(means - previous_means) < tolerance).item()):
            means_cpu = means.detach().cpu().numpy()
            gamma_cpu = gamma.detach().cpu().numpy()
            probability_cpu = probability.detach().cpu().numpy()
            return means_cpu[central_index], gamma_cpu[central_index], means_cpu, gamma_cpu, probability_cpu, iteration + 1, "cuda"

    means_cpu = means.detach().cpu().numpy()
    gamma_cpu = gamma.detach().cpu().numpy()
    probability_cpu = probability.detach().cpu().numpy()
    return means_cpu[central_index], gamma_cpu[central_index], means_cpu, gamma_cpu, probability_cpu, int(max_iterations), "cuda"


def _gmm_arto(*args, **kwargs):
    torch, _unavailable_reason = _available_torch_cuda()
    if torch is not None:
        try:
            return _gmm_arto_torch(torch, *args, **kwargs)
        except Exception:
            torch.cuda.empty_cache()
    return _gmm_arto_cpu(*args, **kwargs)


def _execute_wrls_arto_impl(
    im,
    corr_fit_order=3,
    lam=5.0,
    magnitude_threshold=0.04,
    mid_fov_fraction=0.5,
    mid_slice_fraction=0.65,
    arto_iterations=2,
    tau=3.0,
    delta=2.0,
    central_probability=0.5,
    fista_iterations=5000,
    gmm_iterations=1000,
    progress_callback=None,
):
    raw = np.asarray(im)
    zero_corr = _zero_corr_nvtzyx(raw.shape)
    diag = {
        "applied": False,
        "compute_device": "cpu",
        "corr_fit_order": int(corr_fit_order),
        "stationary_voxels": 0,
        "skipped_reason": "",
        "wrls_lambda": float(lam),
        "wrls_magnitude_threshold": float(magnitude_threshold),
        "wrls_mid_fov_fraction": float(mid_fov_fraction),
        "wrls_mid_slice_fraction": float(mid_slice_fraction),
        "wrls_arto_iterations": int(arto_iterations),
        "wrls_tau": float(tau),
        "wrls_delta": float(delta),
        "wrls_central_probability": float(central_probability),
        "wrls_fista_iterations": int(fista_iterations),
        "wrls_gmm_iterations": int(gmm_iterations),
    }
    if raw.ndim != 5 or raw.shape[0] < 4:
        diag["skipped_reason"] = f"expected NVTZYX with at least 4 encodes, got shape={raw.shape}"
        return zero_corr, np.zeros((1, 1, 1), dtype=bool), diag
    if raw.shape[1] < 2:
        diag["skipped_reason"] = "WRLS+ARTO requires at least two time frames for temporal sigma"
        return zero_corr, np.zeros(raw.shape[2:], dtype=bool), diag
    if int(arto_iterations) < 1:
        diag["skipped_reason"] = "WRLS+ARTO requires at least one ARTO iteration"
        return zero_corr, np.zeros(raw.shape[2:], dtype=bool), diag
    if not 0.0 < float(central_probability) < 1.0:
        raise ValueError("wrls_central_probability must be between 0 and 1")

    reference = np.asarray(raw[0], dtype=np.complex64)
    magnitude_xyz = np.transpose(np.mean(np.abs(reference), axis=0), (2, 1, 0))
    slice_max = np.max(magnitude_xyz, axis=(0, 1), keepdims=True)
    magnitude_mask = magnitude_xyz > (float(magnitude_threshold) * slice_max)
    if not np.any(magnitude_mask):
        diag["skipped_reason"] = "no WRLS magnitude mask candidates"
        return zero_corr, np.zeros(raw.shape[2:], dtype=bool), diag

    basis = _build_normalized_polynomial_basis(magnitude_xyz.shape, max(1, int(corr_fit_order)))
    initial_region = _middle_fov_mask(
        magnitude_xyz.shape,
        mid_fov_fraction=float(mid_fov_fraction),
        mid_slice_fraction=float(mid_slice_fraction),
    )
    corrections_xyz = np.zeros((3,) + magnitude_xyz.shape, dtype=np.float32)
    direction_masks = []
    gmm_iteration_counts = []
    gmm_devices = []
    total_fits = 3 * (1 + int(arto_iterations))
    completed_fits = 0

    for direction in range(3):
        phase_txyz = np.transpose(
            np.angle(np.asarray(raw[direction + 1], dtype=np.complex64) * np.conj(reference)) / np.pi,
            (0, 3, 2, 1),
        )
        phi = np.mean(phase_txyz, axis=0, dtype=np.float64)
        sigma = np.std(phase_txyz, axis=0, ddof=1, dtype=np.float64)
        valid = magnitude_mask & np.isfinite(phi) & np.isfinite(sigma) & (sigma > np.finfo(np.float32).eps)
        fit_mask = valid & initial_region
        phi_corrected, correction = _wrls_fit(
            phi,
            sigma,
            fit_mask,
            basis,
            order=1,
            lam=float(lam),
            fista_iterations=int(fista_iterations),
        )
        completed_fits += 1
        _emit_progress(
            progress_callback,
            "background_phase_fit",
            current=completed_fits,
            total=total_fits,
            message=f"WRLS+ARTO direction {direction + 1}/3 initialization",
        )

        initial_means = None
        final_mask = fit_mask
        for arto_index in range(int(arto_iterations)):
            epsilon = phi_corrected[valid] / sigma[valid]
            mu_center, gamma_center, means, _gamma, _probability, gmm_count, gmm_device = _gmm_arto(
                epsilon,
                delta=float(delta),
                central_probability=float(central_probability),
                max_iterations=int(gmm_iterations),
                initial_means=initial_means,
            )
            gmm_iteration_counts.append(int(gmm_count))
            gmm_devices.append(str(gmm_device))
            weighted_residual = np.full(phi.shape, np.nan, dtype=np.float64)
            np.divide(phi_corrected, sigma, out=weighted_residual, where=valid)
            lower = mu_center - float(tau) * gamma_center
            upper = mu_center + float(tau) * gamma_center
            final_mask = valid & (weighted_residual > lower) & (weighted_residual < upper)
            phi_corrected, correction = _wrls_fit(
                phi,
                sigma,
                final_mask,
                basis,
                order=int(corr_fit_order),
                lam=float(lam),
                fista_iterations=int(fista_iterations),
            )
            # The source assigns sigma0/phi0, but gmmArto reads gamma0/prob0;
            # only the component means are therefore carried into the next pass.
            initial_means = means
            completed_fits += 1
            _emit_progress(
                progress_callback,
                "background_phase_fit",
                current=completed_fits,
                total=total_fits,
                message=f"WRLS+ARTO direction {direction + 1}/3 ARTO {arto_index + 1}/{arto_iterations}",
            )

        corrections_xyz[direction] = np.asarray(correction * np.pi, dtype=np.float32)
        direction_masks.append(final_mask)

    stationary_xyz = np.logical_and.reduce(direction_masks)
    correction_nvtzyx = np.transpose(corrections_xyz, (0, 3, 2, 1))[:, np.newaxis, ...]
    stationary_zyx = np.transpose(stationary_xyz, (2, 1, 0))
    diag["applied"] = True
    diag["stationary_voxels"] = int(np.sum(stationary_zyx))
    diag["stationary_voxels_by_direction"] = [int(np.sum(mask)) for mask in direction_masks]
    diag["gmm_iteration_counts"] = gmm_iteration_counts
    if "cuda" in gmm_devices:
        diag["compute_device"] = "cuda"
        diag["gpu_accelerated_stages"] = ["arto_gmm"]
    return np.asarray(correction_nvtzyx, dtype=np.float32), stationary_zyx, diag


def _wrls_fit_torch(phi, sigma, fit_mask, basis, order, lam, fista_iterations, torch):
    exponent_count = len(_polynomial_exponents_3d(order))
    fit_basis = basis[:exponent_count]
    flat_indices = torch.nonzero(fit_mask.reshape(-1), as_tuple=False).reshape(-1)
    if int(flat_indices.numel()) < exponent_count:
        raise ValueError(
            f"not enough WRLS candidates for order {order}: {int(flat_indices.numel())} < {exponent_count}"
        )
    phi_flat = phi.reshape(-1).to(dtype=torch.float64)
    sigma_flat = sigma.reshape(-1).to(dtype=torch.float64)
    design = fit_basis[:, flat_indices].transpose(0, 1)
    inverse_sigma = torch.reciprocal(sigma_flat[flat_indices])
    weighted_design = design * inverse_sigma[:, None]
    weighted_phi = phi_flat[flat_indices] * inverse_sigma
    gram = weighted_design.transpose(0, 1) @ weighted_design
    rhs = weighted_design.transpose(0, 1) @ weighted_phi
    try:
        initial = torch.linalg.solve(gram, rhs)
    except RuntimeError:
        initial = torch.linalg.lstsq(gram, rhs).solution

    x = initial.clone()
    y = x.clone()
    t_value = torch.ones((), dtype=torch.float64, device=gram.device)
    largest_eigenvalue = torch.linalg.eigvalsh(gram).amax()
    lipschitz = 2.05 * largest_eigenvalue
    if not bool(torch.isfinite(lipschitz).item()) or float(lipschitz.item()) <= np.finfo(np.float64).eps:
        coefficients = x
    else:
        shrink_threshold = float(lam) / lipschitz
        for _ in range(max(0, int(fista_iterations))):
            alpha = y - (2.0 / lipschitz) * (gram @ y - rhs)
            x_new = torch.sign(alpha) * torch.clamp(torch.abs(alpha) - shrink_threshold, min=0.0)
            t_new = (1.0 + torch.sqrt(1.0 + 4.0 * t_value * t_value)) / 2.0
            y = x_new + ((t_value - 1.0) / t_new) * (x_new - x)
            x = x_new
            t_value = t_new
        coefficients = x
    correction = (coefficients @ fit_basis).reshape(phi.shape)
    return phi - correction, correction


def _execute_wrls_arto_gpu(
    im,
    corr_fit_order=3,
    lam=5.0,
    magnitude_threshold=0.04,
    mid_fov_fraction=0.5,
    mid_slice_fraction=0.65,
    arto_iterations=2,
    tau=3.0,
    delta=2.0,
    central_probability=0.5,
    fista_iterations=5000,
    gmm_iterations=1000,
    progress_callback=None,
):
    torch, reason = _available_torch_cuda()
    if torch is None:
        raise RuntimeError(f"CUDA unavailable: {reason}")
    raw = torch.as_tensor(np.asarray(im), dtype=torch.complex128, device="cuda")
    if raw.ndim != 5 or raw.shape[0] < 4:
        raise ValueError(f"expected NVTZYX with at least 4 encodes, got shape={tuple(raw.shape)}")
    if raw.shape[1] < 2:
        raise ValueError("WRLS+ARTO requires at least two time frames for temporal sigma")
    if int(arto_iterations) < 1:
        raise ValueError("WRLS+ARTO requires at least one ARTO iteration")
    if not 0.0 < float(central_probability) < 1.0:
        raise ValueError("wrls_central_probability must be between 0 and 1")

    reference = raw[0]
    magnitude_xyz = torch.abs(reference).mean(dim=0).permute(2, 1, 0).to(torch.float64)
    slice_max = magnitude_xyz.amax(dim=(0, 1), keepdim=True)
    magnitude_mask = magnitude_xyz > (float(magnitude_threshold) * slice_max)
    if not bool(torch.any(magnitude_mask).item()):
        raise ValueError("no WRLS magnitude mask candidates")

    shape_xyz = tuple(int(v) for v in magnitude_xyz.shape)
    basis_np = _build_normalized_polynomial_basis(shape_xyz, max(1, int(corr_fit_order)))
    basis = torch.as_tensor(basis_np, dtype=torch.float64, device="cuda")
    initial_region_np = _middle_fov_mask(
        shape_xyz,
        mid_fov_fraction=float(mid_fov_fraction),
        mid_slice_fraction=float(mid_slice_fraction),
    )
    initial_region = torch.as_tensor(initial_region_np, dtype=torch.bool, device="cuda")
    corrections_xyz = torch.zeros((3,) + shape_xyz, dtype=torch.float64, device="cuda")
    direction_masks = []
    gmm_iteration_counts = []
    gmm_devices = []
    total_fits = 3 * (1 + int(arto_iterations))
    completed_fits = 0

    for direction in range(3):
        phase_txyz = torch.angle(raw[direction + 1] * torch.conj(reference)).permute(3, 2, 1, 0) / np.pi
        phi = phase_txyz.mean(dim=3).to(torch.float64)
        sigma = phase_txyz.std(dim=3, correction=1).to(torch.float64)
        valid = magnitude_mask & torch.isfinite(phi) & torch.isfinite(sigma) & (sigma > np.finfo(np.float32).eps)
        fit_mask = valid & initial_region
        phi_corrected, correction = _wrls_fit_torch(
            phi, sigma, fit_mask, basis, order=1, lam=float(lam),
            fista_iterations=int(fista_iterations), torch=torch,
        )
        completed_fits += 1
        _emit_progress(progress_callback, "background_phase_fit", current=completed_fits, total=total_fits,
                       message=f"WRLS+ARTO direction {direction + 1}/3 initialization")

        initial_means = None
        final_mask = fit_mask
        for arto_index in range(int(arto_iterations)):
            epsilon = phi_corrected[valid] / sigma[valid]
            mu_center, gamma_center, means, _gamma, _probability, gmm_count, gmm_device = _gmm_arto_torch(
                torch, epsilon, delta=float(delta), central_probability=float(central_probability),
                max_iterations=int(gmm_iterations), initial_means=initial_means,
            )
            gmm_iteration_counts.append(int(gmm_count))
            gmm_devices.append(str(gmm_device))
            weighted_residual = torch.full_like(phi, float("nan"), dtype=torch.float64)
            weighted_residual[valid] = phi_corrected[valid] / sigma[valid]
            lower = mu_center - float(tau) * gamma_center
            upper = mu_center + float(tau) * gamma_center
            final_mask = valid & (weighted_residual > lower) & (weighted_residual < upper)
            phi_corrected, correction = _wrls_fit_torch(
                phi, sigma, final_mask, basis, order=int(corr_fit_order), lam=float(lam),
                fista_iterations=int(fista_iterations), torch=torch,
            )
            initial_means = means
            completed_fits += 1
            _emit_progress(progress_callback, "background_phase_fit", current=completed_fits, total=total_fits,
                           message=f"WRLS+ARTO direction {direction + 1}/3 ARTO {arto_index + 1}/{arto_iterations}")
        corrections_xyz[direction] = correction * np.pi
        direction_masks.append(final_mask)

    stationary_xyz = torch.stack(direction_masks, dim=0).all(dim=0)
    correction_nvtzyx = corrections_xyz.permute(0, 3, 2, 1).unsqueeze(1)
    stationary_zyx = stationary_xyz.permute(2, 1, 0)
    torch.cuda.synchronize()
    diag = {
        "applied": True, "compute_device": "cuda", "corr_fit_order": int(corr_fit_order),
        "stationary_voxels": int(stationary_zyx.sum().item()), "skipped_reason": "",
        "wrls_lambda": float(lam), "wrls_magnitude_threshold": float(magnitude_threshold),
        "wrls_mid_fov_fraction": float(mid_fov_fraction), "wrls_mid_slice_fraction": float(mid_slice_fraction),
        "wrls_arto_iterations": int(arto_iterations), "wrls_tau": float(tau), "wrls_delta": float(delta),
        "wrls_central_probability": float(central_probability), "wrls_fista_iterations": int(fista_iterations),
        "wrls_gmm_iterations": int(gmm_iterations), "stationary_voxels_by_direction": [int(m.sum().item()) for m in direction_masks],
        "gmm_iteration_counts": gmm_iteration_counts, "gpu_accelerated_stages": ["phase_statistics", "wrls_fit", "arto_gmm"],
    }
    return correction_nvtzyx.detach().cpu().numpy().astype(np.float32), stationary_zyx.detach().cpu().numpy(), diag


def execute_wrls_arto(
    im,
    corr_fit_order=3,
    lam=5.0,
    magnitude_threshold=0.04,
    mid_fov_fraction=0.5,
    mid_slice_fraction=0.65,
    arto_iterations=2,
    tau=3.0,
    delta=2.0,
    central_probability=0.5,
    fista_iterations=5000,
    gmm_iterations=1000,
    progress_callback=None,
):
    kwargs = {
        "corr_fit_order": corr_fit_order,
        "lam": lam,
        "magnitude_threshold": magnitude_threshold,
        "mid_fov_fraction": mid_fov_fraction,
        "mid_slice_fraction": mid_slice_fraction,
        "arto_iterations": arto_iterations,
        "tau": tau,
        "delta": delta,
        "central_probability": central_probability,
        "fista_iterations": fista_iterations,
        "gmm_iterations": gmm_iterations,
        "progress_callback": progress_callback,
    }
    torch, _reason = _available_torch_cuda()
    if torch is not None:
        try:
            return _execute_wrls_arto_gpu(im, **kwargs)
        except Exception:
            torch.cuda.empty_cache()
    return _execute_wrls_arto_impl(im, **kwargs)


def execute_msac(im, corr_fit_order=3, th=0.1, progress_callback=None):
    rng = np.random.RandomState(274612)
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
    magnitude_im_t = np.mean(np.abs(im), axis=-1)
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
        "n_enc": int(d),
        "rng": rng,
    }

    mfunc, cfunc, _fmfunc, fcfunc, dfunc = get_functions(run_4d, parameters["msac_fit_order"], corr_fit_order)
    functions = {
        "msac_fit": mfunc,
        "msac_dist": dfunc,
    }
    def _report_trial(current, total):
        _emit_progress(
            progress_callback,
            "background_phase_fit",
            current=current,
            total=total,
            message=f"Fitting background phase model: {current}/{total}",
        )

    diag["compute_device"] = "cpu"
    _cost, inlier_idx = msac(mp, parameters, functions, progress_callback=_report_trial)

    _emit_progress(
        progress_callback,
        "background_phase_finalize",
        current=int(parameters["trials"]),
        total=int(parameters["trials"]),
        message="Finalizing background phase correction",
    )

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
        dist_cache = {}

        def _dist4d_cached(coeffs, points):
            key = (id(points), tuple(points.shape), tuple(points.strides), int(fitorder_msac))
            prepared = dist_cache.get(key)
            if prepared is None:
                prepared = get_in_out_4d(fitorder_msac, points)[:2]
                dist_cache[key] = prepared
            xyz, design = prepared
            return np.abs((xyz - design @ coeffs) / 2)

        mfunc = lambda points, **kwargs: fit4d(fitorder_msac, points, **kwargs)
        cfunc = lambda points, **kwargs: fit4d(fitorder_corr, points, **kwargs)
        fmfunc = lambda coeffs, points: eval4d(fitorder_msac, coeffs, points)
        fcfunc = lambda coeffs, points: eval4d(fitorder_corr, coeffs, points)
        dfunc = _dist4d_cached
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
    if order != 0 and np.all(indx == indx[:, :1]):
        shared_inliers = indx[:, 0]
        coeffs[:, :] = np.linalg.lstsq(
            a[shared_inliers, :],
            xyz[shared_inliers, :],
            rcond=None,
        )[0]
        return coeffs
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


def msac(points, parameters, functions, progress_callback=None):
    samples = parameters["samples"]
    threshold = parameters["msac_thresh"]
    trials = parameters["trials"]
    n_enc = parameters["n_enc"]
    rng = parameters.get("rng")

    msac_fit = functions["msac_fit"]
    msac_dist = functions["msac_dist"]

    no_p = points.shape[0]
    best_cost = np.ones(n_enc, dtype=np.float32) * threshold * no_p
    best_inliers = np.zeros((no_p, n_enc), dtype=bool)
    permutation = np.random.permutation if rng is None else rng.permutation

    for trial_index in np.arange(trials):
        indx = permutation(no_p)[0:samples]
        sample = points[indx, :]

        coeffs = msac_fit(sample)
        residuals = np.asarray(msac_dist(coeffs, points), dtype=np.float32)

        residuals[residuals > threshold] = threshold
        inliers = residuals < threshold
        cost = np.sum(residuals, 0)
        comp_cost = best_cost > cost

        best_cost[comp_cost] = cost[comp_cost]
        best_inliers[:, comp_cost] = inliers[:, comp_cost]

        if progress_callback is not None:
            progress_callback(int(trial_index) + 1, int(trials))

    return best_cost, best_inliers
