import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np

from ..case_types import BackgroundPhaseCorrectionConfig, InputCase, LoadedCase, LoaderCapabilities
from .phase_correction import (
    apply_background_phase_correction_to_complex,
    apply_background_phase_correction_to_mag_flow,
    background_phase_correction_cache_metadata,
    background_phase_report_for_metadata,
    coerce_background_phase_correction_config,
)


def _axis_pair(a):
    mp = {
        "LR": ("LR", "RL"), "RL": ("LR", "RL"),
        "AP": ("AP", "PA"), "PA": ("AP", "PA"),
        "HF": ("HF", "FH"), "FH": ("HF", "FH"),
    }
    a = a.upper()
    if a not in mp:
        raise ValueError(a)
    return mp[a][0]


def _need_flip(curr_label, target_label):
    c, t = curr_label.upper(), target_label.upper()
    if _axis_pair(c) != _axis_pair(t):
        raise ValueError(f"{c} vs {t}")
    return c != t


def _permute_spatial(arr, curr_order, target_order, spatial_axes=(0, 1, 2)):
    curr_order = [x.upper() for x in curr_order]
    target_order = [x.upper() for x in target_order]
    cb = [_axis_pair(x) for x in curr_order]
    tb = [_axis_pair(x) for x in target_order]
    src_pos = [cb.index(x) for x in tb]
    axes = list(range(arr.ndim))
    new_spatial = [spatial_axes[p] for p in src_pos]
    for k, ax in enumerate(spatial_axes):
        axes[ax] = new_spatial[k]
    return np.transpose(arr, axes), src_pos


def _flip_axes(arr, axes_to_flip):
    for ax in axes_to_flip:
        arr = np.flip(arr, axis=ax)
    return arr


def reorient(mag, flow, segmask, venc, resolution, spatial_order, venc_order,
             target_spatial_order, target_venc_order, return_velocity=False, normalize_mag=True):
    spatial_order = [s.upper() for s in spatial_order]
    venc_order = [v.upper() for v in venc_order]
    target_spatial_order = [s.upper() for s in target_spatial_order]
    target_venc_order = [v.upper() for v in target_venc_order]
    resolution = np.asarray(resolution, dtype=np.float32)
    venc = np.asarray(venc, dtype=np.float32)
    if venc.ndim == 0:
        venc = np.full(3, float(venc), dtype=np.float32)

    mag_r, src_pos_mag = _permute_spatial(mag, spatial_order, target_spatial_order, spatial_axes=(0, 1, 2))
    flow_r, src_pos_flow = _permute_spatial(flow, spatial_order, target_spatial_order, spatial_axes=(0, 1, 2))
    seg_r, src_pos_seg = _permute_spatial(segmask, spatial_order, target_spatial_order, spatial_axes=(0, 1, 2))

    cb = [_axis_pair(s) for s in spatial_order]
    tb = [_axis_pair(s) for s in target_spatial_order]
    res_perm = np.array([cb.index(x) for x in tb], dtype=int)
    resolution_r = resolution[res_perm]

    flip_mag = [i for i in range(3) if _need_flip(spatial_order[src_pos_mag[i]], target_spatial_order[i])]
    flip_flow = [i for i in range(3) if _need_flip(spatial_order[src_pos_flow[i]], target_spatial_order[i])]
    flip_seg = [i for i in range(3) if _need_flip(spatial_order[src_pos_seg[i]], target_spatial_order[i])]
    mag_r = _flip_axes(mag_r, flip_mag)
    flow_r = _flip_axes(flow_r, flip_flow)
    seg_r = _flip_axes(seg_r, flip_seg)

    vb = [_axis_pair(v) for v in venc_order]
    tb2 = [_axis_pair(v) for v in target_venc_order]
    comp_perm = np.array([vb.index(x) for x in tb2], dtype=int)

    flow_r = flow_r[..., comp_perm]
    venc_r = venc[comp_perm]

    sign3 = np.array([(-1.0 if _need_flip(venc_order[comp_perm[i]], target_venc_order[i]) else 1.0)
                      for i in range(3)], dtype=np.float32)
    sign_shape = (1,) * (flow_r.ndim - 1) + (int(sign3.shape[0]),)
    flow_r = flow_r * sign3.reshape(sign_shape)

    if return_velocity:
        venc_shape = (1,) * (flow_r.ndim - 1) + (int(venc_r.shape[0]),)
        flow_r = (flow_r / np.pi) * venc_r.reshape(venc_shape)

    if normalize_mag:
        mag_max = np.max(np.abs(mag_r))
        if mag_max > 0:
            mag_r = mag_r / mag_max

    return flow_r, mag_r, seg_r, venc_r, resolution_r


def _flow_looks_like_phase_radians(flow, venc):
    flow = np.asarray(flow, dtype=np.float32)
    if flow.size == 0:
        return False
    finite = np.isfinite(flow)
    if not np.any(finite):
        return False

    venc = np.asarray(venc, dtype=np.float32)
    if venc.ndim == 0:
        venc = np.full(3, float(venc), dtype=np.float32)
    venc_abs = np.abs(venc[np.isfinite(venc)])
    if venc_abs.size == 0:
        return False
    if float(np.max(venc_abs)) <= float(np.pi) * 1.25:
        return False

    peak_abs = float(np.max(np.abs(flow[finite])))
    return float(np.pi) * 0.75 <= peak_abs <= float(np.pi) * 1.25


def _normalize_real_img_layout(img):
    img = np.asarray(img)
    if img.ndim in (4, 5) and int(img.shape[-1]) == 4:
        return np.ascontiguousarray(img), False
    if img.ndim in (4, 5) and int(img.shape[0]) == 4:
        axes = tuple(range(img.ndim - 1, 0, -1)) + (0,)
        return np.ascontiguousarray(np.transpose(img, axes)), True
    raise ValueError(f"real-valued img layout must be XYZT4, XYZ4, 4TZYX, or 4ZYX, got {img.shape}")


def _ensure_flow_mag_time_and_segmask(flow, mag, segmask):
    flow = np.asarray(flow)
    mag = np.asarray(mag)
    segmask = np.asarray(segmask)
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
    if segmask.ndim == 3:
        segmask = np.repeat(segmask[..., np.newaxis], nt, axis=3)
    elif segmask.ndim == 4 and segmask.shape[3] == 1 and nt > 1:
        segmask = np.repeat(segmask, nt, axis=3)
    elif segmask.ndim != 4:
        raise ValueError(f"segmask must be XYZT or XYZ, got {segmask.shape}")
    if segmask.shape[3] != nt:
        if segmask.shape[3] == 1:
            segmask = np.repeat(segmask, nt, axis=3)
        else:
            raise ValueError(f"segmask time dimension {segmask.shape[3]} does not match flow {nt}")
    return (
        np.ascontiguousarray(flow, dtype=np.float32),
        np.ascontiguousarray(mag, dtype=np.float32),
        np.ascontiguousarray(segmask, dtype=np.int16),
    )


def _ensure_flow_mag_time(flow, mag):
    flow = np.asarray(flow)
    mag = np.asarray(mag)
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


def _ensure_optional_time_volume(arr, nt, name, dtype):
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = np.repeat(arr[..., np.newaxis], nt, axis=3)
    elif arr.ndim == 4 and arr.shape[3] == 1 and nt > 1:
        arr = np.repeat(arr, nt, axis=3)
    elif arr.ndim != 4:
        raise ValueError(f"{name} must be XYZT or XYZ, got {arr.shape}")
    if arr.shape[3] != nt:
        if arr.shape[3] == 1:
            arr = np.repeat(arr, nt, axis=3)
        else:
            raise ValueError(f"{name} time dimension {arr.shape[3]} does not match flow {nt}")
    return np.ascontiguousarray(arr, dtype=dtype)


def _ensure_optional_sigma_time(sigma, nt):
    if sigma is None:
        return None
    sigma = np.asarray(sigma)
    if sigma.ndim == 4 and sigma.shape[-1] == 3:
        sigma = sigma[..., np.newaxis, :]
    elif sigma.ndim != 5 or sigma.shape[-1] != 3:
        raise ValueError(f"sigma must be XYZTV or XYZV with 3 components, got {sigma.shape}")
    if sigma.shape[3] != nt:
        if sigma.shape[3] == 1:
            sigma = np.repeat(sigma, nt, axis=3)
        else:
            raise ValueError(f"sigma time dimension {sigma.shape[3]} does not match flow {nt}")
    return np.ascontiguousarray(sigma, dtype=np.float32)


def normalize_loaded_case(
    *,
    flow,
    mag,
    resolution,
    origin,
    venc,
    rr,
    segmentation=None,
    tke_array=None,
    sigma=None,
    metadata=None,
    source_format="",
    source_group=None,
    capabilities=None,
):
    flow_out, mag_out = _ensure_flow_mag_time(flow, mag)
    nt = int(flow_out.shape[3])
    seg_out = _ensure_optional_time_volume(segmentation, nt, "segmentation", np.int16)
    tke_out = _ensure_optional_time_volume(tke_array, nt, "tke_array", np.float32)
    sigma_out = _ensure_optional_sigma_time(sigma, nt)
    if capabilities is None:
        capabilities = LoaderCapabilities(
            has_segmentation=seg_out is not None,
            has_tke=tke_out is not None,
            has_complex_source=False,
            supports_wss=False,
            supports_plane_metrics=False,
        )
    else:
        capabilities = LoaderCapabilities(
            has_segmentation=bool(capabilities.has_segmentation),
            has_tke=bool(capabilities.has_tke),
            has_complex_source=bool(capabilities.has_complex_source),
            supports_wss=bool(capabilities.supports_wss),
            supports_plane_metrics=bool(capabilities.supports_plane_metrics),
        )
    capabilities.has_tke = bool(capabilities.has_tke or tke_out is not None)

    resolution = np.asarray(resolution, dtype=float).reshape(-1)
    if resolution.size == 1:
        resolution = np.repeat(resolution, 3)
    origin = np.asarray(origin, dtype=float).reshape(-1)
    if origin.size == 1:
        origin = np.repeat(origin, 3)
    venc = np.asarray(venc, dtype=float).reshape(-1)
    if venc.size == 1:
        venc = np.repeat(venc, 3)

    return LoadedCase(
        flow=flow_out,
        mag=mag_out,
        segmentation=seg_out,
        resolution=np.asarray(resolution[:3], dtype=float).reshape(3),
        origin=np.asarray(origin[:3], dtype=float).reshape(3),
        venc=np.asarray(venc[:3], dtype=float).reshape(3),
        rr=float(rr),
        sigma=sigma_out,
        tke_array=tke_out,
        metadata=dict(metadata or {}),
        source_format=str(source_format or ""),
        source_group=source_group,
        capabilities=capabilities,
    )


def _reorient_spatial_only(arr, spatial_order, target_spatial_order):
    arr_r, src_pos = _permute_spatial(arr, spatial_order, target_spatial_order, spatial_axes=(0, 1, 2))
    flip_axes = [i for i in range(3) if _need_flip(spatial_order[src_pos[i]], target_spatial_order[i])]
    return _flip_axes(arr_r, flip_axes)


def _compute_spatial_bbox(mask, pad=0):
    m = np.asarray(mask, dtype=bool)
    if m.ndim > 3:
        m = np.any(m, axis=tuple(range(3, m.ndim)))
    if not np.any(m):
        return tuple(slice(0, int(m.shape[i])) for i in range(3))
    idx = np.argwhere(m)
    lo = np.maximum(idx.min(axis=0) - int(pad), 0)
    hi = np.minimum(idx.max(axis=0) + int(pad) + 1, np.array(m.shape[:3], dtype=int))
    return tuple(slice(int(lo[i]), int(hi[i])) for i in range(3))


def _target_bbox_to_source_slices(shape_raw, spatial_order, target_spatial_order, bbox_target):
    spatial_order = [str(x).upper() for x in spatial_order]
    target_spatial_order = [str(x).upper() for x in target_spatial_order]
    cb = [_axis_pair(s) for s in spatial_order]
    tb = [_axis_pair(s) for s in target_spatial_order]
    src_pos = [cb.index(x) for x in tb]
    out = [slice(0, int(shape_raw[i])) for i in range(3)]
    for target_axis, raw_axis in enumerate(src_pos):
        s = int(bbox_target[target_axis].start)
        e = int(bbox_target[target_axis].stop)
        if _need_flip(spatial_order[raw_axis], target_spatial_order[target_axis]):
            out[raw_axis] = slice(int(shape_raw[raw_axis]) - e, int(shape_raw[raw_axis]) - s)
        else:
            out[raw_axis] = slice(s, e)
    return tuple(out)


def _sigma_from_complex(img_complex, venc):
    venc = np.asarray(venc, dtype=np.float32)
    if venc.ndim == 0:
        venc = np.full(3, float(venc), dtype=np.float32)
    ref = np.abs(img_complex[..., 0]).astype(np.float32)
    enc = np.abs(img_complex[..., 1:4]).astype(np.float32)
    kv = np.pi / venc.reshape((1, 1, 1, 1, 3))
    ratio = ref[..., None] / np.clip(enc, 1e-12, None)
    ratio = np.clip(ratio, 1.0, None)
    sigma = np.sqrt(2.0 * np.log(ratio)) / kv
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    return sigma.astype(np.float32)


def _reorient_component_abs(arr, spatial_order, target_spatial_order, venc_order, target_venc_order):
    spatial_order = [s.upper() for s in spatial_order]
    venc_order = [v.upper() for v in venc_order]
    target_spatial_order = [s.upper() for s in target_spatial_order]
    target_venc_order = [v.upper() for v in target_venc_order]

    arr_r, src_pos = _permute_spatial(arr, spatial_order, target_spatial_order, spatial_axes=(0, 1, 2))
    flip_axes = [i for i in range(3) if _need_flip(spatial_order[src_pos[i]], target_spatial_order[i])]
    arr_r = _flip_axes(arr_r, flip_axes)

    vb = [_axis_pair(v) for v in venc_order]
    tb = [_axis_pair(v) for v in target_venc_order]
    comp_perm = np.array([vb.index(x) for x in tb], dtype=int)

    return arr_r[..., comp_perm]


def _reorient_real_valued_fields(
    *,
    mag,
    flow,
    segmask,
    sigma,
    tke_array,
    venc,
    resolution,
    spatial_order,
    venc_order,
    target_spatial_order,
    target_venc_order,
    return_velocity=False,
):
    segmask_for_reorient = segmask if segmask is not None else np.zeros(np.asarray(mag).shape, dtype=np.int16)
    flow_r, mag_r, seg_r, venc_r, resolution_r = reorient(
        mag,
        flow,
        segmask_for_reorient,
        venc=venc,
        resolution=resolution,
        spatial_order=spatial_order,
        venc_order=venc_order,
        target_spatial_order=target_spatial_order,
        target_venc_order=target_venc_order,
        return_velocity=return_velocity,
        normalize_mag=False,
    )
    sigma_r = None
    if sigma is not None:
        sigma_r = _reorient_component_abs(
            sigma,
            spatial_order=spatial_order,
            target_spatial_order=target_spatial_order,
            venc_order=venc_order,
            target_venc_order=target_venc_order,
        ).astype(np.float32)
    tke_r = None
    if tke_array is not None:
        tke_r = _reorient_spatial_only(
            tke_array,
            spatial_order=spatial_order,
            target_spatial_order=target_spatial_order,
        ).astype(np.float32)
    return flow_r, mag_r, (seg_r if segmask is not None else None), venc_r, resolution_r, sigma_r, tke_r


def _progress_prefix(progress_callback, prefix):
    if progress_callback is None:
        return None

    def _wrapped(payload):
        data = dict(payload or {})
        stage = str(data.get("stage", ""))
        data["stage"] = f"{prefix}{stage}" if prefix else stage
        progress_callback(data)

    return _wrapped


def _loader_correction_config(correction_config):
    if correction_config is None:
        return coerce_background_phase_correction_config({"enabled": False})
    return coerce_background_phase_correction_config(correction_config)

def _background_phase_corr_attr_scalar(attrs, name, default=None):
    if attrs is None:
        return default
    value = None
    try:
        value = attrs.get(name)
    except Exception:
        value = None
    if value is None:
        return default
    arr = np.asarray(value).reshape(-1)
    if arr.size == 0:
        return default
    item = arr[0]
    if isinstance(item, (bytes, np.bytes_)):
        return item.decode("utf-8", errors="ignore")
    if isinstance(default, (bool, np.bool_)):
        return bool(item)
    if isinstance(default, (int, np.integer)) and not isinstance(default, bool):
        return int(item)
    if isinstance(default, (float, np.floating)):
        return float(item)
    return item


def _read_background_phase_corr_cache(scope, cache_name, expected_shape, cfg, expected_source_group=None, allow_untagged_root=True):
    report = {
        "cache_hit": False,
        "cache_name": str(cache_name),
        "cache_reason": "missing",
    }
    ds = _find_h5_dataset(scope, cache_name)
    if ds is None:
        return None, report

    corr = np.asarray(ds[:], dtype=np.float32)
    if corr.ndim == len(expected_shape) - 1 and corr.shape[-1] == 3 and len(expected_shape) == corr.ndim + 1:
        corr = corr[..., np.newaxis, :]
    corr_shape = tuple(corr.shape)
    expected_shape = tuple(expected_shape)
    singleton_time_match = (
        len(corr_shape) == 5
        and len(expected_shape) == 5
        and corr_shape[:3] == expected_shape[:3]
        and corr_shape[3] == 1
        and corr_shape[4] == expected_shape[4]
    )
    if corr_shape != expected_shape and not singleton_time_match:
        report["cache_reason"] = f"shape_mismatch:{tuple(corr.shape)}"
        return None, report

    stored_group = _background_phase_corr_attr_scalar(ds.attrs, "corr_source_group", None)
    if expected_source_group is not None and stored_group is not None:
        if str(stored_group).strip("/") != str(expected_source_group).strip("/"):
            report["cache_reason"] = f"group_mismatch:{stored_group}"
            return None, report
    if expected_source_group is not None and stored_group is None:
        scope_group = _h5_group_source_name(scope)
        if scope_group is None and not allow_untagged_root:
            report["cache_reason"] = "missing_group_tag"
            return None, report

    expected_metadata = background_phase_correction_cache_metadata(cfg)
    expected_algorithm = str(expected_metadata["corr_algorithm"])
    algorithm = _background_phase_corr_attr_scalar(ds.attrs, "corr_algorithm", "")
    if algorithm in ("", None):
        if expected_algorithm != "msac":
            report["cache_reason"] = "algorithm_mismatch:untagged"
            return None, report
        algorithm = "msac"
    elif str(algorithm).lower() != expected_algorithm:
        report["cache_reason"] = f"algorithm_mismatch:{algorithm}"
        return None, report

    version = _background_phase_corr_attr_scalar(ds.attrs, "corr_version", 1)
    if version is not None and int(version) != int(expected_metadata["corr_version"]):
        report["cache_reason"] = f"version_mismatch:{version}"
        return None, report

    fit_order = _background_phase_corr_attr_scalar(ds.attrs, "corr_fit_order", None)
    if fit_order is not None and int(fit_order) != int(cfg.corr_fit_order):
        report["cache_reason"] = f"fit_order_mismatch:{fit_order}"
        return None, report

    algorithm_metadata = {
        key: value
        for key, value in expected_metadata.items()
        if key not in {"corr_algorithm", "corr_version", "corr_fit_order"}
    }
    stored_algorithm_metadata = {}
    for key, expected_value in algorithm_metadata.items():
        stored_value = _background_phase_corr_attr_scalar(ds.attrs, key, None)
        if stored_value is None:
            if expected_algorithm == "msac" and key == "corr_threshold":
                stored_value = expected_value
            else:
                report["cache_reason"] = f"parameter_missing:{key}"
                return None, report
        if isinstance(expected_value, (float, np.floating)):
            matches = np.isclose(float(stored_value), float(expected_value))
        elif isinstance(expected_value, (int, np.integer)):
            matches = int(stored_value) == int(expected_value)
        else:
            matches = str(stored_value) == str(expected_value)
        if not bool(matches):
            report["cache_reason"] = f"parameter_mismatch:{key}={stored_value}"
            return None, report
        stored_algorithm_metadata[key] = expected_value

    report.update({
        "cache_hit": True,
        "cache_reason": "hit",
        "corr_algorithm": expected_algorithm,
        "corr_version": int(version) if version is not None else int(expected_metadata["corr_version"]),
        "corr_fit_order": int(fit_order) if fit_order is not None else int(cfg.corr_fit_order),
        "corr_components": int(corr.shape[-1]),
    })
    report.update(stored_algorithm_metadata)
    if stored_group is not None:
        report["corr_source_group"] = str(stored_group)
    source_mode = _background_phase_corr_attr_scalar(ds.attrs, "corr_source_mode", None)
    if source_mode is not None:
        report["corr_source_mode"] = str(source_mode)
    stationary_voxels = _background_phase_corr_attr_scalar(ds.attrs, "corr_stationary_voxels", None)
    if stationary_voxels is not None:
        report["stationary_voxels"] = int(stationary_voxels)
    return corr, report


def _read_background_phase_corr_cache_from_scopes(scopes, cache_name, expected_shape, cfg, expected_source_group=None, allow_untagged_root=True):
    if bool(getattr(cfg, "force_recompute", False)):
        return None, {"cache_hit": False, "cache_reason": "force_recompute", "cache_name": str(cache_name)}
    last_report = None
    for scope in scopes:
        corr, report = _read_background_phase_corr_cache(
            scope,
            cache_name,
            expected_shape,
            cfg,
            expected_source_group=expected_source_group,
            allow_untagged_root=allow_untagged_root,
        )
        if (
            last_report is not None
            and report.get("cache_reason") == "missing"
            and last_report.get("cache_reason") not in (None, "missing")
        ):
            report = dict(report)
            report["cache_reason"] = last_report.get("cache_reason")
        last_report = report
        if bool(report.get("cache_hit", False)):
            return corr, report
    return None, last_report or {"cache_hit": False, "cache_reason": "missing", "cache_name": str(cache_name)}


def _write_background_phase_corr_cache(
    scope,
    cache_name,
    report,
    expected_source_group=None,
    progress_callback=None,
):
    corr = report.get("corr") if isinstance(report, dict) else None
    if corr is None:
        return False
    corr_arr = np.asarray(corr, dtype=np.float32)
    if corr_arr.ndim == 4 and corr_arr.shape[-1] == 3:
        corr_arr = corr_arr[..., np.newaxis, :]
    if corr_arr.ndim != 5 or corr_arr.shape[-1] != 3:
        return False
    try:
        if progress_callback is not None:
            progress_callback({
                "stage": "background_phase_cache_write",
                "message": f"Writing background phase cache: {cache_name}",
            })
        existing_name = _h5_member_name_map(scope).get(_canonical_h5_key(cache_name))
        if existing_name is not None:
            del scope[existing_name]
        ds = scope.create_dataset(str(cache_name), data=corr_arr, compression="gzip")
        ds.attrs["corr_algorithm"] = str(report.get("corr_algorithm", "msac"))
        ds.attrs["corr_version"] = int(report.get("corr_version", 1))
        ds.attrs["corr_fit_order"] = int(report.get("corr_fit_order", 3))
        for key, value in report.items():
            if not str(key).startswith("corr_") or key in {
                "corr_algorithm",
                "corr_version",
                "corr_fit_order",
                "corr_components",
                "corr_source",
            }:
                continue
            if isinstance(value, (str, int, float, bool, np.integer, np.floating, np.bool_)):
                ds.attrs[str(key)] = value
        ds.attrs["corr_components"] = int(report.get("corr_components", corr_arr.shape[-1]))
        ds.attrs["corr_source_mode"] = str(report.get("source_mode", ""))
        ds.attrs["corr_cache_hit"] = int(bool(report.get("cache_hit", False)))
        if expected_source_group is None:
            expected_source_group = _h5_group_source_name(scope)
        if expected_source_group is not None:
            ds.attrs["corr_source_group"] = str(expected_source_group)
        if report.get("stationary_voxels") is not None:
            ds.attrs["corr_stationary_voxels"] = int(report.get("stationary_voxels"))
    except Exception:
        return False
    if progress_callback is not None:
        progress_callback({
            "stage": "background_phase_cache_done",
            "current": 1,
            "total": 1,
            "message": f"Background phase cache saved: {cache_name}",
        })
    return True


def _canonical_h5_key(name):
    return re.sub(r"[^a-z0-9]+", "", str(name or "").strip().lower())


def _h5_member_name_map(group):
    mapping = {}
    for name in group.keys():
        mapping.setdefault(_canonical_h5_key(name), str(name))
    return mapping


def _h5_attr_name_map(group):
    mapping = {}
    for name in group.attrs.keys():
        mapping.setdefault(_canonical_h5_key(name), str(name))
    return mapping


def _find_h5_dataset(group, *aliases):
    name_map = _h5_member_name_map(group)
    for alias in aliases:
        actual = name_map.get(_canonical_h5_key(alias))
        if actual is None:
            continue
        obj = group[actual]
        if isinstance(obj, h5py.Dataset):
            return obj
    return None


def _find_h5_dataset_from_scopes(scopes, *aliases):
    for group in scopes:
        ds = _find_h5_dataset(group, *aliases)
        if ds is not None:
            return ds
    return None


def _find_h5_attr(group, *aliases):
    name_map = _h5_attr_name_map(group)
    for alias in aliases:
        actual = name_map.get(_canonical_h5_key(alias))
        if actual is not None:
            return group.attrs[actual]
    return None


def _read_h5_value(group, *aliases, default=None):
    ds = _find_h5_dataset(group, *aliases)
    if ds is not None:
        return ds[()]
    attr = _find_h5_attr(group, *aliases)
    if attr is not None:
        return attr
    return default


def _read_h5_value_from_scopes(scopes, *aliases, default=None):
    for group in scopes:
        value = _read_h5_value(group, *aliases, default=None)
        if value is not None:
            return value
    return default


def _decode_h5_string(value):
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def _coerce_h5_text_array(value, default):
    if value is None:
        return np.asarray(default, dtype=str)
    arr = np.asarray(value)
    if arr.ndim == 0:
        arr = np.asarray([arr.item()])
    tokens = []
    for item in arr.reshape(-1).tolist():
        decoded = _decode_h5_string(item).strip()
        if not decoded:
            continue
        parts = [part.strip() for part in re.split(r"[\s,;]+", decoded) if str(part).strip()]
        tokens.extend(parts or [decoded])
    return np.asarray(tokens or list(default), dtype=str)


def _coerce_h5_scalar_float(value, default):
    if value is None:
        return float(default)
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0:
        return float(default)
    return float(arr[0])


def _coerce_h5_numeric_array(value, default):
    if value is None:
        return np.asarray(default, dtype=float).reshape(-1)
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.asarray(default, dtype=float).reshape(-1)
    return arr.astype(float, copy=False)


def _coerce_h5_triplet(value, default, name, repeat_scalar=False):
    arr = _coerce_h5_numeric_array(value, default)
    if arr.size == 1 and repeat_scalar:
        return np.full(3, float(arr[0]), dtype=float)
    if arr.size != 3:
        raise ValueError(f"{name} must contain 3 values, got shape {arr.shape}")
    return np.asarray(arr, dtype=float)


def _h5_group_source_name(group):
    name = str(getattr(group, "name", "") or "").strip("/")
    return name or None


def _is_h5_data_group_candidate(group):
    if _find_h5_dataset(group, "img_complex") is not None:
        return True
    if _find_h5_dataset(group, "mag") is not None and _find_h5_dataset(group, "flow") is not None:
        return True
    img_ds = _find_h5_dataset(group, "img")
    if img_ds is None:
        return False
    if np.issubdtype(img_ds.dtype, np.complexfloating):
        return img_ds.ndim >= 4 and int(img_ds.shape[-1]) >= 4
    return img_ds.ndim in (4, 5) and int(img_ds.shape[-1]) == 4


def _h5_group_depth(name):
    token = str(name or "").strip("/")
    if not token:
        return 0
    return len(token.split("/"))


def _discover_h5_data_group_names(handle):
    candidate_names = []

    if _is_h5_data_group_candidate(handle):
        candidate_names.append("")

    def _visit(name, obj):
        if isinstance(obj, h5py.Group) and _is_h5_data_group_candidate(obj):
            candidate_names.append(str(name))

    handle.visititems(_visit)
    return sorted(set(candidate_names), key=lambda item: (_h5_group_depth(item), item))


def _h5_group_embedded_features(handle, group_name=None):
    group = handle if not group_name else handle[str(group_name).strip("/")]
    scopes = [group] if group is handle else [group, handle]
    seg_ds = _find_h5_dataset_from_scopes(scopes, "segmask", "segmentation", "seg")
    corr_ds = _find_h5_dataset_from_scopes(scopes, "corr")
    corr_low_ds = _find_h5_dataset_from_scopes(scopes, "corr_low")
    corr_high_ds = _find_h5_dataset_from_scopes(scopes, "corr_high")

    def _cache_method(dataset):
        if dataset is None:
            return None
        value = dataset.attrs.get("corr_algorithm", "msac")
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        token = str(value or "msac").strip().lower().replace("-", "_").replace("+", "_")
        token = {"wrls": "wrls_arto", "arto": "wrls_arto", "wrlsarto": "wrls_arto"}.get(token, token)
        return token if token in {"msac", "wrls_arto"} else None

    correction_method = None
    has_correction_cache = False
    if corr_ds is not None:
        correction_method = _cache_method(corr_ds)
        has_correction_cache = correction_method is not None
    elif corr_low_ds is not None and corr_high_ds is not None:
        low_method = _cache_method(corr_low_ds)
        high_method = _cache_method(corr_high_ds)
        if low_method is not None and low_method == high_method:
            correction_method = low_method
            has_correction_cache = True

    features = {
        "has_embedded_segmentation": seg_ds is not None,
        "has_background_correction_cache": bool(has_correction_cache),
    }
    if correction_method is not None:
        features["background_correction_method"] = correction_method
    return features


def inspect_h5_input_case(case):
    """Inspect embedded H5 artifacts without materializing image arrays."""
    if isinstance(case, InputCase):
        path = os.path.abspath(str(case.input_path))
        source_group = case.source_group
    else:
        path = os.path.abspath(str(case))
        source_group = None
    with h5py.File(path, "r") as handle:
        group, group_name = _resolve_h5_data_group(handle, source_group=source_group)
        features = _h5_group_embedded_features(
            handle,
            group_name=None if group is handle else group_name,
        )
    features["source_group"] = group_name
    return features


def discover_h5_input_cases(path):
    path = os.path.abspath(str(path))
    stem = os.path.splitext(os.path.basename(path))[0]
    with h5py.File(path, "r") as handle:
        candidate_names = _discover_h5_data_group_names(handle)
        if not candidate_names or candidate_names == [""]:
            return [
                InputCase(
                    input_path=path,
                    input_kind="h5",
                    display_name=stem,
                    output_name=stem,
                    source_group=None,
                    metadata=_h5_group_embedded_features(handle),
                )
            ]

        cases = []
        for group_name in candidate_names:
            group_token = str(group_name or "").strip("/")
            short_name = group_token.rsplit("/", 1)[-1] if group_token else stem
            safe_group = re.sub(r"[^A-Za-z0-9._-]+", "_", group_token).strip("._-") or "group"
            metadata = {
                "source_group": group_token or None,
                "source_group_name": short_name,
            }
            metadata.update(_h5_group_embedded_features(handle, group_token or None))
            cases.append(
                InputCase(
                    input_path=path,
                    input_kind="h5",
                    display_name=f"{stem} | {group_token}",
                    output_name=f"{stem}__{safe_group}",
                    source_group=group_token or None,
                    metadata=metadata,
                )
            )
    return cases


def _resolve_h5_data_group(handle, source_group=None):
    requested_group = str(source_group or "").strip("/")
    if requested_group:
        if requested_group not in handle:
            raise ValueError(f"h5 data group not found: {requested_group}")
        selected = handle[requested_group]
        if not isinstance(selected, h5py.Group):
            raise ValueError(f"h5 data group is not a group: {requested_group}")
        if not _is_h5_data_group_candidate(selected):
            raise ValueError(f"h5 group does not contain a supported case layout: {requested_group}")
        return selected, requested_group

    candidate_names = _discover_h5_data_group_names(handle)
    if not candidate_names:
        return handle, None
    if candidate_names == [""]:
        return handle, None

    shallowest_depth = _h5_group_depth(candidate_names[0])
    shallowest = [name for name in candidate_names if _h5_group_depth(name) == shallowest_depth]
    if len(shallowest) > 1:
        raise ValueError(
            "ambiguous h5 layout: multiple candidate data groups found: "
            + ", ".join(name for name in shallowest if name)
        )

    selected = shallowest[0]
    if not selected:
        return handle, None
    return handle[selected], selected


def _coerce_venc_array(group):
    venc_value = _read_h5_value(group, "VENC", "venc", default=None)
    if venc_value is not None:
        arr = _coerce_h5_numeric_array(venc_value, np.array([150.0, 150.0, 150.0], dtype=float))
        if arr.size == 1:
            return np.full(3, float(arr[0]), dtype=float)
        return np.asarray(arr, dtype=float)
    return np.array([150.0, 150.0, 150.0], dtype=float)


def _split_dual_venc_triplets(venc):
    venc = np.asarray(venc, dtype=np.float32).reshape(-1)
    if venc.size != 6:
        raise ValueError(f"dual-venc Nv=7 expects 6 venc entries, got {venc.shape}")
    if not np.all(np.isfinite(venc)) or np.any(venc <= 0.0):
        raise ValueError(f"dual-venc Nv=7 expects positive finite venc entries, got {venc.tolist()}")

    first = np.asarray(venc[:3], dtype=np.float32)
    second = np.asarray(venc[3:6], dtype=np.float32)
    equal = np.isclose(first, second, rtol=1e-5, atol=1e-6)
    first_is_low = bool(np.all((first < second) | equal) and np.any(first < second))
    second_is_low = bool(np.all((second < first) | equal) and np.any(second < first))
    if first_is_low:
        return first, second, 0, 1
    if second_is_low:
        return second, first, 1, 0
    raise ValueError(
        "dual-venc Nv=7 cannot determine low/high groups from VENC triplets: "
        f"first={first.tolist()}, second={second.tolist()}"
    )


def _dual_venc_triplet_ratio(lv, hv):
    lv = np.asarray(lv, dtype=np.float32)
    hv = np.asarray(hv, dtype=np.float32)
    return np.divide(
        hv,
        np.clip(lv, 1e-12, None),
        out=np.ones_like(hv, dtype=np.float32),
        where=np.abs(lv) > 1e-12,
    ).astype(np.float32)


def _dual_venc_correct_alias(flow_lv_vtzyx, flow_hv_vtzyx, lv_triplet, ratio1, ratio2):
    flow_lv_vtzyx = np.asarray(flow_lv_vtzyx, dtype=np.float32)
    flow_hv_vtzyx = np.asarray(flow_hv_vtzyx, dtype=np.float32)
    lv_triplet = np.asarray(lv_triplet, dtype=np.float32).reshape(3, 1, 1, 1, 1)
    ratio1_arr = np.asarray(ratio1, dtype=np.float32).reshape(3, 1, 1, 1, 1)
    ratio2_arr = np.asarray(ratio2, dtype=np.float32).reshape(3, 1, 1, 1, 1)

    dlv = flow_hv_vtzyx - flow_lv_vtzyx

    th1 = lv_triplet * (1.0 - ratio1_arr)
    th2 = lv_triplet * (3.0 + ratio1_arr)
    th3 = lv_triplet * (3.0 - ratio2_arr)
    th4 = lv_triplet * (5.0 + ratio2_arr)

    mask_p2 = (dlv >= th1) & (dlv <= th2)
    mask_m2 = (dlv >= -th2) & (dlv <= -th1)
    mask_p4 = (dlv >= th3) & (dlv <= th4)
    mask_m4 = (dlv >= -th4) & (dlv <= -th3)

    dual_alias_corr = np.zeros_like(flow_lv_vtzyx, dtype=np.float32)
    dual_alias_corr = np.where(mask_p2, 2.0 * lv_triplet, dual_alias_corr)
    dual_alias_corr = np.where(mask_m2, -2.0 * lv_triplet, dual_alias_corr)
    dual_alias_corr = np.where(mask_p4, 4.0 * lv_triplet, dual_alias_corr)
    dual_alias_corr = np.where(mask_m4, -4.0 * lv_triplet, dual_alias_corr)
    return np.asarray(flow_lv_vtzyx + dual_alias_corr, dtype=np.float32), np.asarray(dual_alias_corr, dtype=np.float32)


def _load_legacy_dual_venc_h5(
    img_complex,
    segmask,
    venc,
    resolution,
    origin,
    rr,
    spatial_order,
    venc_order,
    cfg,
    progress_callback=None,
    h5_group=None,
    h5_scopes=None,
    source_group=None,
    allow_untagged_root=True,
):
    if img_complex.ndim != 5 or img_complex.shape[-1] != 7:
        raise ValueError(f"legacy dual-venc H5 expects XYZT7 complex data, got {img_complex.shape}")

    mag = np.abs(img_complex[..., 0]).astype(np.float32)
    lv_venc, hv_venc, lv_group_index, hv_group_index = _split_dual_venc_triplets(venc)
    encoded_groups = (img_complex[..., 1:4], img_complex[..., 4:7])
    lv_complex = np.concatenate([img_complex[..., :1], encoded_groups[lv_group_index]], axis=-1)
    hv_complex = np.concatenate([img_complex[..., :1], encoded_groups[hv_group_index]], axis=-1)
    cache_scopes = list(h5_scopes or ([h5_group] if h5_group is not None else []))
    lv_cached_corr, lv_cache_report = _read_background_phase_corr_cache_from_scopes(
        cache_scopes,
        "corr_low",
        tuple(lv_complex.shape[:-1]) + (3,),
        cfg,
        expected_source_group=source_group,
        allow_untagged_root=allow_untagged_root,
    )
    hv_cached_corr, hv_cache_report = _read_background_phase_corr_cache_from_scopes(
        cache_scopes,
        "corr_high",
        tuple(hv_complex.shape[:-1]) + (3,),
        cfg,
        expected_source_group=source_group,
        allow_untagged_root=allow_untagged_root,
    )

    progress_lock = threading.Lock()

    def _locked_progress(payload):
        if progress_callback is None:
            return
        with progress_lock:
            progress_callback(payload)

    callback = _locked_progress if progress_callback is not None else None
    lv_kwargs = {
        "config": cfg,
        "progress_callback": _progress_prefix(callback, "h5_dual_lv_"),
        "source_mode": "legacy_dual_venc_h5_low",
        "cached_corr": lv_cached_corr,
    }
    hv_kwargs = {
        "config": cfg,
        "progress_callback": _progress_prefix(callback, "h5_dual_hv_"),
        "source_mode": "legacy_dual_venc_h5_high",
        "cached_corr": hv_cached_corr,
    }
    if bool(cfg.enabled):
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="autoflow-bgc") as executor:
            lv_future = executor.submit(
                apply_background_phase_correction_to_complex,
                lv_complex,
                **lv_kwargs,
            )
            hv_future = executor.submit(
                apply_background_phase_correction_to_complex,
                hv_complex,
                **hv_kwargs,
            )
            lv_corr, _lv_stationary, lv_report = lv_future.result()
            hv_corr, _hv_stationary, hv_report = hv_future.result()
    else:
        lv_corr, _lv_stationary, lv_report = apply_background_phase_correction_to_complex(
            lv_complex,
            **lv_kwargs,
        )
        hv_corr, _hv_stationary, hv_report = apply_background_phase_correction_to_complex(
            hv_complex,
            **hv_kwargs,
        )

    lv_report["cache_name"] = "corr_low"
    if not bool(lv_report.get("cache_hit", False)):
        lv_report["cache_reason"] = lv_cache_report.get("cache_reason", "missing")
    if "stationary_voxels" in lv_cache_report and bool(lv_report.get("cache_hit", False)):
        lv_report["stationary_voxels"] = int(lv_cache_report["stationary_voxels"])
    if bool(lv_report.get("applied", False)) and not bool(lv_report.get("cache_hit", False)) and h5_group is not None:
        lv_report["cache_written"] = bool(_write_background_phase_corr_cache(
            h5_group,
            "corr_low",
            lv_report,
            expected_source_group=source_group,
            progress_callback=_progress_prefix(progress_callback, "h5_dual_lv_"),
        ))

    hv_report["cache_name"] = "corr_high"
    if not bool(hv_report.get("cache_hit", False)):
        hv_report["cache_reason"] = hv_cache_report.get("cache_reason", "missing")
    if "stationary_voxels" in hv_cache_report and bool(hv_report.get("cache_hit", False)):
        hv_report["stationary_voxels"] = int(hv_cache_report["stationary_voxels"])
    if bool(hv_report.get("applied", False)) and not bool(hv_report.get("cache_hit", False)) and h5_group is not None:
        hv_report["cache_written"] = bool(_write_background_phase_corr_cache(
            h5_group,
            "corr_high",
            hv_report,
            expected_source_group=source_group,
            progress_callback=_progress_prefix(progress_callback, "h5_dual_hv_"),
        ))
    lv_use = lv_corr if bool(lv_report.get("applied", False)) else lv_complex
    hv_use = hv_corr if bool(hv_report.get("applied", False)) else hv_complex

    flow_lv_raw = np.angle(lv_use[..., 1:4] * np.conj(lv_use[..., 0][..., None])).astype(np.float32)
    flow_hv_raw = np.angle(hv_use[..., 1:4] * np.conj(hv_use[..., 0][..., None])).astype(np.float32)

    segmask_for_reorient = segmask if segmask is not None else np.zeros(mag.shape, dtype=np.int16)
    flow_lv, mag_out, seg_r, lv_venc_new, res_new = reorient(
        mag,
        flow_lv_raw,
        segmask_for_reorient,
        venc=lv_venc,
        resolution=resolution,
        spatial_order=spatial_order,
        venc_order=venc_order,
        target_spatial_order=("LR", "AP", "FH"),
        target_venc_order=("LR", "AP", "FH"),
        return_velocity=True,
    )
    flow_hv, _mag_hv, _seg_hv, hv_venc_new, _res_hv = reorient(
        mag,
        flow_hv_raw,
        segmask_for_reorient,
        venc=hv_venc,
        resolution=resolution,
        spatial_order=spatial_order,
        venc_order=venc_order,
        target_spatial_order=("LR", "AP", "FH"),
        target_venc_order=("LR", "AP", "FH"),
        return_velocity=True,
    )

    flow_lv_vtzyx = np.transpose(flow_lv, (4, 3, 2, 1, 0))
    flow_hv_vtzyx = np.transpose(flow_hv, (4, 3, 2, 1, 0))
    lv_triplet = np.asarray(lv_venc_new, dtype=np.float32)
    hv_triplet = np.asarray(hv_venc_new, dtype=np.float32)
    ratio_hv_lv = _dual_venc_triplet_ratio(lv_triplet, hv_triplet)
    ratio1 = np.full(3, float(cfg.dual_venc_ratio1), dtype=np.float32)
    ratio2 = np.full(3, float(cfg.dual_venc_ratio2), dtype=np.float32)

    flow_dual_vtzyx, dual_alias_corr_vtzyx = _dual_venc_correct_alias(
        flow_lv_vtzyx,
        flow_hv_vtzyx,
        lv_triplet,
        ratio1,
        ratio2,
    )
    flow_dual = np.transpose(flow_dual_vtzyx, (4, 3, 2, 1, 0)).astype(np.float32)
    dual_alias_corr = np.transpose(dual_alias_corr_vtzyx, (4, 3, 2, 1, 0)).astype(np.float32)

    meta = {
        "background_phase_correction": {
            "dual_venc_low": background_phase_report_for_metadata(lv_report),
            "dual_venc_high": background_phase_report_for_metadata(hv_report),
        },
        "spatial_order_raw": [str(x) for x in spatial_order[:3]],
        "venc_order_raw": [str(x) for x in venc_order[:3]],
        "dual_venc": {
            "enabled": True,
            "input_channels": int(img_complex.shape[-1]),
            "lv_channel_indices": list(range(1 + 3 * lv_group_index, 4 + 3 * lv_group_index)),
            "hv_channel_indices": list(range(1 + 3 * hv_group_index, 4 + 3 * hv_group_index)),
            "lv_venc": lv_triplet.astype(float).tolist(),
            "hv_venc": hv_triplet.astype(float).tolist(),
            "ratio1": ratio1.astype(float).tolist(),
            "ratio2": ratio2.astype(float).tolist(),
            "hv_lv_ratio": ratio_hv_lv.astype(float).tolist(),
            "alias_shift_unique_cm_s": sorted({float(x) for x in np.unique(np.round(dual_alias_corr, decimals=6))}),
        },
    }
    return normalize_loaded_case(
        flow=flow_dual,
        mag=mag_out,
        segmentation=seg_r if segmask is not None else None,
        resolution=np.asarray(res_new, dtype=float),
        origin=origin,
        venc=np.asarray(hv_triplet, dtype=float),
        rr=float(rr),
        sigma=None,
        tke_array=None,
        metadata=meta,
        source_format="legacy_h5_dual_venc",
        capabilities=LoaderCapabilities(
            has_segmentation=segmask is not None,
            has_tke=False,
            has_complex_source=False,
            supports_wss=True,
            supports_plane_metrics=True,
        ),
    )


def load_h5_data(path, correction_config=None, progress_callback=None, source_group=None, force_recompute_seg=False):
    target_spatial_order = ("LR", "AP", "FH")
    target_venc_order = ("LR", "AP", "FH")
    cfg = _loader_correction_config(correction_config)
    h5_mode = "r+" if bool(cfg.enabled) else "r"
    try:
        handle_ctx = h5py.File(path, h5_mode)
    except OSError:
        handle_ctx = h5py.File(path, "r")
    with handle_ctx as handle:
        candidate_names = _discover_h5_data_group_names(handle)
        group, group_name = _resolve_h5_data_group(handle, source_group=source_group)
        allow_untagged_root = bool(group is handle or len(candidate_names) <= 1)
        scopes = [group] if group is handle else [group, handle]
        VENC = _coerce_venc_array(group)
        if VENC.shape == (3,) and group is not handle:
            root_venc = _coerce_venc_array(handle)
            if np.allclose(VENC, np.array([150.0, 150.0, 150.0], dtype=float)) and not np.allclose(root_venc, VENC):
                VENC = root_venc
        resolution = _coerce_h5_triplet(
            _read_h5_value_from_scopes(scopes, "Resolution", default=np.array([1, 1, 1], dtype=float)),
            default=np.array([1.0, 1.0, 1.0], dtype=float),
            name="Resolution",
            repeat_scalar=True,
        )
        origin = _coerce_h5_triplet(
            _read_h5_value_from_scopes(scopes, "Origin", default=np.array([0.0, 0.0, 0.0], dtype=float)),
            default=np.array([0.0, 0.0, 0.0], dtype=float),
            name="Origin",
            repeat_scalar=False,
        )
        rr = _coerce_h5_scalar_float(_read_h5_value_from_scopes(scopes, "RR", default=1000.0), 1000.0)
        spatial_order = _coerce_h5_text_array(
            _read_h5_value_from_scopes(scopes, "SpatialOrder", default=None),
            default=("FH", "AP", "LR"),
        )
        venc_order = _coerce_h5_text_array(
            _read_h5_value_from_scopes(scopes, "VENCOrder", "VencOrder", default=None),
            default=("FH", "AP", "LR"),
        )

        seg_ds = _find_h5_dataset_from_scopes(scopes, "segmask", "segmentation", "seg")
        if seg_ds is not None and bool(force_recompute_seg):
            seg_source = str(seg_ds.attrs.get("autoflow_source", "") or "").strip().lower()
            if seg_source == "auto_segmentation":
                seg_ds = None
        segmask = None if seg_ds is None else np.asarray(seg_ds[:], dtype=np.int16)

        img_complex_ds = _find_h5_dataset(group, "img_complex")
        img_ds = _find_h5_dataset(group, "img")
        if img_complex_ds is None and img_ds is not None and np.issubdtype(img_ds.dtype, np.complexfloating):
            img_complex_ds = img_ds

        meta = {
            "source_group": group_name,
            "spatial_order_raw": [str(x) for x in spatial_order[:3]],
            "venc_order_raw": [str(x) for x in venc_order[:3]],
            "force_recompute_seg": bool(force_recompute_seg),
        }

        if img_complex_ds is not None:
            src_slices = tuple(slice(0, int(img_complex_ds.shape[i])) for i in range(3))
            img_complex = np.asarray(img_complex_ds[src_slices + (slice(None),) * (img_complex_ds.ndim - 3)])
            if img_complex.ndim == 5 and img_complex.shape[-1] == 7:
                loaded = _load_legacy_dual_venc_h5(
                    img_complex,
                    segmask,
                    VENC,
                    resolution,
                    origin,
                    rr,
                    spatial_order,
                    venc_order,
                    cfg,
                    progress_callback=progress_callback,
                    h5_group=group,
                    h5_scopes=scopes,
                    source_group=group_name,
                    allow_untagged_root=allow_untagged_root,
                )
                loaded.source_group = group_name
                loaded.metadata = dict(loaded.metadata or {})
                loaded.metadata.setdefault("h5_layout", "complex_img")
                return loaded
            cached_corr, cache_report = _read_background_phase_corr_cache_from_scopes(
                scopes,
                "corr",
                tuple(img_complex.shape[:-1]) + (3,),
                cfg,
                expected_source_group=group_name,
                allow_untagged_root=allow_untagged_root,
            )
            img_complex_corr, _stationary_mask_raw, corr_report = apply_background_phase_correction_to_complex(
                img_complex,
                config=cfg,
                progress_callback=_progress_prefix(progress_callback, "h5_"),
                source_mode="legacy_complex_h5",
                cached_corr=cached_corr,
            )
            corr_report["cache_name"] = "corr"
            if not bool(corr_report.get("cache_hit", False)):
                corr_report["cache_reason"] = cache_report.get("cache_reason", "missing")
            if "stationary_voxels" in cache_report and bool(corr_report.get("cache_hit", False)):
                corr_report["stationary_voxels"] = int(cache_report["stationary_voxels"])
            if bool(corr_report.get("applied", False)) and not bool(corr_report.get("cache_hit", False)):
                corr_report["cache_written"] = bool(_write_background_phase_corr_cache(
                    group,
                    "corr",
                    corr_report,
                    expected_source_group=group_name,
                    progress_callback=_progress_prefix(progress_callback, "h5_"),
                ))
            img_complex_use = img_complex_corr if bool(corr_report.get("applied", False)) else img_complex
            mag = np.abs(img_complex[..., 0]).astype(np.float32)
            flow_raw = np.angle(img_complex_use[..., 1:4] * np.conj(img_complex_use[..., 0][..., None])).astype(np.float32)
            sigma_raw = _sigma_from_complex(img_complex_use, VENC)
            segmask_for_reorient = segmask if segmask is not None else np.zeros(mag.shape, dtype=np.int16)
            flow, mag_out, seg_r, venc_new, res_new = reorient(
                mag,
                flow_raw,
                segmask_for_reorient,
                venc=VENC,
                resolution=resolution,
                spatial_order=spatial_order,
                venc_order=venc_order,
                target_spatial_order=target_spatial_order,
                target_venc_order=target_venc_order,
                return_velocity=True,
            )
            sigma = _reorient_component_abs(
                sigma_raw,
                spatial_order=spatial_order,
                target_spatial_order=target_spatial_order,
                venc_order=venc_order,
                target_venc_order=target_venc_order,
            ).astype(np.float32)
            meta.update({
                "background_phase_correction": background_phase_report_for_metadata(corr_report),
                "force_recompute_corr": bool(getattr(cfg, "force_recompute", False)),
                "h5_layout": "complex_img",
                "spatial_order_raw": [str(x) for x in spatial_order[:3]],
                "venc_order_raw": [str(x) for x in venc_order[:3]],
            })
            return normalize_loaded_case(
                flow=flow,
                mag=mag_out,
                segmentation=seg_r if segmask is not None else None,
                resolution=np.asarray(res_new, dtype=float),
                origin=origin,
                venc=np.asarray(venc_new, dtype=float),
                rr=float(rr),
                sigma=sigma,
                tke_array=None,
                metadata=meta,
                source_format="legacy_h5",
                source_group=group_name,
                capabilities=LoaderCapabilities(
                    has_segmentation=segmask is not None,
                    has_tke=True,
                    has_complex_source=True,
                    supports_wss=True,
                    supports_plane_metrics=True,
                ),
            )

        flow_ds = _find_h5_dataset(group, "flow")
        mag_ds = _find_h5_dataset(group, "mag")
        normalized_layout = flow_ds is not None and mag_ds is not None

        if normalized_layout or img_ds is not None:
            sigma_ds = _find_h5_dataset_from_scopes(scopes, "sigma")
            tke_ds = _find_h5_dataset_from_scopes(scopes, "tke_array", "tke")
            sigma = None if sigma_ds is None else np.asarray(sigma_ds[:], dtype=np.float32)
            tke_array = None if tke_ds is None else np.asarray(tke_ds[:], dtype=np.float32)
            flow_is_phase_radians = False
            transposed_from_channel_first = False
            if normalized_layout:
                flow_raw = np.asarray(flow_ds[:], dtype=np.float32)
                mag_raw = np.asarray(mag_ds[:], dtype=np.float32)
                layout_name = "normalized_h5"
            else:
                img = np.asarray(img_ds[:])
                if np.issubdtype(img.dtype, np.complexfloating):
                    raise ValueError(f"unsupported complex img layout: {path}")
                img, transposed_from_channel_first = _normalize_real_img_layout(img)
                mag_raw = np.asarray(img[..., 0], dtype=np.float32)
                flow_raw = np.asarray(img[..., 1:4], dtype=np.float32)
                flow_is_phase_radians = _flow_looks_like_phase_radians(flow_raw, VENC)
                layout_name = "combined_img_real"
            flow_raw, mag_raw, segmask_r, venc_new, res_new, sigma_r, tke_array_r = _reorient_real_valued_fields(
                mag=mag_raw,
                flow=flow_raw,
                segmask=segmask,
                sigma=sigma,
                tke_array=tke_array,
                venc=VENC,
                resolution=resolution,
                spatial_order=spatial_order,
                venc_order=venc_order,
                target_spatial_order=target_spatial_order,
                target_venc_order=target_venc_order,
                return_velocity=flow_is_phase_radians,
            )
            cached_corr, cache_report = _read_background_phase_corr_cache_from_scopes(
                scopes,
                "corr",
                tuple(flow_raw.shape[:-1]) + (3,),
                cfg,
                expected_source_group=group_name,
                allow_untagged_root=allow_untagged_root,
            )
            flow_corr, _stationary_mask, corr_report = apply_background_phase_correction_to_mag_flow(
                mag_raw,
                flow_raw,
                venc_new,
                config=cfg,
                progress_callback=_progress_prefix(progress_callback, "h5_"),
                cached_corr=cached_corr,
            )
            corr_report["cache_name"] = "corr"
            if not bool(corr_report.get("cache_hit", False)):
                corr_report["cache_reason"] = cache_report.get("cache_reason", "missing")
            if "stationary_voxels" in cache_report and bool(corr_report.get("cache_hit", False)):
                corr_report["stationary_voxels"] = int(cache_report["stationary_voxels"])
            if bool(corr_report.get("applied", False)) and not bool(corr_report.get("cache_hit", False)):
                corr_report["cache_written"] = bool(_write_background_phase_corr_cache(
                    group,
                    "corr",
                    corr_report,
                    expected_source_group=group_name,
                    progress_callback=_progress_prefix(progress_callback, "h5_"),
                ))
            meta.update({
                "background_phase_correction": background_phase_report_for_metadata(corr_report),
                "force_recompute_corr": bool(getattr(cfg, "force_recompute", False)),
                "h5_layout": layout_name,
                "flow_rescaled_from_pi_to_venc": bool(flow_is_phase_radians),
                "spatial_order_raw": [str(x) for x in spatial_order[:3]],
                "venc_order_raw": [str(x) for x in venc_order[:3]],
            })
            if layout_name == "combined_img_real":
                meta["flow_value_unit_raw"] = "phase_radians" if flow_is_phase_radians else "velocity"
                meta["real_img_channel_axis_raw"] = "first" if transposed_from_channel_first else "last"
            return normalize_loaded_case(
                flow=flow_corr,
                mag=mag_raw,
                segmentation=segmask_r,
                resolution=res_new,
                origin=origin,
                venc=venc_new,
                rr=float(rr),
                sigma=sigma_r,
                tke_array=tke_array_r,
                metadata=meta,
                source_format="normalized_h5",
                source_group=group_name,
                capabilities=LoaderCapabilities(
                    has_segmentation=segmask_r is not None,
                    has_tke=tke_array_r is not None,
                    has_complex_source=False,
                    supports_wss=True,
                    supports_plane_metrics=True,
                ),
            )

        raise ValueError(f"unsupported h5 layout: {path}")
