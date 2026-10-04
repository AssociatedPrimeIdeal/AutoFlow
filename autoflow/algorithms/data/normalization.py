"""Canonical LoadedCase arrays, time broadcasting and complex-derived sigma."""

import numpy as np
from ...case_types import LoadedCase, LoaderCapabilities


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
    if img.ndim != 5 or int(img.shape[-1]) != 4:
        raise ValueError(
            f"real-valued img must use XYZTV layout with V=4 (channels last), got {img.shape}"
        )
    return np.ascontiguousarray(img)


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
    correction=None,
    correction_high=None,
    phase_wrapped=None,
    phase_wrapped_high=None,
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
    correction_out = _ensure_optional_sigma_time(correction, nt)
    correction_high_out = _ensure_optional_sigma_time(correction_high, nt)
    phase_wrapped_out = _ensure_optional_sigma_time(phase_wrapped, nt)
    phase_wrapped_high_out = _ensure_optional_sigma_time(phase_wrapped_high, nt)
    if capabilities is None:
        capabilities = LoaderCapabilities(
            has_segmentation=seg_out is not None,
            has_tke=tke_out is not None,
            has_complex_source=False,
            supports_wss=False,
            supports_plane_metrics=False,
            has_wrapped_phase=phase_wrapped_out is not None,
            supports_phase_unwrap=phase_wrapped_out is not None,
        )
    else:
        capabilities = LoaderCapabilities(
            has_segmentation=bool(capabilities.has_segmentation),
            has_tke=bool(capabilities.has_tke),
            has_complex_source=bool(capabilities.has_complex_source),
            supports_wss=bool(capabilities.supports_wss),
            supports_plane_metrics=bool(capabilities.supports_plane_metrics),
            has_wrapped_phase=bool(getattr(capabilities, "has_wrapped_phase", False)),
            supports_phase_unwrap=bool(getattr(capabilities, "supports_phase_unwrap", False)),
        )
    capabilities.has_tke = bool(capabilities.has_tke or tke_out is not None)
    capabilities.has_wrapped_phase = bool(capabilities.has_wrapped_phase or phase_wrapped_out is not None)
    capabilities.supports_phase_unwrap = bool(capabilities.supports_phase_unwrap or phase_wrapped_out is not None)

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
        correction=correction_out,
        correction_high=correction_high_out,
        phase_wrapped=phase_wrapped_out,
        phase_wrapped_high=phase_wrapped_high_out,
        tke_array=tke_out,
        metadata=dict(metadata or {}),
        source_format=str(source_format or ""),
        source_group=source_group,
        capabilities=capabilities,
    )


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
