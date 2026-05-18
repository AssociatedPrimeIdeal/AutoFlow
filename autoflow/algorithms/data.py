import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np

from ..case_types import BackgroundPhaseCorrectionConfig, LoadedCase, LoaderCapabilities
from .phase_correction import (
    apply_background_phase_correction_to_complex,
    apply_background_phase_correction_to_mag_flow,
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
             target_spatial_order, target_venc_order, return_velocity=False):
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
    flow_r = flow_r * sign3.reshape((1, 1, 1, 1, -1))

    if return_velocity:
        flow_r = (flow_r / np.pi) * venc_r.reshape((1, 1, 1, 1, -1))

    mag_max = np.max(np.abs(mag_r))
    if mag_max > 0:
        mag_r = mag_r / mag_max

    return flow_r, mag_r, seg_r, venc_r, resolution_r


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


def _progress_prefix(progress_callback, prefix):
    if progress_callback is None:
        return None

    def _wrapped(payload):
        data = dict(payload or {})
        stage = str(data.get("stage", ""))
        data["stage"] = f"{prefix}{stage}" if prefix else stage
        progress_callback(data)

    return _wrapped


def load_h5_data(path, correction_config=None, progress_callback=None):
    target_spatial_order = ("LR", "AP", "FH")
    target_venc_order = ("LR", "AP", "FH")
    cfg = coerce_background_phase_correction_config(correction_config)
    with h5py.File(path, "r") as g:
        VENC = g["VENC"][:] if "VENC" in g else np.array([150, 150, 150], dtype=float)
        resolution = g["Resolution"][:] if "Resolution" in g else np.array([1, 1, 1], dtype=float)
        origin = g["Origin"][:] if "Origin" in g else np.array([0.0, 0.0, 0.0], dtype=float)
        rr = float(g["RR"][()]) if "RR" in g else 1000.0
        spatial_order = g["SpatialOrder"][:].astype(str) if "SpatialOrder" in g else np.array(["FH", "AP", "LR"])
        venc_order = g["VENCOrder"][:].astype(str) if "VENCOrder" in g else np.array(["FH", "AP", "LR"])
        seg_name = "segmask" if "segmask" in g else "segmentation" if "segmentation" in g else None

        if "img_complex" in g:
            img_ds = g["img_complex"]
            if seg_name is not None:
                segmask_ds = g[seg_name]
                segmask_full = segmask_ds[:].astype(np.int16)
                if cfg.enabled:
                    src_slices = tuple(slice(0, int(img_ds.shape[i])) for i in range(3))
                    segmask = segmask_full
                else:
                    src_slices = _compute_spatial_bbox(segmask_full, pad=2)
                    segmask = segmask_ds[src_slices + (slice(None),) * (segmask_ds.ndim - 3)].astype(np.int16)
                del segmask_full
            else:
                src_slices = tuple(slice(0, int(img_ds.shape[i])) for i in range(3))
                segmask = None

            img_complex = np.asarray(img_ds[src_slices + (slice(None),) * (img_ds.ndim - 3)])
            img_complex_corr, _stationary_mask_raw, corr_report = apply_background_phase_correction_to_complex(
                img_complex,
                config=cfg,
                progress_callback=_progress_prefix(progress_callback, "h5_"),
                source_mode="legacy_complex_h5",
            )
            img_complex_use = img_complex_corr if bool(corr_report.get("applied", False)) else img_complex
            mag = np.abs(img_complex[..., 0]).astype(np.float32)
            flow_raw = np.angle(img_complex_use[..., 1:4] * np.conj(img_complex_use[..., 0][..., None])).astype(np.float32)
            sigma_raw = _sigma_from_complex(img_complex_use, VENC)
            segmask_for_reorient = segmask if segmask is not None else np.zeros(mag.shape, dtype=np.int16)
            flow, mag_out, seg_r, venc_new, res_new = reorient(
                mag, flow_raw, segmask_for_reorient, venc=VENC, resolution=resolution,
                spatial_order=spatial_order, venc_order=venc_order,
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
            meta = {
                "background_phase_correction": background_phase_report_for_metadata(corr_report),
            }
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
                capabilities=LoaderCapabilities(
                    has_segmentation=segmask is not None,
                    has_tke=True,
                    has_complex_source=True,
                    supports_wss=True,
                    supports_plane_metrics=True,
                ),
            )

        if "flow" in g and "mag" in g:
            sigma = np.asarray(g["sigma"][:], dtype=np.float32) if "sigma" in g else None
            tke_array = np.asarray(g["tke_array"][:], dtype=np.float32) if "tke_array" in g else None
            segmentation = None if seg_name is None else np.asarray(g[seg_name][:], dtype=np.int16)
            flow_raw = np.asarray(g["flow"][:], dtype=np.float32)
            mag_raw = np.asarray(g["mag"][:], dtype=np.float32)
            flow_corr, _stationary_mask, corr_report = apply_background_phase_correction_to_mag_flow(
                mag_raw,
                flow_raw,
                VENC,
                config=cfg,
                progress_callback=_progress_prefix(progress_callback, "h5_"),
            )
            return normalize_loaded_case(
                flow=flow_corr,
                mag=mag_raw,
                segmentation=segmentation,
                resolution=resolution,
                origin=origin,
                venc=VENC,
                rr=float(rr),
                sigma=sigma,
                tke_array=tke_array,
                metadata={"background_phase_correction": background_phase_report_for_metadata(corr_report)},
                source_format="normalized_h5",
                capabilities=LoaderCapabilities(
                    has_segmentation=segmentation is not None,
                    has_tke=tke_array is not None,
                    has_complex_source=False,
                    supports_wss=True,
                    supports_plane_metrics=True,
                ),
            )

        raise ValueError(f"unsupported h5 layout: {path}")
