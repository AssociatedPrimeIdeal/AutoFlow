"""H5 layout dispatch and normalized case loading."""

import h5py
import numpy as np
from ...case_types import LoaderCapabilities
from ..phase_correction import apply_background_phase_correction_to_complex, apply_background_phase_correction_to_mag_flow, background_phase_report_for_metadata

from .correction_cache import (
    _loader_correction_config,
    _progress_prefix,
    _prepare_background_phase_corr_cache,
    _merge_background_phase_cache_report,
    _write_background_phase_corr_cache,
)
from .dual_venc import _load_legacy_dual_venc_h5
from .h5_metadata import (
    _coerce_h5_scalar_float,
    _coerce_h5_text_array,
    _coerce_h5_triplet,
    _discover_h5_data_group_names,
    _find_h5_dataset,
    _find_h5_dataset_from_scopes,
    _read_h5_value_from_scopes,
    _resolve_h5_data_group,
)
from .normalization import (
    _flow_looks_like_phase_radians,
    _normalize_real_img_layout,
    _sigma_from_complex,
    normalize_loaded_case,
)
from .orientation import (
    _reorient_component_abs,
    _reorient_component_signed,
    _reorient_real_valued_fields,
    reorient,
)
from .venc import _coerce_dual_venc_mode, _coerce_venc_array


def load_h5_data(
    path,
    correction_config=None,
    progress_callback=None,
    source_group=None,
    force_recompute_seg=False,
    ignore_embedded_segmentation=False,
    dual_venc_mode="dv",
    reuse_existing_corr=False,
):
    target_spatial_order = ("LR", "AP", "FH")
    target_venc_order = ("LR", "AP", "FH")
    dual_venc_mode = _coerce_dual_venc_mode(dual_venc_mode)
    cfg = _loader_correction_config(correction_config)
    h5_mode = "r+" if bool(cfg.enabled) and bool(cfg.write_cache) and not reuse_existing_corr else "r"
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
        if seg_ds is not None and bool(ignore_embedded_segmentation):
            seg_ds = None
        elif seg_ds is not None and bool(force_recompute_seg):
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
            "ignore_embedded_segmentation": bool(ignore_embedded_segmentation),
        }

        if img_complex_ds is not None:
            img_complex = np.asarray(img_complex_ds[:])
            if img_complex.ndim != 5 or int(img_complex.shape[-1]) not in (4, 7):
                raise ValueError(
                    f"legacy complex H5 expects XYZT4 or XYZT7 complex data (channels last), got {img_complex.shape}"
                )
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
                    dual_venc_mode=dual_venc_mode,
                    reuse_existing_corr=reuse_existing_corr,
                )
                loaded.source_group = group_name
                loaded.metadata = dict(loaded.metadata or {})
                loaded.metadata.setdefault("h5_layout", "complex_img")
                return loaded
            cfg, cached_corr, cache_report = _prepare_background_phase_corr_cache(
                scopes, "corr", tuple(img_complex.shape[:-1]) + (3,), cfg,
                expected_source_group=group_name, allow_untagged_root=allow_untagged_root,
                reuse_existing_corr=reuse_existing_corr,
                progress_callback=_progress_prefix(progress_callback, "h5_"),
            )
            img_complex_corr, _stationary_mask_raw, corr_report = apply_background_phase_correction_to_complex(
                img_complex,
                config=cfg,
                progress_callback=_progress_prefix(progress_callback, "h5_"),
                source_mode="legacy_complex_h5",
                cached_corr=cached_corr,
            )
            corr_report["cache_name"] = "corr"
            _merge_background_phase_cache_report(corr_report, cache_report, reuse_existing_corr)
            if bool(cfg.write_cache) and bool(corr_report.get("applied", False)) and not bool(corr_report.get("cache_hit", False)):
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
            # Reorient once in phase units.  The velocity conversion performed
            # by ``reorient(return_velocity=True)`` is a final scale; keeping
            # the phase result from this same call avoids a second full-volume
            # transpose/flip while preserving the original operation order.
            phase_wrapped, mag_out, seg_r, venc_new, res_new = reorient(
                mag,
                flow_raw,
                segmask_for_reorient,
                venc=VENC,
                resolution=resolution,
                spatial_order=spatial_order,
                venc_order=venc_order,
                target_spatial_order=target_spatial_order,
                target_venc_order=target_venc_order,
                return_velocity=False,
            )
            flow = (np.asarray(phase_wrapped, dtype=np.float32) / np.pi) * np.asarray(venc_new, dtype=np.float32).reshape((1, 1, 1, 1, 3))
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
                correction=(
                    _reorient_component_signed(
                        corr_report["corr"],
                        spatial_order=spatial_order,
                        target_spatial_order=target_spatial_order,
                        venc_order=venc_order,
                        target_venc_order=target_venc_order,
                    )
                    if isinstance(corr_report, dict) and corr_report.get("corr") is not None else None
                ),
                phase_wrapped=np.asarray(phase_wrapped, dtype=np.float32),
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
            if normalized_layout:
                flow_raw = np.asarray(flow_ds[:], dtype=np.float32)
                mag_raw = np.asarray(mag_ds[:], dtype=np.float32)
                layout_name = "normalized_h5"
            else:
                img = np.asarray(img_ds[:])
                if np.issubdtype(img.dtype, np.complexfloating):
                    raise ValueError(f"unsupported complex img layout: {path}")
                img = _normalize_real_img_layout(img)
                mag_raw = np.asarray(img[..., 0], dtype=np.float32)
                flow_raw = np.asarray(img[..., 1:4], dtype=np.float32)
                flow_is_phase_radians = _flow_looks_like_phase_radians(flow_raw, VENC)
                layout_name = "combined_img_real"
            # Preserve the raw phase before reorientation/rescaling.  The helper
            # below may convert radians to velocity and permute axes, so using
            # its output to reconstruct ``phase_wrapped`` would double-transform
            # normalized real inputs.
            phase_source_raw = np.array(flow_raw, copy=True) if flow_is_phase_radians else None
            mag_source_raw = np.array(mag_raw, copy=True) if flow_is_phase_radians else None
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
            phase_wrapped_r = None
            if flow_is_phase_radians:
                seg_for_phase = segmask if segmask is not None else np.zeros(np.asarray(mag_raw).shape, dtype=np.int16)
                phase_wrapped_r, _mp, _sp, _vp, _rp = reorient(
                    np.asarray(mag_source_raw),
                    np.asarray(phase_source_raw),
                    seg_for_phase,
                    venc=VENC,
                    resolution=resolution,
                    spatial_order=spatial_order,
                    venc_order=venc_order,
                    target_spatial_order=target_spatial_order,
                    target_venc_order=target_venc_order,
                    return_velocity=False,
                    normalize_mag=False,
                )
            cfg, cached_corr, cache_report = _prepare_background_phase_corr_cache(
                scopes, "corr", tuple(flow_raw.shape[:-1]) + (3,), cfg,
                expected_source_group=group_name, allow_untagged_root=allow_untagged_root,
                reuse_existing_corr=reuse_existing_corr,
                progress_callback=_progress_prefix(progress_callback, "h5_"),
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
            _merge_background_phase_cache_report(corr_report, cache_report, reuse_existing_corr)
            if phase_wrapped_r is not None and corr_report.get("applied"):
                phase = np.pi * flow_corr / np.asarray(venc_new).reshape((1, 1, 1, 1, 3))
                phase_wrapped_r = np.angle(np.exp(1j * phase)).astype(np.float32)
            if bool(cfg.write_cache) and bool(corr_report.get("applied", False)) and not bool(corr_report.get("cache_hit", False)):
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
            return normalize_loaded_case(
                flow=flow_corr,
                mag=mag_raw,
                segmentation=segmask_r,
                resolution=res_new,
                origin=origin,
                venc=venc_new,
                rr=float(rr),
                sigma=sigma_r,
                correction=corr_report.get("corr") if isinstance(corr_report, dict) else None,
                phase_wrapped=phase_wrapped_r,
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
