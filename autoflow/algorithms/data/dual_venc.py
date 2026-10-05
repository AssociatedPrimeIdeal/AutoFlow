"""Legacy dual-VENC complex-H5 decoding and correction."""

import time
from queue import SimpleQueue
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from ...case_types import LoaderCapabilities
from ...task_control import check_cancelled, current_cancellation_token, task_scope
from ..phase_correction import apply_background_phase_correction_to_complex, background_phase_report_for_metadata

from .correction_cache import (
    _progress_prefix,
    _prepare_background_phase_corr_cache,
    _merge_background_phase_cache_report,
    _write_background_phase_corr_cache,
)
from .normalization import normalize_loaded_case
from .orientation import _reorient_component_signed, reorient
from .venc import (
    _coerce_dual_venc_mode,
    _dual_alias_shift_unique_values,
    _dual_venc_correct_alias,
    _dual_venc_triplet_ratio,
    _split_dual_venc_triplets,
)


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
    dual_venc_mode="dv",
    reuse_existing_corr=False,
):
    if img_complex.ndim != 5 or img_complex.shape[-1] != 7:
        raise ValueError(f"legacy dual-venc H5 expects XYZT7 complex data, got {img_complex.shape}")

    dual_venc_mode = _coerce_dual_venc_mode(dual_venc_mode)

    mag = np.abs(img_complex[..., 0]).astype(np.float32)
    lv_venc, hv_venc, lv_group_index, hv_group_index = _split_dual_venc_triplets(venc)
    encoded_groups = (img_complex[..., 1:4], img_complex[..., 4:7])
    lv_complex = np.concatenate([img_complex[..., :1], encoded_groups[lv_group_index]], axis=-1)
    hv_complex = np.concatenate([img_complex[..., :1], encoded_groups[hv_group_index]], axis=-1)
    cache_scopes = list(h5_scopes or ([h5_group] if h5_group is not None else []))
    lv_cfg, lv_cached_corr, lv_cache_report = _prepare_background_phase_corr_cache(
        cache_scopes, "corr_low", tuple(lv_complex.shape[:-1]) + (3,), cfg,
        expected_source_group=source_group, allow_untagged_root=allow_untagged_root,
        reuse_existing_corr=reuse_existing_corr,
        progress_callback=_progress_prefix(progress_callback, "h5_dual_lv_"),
    )
    hv_cfg, hv_cached_corr, hv_cache_report = _prepare_background_phase_corr_cache(
        cache_scopes, "corr_high", tuple(hv_complex.shape[:-1]) + (3,), cfg,
        expected_source_group=source_group, allow_untagged_root=allow_untagged_root,
        reuse_existing_corr=reuse_existing_corr,
        progress_callback=_progress_prefix(progress_callback, "h5_dual_hv_"),
    )

    # Worker threads enqueue progress; callbacks (including Qt widgets) run
    # on the calling thread while it waits for the two correction jobs.
    progress_events = SimpleQueue()
    callback = progress_events.put if progress_callback is not None else None

    def drain_progress():
        check_cancelled()
        while not progress_events.empty():
            progress_callback(progress_events.get())

    token = current_cancellation_token()
    def correct_encoding(complex_data, **kwargs):
        with task_scope(token):
            return apply_background_phase_correction_to_complex(complex_data, **kwargs)

    lv_kwargs = {
        "config": lv_cfg,
        "progress_callback": _progress_prefix(callback, "h5_dual_lv_"),
        "source_mode": "legacy_dual_venc_h5_low",
        "cached_corr": lv_cached_corr,
    }
    hv_kwargs = {
        "config": hv_cfg,
        "progress_callback": _progress_prefix(callback, "h5_dual_hv_"),
        "source_mode": "legacy_dual_venc_h5_high",
        "cached_corr": hv_cached_corr,
    }
    if bool(lv_cfg.enabled) or bool(hv_cfg.enabled):
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="autoflow-bgc") as executor:
            lv_future = executor.submit(
                correct_encoding,
                lv_complex,
                **lv_kwargs,
            )
            hv_future = executor.submit(
                correct_encoding,
                hv_complex,
                **hv_kwargs,
            )
            while not (lv_future.done() and hv_future.done()):
                drain_progress()
                time.sleep(0.01)
            drain_progress()
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
    _merge_background_phase_cache_report(lv_report, lv_cache_report, reuse_existing_corr)
    if (
        bool(lv_cfg.write_cache)
        and bool(lv_report.get("applied", False))
        and not bool(lv_report.get("cache_hit", False))
        and h5_group is not None
    ):
        lv_report["cache_written"] = bool(_write_background_phase_corr_cache(
            h5_group,
            "corr_low",
            lv_report,
            expected_source_group=source_group,
            progress_callback=_progress_prefix(progress_callback, "h5_dual_lv_"),
        ))

    hv_report["cache_name"] = "corr_high"
    _merge_background_phase_cache_report(hv_report, hv_cache_report, reuse_existing_corr)
    if (
        bool(hv_cfg.write_cache)
        and bool(hv_report.get("applied", False))
        and not bool(hv_report.get("cache_hit", False))
        and h5_group is not None
    ):
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
    # Preserve canonical wrapped phases for the optional traditional unwrap
    # stage.  The public loaded flow remains the dual-VENC reconstruction.
    phase_lv, _m_phase, _s_phase, _v_phase, _r_phase = reorient(
        mag, flow_lv_raw, segmask_for_reorient,
        venc=lv_venc, resolution=resolution,
        spatial_order=spatial_order, venc_order=venc_order,
        target_spatial_order=("LR", "AP", "FH"),
        target_venc_order=("LR", "AP", "FH"),
        return_velocity=False,
    )
    phase_hv, _m_phase_h, _s_phase_h, _v_phase_h, _r_phase_h = reorient(
        mag, flow_hv_raw, segmask_for_reorient,
        venc=hv_venc, resolution=resolution,
        spatial_order=spatial_order, venc_order=venc_order,
        target_spatial_order=("LR", "AP", "FH"),
        target_venc_order=("LR", "AP", "FH"),
        return_velocity=False,
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
    correction_low = None
    if isinstance(lv_report, dict) and lv_report.get("corr") is not None:
        correction_low = _reorient_component_signed(
            lv_report["corr"],
            spatial_order=spatial_order,
            target_spatial_order=("LR", "AP", "FH"),
            venc_order=venc_order,
            target_venc_order=("LR", "AP", "FH"),
        )
    correction_high = None
    if isinstance(hv_report, dict) and hv_report.get("corr") is not None:
        correction_high = _reorient_component_signed(
            hv_report["corr"],
            spatial_order=spatial_order,
            target_spatial_order=("LR", "AP", "FH"),
            venc_order=venc_order,
            target_venc_order=("LR", "AP", "FH"),
        )

    selected_flow = {
        "lv": flow_lv,
        "hv": flow_hv,
        "dv": flow_dual,
    }[dual_venc_mode]
    selected_venc = {
        "lv": lv_triplet,
        "hv": hv_triplet,
        "dv": hv_triplet,
    }[dual_venc_mode]
    selected_phase = {
        "lv": phase_lv,
        "hv": phase_hv,
        "dv": phase_lv,
    }[dual_venc_mode]

    meta = {
        "background_phase_correction": {
            "dual_venc_low": background_phase_report_for_metadata(lv_report),
            "dual_venc_high": background_phase_report_for_metadata(hv_report),
        },
        "spatial_order_raw": [str(x) for x in spatial_order[:3]],
        "venc_order_raw": [str(x) for x in venc_order[:3]],
        "is_dual_venc": True,
        "dual_venc_mode": dual_venc_mode,
        "dual_venc": {
            "enabled": True,
            "mode": dual_venc_mode,
            "selected_mode": dual_venc_mode,
            "input_channels": int(img_complex.shape[-1]),
            "lv_channel_indices": list(range(1 + 3 * lv_group_index, 4 + 3 * lv_group_index)),
            "hv_channel_indices": list(range(1 + 3 * hv_group_index, 4 + 3 * hv_group_index)),
            "lv_venc": lv_triplet.astype(float).tolist(),
            "hv_venc": hv_triplet.astype(float).tolist(),
            "ratio1": ratio1.astype(float).tolist(),
            "ratio2": ratio2.astype(float).tolist(),
            "hv_lv_ratio": ratio_hv_lv.astype(float).tolist(),
            "alias_shift_unique_cm_s": _dual_alias_shift_unique_values(dual_alias_corr, lv_triplet),
            "display_correction": {
                "low": correction_low is not None,
                "high": correction_high is not None,
                "units": "rad",
            },
        },
    }
    return normalize_loaded_case(
        flow=selected_flow,
        mag=mag_out,
        segmentation=seg_r if segmask is not None else None,
        resolution=np.asarray(res_new, dtype=float),
        origin=origin,
        venc=np.asarray(selected_venc, dtype=float),
        rr=float(rr),
        sigma=None,
        correction=correction_low,
        correction_high=correction_high,
        phase_wrapped=np.asarray(selected_phase, dtype=np.float32),
        phase_wrapped_high=np.asarray(phase_hv, dtype=np.float32),
        tke_array=None,
        metadata=meta,
        source_format="legacy_h5_dual_venc",
        capabilities=LoaderCapabilities(
            has_segmentation=segmask is not None,
            has_tke=False,
            has_complex_source=False,
            supports_wss=True,
            supports_plane_metrics=True,
            has_wrapped_phase=True,
            supports_phase_unwrap=True,
        ),
    )
