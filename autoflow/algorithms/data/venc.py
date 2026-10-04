"""VENC metadata and low/high-VENC alias-correction helpers."""

import numpy as np

from .h5_metadata import _coerce_h5_numeric_array, _read_h5_value


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


def _coerce_dual_venc_mode(value):
    """Normalize the user-facing dual-VENC source selection."""
    token = str(value or "dv").strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "lv": "lv",
        "low": "lv",
        "low-venc": "lv",
        "hv": "hv",
        "high": "hv",
        "high-venc": "hv",
        "dv": "dv",
        "dual": "dv",
        "dual-venc": "dv",
    }
    mode = aliases.get(token)
    if mode is None:
        raise ValueError(f"dual_venc_mode must be one of 'lv', 'hv', or 'dv', got {value!r}")
    return mode


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


def _dual_alias_shift_unique_values(alias_corr, lv_triplet):
    """Return the exact small set of rounded alias shifts without sorting all voxels."""
    corr = np.asarray(alias_corr, dtype=np.float32)
    venc = np.asarray(lv_triplet, dtype=np.float32).reshape(3)
    values = set()
    for component in range(min(3, corr.shape[0])):
        rounded = np.round(corr[component], decimals=6)
        candidates = (0.0, 2.0 * float(venc[component]), -2.0 * float(venc[component]),
                      4.0 * float(venc[component]), -4.0 * float(venc[component]))
        for candidate in candidates:
            candidate = float(np.round(candidate, decimals=6))
            if np.any(rounded == candidate):
                values.add(candidate)
    return sorted(values)
