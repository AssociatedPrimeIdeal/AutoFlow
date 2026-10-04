"""Reference-scalar selection and threshold-based segmentation."""

import numpy as np
from scipy.ndimage import binary_closing, binary_opening
from ..preprocess import largest_connected_component, remove_small_cc_from_binary_mask

from ._common import broadcast_segmentation_to_time, segmentation_timestamp


def _otsu_threshold(values):
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return 0.0
    vmin = float(np.min(data))
    vmax = float(np.max(data))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
        return vmin
    hist, edges = np.histogram(data, bins=256, range=(vmin, vmax))
    centers = 0.5 * (edges[:-1] + edges[1:])
    weight1 = np.cumsum(hist, dtype=float)
    weight2 = float(np.sum(hist)) - weight1
    moment1 = np.cumsum(hist * centers, dtype=float)
    valid1 = weight1 > 0
    valid2 = weight2 > 0
    mean1 = np.zeros_like(moment1, dtype=float)
    mean2 = np.zeros_like(moment1, dtype=float)
    mean1[valid1] = moment1[valid1] / weight1[valid1]
    rem = moment1[-1] - moment1
    mean2[valid2] = rem[valid2] / weight2[valid2]
    between = weight1[:-1] * weight2[:-1] * (mean1[:-1] - mean2[:-1]) ** 2
    if between.size == 0:
        return float(centers[0])
    return float(centers[int(np.argmax(between))])


def _finite_scalar_max(values):
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return 0.0
    return float(np.max(data))


def _resolve_threshold_config(values, threshold):
    if isinstance(threshold, dict):
        mode = str(threshold.get("mode", "manual") or "manual").strip().lower()
        if mode == "auto":
            threshold = "auto"
        else:
            min_percent = float(threshold.get("min_percent", 10.0))
            max_percent = float(threshold.get("max_percent", 100.0))
            if not np.isfinite(min_percent) or not np.isfinite(max_percent):
                raise ValueError("manual threshold percentages must be finite")
            if max_percent < min_percent:
                raise ValueError("manual threshold max percent must be greater than or equal to min percent")
            scalar_max = _finite_scalar_max(values)
            return {
                "mode": "manual_percent",
                "min_percent": float(min_percent),
                "max_percent": float(max_percent),
                "scalar_max": float(scalar_max),
                "min_value": float(scalar_max * (min_percent / 100.0)),
                "max_value": float(scalar_max * (max_percent / 100.0)),
                "spec": {
                    "mode": "manual",
                    "min_percent": float(min_percent),
                    "max_percent": float(max_percent),
                },
            }
    if isinstance(threshold, str) and threshold.strip().lower() == "auto":
        threshold_value = _otsu_threshold(values)
        return {
            "mode": "auto",
            "min_value": float(threshold_value),
            "max_value": None,
            "spec": "auto",
        }
    threshold_value = float(threshold)
    return {
        "mode": "manual_absolute",
        "min_value": float(threshold_value),
        "max_value": None,
        "spec": float(threshold_value),
    }


def compute_reference_scalar(mag, flow, scalar_name):
    scalar_name = str(scalar_name or "pcmra").lower()
    mag_arr = np.asarray(mag, dtype=np.float32)
    if mag_arr.ndim == 3:
        mag_arr = mag_arr[..., None]
    if scalar_name == "mag":
        return np.mean(mag_arr, axis=3).astype(np.float32)
    if flow is None:
        raise ValueError(f"{scalar_name} requires flow data")
    flow_arr = np.asarray(flow, dtype=np.float32)
    if flow_arr.ndim != 5:
        raise ValueError(f"flow must be XYZT3, got shape={flow_arr.shape}")
    speed = np.linalg.norm(flow_arr, axis=-1)
    pcmra = mag_arr * speed
    if scalar_name == "pcmra":
        return np.mean(pcmra, axis=3).astype(np.float32)
    if scalar_name == "pcmra_std":
        return np.std(pcmra, axis=3).astype(np.float32)
    raise ValueError(f"unsupported threshold scalar: {scalar_name}")


def generate_threshold_segmentation(
    mag,
    flow,
    resolution,
    time_count,
    scalar_name="pcmra",
    threshold="auto",
    keep_largest_cc=False,
    min_component_volume_mm3=0.0,
    closing=False,
    opening=False,
):
    scalar = compute_reference_scalar(mag, flow, scalar_name)
    threshold_info = _resolve_threshold_config(scalar, threshold)
    mask = np.asarray(scalar >= float(threshold_info["min_value"]), dtype=bool)
    if threshold_info.get("max_value") is not None:
        mask &= np.asarray(scalar <= float(threshold_info["max_value"]), dtype=bool)
    if closing:
        mask = binary_closing(mask).astype(bool)
    if opening:
        mask = binary_opening(mask).astype(bool)
    if keep_largest_cc:
        mask = largest_connected_component(mask)
    if float(min_component_volume_mm3) > 0:
        mask = remove_small_cc_from_binary_mask(mask, resolution, float(min_component_volume_mm3))
    labels_3d = np.asarray(mask, dtype=np.int16)
    seg = broadcast_segmentation_to_time(labels_3d, time_count)
    provenance = {
        "source": "threshold",
        "scalar": str(scalar_name),
        "threshold": threshold_info["spec"],
        "threshold_mode": str(threshold_info["mode"]),
        "threshold_value": float(threshold_info["min_value"]),
        "threshold_value_min": float(threshold_info["min_value"]),
        "keep_largest_cc": bool(keep_largest_cc),
        "min_component_volume_mm3": float(min_component_volume_mm3),
        "closing": bool(closing),
        "opening": bool(opening),
        "created_at": segmentation_timestamp(),
    }
    if threshold_info.get("max_value") is not None:
        provenance["threshold_value_max"] = float(threshold_info["max_value"])
    if threshold_info["mode"] == "manual_percent":
        provenance["threshold_percent_min"] = float(threshold_info["min_percent"])
        provenance["threshold_percent_max"] = float(threshold_info["max_percent"])
        provenance["scalar_max"] = float(threshold_info["scalar_max"])
    return seg, provenance, scalar, threshold_info
