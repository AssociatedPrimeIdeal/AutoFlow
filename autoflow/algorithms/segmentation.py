import json
import os
from datetime import datetime, timezone

import h5py
import numpy as np
from scipy.ndimage import binary_closing, binary_opening

from .preprocess import largest_connected_component, remove_small_cc_from_binary_mask


def segmentation_timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def collapse_segmentation_to_3d(segmentation):
    seg = np.asarray(segmentation, dtype=np.int16)
    if seg.ndim == 3:
        return seg.copy()
    if seg.ndim != 4:
        raise ValueError(f"segmentation must be 3D or 4D, got shape={seg.shape}")
    return np.max(seg, axis=3).astype(np.int16)


def broadcast_segmentation_to_time(labels_3d, time_count):
    labels = np.asarray(labels_3d, dtype=np.int16)
    if labels.ndim != 3:
        raise ValueError(f"labels_3d must be 3D, got shape={labels.shape}")
    nt = max(1, int(time_count))
    return np.repeat(labels[..., None], nt, axis=3).astype(np.int16)


def normalize_segmentation_volume(segmentation, spatial_shape=None, time_count=1):
    seg = np.asarray(segmentation, dtype=np.int16)
    if spatial_shape is not None and tuple(seg.shape[:3]) != tuple(int(x) for x in spatial_shape):
        raise ValueError(
            f"segmentation spatial shape mismatch: got {seg.shape[:3]} expected {tuple(spatial_shape)}"
        )
    if seg.ndim == 3:
        return broadcast_segmentation_to_time(seg, time_count)
    if seg.ndim != 4:
        raise ValueError(f"segmentation must be 3D or 4D, got shape={seg.shape}")
    nt = max(1, int(time_count))
    if seg.shape[3] == nt:
        return seg.copy()
    if seg.shape[3] == 1:
        return np.repeat(seg, nt, axis=3).astype(np.int16)
    if nt == 1:
        return seg.copy()
    raise ValueError(f"segmentation time count mismatch: got {seg.shape[3]} expected {nt}")


def _collect_h5_datasets(handle):
    datasets = []

    def _visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            datasets.append((name, obj))

    handle.visititems(_visitor)
    return datasets


def _load_segmentation_from_h5(path):
    with h5py.File(path, "r") as f:
        preferred = []
        for name, ds in _collect_h5_datasets(f):
            lname = name.lower()
            if lname.endswith("segmentation") or lname.endswith("segmask"):
                preferred.append((name, ds))
        if preferred:
            return np.asarray(preferred[0][1][:], dtype=np.int16), preferred[0][0]
        datasets = _collect_h5_datasets(f)
        if len(datasets) == 1:
            return np.asarray(datasets[0][1][:], dtype=np.int16), datasets[0][0]
    raise ValueError(f"could not find a segmentation dataset in {path}")


def load_segmentation_file(path, spatial_shape=None, time_count=1):
    ext = os.path.splitext(path)[1].lower()
    dataset_name = ""
    if ext == ".npy":
        arr = np.asarray(np.load(path), dtype=np.int16)
    elif ext == ".npz":
        payload = np.load(path)
        keys = list(payload.keys())
        for key in ["segmentation", "segmask", "labels", "arr_0"]:
            if key in payload:
                dataset_name = key
                arr = np.asarray(payload[key], dtype=np.int16)
                break
        else:
            if not keys:
                raise ValueError(f"npz file has no arrays: {path}")
            dataset_name = keys[0]
            arr = np.asarray(payload[keys[0]], dtype=np.int16)
    elif ext in (".h5", ".hdf5"):
        arr, dataset_name = _load_segmentation_from_h5(path)
    else:
        raise ValueError(f"unsupported segmentation file type: {path}")

    seg = normalize_segmentation_volume(arr, spatial_shape=spatial_shape, time_count=time_count)
    provenance = {
        "source": "import",
        "path": os.path.abspath(path),
        "dataset": dataset_name,
        "created_at": segmentation_timestamp(),
    }
    return seg, provenance


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
    threshold_value = _otsu_threshold(scalar) if threshold == "auto" else float(threshold)
    mask = np.asarray(scalar >= threshold_value, dtype=bool)
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
        "threshold": threshold if threshold == "auto" else float(threshold_value),
        "threshold_value": float(threshold_value),
        "keep_largest_cc": bool(keep_largest_cc),
        "min_component_volume_mm3": float(min_component_volume_mm3),
        "closing": bool(closing),
        "opening": bool(opening),
        "created_at": segmentation_timestamp(),
    }
    return seg, provenance, scalar, float(threshold_value)


def save_segmentation_file(path, segmentation, resolution=None, origin=None, provenance=None):
    arr = np.asarray(segmentation, dtype=np.int16)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        np.save(path, arr)
        return path
    if ext == ".npz":
        np.savez_compressed(path, segmentation=arr)
        return path
    if ext not in (".h5", ".hdf5"):
        raise ValueError(f"unsupported segmentation save type: {path}")
    with h5py.File(path, "w") as f:
        f["segmentation"] = arr
        if resolution is not None:
            f["Resolution"] = np.asarray(resolution, dtype=np.float32).reshape(3)
        if origin is not None:
            f["Origin"] = np.asarray(origin, dtype=np.float32).reshape(3)
        if provenance:
            f.attrs["provenance_json"] = json.dumps(provenance, ensure_ascii=False)
    return path
