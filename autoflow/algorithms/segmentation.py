import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_closing, binary_opening

from .data import (
    _canonical_h5_key,
    _find_h5_dataset_from_scopes,
    _h5_member_name_map,
    _resolve_h5_data_group,
    _reorient_spatial_only,
    reorient,
)
from .preprocess import largest_connected_component, remove_small_cc_from_binary_mask


_AUTOFLOW_INTERNAL_SPATIAL_ORDER = ("LR", "AP", "FH")
_AUTOFLOW_INTERNAL_VENC_ORDER = ("LR", "AP", "FH")
_NNUNET_TARGET_SPATIAL_ORDER = ("HF", "AP", "RL")
_NNUNET_TARGET_VENC_ORDER = ("HF", "AP", "RL")
_NNUNET_GPU_PREPROCESSING_ENV = "AUTOFLOW_NNUNET_GPU_PREPROCESSING"


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


def save_segmentation_to_source_h5(path, segmentation, resolution=None, origin=None, provenance=None, dataset_name="segmask", source_spatial_order=None, source_group=None):
    arr = np.asarray(segmentation, dtype=np.int16)
    source_order = tuple(str(x).upper() for x in (source_spatial_order or ()))
    if source_order and source_order != _AUTOFLOW_INTERNAL_SPATIAL_ORDER:
        arr = _reorient_spatial_only(
            arr,
            spatial_order=_AUTOFLOW_INTERNAL_SPATIAL_ORDER,
            target_spatial_order=source_order,
        )
    arr = np.ascontiguousarray(arr, dtype=np.int16)
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".h5", ".hdf5"):
        raise ValueError(f"source segmentation cache requires an H5 input: {path}")
    with h5py.File(path, "r+") as handle:
        group, group_name = _resolve_h5_data_group(handle, source_group=source_group)
        scopes = [group] if group is handle else [group, handle]
        existing = _find_h5_dataset_from_scopes(scopes, "segmask", "segmentation")
        target_group = group
        target_name = str(dataset_name or "segmask")
        if existing is not None:
            target_group = existing.parent
            target_name = str(existing.name.rsplit("/", 1)[-1])
            del target_group[target_name]
        else:
            name_map = _h5_member_name_map(target_group)
            actual = name_map.get(_canonical_h5_key(target_name))
            if actual is not None:
                del target_group[actual]
                target_name = actual
        ds = target_group.create_dataset(target_name, data=arr, compression="gzip")
        ds.attrs["autoflow_source"] = "auto_segmentation"
        ds.attrs["autoflow_created_at"] = segmentation_timestamp()
        if group_name is not None:
            ds.attrs["autoflow_source_group"] = str(group_name)
        if source_order:
            ds.attrs["SpatialOrder"] = np.asarray(source_order, dtype="S4")
        if resolution is not None:
            ds.attrs["Resolution"] = np.asarray(resolution, dtype=np.float32).reshape(3)
        if origin is not None:
            ds.attrs["Origin"] = np.asarray(origin, dtype=np.float32).reshape(3)
        if provenance:
            ds.attrs["provenance_json"] = json.dumps(provenance, ensure_ascii=False)
    return path

def _nnunet_normalize_channel_name(name):
    token = str(name or "").strip().lower()
    aliases = {
        "mag_mean": "mag_mean_xyz",
        "mean_mag": "mag_mean_xyz",
        "mag_std": "mag_std_xyz",
        "std_mag": "mag_std_xyz",
        "pcmra_mean": "pcmra_mean_xyz",
        "pcmra_std": "pcmra_std_xyz",
        "flow_x_mean": "flow_x_mean_xyz",
        "flow_y_mean": "flow_y_mean_xyz",
        "flow_z_mean": "flow_z_mean_xyz",
        "flow_mag_mean": "flow_mag_mean_xyz",
        "flow_x_std": "flow_x_std_xyz",
        "flow_y_std": "flow_y_std_xyz",
        "flow_z_std": "flow_z_std_xyz",
        "flow_mag_std": "flow_mag_std_xyz",
    }
    return aliases.get(token, token)


def _sanitize_nnunet_artifact_token(text, default="item"):
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text or "")).strip("._-")
    return token or str(default)


def _nnunet_artifact_paths(artifact_prefix, channel_names, file_ending):
    prefix = str(artifact_prefix or "").strip()
    if not prefix:
        return [], None
    prefix_path = Path(prefix)
    prefix_path.parent.mkdir(parents=True, exist_ok=True)
    feature_paths = []
    for idx, channel_name in enumerate(channel_names):
        channel_token = _sanitize_nnunet_artifact_token(
            _nnunet_normalize_channel_name(channel_name),
            default=f"channel_{idx:04d}",
        )
        feature_paths.append(
            prefix_path.parent / f"{prefix_path.name}_feature_{idx:04d}_{channel_token}{file_ending}"
        )
    prediction_path = prefix_path.parent / f"{prefix_path.name}{file_ending}"
    return feature_paths, prediction_path


def _nnunet_spatial_affine(resolution, spatial_shape):
    res = np.asarray(resolution, dtype=np.float32).reshape(-1)
    if res.size == 1:
        res = np.repeat(res, 3)
    shape = np.asarray(spatial_shape, dtype=np.int32).reshape(-1)
    if shape.size != 3:
        raise ValueError(f"spatial_shape must be length 3, got {tuple(shape.tolist())}")
    x_size, y_size, z_size = int(shape[0]), int(shape[1]), int(shape[2])
    dx, dy, dz = float(res[0]), float(res[1]), float(res[2])
    # Match the affine convention used by the nnUNet training/export scripts so
    # auto-seg inference sees the same voxel-to-world geometry.
    affine = np.array([
        [0.0, 0.0, -dz, dz * (z_size - 1) / 2.0],
        [0.0, -dy, 0.0, dy * (y_size - 1) / 2.0],
        [-dx, 0.0, 0.0, dx * (x_size - 1) / 2.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)
    return affine


def _ensure_nnunet_mag_flow(mag, flow):
    mag = np.asarray(mag, dtype=np.float32)
    flow = np.asarray(flow, dtype=np.float32)
    if flow.ndim == 4 and flow.shape[-1] == 3:
        flow = flow[..., np.newaxis, :]
    if flow.ndim != 5 or flow.shape[-1] != 3:
        raise ValueError(f"flow must be XYZTV or XYZV with 3 components, got shape={flow.shape}")
    nt = int(flow.shape[3])
    if mag.ndim == 3:
        mag = np.repeat(mag[..., np.newaxis], nt, axis=3)
    elif mag.ndim == 4 and mag.shape[3] == 1 and nt > 1:
        mag = np.repeat(mag, nt, axis=3)
    elif mag.ndim != 4:
        raise ValueError(f"mag must be XYZT or XYZ, got shape={mag.shape}")
    if mag.shape[3] != nt:
        if mag.shape[3] == 1:
            mag = np.repeat(mag, nt, axis=3)
        else:
            raise ValueError(f"mag time dimension {mag.shape[3]} does not match flow {nt}")
    return mag, flow


def _prepare_nnunet_inputs(mag, flow, resolution):
    seg_dummy = np.zeros(np.asarray(mag).shape[:3], dtype=np.int16)
    flow_r, mag_r, _seg_r, _venc_r, resolution_r = reorient(
        mag,
        flow,
        seg_dummy,
        venc=np.ones(3, dtype=np.float32),
        resolution=resolution,
        spatial_order=_AUTOFLOW_INTERNAL_SPATIAL_ORDER,
        venc_order=_AUTOFLOW_INTERNAL_VENC_ORDER,
        target_spatial_order=_NNUNET_TARGET_SPATIAL_ORDER,
        target_venc_order=_NNUNET_TARGET_VENC_ORDER,
        return_velocity=False,
        normalize_mag=False,
    )
    return (
        np.asarray(mag_r, dtype=np.float32),
        np.asarray(flow_r, dtype=np.float32),
        np.asarray(resolution_r, dtype=np.float32),
    )


def _restore_autoflow_segmentation(segmentation):
    seg = _reorient_spatial_only(
        segmentation,
        spatial_order=_NNUNET_TARGET_SPATIAL_ORDER,
        target_spatial_order=_AUTOFLOW_INTERNAL_SPATIAL_ORDER,
    )
    return np.asarray(seg, dtype=np.int16)


def _ordered_mapping_values(payload):
    if isinstance(payload, dict):
        def _sort_key(item):
            key = item[0]
            try:
                return (0, int(key))
            except Exception:
                return (1, str(key))

        return [value for _, value in sorted(payload.items(), key=_sort_key)]
    if isinstance(payload, list):
        return list(payload)
    raise ValueError(f"expected mapping or list, got {type(payload).__name__}")


def _load_nnunet_model_metadata(model_folder):
    model_path = Path(model_folder).expanduser()
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path
    model_path = model_path.resolve()
    if model_path.name.startswith("fold_") and not (model_path / "dataset.json").is_file():
        model_path = model_path.parent
    if not model_path.is_dir():
        raise FileNotFoundError(f"nnUNet model folder not found: {model_path}")

    dataset_json_path = model_path / "dataset.json"
    plans_json_path = model_path / "plans.json"
    if not dataset_json_path.is_file():
        raise FileNotFoundError(f"missing nnUNet dataset.json: {dataset_json_path}")
    if not plans_json_path.is_file():
        raise FileNotFoundError(f"missing nnUNet plans.json: {plans_json_path}")

    with dataset_json_path.open("r", encoding="utf-8") as f:
        dataset_json = json.load(f)
    if not isinstance(dataset_json, dict):
        raise ValueError(f"dataset.json must contain a JSON object: {dataset_json_path}")

    return model_path, dataset_json


def _gpu_preprocessing_enabled(device, environ=None):
    environment = os.environ if environ is None else environ
    token = str(environment.get(_NNUNET_GPU_PREPROCESSING_ENV, "auto") or "auto").strip().lower()
    if token in {"0", "false", "no", "off", "cpu", "disabled"}:
        return False
    return str(device or "").strip().lower() == "cuda"


def _link_nnunet_model_file(source, destination):
    source = Path(source).resolve()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(source, destination)
        return
    except OSError:
        pass
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _prepare_nnunet_gpu_model_folder(model_path, target_path, folds, checkpoint_name):
    model_path = Path(model_path).resolve()
    target_path = Path(target_path)
    target_path.mkdir(parents=True, exist_ok=True)

    with (model_path / "plans.json").open("r", encoding="utf-8") as handle:
        plans = json.load(handle)
    configurations = plans.get("configurations") or {}
    changed = 0
    for configuration in configurations.values():
        if not isinstance(configuration, dict) or "resampling_fn_data" not in configuration:
            continue
        original_kwargs = dict(configuration.get("resampling_fn_data_kwargs") or {})
        gpu_kwargs = {
            "is_seg": False,
            "num_threads": max(1, int(original_kwargs.get("num_threads", 4) or 4)),
            "device": "cuda",
            "memefficient_seg_resampling": False,
            "force_separate_z": original_kwargs.get("force_separate_z"),
            "mode": "linear",
            "aniso_axis_mode": "nearest-exact",
        }
        if "separate_z_anisotropy_threshold" in original_kwargs:
            gpu_kwargs["separate_z_anisotropy_threshold"] = original_kwargs[
                "separate_z_anisotropy_threshold"
            ]
        configuration["resampling_fn_data"] = "resample_torch_fornnunet"
        configuration["resampling_fn_data_kwargs"] = gpu_kwargs
        changed += 1
    if changed == 0:
        raise ValueError(f"nnUNet plans do not define an input resampler: {model_path / 'plans.json'}")
    plans["autoflow_gpu_input_resampling"] = {
        "enabled": True,
        "function": "resample_torch_fornnunet",
        "device": "cuda",
        "mode": "linear",
    }
    with (target_path / "plans.json").open("w", encoding="utf-8") as handle:
        json.dump(plans, handle, indent=2)

    _link_nnunet_model_file(model_path / "dataset.json", target_path / "dataset.json")
    for fold in folds:
        fold_name = f"fold_{fold}"
        source_checkpoint = model_path / fold_name / checkpoint_name
        if not source_checkpoint.is_file():
            raise FileNotFoundError(f"missing nnUNet checkpoint: {source_checkpoint}")
        _link_nnunet_model_file(
            source_checkpoint,
            target_path / fold_name / checkpoint_name,
        )
    return target_path


def _detect_nnunet_folds(model_path):
    folds = []
    has_fold_all = False
    for child in sorted(model_path.iterdir()):
        if not child.is_dir() or not child.name.startswith("fold_"):
            continue
        suffix = child.name[len("fold_"):]
        if suffix == "all":
            has_fold_all = True
            continue
        if suffix.isdigit():
            folds.append(int(suffix))
    if folds:
        return [str(fold) for fold in sorted(set(folds))]
    if has_fold_all:
        return ["all"]
    raise FileNotFoundError(f"no nnUNet fold_* folders found in {model_path}")


_NNUNET_DEFAULT_CHANNEL_ORDER = (
    "mag_std_xyz",
    "mag_mean_xyz",
    "pcmra_std_xyz",
    "pcmra_mean_xyz",
    "flow_x_mean_xyz",
    "flow_y_mean_xyz",
    "flow_z_mean_xyz",
    "flow_mag_mean_xyz",
    "flow_x_std_xyz",
    "flow_y_std_xyz",
    "flow_z_std_xyz",
    "flow_mag_std_xyz",
)


def _resolve_nnunet_channel_token(channel_name, channel_index):
    token = _nnunet_normalize_channel_name(channel_name)
    token = {
        "mag": "mag_mean_xyz",
        "pcmra": "pcmra_mean_xyz",
    }.get(token, token)
    if token in _NNUNET_DEFAULT_CHANNEL_ORDER:
        return token
    if 0 <= int(channel_index) < len(_NNUNET_DEFAULT_CHANNEL_ORDER):
        return _NNUNET_DEFAULT_CHANNEL_ORDER[int(channel_index)]
    raise ValueError(
        f"unsupported nnUNet channel '{channel_name}'. "
        f"Supported channels: {list(_NNUNET_DEFAULT_CHANNEL_ORDER)}"
    )


def _nnunet_channel_volumes(channel_names, mag, flow):
    tokens = [
        _resolve_nnunet_channel_token(channel_name, channel_index)
        for channel_index, channel_name in enumerate(channel_names)
    ]
    requested = set(tokens)
    channels = {}

    if "mag_mean_xyz" in requested:
        channels["mag_mean_xyz"] = np.mean(mag, axis=3)
    if "mag_std_xyz" in requested:
        channels["mag_std_xyz"] = np.std(mag, axis=3)

    for axis_name, axis_index in (("x", 0), ("y", 1), ("z", 2)):
        mean_key = f"flow_{axis_name}_mean_xyz"
        std_key = f"flow_{axis_name}_std_xyz"
        if mean_key in requested or std_key in requested:
            component = flow[..., axis_index]
            if mean_key in requested:
                channels[mean_key] = np.mean(component, axis=3)
            if std_key in requested:
                channels[std_key] = np.std(component, axis=3)

    speed_keys = {
        "flow_mag_mean_xyz",
        "flow_mag_std_xyz",
        "pcmra_mean_xyz",
        "pcmra_std_xyz",
    }
    if requested.intersection(speed_keys):
        speed = np.linalg.norm(flow, axis=-1)
        if "flow_mag_mean_xyz" in requested:
            channels["flow_mag_mean_xyz"] = np.mean(speed, axis=3)
        if "flow_mag_std_xyz" in requested:
            channels["flow_mag_std_xyz"] = np.std(speed, axis=3)
        if "pcmra_mean_xyz" in requested or "pcmra_std_xyz" in requested:
            pcmra = mag * speed
            if "pcmra_mean_xyz" in requested:
                channels["pcmra_mean_xyz"] = np.mean(pcmra, axis=3)
            if "pcmra_std_xyz" in requested:
                channels["pcmra_std_xyz"] = np.std(pcmra, axis=3)

    return [np.asarray(channels[token], dtype=np.float32) for token in tokens]


def _nnunet_channel_volume(channel_name, mag, flow, channel_index=0):
    token = _resolve_nnunet_channel_token(channel_name, channel_index)
    return _nnunet_channel_volumes([token], mag, flow)[0]


def _parse_nnunet_label_map(label_map_spec, model_labels):
    if label_map_spec is None:
        return {}
    if isinstance(label_map_spec, str):
        text = label_map_spec.strip()
        if not text:
            return {}
        try:
            payload = json.loads(text)
        except Exception as exc:
            raise ValueError(f"auto_label_map must be valid JSON: {exc}") from exc
    elif isinstance(label_map_spec, dict):
        payload = label_map_spec
    else:
        raise ValueError(f"auto_label_map must be JSON object or string, got {type(label_map_spec).__name__}")
    if not payload:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("auto_label_map must decode to a JSON object")

    name_to_id = {}
    if isinstance(model_labels, dict):
        for key, value in model_labels.items():
            try:
                label_id = int(value)
            except Exception:
                continue
            name_to_id[str(key).strip().lower()] = label_id

    mapping = {}
    for raw_key, raw_value in payload.items():
        try:
            target_id = int(raw_value)
        except Exception as exc:
            raise ValueError(f"auto_label_map values must be integers, got {raw_value!r}") from exc
        key_token = str(raw_key).strip()
        if key_token.lstrip("-").isdigit():
            source_id = int(key_token)
        else:
            lookup = key_token.lower()
            if lookup not in name_to_id:
                raise ValueError(
                    f"auto_label_map key {raw_key!r} is not numeric and not present in model labels"
                )
            source_id = int(name_to_id[lookup])
        mapping[source_id] = target_id
    return mapping


def _apply_label_map(segmentation, label_map):
    seg = np.asarray(segmentation, dtype=np.int16).copy()
    if not label_map:
        return seg
    for source_id, target_id in label_map.items():
        seg[seg == int(source_id)] = int(target_id)
    return seg


def _write_nifti_volume(volume, affine, path):
    img = nib.Nifti1Image(np.asarray(volume, dtype=np.float32), affine)
    nib.save(img, str(path))


def _write_nifti_segmentation(volume, affine, path):
    img = nib.Nifti1Image(np.asarray(volume, dtype=np.int16), affine)
    nib.save(img, str(path))


def _link_or_copy_file(source, destination):
    source = Path(source)
    destination = Path(destination)
    if source.resolve() == destination.resolve():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.unlink(missing_ok=True)
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _read_nifti_segmentation(path):
    arr = np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"expected 3D nnUNet segmentation, got shape={arr.shape}")
    return np.rint(arr).astype(np.int16)


def _run_subprocess(command, *, env=None, cwd=None, runner=None):
    runner = subprocess.run if runner is None else runner
    return runner(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
    )


def _subprocess_failure_details(result):
    stdout = str(getattr(result, "stdout", "") or "").strip()
    stderr = str(getattr(result, "stderr", "") or "").strip()
    combined = "\n".join(part for part in (stdout, stderr) if part)
    return combined[-2000:] if combined else f"exit code {getattr(result, 'returncode', 'unknown')}"


def _nnunet_inference_command(
    input_dir,
    output_dir,
    model_path,
    folds,
    checkpoint_name,
    resolved_device,
    num_processes_preprocessing,
    num_processes_segmentation_export,
    step_size,
    disable_tta,
):
    command = [
        *_nnunet_predict_command(),
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-m", str(model_path),
        "-f", *folds,
        "-chk", str(checkpoint_name),
        "-npp", str(int(num_processes_preprocessing)),
        "-nps", str(int(num_processes_segmentation_export)),
        "-device", str(resolved_device),
    ]
    if step_size is not None:
        command.extend(["-step_size", str(float(step_size))])
    if disable_tta:
        command.append("--disable_tta")
    return command


def _emit_progress(progress_callback, *, stage, message, current=None, total=None, elapsed_sec=None, **extra):
    if progress_callback is None:
        return
    payload = {
        "stage": str(stage or ""),
        "message": str(message or ""),
    }
    if current is not None:
        payload["current"] = int(current)
    if total is not None:
        payload["total"] = int(total)
    if elapsed_sec is not None:
        payload["elapsed_sec"] = float(elapsed_sec)
    payload.update(extra)
    progress_callback(payload)


def default_nnunet_model_folder():
    model_name = "nnUNetTrainerPartBalanced__nnUNetPlans__3d_fullres_iso1mm"
    return Path(__file__).resolve().parents[1] / "segmodel" / model_name


def _resolve_bundled_relative_path(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    if path.is_dir():
        return path.resolve()

    package_root = Path(__file__).resolve().parents[2]
    packaged_path = package_root / path
    bundled_default = default_nnunet_model_folder()
    legacy_default = Path("autoflow") / "segmodel" / bundled_default.name
    if path == legacy_default:
        return bundled_default
    if packaged_path.is_dir():
        return packaged_path.resolve()
    return path.resolve()


def resolve_nnunet_model_folder(model_folder=""):
    candidate = str(model_folder or "").strip()
    if candidate:
        return str(_resolve_bundled_relative_path(candidate))
    default_path = default_nnunet_model_folder()
    if default_path.is_dir():
        return str(default_path)
    raise FileNotFoundError(
        "bundled nnUNet model folder is missing: "
        f"{default_path}. Set an explicit model folder to override the bundled default"
    )


def _nnunet_predict_command():
    if getattr(sys, "frozen", False):
        return [sys.executable, "--autoflow-internal-nnunet-predict"]
    executable = shutil.which("nnUNetv2_predict_from_modelfolder")
    if executable:
        return [executable]
    return [
        sys.executable,
        "-c",
        (
            "from nnunetv2.inference.predict_from_raw_data import "
            "predict_entry_point_modelfolder as main; main()"
        ),
    ]


def resolve_auto_segmentation_device(device="cpu"):
    token = str(device or "").strip().lower()
    if token in {"", "auto", "cuda_if_available"}:
        try:
            import torch

            return "cuda" if bool(torch.cuda.is_available()) else "cpu"
        except Exception:
            return "cpu"
    if token in {"cpu", "cuda"}:
        return token
    return str(device)


def generate_nnunet_auto_segmentation(
    mag,
    flow,
    resolution,
    origin,
    model_folder,
    *,
    backend="nnUNet",
    checkpoint_name="checkpoint_final.pth",
    device="cpu",
    auto_label_map="",
    case_id="autoflow_case",
    step_size=0.5,
    disable_tta=True,
    num_processes_preprocessing=3,
    num_processes_segmentation_export=1,
    artifact_prefix="",
    runner=None,
    progress_callback=None,
):
    if str(backend or "").strip().lower() != "nnunet":
        raise ValueError(f"unsupported auto segmentation backend: {backend}")

    t_total_start = time.perf_counter()
    total_stages = 5
    _emit_progress(
        progress_callback,
        stage="autoseg_start",
        message="Resolving nnUNet model and input metadata...",
        current=0,
        total=total_stages,
        elapsed_sec=0.0,
    )

    model_folder = resolve_nnunet_model_folder(model_folder)
    resolved_device = resolve_auto_segmentation_device(device)
    mag, flow = _ensure_nnunet_mag_flow(mag, flow)
    time_count = int(flow.shape[3])
    mag_nnunet, flow_nnunet, resolution_nnunet = _prepare_nnunet_inputs(mag, flow, resolution)
    model_path, dataset_json = _load_nnunet_model_metadata(model_folder)
    channel_names = _ordered_mapping_values(dataset_json.get("channel_names") or dataset_json.get("modality") or {})
    if not channel_names:
        raise ValueError(f"model folder does not define any channel names: {model_path}")
    file_ending = str(dataset_json.get("file_ending", ".nii.gz"))
    if not file_ending.startswith("."):
        file_ending = f".{file_ending}"
    label_map = _parse_nnunet_label_map(auto_label_map, dataset_json.get("labels", {}))
    folds = _detect_nnunet_folds(model_path)
    affine = _nnunet_spatial_affine(resolution_nnunet, mag_nnunet.shape[:3])
    artifact_feature_paths, artifact_prediction_path = _nnunet_artifact_paths(
        artifact_prefix,
        channel_names,
        file_ending,
    )
    _emit_progress(
        progress_callback,
        stage="autoseg_model_ready",
        message=f"Resolved nnUNet model: {model_path.name} | device={resolved_device}",
        current=1,
        total=total_stages,
        elapsed_sec=time.perf_counter() - t_total_start,
        backend="nnUNet",
        device=str(resolved_device),
        checkpoint=str(checkpoint_name),
        model_folder=str(model_path),
    )

    with TemporaryDirectory(prefix="autoflow_nnunet_") as tmp_root:
        tmp_root = Path(tmp_root)
        input_dir = tmp_root / "input"
        output_dir = tmp_root / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        _emit_progress(
            progress_callback,
            stage="autoseg_prepare_inputs",
            message=f"Preparing {len(channel_names)} nnUNet input channel(s)...",
            current=2,
            total=total_stages,
            elapsed_sec=time.perf_counter() - t_total_start,
            detail_current=0,
            detail_total=len(channel_names),
        )
        channel_volumes = _nnunet_channel_volumes(channel_names, mag_nnunet, flow_nnunet)
        for idx, (channel_name, volume) in enumerate(zip(channel_names, channel_volumes)):
            input_path = input_dir / f"{case_id}_{idx:04d}{file_ending}"
            _write_nifti_volume(volume, affine, input_path)
            if idx < len(artifact_feature_paths):
                _link_or_copy_file(input_path, artifact_feature_paths[idx])
            _emit_progress(
                progress_callback,
                stage="autoseg_prepare_inputs",
                message=f"Writing nnUNet input channel {idx + 1}/{len(channel_names)}: {channel_name}",
                current=2,
                total=total_stages,
                elapsed_sec=time.perf_counter() - t_total_start,
                detail_current=idx + 1,
                detail_total=len(channel_names),
                channel_name=str(channel_name),
            )

        gpu_preprocessing_requested = _gpu_preprocessing_enabled(resolved_device)
        use_gpu_preprocessing = gpu_preprocessing_requested
        prediction_model_path = model_path
        gpu_preprocessing_fallback_reason = ""
        if use_gpu_preprocessing:
            try:
                prediction_model_path = _prepare_nnunet_gpu_model_folder(
                    model_path,
                    tmp_root / "model_gpu_preprocessing",
                    folds,
                    checkpoint_name,
                )
            except Exception as exc:
                use_gpu_preprocessing = False
                gpu_preprocessing_fallback_reason = f"{type(exc).__name__}: {exc}"
                _emit_progress(
                    progress_callback,
                    stage="autoseg_gpu_preprocessing_fallback",
                    message="Could not prepare CUDA input resampling; using CPU resampling...",
                    current=3,
                    total=total_stages,
                    elapsed_sec=time.perf_counter() - t_total_start,
                    preprocessing_device="cpu",
                    fallback_reason=gpu_preprocessing_fallback_reason,
                )
        command = _nnunet_inference_command(
            input_dir,
            output_dir,
            prediction_model_path,
            folds,
            checkpoint_name,
            resolved_device,
            num_processes_preprocessing,
            num_processes_segmentation_export,
            step_size,
            disable_tta,
        )
        preprocessing_device = "cuda" if use_gpu_preprocessing else "cpu"
        gpu_preprocessing_command = list(command) if use_gpu_preprocessing else []
        _emit_progress(
            progress_callback,
            stage="autoseg_run_inference",
            message=(
                "Running nnUNet inference with CUDA input resampling..."
                if use_gpu_preprocessing
                else f"Running nnUNet inference on {resolved_device}..."
            ),
            current=3,
            total=total_stages,
            elapsed_sec=time.perf_counter() - t_total_start,
            command=[str(x) for x in command],
            preprocessing_device=preprocessing_device,
        )
        result = _run_subprocess(command, runner=runner)
        if getattr(result, "returncode", 0) != 0 and use_gpu_preprocessing:
            gpu_preprocessing_fallback_reason = _subprocess_failure_details(result)
            output_dir = tmp_root / "output_cpu_preprocessing"
            output_dir.mkdir(parents=True, exist_ok=True)
            command = _nnunet_inference_command(
                input_dir,
                output_dir,
                model_path,
                folds,
                checkpoint_name,
                resolved_device,
                num_processes_preprocessing,
                num_processes_segmentation_export,
                step_size,
                disable_tta,
            )
            preprocessing_device = "cpu"
            _emit_progress(
                progress_callback,
                stage="autoseg_gpu_preprocessing_fallback",
                message="CUDA input resampling failed; retrying with CPU resampling...",
                current=3,
                total=total_stages,
                elapsed_sec=time.perf_counter() - t_total_start,
                command=[str(x) for x in command],
                preprocessing_device=preprocessing_device,
                fallback_reason=gpu_preprocessing_fallback_reason,
            )
            result = _run_subprocess(command, runner=runner)
        if getattr(result, "returncode", 0) != 0:
            stdout = getattr(result, "stdout", "") or ""
            stderr = getattr(result, "stderr", "") or ""
            gpu_failure = ""
            if gpu_preprocessing_fallback_reason:
                gpu_failure = (
                    "\nCUDA preprocessing attempt failed before the CPU retry:\n"
                    f"{gpu_preprocessing_fallback_reason}\n"
                )
            raise RuntimeError(
                "nnUNet inference failed\n"
                f"command: {' '.join(map(str, command))}\n"
                f"{gpu_failure}"
                f"stdout:\n{stdout}\n"
                f"stderr:\n{stderr}"
            )

        prediction_path = output_dir / f"{case_id}{file_ending}"
        if not prediction_path.is_file() and file_ending == ".nii.gz":
            prediction_path = output_dir / f"{case_id}.nii.gz"
        if not prediction_path.is_file():
            raise FileNotFoundError(f"nnUNet prediction not found: {prediction_path}")

        _emit_progress(
            progress_callback,
            stage="autoseg_read_prediction",
            message="Reading nnUNet prediction...",
            current=4,
            total=total_stages,
            elapsed_sec=time.perf_counter() - t_total_start,
            prediction_file=str(prediction_path),
        )
        seg_3d_nnunet = _read_nifti_segmentation(prediction_path)
        seg_3d_nnunet = _apply_label_map(seg_3d_nnunet, label_map)
        if artifact_prediction_path is not None:
            _write_nifti_segmentation(seg_3d_nnunet, affine, artifact_prediction_path)
        seg_3d = _restore_autoflow_segmentation(seg_3d_nnunet)
        seg_4d = broadcast_segmentation_to_time(seg_3d, time_count)
        elapsed_total = time.perf_counter() - t_total_start
        _emit_progress(
            progress_callback,
            stage="autoseg_finalize",
            message="Finalizing auto segmentation volume...",
            current=5,
            total=total_stages,
            elapsed_sec=elapsed_total,
            prediction_file=str(prediction_path),
        )
        provenance = {
            "source": "auto",
            "backend": "nnUNet",
            "model_folder": str(model_path),
            "checkpoint": str(checkpoint_name),
            "device": str(resolved_device),
            "folds": list(folds),
            "channel_names": list(channel_names),
            "nnunet_spatial_order": list(_NNUNET_TARGET_SPATIAL_ORDER),
            "nnunet_venc_order": list(_NNUNET_TARGET_VENC_ORDER),
            "autoflow_internal_spatial_order": list(_AUTOFLOW_INTERNAL_SPATIAL_ORDER),
            "autoflow_internal_venc_order": list(_AUTOFLOW_INTERNAL_VENC_ORDER),
            "label_map": {str(k): int(v) for k, v in label_map.items()},
            "case_id": str(case_id),
            "created_at": segmentation_timestamp(),
            "command": [str(x) for x in command],
            "gpu_preprocessing_command": [str(x) for x in gpu_preprocessing_command],
            "preprocessing_device_requested": "cuda" if gpu_preprocessing_requested else "cpu",
            "preprocessing_device": str(preprocessing_device),
            "preprocessing_resampler": (
                "resample_torch_fornnunet"
                if preprocessing_device == "cuda"
                else "resample_data_or_seg_to_shape"
            ),
            "gpu_preprocessing_fallback": bool(gpu_preprocessing_fallback_reason),
            "gpu_preprocessing_fallback_reason": str(gpu_preprocessing_fallback_reason),
            "feature_files": [str(path) for path in artifact_feature_paths],
            "segmentation_nifti": "" if artifact_prediction_path is None else str(artifact_prediction_path),
            "prediction_file": "" if artifact_prediction_path is None else str(artifact_prediction_path),
            "elapsed_sec": float(elapsed_total),
        }
        return seg_4d, provenance
