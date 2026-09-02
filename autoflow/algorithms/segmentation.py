import json
import hashlib
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
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
_NNUNET_4D_BACKENDS = {"nnunet4d", "nnunet_4d", "4d"}
_NNUNET_4D_PIPELINE_DEFAULT = Path(
    "/nas-data2/ryy/CMR4DFlow2026/Segdata/scripts/nnunet/4D/"
    "run_7020_4d_full_ssd_20260824.sh"
)
_NNUNET_4D_RESULTS_DEFAULT = Path("/nas-data/ryy_rawdata/aorta_seg/nnres")
_NNUNET_4D_DATASET_DEFAULT = "Dataset7020_Aorta_4DTemporalFT"
_NNUNET_4D_TRAINER_DEFAULT = "nnUNetTrainerPartBalancedTversky"
_NNUNET_4D_PLANS_DEFAULT = "nnUNetPlansIso1mm"
_NNUNET_4D_CONFIGURATION_DEFAULT = "3d_fullres"


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
    path = str(path)
    lower_path = path.lower()
    ext = ".nii.gz" if lower_path.endswith(".nii.gz") else os.path.splitext(path)[1].lower()
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
    elif ext in (".nii", ".nii.gz"):
        image = nib.load(path)
        arr = np.asarray(image.get_fdata(), dtype=np.float32)
        if arr.ndim not in (3, 4):
            raise ValueError(f"NIfTI segmentation must be 3D or 4D, got shape={arr.shape}")
        arr = np.rint(arr).astype(np.int16)
        dataset_name = "nifti"
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
    path = str(path)
    lower_path = path.lower()
    ext = ".nii.gz" if lower_path.endswith(".nii.gz") else os.path.splitext(path)[1].lower()
    if ext == ".npy":
        np.save(path, arr)
        return path
    if ext == ".npz":
        np.savez_compressed(path, segmentation=arr)
        return path
    if ext in (".nii", ".nii.gz"):
        spacing = np.asarray(resolution if resolution is not None else (1.0, 1.0, 1.0), dtype=float).reshape(-1)
        spacing = np.where(np.isfinite(spacing[:3]) & (spacing[:3] > 0), spacing[:3], 1.0)
        translation = np.asarray(origin if origin is not None else (0.0, 0.0, 0.0), dtype=float).reshape(-1)
        translation = np.where(np.isfinite(translation[:3]), translation[:3], 0.0)
        affine = np.eye(4, dtype=float)
        affine[:3, :3] = np.diag(spacing)
        affine[:3, 3] = translation
        image = nib.Nifti1Image(arr, affine)
        if provenance:
            image.header["descrip"] = str(json.dumps(provenance, ensure_ascii=False))[:79]
        nib.save(image, path)
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


def save_nifti_volume(path, volume, resolution=None, origin=None):
    """Write a scalar image sequence for external editors such as SpatioTemporal Labeler."""
    arr = np.asarray(volume)
    if arr.ndim not in (3, 4):
        raise ValueError(f"NIfTI volume must be 3D or 4D, got shape={arr.shape}")
    spacing = np.asarray(resolution if resolution is not None else (1.0, 1.0, 1.0), dtype=float).reshape(-1)
    spacing = np.where(np.isfinite(spacing[:3]) & (spacing[:3] > 0), spacing[:3], 1.0)
    translation = np.asarray(origin if origin is not None else (0.0, 0.0, 0.0), dtype=float).reshape(-1)
    translation = np.where(np.isfinite(translation[:3]), translation[:3], 0.0)
    affine = np.eye(4, dtype=float)
    affine[:3, :3] = np.diag(spacing)
    affine[:3, 3] = translation
    nib.save(nib.Nifti1Image(np.asarray(arr, dtype=np.float32), affine), str(path))
    return str(path)


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
        "flow_speed_mean": "flow_mag_mean_xyz",
        "flow_x_std": "flow_x_std_xyz",
        "flow_y_std": "flow_y_std_xyz",
        "flow_z_std": "flow_z_std_xyz",
        "flow_mag_std": "flow_mag_std_xyz",
        "flow_speed_std": "flow_mag_std_xyz",
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
    token = str(environment.get(_NNUNET_GPU_PREPROCESSING_ENV, "cuda") or "cuda").strip().lower()
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
        if "resampling_fn_probabilities" in configuration:
            probability_kwargs = dict(gpu_kwargs)
            probability_kwargs.update({"is_seg": False, "mode": "linear"})
            configuration["resampling_fn_probabilities"] = "resample_torch_fornnunet"
            configuration["resampling_fn_probabilities_kwargs"] = probability_kwargs
        changed += 1
    if changed == 0:
        raise ValueError(f"nnUNet plans do not define an input resampler: {model_path / 'plans.json'}")
    plans["autoflow_gpu_input_resampling"] = {
        "enabled": True,
        "function": "resample_torch_fornnunet",
        "probabilities_function": "resample_torch_fornnunet",
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
        # Compute the norm once.  The 4D model requests several speed/PCMRA
        # channels and recalculating this volume for each channel is a large
        # avoidable allocation for clinical-size inputs.
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


def _parse_nnunet_temporal_channel(name):
    """Return ``(offset, feature)`` for a temporal channel name.

    Dataset7020 names channels as ``tm2_flow_x``, ``tp1_mag`` and ``tp0_pcmra``.
    The parser also accepts ``t-2_*``/``t+1_*`` forms so exported datasets can
    use a more readable spelling without changing the predictor.
    """
    token = str(name or "").strip().lower()
    match = re.match(r"^t([mp])(\d+)_(flow_[xyz]|mag|pcmra)$", token)
    if match:
        offset = int(match.group(2)) * (-1 if match.group(1) == "m" else 1)
        return offset, match.group(3)
    match = re.match(r"^t([+-]?\d+)_(flow_[xyz]|mag|pcmra)$", token)
    if match:
        return int(match.group(1)), match.group(2)
    return None


def _nnunet_4d_channel_volumes(
    channel_names, mag, flow, frame_index, global_by_name=None, temporal_cache=None
):
    """Build one 4D-model sample from a circular temporal neighbourhood."""
    mag = np.asarray(mag, dtype=np.float32)
    flow = np.asarray(flow, dtype=np.float32)
    if mag.ndim != 4 or flow.ndim != 5 or flow.shape[:4] != mag.shape:
        raise ValueError(
            f"4D nnUNet inputs must be mag XYZT and flow XYZT3, got {mag.shape} and {flow.shape}"
        )
    nt = int(mag.shape[3])
    if nt < 1:
        raise ValueError("4D nnUNet input has no time frames")
    # Cache all full-cycle statistics once per case.  Every frame shares these
    # twelve channels; only the temporal window changes.
    global_names = [
        "mag_std_xyz", "mag_mean_xyz", "pcmra_std_xyz", "pcmra_mean_xyz",
        "flow_x_mean_xyz", "flow_y_mean_xyz", "flow_z_mean_xyz", "flow_mag_mean_xyz",
        "flow_x_std_xyz", "flow_y_std_xyz", "flow_z_std_xyz", "flow_mag_std_xyz",
    ]
    if global_by_name is None:
        global_channels = _nnunet_channel_volumes(global_names, mag, flow)
        global_by_name = dict(zip(global_names, global_channels))
    speed = None
    pcmra = None
    result = []
    for index, raw_name in enumerate(channel_names):
        name = _nnunet_normalize_channel_name(raw_name)
        if name in global_by_name:
            result.append(global_by_name[name])
            continue
        parsed = _parse_nnunet_temporal_channel(name)
        if parsed is None:
            # Preserve the existing positional aliases for old 12-channel
            # models, while producing a useful error for a malformed 4D spec.
            if index < len(global_names):
                result.append(global_by_name[global_names[index]])
                continue
            raise ValueError(f"unsupported 4D nnUNet channel '{raw_name}'")
        offset, feature = parsed
        frame = (int(frame_index) + int(offset)) % nt
        if feature == "mag":
            result.append(
                mag[..., frame] if temporal_cache is None else temporal_cache["mag"][..., frame]
            )
        elif feature == "pcmra":
            if temporal_cache is not None:
                pcmra = temporal_cache["pcmra"]
            elif pcmra is None:
                if speed is None:
                    speed = np.linalg.norm(flow, axis=-1)
                pcmra = mag * speed
            result.append(pcmra[..., frame])
        else:
            if feature == "flow_x": component = 0
            elif feature == "flow_y": component = 1
            else: component = 2
            value = flow[..., frame, component]
            if temporal_cache is not None:
                value = temporal_cache[f"flow_{feature[-1]}"][..., frame]
            result.append(value)
    return [np.asarray(value, dtype=np.float32) for value in result]


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
    python_executable=None,
    bootstrap_resampler_path=None,
):
    command = [
        *_nnunet_predict_command(
            python_executable=python_executable,
            bootstrap_resampler_path=bootstrap_resampler_path,
        ),
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


def default_nnunet_4d_pipeline_script():
    """Return the optional local Dataset7020 training/prediction script."""
    return str(_NNUNET_4D_PIPELINE_DEFAULT)


def _model_folder_from_4d_pipeline_script(script_path):
    """Resolve a Dataset7020 model directory from its shell configuration.

    The supplied ``run_7020_4d_full_ssd_20260824.sh`` is a training/prediction
    orchestrator rather than an inference executable.  Parsing its stable
    ``NNUNET_RESULTS`` and ``DATASET`` assignments lets callers point
    ``auto_model`` at that script while AutoFlow still invokes nnUNet once per
    4D case.  Missing or non-standard assignments simply fall back to the
    documented Dataset7020 defaults.
    """
    path = Path(script_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"4D nnUNet pipeline script not found: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")

    def assignment(name, fallback):
        match = re.search(rf"^\s*{re.escape(name)}=\"([^\"]+)\"", text, flags=re.MULTILINE)
        return match.group(1) if match else fallback

    # Reuse the same local shell-assignment expansion used for the child
    # predictor environment, so ``NNUNET_RESULTS="${SSD_ROOT}/nnres"`` is
    # resolved before constructing the model folder.
    assignments = _nnunet_4d_pipeline_assignments(path)
    results_root = Path(
        assignments.get("NNUNET_RESULTS", assignment("NNUNET_RESULTS", str(_NNUNET_4D_RESULTS_DEFAULT)))
    ).expanduser()
    dataset = assignments.get("DATASET", assignment("DATASET", _NNUNET_4D_DATASET_DEFAULT))
    trainer = _NNUNET_4D_TRAINER_DEFAULT
    plans = _NNUNET_4D_PLANS_DEFAULT
    configuration = _NNUNET_4D_CONFIGURATION_DEFAULT
    model = results_root / dataset / f"{trainer}__{plans}__{configuration}"
    return model.resolve()


def _python_from_4d_pipeline_script(script_path):
    path = Path(script_path).expanduser().resolve()
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"PYTHON_BIN=\"\$\{PYTHON_BIN:-([^}]+)\}\"", text)
    candidate = str(match.group(1)).strip() if match else ""
    return candidate if candidate and Path(candidate).is_file() else ""


def _nnunet_4d_pipeline_assignments(script_path):
    """Read stable environment assignments from the Dataset7020 shell script."""
    path = Path(script_path).expanduser().resolve()
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    values = {}
    for name in ("SSD_ROOT", "NNUNET_RAW", "NNUNET_PREPROCESSED", "NNUNET_RESULTS", "DATASET"):
        match = re.search(rf"^\s*{re.escape(name)}=\"([^\"]+)\"", text, flags=re.MULTILINE)
        if match:
            values[name] = match.group(1)
    # The shell script intentionally derives cache paths from SSD_ROOT. Expand
    # only the local assignments so ${SSD_ROOT} never leaks into child env.
    for name, value in list(values.items()):
        for _ in range(4):
            replaced = re.sub(
                r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)",
                lambda match: values.get(match.group(1) or match.group(2), match.group(0)),
                value,
            )
            if replaced == value:
                break
            value = replaced
        values[name] = value
    return values


def _nnunet_4d_subprocess_env(pipeline_script, model_path):
    """Build an isolated nnUNet environment for the external 4D predictor."""
    env = os.environ.copy()
    script_path = Path(pipeline_script).expanduser().resolve() if pipeline_script else None
    if script_path is not None and script_path.is_file():
        # The training script imports project helpers and custom trainer modules
        # from both scripts/ and scripts/nnunet/. Keep the parent process env
        # untouched while making those imports available to the child.
        roots = [script_path.parent, script_path.parent.parent]
        existing = [item for item in str(env.get("PYTHONPATH", "")).split(os.pathsep) if item]
        env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys([*(str(root) for root in roots), *existing]))
        assignments = _nnunet_4d_pipeline_assignments(script_path)
        if assignments.get("NNUNET_RAW"):
            env["nnUNet_raw"] = assignments["NNUNET_RAW"]
        if assignments.get("NNUNET_PREPROCESSED"):
            env["nnUNet_preprocessed"] = assignments["NNUNET_PREPROCESSED"]
        if assignments.get("NNUNET_RESULTS"):
            env["nnUNet_results"] = assignments["NNUNET_RESULTS"]
    if model_path:
        # A model folder is always .../nnUNet_results/Dataset/TrainerConfig.
        # This fallback also handles scripts with a non-standard variable name.
        model_root = Path(model_path).expanduser().resolve().parent.parent
        env["nnUNet_results"] = str(model_root)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    return env


def _load_4d_pipeline_module(script_path):
    """Load the external grouped-preprocessing helpers when available."""
    path = Path(script_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"4D pipeline script not found: {path}")
    # ``run_seg2nndata_all.py`` discovers its project root from the
    # ``scripts/`` directory (which contains both ``methods/`` and
    # ``nnunet/``).  The 4D runner lives below ``scripts/nnunet/4D``.
    scripts_root = path.parent.parent.parent
    for root in (scripts_root, path.parent.parent, path.parent):
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    module_path = path.parent / "run_4dflow.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"grouped 4D runner not found: {module_path}")
    module_name = "autoflow_external_4dflow_" + hashlib.sha1(str(module_path).encode("utf-8")).hexdigest()[:12]
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load grouped 4D runner: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    # run_seg2nndata_all.py discovers its project root from cwd at import time.
    # Import it from the Dataset7020 scripts root, then restore AutoFlow's cwd.
    old_cwd = os.getcwd()
    try:
        os.chdir(scripts_root)
        spec.loader.exec_module(module)
    finally:
        os.chdir(old_cwd)
    return module


def _install_nnunet_4d_resampler(script_path):
    """Install the Dataset7020 GPU resampler into the active nnUNet package."""
    path = Path(script_path).expanduser().resolve()
    scripts_root = path.parent.parent.parent
    source_file = scripts_root / "nnunet" / "gpu_resampling.py"
    if not source_file.is_file():
        raise FileNotFoundError(f"Dataset7020 GPU resampler not found: {source_file}")
    import nnunetv2

    target_dir = Path(nnunetv2.__file__).resolve().parent / "preprocessing" / "resampling"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / "gpu_resampling.py"
    if not target_file.exists() or target_file.read_bytes() != source_file.read_bytes():
        shutil.copy2(source_file, target_file)
    importlib.invalidate_caches()
    return target_file


def _nnunet_4d_temporal_radius(channel_names):
    offsets = [parsed[0] for name in channel_names if (parsed := _parse_nnunet_temporal_channel(name))]
    if not offsets:
        raise ValueError("4D nnUNet model does not define temporal channels")
    radius = max(abs(int(value)) for value in offsets)
    expected = 12 + 5 * (2 * radius + 1)
    if len(channel_names) != expected:
        raise ValueError(
            f"4D nnUNet channel layout has {len(channel_names)} channels; "
            f"expected {expected} for temporal radius {radius}"
        )
    return radius


def _nnunet_4d_grouped_requested(runner=None):
    """Return whether the direct grouped predictor should be attempted."""
    token = str(os.environ.get("AUTOFLOW_NNUNET4D_GROUPED", "auto") or "auto").strip().lower()
    if token in {"0", "false", "no", "off", "standard", "subprocess"}:
        return False
    # A test/custom runner intentionally exercises the subprocess contract.
    return runner is None


def _generate_nnunet_4d_grouped(
    mag_nnunet,
    flow_nnunet,
    resolution_nnunet,
    model_path,
    dataset_json,
    selected_folds,
    checkpoint_name,
    resolved_device,
    label_map,
    safe_case_id,
    affine,
    pipeline_script,
    num_processes_segmentation_export,
    artifact_prefix,
    progress_callback,
    t_total_start,
):
    """Predict all temporal samples with one common crop and shared resampling."""
    pipeline = _load_4d_pipeline_module(pipeline_script)
    # Dataset7020 plans may reference the project GPU resampler by name. The
    # grouped runner already owns the installer used by its batch workflow;
    # install it before PlansManager resolves the resampling function.
    if str(resolved_device).strip().lower() == "cuda":
        _install_nnunet_4d_resampler(pipeline_script)
    channel_names = _ordered_mapping_values(dataset_json.get("channel_names") or dataset_json.get("modality") or {})
    radius = _nnunet_4d_temporal_radius(channel_names)
    nt = int(mag_nnunet.shape[3])
    if nt < 1:
        raise ValueError("4D nnUNet input has no time frames")

    global_names = [
        "mag_std_xyz", "mag_mean_xyz", "pcmra_std_xyz", "pcmra_mean_xyz",
        "flow_x_mean_xyz", "flow_y_mean_xyz", "flow_z_mean_xyz", "flow_mag_mean_xyz",
        "flow_x_std_xyz", "flow_y_std_xyz", "flow_z_std_xyz", "flow_mag_std_xyz",
    ]
    global_values = _nnunet_channel_volumes(global_names, mag_nnunet, flow_nnunet)
    global_by_name = dict(zip(global_names, global_values))
    speed = np.linalg.norm(flow_nnunet, axis=-1).astype(np.float32, copy=False)
    pcmra = (mag_nnunet * speed).astype(np.float32, copy=False)

    with TemporaryDirectory(prefix="autoflow_nnunet4d_grouped_") as tmp_root:
        tmp_root = Path(tmp_root)
        input_dir = tmp_root / "input"
        output_dir = tmp_root / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        source_paths = {}
        source_volumes = {}
        samples = []
        for frame in range(nt):
            sample_id = f"{safe_case_id}__t{frame:03d}"
            samples.append({"sample_id": sample_id, "frame": frame, "label": None})
            for source_key in pipeline.temporal_source_keys(frame, nt, radius):
                source_paths.setdefault(source_key, None)

        temporal_features = (
            flow_nnunet[..., 0], flow_nnunet[..., 1], flow_nnunet[..., 2],
            mag_nnunet, pcmra,
        )
        for source_key in source_paths:
            kind, feature_index, frame = source_key
            if kind == "global":
                source_volumes[source_key] = global_values[int(feature_index)]
            else:
                source_volumes[source_key] = temporal_features[int(feature_index)][..., int(frame)]

        def _write_source(item):
            index, (source_key, volume) = item
            kind, feature_index, frame = source_key
            frame_token = "global" if kind == "global" else f"t{int(frame):03d}"
            path = input_dir / f"source_{kind}_{int(feature_index):02d}_{frame_token}_{index:03d}.nii.gz"
            _write_nifti_volume(volume, affine, path)
            return source_key, path

        _emit_progress(
            progress_callback,
            stage="autoseg_prepare_inputs",
            message=(
                f"Preparing grouped 4D inputs ({len(source_paths)} unique maps, "
                f"{nt} temporal samples)..."
            ),
            current=2,
            total=5,
            elapsed_sec=time.perf_counter() - t_total_start,
            detail_current=0,
            detail_total=len(source_paths),
            grouped_preprocessing=True,
            temporal_radius=radius,
        )
        items = list(zip(source_paths.keys(), source_volumes.values()))
        worker_count = min(4, max(1, len(items)))
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="autoflow-nnunet4d-source") as executor:
            written = list(executor.map(_write_source, enumerate(items)))
        for source_key, path in written:
            source_paths[source_key] = path

        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        torch_device = torch.device(str(resolved_device))
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False,
            perform_everything_on_device=torch_device.type == "cuda",
            device=torch_device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        predictor.initialize_from_trained_model_folder(
            str(model_path), use_folds=tuple(selected_folds), checkpoint_name=checkpoint_name
        )
        if torch_device.type != "cuda":
            # Dataset7020's persisted plan points at a project CUDA resampler.
            # Keep CPU inference usable by changing only this in-memory plan;
            # the source model metadata remains untouched.  The grouped path
            # invokes this function directly, so it cannot rely on nnUNet's
            # subprocess fallback to make the same adjustment.
            configuration = getattr(predictor.configuration_manager, "configuration", None)
            if isinstance(configuration, dict):
                configuration["resampling_fn_data"] = "resample_data_or_seg_to_shape"
                configuration["resampling_fn_data_kwargs"] = {
                    "is_seg": False,
                    "order": 3,
                    "order_z": 0,
                    "force_separate_z": None,
                }
                if "resampling_fn_probabilities" in configuration:
                    configuration["resampling_fn_probabilities"] = "resample_data_or_seg_to_shape"
                    configuration["resampling_fn_probabilities_kwargs"] = {
                        "is_seg": False,
                        "order": 3,
                        "order_z": 0,
                        "force_separate_z": None,
                    }
        _emit_progress(
            progress_callback,
            stage="autoseg_run_inference",
            message=f"Running grouped 4D nnUNet inference on {resolved_device}...",
            current=3,
            total=5,
            elapsed_sec=time.perf_counter() - t_total_start,
            grouped_preprocessing=True,
            temporal_radius=radius,
            unique_source_maps=len(source_paths),
        )
        raw_iterator = pipeline.iter_grouped_temporal_preprocessed_samples_from_source_paths(
            samples,
            source_paths,
            predictor.plans_manager,
            predictor.configuration_manager,
            predictor.dataset_json,
            radius,
        )
        import torch

        def _predictor_iterator():
            for sample, data, segmentation, properties in raw_iterator:
                if segmentation is not None:
                    raise RuntimeError(f"{sample['sample_id']}: prediction input unexpectedly has a segmentation")
                yield {
                    "data": torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32)),
                    "data_properties": properties,
                    "ofile": str(output_dir / str(sample["sample_id"])),
                }

        predictor.predict_from_data_iterator(
            _predictor_iterator(),
            save_probabilities=False,
            num_processes_segmentation_export=max(1, int(num_processes_segmentation_export or 1)),
        )

        _emit_progress(
            progress_callback,
            stage="autoseg_read_prediction",
            message="Reading grouped 4D nnUNet predictions...",
            current=4,
            total=5,
            elapsed_sec=time.perf_counter() - t_total_start,
            grouped_preprocessing=True,
        )
        predictions = []
        for frame in range(nt):
            prediction_path = output_dir / f"{safe_case_id}__t{frame:03d}.nii.gz"
            if not prediction_path.is_file():
                raise FileNotFoundError(f"grouped 4D nnUNet prediction not found: {prediction_path}")
            predictions.append(_read_nifti_segmentation(prediction_path))
        seg_nnunet = _apply_label_map(np.stack(predictions, axis=3), label_map)
        artifact_prediction = None
        if artifact_prefix:
            artifact_prediction = Path(f"{artifact_prefix}.nii.gz")
            artifact_prediction.parent.mkdir(parents=True, exist_ok=True)
            _write_nifti_segmentation(seg_nnunet, affine, artifact_prediction)
        seg_4d = _restore_autoflow_segmentation(seg_nnunet)
        elapsed_total = time.perf_counter() - t_total_start
        _emit_progress(
            progress_callback,
            stage="autoseg_finalize",
            message="Finalizing grouped 4D auto segmentation volume...",
            current=5,
            total=5,
            elapsed_sec=elapsed_total,
            grouped_preprocessing=True,
        )
        provenance = {
            "source": "auto",
            "backend": "nnUNet4D",
            "model_folder": str(model_path),
            "pipeline_script": str(pipeline_script or ""),
            "python_executable": str(_python_from_4d_pipeline_script(pipeline_script) or sys.executable),
            "checkpoint": str(checkpoint_name),
            "device": str(resolved_device),
            "folds": list(selected_folds),
            "fold_mode": "grouped",
            "channel_names": list(channel_names),
            "time_count": nt,
            "temporal_radius": radius,
            "grouped_preprocessing": True,
            "unique_source_maps": len(source_paths),
            "label_map": {str(k): int(v) for k, v in label_map.items()},
            "case_id": str(safe_case_id),
            "created_at": segmentation_timestamp(),
            "command": [],
            "preprocessing_device": "model",
            "feature_files": [],
            "segmentation_nifti": "" if artifact_prediction is None else str(artifact_prediction),
            "prediction_file": "" if artifact_prediction is None else str(artifact_prediction),
            "elapsed_sec": float(elapsed_total),
        }
        return np.asarray(seg_4d, dtype=np.int16), provenance


def resolve_nnunet_4d_model_folder(model_folder=""):
    """Resolve a 4D model folder or a Dataset7020 orchestration script path."""
    candidate = str(model_folder or "").strip()
    if candidate.lower().endswith((".sh", ".bash")):
        return str(_model_folder_from_4d_pipeline_script(candidate))
    if candidate:
        return str(_resolve_bundled_relative_path(candidate))
    model = _model_folder_from_4d_pipeline_script(_NNUNET_4D_PIPELINE_DEFAULT)
    if model.is_dir():
        return str(model)
    raise FileNotFoundError(
        "4D nnUNet model folder is missing: "
        f"{model}. Set an explicit 4D model folder or pipeline script via auto_model."
    )


def _resolve_nnunet_checkpoint(model_path, checkpoint_name, folds):
    """Use a requested checkpoint, falling back to ``checkpoint_best.pth``."""
    requested = str(checkpoint_name or "checkpoint_final.pth")
    candidates = [requested]
    if requested == "checkpoint_final.pth":
        candidates.append("checkpoint_best.pth")
    for candidate in candidates:
        if all((Path(model_path) / f"fold_{fold}" / candidate).is_file() for fold in folds):
            return candidate
    return requested


def _resolve_nnunet_folds(model_path, folds=None):
    """Resolve ``single``, ``all`` or an explicit fold list deterministically."""
    model_path = Path(model_path)
    detected = _detect_nnunet_folds(model_path)
    has_fold_all = (model_path / "fold_all").is_dir()
    if folds is None or str(folds).strip().lower() in {"", "auto"}:
        return detected
    if isinstance(folds, str):
        token = folds.strip().lower()
        if token in {"all", "ensemble", "5fold", "fivefold"}:
            return detected
        if token in {"single", "one"}:
            # ``fold_all`` is the full-data single-model checkpoint.  Keep it
            # as the explicit single branch even when five numeric folds are
            # also present for a future ensemble run.
            return ["all"] if has_fold_all else [detected[0]]
        values = [part.strip().lower().removeprefix("fold_") for part in folds.split(",") if part.strip()]
    else:
        values = [str(value).strip().lower().removeprefix("fold_") for value in folds if str(value).strip()]
    if not values:
        return detected
    valid = set(detected)
    unknown = [value for value in values if value not in valid]
    if unknown:
        raise FileNotFoundError(f"requested nnUNet folds are unavailable: {unknown}; found {detected}")
    return list(dict.fromkeys(values))


def _nnunet_predict_command(python_executable=None, bootstrap_resampler_path=None):
    if getattr(sys, "frozen", False):
        return [sys.executable, "--autoflow-internal-nnunet-predict"]
    code = (
        "from nnunetv2.inference.predict_from_raw_data import "
        "predict_entry_point_modelfolder as main; main()"
    )
    if bootstrap_resampler_path:
        source = repr(str(Path(bootstrap_resampler_path).expanduser().resolve()))
        code = (
            "from pathlib import Path; import shutil, importlib, nnunetv2; "
            f"_src=Path({source}); _dst=Path(nnunetv2.__file__).resolve().parent/'preprocessing'/'resampling'/'gpu_resampling.py'; "
            "_dst.parent.mkdir(parents=True, exist_ok=True); "
            "shutil.copy2(_src, _dst) if _src.is_file() else None; importlib.invalidate_caches(); "
            "from nnunetv2.inference.predict_from_raw_data import "
            "predict_entry_point_modelfolder as main; main()"
        )
    if python_executable and Path(str(python_executable)).is_file():
        return [
            str(python_executable),
            "-c",
            code,
        ]
    executable = shutil.which("nnUNetv2_predict_from_modelfolder")
    if executable:
        if bootstrap_resampler_path:
            # The console entry point cannot execute the bootstrap prelude;
            # use the active interpreter so the child can install the helper.
            return [sys.executable, "-c", code]
        return [executable]
    return [
        sys.executable,
        "-c",
        code,
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
    folds=None,
    device="cpu",
    auto_label_map="",
    case_id="autoflow_case",
    step_size=0.5,
    disable_tta=True,
    num_processes_preprocessing=0,
    num_processes_segmentation_export=1,
    artifact_prefix="",
    runner=None,
    progress_callback=None,
):
    if str(backend or "").strip().lower() != "nnunet":
        if str(backend or "").strip().lower() in _NNUNET_4D_BACKENDS:
            return generate_nnunet_4d_auto_segmentation(
                mag,
                flow,
                resolution,
                origin,
                model_folder,
                checkpoint_name=checkpoint_name,
                folds=folds,
                device=device,
                auto_label_map=auto_label_map,
                case_id=case_id,
                num_processes_preprocessing=num_processes_preprocessing,
                num_processes_segmentation_export=num_processes_segmentation_export,
                artifact_prefix=artifact_prefix,
                runner=runner,
                progress_callback=progress_callback,
            )
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
    effective_num_processes_preprocessing = int(num_processes_preprocessing or (1 if resolved_device == "cuda" else 3))
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
    folds = _resolve_nnunet_folds(model_path, folds)
    checkpoint_name = _resolve_nnunet_checkpoint(model_path, checkpoint_name, folds)
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
        def _write_channel(item):
            idx, volume = item
            input_path = input_dir / f"{case_id}_{idx:04d}{file_ending}"
            _write_nifti_volume(volume, affine, input_path)
            if idx < len(artifact_feature_paths):
                _link_or_copy_file(input_path, artifact_feature_paths[idx])

        # NIfTI compression is independent per channel and releases the GIL;
        # overlap the twelve feature writes while preserving deterministic data.
        worker_count = min(4, max(1, len(channel_volumes)))
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="autoflow-nnunet-nifti") as executor:
            list(executor.map(_write_channel, enumerate(channel_volumes)))
        for idx, channel_name in enumerate(channel_names):
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
            effective_num_processes_preprocessing,
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
                effective_num_processes_preprocessing,
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


def generate_nnunet_4d_auto_segmentation(
    mag,
    flow,
    resolution,
    origin,
    model_folder="",
    *,
    checkpoint_name="checkpoint_final.pth",
    folds="single",
    device="auto",
    auto_label_map="",
    case_id="autoflow_case",
    num_processes_preprocessing=0,
    num_processes_segmentation_export=1,
    artifact_prefix="",
    runner=None,
    progress_callback=None,
):
    """Run the Dataset7020 temporal model for every frame in one invocation.

    The Dataset7020 checkpoint is still a regular nnUNet model; its temporal
    context is encoded in the channel names (``tm2_*`` through ``tp2_*``).
    AutoFlow therefore writes one sample per frame, invokes nnUNet once, then
    stacks the frame predictions back into an ``XYZT`` segmentation.  ``folds``
    accepts ``single`` (``fold_all`` or the first available fold), ``all`` for
    an ensemble, or a comma-separated explicit list.
    """
    t_total_start = time.perf_counter()
    total_stages = 5
    _emit_progress(
        progress_callback,
        stage="autoseg_start",
        message="Resolving 4D nnUNet model and input metadata...",
        current=0,
        total=total_stages,
        elapsed_sec=0.0,
    )
    pipeline_script = ""
    if str(model_folder or "").lower().endswith((".sh", ".bash")):
        pipeline_script = str(Path(model_folder).expanduser().resolve())
    model_path = Path(resolve_nnunet_4d_model_folder(model_folder))
    resolved_device = resolve_auto_segmentation_device(device)
    mag, flow = _ensure_nnunet_mag_flow(mag, flow)
    time_count = int(flow.shape[3])
    mag_nnunet, flow_nnunet, resolution_nnunet = _prepare_nnunet_inputs(mag, flow, resolution)
    model_path, dataset_json = _load_nnunet_model_metadata(model_path)
    channel_names = _ordered_mapping_values(dataset_json.get("channel_names") or dataset_json.get("modality") or {})
    if not channel_names:
        raise ValueError(f"4D model folder does not define any channel names: {model_path}")
    if not any(_parse_nnunet_temporal_channel(name) for name in channel_names):
        raise ValueError(
            f"4D nnUNet model has no temporal channels: {model_path / 'dataset.json'}"
        )
    file_ending = str(dataset_json.get("file_ending", ".nii.gz"))
    if not file_ending.startswith("."):
        file_ending = f".{file_ending}"
    label_map = _parse_nnunet_label_map(auto_label_map, dataset_json.get("labels", {}))
    selected_folds = _resolve_nnunet_folds(model_path, folds)
    checkpoint_name = _resolve_nnunet_checkpoint(model_path, checkpoint_name, selected_folds)
    affine = _nnunet_spatial_affine(resolution_nnunet, mag_nnunet.shape[:3])
    _emit_progress(
        progress_callback,
        stage="autoseg_model_ready",
        message=f"Resolved 4D nnUNet model: {model_path.name} | folds={','.join(selected_folds)} | device={resolved_device}",
        current=1,
        total=total_stages,
        elapsed_sec=time.perf_counter() - t_total_start,
        backend="nnUNet4D",
        device=str(resolved_device),
        checkpoint=str(checkpoint_name),
        folds=list(selected_folds),
        model_folder=str(model_path),
    )

    safe_case_id = _sanitize_nnunet_artifact_token(case_id, default="autoflow_case")
    grouped_script = pipeline_script
    grouped_fallback_reason = ""
    if not grouped_script and _NNUNET_4D_PIPELINE_DEFAULT.is_file():
        grouped_script = str(_NNUNET_4D_PIPELINE_DEFAULT)
        # Use the default script for subprocess bootstrap as well, not only for
        # grouped preprocessing. This keeps custom resampling available when a
        # caller leaves the model setting empty.
        pipeline_script = grouped_script
    if _nnunet_4d_grouped_requested(runner) and grouped_script:
        try:
            return _generate_nnunet_4d_grouped(
                mag_nnunet,
                flow_nnunet,
                resolution_nnunet,
                model_path,
                dataset_json,
                selected_folds,
                checkpoint_name,
                resolved_device,
                label_map,
                safe_case_id,
                affine,
                grouped_script,
                num_processes_segmentation_export,
                artifact_prefix,
                progress_callback,
                t_total_start,
            )
        except Exception as exc:
            grouped_fallback_reason = f"{type(exc).__name__}: {exc}"
            token = str(os.environ.get("AUTOFLOW_NNUNET4D_GROUPED", "auto") or "auto").strip().lower()
            if token in {"1", "true", "yes", "on", "required"}:
                raise
            _emit_progress(
                progress_callback,
                stage="autoseg_grouped_fallback",
                message="Grouped 4D preprocessing failed; retrying with standard nnUNet input files...",
                current=2,
                total=total_stages,
                elapsed_sec=time.perf_counter() - t_total_start,
                grouped_preprocessing=False,
                fallback_reason=grouped_fallback_reason,
            )

    with TemporaryDirectory(prefix="autoflow_nnunet4d_") as tmp_root:
        tmp_root = Path(tmp_root)
        input_dir = tmp_root / "input"
        output_dir = tmp_root / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        global_names = [
            "mag_std_xyz", "mag_mean_xyz", "pcmra_std_xyz", "pcmra_mean_xyz",
            "flow_x_mean_xyz", "flow_y_mean_xyz", "flow_z_mean_xyz", "flow_mag_mean_xyz",
            "flow_x_std_xyz", "flow_y_std_xyz", "flow_z_std_xyz", "flow_mag_std_xyz",
        ]
        global_by_name = dict(zip(global_names, _nnunet_channel_volumes(global_names, mag_nnunet, flow_nnunet)))
        temporal_speed = np.linalg.norm(flow_nnunet, axis=-1).astype(np.float32, copy=False)
        temporal_cache = {
            "mag": mag_nnunet,
            "pcmra": (mag_nnunet * temporal_speed).astype(np.float32, copy=False),
            "flow_x": flow_nnunet[..., 0],
            "flow_y": flow_nnunet[..., 1],
            "flow_z": flow_nnunet[..., 2],
        }
        _emit_progress(
            progress_callback,
            stage="autoseg_prepare_inputs",
            message=f"Preparing {time_count} temporal nnUNet samples ({len(channel_names)} channels each)...",
            current=2,
            total=total_stages,
            elapsed_sec=time.perf_counter() - t_total_start,
            detail_current=0,
            detail_total=time_count,
        )

        def _write_frame(frame_index):
            volumes = _nnunet_4d_channel_volumes(
                channel_names,
                mag_nnunet,
                flow_nnunet,
                frame_index,
                global_by_name=global_by_name,
                temporal_cache=temporal_cache,
            )
            frame_id = f"{safe_case_id}_t{int(frame_index):03d}"
            for channel_index, volume in enumerate(volumes):
                path = input_dir / f"{frame_id}_{channel_index:04d}{file_ending}"
                _write_nifti_volume(volume, affine, path)
            return frame_id

        # NIfTI encoding is independent per phase and releases the GIL.  A
        # small pool avoids serial gzip overhead without competing with nnUNet.
        worker_count = min(4, max(1, time_count))
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="autoflow-nnunet4d-nifti") as executor:
            list(executor.map(_write_frame, range(time_count)))
        for frame_index in range(time_count):
            _emit_progress(
                progress_callback,
                stage="autoseg_prepare_inputs",
                message=f"Writing 4D frame {frame_index + 1}/{time_count}",
                current=2,
                total=total_stages,
                elapsed_sec=time.perf_counter() - t_total_start,
                detail_current=frame_index + 1,
                detail_total=time_count,
            )

        command = _nnunet_inference_command(
            input_dir,
            output_dir,
            model_path,
            selected_folds,
            checkpoint_name,
            resolved_device,
            int(num_processes_preprocessing or (1 if resolved_device == "cuda" else 3)),
            num_processes_segmentation_export,
            0.5,
            True,
            python_executable=_python_from_4d_pipeline_script(pipeline_script),
            bootstrap_resampler_path=(
                Path(pipeline_script).expanduser().resolve().parent.parent.parent
                / "nnunet" / "gpu_resampling.py"
                if pipeline_script
                else None
            ),
        )
        _emit_progress(
            progress_callback,
            stage="autoseg_run_inference",
            message=f"Running 4D nnUNet inference on {resolved_device}...",
            current=3,
            total=total_stages,
            elapsed_sec=time.perf_counter() - t_total_start,
            command=[str(x) for x in command],
            preprocessing_device="model",
        )
        subprocess_env = _nnunet_4d_subprocess_env(pipeline_script, model_path)
        result = _run_subprocess(command, env=subprocess_env, runner=runner)
        if getattr(result, "returncode", 0) != 0:
            raise RuntimeError(
                "4D nnUNet inference failed\n"
                f"command: {' '.join(map(str, command))}\n"
                f"stdout:\n{getattr(result, 'stdout', '') or ''}\n"
                f"stderr:\n{getattr(result, 'stderr', '') or ''}"
            )

        _emit_progress(
            progress_callback,
            stage="autoseg_read_prediction",
            message="Reading 4D nnUNet predictions...",
            current=4,
            total=total_stages,
            elapsed_sec=time.perf_counter() - t_total_start,
        )
        predictions = []
        for frame_index in range(time_count):
            prediction_path = output_dir / f"{safe_case_id}_t{frame_index:03d}{file_ending}"
            if not prediction_path.is_file() and file_ending == ".nii.gz":
                prediction_path = output_dir / f"{safe_case_id}_t{frame_index:03d}.nii.gz"
            if not prediction_path.is_file():
                raise FileNotFoundError(f"4D nnUNet prediction not found: {prediction_path}")
            predictions.append(_read_nifti_segmentation(prediction_path))
        shape = tuple(np.asarray(predictions[0]).shape)
        if any(tuple(np.asarray(item).shape) != shape for item in predictions):
            raise ValueError("4D nnUNet predictions have inconsistent spatial shapes")
        seg_nnunet = np.stack(predictions, axis=3)
        seg_nnunet = _apply_label_map(seg_nnunet, label_map)
        if artifact_prefix:
            artifact_prediction = Path(f"{artifact_prefix}.nii.gz")
            artifact_prediction.parent.mkdir(parents=True, exist_ok=True)
            _write_nifti_segmentation(seg_nnunet, affine, artifact_prediction)
        else:
            artifact_prediction = None
        seg_4d = _restore_autoflow_segmentation(seg_nnunet)
        elapsed_total = time.perf_counter() - t_total_start
        _emit_progress(
            progress_callback,
            stage="autoseg_finalize",
            message="Finalizing 4D auto segmentation volume...",
            current=5,
            total=total_stages,
            elapsed_sec=elapsed_total,
        )
        provenance = {
            "source": "auto",
            "backend": "nnUNet4D",
            "model_folder": str(model_path),
            "pipeline_script": pipeline_script,
            "python_executable": str(_python_from_4d_pipeline_script(pipeline_script) or sys.executable),
            "checkpoint": str(checkpoint_name),
            "device": str(resolved_device),
            "folds": list(selected_folds),
            "fold_mode": str(folds),
            "channel_names": list(channel_names),
            "time_count": int(time_count),
            "label_map": {str(k): int(v) for k, v in label_map.items()},
            "case_id": str(case_id),
            "created_at": segmentation_timestamp(),
            "command": [str(x) for x in command],
            "preprocessing_device": "model",
            "grouped_preprocessing": False,
            "grouped_fallback_reason": grouped_fallback_reason,
            "feature_files": [],
            "segmentation_nifti": "" if artifact_prediction is None else str(artifact_prediction),
            "prediction_file": "" if artifact_prediction is None else str(artifact_prediction),
            "elapsed_sec": float(elapsed_total),
        }
        return np.asarray(seg_4d, dtype=np.int16), provenance
