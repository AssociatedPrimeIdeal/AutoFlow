import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import nibabel as nib
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


def _nnunet_spatial_affine(resolution, origin):
    res = np.asarray(resolution, dtype=np.float32).reshape(-1)
    if res.size == 1:
        res = np.repeat(res, 3)
    origin = np.asarray(origin, dtype=np.float32).reshape(-1)
    if origin.size == 1:
        origin = np.repeat(origin, 3)
    affine = np.eye(4, dtype=np.float32)
    affine[0, 0] = float(res[0])
    affine[1, 1] = float(res[1])
    affine[2, 2] = float(res[2])
    affine[:3, 3] = np.asarray(origin[:3], dtype=np.float32)
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


def _nnunet_channel_volume(channel_name, mag, flow, channel_index=0):
    token = _nnunet_normalize_channel_name(channel_name)
    speed = np.linalg.norm(flow, axis=-1)
    mag_mean = np.mean(mag, axis=3)
    mag_std = np.std(mag, axis=3)
    pcmra = mag * speed
    pcmra_mean = np.mean(pcmra, axis=3)
    pcmra_std = np.std(pcmra, axis=3)
    flow_x = flow[..., 0]
    flow_y = flow[..., 1]
    flow_z = flow[..., 2]
    flow_x_mean = np.mean(flow_x, axis=3)
    flow_y_mean = np.mean(flow_y, axis=3)
    flow_z_mean = np.mean(flow_z, axis=3)
    flow_mag_mean = np.mean(speed, axis=3)
    flow_x_std = np.std(flow_x, axis=3)
    flow_y_std = np.std(flow_y, axis=3)
    flow_z_std = np.std(flow_z, axis=3)
    flow_mag_std = np.std(speed, axis=3)
    channels = {
        "mag": mag_mean,
        "mag_mean_xyz": mag_mean,
        "mag_std_xyz": mag_std,
        "pcmra": pcmra_mean,
        "pcmra_mean_xyz": pcmra_mean,
        "pcmra_std_xyz": pcmra_std,
        "flow_x_mean_xyz": flow_x_mean,
        "flow_y_mean_xyz": flow_y_mean,
        "flow_z_mean_xyz": flow_z_mean,
        "flow_mag_mean_xyz": flow_mag_mean,
        "flow_x_std_xyz": flow_x_std,
        "flow_y_std_xyz": flow_y_std,
        "flow_z_std_xyz": flow_z_std,
        "flow_mag_std_xyz": flow_mag_std,
    }
    if token in channels:
        return np.asarray(channels[token], dtype=np.float32)
    default_order = [
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
    ]
    if 0 <= int(channel_index) < len(default_order):
        return np.asarray(channels[default_order[int(channel_index)]], dtype=np.float32)
    raise ValueError(
        f"unsupported nnUNet channel '{channel_name}'. Supported channels: {sorted(channels.keys())}"
    )


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
    return (
        Path(__file__).resolve().parents[1]
        / "segmodel"
        / "nnUNetTrainer_500epochs__nnUNetPlans__3d_fullres_iso1mm"
    )


def resolve_nnunet_model_folder(model_folder=""):
    candidate = str(model_folder or "").strip()
    if candidate:
        return str(Path(candidate).expanduser())
    default_path = default_nnunet_model_folder()
    if default_path.is_dir():
        return str(default_path)
    raise FileNotFoundError(
        "nnUNet model folder not configured and bundled default model is missing: "
        f"{default_path}"
    )


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
    num_processes_segmentation_export=3,
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
    model_path, dataset_json = _load_nnunet_model_metadata(model_folder)
    channel_names = _ordered_mapping_values(dataset_json.get("channel_names") or dataset_json.get("modality") or {})
    if not channel_names:
        raise ValueError(f"model folder does not define any channel names: {model_path}")
    file_ending = str(dataset_json.get("file_ending", ".nii.gz"))
    if not file_ending.startswith("."):
        file_ending = f".{file_ending}"
    label_map = _parse_nnunet_label_map(auto_label_map, dataset_json.get("labels", {}))
    folds = _detect_nnunet_folds(model_path)
    affine = _nnunet_spatial_affine(resolution, origin)
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
        for idx, channel_name in enumerate(channel_names):
            volume = _nnunet_channel_volume(channel_name, mag, flow, channel_index=idx)
            _write_nifti_volume(volume, affine, input_dir / f"{case_id}_{idx:04d}{file_ending}")
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

        command = [
            shutil.which("nnUNetv2_predict_from_modelfolder") or "nnUNetv2_predict_from_modelfolder",
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
        _emit_progress(
            progress_callback,
            stage="autoseg_run_inference",
            message=f"Running nnUNet inference on {resolved_device}...",
            current=3,
            total=total_stages,
            elapsed_sec=time.perf_counter() - t_total_start,
            command=[str(x) for x in command],
        )
        result = _run_subprocess(command, runner=runner)
        if getattr(result, "returncode", 0) != 0:
            stdout = getattr(result, "stdout", "") or ""
            stderr = getattr(result, "stderr", "") or ""
            raise RuntimeError(
                "nnUNet inference failed\n"
                f"command: {' '.join(map(str, command))}\n"
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
        seg_3d = _read_nifti_segmentation(prediction_path)
        seg_3d = _apply_label_map(seg_3d, label_map)
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
            "label_map": {str(k): int(v) for k, v in label_map.items()},
            "case_id": str(case_id),
            "created_at": segmentation_timestamp(),
            "command": [str(x) for x in command],
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "prediction_file": str(prediction_path),
            "elapsed_sec": float(elapsed_total),
        }
        return seg_4d, provenance


