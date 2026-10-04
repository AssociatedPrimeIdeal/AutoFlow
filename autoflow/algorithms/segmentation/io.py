"""Segmentation files, reusable NIfTI features and reviewed H5 masks."""

import json
import os
import re
import shutil
from pathlib import Path
import h5py
import nibabel as nib
import numpy as np
from ..data.h5_metadata import _canonical_h5_key, _find_h5_dataset_from_scopes, _h5_member_name_map, _resolve_h5_data_group
from ..data.orientation import _reorient_spatial_only

from ._common import normalize_segmentation_volume, segmentation_timestamp
from .channels import _nnunet_normalize_channel_name
from .models import _AUTOFLOW_INTERNAL_SPATIAL_ORDER


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
    arr = np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"expected 3D nnUNet segmentation, got shape={arr.shape}")
    return np.rint(arr).astype(np.int16)
