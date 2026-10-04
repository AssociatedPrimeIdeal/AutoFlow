"""Segmentation timestamps, temporal broadcasting and volume normalization."""

from datetime import datetime, timezone
import numpy as np


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
