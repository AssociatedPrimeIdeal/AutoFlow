import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_opening,
    gaussian_filter,
    label,
)

from ..core.models import SkeletonParams


def _normalize_label_selection(labels):
    if labels is None:
        return None
    values = [labels] if np.isscalar(labels) else list(labels)
    arr = np.asarray(values)
    return {int(x) for x in arr.ravel().tolist() if int(x) != 0}


def filter_segmask_labels(segmask_raw, labels_to_keep=None, labels_to_remove=None):
    """Return a label-preserving mask with optional label include/exclude filters."""
    seg = np.asarray(segmask_raw).copy()
    keep = _normalize_label_selection(labels_to_keep)
    remove = _normalize_label_selection(labels_to_remove)
    if keep is not None:
        seg[~np.isin(seg, list(keep))] = 0
    if remove:
        seg[np.isin(seg, list(remove))] = 0
    return seg


def binarize_segmask(segmask_labels):
    return (np.asarray(segmask_labels) > 0).astype(bool)



def merge_segmask_to_3d(segmask_binary_4d):
    seg = np.asarray(segmask_binary_4d, dtype=bool)
    if seg.ndim == 3:
        return seg
    return np.any(seg, axis=3)


def majority_vote_labels_3d(segmask_labels):
    seg = np.asarray(segmask_labels, dtype=np.int16)
    if seg.ndim == 3:
        return seg.copy()
    if seg.ndim != 4:
        raise ValueError(f"segmentation labels must be 3D or 4D, got {seg.shape}")
    labels = [int(x) for x in np.unique(seg)]
    counts = np.stack([(seg == int(label_value)).sum(axis=3) for label_value in labels], axis=-1)
    winners = np.argmax(counts, axis=-1)
    out = np.zeros(seg.shape[:3], dtype=np.int16)
    for idx, label_value in enumerate(labels):
        out[winners == idx] = int(label_value)
    return out


def _connected_components(mask, connectivity=1):
    m = np.asarray(mask, dtype=bool)
    if not np.any(m):
        return []
    struct = np.ones((3, 3, 3), dtype=bool) if connectivity == 2 else None
    lab, n = label(m, structure=struct)
    return [(int(i), lab == int(i)) for i in range(1, int(n) + 1)]


def _component_volumes_mm3(mask_3d, resolution, connectivity=1):
    resolution = np.asarray(resolution, dtype=float).reshape(3)
    voxel_volume = float(np.prod(resolution))
    items = []
    for cc_id, cc in _connected_components(mask_3d, connectivity=connectivity):
        volume_mm3 = float(int(np.sum(cc)) * voxel_volume)
        items.append((int(cc_id), cc, volume_mm3))
    return items


def component_volume_threshold_mm3(mask_3d, resolution, mode="absolute", min_volume_mm3=50.0, rel_min_ratio=0.01, connectivity=1):
    items = _component_volumes_mm3(mask_3d, resolution, connectivity=connectivity)
    if not items:
        return 0.0
    abs_min = max(0.0, float(min_volume_mm3))
    mode_name = str(mode or "absolute").strip().lower()
    if mode_name == "largest":
        return float(max(float(volume_mm3) for _, _, volume_mm3 in items))
    if mode_name == "relative":
        largest = max(float(volume_mm3) for _, _, volume_mm3 in items)
        return float(max(0.0, float(rel_min_ratio)) * largest)
    if mode_name == "hybrid":
        largest = max(float(volume_mm3) for _, _, volume_mm3 in items)
        return float(max(abs_min, max(0.0, float(rel_min_ratio)) * largest))
    return float(abs_min)


def filter_connected_components(mask_3d, resolution, mode="absolute", min_volume_mm3=50.0, rel_min_ratio=0.01, connectivity=1):
    m = np.asarray(mask_3d, dtype=bool)
    if m.ndim != 3:
        raise ValueError(f"connected-component filtering expects a 3D mask, got {m.shape}")
    items = _component_volumes_mm3(m, resolution, connectivity=connectivity)
    if not items:
        return np.zeros_like(m, dtype=bool)
    mode_name = str(mode or "absolute").strip().lower()
    out = np.zeros_like(m, dtype=bool)
    if mode_name == "largest":
        largest_item = max(items, key=lambda item: item[2])
        out |= np.asarray(largest_item[1], dtype=bool)
        return out
    threshold_mm3 = component_volume_threshold_mm3(
        m,
        resolution,
        mode=mode_name,
        min_volume_mm3=min_volume_mm3,
        rel_min_ratio=rel_min_ratio,
        connectivity=connectivity,
    )
    for _, cc, volume_mm3 in items:
        if float(volume_mm3) + 1e-12 >= float(threshold_mm3):
            out |= np.asarray(cc, dtype=bool)
    return out


def remove_small_cc_from_binary_mask(segmask_binary, resolution, min_cc_volume_mm3):
    seg = np.asarray(segmask_binary, dtype=bool).copy()
    if seg.ndim == 3:
        mask_3d = seg
    elif seg.ndim == 4:
        mask_3d = np.any(seg, axis=3)
    else:
        return seg
    resolution = np.asarray(resolution, dtype=float).reshape(3)
    voxel_volume = float(np.prod(resolution))
    remove_mask = np.zeros_like(mask_3d, dtype=bool)
    for _, cc in _connected_components(mask_3d):
        n_voxels = int(np.sum(cc))
        volume_mm3 = n_voxels * voxel_volume
        if volume_mm3 < min_cc_volume_mm3:
            remove_mask |= cc
    if np.any(remove_mask):
        if seg.ndim == 4:
            seg[remove_mask] = False
        else:
            seg[remove_mask] = False
    return seg


def remove_small_cc_from_labeled_mask(segmask_labels_3d, resolution, min_cc_volume_mm3):
    labels = np.asarray(segmask_labels_3d, dtype=np.int16).copy()
    if labels.ndim != 3:
        raise ValueError(f"labeled mask must be 3D, got {labels.shape}")
    if float(min_cc_volume_mm3) <= 0:
        return labels
    for label_value in sorted(int(x) for x in np.unique(labels) if int(x) != 0):
        mask = labels == int(label_value)
        cleaned = remove_small_cc_from_binary_mask(mask, resolution, float(min_cc_volume_mm3))
        labels[mask & ~np.asarray(cleaned, dtype=bool)] = 0
    return labels


def filter_labeled_components(segmask_labels_3d, resolution, mode="absolute", min_volume_mm3=50.0, rel_min_ratio=0.01):
    labels = np.asarray(segmask_labels_3d, dtype=np.int16).copy()
    if labels.ndim != 3:
        raise ValueError(f"labeled mask must be 3D, got {labels.shape}")
    for label_value in sorted(int(x) for x in np.unique(labels) if int(x) != 0):
        mask = labels == int(label_value)
        cleaned = filter_connected_components(
            mask,
            resolution,
            mode=mode,
            min_volume_mm3=min_volume_mm3,
            rel_min_ratio=rel_min_ratio,
        )
        labels[mask & ~np.asarray(cleaned, dtype=bool)] = 0
    return labels


def _component_bbox(mask):
    idx = np.argwhere(np.asarray(mask, dtype=bool))
    if len(idx) == 0:
        return None
    lo = idx.min(axis=0)
    hi = idx.max(axis=0) + 1
    return tuple(slice(int(lo[k]), int(hi[k])) for k in range(3))


def _preprocess_single_component(mask_3d, params):
    m = np.asarray(mask_3d, dtype=bool).copy()
    if not np.any(m):
        return m
    if getattr(params, "dilation_iters", 0) > 0:
        m = binary_dilation(m, iterations=int(params.dilation_iters)).astype(bool)
    if getattr(params, "erosion_iters", 0) > 0:
        m = binary_erosion(m, iterations=int(params.erosion_iters)).astype(bool)
    if params.do_closing:
        m = binary_closing(m).astype(bool)
    if params.do_opening:
        m = binary_opening(m).astype(bool)
    if getattr(params, "closing_iters", 0) > 0:
        m = binary_closing(m, iterations=int(params.closing_iters)).astype(bool)
    if getattr(params, "opening_iters", 0) > 0:
        m = binary_opening(m, iterations=int(params.opening_iters)).astype(bool)
    if params.gaussian_enabled and params.gaussian_sigma > 0:
        m = gaussian_filter(m.astype(float), sigma=params.gaussian_sigma) > 0.5
    return m.astype(bool)


def preprocess_mask_for_skeleton(mask_3d, params=None, resolution=None):
    if params is None:
        params = SkeletonParams()
    m = np.asarray(mask_3d, dtype=bool).copy()
    if not np.any(m):
        return m
    if resolution is None:
        resolution = np.array([1.0, 1.0, 1.0])
    resolution = np.asarray(resolution, dtype=float).reshape(3)
    out = np.zeros_like(m, dtype=bool)
    filtered_mask = m
    if getattr(params, "remove_small_cc", False):
        filtered_mask = filter_connected_components(
            m,
            resolution,
            mode=getattr(params, "cc_filter_mode", "hybrid"),
            min_volume_mm3=getattr(params, "min_cc_volume_mm3", 50.0),
            rel_min_ratio=getattr(params, "cc_rel_min_ratio", 0.01),
        )
    for _, cc in _connected_components(filtered_mask):
        bbox = _component_bbox(cc)
        if bbox is None:
            continue
        local = cc[bbox]
        proc = _preprocess_single_component(local, params)
        if np.any(proc):
            out[bbox] |= proc
    return out.astype(bool)


def largest_connected_component(mask, connectivity=1):
    m = np.asarray(mask).astype(bool)
    if not m.any():
        return m
    struct = np.ones((3, 3, 3), dtype=bool) if connectivity == 2 else None
    lab, n = label(m, structure=struct)
    if n <= 1:
        return m
    cnt = np.bincount(lab.ravel())
    cnt[0] = 0
    return lab == cnt.argmax()
