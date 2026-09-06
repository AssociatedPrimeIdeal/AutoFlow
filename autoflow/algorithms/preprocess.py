import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_opening,
    gaussian_filter,
    find_objects,
    label,
    distance_transform_edt,
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


def _label_components(mask, connectivity=1):
    m = np.asarray(mask, dtype=bool)
    if not np.any(m):
        return np.zeros(m.shape, dtype=np.int32), 0
    struct = np.ones((3, 3, 3), dtype=bool) if connectivity == 2 else None
    return label(m, structure=struct)


def _connected_components(mask, connectivity=1):
    lab, n = _label_components(mask, connectivity=connectivity)
    return [(int(i), lab == int(i)) for i in range(1, int(n) + 1)]


def _component_counts(mask, connectivity=1):
    lab, n = _label_components(mask, connectivity=connectivity)
    counts = np.bincount(lab.ravel(), minlength=int(n) + 1)
    if counts.size:
        counts[0] = 0
    return lab, counts


def _component_volumes_mm3(mask_3d, resolution, connectivity=1):
    resolution = np.asarray(resolution, dtype=float).reshape(3)
    voxel_volume = float(np.prod(resolution))
    items = []
    for cc_id, cc in _connected_components(mask_3d, connectivity=connectivity):
        volume_mm3 = float(int(np.sum(cc)) * voxel_volume)
        items.append((int(cc_id), cc, volume_mm3))
    return items


def component_volume_threshold_mm3(mask_3d, resolution, mode="absolute", min_volume_mm3=50.0, rel_min_ratio=0.01, connectivity=1):
    _lab, counts = _component_counts(mask_3d, connectivity=connectivity)
    if counts.size <= 1 or not np.any(counts[1:] > 0):
        return 0.0
    voxel_volume = float(np.prod(np.asarray(resolution, dtype=float).reshape(3)))
    largest = float(np.max(counts[1:]) * voxel_volume)
    abs_min = max(0.0, float(min_volume_mm3))
    mode_name = str(mode or "absolute").strip().lower()
    if mode_name == "largest":
        return largest
    if mode_name == "relative":
        return float(max(0.0, float(rel_min_ratio)) * largest)
    if mode_name == "hybrid":
        return float(max(abs_min, max(0.0, float(rel_min_ratio)) * largest))
    return float(abs_min)


def filter_connected_components(mask_3d, resolution, mode="absolute", min_volume_mm3=50.0, rel_min_ratio=0.01, connectivity=1):
    m = np.asarray(mask_3d, dtype=bool)
    if m.ndim != 3:
        raise ValueError(f"connected-component filtering expects a 3D mask, got {m.shape}")
    lab, counts = _component_counts(m, connectivity=connectivity)
    if counts.size <= 1 or not np.any(counts[1:] > 0):
        return np.zeros_like(m, dtype=bool)
    mode_name = str(mode or "absolute").strip().lower()
    if mode_name == "largest":
        return lab == int(np.argmax(counts[1:]) + 1)
    voxel_volume = float(np.prod(np.asarray(resolution, dtype=float).reshape(3)))
    largest = float(np.max(counts[1:]) * voxel_volume)
    abs_min = max(0.0, float(min_volume_mm3))
    if mode_name == "relative":
        threshold_mm3 = max(0.0, float(rel_min_ratio)) * largest
    elif mode_name == "hybrid":
        threshold_mm3 = max(abs_min, max(0.0, float(rel_min_ratio)) * largest)
    else:
        threshold_mm3 = abs_min
    keep = counts.astype(float) * voxel_volume + 1e-12 >= float(threshold_mm3)
    keep[0] = False
    return keep[lab]


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
    lab, counts = _component_counts(mask_3d)
    remove = counts.astype(float) * voxel_volume < float(min_cc_volume_mm3)
    if remove.size:
        remove[0] = False
    remove_mask = remove[lab]
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


def filter_4d_labeled_components(segmask_labels_4d, resolution, mode="absolute", min_volume_mm3=50.0, connectivity=1):
    """Clean connected components independently for every label and time frame.

    ``absolute`` removes components below ``min_volume_mm3`` and ``largest`` keeps
    only the largest component for each non-zero label in each frame.
    """
    labels = np.asarray(segmask_labels_4d, dtype=np.int16)
    if labels.ndim != 4:
        raise ValueError(f"4D labeled mask must be XYZT, got {labels.shape}")
    out = labels.copy()
    mode_name = str(mode or "absolute").strip().lower()
    if mode_name not in {"absolute", "largest"}:
        raise ValueError(f"unsupported 4D component cleanup mode: {mode!r}")
    for t in range(labels.shape[3]):
        frame = labels[..., t]
        for label_value in sorted(int(x) for x in np.unique(frame) if int(x) != 0):
            mask = frame == int(label_value)
            cleaned = filter_connected_components(
                mask, resolution, mode=mode_name,
                min_volume_mm3=float(min_volume_mm3), connectivity=connectivity,
            )
            out[..., t][mask & ~np.asarray(cleaned, dtype=bool)] = 0
    return out


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
    component_labels, component_count = _label_components(filtered_mask)
    for component_id, bbox in enumerate(find_objects(component_labels), start=1):
        if component_id > int(component_count) or bbox is None:
            continue
        local = component_labels[bbox] == int(component_id)
        proc = _preprocess_single_component(local, params)
        if np.any(proc):
            out[bbox] |= proc
    return out.astype(bool)


def separate_longitudinal_label_contacts(mask_3d, label_mask_3d, label_values, resolution=None,
                                         min_contact_voxels=8, elongation_threshold=2.5,
                                         force_pairs=None):
    """Conservatively cut elongated side-by-side contacts between labels.

    This keeps the group mask as the skeleton input.  It only uses the source
    labels to identify a contact interface whose physical extent is strongly
    elongated; compact interfaces (the usual end-to-end label transitions)
    are left untouched.
    """
    mask = np.asarray(mask_3d, dtype=bool)
    labels = np.asarray(label_mask_3d)
    if mask.shape != labels.shape or mask.ndim != 3:
        raise ValueError("mask_3d and label_mask_3d must be matching 3D arrays")
    values = sorted({int(value) for value in (label_values or []) if int(value) != 0})
    if len(values) < 2 or not np.any(mask):
        return mask.copy()
    spacing = np.ones(3, dtype=float) if resolution is None else np.asarray(resolution, dtype=float).reshape(3)
    out = mask.copy()
    structure = np.ones((3, 3, 3), dtype=bool)
    forced = {tuple(sorted((int(pair[0]), int(pair[1])))) for pair in (force_pairs or [])}
    distance_fields = {}

    for value_index, first in enumerate(values[:-1]):
        for second in values[value_index + 1:]:
            first_region = (labels == first) & mask
            second_region = (labels == second) & mask
            # Use the same 26-neighbour notion that the graph builder can
            # connect, so diagonal voxel contacts are not left behind.
            interface = (
                (first_region & binary_dilation(second_region, structure=structure)) |
                (second_region & binary_dilation(first_region, structure=structure))
            )
            if not np.any(interface):
                continue

            interface_labels, count = label(interface, structure=structure)
            for contact_id in range(1, int(count) + 1):
                contact = interface_labels == contact_id
                coords = np.argwhere(contact)
                pair_forced = (int(first), int(second)) in forced
                if len(coords) < int(min_contact_voxels) and not pair_forced:
                    continue
                if not pair_forced:
                    extents = (coords.max(axis=0) - coords.min(axis=0) + 1) * spacing
                    ordered = np.sort(extents)
                    if ordered[-1] < 3.0 * float(np.min(spacing)):
                        continue
                    elongation = float(ordered[-1] / max(ordered[-2], 1e-6))
                    if elongation < float(elongation_threshold):
                        continue

                # Do not cut through a one-voxel-thick vessel.  The distance
                # check is evaluated per side of the contact and remains
                # independent of the symbolic label names.
                if first not in distance_fields:
                    distance_fields[first] = distance_transform_edt(labels == first, sampling=spacing)
                if second not in distance_fields:
                    distance_fields[second] = distance_transform_edt(labels == second, sampling=spacing)
                first_radius = distance_fields[first]
                second_radius = distance_fields[second]
                radius_threshold = 0.75 * float(np.min(spacing))
                safe = contact & (
                    ((labels == first) & (first_radius >= radius_threshold)) |
                    ((labels == second) & (second_radius >= radius_threshold))
                )
                if np.any(safe):
                    out[safe] = False
    return out


def separate_special_label_contacts(mask_3d, label_mask_3d, special_label_values, resolution=None):
    """Separate contacts between the configured special segmentation labels.

    The group mask remains the single skeleton input.  Only interfaces between
    the supplied special labels are cut; every other label pair is untouched.
    This is intentionally a small, deterministic wrapper around the legacy
    contact-removal primitive, with all special pairs forced so that a close
    side-by-side contact cannot be interpreted as one branch by the skeleton
    graph builder.
    """
    values = sorted({int(value) for value in (special_label_values or []) if int(value) != 0})
    if len(values) < 2:
        return np.asarray(mask_3d, dtype=bool).copy()
    pairs = [(first, second) for index, first in enumerate(values[:-1]) for second in values[index + 1:]]
    return separate_longitudinal_label_contacts(
        mask_3d,
        label_mask_3d,
        values,
        resolution=resolution,
        force_pairs=pairs,
    )


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
