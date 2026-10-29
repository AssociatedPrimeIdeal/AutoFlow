import numpy as np
from scipy.ndimage import find_objects
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from .preprocess import (
    _component_counts, _component_keep_mask, _label_components, preprocess_mask_for_skeleton,
)


def generate_skeleton_from_mask3d(mask3d, resolution):
    mask3d = np.asarray(mask3d, dtype=bool)
    component_labels, component_count = _label_components(mask3d)
    return _skeleton_from_components(component_labels, component_count, resolution)


def _skeleton_from_components(component_labels, component_count, resolution, keep=None):
    skel = np.zeros(component_labels.shape, dtype=bool)
    point_chunks = []
    for component_id, bbox in enumerate(find_objects(component_labels), start=1):
        if component_id > int(component_count) or bbox is None:
            continue
        if keep is not None and not keep[component_id]:
            continue
        local = component_labels[bbox] == int(component_id)
        if not np.any(local):
            continue
        local_skel = skeletonize(local).astype(bool)
        if np.any(local_skel):
            skel[bbox] |= local_skel
            point_chunks.append(np.argwhere(local_skel) + [sl.start for sl in bbox])
    # Match full-volume argwhere order without scanning the mostly empty grid.
    indices = np.vstack(point_chunks) if point_chunks else np.empty((0, 3), dtype=int)
    if len(indices):
        indices = indices[np.lexsort(indices.T[::-1])]
    pts = indices.astype(float) * np.asarray(resolution, dtype=float).reshape(1, 3)
    return pts, skel


def _skeletonize_unmodified_pass(mask, params, resolution):
    """Reuse the component labels for selection and Lee thinning."""
    if not getattr(params, "remove_small_cc", False):
        return generate_skeleton_from_mask3d(mask, resolution)
    labels, counts = _component_counts(mask)
    keep = _component_keep_mask(counts, resolution,
        mode=getattr(params, "cc_filter_mode", "hybrid"),
        min_volume_mm3=getattr(params, "min_cc_volume_mm3", 50.0),
        rel_min_ratio=getattr(params, "cc_rel_min_ratio", 0.10))
    return _skeleton_from_components(labels, len(counts) - 1, resolution, keep)


def _merge_nearby_skeleton_points(point_sets, radius_mm):
    """Merge points from multiple skeleton passes without collapsing a line.

    AutoFlow skeleton points are sampled on the voxel grid.  The default
    special merge radius (1.5 mm) is below the usual 2.4 mm voxel spacing, so
    adjacent points on one pass remain distinct while coincident points from
    different passes are averaged.
    """
    chunks = [np.asarray(points, dtype=float).reshape(-1, 3) for points in point_sets if len(points)]
    if not chunks:
        return np.empty((0, 3), dtype=float)
    points = np.vstack(chunks)
    radius = max(0.0, float(radius_mm))
    if radius <= 0.0 or len(points) < 2:
        return points

    parent = np.arange(len(points), dtype=np.int64)

    def find(index):
        index = int(index)
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    for left, right in sorted(cKDTree(points).query_pairs(radius)):
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    roots = np.asarray([find(index) for index in range(len(points))], dtype=np.int64)
    unique_roots, inverse = np.unique(roots, return_inverse=True)
    return np.vstack([
        points[inverse == group_id].mean(axis=0)
        for group_id in range(len(unique_roots))
    ])


def generate_three_pass_special_skeleton(
    label_mask_3d,
    group_label_values,
    special_label_values,
    params,
    resolution,
    *, exact_union=False,
):
    """Skeletonize A+B, A+C, ... and merge the resulting point sets.

    ``A`` is every label in ``group_label_values`` except the configured
    special labels.  Each pass uses the normal AutoFlow preprocessing and
    voxel skeletonizer, but cannot grow back into an excluded special label.
    The returned mask is the union of the per-pass skeleton masks.
    """
    labels = np.asarray(label_mask_3d)
    if labels.ndim != 3:
        raise ValueError(f"label_mask_3d must be 3D, got {labels.shape}")
    group_values = sorted({int(value) for value in (group_label_values or []) if int(value) != 0})
    special_values = [
        int(value) for value in special_label_values
        if int(value) != 0 and int(value) in group_values
    ]
    special_values = list(dict.fromkeys(special_values))
    base_values = [value for value in group_values if value not in special_values]
    if len(special_values) < 2 or not base_values:
        mask = np.isin(labels, group_values)
        processed = preprocess_mask_for_skeleton(mask, params, resolution=resolution)
        return generate_skeleton_from_mask3d(processed, resolution)

    pass_points = []
    merged_mask = np.zeros(labels.shape, dtype=bool)
    special_regions = np.isin(labels, special_values)
    base_mask = np.isin(labels, base_values)
    morphology_enabled = (params.do_closing or params.do_opening
        or any(getattr(params, name, 0) > 0 for name in (
            "dilation_iters", "erosion_iters", "closing_iters", "opening_iters"))
        or (params.gaussian_enabled and params.gaussian_sigma > 0))
    for special_value in special_values:
        union_mask = base_mask | (labels == special_value)
        # Closing can fill a narrow gap with voxels belonging to another
        # special branch. Reapplying the exclusion after morphology keeps
        # A+B independent of C/D and prevents false loops and terminal spurs
        # when the per-pass skeletons are merged into a neighbouring graph.
        if morphology_enabled:
            excluded = special_regions & (labels != special_value)
            processed = preprocess_mask_for_skeleton(
                union_mask, params, resolution=resolution,
                post_morphology=lambda mask: mask & ~excluded)
            points, skeleton_mask = generate_skeleton_from_mask3d(processed, resolution)
        else:
            # With no morphology the exclusion is already exact in union_mask.
            # Apply the same component policy, but do not label its result twice.
            points, skeleton_mask = _skeletonize_unmodified_pass(union_mask, params, resolution)
        if len(points):
            pass_points.append(np.asarray(points, dtype=float))
        merged_mask |= np.asarray(skeleton_mask, dtype=bool)

    radius = getattr(params, "special_merge_radius_mm", 1.5)
    merged_points = (np.unique(np.vstack(pass_points), axis=0) if exact_union and pass_points
                     else _merge_nearby_skeleton_points(pass_points, radius))
    return merged_points, merged_mask
