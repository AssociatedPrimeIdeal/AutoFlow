import numpy as np
from scipy.ndimage import find_objects
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from .preprocess import _label_components, preprocess_mask_for_skeleton


def generate_skeleton_from_mask3d(mask3d, resolution):
    mask3d = np.asarray(mask3d, dtype=bool)
    skel = np.zeros_like(mask3d, dtype=bool)
    component_labels, component_count = _label_components(mask3d)
    for component_id, bbox in enumerate(find_objects(component_labels), start=1):
        if component_id > int(component_count) or bbox is None:
            continue
        local = component_labels[bbox] == int(component_id)
        if not np.any(local):
            continue
        local_skel = skeletonize(local).astype(bool)
        if np.any(local_skel):
            skel[bbox] |= local_skel
    pts = np.argwhere(skel > 0).astype(float) * np.asarray(resolution, dtype=float).reshape(1, 3)
    return pts, skel


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
):
    """Skeletonize A+B, A+C, ... and merge the resulting point sets.

    ``A`` is every label in ``group_label_values`` except the configured
    special labels.  Each pass uses the normal AutoFlow preprocessing and
    voxel skeletonizer.  The returned mask is the union of the three pass
    skeleton masks so graph construction can use the same topology.
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
    for special_value in special_values:
        pass_values = base_values + [special_value]
        union_mask = np.isin(labels, pass_values)
        processed = preprocess_mask_for_skeleton(union_mask, params, resolution=resolution)
        points, skeleton_mask = generate_skeleton_from_mask3d(processed, resolution)
        if len(points):
            pass_points.append(np.asarray(points, dtype=float))
        merged_mask |= np.asarray(skeleton_mask, dtype=bool)

    radius = getattr(params, "special_merge_radius_mm", 1.5)
    merged_points = _merge_nearby_skeleton_points(pass_points, radius)
    return merged_points, merged_mask
