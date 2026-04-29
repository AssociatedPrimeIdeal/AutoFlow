import numpy as np
from skimage.morphology import skeletonize

from .preprocess import _component_bbox, _connected_components


def generate_skeleton_from_mask3d(mask3d, resolution):
    mask3d = np.asarray(mask3d, dtype=bool)
    skel = np.zeros_like(mask3d, dtype=bool)
    for _, cc in _connected_components(mask3d):
        bbox = _component_bbox(cc)
        if bbox is None:
            continue
        local = cc[bbox]
        if not np.any(local):
            continue
        local_skel = skeletonize(local).astype(bool)
        if np.any(local_skel):
            skel[bbox] |= local_skel
    pts = np.argwhere(skel > 0).astype(float) * np.asarray(resolution, dtype=float).reshape(1, 3)
    return pts, skel
