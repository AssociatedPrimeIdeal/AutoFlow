import numpy as np
from scipy.ndimage import find_objects
from skimage.morphology import skeletonize

from .preprocess import _label_components


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
