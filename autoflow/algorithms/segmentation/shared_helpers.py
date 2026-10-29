"""Portable experimental Dataset7020 helpers with standard per-channel ZScore."""

import numpy as np
from scipy.ndimage import binary_fill_holes

GLOBAL_CHANNEL_NAMES = ("mag_std_xyz", "mag_mean_xyz", "pcmra_std_xyz", "pcmra_mean_xyz",
    "flow_x_mean_xyz", "flow_y_mean_xyz", "flow_z_mean_xyz", "flow_mag_mean_xyz",
    "flow_x_std_xyz", "flow_y_std_xyz", "flow_z_std_xyz", "flow_mag_std_xyz")
TEMPORAL_FEATURE_NAMES = ("flow_x", "flow_y", "flow_z", "mag", "pcmra")
GROUPED_TEMPORAL_PREPROCESS_VERSION = "autoflow-standard-zscore-v1-experimental"


def temporal_source_keys(frame, total, radius):
    return [("global", i, -1) for i in range(12)] + [
        ("temporal", feature, (frame+offset) % total)
        for offset in range(-radius, radius+1) for feature in range(5)]


def _require_grouped_temporal_normalization(configuration):
    schemes = configuration.normalization_schemes
    masks = configuration.use_mask_for_norm
    values = list(masks.values()) if isinstance(masks, dict) else list(masks)
    if set(schemes) != {"ZScoreNormalization"} or len(set(values)) > 1:
        raise ValueError("Shared preprocessing requires identical standard ZScore normalization and mask policy for every channel")


def _transposed_image(image, plans):
    return np.asarray(image).transpose([0, *[axis+1 for axis in plans.transpose_forward]])[0]


def common_4d_crop_metadata(arrays):
    union = np.zeros(next(iter(arrays.values())).shape, dtype=bool)
    global_union = np.zeros_like(union)
    for key, value in arrays.items():
        union |= value != 0
        if key[0] == "global":
            global_union |= value != 0
    # All frames contain these global channels. Otherwise a shared crop could
    # change a standard per-frame crop: reject and use the standard backend.
    if not np.array_equal(union, global_union) or not np.any(union):
        raise ValueError("Shared crop cannot be proven equivalent to every standard temporal sample")
    nonzero = binary_fill_holes(union)
    coordinates = np.where(nonzero)
    bbox = [[int(axis.min()), int(axis.max())+1] for axis in coordinates]
    slices = tuple(slice(lo, hi) for lo, hi in bbox)
    return slices, bbox, nonzero[slices]


def normalize_case_source_maps(arrays, slices, segmentation, preprocessor, configuration, properties):
    _require_grouped_temporal_normalization(configuration)
    result = []
    for source in arrays.values():
        value = source[slices].astype(np.float32, copy=True)
        result.append(preprocessor._normalize(value[None], segmentation, configuration, properties)[0])
    return result, {"mode": "standard_per_source_independent_ZScore", "version": GROUPED_TEMPORAL_PREPROCESS_VERSION}


def _segmentation_for_grouped_training(*args, **kwargs):
    raise ValueError("AutoFlow shared preprocessing supports inference only")
