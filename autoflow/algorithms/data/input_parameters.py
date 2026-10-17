"""Validate session input calibration and map physical triplets to source axes."""

import numpy as np

from .orientation import _axis_pair
from .venc import _split_dual_venc_triplets


def normalize_input_parameters(parameters):
    """Resolution/VENC use LR, AP, FH; orders describe the source arrays."""
    parameters = dict(parameters or {})
    unknown = parameters.keys() - {"resolution", "venc", "spatial_order", "venc_order"}
    if unknown:
        raise ValueError(f"Unknown input parameters: {sorted(unknown)}")
    result = {}
    for name, value in parameters.items():
        if isinstance(value, str):
            value = value.replace(";", ",").split(",")
        if name in {"resolution", "venc"}:
            values = np.asarray(value, dtype=float).reshape(-1)
            if values.size == 1:
                values = np.repeat(values, 3)
            if values.size != 3 or not np.all(np.isfinite(values)) or np.any(values <= 0):
                raise ValueError(f"{name} requires three positive finite values (LR, AP, FH)")
            result[name] = values.tolist()
        else:
            values = [str(item).strip().upper() for item in value]
            if len(values) != 3 or {_axis_pair(item) for item in values} != {"LR", "AP", "HF"}:
                raise ValueError(f"{name} requires one LR/RL, one AP/PA and one FH/HF direction")
            result[name] = values
    return result


def apply_input_parameters(parameters, resolution, venc, spatial_order, venc_order, dual_venc_mode):
    """Apply calibration before phase decoding and spatial/component normalization."""
    spatial_order = parameters.get("spatial_order", spatial_order)
    venc_order = parameters.get("venc_order", venc_order)
    target_pairs = ["LR", "AP", "HF"]
    if "resolution" in parameters:
        indices = [target_pairs.index(_axis_pair(axis)) for axis in spatial_order]
        resolution = np.asarray(parameters["resolution"], dtype=float)[indices]
    if "venc" in parameters:
        indices = [target_pairs.index(_axis_pair(axis)) for axis in venc_order]
        requested = np.asarray(parameters["venc"], dtype=float)[indices]
        if np.asarray(venc).size == 6:
            low, high, _low_index, _high_index = _split_dual_venc_triplets(venc)
            reference = low if dual_venc_mode == "lv" else high
            # The GUI shows the selected source's VENC. Preserve high/low
            # ratios and the original channel-group ordering.
            venc = np.asarray(venc, dtype=float).reshape(2, 3) * (requested / reference)
            venc = venc.reshape(6)
        else:
            venc = requested
    return resolution, venc, spatial_order, venc_order
