"""nnUNet channel definitions, coordinates and temporal feature volumes."""

import json
import re
import numpy as np
from ..data.orientation import _reorient_spatial_only, reorient

from .models import (
    _AUTOFLOW_INTERNAL_SPATIAL_ORDER,
    _AUTOFLOW_INTERNAL_VENC_ORDER,
    _NNUNET_DEFAULT_CHANNEL_ORDER,
    _NNUNET_TARGET_SPATIAL_ORDER,
    _NNUNET_TARGET_VENC_ORDER,
)


def _nnunet_normalize_channel_name(name):
    token = str(name or "").strip().lower()
    aliases = {
        "mag_mean": "mag_mean_xyz",
        "mean_mag": "mag_mean_xyz",
        "mag_std": "mag_std_xyz",
        "std_mag": "mag_std_xyz",
        "pcmra_mean": "pcmra_mean_xyz",
        "pcmra_std": "pcmra_std_xyz",
        "flow_x_mean": "flow_x_mean_xyz",
        "flow_y_mean": "flow_y_mean_xyz",
        "flow_z_mean": "flow_z_mean_xyz",
        "flow_mag_mean": "flow_mag_mean_xyz",
        "flow_speed_mean": "flow_mag_mean_xyz",
        "flow_x_std": "flow_x_std_xyz",
        "flow_y_std": "flow_y_std_xyz",
        "flow_z_std": "flow_z_std_xyz",
        "flow_mag_std": "flow_mag_std_xyz",
        "flow_speed_std": "flow_mag_std_xyz",
    }
    return aliases.get(token, token)


def _nnunet_spatial_affine(resolution, spatial_shape):
    res = np.asarray(resolution, dtype=np.float32).reshape(-1)
    if res.size == 1:
        res = np.repeat(res, 3)
    shape = np.asarray(spatial_shape, dtype=np.int32).reshape(-1)
    if shape.size != 3:
        raise ValueError(f"spatial_shape must be length 3, got {tuple(shape.tolist())}")
    x_size, y_size, z_size = int(shape[0]), int(shape[1]), int(shape[2])
    dx, dy, dz = float(res[0]), float(res[1]), float(res[2])
    # Match the affine convention used by the nnUNet training/export scripts so
    # auto-seg inference sees the same voxel-to-world geometry.
    affine = np.array([
        [0.0, 0.0, -dz, dz * (z_size - 1) / 2.0],
        [0.0, -dy, 0.0, dy * (y_size - 1) / 2.0],
        [-dx, 0.0, 0.0, dx * (x_size - 1) / 2.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)
    return affine


def _ensure_nnunet_mag_flow(mag, flow):
    mag = np.asarray(mag, dtype=np.float32)
    flow = np.asarray(flow, dtype=np.float32)
    if flow.ndim == 4 and flow.shape[-1] == 3:
        flow = flow[..., np.newaxis, :]
    if flow.ndim != 5 or flow.shape[-1] != 3:
        raise ValueError(f"flow must be XYZTV or XYZV with 3 components, got shape={flow.shape}")
    nt = int(flow.shape[3])
    if mag.ndim == 3:
        mag = np.repeat(mag[..., np.newaxis], nt, axis=3)
    elif mag.ndim == 4 and mag.shape[3] == 1 and nt > 1:
        mag = np.repeat(mag, nt, axis=3)
    elif mag.ndim != 4:
        raise ValueError(f"mag must be XYZT or XYZ, got shape={mag.shape}")
    if mag.shape[3] != nt:
        if mag.shape[3] == 1:
            mag = np.repeat(mag, nt, axis=3)
        else:
            raise ValueError(f"mag time dimension {mag.shape[3]} does not match flow {nt}")
    return mag, flow


def _prepare_nnunet_inputs(mag, flow, resolution):
    seg_dummy = np.zeros(np.asarray(mag).shape[:3], dtype=np.int16)
    flow_r, mag_r, _seg_r, _venc_r, resolution_r = reorient(
        mag,
        flow,
        seg_dummy,
        venc=np.ones(3, dtype=np.float32),
        resolution=resolution,
        spatial_order=_AUTOFLOW_INTERNAL_SPATIAL_ORDER,
        venc_order=_AUTOFLOW_INTERNAL_VENC_ORDER,
        target_spatial_order=_NNUNET_TARGET_SPATIAL_ORDER,
        target_venc_order=_NNUNET_TARGET_VENC_ORDER,
        return_velocity=False,
        normalize_mag=False,
    )
    return (
        np.asarray(mag_r, dtype=np.float32),
        np.asarray(flow_r, dtype=np.float32),
        np.asarray(resolution_r, dtype=np.float32),
    )


def _restore_autoflow_segmentation(segmentation):
    seg = _reorient_spatial_only(
        segmentation,
        spatial_order=_NNUNET_TARGET_SPATIAL_ORDER,
        target_spatial_order=_AUTOFLOW_INTERNAL_SPATIAL_ORDER,
    )
    return np.asarray(seg, dtype=np.int16)


def _ordered_mapping_values(payload):
    if isinstance(payload, dict):
        def _sort_key(item):
            key = item[0]
            try:
                return (0, int(key))
            except Exception:
                return (1, str(key))

        return [value for _, value in sorted(payload.items(), key=_sort_key)]
    if isinstance(payload, list):
        return list(payload)
    raise ValueError(f"expected mapping or list, got {type(payload).__name__}")


def _resolve_nnunet_channel_token(channel_name, channel_index):
    token = _nnunet_normalize_channel_name(channel_name)
    token = {
        "mag": "mag_mean_xyz",
        "pcmra": "pcmra_mean_xyz",
    }.get(token, token)
    if token in _NNUNET_DEFAULT_CHANNEL_ORDER:
        return token
    if 0 <= int(channel_index) < len(_NNUNET_DEFAULT_CHANNEL_ORDER):
        return _NNUNET_DEFAULT_CHANNEL_ORDER[int(channel_index)]
    raise ValueError(
        f"unsupported nnUNet channel '{channel_name}'. "
        f"Supported channels: {list(_NNUNET_DEFAULT_CHANNEL_ORDER)}"
    )


def _nnunet_channel_volumes(channel_names, mag, flow, *, speed=None, pcmra=None):
    tokens = [
        _resolve_nnunet_channel_token(channel_name, channel_index)
        for channel_index, channel_name in enumerate(channel_names)
    ]
    requested = set(tokens)
    channels = {}

    if "mag_mean_xyz" in requested:
        channels["mag_mean_xyz"] = np.mean(mag, axis=3)
    if "mag_std_xyz" in requested:
        channels["mag_std_xyz"] = np.std(mag, axis=3)

    for axis_name, axis_index in (("x", 0), ("y", 1), ("z", 2)):
        mean_key = f"flow_{axis_name}_mean_xyz"
        std_key = f"flow_{axis_name}_std_xyz"
        if mean_key in requested or std_key in requested:
            component = flow[..., axis_index]
            if mean_key in requested:
                channels[mean_key] = np.mean(component, axis=3)
            if std_key in requested:
                channels[std_key] = np.std(component, axis=3)

    speed_keys = {
        "flow_mag_mean_xyz",
        "flow_mag_std_xyz",
        "pcmra_mean_xyz",
        "pcmra_std_xyz",
    }
    if requested.intersection(speed_keys):
        # Compute the norm once.  The 4D model requests several speed/PCMRA
        # channels and recalculating this volume for each channel is a large
        # avoidable allocation for clinical-size inputs.
        if speed is None:
            speed = np.linalg.norm(flow, axis=-1)
        if "flow_mag_mean_xyz" in requested:
            channels["flow_mag_mean_xyz"] = np.mean(speed, axis=3)
        if "flow_mag_std_xyz" in requested:
            channels["flow_mag_std_xyz"] = np.std(speed, axis=3)
        if "pcmra_mean_xyz" in requested or "pcmra_std_xyz" in requested:
            if pcmra is None:
                pcmra = mag * speed
            if "pcmra_mean_xyz" in requested:
                channels["pcmra_mean_xyz"] = np.mean(pcmra, axis=3)
            if "pcmra_std_xyz" in requested:
                channels["pcmra_std_xyz"] = np.std(pcmra, axis=3)

    return [np.asarray(channels[token], dtype=np.float32) for token in tokens]


def _parse_nnunet_temporal_channel(name):
    """Return ``(offset, feature)`` for a temporal channel name.

    Dataset7020 names channels as ``tm2_flow_x``, ``tp1_mag`` and ``tp0_pcmra``.
    The parser also accepts ``t-2_*``/``t+1_*`` forms so exported datasets can
    use a more readable spelling without changing the predictor.
    """
    token = str(name or "").strip().lower()
    match = re.match(r"^t([mp])(\d+)_(flow_[xyz]|mag|pcmra)$", token)
    if match:
        offset = int(match.group(2)) * (-1 if match.group(1) == "m" else 1)
        return offset, match.group(3)
    match = re.match(r"^t([+-]?\d+)_(flow_[xyz]|mag|pcmra)$", token)
    if match:
        return int(match.group(1)), match.group(2)
    return None


def _nnunet_4d_channel_volumes(
    channel_names, mag, flow, frame_index, global_by_name=None, temporal_cache=None
):
    """Build one 4D-model sample from a circular temporal neighbourhood."""
    mag = np.asarray(mag, dtype=np.float32)
    flow = np.asarray(flow, dtype=np.float32)
    if mag.ndim != 4 or flow.ndim != 5 or flow.shape[:4] != mag.shape:
        raise ValueError(
            f"4D nnUNet inputs must be mag XYZT and flow XYZT3, got {mag.shape} and {flow.shape}"
        )
    nt = int(mag.shape[3])
    if nt < 1:
        raise ValueError("4D nnUNet input has no time frames")
    # Cache all full-cycle statistics once per case.  Every frame shares these
    # twelve channels; only the temporal window changes.
    global_names = [
        "mag_std_xyz", "mag_mean_xyz", "pcmra_std_xyz", "pcmra_mean_xyz",
        "flow_x_mean_xyz", "flow_y_mean_xyz", "flow_z_mean_xyz", "flow_mag_mean_xyz",
        "flow_x_std_xyz", "flow_y_std_xyz", "flow_z_std_xyz", "flow_mag_std_xyz",
    ]
    if global_by_name is None:
        global_channels = _nnunet_channel_volumes(global_names, mag, flow)
        global_by_name = dict(zip(global_names, global_channels))
    speed = None
    pcmra = None
    result = []
    for index, raw_name in enumerate(channel_names):
        name = _nnunet_normalize_channel_name(raw_name)
        if name in global_by_name:
            result.append(global_by_name[name])
            continue
        parsed = _parse_nnunet_temporal_channel(name)
        if parsed is None:
            # Preserve the existing positional aliases for old 12-channel
            # models, while producing a useful error for a malformed 4D spec.
            if index < len(global_names):
                result.append(global_by_name[global_names[index]])
                continue
            raise ValueError(f"unsupported 4D nnUNet channel '{raw_name}'")
        offset, feature = parsed
        frame = (int(frame_index) + int(offset)) % nt
        if feature == "mag":
            result.append(
                mag[..., frame] if temporal_cache is None else temporal_cache["mag"][..., frame]
            )
        elif feature == "pcmra":
            if temporal_cache is not None:
                pcmra = temporal_cache["pcmra"]
            elif pcmra is None:
                if speed is None:
                    speed = np.linalg.norm(flow, axis=-1)
                pcmra = mag * speed
            result.append(pcmra[..., frame])
        else:
            if feature == "flow_x": component = 0
            elif feature == "flow_y": component = 1
            else: component = 2
            value = flow[..., frame, component]
            if temporal_cache is not None:
                value = temporal_cache[f"flow_{feature[-1]}"][..., frame]
            result.append(value)
    return [np.asarray(value, dtype=np.float32) for value in result]


def _nnunet_channel_volume(channel_name, mag, flow, channel_index=0):
    token = _resolve_nnunet_channel_token(channel_name, channel_index)
    return _nnunet_channel_volumes([token], mag, flow)[0]


def _parse_nnunet_label_map(label_map_spec, model_labels):
    if label_map_spec is None:
        return {}
    if isinstance(label_map_spec, str):
        text = label_map_spec.strip()
        if not text:
            return {}
        try:
            payload = json.loads(text)
        except Exception as exc:
            raise ValueError(f"auto_label_map must be valid JSON: {exc}") from exc
    elif isinstance(label_map_spec, dict):
        payload = label_map_spec
    else:
        raise ValueError(f"auto_label_map must be JSON object or string, got {type(label_map_spec).__name__}")
    if not payload:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("auto_label_map must decode to a JSON object")

    name_to_id = {}
    if isinstance(model_labels, dict):
        for key, value in model_labels.items():
            try:
                label_id = int(value)
            except Exception:
                continue
            name_to_id[str(key).strip().lower()] = label_id

    mapping = {}
    for raw_key, raw_value in payload.items():
        try:
            target_id = int(raw_value)
        except Exception as exc:
            raise ValueError(f"auto_label_map values must be integers, got {raw_value!r}") from exc
        key_token = str(raw_key).strip()
        if key_token.lstrip("-").isdigit():
            source_id = int(key_token)
        else:
            lookup = key_token.lower()
            if lookup not in name_to_id:
                raise ValueError(
                    f"auto_label_map key {raw_key!r} is not numeric and not present in model labels"
                )
            source_id = int(name_to_id[lookup])
        mapping[source_id] = target_id
    return mapping


def _apply_label_map(segmentation, label_map):
    seg = np.asarray(segmentation, dtype=np.int16).copy()
    if not label_map:
        return seg
    for source_id, target_id in label_map.items():
        seg[seg == int(source_id)] = int(target_id)
    return seg
