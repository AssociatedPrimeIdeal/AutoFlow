"""H5 key lookup, metadata coercion and acquisition-group resolution."""

import re
import h5py
import numpy as np


def _canonical_h5_key(name):
    return re.sub(r"[^a-z0-9]+", "", str(name or "").strip().lower())


def _h5_member_name_map(group):
    mapping = {}
    for name in group.keys():
        mapping.setdefault(_canonical_h5_key(name), str(name))
    return mapping


def _h5_attr_name_map(group):
    mapping = {}
    for name in group.attrs.keys():
        mapping.setdefault(_canonical_h5_key(name), str(name))
    return mapping


def _find_h5_dataset(group, *aliases):
    name_map = _h5_member_name_map(group)
    for alias in aliases:
        actual = name_map.get(_canonical_h5_key(alias))
        if actual is None:
            continue
        obj = group[actual]
        if isinstance(obj, h5py.Dataset):
            return obj
    return None


def _find_h5_dataset_from_scopes(scopes, *aliases):
    for group in scopes:
        ds = _find_h5_dataset(group, *aliases)
        if ds is not None:
            return ds
    return None


def _find_h5_attr(group, *aliases):
    name_map = _h5_attr_name_map(group)
    for alias in aliases:
        actual = name_map.get(_canonical_h5_key(alias))
        if actual is not None:
            return group.attrs[actual]
    return None


def _read_h5_value(group, *aliases, default=None):
    ds = _find_h5_dataset(group, *aliases)
    if ds is not None:
        return ds[()]
    attr = _find_h5_attr(group, *aliases)
    if attr is not None:
        return attr
    return default


def _read_h5_value_from_scopes(scopes, *aliases, default=None):
    for group in scopes:
        value = _read_h5_value(group, *aliases, default=None)
        if value is not None:
            return value
    return default


def _decode_h5_string(value):
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def _coerce_h5_text_array(value, default):
    if value is None:
        return np.asarray(default, dtype=str)
    arr = np.asarray(value)
    if arr.ndim == 0:
        arr = np.asarray([arr.item()])
    tokens = []
    for item in arr.reshape(-1).tolist():
        decoded = _decode_h5_string(item).strip()
        if not decoded:
            continue
        parts = [part.strip() for part in re.split(r"[\s,;]+", decoded) if str(part).strip()]
        tokens.extend(parts or [decoded])
    return np.asarray(tokens or list(default), dtype=str)


def _coerce_h5_scalar_float(value, default):
    if value is None:
        return float(default)
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0:
        return float(default)
    return float(arr[0])


def _coerce_h5_numeric_array(value, default):
    if value is None:
        return np.asarray(default, dtype=float).reshape(-1)
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.asarray(default, dtype=float).reshape(-1)
    return arr.astype(float, copy=False)


def _coerce_h5_triplet(value, default, name, repeat_scalar=False):
    arr = _coerce_h5_numeric_array(value, default)
    if arr.size == 1 and repeat_scalar:
        return np.full(3, float(arr[0]), dtype=float)
    if arr.size != 3:
        raise ValueError(f"{name} must contain 3 values, got shape {arr.shape}")
    return np.asarray(arr, dtype=float)


def _h5_group_source_name(group):
    name = str(getattr(group, "name", "") or "").strip("/")
    return name or None


def _is_h5_data_group_candidate(group):
    if _find_h5_dataset(group, "img_complex") is not None:
        return True
    if _find_h5_dataset(group, "mag") is not None and _find_h5_dataset(group, "flow") is not None:
        return True
    img_ds = _find_h5_dataset(group, "img")
    if img_ds is None:
        return False
    if np.issubdtype(img_ds.dtype, np.complexfloating):
        return img_ds.ndim >= 4 and int(img_ds.shape[-1]) >= 4
    return img_ds.ndim in (4, 5) and int(img_ds.shape[-1]) == 4


def _h5_group_depth(name):
    token = str(name or "").strip("/")
    if not token:
        return 0
    return len(token.split("/"))


def _discover_h5_data_group_names(handle):
    candidate_names = []

    if _is_h5_data_group_candidate(handle):
        candidate_names.append("")

    def _visit(name, obj):
        if isinstance(obj, h5py.Group) and _is_h5_data_group_candidate(obj):
            candidate_names.append(str(name))

    handle.visititems(_visit)
    return sorted(set(candidate_names), key=lambda item: (_h5_group_depth(item), item))


def _resolve_h5_data_group(handle, source_group=None):
    requested_group = str(source_group or "").strip("/")
    if requested_group:
        if requested_group not in handle:
            raise ValueError(f"h5 data group not found: {requested_group}")
        selected = handle[requested_group]
        if not isinstance(selected, h5py.Group):
            raise ValueError(f"h5 data group is not a group: {requested_group}")
        if not _is_h5_data_group_candidate(selected):
            raise ValueError(f"h5 group does not contain a supported case layout: {requested_group}")
        return selected, requested_group

    candidate_names = _discover_h5_data_group_names(handle)
    if not candidate_names:
        return handle, None
    if candidate_names == [""]:
        return handle, None

    shallowest_depth = _h5_group_depth(candidate_names[0])
    shallowest = [name for name in candidate_names if _h5_group_depth(name) == shallowest_depth]
    if len(shallowest) > 1:
        raise ValueError(
            "ambiguous h5 layout: multiple candidate data groups found: "
            + ", ".join(name for name in shallowest if name)
        )

    selected = shallowest[0]
    if not selected:
        return handle, None
    return handle[selected], selected
