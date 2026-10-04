"""H5 acquisition discovery and embedded-capability inspection."""

import os
import re
import h5py
import numpy as np
from ...case_types import InputCase

from .h5_metadata import (
    _coerce_h5_numeric_array,
    _discover_h5_data_group_names,
    _find_h5_dataset,
    _find_h5_dataset_from_scopes,
    _read_h5_value_from_scopes,
    _resolve_h5_data_group,
)
from .venc import _split_dual_venc_triplets


def _h5_group_embedded_features(handle, group_name=None):
    group = handle if not group_name else handle[str(group_name).strip("/")]
    scopes = [group] if group is handle else [group, handle]
    seg_ds = _find_h5_dataset_from_scopes(scopes, "segmask", "segmentation", "seg")
    corr_ds = _find_h5_dataset_from_scopes(scopes, "corr")
    corr_low_ds = _find_h5_dataset_from_scopes(scopes, "corr_low")
    corr_high_ds = _find_h5_dataset_from_scopes(scopes, "corr_high")

    def _cache_method(dataset):
        if dataset is None:
            return None
        value = dataset.attrs.get("corr_algorithm", "msac")
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        token = str(value or "msac").strip().lower().replace("-", "_").replace("+", "_")
        token = {"wrls": "wrls_arto", "arto": "wrls_arto", "wrlsarto": "wrls_arto"}.get(token, token)
        return token if token in {"msac", "wrls_arto"} else None

    correction_method = None
    has_correction_cache = False
    if corr_ds is not None:
        correction_method = _cache_method(corr_ds)
        has_correction_cache = correction_method is not None
    elif corr_low_ds is not None and corr_high_ds is not None:
        low_method = _cache_method(corr_low_ds)
        high_method = _cache_method(corr_high_ds)
        if low_method is not None and low_method == high_method:
            correction_method = low_method
            has_correction_cache = True

    features = {
        "has_embedded_segmentation": seg_ds is not None,
        "has_background_correction_cache": bool(has_correction_cache),
    }
    # Detect legacy dual-VENC cases from metadata and dataset shape without
    # materializing the image volume.  The GUI uses this to ask which source
    # (LV, HV, or reconstructed DV) should be exposed as ``flow``.
    # Image layout belongs to the selected case group.  Root-level metadata
    # may be shared by several groups, so do not use a root image as a dual
    # marker for a nested normalized case.
    img_ds = _find_h5_dataset(group, "img_complex", "img")
    is_dual_venc = bool(
        img_ds is not None
        and np.issubdtype(img_ds.dtype, np.complexfloating)
        and img_ds.ndim == 5
        and int(img_ds.shape[-1]) == 7
    )
    dual_info = {"enabled": is_dual_venc}
    if is_dual_venc:
        try:
            raw_venc = _read_h5_value_from_scopes(scopes, "VENC", "venc", default=None)
            venc_values = _coerce_h5_numeric_array(raw_venc, np.array([], dtype=float))
            if venc_values.size == 6:
                low, high, _low_group, _high_group = _split_dual_venc_triplets(venc_values)
                dual_info.update({"lv_venc": low.astype(float).tolist(), "hv_venc": high.astype(float).tolist()})
        except (TypeError, ValueError):
            # The loader will provide the detailed error if the VENC metadata
            # is malformed; discovery should still list the H5 case.
            pass
    features["is_dual_venc"] = is_dual_venc
    features["dual_venc"] = dual_info
    if correction_method is not None:
        features["background_correction_method"] = correction_method
    return features


def inspect_h5_input_case(case):
    """Inspect embedded H5 artifacts without materializing image arrays."""
    if isinstance(case, InputCase):
        path = os.path.abspath(str(case.input_path))
        source_group = case.source_group
    else:
        path = os.path.abspath(str(case))
        source_group = None
    with h5py.File(path, "r") as handle:
        group, group_name = _resolve_h5_data_group(handle, source_group=source_group)
        features = _h5_group_embedded_features(
            handle,
            group_name=None if group is handle else group_name,
        )
    features["source_group"] = group_name
    return features


def discover_h5_input_cases(path):
    path = os.path.abspath(str(path))
    stem = os.path.splitext(os.path.basename(path))[0]
    with h5py.File(path, "r") as handle:
        candidate_names = _discover_h5_data_group_names(handle)
        if not candidate_names or candidate_names == [""]:
            return [
                InputCase(
                    input_path=path,
                    input_kind="h5",
                    display_name=stem,
                    output_name=stem,
                    source_group=None,
                    metadata=_h5_group_embedded_features(handle),
                )
            ]

        cases = []
        for group_name in candidate_names:
            group_token = str(group_name or "").strip("/")
            short_name = group_token.rsplit("/", 1)[-1] if group_token else stem
            safe_group = re.sub(r"[^A-Za-z0-9._-]+", "_", group_token).strip("._-") or "group"
            metadata = {
                "source_group": group_token or None,
                "source_group_name": short_name,
            }
            metadata.update(_h5_group_embedded_features(handle, group_token or None))
            cases.append(
                InputCase(
                    input_path=path,
                    input_kind="h5",
                    display_name=f"{stem} | {group_token}",
                    output_name=f"{stem}__{safe_group}",
                    source_group=group_token or None,
                    metadata=metadata,
                )
            )
    return cases
