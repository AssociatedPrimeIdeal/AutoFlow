"""Compatible data entry points, separated by implementation responsibility."""

from ...case_types import BackgroundPhaseCorrectionConfig, InputCase, LoadedCase, LoaderCapabilities
from ..phase_correction import (
    apply_background_phase_correction_to_complex,
    apply_background_phase_correction_to_mag_flow,
    background_phase_correction_cache_metadata,
    background_phase_report_for_metadata,
    coerce_background_phase_correction_config,
)

from .orientation import (
    _axis_pair,
    _need_flip,
    _permute_spatial,
    _flip_axes,
    reorient,
    _reorient_spatial_only,
    _compute_spatial_bbox,
    _target_bbox_to_source_slices,
    _reorient_component_abs,
    _reorient_component_signed,
    _reorient_real_valued_fields,
)

from .normalization import (
    _flow_looks_like_phase_radians,
    _normalize_real_img_layout,
    _ensure_flow_mag_time_and_segmask,
    _ensure_flow_mag_time,
    _ensure_optional_time_volume,
    _ensure_optional_sigma_time,
    normalize_loaded_case,
    _sigma_from_complex,
)

from .correction_cache import (
    _progress_prefix,
    _loader_correction_config,
    _background_phase_corr_attr_scalar,
    _read_background_phase_corr_cache,
    _read_background_phase_corr_cache_from_scopes,
    _write_background_phase_corr_cache,
)

from .h5_metadata import (
    _canonical_h5_key,
    _h5_member_name_map,
    _h5_attr_name_map,
    _find_h5_dataset,
    _find_h5_dataset_from_scopes,
    _find_h5_attr,
    _read_h5_value,
    _read_h5_value_from_scopes,
    _decode_h5_string,
    _coerce_h5_text_array,
    _coerce_h5_scalar_float,
    _coerce_h5_numeric_array,
    _coerce_h5_triplet,
    _h5_group_source_name,
    _is_h5_data_group_candidate,
    _h5_group_depth,
    _discover_h5_data_group_names,
    _resolve_h5_data_group,
)

from .discovery import (
    _h5_group_embedded_features,
    inspect_h5_input_case,
    discover_h5_input_cases,
)

from .venc import (
    _coerce_venc_array,
    _split_dual_venc_triplets,
    _coerce_dual_venc_mode,
    _dual_venc_triplet_ratio,
    _dual_venc_correct_alias,
    _dual_alias_shift_unique_values,
)

from .dual_venc import (
    _load_legacy_dual_venc_h5,
)

from .h5_loader import (
    load_h5_data,
)

__all__ = [
    "reorient",
    "normalize_loaded_case",
    "inspect_h5_input_case",
    "discover_h5_input_cases",
    "load_h5_data",
    "LoadedCase",
    "LoaderCapabilities",
]
