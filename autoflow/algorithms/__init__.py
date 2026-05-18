"""Domain-specific algorithm helpers for AutoFlow.

This package preserves the historical ``autoflow.algorithms`` import surface
while keeping each implementation module focused on one processing stage.
"""

from .data import (
    LoadedCase,
    LoaderCapabilities,
    _axis_pair,
    _need_flip,
    _permute_spatial,
    _flip_axes,
    reorient,
    _ensure_flow_mag_time_and_segmask,
    normalize_loaded_case,
    _reorient_spatial_only,
    _compute_spatial_bbox,
    _target_bbox_to_source_slices,
    _sigma_from_complex,
    _reorient_component_abs,
    load_h5_data,
)
from .phase_correction import (
    apply_background_phase_correction_to_complex,
    apply_background_phase_correction_to_mag_flow,
    background_phase_report_for_metadata,
    coerce_background_phase_correction_config,
)

from .dicom import (
    collect_input_cases,
    inspect_dicom_case,
    load_dicom_case,
    load_input_data,
    resolve_input_case,
    scan_dicom_cases,
)

from .preprocess import (
    filter_segmask_labels,
    binarize_segmask,
    merge_segmask_to_3d,
    _connected_components,
    remove_small_cc_from_binary_mask,
    _component_bbox,
    _preprocess_single_component,
    preprocess_mask_for_skeleton,
    largest_connected_component,
)

from .skeleton import (
    generate_skeleton_from_mask3d,
)

from .graph import (
    build_graph_from_points,
    remove_triangle_cycles,
    graph_to_networkx,
    graph_to_polydata,
)

from .paths import (
    _vector_orientation_text,
    _path_cumulative_distance,
    _path_point_at_distance,
    _path_tangent_from_segment,
    _project_point_to_path,
    _determine_plane_forward,
    smooth_path_savgol,
    inter_points,
)

from .branch import (
    _orient_node_paths_by_flow,
    find_path_forks,
    build_path_info,
    segment_vessels_from_graph_and_mask,
)

from .planes import (
    generate_planes_from_paths,
)

from .surfaces import (
    build_multilabel_surface,
    build_multilabel_surface_t,
    build_binary_surface_t,
    build_surface_from_mask3d,
    _build_branch_grid,
    _select_connected_region,
    extract_plane_cross_section,
    _flow_grid_for_t,
    _extract_plane_flow_region,
    create_vector_volume_from_flow,
    create_uniform_grid,
    create_uniform_vector,
)

from .segmentation import (
    segmentation_timestamp,
    collapse_segmentation_to_3d,
    broadcast_segmentation_to_time,
    normalize_segmentation_volume,
    load_segmentation_file,
    compute_reference_scalar,
    generate_threshold_segmentation,
    save_segmentation_file,
)

from .streamlines import (
    generate_seed_points,
    generate_streamlines_at_t,
    generate_streamlines_from_plane_at_t,
)

from .metrics import (
    extract_vectors,
    get_orthogonal_vectors,
    get_vector_magnitude,
    calculate_gradient,
    cal_wss_from_surf,
    summarize_internal_consistency,
    apply_internal_consistency_to_metrics,
    compute_plane_metrics,
    compute_tke_array_from_sigma,
    compute_tke_metrics,
    compute_wss_metrics,
    compute_derived_metrics,
    _compute_single_plane_metric,
    compute_plane_metrics_multithread,
    load_metrics_as_table,
)
__all__ = [
    "LoadedCase",
    "LoaderCapabilities",
    "reorient",
    "load_h5_data",
    "apply_background_phase_correction_to_complex",
    "apply_background_phase_correction_to_mag_flow",
    "background_phase_report_for_metadata",
    "coerce_background_phase_correction_config",
    "load_input_data",
    "load_dicom_case",
    "resolve_input_case",
    "scan_dicom_cases",
    "collect_input_cases",
    "inspect_dicom_case",
    "normalize_loaded_case",
    "filter_segmask_labels",
    "binarize_segmask",
    "merge_segmask_to_3d",
    "remove_small_cc_from_binary_mask",
    "preprocess_mask_for_skeleton",
    "largest_connected_component",
    "generate_skeleton_from_mask3d",
    "build_graph_from_points",
    "remove_triangle_cycles",
    "graph_to_networkx",
    "graph_to_polydata",
    "smooth_path_savgol",
    "inter_points",
    "find_path_forks",
    "build_path_info",
    "segment_vessels_from_graph_and_mask",
    "generate_planes_from_paths",
    "build_multilabel_surface",
    "build_multilabel_surface_t",
    "build_binary_surface_t",
    "build_surface_from_mask3d",
    "segmentation_timestamp",
    "collapse_segmentation_to_3d",
    "broadcast_segmentation_to_time",
    "normalize_segmentation_volume",
    "load_segmentation_file",
    "compute_reference_scalar",
    "generate_threshold_segmentation",
    "save_segmentation_file",
    "extract_plane_cross_section",
    "create_vector_volume_from_flow",
    "create_uniform_grid",
    "create_uniform_vector",
    "generate_seed_points",
    "generate_streamlines_at_t",
    "generate_streamlines_from_plane_at_t",
    "extract_vectors",
    "get_orthogonal_vectors",
    "get_vector_magnitude",
    "calculate_gradient",
    "cal_wss_from_surf",
    "summarize_internal_consistency",
    "apply_internal_consistency_to_metrics",
    "compute_plane_metrics",
    "compute_tke_array_from_sigma",
    "compute_tke_metrics",
    "compute_wss_metrics",
    "compute_derived_metrics",
    "compute_plane_metrics_multithread",
    "load_metrics_as_table",
]
