"""Metric entry points with implementations separated by responsibility.

Existing imports from ``autoflow.algorithms.metrics`` remain supported.
Implementation modules use direct imports rather than this compatibility
entry point so numerical families and shared sampling stay independent.
"""

from ._common import (
    _ensure_mask4d,
    _ensure_flow5d,
)

from ._parallel import (
    _PlaneProcessToken,
    _mark_plane_process_progress,
    _wait_plane_processes,
)

from .consistency import (
    summarize_internal_consistency,
    apply_internal_consistency_to_metrics,
)

from .derived import (
    compute_derived_metrics,
)

from .export import (
    save_plane_pixelwise_h5,
    _write_plane_pixelwise_h5,
    load_metrics_as_table,
)

from .plane_derived import (
    _nanpercentile_safe,
    _weighted_mean,
    _append_summary,
    summarize_plane_derived_metrics,
    _augment_plane_metrics_serial,
    _augment_plane_metrics_chunk,
    augment_plane_metrics_with_derived,
)

from .planes import (
    compute_plane_metrics,
    _compute_single_plane_metric,
    _compute_plane_metrics_process_chunk,
    _run_plane_metric_processes,
    compute_plane_metrics_multithread,
)

from .pressure import (
    _periodic_central_difference,
    _finite_percentile_abs,
    _normalize_pressure_method,
    _neighbor_shifts,
    _solve_reconstruction_system,
    _build_pressure_reconstruction_system,
    _pressure_reconstruction_rhs,
    _build_least_squares_system,
    _build_ppe_system,
    _least_squares_rhs,
    _ppe_rhs,
    _build_ste_mac_system,
    _solve_ste_mac_pressure,
    reconstruct_relative_pressure_map,
    _points_as_voxels,
    _sample_volume_at_points,
    compute_centerline_pressure_profiles,
    compute_pressure_gradient_metrics,
)

from .sampling import (
    _target_label_for_plane,
    _build_mask_phase_lookup,
    _build_plane_support_mesh,
    _build_plane_support_mesh_cache,
    _add_labeled_plane_support_meshes,
    filter_planes_by_branch_support,
    _build_plane_slice_region,
    _points_in_polygon,
    _plane_roi_point_mask,
    _build_plane_slice_spec,
    _get_cached_plane_slice_spec,
    _sample_field_from_slice_spec,
    _extract_plane_field_region,
)

from .tke import (
    compute_tke_array_from_sigma,
    _prepare_tke_array,
    compute_tke_metrics,
)

from .vortex import (
    compute_vortex_metrics,
)

from .wss import (
    extract_vectors,
    get_orthogonal_vectors,
    get_vector_magnitude,
    align_tangential_samples,
    resolve_wss_inward_distance,
    calculate_gradient,
    cal_wss_from_surf,
    compute_wss_metrics,
)

__all__ = [
    "extract_vectors",
    "get_orthogonal_vectors",
    "get_vector_magnitude",
    "align_tangential_samples",
    "resolve_wss_inward_distance",
    "calculate_gradient",
    "cal_wss_from_surf",
    "summarize_internal_consistency",
    "apply_internal_consistency_to_metrics",
    "compute_plane_metrics",
    "filter_planes_by_branch_support",
    "summarize_plane_derived_metrics",
    "augment_plane_metrics_with_derived",
    "save_plane_pixelwise_h5",
    "compute_tke_array_from_sigma",
    "compute_tke_metrics",
    "compute_wss_metrics",
    "reconstruct_relative_pressure_map",
    "compute_centerline_pressure_profiles",
    "compute_vortex_metrics",
    "compute_pressure_gradient_metrics",
    "compute_derived_metrics",
    "compute_plane_metrics_multithread",
    "load_metrics_as_table",
]
