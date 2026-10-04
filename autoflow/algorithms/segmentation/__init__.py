"""Compatible segmentation entry points, separated by implementation responsibility."""

from ._common import (
    segmentation_timestamp,
    collapse_segmentation_to_3d,
    broadcast_segmentation_to_time,
    normalize_segmentation_volume,
)

from .io import (
    _collect_h5_datasets,
    _load_segmentation_from_h5,
    load_segmentation_file,
    save_segmentation_file,
    save_nifti_volume,
    save_segmentation_to_source_h5,
    _sanitize_nnunet_artifact_token,
    _nnunet_artifact_paths,
    _write_nifti_volume,
    _write_nifti_segmentation,
    _link_or_copy_file,
    _read_nifti_segmentation,
)

from .threshold import (
    _otsu_threshold,
    _finite_scalar_max,
    _resolve_threshold_config,
    compute_reference_scalar,
    generate_threshold_segmentation,
)

from .channels import (
    _nnunet_normalize_channel_name,
    _nnunet_spatial_affine,
    _ensure_nnunet_mag_flow,
    _prepare_nnunet_inputs,
    _restore_autoflow_segmentation,
    _ordered_mapping_values,
    _resolve_nnunet_channel_token,
    _nnunet_channel_volumes,
    _parse_nnunet_temporal_channel,
    _nnunet_4d_channel_volumes,
    _nnunet_channel_volume,
    _parse_nnunet_label_map,
    _apply_label_map,
)

from .models import (
    _load_nnunet_model_metadata,
    _detect_nnunet_folds,
    bundled_nnunet_model_folder,
    default_nnunet_model_folder,
    _resolve_bundled_relative_path,
    resolve_nnunet_model_folder,
    default_nnunet_4d_pipeline_script,
    _model_folder_from_4d_pipeline_script,
    _python_from_4d_pipeline_script,
    _nnunet_4d_pipeline_assignments,
    _nnunet_4d_subprocess_env,
    resolve_nnunet_4d_model_folder,
    _resolve_nnunet_checkpoint,
    _resolve_nnunet_folds,
    _AUTOFLOW_INTERNAL_SPATIAL_ORDER,
    _AUTOFLOW_INTERNAL_VENC_ORDER,
    _NNUNET_TARGET_SPATIAL_ORDER,
    _NNUNET_TARGET_VENC_ORDER,
    _NNUNET_GPU_PREPROCESSING_ENV,
    _NNUNET_4D_BACKENDS,
    _NNUNET_3D_MODEL_DEFAULT,
    _NNUNET_3D_CHECKPOINT_DEFAULT,
    _NNUNET_4D_MODEL_DEFAULT,
    _NNUNET_4D_CHECKPOINT_DEFAULT,
    _NNUNET_4D_PIPELINE_DEFAULT,
    _NNUNET_4D_RESULTS_DEFAULT,
    _NNUNET_4D_DATASET_DEFAULT,
    _NNUNET_4D_TRAINER_DEFAULT,
    _NNUNET_4D_PLANS_DEFAULT,
    _NNUNET_4D_CONFIGURATION_DEFAULT,
    _NNUNET_DEFAULT_CHANNEL_ORDER,
)

from .runtime import (
    _gpu_preprocessing_enabled,
    _link_nnunet_model_file,
    _prepare_nnunet_gpu_model_folder,
    _run_subprocess,
    _subprocess_failure_details,
    _nnunet_inference_command,
    _emit_progress,
    _nnunet_predict_command,
    resolve_auto_segmentation_device,
)

from .nnunet_grouped import (
    _load_4d_pipeline_module,
    _install_nnunet_4d_resampler,
    _nnunet_4d_temporal_radius,
    _nnunet_4d_grouped_requested,
    _generate_nnunet_4d_grouped,
)

from .nnunet_static import (
    generate_nnunet_auto_segmentation,
)

from .nnunet_temporal import (
    generate_nnunet_4d_auto_segmentation,
)

__all__ = [
    "segmentation_timestamp",
    "collapse_segmentation_to_3d",
    "broadcast_segmentation_to_time",
    "normalize_segmentation_volume",
    "load_segmentation_file",
    "compute_reference_scalar",
    "generate_threshold_segmentation",
    "save_segmentation_file",
    "save_nifti_volume",
    "save_segmentation_to_source_h5",
    "bundled_nnunet_model_folder",
    "default_nnunet_model_folder",
    "resolve_nnunet_model_folder",
    "default_nnunet_4d_pipeline_script",
    "resolve_nnunet_4d_model_folder",
    "resolve_auto_segmentation_device",
    "generate_nnunet_auto_segmentation",
    "generate_nnunet_4d_auto_segmentation",
]
