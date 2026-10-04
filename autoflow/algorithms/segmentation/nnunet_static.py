"""Static nnUNet inference and entry-point dispatch."""

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

from ._common import broadcast_segmentation_to_time, segmentation_timestamp
from .channels import (
    _apply_label_map,
    _ensure_nnunet_mag_flow,
    _nnunet_channel_volumes,
    _nnunet_spatial_affine,
    _ordered_mapping_values,
    _parse_nnunet_label_map,
    _prepare_nnunet_inputs,
    _restore_autoflow_segmentation,
)
from .io import (
    _link_or_copy_file,
    _nnunet_artifact_paths,
    _read_nifti_segmentation,
    _write_nifti_segmentation,
    _write_nifti_volume,
)
from .models import (
    _AUTOFLOW_INTERNAL_SPATIAL_ORDER,
    _AUTOFLOW_INTERNAL_VENC_ORDER,
    _NNUNET_3D_CHECKPOINT_DEFAULT,
    _NNUNET_4D_BACKENDS,
    _NNUNET_TARGET_SPATIAL_ORDER,
    _NNUNET_TARGET_VENC_ORDER,
    _load_nnunet_model_metadata,
    _resolve_nnunet_checkpoint,
    _resolve_nnunet_folds,
    resolve_nnunet_model_folder,
)
from .nnunet_temporal import generate_nnunet_4d_auto_segmentation
from .runtime import (
    _emit_progress,
    _gpu_preprocessing_enabled,
    _nnunet_inference_command,
    _prepare_nnunet_gpu_model_folder,
    _run_subprocess,
    _subprocess_failure_details,
    resolve_auto_segmentation_device,
)


def generate_nnunet_auto_segmentation(
    mag,
    flow,
    resolution,
    origin,
    model_folder,
    *,
    backend="nnUNet",
    checkpoint_name="checkpoint_final.pth",
    folds=None,
    device="cpu",
    auto_label_map="",
    case_id="autoflow_case",
    step_size=0.5,
    disable_tta=True,
    num_processes_preprocessing=0,
    num_processes_segmentation_export=1,
    artifact_prefix="",
    runner=None,
    progress_callback=None,
    grouped_preprocessing=None,
):
    if str(backend or "").strip().lower() != "nnunet":
        if str(backend or "").strip().lower() in _NNUNET_4D_BACKENDS:
            return generate_nnunet_4d_auto_segmentation(
                mag,
                flow,
                resolution,
                origin,
                model_folder,
                checkpoint_name=checkpoint_name,
                folds=folds,
                device=device,
                auto_label_map=auto_label_map,
                case_id=case_id,
                num_processes_preprocessing=num_processes_preprocessing,
                num_processes_segmentation_export=num_processes_segmentation_export,
                artifact_prefix=artifact_prefix,
                runner=runner,
                progress_callback=progress_callback,
                grouped_preprocessing=grouped_preprocessing,
            )
        raise ValueError(f"unsupported auto segmentation backend: {backend}")

    t_total_start = time.perf_counter()
    total_stages = 5
    _emit_progress(
        progress_callback,
        stage="autoseg_start",
        message="Resolving nnUNet model and input metadata...",
        current=0,
        total=total_stages,
        elapsed_sec=0.0,
    )

    model_folder = resolve_nnunet_model_folder(model_folder)
    resolved_device = resolve_auto_segmentation_device(device)
    effective_num_processes_preprocessing = int(num_processes_preprocessing or (1 if resolved_device == "cuda" else 3))
    mag, flow = _ensure_nnunet_mag_flow(mag, flow)
    time_count = int(flow.shape[3])
    mag_nnunet, flow_nnunet, resolution_nnunet = _prepare_nnunet_inputs(mag, flow, resolution)
    model_path, dataset_json = _load_nnunet_model_metadata(model_folder)
    channel_names = _ordered_mapping_values(dataset_json.get("channel_names") or dataset_json.get("modality") or {})
    if not channel_names:
        raise ValueError(f"model folder does not define any channel names: {model_path}")
    file_ending = str(dataset_json.get("file_ending", ".nii.gz"))
    if not file_ending.startswith("."):
        file_ending = f".{file_ending}"
    label_map = _parse_nnunet_label_map(auto_label_map, dataset_json.get("labels", {}))
    folds = _resolve_nnunet_folds(model_path, folds)
    checkpoint_name = _resolve_nnunet_checkpoint(
        model_path,
        checkpoint_name,
        folds,
        default_checkpoint=_NNUNET_3D_CHECKPOINT_DEFAULT,
    )
    affine = _nnunet_spatial_affine(resolution_nnunet, mag_nnunet.shape[:3])
    artifact_feature_paths, artifact_prediction_path = _nnunet_artifact_paths(
        artifact_prefix,
        channel_names,
        file_ending,
    )
    _emit_progress(
        progress_callback,
        stage="autoseg_model_ready",
        message=f"Resolved nnUNet model: {model_path.name} | device={resolved_device}",
        current=1,
        total=total_stages,
        elapsed_sec=time.perf_counter() - t_total_start,
        backend="nnUNet",
        device=str(resolved_device),
        checkpoint=str(checkpoint_name),
        model_folder=str(model_path),
    )

    with TemporaryDirectory(prefix="autoflow_nnunet_") as tmp_root:
        tmp_root = Path(tmp_root)
        input_dir = tmp_root / "input"
        output_dir = tmp_root / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        _emit_progress(
            progress_callback,
            stage="autoseg_prepare_inputs",
            message=f"Preparing {len(channel_names)} nnUNet input channel(s)...",
            current=2,
            total=total_stages,
            elapsed_sec=time.perf_counter() - t_total_start,
            detail_current=0,
            detail_total=len(channel_names),
        )
        channel_volumes = _nnunet_channel_volumes(channel_names, mag_nnunet, flow_nnunet)
        def _write_channel(item):
            idx, volume = item
            input_path = input_dir / f"{case_id}_{idx:04d}{file_ending}"
            _write_nifti_volume(volume, affine, input_path)
            if idx < len(artifact_feature_paths):
                _link_or_copy_file(input_path, artifact_feature_paths[idx])

        # NIfTI compression is independent per channel and releases the GIL;
        # overlap the twelve feature writes while preserving deterministic data.
        worker_count = min(4, max(1, len(channel_volumes)))
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="autoflow-nnunet-nifti") as executor:
            list(executor.map(_write_channel, enumerate(channel_volumes)))
        for idx, channel_name in enumerate(channel_names):
            _emit_progress(
                progress_callback,
                stage="autoseg_prepare_inputs",
                message=f"Writing nnUNet input channel {idx + 1}/{len(channel_names)}: {channel_name}",
                current=2,
                total=total_stages,
                elapsed_sec=time.perf_counter() - t_total_start,
                detail_current=idx + 1,
                detail_total=len(channel_names),
                channel_name=str(channel_name),
            )

        gpu_preprocessing_requested = _gpu_preprocessing_enabled(resolved_device)
        use_gpu_preprocessing = gpu_preprocessing_requested
        prediction_model_path = model_path
        gpu_preprocessing_fallback_reason = ""
        if use_gpu_preprocessing:
            try:
                prediction_model_path = _prepare_nnunet_gpu_model_folder(
                    model_path,
                    tmp_root / "model_gpu_preprocessing",
                    folds,
                    checkpoint_name,
                )
            except Exception as exc:
                use_gpu_preprocessing = False
                gpu_preprocessing_fallback_reason = f"{type(exc).__name__}: {exc}"
                _emit_progress(
                    progress_callback,
                    stage="autoseg_gpu_preprocessing_fallback",
                    message="Could not prepare CUDA input resampling; using CPU resampling...",
                    current=3,
                    total=total_stages,
                    elapsed_sec=time.perf_counter() - t_total_start,
                    preprocessing_device="cpu",
                    fallback_reason=gpu_preprocessing_fallback_reason,
                )
        command = _nnunet_inference_command(
            input_dir,
            output_dir,
            prediction_model_path,
            folds,
            checkpoint_name,
            resolved_device,
            effective_num_processes_preprocessing,
            num_processes_segmentation_export,
            step_size,
            disable_tta,
        )
        preprocessing_device = "cuda" if use_gpu_preprocessing else "cpu"
        gpu_preprocessing_command = list(command) if use_gpu_preprocessing else []
        _emit_progress(
            progress_callback,
            stage="autoseg_run_inference",
            message=(
                "Running nnUNet inference with CUDA input resampling..."
                if use_gpu_preprocessing
                else f"Running nnUNet inference on {resolved_device}..."
            ),
            current=3,
            total=total_stages,
            elapsed_sec=time.perf_counter() - t_total_start,
            command=[str(x) for x in command],
            preprocessing_device=preprocessing_device,
        )
        result = _run_subprocess(command, runner=runner, progress_callback=progress_callback,
                                 output_dir=output_dir, progress_total=total_stages)
        if getattr(result, "returncode", 0) != 0 and use_gpu_preprocessing:
            gpu_preprocessing_fallback_reason = _subprocess_failure_details(result)
            output_dir = tmp_root / "output_cpu_preprocessing"
            output_dir.mkdir(parents=True, exist_ok=True)
            command = _nnunet_inference_command(
                input_dir,
                output_dir,
                model_path,
                folds,
                checkpoint_name,
                resolved_device,
                effective_num_processes_preprocessing,
                num_processes_segmentation_export,
                step_size,
                disable_tta,
            )
            preprocessing_device = "cpu"
            _emit_progress(
                progress_callback,
                stage="autoseg_gpu_preprocessing_fallback",
                message="CUDA input resampling failed; retrying with CPU resampling...",
                current=3,
                total=total_stages,
                elapsed_sec=time.perf_counter() - t_total_start,
                command=[str(x) for x in command],
                preprocessing_device=preprocessing_device,
                fallback_reason=gpu_preprocessing_fallback_reason,
            )
            result = _run_subprocess(command, runner=runner, progress_callback=progress_callback,
                                     output_dir=output_dir, progress_total=total_stages)
        if getattr(result, "returncode", 0) != 0:
            stdout = getattr(result, "stdout", "") or ""
            stderr = getattr(result, "stderr", "") or ""
            gpu_failure = ""
            if gpu_preprocessing_fallback_reason:
                gpu_failure = (
                    "\nCUDA preprocessing attempt failed before the CPU retry:\n"
                    f"{gpu_preprocessing_fallback_reason}\n"
                )
            raise RuntimeError(
                "nnUNet inference failed\n"
                f"command: {' '.join(map(str, command))}\n"
                f"{gpu_failure}"
                f"stdout:\n{stdout}\n"
                f"stderr:\n{stderr}"
            )


        prediction_path = output_dir / f"{case_id}{file_ending}"
        if not prediction_path.is_file() and file_ending == ".nii.gz":
            prediction_path = output_dir / f"{case_id}.nii.gz"
        if not prediction_path.is_file():
            raise FileNotFoundError(f"nnUNet prediction not found: {prediction_path}")

        _emit_progress(
            progress_callback,
            stage="autoseg_read_prediction",
            message="Reading nnUNet prediction...",
            current=4,
            total=total_stages,
            elapsed_sec=time.perf_counter() - t_total_start,
            prediction_file=str(prediction_path),
        )
        seg_3d_nnunet = _read_nifti_segmentation(prediction_path)
        seg_3d_nnunet = _apply_label_map(seg_3d_nnunet, label_map)
        if artifact_prediction_path is not None:
            _write_nifti_segmentation(seg_3d_nnunet, affine, artifact_prediction_path)
        seg_3d = _restore_autoflow_segmentation(seg_3d_nnunet)
        seg_4d = broadcast_segmentation_to_time(seg_3d, time_count)
        elapsed_total = time.perf_counter() - t_total_start
        _emit_progress(
            progress_callback,
            stage="autoseg_finalize",
            message="Finalizing auto segmentation volume...",
            current=5,
            total=total_stages,
            elapsed_sec=elapsed_total,
            prediction_file=str(prediction_path),
        )
        provenance = {
            "source": "auto",
            "backend": "nnUNet",
            "model_folder": str(model_path),
            "checkpoint": str(checkpoint_name),
            "device": str(resolved_device),
            "folds": list(folds),
            "channel_names": list(channel_names),
            "nnunet_spatial_order": list(_NNUNET_TARGET_SPATIAL_ORDER),
            "nnunet_venc_order": list(_NNUNET_TARGET_VENC_ORDER),
            "autoflow_internal_spatial_order": list(_AUTOFLOW_INTERNAL_SPATIAL_ORDER),
            "autoflow_internal_venc_order": list(_AUTOFLOW_INTERNAL_VENC_ORDER),
            "label_map": {str(k): int(v) for k, v in label_map.items()},
            "case_id": str(case_id),
            "created_at": segmentation_timestamp(),
            "command": [str(x) for x in command],
            "gpu_preprocessing_command": [str(x) for x in gpu_preprocessing_command],
            "preprocessing_device_requested": "cuda" if gpu_preprocessing_requested else "cpu",
            "preprocessing_device": str(preprocessing_device),
            "preprocessing_resampler": (
                "resample_torch_fornnunet"
                if preprocessing_device == "cuda"
                else "resample_data_or_seg_to_shape"
            ),
            "gpu_preprocessing_fallback": bool(gpu_preprocessing_fallback_reason),
            "gpu_preprocessing_fallback_reason": str(gpu_preprocessing_fallback_reason),
            "feature_files": [str(path) for path in artifact_feature_paths],
            "segmentation_nifti": "" if artifact_prediction_path is None else str(artifact_prediction_path),
            "prediction_file": "" if artifact_prediction_path is None else str(artifact_prediction_path),
            "elapsed_sec": float(elapsed_total),
        }
        return seg_4d, provenance
