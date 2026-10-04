"""Phase-resolved nnUNet inference and segmentation reconstruction."""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from ...task_control import TaskCancelled

from ._common import segmentation_timestamp
from .channels import (
    _apply_label_map,
    _ensure_nnunet_mag_flow,
    _nnunet_4d_channel_volumes,
    _nnunet_channel_volumes,
    _nnunet_normalize_channel_name,
    _nnunet_spatial_affine,
    _ordered_mapping_values,
    _parse_nnunet_label_map,
    _parse_nnunet_temporal_channel,
    _prepare_nnunet_inputs,
    _restore_autoflow_segmentation,
)
from .io import (
    _link_or_copy_file,
    _read_nifti_segmentation,
    _sanitize_nnunet_artifact_token,
    _write_nifti_segmentation,
    _write_nifti_volume,
)
from .models import (
    _NNUNET_4D_CHECKPOINT_DEFAULT,
    _NNUNET_4D_PIPELINE_DEFAULT,
    _load_nnunet_model_metadata,
    _nnunet_4d_subprocess_env,
    _python_from_4d_pipeline_script,
    _resolve_nnunet_checkpoint,
    _resolve_nnunet_folds,
    resolve_nnunet_4d_model_folder,
)
from .nnunet_grouped import _generate_nnunet_4d_grouped, _nnunet_4d_grouped_requested
from .runtime import (
    _emit_progress,
    _nnunet_inference_command,
    _run_subprocess,
    resolve_auto_segmentation_device,
)


def generate_nnunet_4d_auto_segmentation(
    mag,
    flow,
    resolution,
    origin,
    model_folder="",
    *,
    checkpoint_name="checkpoint_final.pth",
    folds="single",
    device="auto",
    auto_label_map="",
    case_id="autoflow_case",
    num_processes_preprocessing=0,
    num_processes_segmentation_export=1,
    artifact_prefix="",
    runner=None,
    progress_callback=None,
    grouped_preprocessing=None,
):
    """Run the Dataset7020 temporal model for every frame in one invocation.

    The Dataset7020 checkpoint is still a regular nnUNet model; its temporal
    context is encoded in the channel names (``tm2_*`` through ``tp2_*``).
    AutoFlow therefore writes one sample per frame, invokes nnUNet once, then
    stacks the frame predictions back into an ``XYZT`` segmentation.  ``folds``
    accepts ``single`` (``fold_all`` or the first available fold), ``all`` for
    an ensemble, or a comma-separated explicit list.
    """
    t_total_start = time.perf_counter()
    total_stages = 5
    _emit_progress(
        progress_callback,
        stage="autoseg_start",
        message="Resolving 4D nnUNet model and input metadata...",
        current=0,
        total=total_stages,
        elapsed_sec=0.0,
    )
    pipeline_script = ""
    if str(model_folder or "").lower().endswith((".sh", ".bash")):
        pipeline_script = str(Path(model_folder).expanduser().resolve())
    model_path = Path(resolve_nnunet_4d_model_folder(model_folder))
    resolved_device = resolve_auto_segmentation_device(device)
    mag, flow = _ensure_nnunet_mag_flow(mag, flow)
    time_count = int(flow.shape[3])
    mag_nnunet, flow_nnunet, resolution_nnunet = _prepare_nnunet_inputs(mag, flow, resolution)
    model_path, dataset_json = _load_nnunet_model_metadata(model_path)
    channel_names = _ordered_mapping_values(dataset_json.get("channel_names") or dataset_json.get("modality") or {})
    if not channel_names:
        raise ValueError(f"4D model folder does not define any channel names: {model_path}")
    if not any(_parse_nnunet_temporal_channel(name) for name in channel_names):
        raise ValueError(
            f"4D nnUNet model has no temporal channels: {model_path / 'dataset.json'}"
        )
    file_ending = str(dataset_json.get("file_ending", ".nii.gz"))
    if not file_ending.startswith("."):
        file_ending = f".{file_ending}"
    label_map = _parse_nnunet_label_map(auto_label_map, dataset_json.get("labels", {}))
    selected_folds = _resolve_nnunet_folds(model_path, folds)
    checkpoint_name = _resolve_nnunet_checkpoint(
        model_path,
        checkpoint_name,
        selected_folds,
        default_checkpoint=_NNUNET_4D_CHECKPOINT_DEFAULT,
    )
    affine = _nnunet_spatial_affine(resolution_nnunet, mag_nnunet.shape[:3])
    _emit_progress(
        progress_callback,
        stage="autoseg_model_ready",
        message=f"Resolved 4D nnUNet model: {model_path.name} | folds={','.join(selected_folds)} | device={resolved_device}",
        current=1,
        total=total_stages,
        elapsed_sec=time.perf_counter() - t_total_start,
        backend="nnUNet4D",
        device=str(resolved_device),
        checkpoint=str(checkpoint_name),
        folds=list(selected_folds),
        model_folder=str(model_path),
    )

    safe_case_id = _sanitize_nnunet_artifact_token(case_id, default="autoflow_case")
    grouped_script = pipeline_script
    grouped_fallback_reason = ""
    if not grouped_script and _NNUNET_4D_PIPELINE_DEFAULT.is_file():
        grouped_script = str(_NNUNET_4D_PIPELINE_DEFAULT)
        # Use the default script for subprocess bootstrap as well, not only for
        # grouped preprocessing. This keeps custom resampling available when a
        # caller leaves the model setting empty.
        pipeline_script = grouped_script
    if _nnunet_4d_grouped_requested(runner, grouped_preprocessing) and grouped_script:
        try:
            return _generate_nnunet_4d_grouped(
                mag_nnunet,
                flow_nnunet,
                resolution_nnunet,
                model_path,
                dataset_json,
                selected_folds,
                checkpoint_name,
                resolved_device,
                label_map,
                safe_case_id,
                affine,
                grouped_script,
                num_processes_segmentation_export,
                artifact_prefix,
                progress_callback,
                t_total_start,
            )
        except TaskCancelled:
            raise
        except Exception as exc:
            grouped_fallback_reason = f"{type(exc).__name__}: {exc}"
            token = str(os.environ.get("AUTOFLOW_NNUNET4D_GROUPED", "auto") or "auto").strip().lower()
            if token in {"1", "true", "yes", "on", "required"}:
                raise
            _emit_progress(
                progress_callback,
                stage="autoseg_grouped_fallback",
                message="Grouped 4D preprocessing failed; retrying with standard nnUNet input files...",
                current=2,
                total=total_stages,
                elapsed_sec=time.perf_counter() - t_total_start,
                grouped_preprocessing=False,
                fallback_reason=grouped_fallback_reason,
            )

    with TemporaryDirectory(prefix="autoflow_nnunet4d_") as tmp_root:
        tmp_root = Path(tmp_root)
        input_dir = tmp_root / "input"
        output_dir = tmp_root / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        global_names = [
            "mag_std_xyz", "mag_mean_xyz", "pcmra_std_xyz", "pcmra_mean_xyz",
            "flow_x_mean_xyz", "flow_y_mean_xyz", "flow_z_mean_xyz", "flow_mag_mean_xyz",
            "flow_x_std_xyz", "flow_y_std_xyz", "flow_z_std_xyz", "flow_mag_std_xyz",
        ]
        temporal_speed = np.linalg.norm(flow_nnunet, axis=-1).astype(np.float32, copy=False)
        temporal_cache = {
            "mag": mag_nnunet,
            "pcmra": (mag_nnunet * temporal_speed).astype(np.float32, copy=False),
            "flow_x": flow_nnunet[..., 0],
            "flow_y": flow_nnunet[..., 1],
            "flow_z": flow_nnunet[..., 2],
        }
        global_by_name = dict(zip(global_names, _nnunet_channel_volumes(
            global_names, mag_nnunet, flow_nnunet, speed=temporal_speed, pcmra=temporal_cache["pcmra"]
        )))
        _emit_progress(
            progress_callback,
            stage="autoseg_prepare_inputs",
            message=f"Preparing {time_count} temporal nnUNet samples ({len(channel_names)} channels each)...",
            current=2,
            total=total_stages,
            elapsed_sec=time.perf_counter() - t_total_start,
            detail_current=0,
            detail_total=time_count,
        )

        # Temporal neighbours and cycle statistics recur across samples. Encode
        # each feature map once, then retain the standard per-frame channel
        # filenames through hard links (or byte copies on other filesystems).
        # nnUNet still performs its original, independent sample preprocessing.
        source_jobs = {}
        channel_links = []
        for frame_index in range(time_count):
            volumes = _nnunet_4d_channel_volumes(
                channel_names,
                mag_nnunet,
                flow_nnunet,
                frame_index,
                global_by_name=global_by_name,
                temporal_cache=temporal_cache,
            )
            frame_id = f"{safe_case_id}_t{int(frame_index):03d}"
            for channel_index, (raw_name, volume) in enumerate(zip(channel_names, volumes)):
                name = _nnunet_normalize_channel_name(raw_name)
                temporal = _parse_nnunet_temporal_channel(name)
                if name in global_by_name:
                    source_key = ("global", name)
                elif temporal is not None:
                    offset, feature = temporal
                    source_key = ("temporal", feature, (frame_index + offset) % time_count)
                else:
                    source_key = ("global", global_names[channel_index])
                path = input_dir / f"{frame_id}_{channel_index:04d}{file_ending}"
                if source_key not in source_jobs:
                    source_jobs[source_key] = (path, volume)
                else:
                    channel_links.append((source_jobs[source_key][0], path))

        def _write_source(job):
            path, volume = job
            _write_nifti_volume(volume, affine, path)

        # NIfTI encoding is independent per feature and releases the GIL. A
        # small pool avoids serial gzip overhead without competing with nnUNet.
        worker_count = min(4, max(1, time_count))
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="autoflow-nnunet4d-nifti") as executor:
            list(executor.map(_write_source, source_jobs.values()))
        for source, destination in channel_links:
            _link_or_copy_file(source, destination)
        for frame_index in range(time_count):
            _emit_progress(
                progress_callback,
                stage="autoseg_prepare_inputs",
                message=f"Writing 4D frame {frame_index + 1}/{time_count}",
                current=2,
                total=total_stages,
                elapsed_sec=time.perf_counter() - t_total_start,
                detail_current=frame_index + 1,
                detail_total=time_count,
            )

        command = _nnunet_inference_command(
            input_dir,
            output_dir,
            model_path,
            selected_folds,
            checkpoint_name,
            resolved_device,
            int(num_processes_preprocessing or (1 if resolved_device == "cuda" else 3)),
            num_processes_segmentation_export,
            0.5,
            True,
            python_executable=_python_from_4d_pipeline_script(pipeline_script),
            bootstrap_resampler_path=(
                Path(pipeline_script).expanduser().resolve().parent.parent.parent
                / "nnunet" / "gpu_resampling.py"
                if pipeline_script
                else None
            ),
            optimize_transfers=True,
        )
        _emit_progress(
            progress_callback,
            stage="autoseg_run_inference",
            message=f"Running 4D nnUNet inference on {resolved_device}...",
            current=3,
            total=total_stages,
            elapsed_sec=time.perf_counter() - t_total_start,
            command=[str(x) for x in command],
            preprocessing_device="model",
        )
        subprocess_env = _nnunet_4d_subprocess_env(pipeline_script, model_path)
        result = _run_subprocess(command, env=subprocess_env, runner=runner, progress_callback=progress_callback,
                                 output_dir=output_dir, expected_predictions=time_count, progress_total=total_stages)
        if getattr(result, "returncode", 0) != 0:
            raise RuntimeError(
                "4D nnUNet inference failed\n"
                f"command: {' '.join(map(str, command))}\n"
                f"stdout:\n{getattr(result, 'stdout', '') or ''}\n"
                f"stderr:\n{getattr(result, 'stderr', '') or ''}"
            )

        _emit_progress(
            progress_callback,
            stage="autoseg_read_prediction",
            message="Reading 4D nnUNet predictions...",
            current=4,
            total=total_stages,
            elapsed_sec=time.perf_counter() - t_total_start,
        )
        predictions = []
        for frame_index in range(time_count):
            prediction_path = output_dir / f"{safe_case_id}_t{frame_index:03d}{file_ending}"
            if not prediction_path.is_file() and file_ending == ".nii.gz":
                prediction_path = output_dir / f"{safe_case_id}_t{frame_index:03d}.nii.gz"
            if not prediction_path.is_file():
                raise FileNotFoundError(f"4D nnUNet prediction not found: {prediction_path}")
            predictions.append(_read_nifti_segmentation(prediction_path))
        shape = tuple(np.asarray(predictions[0]).shape)
        if any(tuple(np.asarray(item).shape) != shape for item in predictions):
            raise ValueError("4D nnUNet predictions have inconsistent spatial shapes")
        seg_nnunet = np.stack(predictions, axis=3)
        seg_nnunet = _apply_label_map(seg_nnunet, label_map)
        if artifact_prefix:
            artifact_prediction = Path(f"{artifact_prefix}.nii.gz")
            artifact_prediction.parent.mkdir(parents=True, exist_ok=True)
            _write_nifti_segmentation(seg_nnunet, affine, artifact_prediction)
        else:
            artifact_prediction = None
        seg_4d = _restore_autoflow_segmentation(seg_nnunet)
        elapsed_total = time.perf_counter() - t_total_start
        _emit_progress(
            progress_callback,
            stage="autoseg_finalize",
            message="Finalizing 4D auto segmentation volume...",
            current=5,
            total=total_stages,
            elapsed_sec=elapsed_total,
        )
        provenance = {
            "source": "auto",
            "backend": "nnUNet4D",
            "model_folder": str(model_path),
            "pipeline_script": pipeline_script,
            "python_executable": str(_python_from_4d_pipeline_script(pipeline_script) or sys.executable),
            "checkpoint": str(checkpoint_name),
            "device": str(resolved_device),
            "folds": list(selected_folds),
            "fold_mode": str(folds),
            "channel_names": list(channel_names),
            "time_count": int(time_count),
            "label_map": {str(k): int(v) for k, v in label_map.items()},
            "case_id": str(case_id),
            "created_at": segmentation_timestamp(),
            "command": [str(x) for x in command],
            "preprocessing_device": "model",
            "grouped_preprocessing": False,
            "grouped_fallback_reason": grouped_fallback_reason,
            "feature_files": [],
            "segmentation_nifti": "" if artifact_prediction is None else str(artifact_prediction),
            "prediction_file": "" if artifact_prediction is None else str(artifact_prediction),
            "elapsed_sec": float(elapsed_total),
        }
        return np.asarray(seg_4d, dtype=np.int16), provenance
