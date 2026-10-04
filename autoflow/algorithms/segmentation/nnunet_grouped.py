"""Optional grouped Dataset7020 preprocessing and inference adapter."""

import hashlib
import importlib.util
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np

from ._common import segmentation_timestamp
from .channels import (
    _apply_label_map,
    _nnunet_channel_volumes,
    _ordered_mapping_values,
    _parse_nnunet_temporal_channel,
    _restore_autoflow_segmentation,
)
from .io import (
    _read_nifti_segmentation,
    _write_nifti_segmentation,
    _write_nifti_volume,
)
from .models import _python_from_4d_pipeline_script
from .runtime import _emit_progress


def _load_4d_pipeline_module(script_path):
    """Load the external grouped-preprocessing helpers when available."""
    path = Path(script_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"4D pipeline script not found: {path}")
    # ``run_seg2nndata_all.py`` discovers its project root from the
    # ``scripts/`` directory (which contains both ``methods/`` and
    # ``nnunet/``).  The 4D runner lives below ``scripts/nnunet/4D``.
    scripts_root = path.parent.parent.parent
    for root in (scripts_root, path.parent.parent, path.parent):
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    module_path = path.parent / "run_4dflow.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"grouped 4D runner not found: {module_path}")
    module_name = "autoflow_external_4dflow_" + hashlib.sha1(str(module_path).encode("utf-8")).hexdigest()[:12]
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load grouped 4D runner: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    # run_seg2nndata_all.py discovers its project root from cwd at import time.
    # Import it from the Dataset7020 scripts root, then restore AutoFlow's cwd.
    old_cwd = os.getcwd()
    try:
        os.chdir(scripts_root)
        spec.loader.exec_module(module)
    finally:
        os.chdir(old_cwd)
    return module


def _install_nnunet_4d_resampler(script_path):
    """Install the Dataset7020 GPU resampler into the active nnUNet package."""
    path = Path(script_path).expanduser().resolve()
    scripts_root = path.parent.parent.parent
    source_file = scripts_root / "nnunet" / "gpu_resampling.py"
    if not source_file.is_file():
        raise FileNotFoundError(f"Dataset7020 GPU resampler not found: {source_file}")
    import nnunetv2

    target_dir = Path(nnunetv2.__file__).resolve().parent / "preprocessing" / "resampling"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / "gpu_resampling.py"
    if not target_file.exists() or target_file.read_bytes() != source_file.read_bytes():
        shutil.copy2(source_file, target_file)
    importlib.invalidate_caches()
    return target_file


def _nnunet_4d_temporal_radius(channel_names):
    offsets = [parsed[0] for name in channel_names if (parsed := _parse_nnunet_temporal_channel(name))]
    if not offsets:
        raise ValueError("4D nnUNet model does not define temporal channels")
    radius = max(abs(int(value)) for value in offsets)
    expected = 12 + 5 * (2 * radius + 1)
    if len(channel_names) != expected:
        raise ValueError(
            f"4D nnUNet channel layout has {len(channel_names)} channels; "
            f"expected {expected} for temporal radius {radius}"
        )
    return radius


def _nnunet_4d_grouped_requested(runner=None, grouped_preprocessing=None):
    """Return whether the direct grouped predictor should be attempted.

    The grouped implementation keeps a torch predictor in the caller process.
    The GUI passes ``grouped_preprocessing=False`` so native CUDA/Qt failures
    stay inside the nnUNet subprocess.  Other callers retain the environment
    controlled behavior for backwards compatibility.
    """
    if grouped_preprocessing is not None:
        return bool(grouped_preprocessing) and runner is None
    token = str(os.environ.get("AUTOFLOW_NNUNET4D_GROUPED", "auto") or "auto").strip().lower()
    if token in {"0", "false", "no", "off", "standard", "subprocess"}:
        return False
    # A test/custom runner intentionally exercises the subprocess contract.
    return runner is None


def _generate_nnunet_4d_grouped(
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
    pipeline_script,
    num_processes_segmentation_export,
    artifact_prefix,
    progress_callback,
    t_total_start,
):
    """Predict all temporal samples with one common crop and shared resampling."""
    pipeline = _load_4d_pipeline_module(pipeline_script)
    # Dataset7020 plans may reference the project GPU resampler by name. The
    # grouped runner already owns the installer used by its batch workflow;
    # install it before PlansManager resolves the resampling function.
    if str(resolved_device).strip().lower() == "cuda":
        _install_nnunet_4d_resampler(pipeline_script)
    channel_names = _ordered_mapping_values(dataset_json.get("channel_names") or dataset_json.get("modality") or {})
    radius = _nnunet_4d_temporal_radius(channel_names)
    nt = int(mag_nnunet.shape[3])
    if nt < 1:
        raise ValueError("4D nnUNet input has no time frames")

    global_names = [
        "mag_std_xyz", "mag_mean_xyz", "pcmra_std_xyz", "pcmra_mean_xyz",
        "flow_x_mean_xyz", "flow_y_mean_xyz", "flow_z_mean_xyz", "flow_mag_mean_xyz",
        "flow_x_std_xyz", "flow_y_std_xyz", "flow_z_std_xyz", "flow_mag_std_xyz",
    ]
    speed = np.linalg.norm(flow_nnunet, axis=-1).astype(np.float32, copy=False)
    pcmra = (mag_nnunet * speed).astype(np.float32, copy=False)
    global_values = _nnunet_channel_volumes(global_names, mag_nnunet, flow_nnunet, speed=speed, pcmra=pcmra)
    global_by_name = dict(zip(global_names, global_values))

    with TemporaryDirectory(prefix="autoflow_nnunet4d_grouped_") as tmp_root:
        tmp_root = Path(tmp_root)
        input_dir = tmp_root / "input"
        output_dir = tmp_root / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        source_paths = {}
        source_volumes = {}
        samples = []
        for frame in range(nt):
            sample_id = f"{safe_case_id}__t{frame:03d}"
            samples.append({"sample_id": sample_id, "frame": frame, "label": None})
            for source_key in pipeline.temporal_source_keys(frame, nt, radius):
                source_paths.setdefault(source_key, None)

        temporal_features = (
            flow_nnunet[..., 0], flow_nnunet[..., 1], flow_nnunet[..., 2],
            mag_nnunet, pcmra,
        )
        for source_key in source_paths:
            kind, feature_index, frame = source_key
            if kind == "global":
                source_volumes[source_key] = global_values[int(feature_index)]
            else:
                source_volumes[source_key] = temporal_features[int(feature_index)][..., int(frame)]

        def _write_source(item):
            index, (source_key, volume) = item
            kind, feature_index, frame = source_key
            frame_token = "global" if kind == "global" else f"t{int(frame):03d}"
            path = input_dir / f"source_{kind}_{int(feature_index):02d}_{frame_token}_{index:03d}.nii.gz"
            _write_nifti_volume(volume, affine, path)
            return source_key, path

        _emit_progress(
            progress_callback,
            stage="autoseg_prepare_inputs",
            message=(
                f"Preparing grouped 4D inputs ({len(source_paths)} unique maps, "
                f"{nt} temporal samples)..."
            ),
            current=2,
            total=5,
            elapsed_sec=time.perf_counter() - t_total_start,
            detail_current=0,
            detail_total=len(source_paths),
            grouped_preprocessing=True,
            temporal_radius=radius,
        )
        items = list(zip(source_paths.keys(), source_volumes.values()))
        worker_count = min(4, max(1, len(items)))
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="autoflow-nnunet4d-source") as executor:
            written = list(executor.map(_write_source, enumerate(items)))
        for source_key, path in written:
            source_paths[source_key] = path

        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        torch_device = torch.device(str(resolved_device))
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False,
            perform_everything_on_device=torch_device.type == "cuda",
            device=torch_device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        predictor.initialize_from_trained_model_folder(
            str(model_path), use_folds=tuple(selected_folds), checkpoint_name=checkpoint_name
        )
        if torch_device.type != "cuda":
            # Dataset7020's persisted plan points at a project CUDA resampler.
            # Keep CPU inference usable by changing only this in-memory plan;
            # the source model metadata remains untouched.  The grouped path
            # invokes this function directly, so it cannot rely on nnUNet's
            # subprocess fallback to make the same adjustment.
            configuration = getattr(predictor.configuration_manager, "configuration", None)
            if isinstance(configuration, dict):
                configuration["resampling_fn_data"] = "resample_data_or_seg_to_shape"
                configuration["resampling_fn_data_kwargs"] = {
                    "is_seg": False,
                    "order": 3,
                    "order_z": 0,
                    "force_separate_z": None,
                }
                if "resampling_fn_probabilities" in configuration:
                    configuration["resampling_fn_probabilities"] = "resample_data_or_seg_to_shape"
                    configuration["resampling_fn_probabilities_kwargs"] = {
                        "is_seg": False,
                        "order": 3,
                        "order_z": 0,
                        "force_separate_z": None,
                    }
        _emit_progress(
            progress_callback,
            stage="autoseg_run_inference",
            message=f"Running grouped 4D nnUNet inference on {resolved_device}...",
            current=3,
            total=5,
            elapsed_sec=time.perf_counter() - t_total_start,
            grouped_preprocessing=True,
            temporal_radius=radius,
            unique_source_maps=len(source_paths),
        )
        raw_iterator = pipeline.iter_grouped_temporal_preprocessed_samples_from_source_paths(
            samples,
            source_paths,
            predictor.plans_manager,
            predictor.configuration_manager,
            predictor.dataset_json,
            radius,
        )
        import torch

        def _predictor_iterator():
            for sample, data, segmentation, properties in raw_iterator:
                if segmentation is not None:
                    raise RuntimeError(f"{sample['sample_id']}: prediction input unexpectedly has a segmentation")
                yield {
                    "data": torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32)),
                    "data_properties": properties,
                    "ofile": str(output_dir / str(sample["sample_id"])),
                }

        predictor.predict_from_data_iterator(
            _predictor_iterator(),
            save_probabilities=False,
            num_processes_segmentation_export=max(1, int(num_processes_segmentation_export or 1)),
        )

        _emit_progress(
            progress_callback,
            stage="autoseg_read_prediction",
            message="Reading grouped 4D nnUNet predictions...",
            current=4,
            total=5,
            elapsed_sec=time.perf_counter() - t_total_start,
            grouped_preprocessing=True,
        )
        predictions = []
        for frame in range(nt):
            prediction_path = output_dir / f"{safe_case_id}__t{frame:03d}.nii.gz"
            if not prediction_path.is_file():
                raise FileNotFoundError(f"grouped 4D nnUNet prediction not found: {prediction_path}")
            predictions.append(_read_nifti_segmentation(prediction_path))
        seg_nnunet = _apply_label_map(np.stack(predictions, axis=3), label_map)
        artifact_prediction = None
        if artifact_prefix:
            artifact_prediction = Path(f"{artifact_prefix}.nii.gz")
            artifact_prediction.parent.mkdir(parents=True, exist_ok=True)
            _write_nifti_segmentation(seg_nnunet, affine, artifact_prediction)
        seg_4d = _restore_autoflow_segmentation(seg_nnunet)
        elapsed_total = time.perf_counter() - t_total_start
        _emit_progress(
            progress_callback,
            stage="autoseg_finalize",
            message="Finalizing grouped 4D auto segmentation volume...",
            current=5,
            total=5,
            elapsed_sec=elapsed_total,
            grouped_preprocessing=True,
        )
        provenance = {
            "source": "auto",
            "backend": "nnUNet4D",
            "model_folder": str(model_path),
            "pipeline_script": str(pipeline_script or ""),
            "python_executable": str(_python_from_4d_pipeline_script(pipeline_script) or sys.executable),
            "checkpoint": str(checkpoint_name),
            "device": str(resolved_device),
            "folds": list(selected_folds),
            "fold_mode": "grouped",
            "channel_names": list(channel_names),
            "time_count": nt,
            "temporal_radius": radius,
            "grouped_preprocessing": True,
            "unique_source_maps": len(source_paths),
            "label_map": {str(k): int(v) for k, v in label_map.items()},
            "case_id": str(safe_case_id),
            "created_at": segmentation_timestamp(),
            "command": [],
            "preprocessing_device": "model",
            "feature_files": [],
            "segmentation_nifti": "" if artifact_prediction is None else str(artifact_prediction),
            "prediction_file": "" if artifact_prediction is None else str(artifact_prediction),
            "elapsed_sec": float(elapsed_total),
        }
        return np.asarray(seg_4d, dtype=np.int16), provenance
