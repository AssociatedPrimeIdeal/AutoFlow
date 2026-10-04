"""nnUNet subprocesses, progress, device selection and GPU preprocessing."""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryFile
from ...task_control import check_cancelled, stop_process

from .models import _NNUNET_GPU_PREPROCESSING_ENV


def _gpu_preprocessing_enabled(device, environ=None):
    environment = os.environ if environ is None else environ
    token = str(environment.get(_NNUNET_GPU_PREPROCESSING_ENV, "cuda") or "cuda").strip().lower()
    if token in {"0", "false", "no", "off", "cpu", "disabled"}:
        return False
    return str(device or "").strip().lower() == "cuda"


def _link_nnunet_model_file(source, destination):
    source = Path(source).resolve()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(source, destination)
        return
    except OSError:
        pass
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _prepare_nnunet_gpu_model_folder(model_path, target_path, folds, checkpoint_name):
    model_path = Path(model_path).resolve()
    target_path = Path(target_path)
    target_path.mkdir(parents=True, exist_ok=True)

    with (model_path / "plans.json").open("r", encoding="utf-8") as handle:
        plans = json.load(handle)
    configurations = plans.get("configurations") or {}
    changed = 0
    for configuration in configurations.values():
        if not isinstance(configuration, dict) or "resampling_fn_data" not in configuration:
            continue
        original_kwargs = dict(configuration.get("resampling_fn_data_kwargs") or {})
        gpu_kwargs = {
            "is_seg": False,
            "num_threads": max(1, int(original_kwargs.get("num_threads", 4) or 4)),
            "device": "cuda",
            "memefficient_seg_resampling": False,
            "force_separate_z": original_kwargs.get("force_separate_z"),
            "mode": "linear",
            "aniso_axis_mode": "nearest-exact",
        }
        if "separate_z_anisotropy_threshold" in original_kwargs:
            gpu_kwargs["separate_z_anisotropy_threshold"] = original_kwargs[
                "separate_z_anisotropy_threshold"
            ]
        configuration["resampling_fn_data"] = "resample_torch_fornnunet"
        configuration["resampling_fn_data_kwargs"] = gpu_kwargs
        if "resampling_fn_probabilities" in configuration:
            probability_kwargs = dict(gpu_kwargs)
            probability_kwargs.update({"is_seg": False, "mode": "linear"})
            configuration["resampling_fn_probabilities"] = "resample_torch_fornnunet"
            configuration["resampling_fn_probabilities_kwargs"] = probability_kwargs
        changed += 1
    if changed == 0:
        raise ValueError(f"nnUNet plans do not define an input resampler: {model_path / 'plans.json'}")
    plans["autoflow_gpu_input_resampling"] = {
        "enabled": True,
        "function": "resample_torch_fornnunet",
        "probabilities_function": "resample_torch_fornnunet",
        "device": "cuda",
        "mode": "linear",
    }
    with (target_path / "plans.json").open("w", encoding="utf-8") as handle:
        json.dump(plans, handle, indent=2)

    _link_nnunet_model_file(model_path / "dataset.json", target_path / "dataset.json")
    for fold in folds:
        fold_name = f"fold_{fold}"
        source_checkpoint = model_path / fold_name / checkpoint_name
        if not source_checkpoint.is_file():
            raise FileNotFoundError(f"missing nnUNet checkpoint: {source_checkpoint}")
        _link_nnunet_model_file(
            source_checkpoint,
            target_path / fold_name / checkpoint_name,
        )
    return target_path


def _run_subprocess(command, *, env=None, cwd=None, runner=None, progress_callback=None,
                    output_dir=None, expected_predictions=1, progress_total=5):
    check_cancelled()
    if runner is not None:
        return runner(command, check=False, capture_output=True, text=True, env=env, cwd=cwd)
    started = time.perf_counter()
    options = {"start_new_session": True} if os.name != "nt" else {
        "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP
    }
    with TemporaryFile(mode="w+b") as stdout, TemporaryFile(mode="w+b") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env, cwd=cwd, **options)
        try:
            next_update = 0.0
            while process.poll() is None:
                check_cancelled()
                now = time.perf_counter()
                if progress_callback is not None and now >= next_update:
                    completed = sum(
                        path.name.endswith((".nii", ".nii.gz"))
                        for path in Path(output_dir).iterdir()
                    ) if output_dir is not None else 0
                    _emit_progress(
                        progress_callback, stage="autoseg_inference_progress", current=3, total=progress_total,
                        message=f"nnUNet inference active — prediction files observed {completed}/{expected_predictions}",
                        detail_current=completed, detail_total=expected_predictions,
                        detail_message=f"Exported predictions: {completed} / {expected_predictions}",
                        elapsed_sec=now - started,
                    )
                    next_update = now + 0.5
                time.sleep(0.1)
            check_cancelled()
        except BaseException:
            stop_process(process)
            raise
        stdout.seek(0)
        stderr.seek(0)
        return subprocess.CompletedProcess(command, process.returncode,
            stdout.read().decode("utf-8", errors="replace"), stderr.read().decode("utf-8", errors="replace"))


def _subprocess_failure_details(result):
    stdout = str(getattr(result, "stdout", "") or "").strip()
    stderr = str(getattr(result, "stderr", "") or "").strip()
    combined = "\n".join(part for part in (stdout, stderr) if part)
    return combined[-2000:] if combined else f"exit code {getattr(result, 'returncode', 'unknown')}"


def _nnunet_inference_command(
    input_dir,
    output_dir,
    model_path,
    folds,
    checkpoint_name,
    resolved_device,
    num_processes_preprocessing,
    num_processes_segmentation_export,
    step_size,
    disable_tta,
    python_executable=None,
    bootstrap_resampler_path=None,
    optimize_transfers=False,
):
    command = [
        *_nnunet_predict_command(
            python_executable=python_executable,
            bootstrap_resampler_path=bootstrap_resampler_path,
            optimize_transfers=optimize_transfers,
        ),
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-m", str(model_path),
        "-f", *folds,
        "-chk", str(checkpoint_name),
        "-npp", str(int(num_processes_preprocessing)),
        "-nps", str(int(num_processes_segmentation_export)),
        "-device", str(resolved_device),
    ]
    if step_size is not None:
        command.extend(["-step_size", str(float(step_size))])
    if disable_tta:
        command.append("--disable_tta")
    return command


def _emit_progress(progress_callback, *, stage, message, current=None, total=None, elapsed_sec=None, **extra):
    check_cancelled()
    if progress_callback is None:
        return
    payload = {
        "stage": str(stage or ""),
        "message": str(message or ""),
    }
    if current is not None:
        payload["current"] = int(current)
    if total is not None:
        payload["total"] = int(total)
    if elapsed_sec is not None:
        payload["elapsed_sec"] = float(elapsed_sec)
    payload.update(extra)
    progress_callback(payload)


def _nnunet_predict_command(python_executable=None, bootstrap_resampler_path=None, optimize_transfers=False):
    if getattr(sys, "frozen", False):
        return [sys.executable, "--autoflow-internal-nnunet-predict"]
    code = (
        "from nnunetv2.inference.predict_from_raw_data import "
        "predict_entry_point_modelfolder as main; main()"
    )
    if bootstrap_resampler_path:
        source = repr(str(Path(bootstrap_resampler_path).expanduser().resolve()))
        code = (
            "from pathlib import Path; import shutil, importlib, nnunetv2; "
            f"_src=Path({source}); _dst=Path(nnunetv2.__file__).resolve().parent/'preprocessing'/'resampling'/'gpu_resampling.py'; "
            "_dst.parent.mkdir(parents=True, exist_ok=True); "
            "shutil.copy2(_src, _dst) if _src.is_file() else None; importlib.invalidate_caches(); "
            "from nnunetv2.inference.predict_from_raw_data import "
            "predict_entry_point_modelfolder as main; main()"
        )
    if optimize_transfers:
        # The predictor may use a separate interpreter and working directory.
        # Load the runtime helper from this AutoFlow checkout/package explicitly.
        runtime_root = repr(str(Path(__file__).resolve().parents[3]))
        code = code[:-6] + (
            f"import sys; sys.path.insert(0, {runtime_root}); "
            "from autoflow.nnunet_runtime import configure_exact_inference_runtime; "
            "configure_exact_inference_runtime(); main()"
        )
    if python_executable and Path(str(python_executable)).is_file():
        return [
            str(python_executable),
            "-c",
            code,
        ]
    executable = shutil.which("nnUNetv2_predict_from_modelfolder")
    if executable:
        if bootstrap_resampler_path or optimize_transfers:
            # The console entry point cannot execute the bootstrap prelude;
            # use the active interpreter so the child can install the helper.
            return [sys.executable, "-c", code]
        return [executable]
    return [
        sys.executable,
        "-c",
        code,
    ]


def resolve_auto_segmentation_device(device="cpu"):
    token = str(device or "").strip().lower()
    if token in {"", "auto", "cuda_if_available"}:
        try:
            import torch

            return "cuda" if bool(torch.cuda.is_available()) else "cpu"
        except Exception:
            return "cpu"
    if token in {"cpu", "cuda"}:
        return token
    return str(device)
