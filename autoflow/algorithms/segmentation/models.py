"""nnUNet profiles, checkpoints, folds and pipeline/model path resolution."""

import json
import os
import re
from pathlib import Path


_AUTOFLOW_INTERNAL_SPATIAL_ORDER = ("LR", "AP", "FH")


_AUTOFLOW_INTERNAL_VENC_ORDER = ("LR", "AP", "FH")


_NNUNET_TARGET_SPATIAL_ORDER = ("HF", "AP", "RL")


_NNUNET_TARGET_VENC_ORDER = ("HF", "AP", "RL")


_NNUNET_GPU_PREPROCESSING_ENV = "AUTOFLOW_NNUNET_GPU_PREPROCESSING"


_NNUNET_4D_BACKENDS = {"nnunet4d", "nnunet_4d", "4d"}


_NNUNET_3D_MODEL_DEFAULT = Path(
    "/nas-data/ryy_rawdata/aorta_seg/nnres/Dataset7010_All_Mean/"
    "nnUNetTrainerPartBalancedTversky__nnUNetPlans__3d_fullres"
)


_NNUNET_3D_CHECKPOINT_DEFAULT = "checkpoint_final.pth"


_NNUNET_4D_MODEL_DEFAULT = Path(
    "/nas-data/ryy_rawdata/aorta_seg/nnres_noCC/Dataset7020_Aorta_4DTemporalFT/"
    "nnUNetTrainerPartBalancedTversky__nnUNetPlansIso1mm__3d_fullres"
)


_NNUNET_4D_CHECKPOINT_DEFAULT = "checkpoint_best.pth"


_NNUNET_4D_PIPELINE_DEFAULT = Path(
    "/nas-data2/ryy/CMR4DFlow2026/Segdata/scripts/nnunet/4D/"
    "run_7020_4d_full_ssd_20260824.sh"
)


_NNUNET_4D_RESULTS_DEFAULT = Path("/nas-data/ryy_rawdata/aorta_seg/nnres")


_NNUNET_4D_DATASET_DEFAULT = "Dataset7020_Aorta_4DTemporalFT"


_NNUNET_4D_TRAINER_DEFAULT = "nnUNetTrainerPartBalancedTversky"


_NNUNET_4D_PLANS_DEFAULT = "nnUNetPlansIso1mm"


_NNUNET_4D_CONFIGURATION_DEFAULT = "3d_fullres"


_NNUNET_DEFAULT_CHANNEL_ORDER = (
    "mag_std_xyz",
    "mag_mean_xyz",
    "pcmra_std_xyz",
    "pcmra_mean_xyz",
    "flow_x_mean_xyz",
    "flow_y_mean_xyz",
    "flow_z_mean_xyz",
    "flow_mag_mean_xyz",
    "flow_x_std_xyz",
    "flow_y_std_xyz",
    "flow_z_std_xyz",
    "flow_mag_std_xyz",
)


def _load_nnunet_model_metadata(model_folder):
    model_path = Path(model_folder).expanduser()
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path
    model_path = model_path.resolve()
    if model_path.name.startswith("fold_") and not (model_path / "dataset.json").is_file():
        model_path = model_path.parent
    if not model_path.is_dir():
        raise FileNotFoundError(f"nnUNet model folder not found: {model_path}")

    dataset_json_path = model_path / "dataset.json"
    plans_json_path = model_path / "plans.json"
    if not dataset_json_path.is_file():
        raise FileNotFoundError(f"missing nnUNet dataset.json: {dataset_json_path}")
    if not plans_json_path.is_file():
        raise FileNotFoundError(f"missing nnUNet plans.json: {plans_json_path}")

    with dataset_json_path.open("r", encoding="utf-8") as f:
        dataset_json = json.load(f)
    if not isinstance(dataset_json, dict):
        raise ValueError(f"dataset.json must contain a JSON object: {dataset_json_path}")

    return model_path, dataset_json


def _detect_nnunet_folds(model_path):
    folds = []
    has_fold_all = False
    for child in sorted(model_path.iterdir()):
        if not child.is_dir() or not child.name.startswith("fold_"):
            continue
        suffix = child.name[len("fold_"):]
        if suffix == "all":
            has_fold_all = True
            continue
        if suffix.isdigit():
            folds.append(int(suffix))
    if folds:
        return [str(fold) for fold in sorted(set(folds))]
    if has_fold_all:
        return ["all"]
    raise FileNotFoundError(f"no nnUNet fold_* folders found in {model_path}")


def bundled_nnunet_model_folder():
    model_name = "nnUNetTrainerPartBalanced__nnUNetPlans__3d_fullres_iso1mm"
    return Path(__file__).resolve().parents[2] / "segmodel" / model_name


def default_nnunet_model_folder():
    """Return the preferred static model, with the packaged model as fallback."""
    if _NNUNET_3D_MODEL_DEFAULT.is_dir():
        return _NNUNET_3D_MODEL_DEFAULT
    return bundled_nnunet_model_folder()


def _resolve_bundled_relative_path(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    if path.is_dir():
        return path.resolve()

    package_root = Path(__file__).resolve().parents[3]
    packaged_path = package_root / path
    bundled_default = bundled_nnunet_model_folder()
    legacy_default = Path("autoflow") / "segmodel" / bundled_default.name
    if path == legacy_default:
        return bundled_default
    if packaged_path.is_dir():
        return packaged_path.resolve()
    return path.resolve()


def resolve_nnunet_model_folder(model_folder=""):
    candidate = str(model_folder or "").strip()
    if candidate.lower() in {"auto", "default"}:
        candidate = ""
    if candidate:
        return str(_resolve_bundled_relative_path(candidate))
    default_path = default_nnunet_model_folder()
    if default_path.is_dir():
        return str(default_path)
    raise FileNotFoundError(
        "default 3D nnUNet model folder is missing: "
        f"{default_path}. Set an explicit model folder to override the automatic profile"
    )


def default_nnunet_4d_pipeline_script():
    """Return the optional local Dataset7020 training/prediction script."""
    return str(_NNUNET_4D_PIPELINE_DEFAULT)


def _model_folder_from_4d_pipeline_script(script_path):
    """Resolve a Dataset7020 model directory from its shell configuration.

    The supplied ``run_7020_4d_full_ssd_20260824.sh`` is a training/prediction
    orchestrator rather than an inference executable.  Parsing its stable
    ``NNUNET_RESULTS`` and ``DATASET`` assignments lets callers point
    ``auto_model`` at that script while AutoFlow still invokes nnUNet once per
    4D case.  Missing or non-standard assignments simply fall back to the
    documented Dataset7020 defaults.
    """
    path = Path(script_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"4D nnUNet pipeline script not found: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")

    def assignment(name, fallback):
        match = re.search(rf"^\s*{re.escape(name)}=\"([^\"]+)\"", text, flags=re.MULTILINE)
        return match.group(1) if match else fallback

    # Reuse the same local shell-assignment expansion used for the child
    # predictor environment, so ``NNUNET_RESULTS="${SSD_ROOT}/nnres"`` is
    # resolved before constructing the model folder.
    assignments = _nnunet_4d_pipeline_assignments(path)
    results_root = Path(
        assignments.get("NNUNET_RESULTS", assignment("NNUNET_RESULTS", str(_NNUNET_4D_RESULTS_DEFAULT)))
    ).expanduser()
    dataset = assignments.get("DATASET", assignment("DATASET", _NNUNET_4D_DATASET_DEFAULT))
    trainer = _NNUNET_4D_TRAINER_DEFAULT
    plans = _NNUNET_4D_PLANS_DEFAULT
    configuration = _NNUNET_4D_CONFIGURATION_DEFAULT
    model = results_root / dataset / f"{trainer}__{plans}__{configuration}"
    return model.resolve()


def _python_from_4d_pipeline_script(script_path):
    path = Path(script_path).expanduser().resolve()
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"PYTHON_BIN=\"\$\{PYTHON_BIN:-([^}]+)\}\"", text)
    candidate = str(match.group(1)).strip() if match else ""
    return candidate if candidate and Path(candidate).is_file() else ""


def _nnunet_4d_pipeline_assignments(script_path):
    """Read stable environment assignments from the Dataset7020 shell script."""
    path = Path(script_path).expanduser().resolve()
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    values = {}
    for name in ("SSD_ROOT", "NNUNET_RAW", "NNUNET_PREPROCESSED", "NNUNET_RESULTS", "DATASET"):
        match = re.search(rf"^\s*{re.escape(name)}=\"([^\"]+)\"", text, flags=re.MULTILINE)
        if match:
            values[name] = match.group(1)
    # The shell script intentionally derives cache paths from SSD_ROOT. Expand
    # only the local assignments so ${SSD_ROOT} never leaks into child env.
    for name, value in list(values.items()):
        for _ in range(4):
            replaced = re.sub(
                r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)",
                lambda match: values.get(match.group(1) or match.group(2), match.group(0)),
                value,
            )
            if replaced == value:
                break
            value = replaced
        values[name] = value
    return values


def _nnunet_4d_subprocess_env(pipeline_script, model_path):
    """Build an isolated nnUNet environment for the external 4D predictor."""
    env = os.environ.copy()
    script_path = Path(pipeline_script).expanduser().resolve() if pipeline_script else None
    if script_path is not None and script_path.is_file():
        # The training script imports project helpers and custom trainer modules
        # from both scripts/ and scripts/nnunet/. Keep the parent process env
        # untouched while making those imports available to the child.
        roots = [script_path.parent, script_path.parent.parent]
        existing = [item for item in str(env.get("PYTHONPATH", "")).split(os.pathsep) if item]
        env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys([*(str(root) for root in roots), *existing]))
        assignments = _nnunet_4d_pipeline_assignments(script_path)
        if assignments.get("NNUNET_RAW"):
            env["nnUNet_raw"] = assignments["NNUNET_RAW"]
        if assignments.get("NNUNET_PREPROCESSED"):
            env["nnUNet_preprocessed"] = assignments["NNUNET_PREPROCESSED"]
        if assignments.get("NNUNET_RESULTS"):
            env["nnUNet_results"] = assignments["NNUNET_RESULTS"]
    if model_path:
        # A model folder is always .../nnUNet_results/Dataset/TrainerConfig.
        # This fallback also handles scripts with a non-standard variable name.
        model_root = Path(model_path).expanduser().resolve().parent.parent
        env["nnUNet_results"] = str(model_root)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    return env


def resolve_nnunet_4d_model_folder(model_folder=""):
    """Resolve a 4D model folder or a Dataset7020 orchestration script path."""
    candidate = str(model_folder or "").strip()
    if candidate.lower() in {"auto", "default"}:
        candidate = ""
    if candidate.lower().endswith((".sh", ".bash")):
        return str(_model_folder_from_4d_pipeline_script(candidate))
    if candidate:
        return str(_resolve_bundled_relative_path(candidate))
    if _NNUNET_4D_MODEL_DEFAULT.is_dir():
        return str(_NNUNET_4D_MODEL_DEFAULT)
    model = _model_folder_from_4d_pipeline_script(_NNUNET_4D_PIPELINE_DEFAULT)
    if model.is_dir():
        return str(model)
    raise FileNotFoundError(
        "4D nnUNet model folder is missing: "
        f"{model}. Set an explicit 4D model folder or pipeline script via auto_model."
    )


def _resolve_nnunet_checkpoint(
    model_path,
    checkpoint_name,
    folds,
    *,
    default_checkpoint=_NNUNET_3D_CHECKPOINT_DEFAULT,
):
    """Resolve an explicit checkpoint or a backend-specific automatic default."""
    requested = str(checkpoint_name or "auto").strip()
    if requested.lower() in {"", "auto", "default"}:
        alternate = (
            "checkpoint_best.pth"
            if default_checkpoint == "checkpoint_final.pth"
            else "checkpoint_final.pth"
        )
        candidates = [str(default_checkpoint), alternate]
    else:
        candidates = [requested]
        if requested == "checkpoint_final.pth":
            candidates.append("checkpoint_best.pth")
    for candidate in candidates:
        if all((Path(model_path) / f"fold_{fold}" / candidate).is_file() for fold in folds):
            return candidate
    return candidates[0]


def _resolve_nnunet_folds(model_path, folds=None):
    """Resolve ``single``, ``all`` or an explicit fold list deterministically."""
    model_path = Path(model_path)
    detected = _detect_nnunet_folds(model_path)
    has_fold_all = (model_path / "fold_all").is_dir()
    if folds is None or str(folds).strip().lower() in {"", "auto"}:
        return detected
    if isinstance(folds, str):
        token = folds.strip().lower()
        if token in {"all", "ensemble", "5fold", "fivefold"}:
            return detected
        if token in {"single", "one"}:
            # ``fold_all`` is the full-data single-model checkpoint.  Keep it
            # as the explicit single branch even when five numeric folds are
            # also present for a future ensemble run.
            return ["all"] if has_fold_all else [detected[0]]
        values = [part.strip().lower().removeprefix("fold_") for part in folds.split(",") if part.strip()]
    else:
        values = [str(value).strip().lower().removeprefix("fold_") for value in folds if str(value).strip()]
    if not values:
        return detected
    valid = set(detected)
    unknown = [value for value in values if value not in valid]
    if unknown:
        raise FileNotFoundError(f"requested nnUNet folds are unavailable: {unknown}; found {detected}")
    return list(dict.fromkeys(values))
