"""Optional, preserved DICOM-to-H5 conversion using the pinned submodule."""
import hashlib
import importlib
import os
from pathlib import Path
import sys
import tempfile

import h5py
import numpy as np

from .data import discover_h5_input_cases
from ..task_control import check_cancelled


def _converter_module():
    source = Path(__file__).resolve().parents[2] / "third_party/4DFlow_Dicom2H5/src"
    if (source / "dicom2h5/converter.py").is_file() and str(source) not in sys.path:
        # Keep this importable for the converter's process-pool workers.
        sys.path.insert(0, str(source))
    try:
        return importlib.import_module("dicom2h5.converter")
    except ImportError as exc:
        raise RuntimeError(
            "Dicom2H5 is unavailable. Run git submodule update --init "
            "third_party/4DFlow_Dicom2H5 and pip install -e "
            "third_party/4DFlow_Dicom2H5, or pip install 'autoflow-mri[dicom]'."
        ) from exc


def convert_dicom_input(dicom_directory, output_h5, progress_callback=None):
    """Convert a directory into a new H5 and return all validated InputCases.

    Existing destinations are never replaced. A failed/empty conversion does not
    publish an H5. The original DICOM files are read only.
    """
    root = Path(dicom_directory).expanduser().resolve()
    target = Path(output_h5).expanduser().absolute()
    if not root.is_dir():
        raise ValueError(f"DICOM input must be a directory: {root}")
    if target.exists():
        raise FileExistsError(f"Converted H5 already exists: {target}. Load it as H5 or choose a new destination.")
    target.parent.mkdir(parents=True, exist_ok=True)
    converter = _converter_module()
    descriptor, temporary = tempfile.mkstemp(prefix=".dicom2h5-", suffix=".h5", dir=target.parent)
    os.close(descriptor)
    try:
        if progress_callback:
            progress_callback({"message": "Converting DICOM to H5...", "current": 0, "total": 0})
        if getattr(converter, "__name__", "") == "dicom2h5.converter":
            from .segmentation.runtime import _run_subprocess
            def conversion_progress(record):
                if progress_callback:
                    progress_callback(dict(record, stage="dicom_conversion", message="Converting DICOM to H5…"))
            bootstrap = "import sys; sys.path.insert(0, sys.argv.pop(1)); from autoflow.algorithms.dicom_conversion import _converter_module; _converter_module().convert_dicom_to_h5(sys.argv[1], sys.argv[2])"
            result = _run_subprocess([sys.executable, "-c", bootstrap, str(Path(__file__).resolve().parents[2]), str(root), temporary],
                                     progress_callback=conversion_progress)
            if result.returncode:
                raise RuntimeError("DICOM conversion failed: "+result.stderr[-8000:])
        else:
            # Custom converter adapters keep their direct callable contract.
            converter.convert_dicom_to_h5(str(root), temporary)
        check_cancelled()
        if progress_callback:
            progress_callback({"message": "Validating converted H5 geometry and calibration...", "current": 0, "total": 0})
        report = converter.validate_native_h5(temporary)
        if not report.get("valid") or not report.get("groups"):
            raise ValueError(f"Dicom2H5 produced no valid flow cases: {report.get('errors', [])}")
        with h5py.File(temporary, "r") as handle:
            for name in report["groups"]:
                check_cancelled()
                group = handle[name] if name else handle
                for key in ("RR", "Resolution", "VENC", "Origin"):
                    value = np.asarray(group[key], dtype=float)
                    if not np.all(np.isfinite(value)) or (key != "Origin" and np.any(value <= 0)):
                        raise ValueError(f"Dicom2H5 group {name or '<root>'} has invalid {key}. Verify DICOM calibration before converting again.")
        cases = discover_h5_input_cases(temporary)
        if not cases:
            raise ValueError("Dicom2H5 produced no AutoFlow-compatible mag + flow groups.")
        # Atomic publication without replacing another run's output.
        check_cancelled()
        try:
            os.link(temporary, target)
        except OSError:
            if os.name != "nt":
                raise
            # Windows rename is atomic and fails if the target exists. This
            # supports filesystems that do not implement hard links.
            os.rename(temporary, target)
        cases = discover_h5_input_cases(target)
        for case in cases:
            case.input_path = str(target)
            case.metadata.update({
                "dicom_backend": "dicom2h5",
                "dicom_source_directory": str(root),
                "converted_h5": str(target),
            })
        if progress_callback:
            progress_callback({"message": f"Converted {len(cases)} H5 case(s)", "current": 1, "total": 1})
        return cases
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def converted_input_cases(dicom_directory, output_directory):
    """Choose a deterministic destination within a caller-owned output directory."""
    root = Path(dicom_directory).expanduser().resolve()
    digest = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:10]
    target = Path(output_directory) / f"{root.name or 'dicom'}-{digest}.h5"
    return convert_dicom_input(root, target)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python -m autoflow.algorithms.dicom_conversion INPUT OUTPUT")
    _converter_module().convert_dicom_to_h5(sys.argv[1], sys.argv[2])
