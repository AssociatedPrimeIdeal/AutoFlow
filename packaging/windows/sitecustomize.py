"""Application-local initialization, inherited by ordinary Python workers."""

import os
from pathlib import Path
import sys


root = Path(__file__).resolve().parents[3]
if (root / "portable.json").is_file():
    os.environ["AUTOFLOW_PORTABLE_ROOT"] = str(root)
    os.environ.setdefault("AUTOFLOW_MODELS_DIR", str(root / "models"))
    os.environ["QT_API"] = "pyside6"
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ["PATH"] = os.pathsep.join((str(root / "_runtime"), str(root / "_runtime" / "Scripts"), os.environ.get("PATH", "")))
    os.environ.setdefault("nnUNet_compile", "false")
    cache = Path(os.environ.get("LOCALAPPDATA", str(root))) / "AutoFlow" / "cache"
    os.environ.setdefault("CUPY_CACHE_DIR", str(cache / "cupy"))
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    # pythonw has no standard streams. Predictors use python.exe so their
    # diagnostics can be captured by the application's existing process runner.
    sys.executable = str(root / "_runtime" / "python.exe")
    _dll_handles = []
    if hasattr(os, "add_dll_directory"):
        site = Path(__file__).resolve().parent
        candidates = [site / "bin", site / "PySide6", site / "shiboken6", site / "torch" / "lib"]
        candidates += list((site / "nvidia").glob("*/bin"))
        candidates += list((site / "nvidia").glob("*/lib"))
        for directory in candidates:
            if directory.is_dir():
                _dll_handles.append(os.add_dll_directory(str(directory)))
