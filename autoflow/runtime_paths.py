"""Paths supplied by the Windows portable distribution, without changing pip installs."""

import os
from pathlib import Path


def portable_root():
    value = os.environ.get("AUTOFLOW_PORTABLE_ROOT", "").strip()
    return Path(value).expanduser().resolve() if value else None


def portable_models_dir():
    value = os.environ.get("AUTOFLOW_MODELS_DIR", "").strip()
    if value:
        return Path(value).expanduser().resolve()
    root = portable_root()
    return root / "models" if root is not None else None


def portable_model_folder(profile):
    root = portable_models_dir()
    return root / profile if root is not None else None


DOCUMENTATION_URL = "https://associatedprimeideal.github.io/autoflow-docs/"


def documentation_index():
    """Prefer this build's local pages; installed/source GUIs can use the web."""
    root = portable_root()
    if root is not None:
        candidates = (root / "docs" / "site" / "index.html",)
    else:
        repository = Path(__file__).resolve().parents[1]
        candidates = (repository / "build" / "docs-offline" / "index.html",
                      repository / "site" / "index.html")
    return next((path for path in candidates if path.is_file()
                 and (path.parent / "offline-manifest.json").is_file()), None)
