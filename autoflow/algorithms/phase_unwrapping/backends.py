"""Method and mask policies, optional dependencies and device selection."""

from __future__ import annotations

import importlib.util
import importlib
import sys
from typing import Optional


METHODS = ("none", "gc3D", "lap4D", "nprs", "pudip", "gust")


_LEARNED_BACKEND_PACKAGES = {"pudip": "pudipflow", "gust": "gustflow"}


_LEARNED_BACKEND_REQUIREMENTS = {
    "pudip": ("torch", "tqdm", "matplotlib"),
    "gust": ("torch", "cupy", "matplotlib"),
}


def _canonical_method(method: str) -> str:
    aliases = {
        "gc3d": "gc3D", "lap4d": "lap4D", "nprs": "nprs",
        "pudip": "pudip", "pudip-flow": "pudip", "pudipflow": "pudip",
        "gust": "gust", "gust-flow": "gust", "gustflow": "gust",
    }
    token = str(method or "none").strip()
    return aliases.get(token.lower(), token)


def mask_sources_for_method(method: str) -> tuple[str, ...]:
    if _canonical_method(method) in {"pudip", "gust"}:
        return ("pcmra_std", "pcmra_mean", "none", "segmask")
    return ("none", "segmask")


def resolve_mask_source(method: str, source: str = "auto") -> str:
    token = str(source or "auto").strip().lower()
    token = {"segmentation": "segmask", "active_segmentation": "segmask", "seg": "segmask",
             "mask": "segmask", "pcmrastd": "pcmra_std", "pcmramean": "pcmra_mean"}.get(token, token)
    allowed = mask_sources_for_method(method)
    if token == "auto":
        return allowed[0]
    if token not in allowed:
        raise ValueError(f"{method} supports mask sources {', '.join(allowed)}; got {source!r}")
    return token


def backend_available(method: str) -> bool:
    """Return whether an optional learned backend is installed and importable."""
    token = _canonical_method(method)
    package_name = _LEARNED_BACKEND_PACKAGES.get(token)
    if package_name is None:
        return True
    try:
        package_available = package_name in sys.modules or importlib.util.find_spec(package_name) is not None
        if not package_available:
            return False
        return all(
            dependency in sys.modules or importlib.util.find_spec(dependency) is not None
            for dependency in _LEARNED_BACKEND_REQUIREMENTS[token]
        )
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _load_external_backend(package_name: str):
    """Load a phase-unwrapping package installed by the optional ``pu`` extra."""
    try:
        return importlib.import_module(package_name)
    except Exception as exc:
        raise RuntimeError(
            f"{package_name} is unavailable; install the optional phase-unwrapping "
            f"dependencies with pip install .[pu]"
        ) from exc


def _backend_device(device: str, *, require_cuda: bool = False) -> str:
    token = str(device or "auto").strip().lower()
    if token in {"cpu", "none"}:
        if require_cuda:
            raise RuntimeError("the selected phase-unwrapping backend requires CUDA")
        return "cpu"
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("the selected phase-unwrapping backend requires PyTorch") from exc
    has_cuda = bool(torch.cuda.is_available())
    if token in {"cuda", "gpu", "auto"}:
        if has_cuda:
            return "cuda"
        if require_cuda:
            raise RuntimeError("the selected phase-unwrapping backend requires a CUDA device")
        return "cpu"
    if token.startswith("cuda:"):
        if not has_cuda:
            raise RuntimeError(f"requested phase-unwrapping device {device!r}, but CUDA is unavailable")
        return token
    raise ValueError(f"unsupported phase-unwrapping device: {device!r}")


def _resolve_gpu_device(device: str) -> Optional[str]:
    token = str(device or "auto").strip().lower()
    try:
        import torch
    except Exception:
        return None
    if token in {"cpu", "none"}:
        return None
    if token in {"cuda", "gpu", "auto"} and torch.cuda.is_available():
        return "cuda"
    if token.startswith("cuda:") and torch.cuda.is_available():
        return token
    return None
