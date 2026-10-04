"""GUST-Flow confidence, CUDA diagnostics and velocity-to-phase adapter."""

from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np
from ._common import _backend_spacing, _backend_weight_mask, _deep_backend_params
from .backends import _backend_device, _load_external_backend


def _gust_unwrap(
    phase: np.ndarray,
    weightmask: np.ndarray,
    venc: np.ndarray,
    config: Dict[str, Any],
    device: str,
    *,
    center_confidence: Optional[np.ndarray] = None,
):
    module = _load_external_backend("gustflow")
    backend = _deep_backend_params(config, "gust")
    resolved_device = _backend_device(device, require_cuda=True)
    wrapped = np.transpose(phase, (4, 3, 0, 1, 2))
    weight = _backend_weight_mask(weightmask, phase.shape)
    confidence = backend.get("center_confidence") if center_confidence is None else center_confidence
    if confidence is None:
        confidence = np.any(weightmask, axis=3).astype(np.float32)
    confidence = np.asarray(confidence, dtype=np.float32)
    if confidence.shape != phase.shape[:3]:
        raise ValueError(
            f"GUST-Flow center_confidence must have shape {phase.shape[:3]}, got {confidence.shape}"
        )
    runner = module.GUSTFlow(
        venc=venc,
        voxel_spacing=_backend_spacing(backend.get("voxel_spacing")),
        num_primitives=int(backend.get("num_primitives", 8192)),
        num_iter=int(backend.get("num_iter", 1000)),
        lr=float(backend.get("lr", 0.03)),
        device=resolved_device,
        seed=int(backend.get("seed", 314159)),
    )
    try:
        result = runner.fit(wrapped, weight, center_confidence=confidence)
    except Exception as exc:
        detail = str(exc).strip() or type(exc).__name__
        lowered = detail.lower()
        if "out of memory" in lowered or "cuda" in lowered or "cupy" in lowered:
            raise RuntimeError(
                "GUST-Flow CUDA execution failed (usually insufficient VRAM or an "
                "incompatible CUDA/CuPy/PyTorch build). Try Lap4D, reduce the input "
                "volume, or run GUST-Flow in a matching CUDA environment. "
                f"Original error: {detail}"
            ) from exc
        raise RuntimeError(f"GUST-Flow failed: {detail}") from exc
    recovered = getattr(result, "recovered", result)
    if hasattr(recovered, "detach"):
        recovered = recovered.detach().cpu().numpy()
    recovered = np.asarray(recovered, dtype=np.float32)
    if recovered.shape != wrapped.shape:
        raise ValueError(f"GUST-Flow returned {recovered.shape}, expected {wrapped.shape}")
    phase_unwrapped = np.transpose(
        recovered * np.pi / venc.reshape(3, 1, 1, 1, 1),
        (2, 3, 4, 1, 0),
    )
    return phase_unwrapped, resolved_device, {"iterations": int(backend.get("num_iter", 1000))}
