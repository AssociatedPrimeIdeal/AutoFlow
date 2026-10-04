"""PUDIP-Flow layout, parameter and velocity-to-phase adapter."""

from __future__ import annotations

from typing import Any, Dict
import numpy as np
from ._common import _backend_spacing, _backend_weight_mask, _deep_backend_params
from .backends import _backend_device, _load_external_backend


def _pudip_unwrap(phase: np.ndarray, weightmask: np.ndarray, venc: np.ndarray, config: Dict[str, Any], device: str):
    module = _load_external_backend("pudipflow")
    backend = _deep_backend_params(config, "pudip")
    resolved_device = _backend_device(device)
    wrapped = np.transpose(phase, (4, 3, 0, 1, 2))
    weight = _backend_weight_mask(weightmask, phase.shape)
    kwargs = {
        "venc": venc,
        "level": int(backend.get("level", 4)),
        "features": int(backend.get("features", 128)),
        "input_depth": int(backend.get("input_depth", 128)),
        "lr": float(backend.get("lr", 1e-3)),
        "num_iter": int(backend.get("num_iter", 1000)),
        "tv_weights": tuple(float(v) for v in backend.get("tv_weights", (1.0, 1.0, 1.0, 1.0))),
        "loss_type": str(backend.get("loss_type", "l1")),
        "device": resolved_device,
        "lr_scheduler": str(backend.get("lr_scheduler", "cosine")),
        "div_weight": float(backend.get("div_weight", 0.0)),
        "spacing": _backend_spacing(backend.get("spacing")),
        "reshape_mode": str(backend.get("reshape_mode", "bt_as_channel")),
    }
    runner = module.PUDIPFlow(**kwargs)
    recovered, history = runner.run(
        wrapped,
        weight,
        plot=False,
        save_video=False,
    )
    if hasattr(recovered, "detach"):
        recovered = recovered.detach().cpu().numpy()
    recovered = np.asarray(recovered, dtype=np.float32)
    if recovered.shape != wrapped.shape:
        raise ValueError(f"PUDIP-Flow returned {recovered.shape}, expected {wrapped.shape}")
    phase_unwrapped = np.transpose(
        recovered * np.pi / venc.reshape(3, 1, 1, 1, 1),
        (2, 3, 4, 1, 0),
    )
    return phase_unwrapped, resolved_device, {"iterations": int(kwargs["num_iter"]), "history": history}
