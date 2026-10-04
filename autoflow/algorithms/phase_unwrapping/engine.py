"""Phase-unwrapping dispatch, component orchestration and wrap diagnostics."""

from __future__ import annotations

import importlib.util
import importlib
import time
from typing import Any, Callable, Dict, Optional
import numpy as np
from .cpu import unwrap_data
from ._common import _as_mask_or_weight_xyz_t, _as_mask_xyz_t, _as_phase_xyz_t3, _coerce_venc
from .backends import METHODS, _canonical_method, _resolve_gpu_device
from .graphcut import _gc3d_unwrap_torch
from .gust import _gust_unwrap
from .laplacian import _lap4d_gpu
from .nprs import _nprs_gpu_fft
from .pudip import _pudip_unwrap


def unwrap_phase(
    phase_wrapped: np.ndarray,
    mask: Optional[np.ndarray],
    venc,
    method: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    device: str = "auto",
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    backend_weightmask: Optional[np.ndarray] = None,
    backend_center_confidence: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Run one traditional method and return flow plus wrap diagnostics."""
    method_token = str(method or "none").strip()
    if method_token.lower() == "none":
        raise ValueError("phase unwrapping method is disabled")
    method_token = _canonical_method(method_token)
    if method_token not in METHODS[1:]:
        raise ValueError(
            f"unsupported phase unwrapping method: {method}; choose gc3D, lap4D, nprs, pudip, or gust"
        )

    phase = _as_phase_xyz_t3(phase_wrapped)
    x, y, z, t, _ = phase.shape
    mask4 = _as_mask_xyz_t(mask, (x, y, z, t))
    backend_weight = mask4 if backend_weightmask is None else _as_mask_or_weight_xyz_t(
        backend_weightmask, (x, y, z, t), name="backend weight mask"
    )
    if not np.any(mask4):
        flow_wrapped = phase * _coerce_venc(venc).reshape((1, 1, 1, 1, 3)) / np.pi
        return {
            "flow_unwrapped": flow_wrapped.copy(), "flow_wrapped": flow_wrapped,
            "phase_unwrapped": phase.copy(), "wrap_count": np.zeros_like(phase, dtype=np.int16),
            "wrap_mask": np.zeros_like(phase, dtype=bool), "mask_used": mask4,
            "method": method_token, "device": "cpu", "elapsed_sec": 0.0,
            "statistics": {"evaluated_voxels": 0, "wrapped_voxels_any": 0, "max_abs_k": 0, "per_component": []},
        }
    venc3 = _coerce_venc(venc)
    cfg = dict(params or {})
    cfg.setdefault("tfc", True)
    cfg.setdefault("lap4d_ts", 2.0)
    cfg.setdefault("nprs_upsampling_factor", 2)
    cfg.setdefault("nprs_pi_unwrap", True)
    cfg.setdefault("nprs_auto_crop", True)
    cfg.setdefault("backend_params", {})
    started = time.perf_counter()
    phase_unwrapped = np.empty_like(phase, dtype=np.float32)
    wrap_count = np.zeros_like(phase, dtype=np.int16)
    used_device = "cpu"
    backend_metadata: Dict[str, Any] = {}
    gpu_device = _resolve_gpu_device(device)
    if method_token in {"pudip", "gust"}:
        if progress_callback is not None:
            progress_callback({
                "stage": "phase_unwrap_backend",
                "current": 0,
                "total": 1,
                "message": f"Running {method_token.upper()}-Flow phase unwrapping",
            })
        if method_token == "pudip":
            phase_unwrapped, used_device, backend_metadata = _pudip_unwrap(
                phase, backend_weight, venc3, cfg, device,
            )
        else:
            phase_unwrapped, used_device, backend_metadata = _gust_unwrap(
                phase, backend_weight, venc3, cfg, device,
                center_confidence=backend_center_confidence,
            )
        wrap_count = np.rint((phase_unwrapped - phase) / (2.0 * np.pi)).astype(np.int16)
    if method_token == "gc3D" and importlib.util.find_spec("maxflow") is None:
        raise RuntimeError("gc3D requires PyMaxflow; install it or choose lap4D/nprs")

    for component in range(3) if method_token not in {"pudip", "gust"} else ():
        if progress_callback is not None:
            progress_callback({"stage": "phase_unwrap_component", "current": component, "total": 3,
                               "message": f"Unwrapping phase component {component + 1}/3 ({method_token})"})
        phi = np.asarray(phase[..., component], dtype=np.float32)
        if method_token == "lap4D" and gpu_device:
            unwrapped, nr = _lap4d_gpu(phi, mask4, ts=float(cfg["lap4d_ts"]), device=gpu_device)
            used_device = gpu_device
        elif method_token == "gc3D" and gpu_device:
            # CUDA accelerates only masked graph construction; PyMaxflow/PUMA
            # remains on CPU so the numerical solve is unchanged.
            # Legacy ``gc3D_unwrap`` accumulates into a NumPy float64 array;
            # retain that precision for the wrap-count rounding step.
            unwrapped = np.empty_like(phi, dtype=np.float64)
            for time_index in range(t):
                unwrapped[..., time_index] = _gc3d_unwrap_torch(
                    phi[..., time_index], mask4[..., time_index], gpu_device
                )
            nr = np.round((np.asarray(unwrapped) - phi) / (2.0 * np.pi)).astype(np.int16)
            if bool(cfg["tfc"]):
                from ._common import total_field_correction

                unwrapped = total_field_correction(np.asarray(unwrapped, dtype=np.float32), mask4)
            used_device = f"{gpu_device}:graph"
        elif method_token == "nprs" and gpu_device and int(cfg["nprs_upsampling_factor"]) > 1:
            # Keep the reliability-guided skimage core byte-for-byte in spirit;
            # only its expensive Fourier resampling is evaluated on CUDA.
            unwrapped = np.empty_like(phi, dtype=np.float32)
            for time_index in range(t):
                unwrapped[..., time_index] = _nprs_gpu_fft(
                    phi[..., time_index],
                    mask4[..., time_index],
                    upsampling_factor=int(cfg["nprs_upsampling_factor"]),
                    pi_unwrap=bool(cfg["nprs_pi_unwrap"]),
                    auto_crop=bool(cfg["nprs_auto_crop"]),
                    device=gpu_device,
                )
            nr = np.round((np.asarray(unwrapped) - phi) / (2.0 * np.pi)).astype(np.int16)
            if bool(cfg["tfc"]):
                from ._common import total_field_correction

                unwrapped = total_field_correction(np.asarray(unwrapped, dtype=np.float32), mask4)
            used_device = f"{gpu_device}:fft"
        else:
            kwargs: Dict[str, Any] = {"tfc": bool(cfg["tfc"]), "verbose": False}
            if method_token == "lap4D":
                kwargs["ts"] = float(cfg["lap4d_ts"])
            elif method_token == "nprs":
                kwargs.update({
                    "upsampling_factor": max(1, int(cfg["nprs_upsampling_factor"])),
                    "pi_unwrap": bool(cfg["nprs_pi_unwrap"]),
                    "auto_crop": bool(cfg["nprs_auto_crop"]),
                })
            unwrapped, nr = unwrap_data(
                np.array(phi, copy=True),
                mode=method_token,
                venc=None,
                full=True,
                mask=mask4,
                **kwargs,
            )
        phase_unwrapped[..., component] = np.asarray(unwrapped, dtype=np.float32)
        wrap_count[..., component] = np.asarray(nr, dtype=np.int16)

    evaluated = mask4[..., None]
    wrap_count = np.where(evaluated, wrap_count, 0).astype(np.int16)
    wrap_mask = (np.abs(wrap_count) > 0) & evaluated
    # Preserve the loaded background outside the requested mask.
    phase_unwrapped = np.where(evaluated, phase_unwrapped, phase)
    flow_unwrapped = phase_unwrapped * venc3.reshape((1, 1, 1, 1, 3)) / np.pi
    flow_wrapped = phase * venc3.reshape((1, 1, 1, 1, 3)) / np.pi
    per_component = []
    for component in range(3):
        values = wrap_count[..., component][wrap_mask[..., component]]
        per_component.append({
            "component": int(component),
            "wrapped_voxels": int(values.size),
            "fraction_of_mask": float(values.size / max(1, int(np.count_nonzero(mask4)))),
            "min_k": int(values.min()) if values.size else 0,
            "max_k": int(values.max()) if values.size else 0,
            "phases": [int(i) for i in np.unique(np.where(wrap_mask[..., component])[3])],
        })
    return {
        "flow_unwrapped": np.asarray(flow_unwrapped, dtype=np.float32),
        "flow_wrapped": np.asarray(flow_wrapped, dtype=np.float32),
        "phase_unwrapped": np.asarray(phase_unwrapped, dtype=np.float32),
        "wrap_count": wrap_count,
        "wrap_mask": wrap_mask,
        "mask_used": mask4,
        "method": method_token,
        "device": used_device,
        "elapsed_sec": float(time.perf_counter() - started),
        "backend_metadata": backend_metadata,
        "statistics": {
            "evaluated_voxels": int(np.count_nonzero(mask4)),
            "wrapped_voxels_any": int(np.count_nonzero(np.any(wrap_mask, axis=-1))),
            "max_abs_k": int(np.max(np.abs(wrap_count))) if wrap_count.size else 0,
            "per_component": per_component,
        },
    }
