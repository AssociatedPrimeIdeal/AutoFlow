"""Canonical phase/VENC arrays, mask weights and learned-backend parameters."""

from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np


def total_field_correction(to_unwrap, mask):
    """
    This function performs phase unwrapping on a four-dimensional numpy array 'to_unwrap' using the array 'mask'
    as a binary mask to apply weights during the unwrapping process. The function calculates the energy as a weighted
    sum of the 'to_unwrap' array, adjusts it by phase unwrapping to avoid discontinuities, and calculates the number
    of full 2π wraps required to align the unwrapped phase with the original phase. The result is an adjusted
    'to_unwrap' array with corrected phase values to maximize phase consistency across the time dimension.

    Parameters:
    - to_unwrap (numpy.ndarray): A 4-dimensional numpy array containing the data to be phase-corrected.
    - mask (numpy.ndarray): A 4-dimensional binary mask array of the same shape as 'to_unwrap'. It specifies the
      regions of 'to_unwrap' over which the sum (energy) and phase corrections should be applied.

    Returns:
    - numpy.ndarray: The phase-corrected version of 'to_unwrap', with adjustments made by adding necessary multiples of 2π.
    """
    if to_unwrap.ndim != 4:
        raise ValueError("Input array must have 4 dimensions.")
    if to_unwrap.shape != mask.shape:
        raise ValueError("Input array and mask have different shapes.")

    energy = np.sum(to_unwrap * mask, axis=(0, 1, 2))
    n_voxels = int(len(np.argwhere(mask)) / mask.shape[-1])
    unwrap_energy = np.unwrap(energy, discont=np.pi * n_voxels, period=2 * np.pi * n_voxels)
    n_wraps = np.round((unwrap_energy - energy) / (2 * np.pi * n_voxels))
    return to_unwrap + n_wraps * np.pi * 2


def _backend_spacing(value) -> tuple[float, float, float]:
    arr = np.asarray(value if value is not None else (1.0, 1.0, 1.0), dtype=np.float32).reshape(-1)
    if arr.size == 1:
        arr = np.repeat(arr, 3)
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])) or np.any(arr[:3] <= 0):
        raise ValueError(f"backend voxel spacing must contain three positive values, got {arr.tolist()}")
    return tuple(float(v) for v in arr[:3])


def _backend_weight_mask(mask: np.ndarray, phase_shape) -> np.ndarray:
    x, y, z, t, components = (int(v) for v in phase_shape)
    if components != 3 or tuple(mask.shape) != (x, y, z, t):
        raise ValueError("phase backend mask does not match the phase shape")
    return np.broadcast_to(
        np.transpose(np.asarray(mask, dtype=np.float32), (3, 0, 1, 2))[None, ...],
        (3, t, x, y, z),
    ).copy()


def _deep_backend_params(config: Dict[str, Any], method: str) -> Dict[str, Any]:
    raw = dict(config.get("backend_params") or {})
    params = dict(raw.get(method) or {}) if isinstance(raw.get(method), dict) else {}
    for key, value in raw.items():
        if key != method and not isinstance(value, dict):
            params.setdefault(str(key), value)
    prefix = f"{method}_"
    for key, value in config.items():
        if str(key).startswith(prefix):
            params.setdefault(str(key)[len(prefix):], value)
    return params


def _as_phase_xyz_t3(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 5 or arr.shape[-1] != 3:
        raise ValueError(f"wrapped phase must have shape XYZTV3, got {arr.shape}")
    return np.ascontiguousarray(arr)


def _as_mask_xyz_t(mask: Optional[np.ndarray], shape_xyz_t) -> np.ndarray:
    x, y, z, t = (int(v) for v in shape_xyz_t)
    if mask is None:
        return np.ones((x, y, z, t), dtype=bool)
    arr = np.asarray(mask)
    if arr.ndim == 3:
        if tuple(arr.shape) != (x, y, z):
            raise ValueError(f"mask shape {arr.shape} does not match {(x, y, z)}")
        arr = np.repeat(arr[..., None], t, axis=3)
    elif arr.ndim == 4:
        if tuple(arr.shape) != (x, y, z, t):
            raise ValueError(f"mask shape {arr.shape} does not match {(x, y, z, t)}")
    else:
        raise ValueError(f"mask must be XYZ or XYZT, got {arr.shape}")
    return np.asarray(arr > 0, dtype=bool)


def _as_mask_or_weight_xyz_t(value: np.ndarray, shape_xyz_t, *, name: str) -> np.ndarray:
    """Normalize a learned-backend weight map without binarizing it."""
    x, y, z, t = (int(v) for v in shape_xyz_t)
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 3:
        if tuple(arr.shape) != (x, y, z):
            raise ValueError(f"{name} shape {arr.shape} does not match {(x, y, z)}")
        arr = np.repeat(arr[..., None], t, axis=3)
    elif arr.ndim != 4 or tuple(arr.shape) != (x, y, z, t):
        raise ValueError(f"{name} must be XYZ or XYZT, got {arr.shape}")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError(f"{name} must contain finite non-negative values")
    return np.ascontiguousarray(arr, dtype=np.float32)


def _coerce_venc(venc) -> np.ndarray:
    arr = np.asarray(venc, dtype=np.float32).reshape(-1)
    if arr.size == 1:
        arr = np.repeat(arr, 3)
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])) or np.any(arr[:3] <= 0):
        raise ValueError(f"venc must contain three positive finite values, got {arr.tolist()}")
    return np.asarray(arr[:3], dtype=np.float32)
