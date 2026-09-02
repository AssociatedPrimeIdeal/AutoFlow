"""Optional traditional phase-unwrapping backends.

The public API uses AutoFlow's canonical ``XYZTV3`` layout (phase in radians,
velocity in cm/s).  The bundled PUDIP-Flow implementation uses ``Nv,Nt,X,Y,Z``;
conversion and diagnostics live here so the rest of AutoFlow never needs to
know about that legacy layout.
"""

from __future__ import annotations

import importlib.util
import time
from typing import Any, Callable, Dict, Optional

import numpy as np

from .traditional import unwrap_data


METHODS = ("none", "gc3D", "lap4D", "nprs")


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


def _coerce_venc(venc) -> np.ndarray:
    arr = np.asarray(venc, dtype=np.float32).reshape(-1)
    if arr.size == 1:
        arr = np.repeat(arr, 3)
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])) or np.any(arr[:3] <= 0):
        raise ValueError(f"venc must contain three positive finite values, got {arr.tolist()}")
    return np.asarray(arr[:3], dtype=np.float32)


def _fftshift_torch(value, dims):
    import torch

    return torch.fft.fftshift(value, dim=dims)


def _ifftshift_torch(value, dims):
    import torch

    return torch.fft.ifftshift(value, dim=dims)


def _lap4d_gpu(phi_w: np.ndarray, mask: np.ndarray, *, ts: float, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Torch implementation of the PUDIP 4D Laplacian unwrap."""
    import torch

    dev = torch.device(device)
    original_shape = tuple(int(v) for v in phi_w.shape)
    even_shape = tuple((v // 2) * 2 for v in original_shape)
    slices = tuple(slice(0, v) for v in even_shape)
    phi = torch.as_tensor(np.asarray(phi_w[slices], dtype=np.float32), device=dev)
    sx, sy, sz, st = (int(v) for v in phi.shape)
    ranges = [torch.arange(-(n // 2), n // 2, device=dev, dtype=torch.float32) for n in (sx, sy, sz, st)]
    X, Y, Z, T = torch.meshgrid(*ranges, indexing="ij")
    mod = 2 * torch.cos(np.pi * X / sx) + 2 * torch.cos(np.pi * Y / sy) + 2 * torch.cos(np.pi * Z / sz)
    mod = mod + float(ts) * torch.cos(np.pi * T / st) - 6.0 - float(ts)
    dims = (0, 1, 2, 3)

    def lap(value, inverse=False):
        spectrum = _fftshift_torch(torch.fft.fftn(value), dims)
        if inverse:
            safe = torch.where(mod == 0, torch.ones_like(mod), mod)
            spectrum = spectrum / safe
        else:
            spectrum = spectrum * mod
        return torch.real(torch.fft.ifftn(_ifftshift_torch(spectrum, dims)))

    lap_phiw = lap(phi)
    lap_phi = torch.cos(phi) * lap(torch.sin(phi)) - torch.sin(phi) * lap(torch.cos(phi))
    ilap = lap(lap_phi - lap_phiw, inverse=True)
    nr_even = torch.round(ilap / (2.0 * np.pi)).to(torch.int16).cpu().numpy()
    nr = np.zeros(original_shape, dtype=np.int16)
    nr[slices] = nr_even
    phase = np.asarray(phi_w, dtype=np.float32) + 2.0 * np.pi * nr.astype(np.float32)
    # The bundled total-field correction is intentionally kept identical.
    if np.any(mask):
        from .traditional.flowunwrap import total_field_correction

        phase = total_field_correction(phase, mask.astype(np.int16))
    return np.asarray(phase, dtype=np.float32), nr


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


def _torch_pad_or_crop(value, widths):
    """Match the symmetric pad/box-crop behavior used by legacy ``pad_array``."""
    import torch

    widths = tuple(int(w) for w in widths)
    if all(w >= 0 for w in widths):
        out_shape = tuple(int(n + 2 * w) for n, w in zip(value.shape, widths))
        out = torch.zeros(out_shape, dtype=value.dtype, device=value.device)
        slices = tuple(slice(w, w + int(n)) for n, w in zip(value.shape, widths))
        out[slices] = value
        return out
    if all(w <= 0 for w in widths):
        slices = tuple(slice(-w, int(n + w)) for n, w in zip(value.shape, widths))
        return value[slices]
    raise ValueError("pad/crop widths must have a consistent sign")


def _torch_fft_resample(value: np.ndarray, shape_to_resample, factor: float, device: str) -> np.ndarray:
    """GPU equivalent of the legacy Fourier ``upsample``/``downsample`` helpers."""
    import torch

    arr = np.asarray(value)
    tensor = torch.as_tensor(arr, device=torch.device(device))
    spectrum = torch.fft.fftshift(torch.fft.fftn(tensor, norm="ortho"))
    target = tuple(int(n) for n in shape_to_resample)
    # Legacy ``upsample`` receives an array at ``target`` size and pads it;
    # ``downsample`` receives the expanded array and crops it back.
    if tuple(int(n) for n in arr.shape) == target:
        widths = tuple(int(n / float(factor)) for n in target)
    else:
        widths = tuple(-int(n / float(factor)) for n in target)
    spectrum = _torch_pad_or_crop(spectrum, widths)
    out = torch.fft.ifftn(torch.fft.ifftshift(spectrum), norm="forward")
    out = out / np.sqrt(float(np.prod(shape_to_resample)))
    return out.detach().cpu().numpy()


def _nprs_gpu_fft(
    to_unwrap: np.ndarray,
    mask: np.ndarray,
    *,
    upsampling_factor: int = 2,
    pi_unwrap: bool = True,
    auto_crop: bool = False,
    n_voxels: int = 5,
    device: str = "cuda",
) -> np.ndarray:
    """NPRS with only Fourier resampling on CUDA; skimage unwrap remains unchanged."""
    from skimage.restoration import unwrap_phase as skimage_unwrap_phase

    to_unwrap = np.asarray(to_unwrap, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if to_unwrap.shape != mask.shape or to_unwrap.ndim > 3:
        raise ValueError("NPRS input and mask must have matching arrays of at most 3 dimensions")
    factor = int(upsampling_factor)
    if factor < 1:
        raise ValueError("upsampling factor must be >=1")
    if factor == 1:
        factor = 0
    elif factor > 1:
        factor = 2 / (factor - 1)
    target_slice = None
    reference = to_unwrap
    if auto_crop:
        where = np.where(mask > 0.5)
        if not where[0].size:
            return np.zeros_like(to_unwrap)
        bounds = [(max(int(np.min(axis)) - n_voxels, 0), min(int(np.max(axis)) + n_voxels, to_unwrap.shape[i]))
                  for i, axis in enumerate(where)]
        target_slice = tuple(slice(lo, hi) for lo, hi in bounds)
        to_unwrap = to_unwrap[target_slice]
        mask = mask[target_slice]
    shape = tuple(int(v) for v in to_unwrap.shape)
    wrapped = np.array(to_unwrap, copy=True)
    if factor == 0:
        result = skimage_unwrap_phase(np.ma.masked_array(to_unwrap, mask=~mask))
    else:
        upmask = np.abs(_torch_fft_resample(mask.astype(np.float32), shape, factor, device))
        upmask = (upmask >= 0.5).astype(np.uint8)
        complex_phase = np.exp(1j * to_unwrap)
        expanded = np.angle(_torch_fft_resample(complex_phase, shape, factor, device))
        result = skimage_unwrap_phase(np.ma.masked_array(expanded, mask=(1 - upmask)))
        normalization = max(float(result.max()), float(np.abs(result.min())))
        # Always execute the inverse resampling, including the all-zero case;
        # otherwise the expanded FFT shape would leak into the cropped output.
        scale = normalization if normalization > 1e-12 else 1.0
        normalized = np.exp(1j * result / scale * np.pi)
        result = np.angle(_torch_fft_resample(normalized, shape, factor, device))
        result = result / np.pi * normalization
    result = np.asarray(result, dtype=np.float32)
    if auto_crop:
        if pi_unwrap:
            result = wrapped + 2 * np.pi * np.round((result - wrapped) / (2 * np.pi))
        out = np.zeros_like(reference)
        out[target_slice] = result
        return out
    if pi_unwrap:
        result = wrapped + 2 * np.pi * np.round((result - wrapped) / (2 * np.pi))
    return result


def _gc3d_torch_edges(phi_w: np.ndarray, mask: np.ndarray, device: str):
    """Build the legacy gc3D masked graph with torch, preserving node order.

    The max-flow/PUMA solve remains the original CPU implementation.  Keeping
    that solver unchanged avoids numerical changes while moving the expensive
    coordinate/edge discovery to CUDA.
    """
    import torch

    phase = np.asarray(phi_w)
    include = np.asarray(mask, dtype=bool) & (phase != 0)
    dev = torch.device(device)
    include_t = torch.as_tensor(include, device=dev)
    coords = torch.nonzero(include_t, as_tuple=False)
    n_nodes = int(coords.shape[0])
    if n_nodes == 0:
        return np.asarray([], dtype=phase.dtype), np.empty((0, 2), dtype=np.int64), {}
    node_ids = torch.full(include_t.shape, -1, dtype=torch.int64, device=dev)
    node_ids[tuple(coords.T)] = torch.arange(n_nodes, device=dev, dtype=torch.int64)
    neighbor_ids = []
    for axis, direction in ((2, 1), (2, -1), (1, 1), (1, -1), (0, 1), (0, -1)):
        shifted = torch.full_like(node_ids, -1)
        src = [slice(None)] * 3
        dst = [slice(None)] * 3
        if direction > 0:
            src[axis] = slice(0, -1); dst[axis] = slice(1, None)
        else:
            src[axis] = slice(1, None); dst[axis] = slice(0, -1)
        shifted[tuple(src)] = node_ids[tuple(dst)]
        neighbor_ids.append(shifted[tuple(coords.T)])
    neigh = torch.stack(neighbor_ids, dim=1)
    src_ids = torch.arange(n_nodes, device=dev, dtype=torch.int64).view(-1, 1).expand(-1, 6)
    edge_pairs = torch.stack((src_ids, neigh), dim=-1).reshape(-1, 2)
    edge_pairs = edge_pairs[edge_pairs[:, 1] >= 0]
    coords_cpu = coords.detach().cpu().numpy()
    edges_cpu = edge_pairs.detach().cpu().numpy().astype(np.int64, copy=False)
    coord_to_node = {tuple(int(v) for v in coord): int(i) for i, coord in enumerate(coords_cpu)}
    return np.asarray(phase[include], dtype=phase.dtype), edges_cpu, coord_to_node


def _gc3d_unwrap_torch(phi_w: np.ndarray, mask: np.ndarray, device: str):
    from .traditional.flowunwrap import puma

    values, edges, coord_to_node = _gc3d_torch_edges(phi_w, mask, device)
    out = np.array(phi_w, copy=True)
    if values.size == 0 or edges.size == 0:
        return out
    shifts = puma(values / (2.0 * np.pi), edges, 2.0 * np.pi)
    shifts = shifts - shifts[0]
    for index, loc in enumerate(coord_to_node.keys()):
        out[loc] = shifts[index] * (2.0 * np.pi) + out[loc]
    return out


def unwrap_phase(
    phase_wrapped: np.ndarray,
    mask: Optional[np.ndarray],
    venc,
    method: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    device: str = "auto",
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Run one traditional method and return flow plus wrap diagnostics."""
    method_token = str(method or "none").strip()
    if method_token.lower() == "none":
        raise ValueError("phase unwrapping method is disabled")
    aliases = {"gc3d": "gc3D", "lap4d": "lap4D", "nprs": "nprs"}
    method_token = aliases.get(method_token.lower(), method_token)
    if method_token not in METHODS[1:]:
        raise ValueError(f"unsupported phase unwrapping method: {method}; choose gc3D, lap4D, or nprs")

    phase = _as_phase_xyz_t3(phase_wrapped)
    x, y, z, t, _ = phase.shape
    mask4 = _as_mask_xyz_t(mask, (x, y, z, t))
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
    started = time.perf_counter()
    phase_unwrapped = np.empty_like(phase, dtype=np.float32)
    wrap_count = np.zeros_like(phase, dtype=np.int16)
    used_device = "cpu"
    gpu_device = _resolve_gpu_device(device)
    if method_token == "gc3D" and importlib.util.find_spec("maxflow") is None:
        raise RuntimeError("gc3D requires PyMaxflow; install it or choose lap4D/nprs")

    for component in range(3):
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
                from .traditional.flowunwrap import total_field_correction

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
                from .traditional.flowunwrap import total_field_correction

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
        "statistics": {
            "evaluated_voxels": int(np.count_nonzero(mask4)),
            "wrapped_voxels_any": int(np.count_nonzero(np.any(wrap_mask, axis=-1))),
            "max_abs_k": int(np.max(np.abs(wrap_count))) if wrap_count.size else 0,
            "per_component": per_component,
        },
    }


__all__ = ["METHODS", "unwrap_phase"]
