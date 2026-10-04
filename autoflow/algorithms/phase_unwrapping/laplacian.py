"""CPU 3D/4D and Torch 4D Laplacian phase recovery."""

from __future__ import annotations

import numpy as np


def lap4(in_matrix, direction, mod, real_flag=False):
    """
    Runs 4D Laplacian on input matrix.

    Args:
        in_matrix (ndarray): 3D input array.
        direction (int): Forward or inverse transform (1 or -1).
        mod (ndarray): Laplacian kernel in frequency space.
        real_flag (bool): Restrict output to real (default is False).

    Returns:
        out (ndarray): Output matrix.
    """
    sx, sy, sz, st = in_matrix.shape

    K = np.fft.fftshift(np.fft.fftn(in_matrix))

    if direction == 1:
        K = K * mod
    elif direction == -1:
        mod[mod == 0] = 1
        K = K / mod

    else:
        raise ValueError("Invalid direction. Should be 1 or -1.")

    if real_flag:
        out = np.real(np.fft.ifftn(np.fft.ifftshift(K)))
    else:
        out = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(K)))

    return out


def lap3(in_matrix, direction, mod, real_flag=False, **kwargs):
    """
    Runs 3D Laplacian on input matrix.

    Args:
        in_matrix (ndarray): 3D input array.
        direction (int): Forward or inverse transform (1 or -1).
        mod (ndarray): Laplacian kernel in frequency space.
        real_flag (bool): Restrict output to real (default is False).

    Returns:
        out (ndarray): Output matrix.
    """

    sx, sy, sz = in_matrix.shape

    K = np.fft.fftshift(np.fft.fftn(in_matrix))

    if direction == 1:
        K = K * mod
    elif direction == -1:
        mod[mod == 0] = 1
        K = K / mod
    else:
        raise ValueError("Invalid direction. Should be 1 or -1.")

    if real_flag:
        out = np.real(np.fft.ifftn(np.fft.ifftshift(K)))
    else:
        out = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(K)))

    return out


def unwrap_4D(phi_w, real_flag=True, ts=2, **kwargs):
    """
    Unwraps a 4D array. Based on the work of M. Loecher et al. (10.1002/jmri.25045)
    and the corresponding MATLAB repository: https://github.com/mloecher/4dflow-lapunwrap

    Args:
        phi_w (ndarray): Wrapped input array (-pi to pi).
        ts (int): Scales the temporal data to spatial dimensions (default is 2).
        real_flag (bool): Restrict Laplacians to real (default is True).

    Returns:
        nr (ndarray): Integer array containing the NUMBER of wraps per voxel.
                      (Note that this is not the actual unwrapped data.)
    """

    if phi_w.ndim != 4:
        raise ValueError("Input array phi_w must have 4 dimensions.")

    nr_reference = np.zeros(phi_w.shape)
    phi_w = phi_w[:phi_w.shape[0] // 2 * 2, :phi_w.shape[1] // 2 * 2, :phi_w.shape[2] // 2 * 2, :]

    sx, sy, sz, st = phi_w.shape

    X, Y, Z, T = np.meshgrid(np.arange(-sx // 2, sx // 2),
                             np.arange(-sy // 2, sy // 2),
                             np.arange(-sz // 2, sz // 2),
                             np.arange(-st // 2, st // 2),
                             indexing='ij')

    mod = 2 * np.cos(np.pi * X / sx) + 2 * np.cos(np.pi * Y / sy) + \
          2 * np.cos(np.pi * Z / sz) + ts * np.cos(np.pi * T / st) - 6 - ts

    lap_phiw = lap4(phi_w, 1, mod, real_flag)
    lap_phi = np.cos(phi_w) * lap4(np.sin(phi_w), 1, mod, real_flag) - np.sin(phi_w) * lap4(np.cos(phi_w), 1, mod,
                                                                                            real_flag)
    ilap_phidiff = lap4(lap_phi - lap_phiw, -1, mod, real_flag)
    nr = np.round(ilap_phidiff / (2 * np.pi)).astype(np.int8)

    nr_reference[:nr.shape[0], :nr.shape[1], :nr.shape[2], :] = nr

    return nr_reference


def unwrap_3D(phi_w, real_flag=True):
    """
    Unwraps a 3D array. Based on the work of M. Loecher et al. (10.1002/jmri.25045)
    and the corresponding MATLAB repository: https://github.com/mloecher/4dflow-lapunwrap

    Args:
        phi_w (ndarray): Wrapped input array (-pi to pi).
        real_flag (bool): Restrict Laplacians to real (default is True).

    Returns:
        nr (ndarray): Integer array containing the NUMBER of wraps per voxel.
                      (Note that this is not the actual unwrapped data.)
    """

    if phi_w.ndim != 3:
        raise ValueError("Input array phi_w must have 3 dimensions.")

    nr_reference = np.zeros(phi_w.shape)
    phi_w = phi_w[:phi_w.shape[0] // 2 * 2, :phi_w.shape[1] // 2 * 2, :phi_w.shape[2] // 2 * 2]

    sx, sy, sz = phi_w.shape

    X, Y, Z = np.meshgrid(np.arange(-sx // 2, sx // 2),
                          np.arange(-sy // 2, sy // 2),
                          np.arange(-sz // 2, sz // 2),
                          indexing='ij')

    mod = 2 * np.cos(np.pi * X / sx) + 2 * np.cos(np.pi * Y / sy) + \
          2 * np.cos(np.pi * Z / sz) - 6

    lap_phiw = lap3(phi_w, 1, mod, real_flag)
    lap_phi = np.cos(phi_w) * lap3(np.sin(phi_w), 1, mod, real_flag) - \
              np.sin(phi_w) * lap3(np.cos(phi_w), 1, mod, real_flag)
    ilap_phidiff = lap3(lap_phi - lap_phiw, -1, mod, real_flag)
    nr = np.round(ilap_phidiff / (2 * np.pi)).astype(np.int8)

    nr_reference[:nr.shape[0], :nr.shape[1], :nr.shape[2]] = nr

    return nr_reference


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
        from ._common import total_field_correction

        phase = total_field_correction(phase, mask.astype(np.int16))
    return np.asarray(phase, dtype=np.float32), nr
