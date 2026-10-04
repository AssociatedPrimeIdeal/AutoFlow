"""Phase-unwrapping entry points with separate backend owners."""

from .backends import (
    METHODS,
    _LEARNED_BACKEND_PACKAGES,
    _LEARNED_BACKEND_REQUIREMENTS,
    _canonical_method,
    mask_sources_for_method,
    resolve_mask_source,
    backend_available,
    _load_external_backend,
    _backend_device,
    _resolve_gpu_device,
)

from ._common import (
    _backend_spacing,
    _backend_weight_mask,
    _deep_backend_params,
    _as_phase_xyz_t3,
    _as_mask_xyz_t,
    _as_mask_or_weight_xyz_t,
    _coerce_venc,
)

from .pudip import (
    _pudip_unwrap,
)

from .gust import (
    _gust_unwrap,
)

from .laplacian import (
    _fftshift_torch,
    _ifftshift_torch,
    _lap4d_gpu,
)

from .nprs import (
    _torch_pad_or_crop,
    _torch_fft_resample,
    _nprs_gpu_fft,
)

from .graphcut import (
    _gc3d_torch_edges,
    _gc3d_unwrap_torch,
)

from .engine import (
    unwrap_phase,
)

from .cpu import unwrap_data

__all__ = ["METHODS", "backend_available", "unwrap_phase"]
