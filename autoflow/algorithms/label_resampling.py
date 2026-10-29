"""Physical label-linear interpolation with deterministic class competition.

CUDA is preferred. The CPU fallback samples foreground channels only near
their support; neither backend interpolates integer class IDs directly.
"""
import logging
import numpy as np
from scipy.ndimage import affine_transform, find_objects
from ..task_control import check_cancelled

_LOG = logging.getLogger(__name__)
_CUPY_KERNEL = None
_TRITON_KERNEL = None
_CUDA_SOURCE = r'''
extern "C" __global__ void label_linear(
    const short* source, short* output, const long long count,
    const int sx, const int sy, const int sz, const int ty, const int tz,
    const double dx, const double dy, const double dz) {
    const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const double x = (index / ((long long)ty * tz)) * dx;
    const double y = ((index / tz) % ty) * dy;
    const double z = (index % tz) * dz;
    if (x > sx-1 || y > sy-1 || z > sz-1) { output[index] = 0; return; }
    const int ax = (int)x, ay = (int)y, az = (int)z;
    const double fx=x-ax, fy=y-ay, fz=z-az;
    short labels[8]; double weights[8];
    for (int c=0; c<8; ++c) {
        const int cx=min(ax+c/4, sx-1), cy=min(ay+(c/2)%2, sy-1), cz=min(az+c%2, sz-1);
        labels[c]=source[((long long)cx*sy+cy)*sz+cz];
        weights[c]=(c/4 ? fx : 1.-fx)*((c/2)%2 ? fy : 1.-fy)*(c%2 ? fz : 1.-fz);
    }
    float best=-1.f; short selected=32767;
    for (int c=0; c<8; ++c) {
        double score=0.;
        for (int j=0; j<8; ++j) if (labels[c]==labels[j]) score+=weights[j];
        const float rounded=(float)score;
        if (rounded>best || (rounded==best && labels[c]<selected)) { best=rounded; selected=labels[c]; }
    }
    output[index]=selected;
}
'''


def _cpu_frame(labels, scale, shape):
    best = affine_transform((labels == 0).astype(np.float32), np.diag(scale), output_shape=shape,
                            order=1, mode='constant', cval=1., prefilter=False)
    result = np.zeros(shape, np.int16)
    objects = find_objects(labels)
    for value in np.unique(labels):
        check_cancelled()
        if value == 0:
            continue
        box = objects[int(value)-1]
        lo = np.maximum(0, np.asarray([s.start for s in box])-1)
        hi = np.minimum(labels.shape, np.asarray([s.stop for s in box])+1)
        first = np.maximum(0, np.ceil((lo-1)/scale).astype(int))
        stop = np.minimum(shape, np.floor(hi/scale).astype(int)+1)
        source_slices = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        target_slices = tuple(slice(int(a), int(b)) for a, b in zip(first, stop))
        field = (labels[source_slices] == value).astype(np.float32)
        sampled = affine_transform(field, np.diag(scale), offset=first*scale-lo,
                                   output_shape=tuple(stop-first), order=1, mode='constant', cval=0., prefilter=False)
        old, selected = best[target_slices], result[target_slices]
        use = sampled > old
        old[use], selected[use] = sampled[use], value
    return result


def _cupy_frame(labels, scale, shape):
    global _CUPY_KERNEL
    import sys
    if sys.platform == 'win32':
        # The portable application ships CUDA with Torch. Loading its runtime
        # first prevents an older system toolkit from winning NVRTC discovery.
        try:
            import torch
        except ImportError:
            pass
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError('CUDA device unavailable')
    if _CUPY_KERNEL is None:
        _CUPY_KERNEL = cp.RawKernel(_CUDA_SOURCE, 'label_linear', options=('--fmad=false',))
    source = cp.asarray(np.ascontiguousarray(labels), dtype=cp.int16)
    output = cp.empty(shape, dtype=cp.int16)
    count = int(np.prod(shape, dtype=np.int64))
    _CUPY_KERNEL(((count+255)//256,), (256,), (source, output, np.int64(count),
        *map(np.int32, labels.shape), np.int32(shape[1]), np.int32(shape[2]), *map(np.float64, scale)))
    return cp.asnumpy(output)


def _make_triton_kernel():
    # Triton is optional and imported only on a CUDA path. Module-level names
    # also support older Triton versions that cannot resolve a JIT closure.
    global tl
    import triton
    import triton.language as tl
    @triton.jit
    def sample(INPUT, OUTPUT, SX: tl.constexpr, SY: tl.constexpr, SZ: tl.constexpr,
               TY: tl.constexpr, TZ: tl.constexpr, DX: tl.constexpr, DY: tl.constexpr, DZ: tl.constexpr,
               N: tl.constexpr, BLOCK: tl.constexpr):
        index = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
        valid = index < N
        x = (index//(TY*TZ)).to(tl.float64)*DX
        y = ((index//TZ)%TY).to(tl.float64)*DY
        z = (index%TZ).to(tl.float64)*DZ
        inside = valid & (x <= SX-1) & (y <= SY-1) & (z <= SZ-1)
        ax, ay, az = tl.floor(x).to(tl.int32), tl.floor(y).to(tl.int32), tl.floor(z).to(tl.int32)
        fx, fy, fz = x-ax, y-ay, z-az
        best = tl.full((BLOCK,), -1., tl.float32)
        selected = tl.full((BLOCK,), 32767, tl.int32)
        for c in tl.static_range(8):
            cx, cy, cz = tl.minimum(ax+c//4, SX-1), tl.minimum(ay+(c//2)%2, SY-1), tl.minimum(az+c%2, SZ-1)
            candidate = tl.load(INPUT+(cx*SY+cy)*SZ+cz, mask=inside, other=0).to(tl.int32)
            score = tl.full((BLOCK,), 0., tl.float64)
            for j in tl.static_range(8):
                ox, oy, oz = tl.minimum(ax+j//4, SX-1), tl.minimum(ay+(j//2)%2, SY-1), tl.minimum(az+j%2, SZ-1)
                value = tl.load(INPUT+(ox*SY+oy)*SZ+oz, mask=inside, other=0).to(tl.int32)
                wx = fx if j//4 else 1.-fx
                wy = fy if (j//2)%2 else 1.-fy
                wz = fz if j%2 else 1.-fz
                score += tl.where(value == candidate, wx*wy*wz, 0.)
            rounded = score.to(tl.float32)
            update = (rounded > best) | ((rounded == best) & (candidate < selected))
            selected, best = tl.where(update, candidate, selected), tl.maximum(best, rounded)
        tl.store(OUTPUT+index, tl.where(inside, selected, 0).to(tl.int16), mask=valid)
    return sample


def _triton_frame(labels, scale, shape):
    global _TRITON_KERNEL
    import torch, triton
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA device unavailable')
    if _TRITON_KERNEL is None:
        _TRITON_KERNEL = _make_triton_kernel()
    source = torch.as_tensor(np.ascontiguousarray(labels), device='cuda')
    output = torch.empty(shape, dtype=torch.int16, device='cuda')
    count = int(np.prod(shape, dtype=np.int64))
    _TRITON_KERNEL[(triton.cdiv(count, 128),)](source, output, *labels.shape, shape[1], shape[2],
                                            *tuple(scale), count, 128, enable_fp_fusion=False)
    return output.cpu().numpy()


def resample_label_linear(labels, spacing, target=1., shape=None, *, backend='gpu', return_info=False):
    """Resample XYZ[ T ] labels once; CUDA failure falls back to label-linear CPU.

    A score tie selects the lowest existing label ID. Source and destination
    voxel centers share their physical origin; coordinates beyond the source
    center extent are background. Velocity arrays are never passed here.
    """
    from .geometry_grid import grid_shape
    array = np.asarray(labels)
    if array.ndim not in (3, 4) or any(n < 1 for n in array.shape):
        raise ValueError('Segmentation must be a nonempty XYZ or XYZT array')
    if not np.issubdtype(array.dtype, np.integer) or np.any(array < 0) or np.any(array > 32767):
        raise ValueError('Segmentation label IDs must be integers in [0, 32767]')
    spacing = np.asarray(spacing, float).reshape(3)
    expected = grid_shape(array.shape, spacing, target)
    shape = expected if shape is None else tuple(map(int, shape))
    if shape != expected:
        raise ValueError('Target shape does not match the physical label grid')
    if backend not in {'gpu', 'cpu'}:
        raise ValueError('Label resampling backend must be gpu or cpu')
    array = np.asarray(array, np.int16)
    info = dict(method='per_label_linear_argmax', requested_backend=backend,
                spacing_mm=[float(target)]*3, native_spacing_mm=spacing.tolist(), fallback_reasons=[])
    if array.shape[:3] == shape and np.allclose(spacing, target, rtol=0, atol=1e-7):
        result, info['backend'] = array.copy(), 'identity'
    else:
        candidates = [('cupy_cuda', _cupy_frame), ('triton_cuda', _triton_frame)] if backend == 'gpu' else []
        candidates.append(('cpu_roi', _cpu_frame))
        frames = array.shape[3] if array.ndim == 4 else 1
        result = np.empty(shape+((frames,) if array.ndim == 4 else ()), np.int16)
        selected = None
        for frame in range(frames):
            check_cancelled()
            source = array[..., frame] if array.ndim == 4 else array
            remaining = candidates if selected is None else [selected]+[c for c in candidates if c != selected]
            for name, function in remaining:
                try:
                    sampled = function(source, float(target)/spacing, shape)
                except Exception as error:
                    if name == 'cpu_roi':
                        raise
                    info['fallback_reasons'].append(f'{name}: {type(error).__name__}: {error}')
                    continue
                selected = (name, function)
                if array.ndim == 4:
                    result[..., frame] = sampled
                else:
                    result[...] = sampled
                break
            info['backend'] = selected[0]
        if info['backend'] == 'cpu_roi' and backend == 'gpu':
            _LOG.info('CUDA label resampling unavailable; using CPU label-linear interpolation: %s', info['fallback_reasons'])
    return (result, info) if return_info else result
