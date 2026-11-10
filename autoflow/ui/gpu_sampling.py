"""Optional CUDA sampling for oblique display slices with a bounded volume cache."""

from collections import OrderedDict
from functools import lru_cache
import sys
import time
import weakref
import numpy as np


_SOURCE = r'''
#define ARGS(T) const T* src, const double* p, T* out, const long long n, const int sx, const int sy, const int sz, const int order
template<typename T> __device__ void sample_impl(ARGS(T)) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double x=p[i], y=p[n+i], z=p[2*n+i];
    if (!isfinite(x) || !isfinite(y) || !isfinite(z) || x<0. || y<0. || z<0. || x>sx-1 || y>sy-1 || z>sz-1) {
        out[i]=(T)0; return;
    }
    if (order==0) {
        const int a=(int)floor(x+.5), b=(int)floor(y+.5), c=(int)floor(z+.5);
        out[i]=src[((long long)a*sy+b)*sz+c]; return;
    }
    const int a=(int)floor(x), b=(int)floor(y), c=(int)floor(z);
    const double fx=x-a, fy=y-b, fz=z-c;
    double sum=0.;
    for (int dx=0;dx<2;++dx) for (int dy=0;dy<2;++dy) for (int dz=0;dz<2;++dz) {
        // SciPy's constant mode mirrors the interpolation footprint at
        // an in-bounds edge, including zero-weight NaN/Inf contributions.
        const int ix=a+dx<sx ? a+dx : max(sx-2,0);
        const int iy=b+dy<sy ? b+dy : max(sy-2,0);
        const int iz=c+dz<sz ? c+dz : max(sz-2,0);
        const double weight=(dx?fx:1.-fx)*(dy?fy:1.-fy)*(dz?fz:1.-fz);
        sum += weight*(double)src[((long long)ix*sy+iy)*sz+iz];
    }
    out[i]=(T)sum;
}
'''

_TYPES = {np.dtype(dtype).str:ctype for dtype,ctype in (
    ('float32','float'),('float64','double'),('int16','short'),('uint16','unsigned short'),
    ('int32','int'),('uint32','unsigned int'),('int8','signed char'),('uint8','unsigned char'),
    ('bool','unsigned char'),('int64','long long'),('uint64','unsigned long long'))}


@lru_cache(maxsize=12)
def _sampling_kernel(dtype):
    import cupy as cp
    ctype=_TYPES[dtype]
    entry='extern "C" __global__ void sample(ARGS('+ctype+')) { sample_impl<'+ctype+'>(src,p,out,n,sx,sy,sz,order); }'
    return cp.RawKernel(_SOURCE+entry,'sample',options=('--std=c++11','--fmad=false'))


class CudaSliceSampler:
    def __init__(self, max_bytes=128 * 1024**2, min_source_size=131_072, min_samples=16_384):
        self.max_bytes = int(max_bytes)
        self.min_source_size = int(min_source_size)
        self.min_samples = int(min_samples)
        self._volumes = OrderedDict()
        self.bytes_used = 0
        self.uploads = 0
        self.gpu_calls = 0
        self.backend = 'cpu'
        self.fallback_reason = ''
        self._retry_after = 0.

    def clear(self):
        self._volumes.clear()
        self.bytes_used = 0

    def _device_volume(self, source, cp):
        owner = source
        while isinstance(owner.base, np.ndarray):
            owner = owner.base
        key = (int(cp.cuda.runtime.getDevice()), id(owner),
               int(source.__array_interface__['data'][0]), source.shape, source.strides, source.dtype.str)
        entry = self._volumes.get(key)
        if entry is not None and entry[0]() is owner:
            self._volumes.move_to_end(key)
            return entry[1]
        if entry is not None:
            self.bytes_used -= entry[2]
            del self._volumes[key]
        nbytes = int(source.nbytes)
        while self._volumes and self.bytes_used + nbytes > self.max_bytes:
            _, old = self._volumes.popitem(last=False)
            self.bytes_used -= old[2]
        device = cp.asarray(np.ascontiguousarray(source))
        self.uploads += 1
        self._volumes[key] = (weakref.ref(owner), device, nbytes)
        self.bytes_used += nbytes
        return device

    def sample(self, source, coordinates, *, order=1):
        """Return flat NumPy pixels, or None so the caller uses its CPU path.

        Inputs are read-only display data. The owner clears this cache on a
        full refresh after an in-place data edit; phase/array/stride changes
        identify their own cached volumes. Source numerical fields are retained.
        """
        self.backend = 'cpu'
        source = np.asarray(source)
        count = int(np.asarray(coordinates[0]).size)
        if (source.ndim != 3 or source.dtype.str not in _TYPES or order not in (0, 1)
                or (order == 1 and source.dtype.kind != 'f')
                or source.size < self.min_source_size or count < self.min_samples
                or source.nbytes > self.max_bytes or time.monotonic() < self._retry_after):
            return None
        try:
            if sys.platform == 'win32':
                # Prefer the CUDA toolkit distributed beside Torch/AutoFlow.
                try:
                    import torch
                except ImportError:
                    pass
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError('CUDA device unavailable')
            free, _ = cp.cuda.runtime.memGetInfo()
            if source.nbytes > free // 4:
                self.fallback_reason = 'Insufficient free device memory for a display volume'
                return None
            # Avoid general ndimage cval checks and legacy-stream syncs for
            # this fixed constant-zero, order-0/1 display operation.
            with cp.cuda.Stream(non_blocking=True):
                device = self._device_volume(source, cp)
                coords = cp.asarray(np.stack(coordinates), dtype=cp.float64)
                sampled = cp.empty(count, dtype=device.dtype)
                kernel = _sampling_kernel(source.dtype.str)
                kernel(((count+255)//256,), (256,), (device, coords, sampled, np.int64(count),
                    *map(np.int32, source.shape), np.int32(order)))
                result = cp.asnumpy(sampled)
            self.gpu_calls += 1
            self.backend = 'cupy_cuda'
            self.fallback_reason = ''
            return result
        except Exception as error:
            self.clear()
            self.fallback_reason = f'{type(error).__name__}: {error}'
            self._retry_after = time.monotonic() + 30.
            return None
