"""Exact temporal label mode on CUDA, with bounded staging memory."""

from functools import lru_cache
import sys
import numpy as np

from ..task_control import check_cancelled


_SOURCE = r'''
extern "C" __global__ void temporal_mode(const short* source, short* output,
                                         const long long voxels, const int frames) {
    const long long voxel = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (voxel >= voxels) return;
    const long long start = voxel * frames;
    short best = 32767;
    int best_count = -1;
    for (int t = 0; t < frames; ++t) {
        const short candidate = source[start + t];
        if (best_count >= 0 && candidate == best) continue;
        int count = 0;
        for (int j = 0; j < frames; ++j) count += source[start + j] == candidate;
        if (count > best_count || (count == best_count && candidate < best)) {
            best_count = count;
            best = candidate;
        }
        if (best_count > frames / 2) break;
    }
    output[voxel] = best;
}
'''


@lru_cache(maxsize=1)
def _kernel():
    if sys.platform == 'win32':
        # Resolve the CUDA runtime bundled with the portable application first.
        try:
            import torch
        except ImportError:
            pass
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError('CUDA device unavailable')
    return cp.RawKernel(_SOURCE, 'temporal_mode', options=('--std=c++11',))


def vote_labels_cuda(labels, *, chunk_bytes=64 * 1024 ** 2):
    """Return the exact int16 mode, including background and lowest-label ties."""
    kernel = _kernel()
    import cupy as cp
    labels = np.asarray(labels, np.int16)
    if labels.ndim != 4 or not labels.size:
        raise ValueError('CUDA temporal mode requires nonempty XYZT labels')
    free, _ = cp.cuda.runtime.memGetInfo()
    budget = min(int(chunk_bytes), int(free)//8)
    row_bytes = int(np.prod(labels.shape[1:], dtype=np.int64)) * 2
    if row_bytes * 2 > free or budget < 1024:
        raise RuntimeError('Insufficient free CUDA memory for temporal voting')
    rows = max(1, budget // max(row_bytes, 1))
    result = np.empty(labels.shape[:3], np.int16)
    with cp.cuda.Stream(non_blocking=True):
        for first in range(0, labels.shape[0], rows):
            check_cancelled()
            last = min(first + rows, labels.shape[0])
            slab = np.ascontiguousarray(labels[first:last])
            source = cp.asarray(slab)
            output = cp.empty(slab.shape[:3], dtype=cp.int16)
            count = int(output.size)
            kernel(((count + 255)//256,), (256,),
                   (source, output, np.int64(count), np.int32(labels.shape[3])))
            result[first:last] = cp.asnumpy(output)
    return result
