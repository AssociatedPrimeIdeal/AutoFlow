"""Optional, bounded CUDA workspaces for velocity reconstruction and display."""

from functools import lru_cache
import sys
import numpy as np

from ..task_control import check_cancelled, report_progress


def _cuda():
    if sys.platform == "win32":
        # Make the portable application's bundled runtime visible first.
        try:
            import torch
        except ImportError:
            pass
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("CUDA device unavailable")
    return cp


@lru_cache(maxsize=1)
def _dual_kernel():
    cp = _cuda()
    return cp.ElementwiseKernel(
        "float32 low, float32 high, float32 venc, float32 ratio1, float32 ratio2",
        "float32 result, float32 correction",
        r'''
        const float difference = dv_sub(high, low);
        const float th1 = dv_mul(venc, dv_sub(1.f, ratio1));
        const float th2 = dv_mul(venc, dv_add(3.f, ratio1));
        const float th3 = dv_mul(venc, dv_sub(3.f, ratio2));
        const float th4 = dv_mul(venc, dv_add(5.f, ratio2));
        correction = 0.f;
        if (dv_le(th1, difference) && dv_le(difference, th2)) correction = dv_mul(2.f, venc);
        if (dv_le(dv_neg(th2), difference) && dv_le(difference, dv_neg(th1))) correction = dv_mul(-2.f, venc);
        if (dv_le(th3, difference) && dv_le(difference, th4)) correction = dv_mul(4.f, venc);
        if (dv_le(dv_neg(th4), difference) && dv_le(difference, dv_neg(th3))) correction = dv_mul(-4.f, venc);
        result = dv_add(low, correction);
        ''', "autoflow_dual_venc", options=("--fmad=false",), preamble=r'''
        // CuPy enables FTZ globally. Explicit PTX preserves NumPy's subnormal
        // arithmetic, including comparisons at overlapping zero thresholds.
        __device__ __forceinline__ float dv_add(float a, float b) {
            float r; asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r;
        }
        __device__ __forceinline__ float dv_sub(float a, float b) {
            float r; asm("sub.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r;
        }
        __device__ __forceinline__ float dv_mul(float a, float b) {
            float r; asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r;
        }
        __device__ __forceinline__ float dv_neg(float a) {
            return __uint_as_float(__float_as_uint(a) ^ 0x80000000u);
        }
        __device__ __forceinline__ bool dv_le(float a, float b) {
            unsigned int x=__float_as_uint(a), y=__float_as_uint(b);
            if ((x & 0x7fffffffu) > 0x7f800000u || (y & 0x7fffffffu) > 0x7f800000u) return false;
            if ((x & 0x7fffffffu) == 0) x=0;
            if ((y & 0x7fffffffu) == 0) y=0;
            x ^= (x & 0x80000000u) ? 0xffffffffu : 0x80000000u;
            y ^= (y & 0x80000000u) ? 0xffffffffu : 0x80000000u;
            return x <= y;
        }
        ''')


def reconstruct_dual_venc_cuda(low, high, venc, ratio1, ratio2, *, chunk_bytes=64 * 1024**2):
    """Reconstruct XYZT3 fields, preserving CPU threshold overlap precedence.

    Return a small per-component sample of observed shifts for the existing
    metadata formatter. Inputs remain host owned and are never modified.
    """
    cp = _cuda()
    kernel = _dual_kernel()
    free, _ = cp.cuda.runtime.memGetInfo()
    budget = min(int(chunk_bytes), int(free) // 12)
    row_bytes = int(np.prod(low.shape[1:], dtype=np.int64)) * 4
    if budget < row_bytes or budget < 1024:
        raise RuntimeError("Insufficient free CUDA memory for dual-VENC reconstruction")
    rows = max(1, budget // row_bytes)
    # NumPy's alias expressions retain the input's preferred memory order.
    # Preserve it too: downstream pressure reductions can otherwise differ in
    # their last bits even when every reconstructed velocity is identical.
    result = np.empty_like(low, dtype=np.float32, order="K")
    observed = np.zeros((3, 5), bool)
    candidates = np.asarray(venc, np.float32)[:, None] * np.array([0, 2, -2, 4, -4], np.float32)
    with cp.cuda.Stream(non_blocking=True):
        venc_gpu, ratio1_gpu, ratio2_gpu = (cp.asarray(value, dtype=cp.float32)
                                           for value in (venc, ratio1, ratio2))
        for first in range(0, low.shape[0], rows):
            check_cancelled()
            last = min(first + rows, low.shape[0])
            low_gpu = cp.asarray(np.ascontiguousarray(low[first:last], dtype=np.float32))
            high_gpu = cp.asarray(np.ascontiguousarray(high[first:last], dtype=np.float32))
            output, shifts = kernel(low_gpu, high_gpu, venc_gpu, ratio1_gpu, ratio2_gpu)
            # Transfer only fifteen booleans, rather than the full shift field.
            found = cp.stack([cp.any(shifts[..., component] == candidate)
                              for component in range(3) for candidate in candidates[component]])
            observed |= cp.asnumpy(found).reshape(3, 5)
            result[first:last] = cp.asnumpy(output)
            del low_gpu, high_gpu, output, shifts, found
            report_progress({"stage": "dual_venc", "current": last, "total": low.shape[0],
                             "message": "Reconstructing dual-VENC velocity"})
    samples = np.where(observed, candidates, np.float32(np.nan)).reshape(3, 1, 1, 1, 5)
    return result, samples


def coherence_score_cuda(magnitude, velocity, *, workspace_bytes=512 * 1024**2):
    """Return the temporal 90th-percentile score with a one-voxel slab halo.

    Temporal filtering remains cyclic, spatial filtering reflective. Smoothing
    and the display ramp use the shared CPU path after this small XYZ result.
    """
    cp = _cuda()
    from cupyx.scipy.ndimage import uniform_filter
    free, _ = cp.cuda.runtime.memGetInfo()
    budget = min(int(workspace_bytes), int(free) // 3)
    # Covers input conversion, filters, norms and percentile scratch, including
    # float64 caller inputs. Slabs keep the complete cardiac cycle together.
    row_bytes = int(np.prod(velocity.shape[1:4], dtype=np.int64)) * 128
    rows = budget // max(row_bytes, 1) - 2
    if rows < 1 or budget < 1024:
        raise RuntimeError("Insufficient free CUDA memory for coherence filtering")
    rows = min(rows, velocity.shape[0])
    score = np.empty(velocity.shape[:3], np.float32)
    modes = ("reflect", "reflect", "reflect", "wrap")
    with cp.cuda.Stream(non_blocking=True):
        for first in range(0, velocity.shape[0], rows):
            check_cancelled()
            last = min(first + rows, velocity.shape[0])
            start, stop = max(0, first - 1), min(velocity.shape[0], last + 1)
            v = cp.nan_to_num(cp.asarray(np.ascontiguousarray(velocity[start:stop])),
                              nan=0., posinf=0., neginf=0.).astype(cp.float32, copy=False)
            m = cp.maximum(cp.nan_to_num(cp.asarray(np.ascontiguousarray(magnitude[start:stop])),
                                         nan=0., posinf=0., neginf=0.), 0).astype(cp.float32, copy=False)
            denominator = uniform_filter(m * cp.linalg.norm(v, axis=-1), size=(3, 3, 3, 3), mode=modes)
            numerator = cp.zeros_like(denominator)
            for component in range(3):
                check_cancelled()
                moment = uniform_filter(m * v[..., component], size=(3, 3, 3, 3), mode=modes)
                numerator += moment * moment
            cp.sqrt(numerator, out=numerator)
            valid = denominator > 1e-12
            denominator[~valid] = 1
            numerator /= denominator
            numerator[~valid] = 0
            cp.clip(numerator, 0, 1, out=numerator)
            slab = cp.percentile(numerator[first - start:last - start], 90, axis=3).astype(cp.float32)
            score[first:last] = cp.asnumpy(slab)
            del v, m, denominator, numerator, moment, valid, slab
            report_progress({"stage": "noise_coherence", "current": last, "total": velocity.shape[0],
                             "message": "Calculating PC-MRA display coherence"})
    return score
