"""Bounded elementwise operations with the original NumPy expression order.

Only independent rows are parallelized. No reduction, precision, phase-boundary
or VENC semantics change. Workers inherit the current cancellation token.
"""

from concurrent.futures import ThreadPoolExecutor
import os

import numpy as np

from ..task_control import check_cancelled, current_cancellation_token


def _venc3(value):
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    if result.size == 1:
        result = np.repeat(result, 3)
    if result.size != 3 or not np.all(np.isfinite(result)) or np.any(result <= 0):
        raise ValueError("VENC must contain one or three positive finite values")
    return result.reshape(1, 3)


def _setting(name, default, minimum, maximum):
    try:
        return max(minimum, min(maximum, int(os.environ.get(name, default))))
    except (TypeError, ValueError):
        return default


def _rows(count, operation):
    check_cancelled()
    chunk = _setting("AUTOFLOW_POINTWISE_BLOCK_ROWS", 262144, 4096, 1048576)
    workers = min(_setting("AUTOFLOW_POINTWISE_WORKERS", 8, 1, 32), os.cpu_count() or 1)
    token = current_cancellation_token()

    def apply(bounds):
        if token is not None:
            token.check()
        operation(*bounds)
        if token is not None:
            token.check()

    bounds = [(start, min(start + chunk, count)) for start in range(0, count, chunk)]
    if count < 4 * chunk or workers == 1:
        for bound in bounds:
            apply(bound)
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="autoflow-pointwise") as pool:
            for _ in pool.map(apply, bounds):
                check_cancelled()
    check_cancelled()


def wrapped_phase_from_flow(flow, venc):
    array = np.asarray(flow)
    if array.shape[-1:] != (3,):
        raise ValueError("flow must end in three velocity components")
    field = array.reshape(-1, 3)
    denominator = np.maximum(_venc3(venc), 1e-6)
    out = np.empty(field.shape, dtype=np.float32)

    def block(a, b):
        out[a:b] = np.angle(np.exp(1j * field[a:b] * np.pi / denominator)).astype(np.float32)

    _rows(len(field), block)
    return out.reshape(array.shape)


def synthetic_complex_from_mag_flow(mag, flow, venc):
    array = np.asarray(flow)
    field = array.reshape(-1, 3)
    ref = np.asarray(mag, dtype=np.float32).reshape(-1)
    if len(ref) != len(field):
        raise ValueError("magnitude and flow shapes do not match")
    denominator = _venc3(venc)
    out = np.empty((len(field), 4), dtype=np.complex64)

    def block(a, b):
        out[a:b, 0] = ref[a:b].astype(np.complex64)
        phase = np.pi * field[a:b] / denominator
        out[a:b, 1:4] = ref[a:b, None] * np.exp(1j * phase)

    _rows(len(field), block)
    return out.reshape(array.shape[:-1] + (4,))


def apply_complex_correction(array, correction):
    result = np.asarray(array, dtype=np.complex64).copy()
    corr = np.asarray(correction, dtype=np.float32)
    if corr.shape != result.shape[:-1] + (3,):
        # In particular retain NumPy's original singleton-time broadcasting.
        check_cancelled()
        result[..., 1:4] *= np.exp(-1j * corr)
        check_cancelled()
        return result
    field = result.reshape(-1, 4)
    values = corr.reshape(-1, 3)

    def block(a, b):
        field[a:b, 1:4] *= np.exp(-1j * values[a:b])

    _rows(len(field), block)
    return result


def velocity_from_complex(array, venc):
    source = np.asarray(array)
    if source.shape[-1:] != (4,):
        raise ValueError("complex flow input must have four components")
    field = source.reshape(-1, 4)
    numerator = _venc3(venc)
    out = np.empty((len(field), 3), dtype=np.float32)

    def block(a, b):
        phase = np.angle(field[a:b, 1:4] * np.conj(field[a:b, 0:1]))
        out[a:b] = (phase / np.pi) * numerator

    _rows(len(field), block)
    return out.reshape(source.shape[:-1] + (3,))
