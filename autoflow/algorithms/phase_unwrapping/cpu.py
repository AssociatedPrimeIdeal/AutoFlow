"""CPU method dispatch and velocity/wrap-count return contract.

Derived from the MIT-licensed PUDIP-Flow TradMethod sources.
"""

import numpy as np
from tqdm import tqdm
from ._common import total_field_correction
from .brute import brute_unwrap
from .graphcut import gc3D_unwrap, gc4D_unwrap
from .laplacian import unwrap_3D, unwrap_4D
from .nprs import unwrap_nprs


def unwrap_data(phi_w, mode='3D', real_flag=True, venc=None, full=False, tfc=False, mask=None, verbose=True, **kwargs):
    """
    Unwraps a 4D array .

    Args:
        phi_w (ndarray): Wrapped input array (-pi to pi).
        real_flag (bool): Restrict Laplacians to real (default is True).
        mode (string): '3D' or '4D' activates corresponding Laplacian approach
        venc (float or None): if float, then returns velocity=phase/np.pi*venc
        full (bool): if True, returns phi_u & nr (unwrapped phase & number of wraps)
        ts (float): weighting bewteen temporal and spatial scales when using a 4D approach

    Returns:
        nr (ndarray): Integer array containing the numer of wraps per voxel.
                      (Note that this is not the actual unwrapped data.)
        phi_u (ndarray): Float array containing the unwrapped phase.
    """
    try:
        if mask == None:
            mask = np.ones_like(phi_w)
    except:
        pass
    if phi_w.ndim != 4:
        raise ValueError("Input array must have 4 dimensions.")
    if mode == 'lap3D':
        nr = np.zeros(phi_w.shape)
        for time in tqdm(range(nr.shape[-1]), disable=not verbose, leave=False, desc="Unwrapping..."):
            nr[..., time] = unwrap_3D(phi_w[..., time], real_flag)
        phi_u = phi_w + 2 * np.pi * nr
        if tfc:
            phi_u = total_field_correction(phi_u, mask)
        if venc:
            phi_u *= venc / np.pi
        if full:
            return phi_u, nr
        else:
            return phi_u

    elif mode == 'lap4D':
        for time in tqdm(range(1), disable=not verbose, leave=False, desc="Unwrapping..."):
            nr = unwrap_4D(phi_w, real_flag, **kwargs)
        phi_u = phi_w + 2 * np.pi * nr
        if tfc:
            phi_u = total_field_correction(phi_u, mask)
        if venc:
            phi_u *= venc / np.pi
        if full:
            return phi_u, nr
        else:
            return phi_u

    elif mode == 'nprs':
        phi_u = np.zeros(phi_w.shape)
        if type(mask).__module__ == np.__name__:
            try:
                if mask.shape == phi_w.shape:
                    pass
            except:
                raise Exception('mask and to_unwrap shapes must match')

        for time in tqdm(range(phi_u.shape[-1]), disable=not verbose, leave=False, desc="Unwrapping..."):
            phi_u[..., time] = unwrap_nprs(phi_w[..., time], mask[..., time], **kwargs)
        nr = np.round((phi_u - phi_w) / (2 * np.pi))
        if tfc:
            phi_u = total_field_correction(phi_u, mask)
        if venc:
            phi_u *= venc / np.pi
        if full:
            return np.ma.getdata(phi_u) * mask, np.ma.getdata(nr) * mask
        else:
            return np.ma.getdata(phi_u) * mask

    elif mode == 'brute':
        phi_u = np.zeros(phi_w.shape)
        if type(mask).__module__ == np.__name__:
            try:
                if mask.shape == phi_w.shape:
                    pass
            except:
                raise Exception('mask and to_unwrap shapes must match')

        phi_u = brute_unwrap(phi_w, mask, **kwargs)
        nr = np.round((phi_u - phi_w) / (2 * np.pi))

        if venc:
            phi_u *= venc / np.pi
        if full:
            return np.ma.getdata(phi_u) * mask, np.ma.getdata(phi_u) * mask
        else:
            return np.ma.getdata(phi_u) * mask

    elif mode == 'gc3D':
        phi_u = np.zeros(phi_w.shape)
        # ``gc3D_unwrap`` updates each time-slice in place.  Keep the wrapped
        # input separately so ``nr`` reports the actual integer wraps instead
        # of always returning zeros after the in-place update.
        phi_wrapped = np.array(phi_w, copy=True)
        if type(mask).__module__ == np.__name__:
            try:
                if mask.shape == phi_w.shape:
                    pass
            except:
                raise Exception('mask and to_unwrap shapes must match')

        for time in tqdm(range(phi_u.shape[-1]), disable=not verbose, leave=False, desc="Unwrapping..."):
            phi_u[..., time] = gc3D_unwrap(phi_w[..., time], mask[..., time], **kwargs)
        nr = np.round((phi_u - phi_wrapped) / (2 * np.pi))
        if tfc:
            phi_u = total_field_correction(phi_u, mask)
        if venc:
            phi_u *= venc / np.pi
        if full:
            return np.ma.getdata(phi_u) * mask, np.ma.getdata(nr) * mask
        else:
            return np.ma.getdata(phi_u) * mask

    elif mode == 'gc4D':
        phi_u = np.zeros(phi_w.shape)
        if type(mask).__module__ == np.__name__:
            try:
                if mask.shape == phi_w.shape:
                    pass
            except:
                raise Exception('mask and to_unwrap shapes must match')

        for time in tqdm(range(1), disable=not verbose, leave=False, desc="Unwrapping..."):
            phi_u = gc4D_unwrap(phi_w, mask, **kwargs)
        nr = np.round((phi_u - phi_w) / (2 * np.pi))

        if tfc:
            phi_u = total_field_correction(phi_u, mask)
        if venc:
            phi_u *= venc / np.pi
        if full:
            return np.ma.getdata(phi_u) * mask, np.ma.getdata(nr) * mask
        else:
            return np.ma.getdata(phi_u) * mask

    else:
        raise ValueError("Input mode must be either 3D or 4D")
