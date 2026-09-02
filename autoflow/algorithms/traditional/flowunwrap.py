import numpy as np
from skimage.restoration import unwrap_phase
try:  # package import
    from .fourierOperators import *
except ImportError:  # pragma: no cover - backwards-compatible script import
    from fourierOperators import *
from tqdm import tqdm

try:
    import maxflow
except ImportError:
    print("PyMaxflow not found, graph-cut unwrapping will fail")


######################################################################################################
#                                        Example Usage                                               #
#         phi_w = [88, 96, 19, 21] # wrapped phase data [X,Y,Z,cardiac_phases] E [-pi,pi]            #
#                    phi_u3D=unwrap_data(phi_w, mode='3D') # 3D Laplacian unwrapping                 #
# phi_u4D=unwrap_data(phi_w, mode='4D', venc=0.5, ts=2) # 4D Laplacian unwrapping (with VENC mult.)  #
######################################################################################################

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


def upsample(to_resample, shape_to_resample, upsampling_factor):
    """
    This function upsamples a given multidimensional array 'to_resample' by a specified 'upsampling_factor'.
    The upsampling is performed in the Fourier domain. The array is first transformed into the Fourier space using an FFT,
    then it is padded to the desired shape based on the 'shape_to_resample' divided by the 'upsampling_factor'.
    After padding, an inverse FFT is used to transform the array back to the spatial domain. The output is scaled by the square root
    of the product of the desired shape dimensions to normalize the transformation's effect on the amplitude of the array.

    Parameters:
    - to_resample (numpy.ndarray): The array to be upsampled.
    - shape_to_resample (tuple of int): The target shape after upsampling.
    - upsampling_factor (int or float): The factor by which to upsample the array.

    Returns:
    - numpy.ndarray: The upsampled array, with the size specified by 'shape_to_resample'.
    """
    to_resample = np.fft.fftshift(np.fft.fftn(to_resample, norm='ortho'))
    to_resample = pad_array(to_resample, tuple((np.array(shape_to_resample) / upsampling_factor).astype('int')))
    to_resample = np.fft.ifftn(np.fft.ifftshift(to_resample), norm='forward') / np.sqrt(
        np.prod(np.array(shape_to_resample)))
    return to_resample


def downsample(to_resample, shape_to_resample, upsampling_factor):
    """
    This function downsamples a given multidimensional array 'to_resample' by reducing its dimensions based on a
    specified 'upsampling_factor'. The downsampling process involves first transforming the array into the Fourier domain using an FFT,
    then cropping the Fourier-transformed array to reduce its size inversely proportional to the 'upsampling_factor'.
    After cropping, an inverse FFT is applied to transform the array back to the spatial domain. The final output is normalized by the square root
    of the product of the dimensions of the resulting downsampled array to maintain the amplitude scale.

    Parameters:
    - to_resample (numpy.ndarray): The array to be downsampled.
    - shape_to_resample (tuple of int): The target shape to which the array should approximate after downsampling.
    - upsampling_factor (int or float): The factor by which the array was originally upsampled.

    Returns:
    - numpy.ndarray: The downsampled array, approximately the size specified by 'shape_to_resample'.
    """
    to_resample = np.fft.fftshift(np.fft.fftn(to_resample, norm='ortho'))
    to_resample = pad_array(to_resample, tuple((-np.array(shape_to_resample) / upsampling_factor).astype('int')))
    to_resample = np.fft.ifftn(np.fft.ifftshift(to_resample), norm='forward') / np.sqrt(
        np.prod(np.array(to_resample.shape)))
    return to_resample


def unwrap_nprs(to_unwrap, mask=None, upsampling_factor=1, pi_unwrap=True, auto_crop=False, n_voxels=5):
    """
    This function uses the nprs (non-continuous path with reliability sorting) algorithm for phase unwrapping implemented
    on the scipy skimage.restauration based on the work of Herraez et al. (10.1364/AO.41.007437),
    applicable to arrays with up to 3 dimensions. It handles optional upsampling for improved accuracy and can work with
    or without a mask. The function also supports auto-cropping to focus on regions defined by the mask and can normalize
    phase values to pi if requested.

    Parameters:
    - to_unwrap (numpy.ndarray): The array containing phase data to be unwrapped.
    - mask (numpy.ndarray, optional): A binary mask array that matches the dimensions of 'to_unwrap'. It specifies the regions
      over which the unwrapping and calculations should be focused.
    - upsampling_factor (int or float): Specifies the factor by which 'to_unwrap' should be upsampled before unwrapping.
      Must be >=1. Default is 1, which means no upsampling.
    - pi_unwrap (bool): If True, normalizes the unwrapped phase values to be between -pi and pi after unwrapping.
    - auto_crop (bool): If True, the function automatically crops 'to_unwrap' to the bounds defined by 'mask' before processing.
    - n_voxels (int): The number of voxels to expand the crop beyond the true edges of the mask. This parameter is used only if 'auto_crop' is True.

    Returns:
    - numpy.ndarray: The unwrapped phase array. If 'auto_crop' is True, the output size matches the original 'to_unwrap' size;
      otherwise, it matches the possibly cropped or upsampled size.

    Raises:
    - Exception: If the upsampling factor is less than 1, or if 'mask' and 'to_unwrap' do not match in shape, or if 'to_unwrap'
      has more than 3 dimensions.
    """
    if upsampling_factor < 1:
        raise Exception('upsampling factor must be >=1, to deactivate: set = 1')
    elif upsampling_factor == 1:
        upsampling_factor = 0
    if upsampling_factor > 1:
        upsampling_factor = 2 / (upsampling_factor - 1)

    try:
        if mask.shape == to_unwrap.shape:
            pass
    except:
        raise Exception('mask and to_unwrap shapes must match')

    if to_unwrap.ndim > 3:
        raise Exception('nprs algorithm only implemented for arrays with maximum 3 dimensions.')

    if auto_crop:
        loc_where = np.where(mask > 0.5)
        target_slice = [(max(np.min(A) - n_voxels, 0), np.max(A) + n_voxels) for A in loc_where]
        target_slice = tuple(
            [slice(start_pad, end_pad) for ((start_pad, end_pad), dim) in zip(target_slice, to_unwrap.shape)])

        reference_to_unwrap = to_unwrap.copy()
        reference_mask = mask.copy()

        to_unwrap = to_unwrap[target_slice]
        mask = mask[target_slice]

    shape_to_unwrap = to_unwrap.shape
    wrapped_data = to_unwrap.copy()

    if upsampling_factor == 0:
        to_unwrap = np.ma.masked_array(to_unwrap, mask=(1 - mask.astype('int')))
        res = unwrap_phase(to_unwrap)
    else:
        upmask = upsample(mask, shape_to_unwrap, upsampling_factor)
        upmask = np.abs(upmask)
        upmask[upmask >= 0.5] = 1
        upmask[upmask < 0.5] = 0

        to_unwrap = np.exp(1j * to_unwrap)
        to_unwrap = upsample(to_unwrap, shape_to_unwrap, upsampling_factor)
        to_unwrap = np.angle(to_unwrap)

        to_unwrap = np.ma.masked_array(to_unwrap, mask=(1 - upmask.astype('int')))
        res = unwrap_phase(to_unwrap)

        normalization = max(res.max(), np.abs(res.min()))
        res = np.exp(1j * res / normalization * np.pi)

        res = downsample(res, shape_to_unwrap, upsampling_factor)
        res = np.angle(res)

        res = res / np.pi * normalization

    if auto_crop:
        if pi_unwrap == True:
            rounding = np.round((res - wrapped_data) / (2 * np.pi))
            res = wrapped_data + 2 * np.pi * rounding
            res_reshape = np.zeros_like(reference_to_unwrap)
            res_reshape[target_slice] = res
            return res_reshape
        else:
            res_reshape = np.zeros_like(reference_to_unwrap)
            res_reshape[target_slice] = res
            return res_reshape
    else:
        if pi_unwrap == True:
            rounding = np.round((res - wrapped_data) / (2 * np.pi))
            res = wrapped_data + 2 * np.pi * rounding
            return res
        else:
            return res


def compute_discontinuities(array, mode='3D', axis=0):
    """
    This function computes the magnitude of discontinuities in a given array by comparing each element with its neighbors.
    The function can operate in either 1D or 3D mode. In 1D mode, it calculates the discontinuity by comparing each element
    with its immediate neighbor along a specified axis. In 3D mode, it considers neighbors along all three axes.

    Parameters:
    - array (numpy.ndarray): The input array for which discontinuities are to be calculated.
    - mode (str): The mode of operation, which can be '1D' or '3D'. Default is '3D'.
    - axis (int): The axis along which to compute discontinuities if in '1D' mode. Default is 0.

    Returns:
    - numpy.ndarray: An array of the same shape as the input 'array', containing the maximum discontinuity
      at each point, calculated as the maximum absolute difference between an element and its neighbors.

    Raises:
    - ValueError: If an invalid mode is provided.
    """
    if mode == '1D':
        a1 = np.abs(array - np.roll(array, -1, axis=axis))
        a2 = np.abs(array - np.roll(array, 1, axis=axis))
        A = np.ma.concatenate((a1[..., np.newaxis], a2[..., np.newaxis]), axis=-1)
        return np.max(A, axis=-1)

    if mode == '3D':
        a1 = np.abs(array - np.roll(array, -1, axis=0))
        a2 = np.abs(array - np.roll(array, 1, axis=0))
        a3 = np.abs(array - np.roll(array, -1, axis=1))
        a4 = np.abs(array - np.roll(array, 1, axis=1))
        a5 = np.abs(array - np.roll(array, -1, axis=2))
        a6 = np.abs(array - np.roll(array, 1, axis=2))
        A = np.ma.concatenate((a1[..., np.newaxis], a2[..., np.newaxis], a3[..., np.newaxis], a4[..., np.newaxis],
                               a5[..., np.newaxis], a6[..., np.newaxis]), axis=-1)
        return np.max(A, axis=-1)


def compute_local_gradient(array, loc, wrap=0, mode_loc='3D', alpha=0.5):
    """
    This function computes the local gradient at a specified location within an array. The gradient can be calculated
    either in a 3D or 4D mode, which influences how spatial and potentially temporal differences are considered.
    The function handles wrap values, which are added to the local point before computing differences, allowing for
    handling of wrapped or cyclic data.

    Parameters:
    - array (numpy.ndarray): The multidimensional array from which to compute the gradient.
    - loc (tuple): The specific location (index) within the array for which the gradient is calculated.
    - wrap (int or float, optional): A value to be added to the array value at the specified location, useful for
      handling wrapped data. Default is 0.
    - mode_loc (str): The mode of calculation, '3D' or '4D'. '3D' calculates spatial gradients in three dimensions,
      while '4D' includes a temporal or additional spatial component.
    - alpha (float, optional): A blending factor used in '4D' mode to weight the spatial and temporal components of the gradient.
      Default is 0.5, giving equal weight to both components.

    Returns:
    - float: The computed gradient at the specified location. In '3D' mode, it is the mean of the absolute differences
      between the local point and its immediate neighbors. In '4D' mode, it combines spatial and temporal gradients according
      to the specified alpha.

    Notes:
    - Ensure that the location and dimensions specified are within the bounds of the array to avoid indexing errors.
    - NaN values are handled by treating them as missing data and excluding them from the mean computation.
    """
    local_point = array[tuple(loc)] + wrap

    if mode_loc == '3D':
        grad = np.nanmean(np.ma.abs([local_point - array[loc[0] - 1, loc[1], loc[2], loc[3]], \
                                     local_point - array[loc[0] + 1, loc[1], loc[2], loc[3]], \
                                     local_point - array[loc[0], loc[1] - 1, loc[2], loc[3]], \
                                     local_point - array[loc[0], loc[1] + 1, loc[2], loc[3]], \
                                     local_point - array[loc[0], loc[1], loc[2] - 1, loc[3]], \
                                     local_point - array[loc[0], loc[1], loc[2] + 1, loc[3]]]))
    if mode_loc == '4D':
        grad_space = np.nanmean(np.ma.abs([local_point - array[loc[0] - 1, loc[1], loc[2], loc[3]], \
                                           local_point - array[loc[0] + 1, loc[1], loc[2], loc[3]], \
                                           local_point - array[loc[0], loc[1] - 1, loc[2], loc[3]], \
                                           local_point - array[loc[0], loc[1] + 1, loc[2], loc[3]], \
                                           local_point - array[loc[0], loc[1], loc[2] - 1, loc[3]], \
                                           local_point - array[loc[0], loc[1], loc[2] + 1, loc[3]]]))

        if loc[3] == array.shape[-1] - 1:
            grad_time = np.nanmean(np.ma.abs([local_point - array[loc[0], loc[1], loc[2], 0], \
                                              local_point - array[loc[0], loc[1], loc[2], loc[3] - 1]]))
        else:
            grad_time = np.nanmean(np.ma.abs([local_point - array[loc[0], loc[1], loc[2], loc[3] + 1], \
                                              local_point - array[loc[0], loc[1], loc[2], loc[3] - 1]]))

        grad = grad_space * alpha + grad_time * (1 - alpha)

    return grad


def brute_unwrap(to_unwrap, mask, n_iter=5, verbose=True, **kwargs):
    """
    This function performs a brute force unwrapping of phase data in a multi-dimensional array, attempting to minimize
    discontinuities by iteratively adjusting values based on local gradients. The method pads the array and mask to handle
    edge cases and uses a mask to focus unwrapping efforts only on relevant areas. The function leverages local gradient
    computations to decide on the best wrap adjustments.

    Parameters:
    - to_unwrap (numpy.ndarray): The multi-dimensional array containing phase data to be unwrapped.
    - mask (numpy.ndarray): A binary mask array indicating the regions of interest for phase unwrapping.
    - n_iter (int, optional): The number of iterations to attempt unwrapping. Default is 5.
    - verbose (bool, optional): If True, progress bars and updates will be shown during the unwrapping process. Default is True.
    - **kwargs: Additional keyword arguments passed to the `compute_local_gradient` function.

    Returns:
    - numpy.ndarray: The unwrapped array, with dimensions reduced to exclude the padding.

    Notes:
    - Padding is added to `to_unwrap` and `mask` to handle boundary conditions effectively during gradient computations.
    - The function uses a brute-force approach to iteratively adjust phase values, assessing each potential modification by
      calculating its effect on the local gradient and choosing the modification that results in a lower gradient magnitude.
    - Discontinuities are computed to identify target voxels for potential unwrapping adjustments.
    - The process is controlled by a set number of iterations or until no further wraps are adjusted.
    """
    mask = np.pad(mask, 1, constant_values=0)
    to_unwrap = np.pad(to_unwrap, 1)

    pbar = tqdm(range(n_iter), disable=not verbose, leave=False, desc=f"Unwrapping voxels")
    for i in pbar:

        disc = compute_discontinuities(np.ma.masked_array(to_unwrap, 1 - mask))

        target = np.argwhere(np.round(disc * mask / np.pi / 2))

        array = np.ma.masked_array(to_unwrap, 1 - mask)

        n_wraps = 0
        for cell in tqdm(target, disable=not verbose, leave=False, desc=f"Iterating over voxels"):
            tmp_gradient = compute_local_gradient(array, cell, **kwargs)
            for wrap in [-2 * np.pi, 2 * np.pi]:
                if tmp_gradient > compute_local_gradient(array, cell, wrap, **kwargs):
                    array[tuple(cell)] += wrap
                    n_wraps += 1
                else:
                    pass

        to_unwrap = array.copy()

        pbar.set_description(f'Unwrapped voxels {n_wraps}')

        if n_wraps == 0:
            break

    return to_unwrap[1:-1, 1:-1, 1:-1, 1:-1]


def get_3d_masked_edges(masked_array):
    """
    This function processes a 3D masked array to extract nodes and edges based on the masked values. It creates a mapping
    from each masked coordinate to a unique node ID and then identifies edges between adjacent nodes based on a 6-connectivity
    model (neighbors in the cardinal directions). The function is useful for converting a 3D volume into a graph representation,
    where each masked voxel represents a node and edges represent direct adjacency between these voxels.

    Parameters:
    - masked_array (numpy.ndarray): A 3D boolean array where True values indicate masked points that will be converted to nodes.

    Returns:
    - tuple:
        - numpy.ndarray: Array of values from the masked positions of the input array.
        - numpy.ndarray: Array of edges, where each edge is represented by a pair of node IDs indicating connectivity.
        - dict: A dictionary mapping from 3D coordinates to node IDs.

    Notes:
    - The function assumes the input array is strictly three-dimensional and boolean.
    - Edges are only established between directly adjacent nodes in the 3D space along the axes, not diagonally.
    """
    # Create a dictionary to map coordinates to node IDs
    coord_to_node = {}
    node_id = 0

    nodes = []
    values = []
    # Iterate through the 3D masked array
    for z in range(masked_array.shape[0]):
        for y in range(masked_array.shape[1]):
            for x in range(masked_array.shape[2]):
                if masked_array[z, y, x]:
                    # Add a node for the masked value
                    coord_to_node[(z, y, x)] = node_id
                    # g.add_node(node_id)
                    nodes.append(node_id)
                    values.append(masked_array[z, y, x])
                    node_id += 1

    # Define 3D neighborhood offsets for 6-connectivity
    neighbor_offsets = [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)]

    # Iterate through the 3D masked array again to create edges
    edges = []
    for z in range(masked_array.shape[0]):
        for y in range(masked_array.shape[1]):
            for x in range(masked_array.shape[2]):
                if masked_array[z, y, x]:
                    # Get the current node ID
                    current_node_id = coord_to_node[(z, y, x)]

                    # Create edges to adjacent masked values
                    for offset in neighbor_offsets:
                        new_z, new_y, new_x = z + offset[0], y + offset[1], x + offset[2]
                        if (new_z, new_y, new_x) in coord_to_node:
                            adjacent_node_id = coord_to_node[(new_z, new_y, new_x)]

                            edges.append([current_node_id, adjacent_node_id])

    return np.array(values), np.array(edges), coord_to_node


def get_4d_masked_edges(masked_array):
    """
    This function processes a 4D masked array to extract nodes and edges based on the masked values. It creates a mapping
    from each masked coordinate to a unique node ID and then identifies edges between adjacent nodes based on 8-connectivity
    (neighbors in the cardinal directions across spatial and temporal axes). This function is ideal for converting a 4D volume
    (3D spatial plus temporal or another dimension) into a graph representation, where each masked voxel represents a node
    and edges represent direct adjacency between these voxels.

    Parameters:
    - masked_array (numpy.ndarray): A 4D boolean array where True values indicate masked points that will be converted to nodes.

    Returns:
    - tuple:
        - numpy.ndarray: Array of values from the masked positions of the input array.
        - numpy.ndarray: Array of edges, where each edge is represented by a pair of node IDs indicating connectivity.
        - dict: A dictionary mapping from 4D coordinates to node IDs.

    Notes:
    - The function assumes the input array is strictly four-dimensional and boolean.
    - Edges are only established between directly adjacent nodes in the 4D space, not diagonally across any dimension.
    """
    # Create a dictionary to map coordinates to node IDs
    coord_to_node = {}
    node_id = 0

    nodes = []
    values = []
    # Iterate through the 3D masked array
    for z in range(masked_array.shape[0]):
        for y in range(masked_array.shape[1]):
            for x in range(masked_array.shape[2]):
                for t in range(masked_array.shape[3]):
                    if masked_array[z, y, x, t]:
                        # Add a node for the masked value
                        coord_to_node[(z, y, x, t)] = node_id
                        nodes.append(node_id)
                        values.append(masked_array[z, y, x, t])
                        node_id += 1

    # Define 3D neighborhood offsets for 6-connectivity
    neighbor_offsets = [
        (0, 0, 0, 1), (0, 0, 0, -1), (0, 0, 1, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, -1, 0, 0),
        (1, 0, 0, 0), (-1, 0, 0, 0)
    ]

    # Iterate through the 3D masked array again to create edges
    edges = []
    for z in range(masked_array.shape[0]):
        for y in range(masked_array.shape[1]):
            for x in range(masked_array.shape[2]):
                for t in range(masked_array.shape[3]):
                    if masked_array[z, y, x, t]:
                        # Get the current node ID
                        current_node_id = coord_to_node[(z, y, x, t)]

                        # Create edges to adjacent masked values
                        for offset in neighbor_offsets:
                            new_z, new_y, new_x, new_t = z + offset[0], y + offset[1], x + offset[2], t + offset[3]
                            if (new_z, new_y, new_x, new_t) in coord_to_node:
                                adjacent_node_id = coord_to_node[(new_z, new_y, new_x, new_t)]

                                edges.append([current_node_id, adjacent_node_id])

    return np.array(values), np.array(edges), coord_to_node


def puma(psi: np.ndarray, edges: np.ndarray, period, max_jump: int = 1, p: float = 1, **kwargs):
    """
    Based on the git repository: "https://github.com/yoyololicon/kamui/tree/dev".
    The `puma` function implements a phase unwrapping method that utilizes graph cuts to minimize a potential
    function over a given network structure defined by nodes (psi values) and edges. It iteratively adjusts the
    unwrapping by minimizing the potential energy across edges, which can be thought of as 'cuts' in the graph.

    Parameters:
        psi (numpy.ndarray): A 1-dimensional array containing the phase values at each node of the graph.
        edges (numpy.ndarray): A 2-dimensional array where each row represents an edge connecting two nodes, with the nodes
            indexed according to their positions in `psi`.
        period (unused in this snippet, typically used to define the cycle length of phase values).
        max_jump (int): The maximum jump magnitude allowed in a single graph cut iteration. Default is 1.
        p (float): The p-norm used in the potential energy calculation. Default is 1.
        **kwargs: Additional keyword arguments for adjusting the potential function V, such as 'potential_mode', 'delta', and 'lam'.

    Returns:
        numpy.ndarray: An array the same size as `psi` containing the adjusted phase values after the graph cut optimization.

    Notes:
        - This function requires the `maxflow` library to create and manipulate the graph.
        - The potential function V can be configured via `kwargs` to use different modes of potential ('truncated', 'smooth', 'tanh').
        - The function returns the phase values adjusted for discontinuities across the graph defined by `edges`.
    """

    if max_jump > 1:
        # jump_steps = list(range(1, max_jump + 1)) * 5
        jump_steps = list(range(1, max_jump + 1))
    else:
        jump_steps = [max_jump]

    total_nodes = psi.size

    def V(x, potential_mode='truncated', delta=1 / np.sqrt(2), lam=0.5, **kwargs):
        """
        Computes the potential value for a given difference 'x' between nodes, based on a specified mode. This function is
        used within graph-based optimization to evaluate the cost or potential associated with a particular phase difference.

        Parameters:
        - x (numpy.ndarray): Array of differences for which the potential is calculated.
        - potential_mode (str): The mode of potential calculation. It can be 'truncated', 'smooth', or 'tanh'.
        - delta (float): A parameter that influences the scaling of differences in the potential calculation. Default is 1/sqrt(2).
        - lam (float): A parameter that sets the maximum potential value in some modes or scales the potential in others. Default is 0.5.

        Returns:
        - numpy.ndarray: The computed potential values for each element in 'x'.

        Raises:
        - Exception: If an unknown 'potential_mode' is specified.

        Notes:
        - 'truncated': Potential is quadratic up to a limit 'lam', beyond which it is capped.
        - 'smooth': Potential is a smooth quadratic function that becomes less steep as 'x' increases.
        - 'tanh': Potential uses a hyperbolic tangent function to provide a smooth transition between values, capped by 'lam'.
        """
        if potential_mode == 'truncated':
            potential = x ** 2 / (2 * delta ** 2)
            potential[potential > lam] = lam

        elif potential_mode == 'smooth':
            potential = lam * x ** 2 / (2 * delta ** 2 + x ** 2)

        elif potential_mode == 'tanh':
            potential = lam * np.tanh(x ** 2 / (2 * delta ** 2))

        else:
            raise Exception(f'Potential mode {potential_mode} not implemented...')

        return potential

    K = np.zeros_like(psi)

    def cal_Ek(K, psi, i, j):
        """
        Computes the total energy of the graph for a given configuration of node potentials ('K') and phase values ('psi')
        across specified edges. This function sums up the potential values for all the edges defined by indices arrays 'i' and 'j'.

        Parameters:
        - K (numpy.ndarray): An array containing the potential values at each node. This represents the current state of the graph.
        - psi (numpy.ndarray): An array containing the original phase values at each node.
        - i (numpy.ndarray): An array of starting indices for the edges, referencing positions in 'K' and 'psi'.
        - j (numpy.ndarray): An array of ending indices for the edges, referencing positions in 'K' and 'psi'.
        - **kwargs: Additional keyword arguments that may be passed to the potential function 'V'.

        Returns:
        - float: The total energy calculated as the sum of potentials across all defined edges. The potential for each edge is
        determined by the difference in the adjusted potentials ('K') and original phase differences ('psi') between nodes.

        Notes:
        - This function leverages the potential function 'V', which calculates the potential based on the difference in the
        total phase shift (adjusted potential difference plus original phase difference) across an edge.
        - The energy computation is central to the optimization in graph-based unwrapping, guiding the iterative adjustment of
        the node potentials to minimize the overall system energy.
        """
        return np.sum(V(K[j] - K[i] - psi[i] + psi[j], **kwargs))

    prev_Ek = cal_Ek(K, psi, edges[:, 0], edges[:, 1])

    energy_list = []

    for step in jump_steps:
        while 1:
            energy_list.append(prev_Ek)
            G = maxflow.Graph[float]()
            G.add_nodes(total_nodes)

            i, j = edges[:, 0], edges[:, 1]
            psi_diff = psi[i] - psi[j]
            a = (K[j] - K[i]) - psi_diff
            e00 = e11 = V(a)
            e01 = V(a - step, **kwargs)
            e10 = V(a + step, **kwargs)
            weight = np.maximum(0, e10 + e01 - e00 - e11)

            G.add_edges(edges[:, 0], edges[:, 1], weight, np.zeros_like(weight))

            tmp_st_weight = np.zeros((2, total_nodes))

            for i in range(edges.shape[0]):
                u, v = edges[i]
                tmp_st_weight[0, u] += max(0, e10[i] - e00[i])
                tmp_st_weight[0, v] += max(0, e11[i] - e10[i])
                tmp_st_weight[1, u] -= min(0, e10[i] - e00[i])
                tmp_st_weight[1, v] -= min(0, e11[i] - e10[i])

            for i in range(total_nodes):
                G.add_tedge(i, tmp_st_weight[0, i], tmp_st_weight[1, i])

            G.maxflow()

            partition = G.get_grid_segments(np.arange(total_nodes))
            K[~partition] += step

            energy = cal_Ek(K, psi, edges[:, 0], edges[:, 1])

            if energy < prev_Ek:
                prev_Ek = energy
            else:
                K[~partition] -= step
                break

    return K


def gc3D_unwrap(phi_w, mask, start_index=0, period=2 * np.pi, **kwargs):
    """
    Unwraps a 3D phase array ('phi_w') using graph-based methods. This function applies the unwrapping only to the
    regions specified by the mask, leveraging the connectivity and discontinuity properties within those regions.

    Parameters:
    - phi_w (numpy.ndarray): A 3-dimensional array containing wrapped phase values that need to be unwrapped.
    - mask (numpy.ndarray): A binary mask array that indicates the regions within `phi_w` where unwrapping should be applied.
      The unwrapping is performed only where mask is True.
    - start_index (int, optional): The index of the node from which to normalize the resulting unwrapped phases. Default is 0.
    - period (float, optional): The period of the phase wrap-around, typically 2*pi for phase data. Default is 2*pi.
    - **kwargs: Additional keyword arguments passed to the `puma` function for phase unwrapping optimization.

    Returns:
    - numpy.ndarray: The unwrapped phase array, where only the regions specified by the mask have been unwrapped.

    Raises:
    - ValueError: If the input array `phi_w` is not three-dimensional as required.

    Notes:
    - The function first masks the input phase array with the provided mask, then extracts the edges and nodes necessary
      for graph construction via the `get_3d_masked_edges` function.
    - It then uses the `puma` function to optimize the unwrapping, ensuring that the phase continuity is maintained across
      the masked region.
    - The `puma` function returns normalized phase values, which are then scaled back to the original period and adjusted
      to ensure a continuous phase across the entire volume.
    - This function modifies the input array `phi_w` directly by updating the phase values at the locations specified by the mask.
    """
    masked_to_unwrap = np.ma.masked_array(phi_w, 1 - mask)

    if phi_w.ndim != 3:
        raise ValueError("Input array phi_w must have 3 dimensions.")

    x, edges, coord_to_node = get_3d_masked_edges(masked_to_unwrap)

    m = puma(x / period, edges, period, **kwargs)
    m -= m[start_index]

    for index, loc in enumerate(coord_to_node.keys()):
        phi_w[loc] = m[index] * period + phi_w[loc]

    return phi_w


def gc4D_unwrap(phi_w, mask, start_index=0, period=2 * np.pi, **kwargs):
    """
    Unwraps a 4D phase array ('phi_w') using graph-based methods specifically tailored for four-dimensional data. This
    function applies unwrapping only to the regions specified by the mask, leveraging the connectivity and discontinuity
    properties within those regions across both spatial and temporal dimensions.

    Parameters:
    - phi_w (numpy.ndarray): A 4-dimensional array containing wrapped phase values that need to be unwrapped.
    - mask (numpy.ndarray): A binary mask array that indicates the regions within `phi_w` where unwrapping should be applied.
      Only the elements of `phi_w` corresponding to a True value in `mask` are considered for unwrapping.
    - start_index (int, optional): The index of the node from which to normalize the resulting unwrapped phases. Default is 0.
    - period (float, optional): The period of the phase wrap-around, typically 2*pi for phase data. Default is 2*pi.
    - **kwargs: Additional keyword arguments passed to the `puma` function for phase unwrapping optimization.

    Returns:
    - numpy.ndarray: The unwrapped phase array, where only the regions specified by the mask have been unwrapped.

    Raises:
    - ValueError: If the input array `phi_w` is not four-dimensional as required.

    Notes:
    - The function first converts the input phase array to a masked array where non-masked regions are ignored. It then
      extracts the edges and nodes necessary for graph construction via the `get_4d_masked_edges` function.
    - It uses the `puma` function to optimize the unwrapping, ensuring that phase continuity is maintained across the
      masked region in all four dimensions.
    - The `puma` function returns normalized phase values, which are then scaled back to the original period and adjusted
      to ensure a continuous phase across the entire dataset.
    - This function modifies the input array `phi_w` directly by updating the phase values at the locations specified by the mask.
    """
    masked_to_unwrap = np.ma.masked_array(phi_w, 1 - mask)

    if phi_w.ndim != 4:
        raise ValueError("Input array phi_w must have 4 dimensions.")

    x, edges, coord_to_node = get_4d_masked_edges(masked_to_unwrap)

    m = puma(x / period, edges, period, **kwargs)
    m -= m[start_index]

    for index, loc in enumerate(coord_to_node.keys()):
        phi_w[loc] = m[index] * period + phi_w[loc]

    return phi_w
