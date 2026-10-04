"""CPU and Torch Fourier resampling for the NPRS skimage solver."""

from __future__ import annotations

import numpy as np

from skimage.restoration import unwrap_phase
from .fourier import pad_array


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
