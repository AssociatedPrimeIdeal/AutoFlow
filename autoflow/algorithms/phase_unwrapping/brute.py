"""Legacy local-gradient and discontinuity-guided phase recovery.

Derived from the MIT-licensed PUDIP-Flow TradMethod sources.
"""

import numpy as np
from tqdm import tqdm


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
