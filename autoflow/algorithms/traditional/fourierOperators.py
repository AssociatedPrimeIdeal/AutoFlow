import numpy as np
from typing import Union
import numpy.typing as npt


def kshift(data, shift: Union[int, tuple], axis: Union[int, tuple] = None):
    """
    Shifts the phase in kspace obtained from ks=np.fft.fftshift(np.fft.fftn(image)),
    such that in image space ks is not shifted, but in Fourier space ks remains centered.
    This snipped is extracted from the public functions of the precon framework.

    Args:
        data (ndarray): kspace to shift.
        shift (int or tuple): shift to apply.
        axis (int or tuple): axis to apply the phase shift on.
        real_flag (bool): Restrict output to real (default is False).

    Returns:
        out (ndarray): Output matrix.
    """
    if axis is None:
        if isinstance(shift, tuple):
            axis = tuple(list(range(0, len(shift))))
        else:
            axis = 0

    if (isinstance(axis, tuple) and not isinstance(shift, tuple)) or (
            not isinstance(axis, tuple) and isinstance(shift, tuple)):
        raise RuntimeError('"axis" and "shift" must have the same number of elements')
    if isinstance(axis, tuple) and isinstance(shift, tuple) and len(shift) != len(axis):
        raise RuntimeError('"axis" and "shift" must have the same number of elements')

    if not isinstance(shift, tuple):
        shift = (shift,)
    if not isinstance(axis, tuple):
        axis = (axis,)

    for s, a in zip(shift, axis):
        if a >= data.ndim:
            continue
        res = data.shape[a]
        dpi = np.linspace(0, 2 * np.pi * (1 - 1 / res) * s, res)
        siz = np.ones((data.ndim,), dtype=np.int32)
        siz[a] = res
        dpi = np.reshape(dpi, tuple(siz))
        data = data * np.exp(-1j * dpi)

    return data


def pad_array(img, pad_width, value=0):
    """
    Allows to pad or box-crop an array depending on pad_width (positive values zero-pad
    and negative values crop)

    Args:
        img (ndarray): image to zero-pad or crop.
        value (double): value to use for padding.
        pad_width (int or tuple): number of elements to pad or crop from.
            Examples of pad_width:
                pad_width = 6 --> pads array symmetrically along all directions with 2 x 6 elements with value
                pad_width = -6 --> crops array symmetrically along all directions by 2 x 6 elements
                pad_width = ((3,5),(7,7)) where len(pad_width) must be = img.ndim --> pads by 3 + 5 elements
                    along axis 1 and 7 + 7 elements along axis 2

    Returns:
        out (ndarray): Padded or cropped matrix.
    """

    def duplicate_tuple_elements(input_tuple):
        duplicated_elements = []
        for element in input_tuple:
            duplicated_element = (element, element)
            duplicated_elements.append(duplicated_element)
        return tuple(duplicated_elements)

    def pad_with(vector, pad_width, iaxis, kwargs):
        pad_value = kwargs.get('padder', 0)
        vector[:pad_width[0]] = pad_value
        vector[-pad_width[1]:] = pad_value

    if isinstance(pad_width, int):
        if pad_width >= 0:
            return np.pad(img, pad_width)
        else:
            pad_width = duplicate_tuple_elements(np.tile(-pad_width, img.ndim))
            reversed_padding = tuple(
                [slice(start_pad, dim - end_pad) for ((start_pad, end_pad), dim) in zip(pad_width, img.shape)])
            return img[reversed_padding]
    else:
        if not isinstance(pad_width, tuple):
            raise Exception('pad_width needs to be a tuple')

        if not len(pad_width) == img.ndim:
            raise Exception('pad_width needs to have same dimensions as image')

        if all(i >= 0 for i in pad_width):
            pad_width = duplicate_tuple_elements(pad_width)
            return np.pad(img, pad_width)
        elif all(i < 0 for i in pad_width):
            pad_width = duplicate_tuple_elements(tuple([-i for i in np.array(pad_width)]))
            reversed_padding = tuple(
                [slice(start_pad, dim - end_pad) for ((start_pad, end_pad), dim) in zip(pad_width, img.shape)])
            return img[reversed_padding]
        else:
            raise Exception('All value element must be >0 OR <0')


def kshift(data, shift: Union[int, tuple], axis: Union[int, tuple] = None):
    """
    Shifts the phase in kspace obtained from ks=np.fft.fftshift(np.fft.fftn(image)),
    such that in image space ks is not shifted, but in Fourier space ks remains centered.
    This snipped is extracted from the public functions of the precon framework.

    Args:
        data (ndarray): kspace to shift.
        shift (int or tuple): shift to apply.
        axis (int or tuple): axis to apply the phase shift on.
        real_flag (bool): Restrict output to real (default is False).

    Returns:
        out (ndarray): Output matrix.
    """
    if axis is None:
        if isinstance(shift, tuple):
            axis = tuple(list(range(0, len(shift))))
        else:
            axis = 0

    if (isinstance(axis, tuple) and not isinstance(shift, tuple)) or (
            not isinstance(axis, tuple) and isinstance(shift, tuple)):
        raise RuntimeError('"axis" and "shift" must have the same number of elements')
    if isinstance(axis, tuple) and isinstance(shift, tuple) and len(shift) != len(axis):
        raise RuntimeError('"axis" and "shift" must have the same number of elements')

    if not isinstance(shift, tuple):
        shift = (shift,)
    if not isinstance(axis, tuple):
        axis = (axis,)

    for s, a in zip(shift, axis):
        if a >= data.ndim:
            continue
        res = data.shape[a]
        dpi = np.linspace(0, 2 * np.pi * (1 - 1 / res) * s, res)
        siz = np.ones((data.ndim,), dtype=np.int32)
        siz[a] = res
        dpi = np.reshape(dpi, tuple(siz))
        data = data * np.exp(-1j * dpi)

    return data


def pad_array(img, pad_width, value=0):
    """
    Allows to pad or box-crop an array depending on pad_width (positive values zero-pad
    and negative values crop)

    Args:
        img (ndarray): image to zero-pad or crop.
        value (double): value to use for padding.
        pad_width (int or tuple): number of elements to pad or crop from.
            Examples of pad_width:
                pad_width = 6 --> pads array symmetrically along all directions with 2 x 6 elements with value
                pad_width = -6 --> crops array symmetrically along all directions by 2 x 6 elements
                pad_width = ((3,5),(7,7)) where len(pad_width) must be = img.ndim --> pads by 3 + 5 elements
                    along axis 1 and 7 + 7 elements along axis 2

    Returns:
        out (ndarray): Padded or cropped matrix.
    """

    def duplicate_tuple_elements(input_tuple):
        duplicated_elements = []
        for element in input_tuple:
            duplicated_element = (element, element)
            duplicated_elements.append(duplicated_element)
        return tuple(duplicated_elements)

    def pad_with(vector, pad_width, iaxis, kwargs):
        pad_value = kwargs.get('padder', 0)
        vector[:pad_width[0]] = pad_value
        vector[-pad_width[1]:] = pad_value

    if isinstance(pad_width, int):
        if pad_width >= 0:
            return np.pad(img, pad_width)
        else:
            pad_width = duplicate_tuple_elements(np.tile(-pad_width, img.ndim))
            reversed_padding = tuple(
                [slice(start_pad, dim - end_pad) for ((start_pad, end_pad), dim) in zip(pad_width, img.shape)])
            return img[reversed_padding]
    else:
        if not isinstance(pad_width, tuple):
            raise Exception('pad_width needs to be a tuple')

        if not len(pad_width) == img.ndim:
            raise Exception('pad_width needs to have same dimensions as image')

        if all(i >= 0 for i in pad_width):
            pad_width = duplicate_tuple_elements(pad_width)
            return np.pad(img, pad_width)
        elif all(i < 0 for i in pad_width):
            pad_width = duplicate_tuple_elements(tuple([-i for i in np.array(pad_width)]))
            reversed_padding = tuple(
                [slice(start_pad, dim - end_pad) for ((start_pad, end_pad), dim) in zip(pad_width, img.shape)])
            return img[reversed_padding]
        else:
            raise Exception('All value element must be >0 OR <0')
