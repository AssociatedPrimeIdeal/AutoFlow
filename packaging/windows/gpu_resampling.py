"""Inference resampler named by the Dataset7010/7020 model plans.

Retains the training project's nnUNet interpolation arguments and CUDA worker
device rotation, with CPU interpolation available on machines without CUDA.
"""

import multiprocessing
import os

import numpy as np
import torch
import torch.nn.functional as F
from nnunetv2.configuration import ANISO_THRESHOLD
from nnunetv2.preprocessing.resampling.resample_torch import resample_torch_fornnunet


def _resampling_device():
    device = torch.device("cpu")
    if os.environ.get("AUTOFLOW_NNUNET_DEVICE", "auto").lower() != "cpu" and torch.cuda.is_available():
        identity = multiprocessing.current_process()._identity
        index = (int(identity[0]) - 1 if identity else 0) % torch.cuda.device_count()
        torch.cuda.set_device(index)
        device = torch.device("cuda", index)
    return device


def resample_segmentation_nearest(data, new_shape, current_spacing, new_spacing,
                                  is_seg=True, num_threads=4, device=torch.device("cpu")):
    """Match the Linux label resampler's voxel-center and padding conventions."""
    if not is_seg:
        raise ValueError("resample_segmentation_nearest is only for label maps")
    shape = tuple(int(value) for value in new_shape)
    if data.ndim not in (3, 4) or len(shape) != data.ndim - 1 or min(shape) < 1:
        raise ValueError(f"Invalid label resampling shapes: {data.shape} -> {shape}")
    if tuple(data.shape[1:]) == shape:
        return data
    was_numpy = isinstance(data, np.ndarray)
    tensor = torch.from_numpy(np.ascontiguousarray(data)) if was_numpy else data
    original_device, original_dtype = tensor.device, tensor.dtype
    with torch.no_grad():
        result = F.interpolate(tensor.to(device=device, dtype=torch.float32)[None],
                               size=shape, mode="nearest-exact")[0].to(dtype=original_dtype)
    return result.cpu().numpy() if was_numpy else result.to(original_device)


def resample_torch_cuda_nearest(data, new_shape, current_spacing, new_spacing,
                               is_seg=True, num_threads=4):
    return resample_segmentation_nearest(data, new_shape, current_spacing, new_spacing,
                                         is_seg=is_seg, num_threads=num_threads,
                                         device=_resampling_device())


def resample_torch_cuda_fornnunet(data, new_shape, current_spacing, new_spacing,
                                is_seg=False, num_threads=4,
                                memefficient_seg_resampling=False,
                                force_separate_z=None,
                                separate_z_anisotropy_threshold=ANISO_THRESHOLD):
    return resample_torch_fornnunet(
        data=data, new_shape=new_shape, current_spacing=current_spacing,
        new_spacing=new_spacing, is_seg=is_seg, num_threads=num_threads,
        device=_resampling_device(), memefficient_seg_resampling=memefficient_seg_resampling,
        force_separate_z=force_separate_z,
        separate_z_anisotropy_threshold=separate_z_anisotropy_threshold,
        mode="linear", aniso_axis_mode="nearest-exact",
    )
