"""Inference resampler named by the Dataset7010/7020 model plans.

Retains the training project's nnUNet interpolation arguments and CUDA worker
device rotation, with CPU interpolation available on machines without CUDA.
"""

import multiprocessing
import os

import torch
from nnunetv2.configuration import ANISO_THRESHOLD
from nnunetv2.preprocessing.resampling.resample_torch import resample_torch_fornnunet


def resample_torch_cuda_fornnunet(data, new_shape, current_spacing, new_spacing,
                                is_seg=False, num_threads=4,
                                memefficient_seg_resampling=False,
                                force_separate_z=None,
                                separate_z_anisotropy_threshold=ANISO_THRESHOLD):
    device = torch.device("cpu")
    if os.environ.get("AUTOFLOW_NNUNET_DEVICE", "auto").lower() != "cpu" and torch.cuda.is_available():
        identity = multiprocessing.current_process()._identity
        index = (int(identity[0]) - 1 if identity else 0) % torch.cuda.device_count()
        torch.cuda.set_device(index)
        device = torch.device("cuda", index)
    return resample_torch_fornnunet(
        data=data, new_shape=new_shape, current_spacing=current_spacing,
        new_spacing=new_spacing, is_seg=is_seg, num_threads=num_threads,
        device=device, memefficient_seg_resampling=memefficient_seg_resampling,
        force_separate_z=force_separate_z,
        separate_z_anisotropy_threshold=separate_z_anisotropy_threshold,
        mode="linear", aniso_axis_mode="nearest-exact",
    )
