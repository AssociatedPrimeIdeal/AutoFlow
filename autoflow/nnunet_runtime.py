"""Exact data-transfer optimizations confined to the nnUNet child process."""

import functools
import inspect
import math
import re


def _gpu_first_predict(original, predictor, input_image, torch):
    if (predictor.device.type != "cuda" or not predictor.perform_everything_on_device
            or input_image.device.type != "cpu"):
        return original(predictor, input_image)
    padded_shape = [max(int(size), int(patch)) for size, patch in zip(
        input_image.shape[1:], predictor.configuration_manager.patch_size
    )]
    input_bytes = input_image.numel() * input_image.element_size()
    padded_bytes = int(input_image.shape[0]) * math.prod(padded_shape) * input_image.element_size()
    # Keep ample room for the network, output accumulators, and export worker.
    # Small/busy GPUs retain nnUNet's original CPU-padding path.
    free, _ = torch.cuda.mem_get_info(predictor.device)
    if free < input_bytes + padded_bytes + 16 * 1024**3:
        return original(predictor, input_image)
    gpu_image = None
    try:
        gpu_image = input_image.to(predictor.device)
        return original(predictor, gpu_image)
    except torch.cuda.OutOfMemoryError:
        pass
    # Leave the exception scope before releasing allocations held by its traceback.
    gpu_image = None
    torch.cuda.empty_cache()
    return original(predictor, input_image)


def configure_exact_inference_runtime():
    """Use the same nnUNet padding/inference with less CPU copying.

    Called only in the isolated standard 4D predictor. Model parameters,
    arithmetic precision, normalization, window order, and output export stay
    with nnUNet. No installed nnUNet source is edited.
    """
    import torch
    import nnunetv2.inference.predict_from_raw_data as runtime

    predictor_class = runtime.nnUNetPredictor
    original = predictor_class.predict_sliding_window_return_logits
    if getattr(original, "_autoflow_gpu_first", False):
        return

    @functools.wraps(original)
    def gpu_first(self, input_image):
        return _gpu_first_predict(original, self, input_image, torch)

    gpu_first._autoflow_gpu_first = True
    predictor_class.predict_sliding_window_return_logits = gpu_first

    iterator = runtime.preprocessing_iterator_fromfiles
    try:
        source = inspect.getsource(iterator)
    except (OSError, TypeError):
        return
    # This upstream expression creates pinned tensors and discards the copies.
    # Only disable it for this recognized implementation, preserving other versions.
    discarded_pin = re.search(
        r"if pin_memory:\s*\[i\.pin_memory\(\) for i in item\.values\(\) "
        r"if isinstance\(i, torch\.Tensor\)\]", source
    )
    if discarded_pin is not None:
        signature = inspect.signature(iterator)

        @functools.wraps(iterator)
        def without_discarded_pinning(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.arguments["pin_memory"] = False
            return iterator(*bound.args, **bound.kwargs)

        runtime.preprocessing_iterator_fromfiles = without_discarded_pinning
