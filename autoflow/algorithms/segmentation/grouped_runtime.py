"""Exact grouped feature transfer and scoped nnUNet export scheduling.

No nnUNet or external runner files are changed. Crop, normalization, resampling,
network arithmetic, window order, and the upstream export writer are retained.
"""

from __future__ import annotations

from functools import update_wrapper
from multiprocessing.pool import ThreadPool
from pathlib import Path
import os
import threading
from types import FunctionType, MethodType, SimpleNamespace
from typing import Any, Dict, Iterator, Sequence, Tuple

import numpy as np

from ...task_control import check_cancelled


_inference_lock = threading.RLock()


def _upload_feature_pool(source, device, sample_channels):
    import torch
    if torch.device(device).type != "cuda":
        return None
    required = source.nbytes + 2 * source[0].nbytes * int(sample_channels) + 16 * 1024**3
    free, _ = torch.cuda.mem_get_info(device)
    if free < required:
        # Release only unused CUDA allocator blocks, never CPU inputs or files.
        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info(device)
    if free < required:
        return None
    try:
        return torch.as_tensor(source, dtype=torch.float32, device=device)
    except torch.cuda.OutOfMemoryError:
        pass
    torch.cuda.empty_cache()
    return None


class _FixedThreadTorch:
    """Forward torch operations while suppressing nested global thread setters."""

    def __init__(self, torch):
        self._torch = torch

    def __getattr__(self, name):
        return getattr(self._torch, name)

    def set_num_threads(self, count):
        # The enclosing call sets/restores this process-wide value once, after
        # waiting for every export thread. Upstream numerical code stays intact.
        pass


def _function_with_globals(function, **overrides):
    namespace = dict(function.__globals__)
    namespace.update(overrides)
    clone = FunctionType(function.__code__, namespace, function.__name__,
                         function.__defaults__, function.__closure__)
    clone.__kwdefaults__ = function.__kwdefaults__
    return update_wrapper(clone, function)


def predict_grouped_iterator(predictor, iterator, export_workers=4):
    """Run upstream inference/export with bounded thread-based CPU export.

    Only this predictor and private copies of upstream function globals change.
    The installed runtime and unrelated predictor instances remain untouched.
    CPU or unrecognized upstream implementations retain the normal process pool.
    """
    import torch
    from nnunetv2.inference import export_prediction as export
    from autoflow.nnunet_runtime import _gpu_first_predict

    original_iterator = predictor.predict_from_data_iterator
    original_logits = predictor.predict_logits_from_preprocessed_data
    original_sliding = predictor.predict_sliding_window_return_logits
    converter = export.convert_predicted_logits_to_segmentation_with_correct_shape
    writer = export.export_prediction_from_logits
    iterator_function = getattr(original_iterator, "__func__", None)
    logits_function = getattr(original_logits, "__func__", None)
    if (predictor.device.type != "cuda" or iterator_function is None or logits_function is None
            or "multiprocessing" not in iterator_function.__code__.co_names
            or "convert_predicted_logits_to_segmentation_with_correct_shape" not in writer.__code__.co_names):
        return original_iterator(iterator, save_probabilities=False,
                                 num_processes_segmentation_export=export_workers)
    fixed_torch = _FixedThreadTorch(torch)
    converter = _function_with_globals(converter, torch=fixed_torch)
    writer = _function_with_globals(writer,
        convert_predicted_logits_to_segmentation_with_correct_shape=converter)
    pool_runtime = SimpleNamespace(get_context=lambda *_: SimpleNamespace(Pool=ThreadPool))
    threaded = _function_with_globals(iterator_function, multiprocessing=pool_runtime,
        export_prediction_from_logits=writer,
        convert_predicted_logits_to_segmentation_with_correct_shape=converter)
    logits = _function_with_globals(logits_function, torch=fixed_torch)
    def sliding(self, data):
        return _gpu_first_predict(lambda unused, image: original_sliding(image), self, data, torch)
    # Grouped inference is an isolated CLI/API operation. Serialize our scopes
    # because torch's CPU thread count is process-wide, not thread-local.
    with _inference_lock:
        previous_threads = torch.get_num_threads()
        previous_attributes = {name: predictor.__dict__.get(name) for name in
                               ("predict_logits_from_preprocessed_data", "predict_sliding_window_return_logits")}
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        predictor.predict_logits_from_preprocessed_data = MethodType(logits, predictor)
        predictor.predict_sliding_window_return_logits = MethodType(sliding, predictor)
        try:
            return threaded(predictor, iterator, save_probabilities=False,
                            num_processes_segmentation_export=max(1, int(export_workers)))
        finally:
            for name, value in previous_attributes.items():
                if value is None:
                    predictor.__dict__.pop(name, None)
                else:
                    setattr(predictor, name, value)
            torch.set_num_threads(previous_threads)


def iter_grouped_temporal_samples(
    pipeline, device,
    sample_inputs: Sequence[Dict[str, Any]],
    source_paths: Dict[Tuple[str, int, int], Path],
    plans_manager: Any,
    configuration_manager: Any,
    dataset_json: Dict[str, Any],
    temporal_radius: int,
    target_sample_ids: set[str] | None = None,
) -> Iterator[Tuple[Dict[str, Any], Any, np.ndarray | None, Dict[str, Any]]]:
    """Preprocess one source 4D case with one common crop and deduplicated data resampling.

    All temporal samples from the source case share the nonzero crop produced by
    every source feature across the cardiac cycle. Data maps are then normalized
    and resampled once per unique source map (12 global + 5*T temporal maps),
    before being assembled back into the normal 37-channel sample layout.
    """
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
    from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape

    if not sample_inputs:
        return
    pipeline._require_grouped_temporal_normalization(configuration_manager)
    samples = sorted(sample_inputs, key=lambda item: int(item["frame"]))
    frames = [int(item["frame"]) for item in samples]
    total_frames = len(samples)
    if frames != list(range(total_frames)):
        raise ValueError(f"Grouped 4D preprocessing requires complete 0..T-1 frames, got {frames}")

    expected_channels = len(pipeline.GLOBAL_CHANNEL_NAMES) + len(pipeline.TEMPORAL_FEATURE_NAMES) * (2 * int(temporal_radius) + 1)
    if expected_channels != len(configuration_manager.normalization_schemes):
        raise ValueError(
            f"Plan has {len(configuration_manager.normalization_schemes)} channels, "
            f"but temporal radius {temporal_radius} requires {expected_channels}"
        )

    expected_source_paths: Dict[Tuple[str, int, int], None] = {}
    for sample in samples:
        for source_key in pipeline.temporal_source_keys(int(sample["frame"]), total_frames, int(temporal_radius)):
            expected_source_paths.setdefault(source_key, None)
    if list(source_paths) != list(expected_source_paths):
        raise ValueError("Grouped 4D preprocessing received incomplete or unordered source inputs")

    reader_writer = plans_manager.image_reader_writer_class()

    arrays_by_source: Dict[Tuple[str, int, int], np.ndarray] = {}
    base_properties: Dict[str, Any] | None = None
    for source_key, image_path in source_paths.items():
        image, properties = reader_writer.read_images((str(image_path),))
        if base_properties is None:
            base_properties = properties
        elif list(properties["spacing"]) != list(base_properties["spacing"]):
            raise ValueError(f"{image_path}: inconsistent 4D source spacing")
        arrays_by_source[source_key] = pipeline._transposed_image(image, plans_manager).astype(np.float32, copy=False)
    if base_properties is None:
        raise RuntimeError("No source image properties were loaded")

    crop_slices, common_bbox, crop_nonzero_mask = pipeline.common_4d_crop_metadata(arrays_by_source)
    original_spacing = [base_properties["spacing"][axis] for axis in plans_manager.transpose_forward]
    target_spacing = list(configuration_manager.spacing)
    if len(target_spacing) < len(crop_nonzero_mask.shape):
        target_spacing = [original_spacing[0], *target_spacing]
    new_shape = compute_new_shape(crop_nonzero_mask.shape, original_spacing, target_spacing)

    preprocessor = DefaultPreprocessor(verbose=False)
    normalization_seg = np.where(crop_nonzero_mask, np.int8(0), np.int8(-1))[None]
    ordered_sources = list(arrays_by_source)
    normalized_sources, normalization_metadata = pipeline.normalize_case_source_maps(
        arrays_by_source, crop_slices, normalization_seg, preprocessor, configuration_manager,
        plans_manager.foreground_intensity_properties_per_channel,
    )
    resampled_sources = configuration_manager.resampling_fn_data(
        np.stack(normalized_sources, axis=0), new_shape, original_spacing, target_spacing
    )
    resampled_sources = np.asarray(resampled_sources, dtype=np.float32)
    resident_pool = _upload_feature_pool(resampled_sources, device, expected_channels)
    source_index = {source_key: index for index, source_key in enumerate(ordered_sources)}

    for sample in samples:
        frame = int(sample["frame"])
        if target_sample_ids is not None and str(sample["sample_id"]) not in target_sample_ids:
            continue
        source_keys = pipeline.temporal_source_keys(frame, total_frames, int(temporal_radius))
        check_cancelled()
        indices = [source_index[key] for key in source_keys]
        data = None
        if resident_pool is not None:
            import torch
            try:
                data = torch.index_select(resident_pool, 0, torch.tensor(indices, device=device))
            except torch.cuda.OutOfMemoryError:
                pass
            if data is None:
                resident_pool = None
                torch.cuda.empty_cache()
        if data is None:
            data = np.stack([resampled_sources[index] for index in indices], axis=0)
        properties = dict(base_properties)
        properties.update({
            "shape_before_cropping": tuple(int(value) for value in arrays_by_source[ordered_sources[0]].shape),
            "bbox_used_for_cropping": common_bbox,
            "shape_after_cropping_and_before_resampling": tuple(int(value) for value in crop_nonzero_mask.shape),
            "temporal_grouped_preprocessing": {
                "version": pipeline.GROUPED_TEMPORAL_PREPROCESS_VERSION,
                "total_frames": total_frames,
                "common_bbox": common_bbox,
                "unique_data_maps": len(ordered_sources),
            },
            "temporal_normalization": normalization_metadata,
        })
        segmentation: np.ndarray | None = None
        label_path = sample.get("label")
        if label_path is not None:
            segmentation, class_locations = pipeline._segmentation_for_grouped_training(
                Path(label_path), reader_writer, plans_manager, configuration_manager, dataset_json,
                crop_slices, crop_nonzero_mask, original_spacing, target_spacing, new_shape, preprocessor,
            )
            properties["class_locations"] = class_locations
        yield sample, data, segmentation, properties
