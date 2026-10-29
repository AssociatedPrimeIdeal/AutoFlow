"""Retain model predictions on an exact 1-mm grid before native-grid export."""
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import affine_transform


def capture_model_geometry(logits, properties, configuration, plans, dataset, output):
    """Interpolate model logits directly to 1 mm, then classify and undo axes.

    The effective preprocessor lattice follows voxel-center alignment: its
    extent is cropped native shape * native spacing (rounded model shape).
    No native exported hard segmentation is used to construct this cache.
    """
    import torch
    from ..geometry_grid import grid_shape
    if isinstance(dataset, str):
        dataset = json.loads(Path(dataset).read_text())
    manager = plans.get_label_manager(dataset)
    array = logits.detach().cpu().numpy() if torch.is_tensor(logits) else np.asarray(logits)
    if array.ndim != 4:
        raise ValueError('Model geometry cache requires 3-D logits')
    source_spacing = np.asarray(properties['spacing'])[plans.transpose_forward]
    crop_shape = np.asarray(properties['shape_after_cropping_and_before_resampling'])
    effective = crop_shape * source_spacing / np.asarray(array.shape[1:])
    bbox = np.asarray(properties['bbox_used_for_cropping'])
    first = (bbox[:, 0] - .5)*source_spacing + .5*effective
    full_shape = grid_shape(properties['shape_before_cropping'], source_spacing)
    # AutoFlow LR/AP/FH -> NIfTI HF/AP/RL reverses X and Z. The reader
    # reverses array axes again. Compensate ceil padding on those two axes
    # before restoring AutoFlow axes, so its physical origin stays exact.
    padding = (np.asarray(full_shape)-1) - (np.asarray(properties['shape_before_cropping'])-1)*source_spacing
    reader_padding = padding[plans.transpose_backward]
    reader_origin = np.array([-reader_padding[0], 0., -reader_padding[2]])
    target_origin = reader_origin[plans.transpose_forward]
    lower = np.maximum(0, np.ceil((bbox[:, 0]-.5)*source_spacing-target_origin).astype(int))
    upper = np.minimum(full_shape, np.floor((bbox[:, 1]-.5)*source_spacing-target_origin).astype(int)+1)
    shape = tuple(np.maximum(0, upper-lower))
    final = np.zeros(full_shape, np.int16)
    if all(shape):
        if not manager.has_regions:
            best = np.full(shape, -np.inf, np.float32)
            segmentation = np.zeros(shape, np.int16)
            for channel in range(array.shape[0]):
                sampled = affine_transform(np.asarray(array[channel], np.float32), np.diag(1/effective),
                    offset=(lower+target_origin-first)/effective, output_shape=shape, order=1,
                    mode='nearest', prefilter=False)
                selected = sampled > best
                segmentation[selected] = channel
                best[selected] = sampled[selected]
        else:
            sampled = np.stack([affine_transform(np.asarray(channel, np.float32), np.diag(1/effective),
                offset=(lower+target_origin-first)/effective, output_shape=shape, order=1,
                mode='nearest', prefilter=False) for channel in array])
            segmentation = manager.convert_logits_to_segmentation(sampled)
            if torch.is_tensor(segmentation):
                segmentation = segmentation.cpu().numpy()
        final[tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))] = segmentation
    final = final.transpose(plans.transpose_backward)
    rw = plans.image_reader_writer_class()
    from copy import deepcopy
    output_properties = deepcopy(properties)
    output_properties['spacing'] = [1., 1., 1.]
    if 'sitk_stuff' in output_properties:
        output_properties['sitk_stuff']['spacing'] = (1., 1., 1.)
    if 'nibabel_stuff' in output_properties:
        affine = np.asarray(output_properties['nibabel_stuff']['original_affine']).copy()
        affine[:3, :3] /= np.linalg.norm(affine[:3, :3], axis=0)
        output_properties['nibabel_stuff']['original_affine'] = affine
    rw.write_seg(final, str(output)+'.geometry1mm.nii.gz', output_properties)


def export_prediction_with_geometry(predicted_array_or_file, properties_dict, configuration_manager,
                                    plans_manager, dataset_json_dict_or_file, output_file_truncated,
                                    save_probabilities=False):
    from nnunetv2.inference.export_prediction import export_prediction_from_logits
    capture_model_geometry(predicted_array_or_file, properties_dict, configuration_manager,
                           plans_manager, dataset_json_dict_or_file, output_file_truncated)
    return export_prediction_from_logits(predicted_array_or_file, properties_dict, configuration_manager,
        plans_manager, dataset_json_dict_or_file, output_file_truncated, save_probabilities)


def configure_geometry_export_runtime():
    import nnunetv2.inference.predict_from_raw_data as runtime
    runtime.export_prediction_from_logits = export_prediction_with_geometry


def collect_geometry_cache(predictions, native, resolution, origin, label_map, artifact_prefix):
    """Vote 4-D model-grid labels and persist a receipt next to native output."""
    from .channels import _restore_autoflow_segmentation, _apply_label_map
    from .io import _read_nifti_segmentation
    from ..preprocess import majority_vote_labels_3d
    from ..geometry_grid import save_geometry_cache
    paths = [Path(str(path)[:-7]+'.geometry1mm.nii.gz') for path in predictions]
    if not artifact_prefix or not paths or not all(path.is_file() for path in paths):
        return {}
    values = [_restore_autoflow_segmentation(_apply_label_map(_read_nifti_segmentation(path), label_map))
              for path in paths]
    temporal = np.stack(values, axis=3) if len(values) > 1 else None
    labels = values[0] if temporal is None else majority_vote_labels_3d(temporal)
    path = str(artifact_prefix)+'.geometry1mm.npz'
    metadata = save_geometry_cache(path, labels, native, resolution, origin, 'model_logits_linear_then_classify', temporal_labels=temporal)
    return {'geometry_cache_path': path, 'geometry_cache': metadata}
