"""Native segmentation for measurement and a separate 1-mm topology grid."""
from pathlib import Path
from dataclasses import dataclass
import hashlib
import json
import os
import uuid

import numpy as np
from scipy.ndimage import affine_transform

SCHEMA = 'autoflow.geometry_grid.v1'
QUANTITATIVE_POLICY = 'native_measurement_1mm_topology_v3'


def has_source_geometry(ws):
    """Whether the validated fine grid is supplied geometry, rather than an upsample.

    A source name alone is insufficient: the cache must have passed its native
    digest/spacing/origin checks. Legacy 3-D caches with reconstructed temporal
    labels cannot supply a faithful phase-wise measurement mask.
    """
    info = getattr(ws, 'segmentation_geometry', {}) or {}
    return bool((info.get('cache_hit') or info.get('snapshot_cache_hit'))
                and info.get('source') != 'native_labels_linear_argmax'
                and not info.get('temporal_fallback'))


@dataclass(frozen=True)
class MeasurementSegmentation:
    binary: object
    labels: object
    labels_3d: object
    branches: object
    spacing: np.ndarray
    source: str


def measurement_segmentation(ws):
    """Use the active source's native labels for all quantitative ROIs.

    nnUNet already exports its prediction on the acquisition grid before
    classification. Its separate 1-mm cache serves topology/display only.
    Other sources use their input labels, including the bound native result
    of an explicitly imported or manually edited 1-mm label volume.
    """
    binary, labels, voted = ws.segmask_binary, ws.segmask_labels, ws.segmask_labels_3d
    if binary is None and ws.segmask_raw is not None:
        labels = ws.segmask_raw
        binary = np.asarray(labels) > 0
        from .preprocess import majority_vote_labels_3d
        voted = majority_vote_labels_3d(labels)
    if binary is None and ws.segmentation_analysis_binary is not None:
        # Compatibility for manually constructed 1-mm workspaces.
        binary, labels, voted = (ws.segmentation_analysis_binary, ws.segmentation_analysis_labels,
                                 ws.segmentation_analysis_labels_3d)
    source = ('auto_native_labels' if ws.segmentation.active_source == 'auto'
              else 'input_native_labels')
    return MeasurementSegmentation(binary, labels, voted, ws.branch_labels,
                                   np.asarray(ws.resolution, float), source)


def retain_native_measurement_masks(ws, specs, *, voted_labels=None):
    """Run declared cleanup on source cells, without skeleton-grid round trips."""
    from .preprocess import (filter_segmask_labels, filter_4d_labeled_components,
                             majority_vote_labels_3d, filter_connected_components)
    labels = filter_segmask_labels(ws.segmask_raw)
    if ws.segmentation.cleanup_4d_components and labels.ndim == 4:
        labels = filter_4d_labeled_components(labels, ws.resolution,
            mode=ws.segmentation.cleanup_4d_mode,
            min_volume_mm3=ws.segmentation.cleanup_4d_min_volume_mm3)
    voted = (np.asarray(voted_labels) if voted_labels is not None and not ws.segmentation.cleanup_4d_components
             else majority_vote_labels_3d(labels))
    count = max(1, int(ws.time_count()))
    temporal = labels if labels.ndim == 4 else np.broadcast_to(labels[..., None], labels.shape+(count,))
    binary = np.zeros(temporal.shape, bool)
    retained = np.zeros_like(voted)
    for spec in specs:
        name, values = str(spec['name']), spec['labels']
        params = ws.skeleton_params.params_for_group(name)
        group = np.isin(temporal, values)
        stable = np.isin(voted, values)
        if params.remove_small_cc:
            options = dict(mode=params.cc_filter_mode, min_volume_mm3=params.min_cc_volume_mm3,
                           rel_min_ratio=params.cc_rel_min_ratio)
            stable = filter_connected_components(stable, ws.resolution, **options)
            for frame in range(group.shape[3]):
                group[..., frame] = filter_connected_components(group[..., frame], ws.resolution, **options)
        binary |= group
        retained[stable] = voted[stable]
        if name in ws.multilabel_groups:
            ws.multilabel_groups[name].update(segmask_binary=group, segmask_3d=stable, clean_mask_3d=stable)
    ws.segmask_labels, ws.segmask_labels_3d = labels, retained
    ws.segmask_binary, ws.segmask_3d = binary, retained > 0


def migrate_measurement_results(ws):
    """Keep old plane placements, but invalidate measurements from a prior grid."""
    previous = ws.segmentation_geometry.get('quantitative_policy')
    if previous == QUANTITATIVE_POLICY or not ws.segmentation_geometry:
        return False
    # A saved fine-grid cache from before explicit policy receipts may also
    # have supplied the old measurement geometry on a non-native grid.
    if not previous and np.allclose(ws.resolution, 1.):
        return False
    from ..core.models import DerivedResults, StepId, ObjectKind
    reason = 'Measurement segmentation now uses the native acquisition grid'
    for plane in ws.planes:
        if plane.metrics:
            plane.metrics = dict(plane.metrics, validity='stale', stale_reason=reason)
    for state in ws.multilabel_groups.values():
        for plane in state.get('planes', []):
            if plane.metrics:
                plane.metrics = dict(plane.metrics, validity='stale', stale_reason=reason)
    ws.derived = DerivedResults(plane_qc=ws.derived.plane_qc)
    for step in (StepId.COMPUTE_PLANE_METRICS, StepId.COMPUTE_DERIVED_METRICS,
                 StepId.COMPUTE_PWV, StepId.GENERATE_STREAMLINES, StepId.PLANE_STREAMLINES):
        ws.pipeline.completed.pop(step.value, None)
        ws.pipeline.skipped.pop(step.value, None)
    for uid,obj in list(ws.scene_objects.items()):
        if obj.kind == ObjectKind.METRIC:
            ws.scene_objects.pop(uid)
    ws.input_state.metadata['measurement_policy_migrated_from'] = previous or 'legacy_1mm_geometry'
    return True


def segmentation_digest(labels):
    array = np.ascontiguousarray(labels, dtype=np.int16)
    digest = hashlib.sha256(str(array.shape).encode('ascii'))
    digest.update(memoryview(array).cast('B'))
    return digest.hexdigest()


def grid_shape(shape, spacing, target=1.0):
    spacing = np.asarray(spacing, dtype=float).reshape(3)
    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError('Geometry spacing must be positive and finite')
    if not np.isfinite(target) or float(target) <= 0:
        raise ValueError('Geometry target spacing must be positive and finite')
    steps = (np.asarray(shape[:3])-1)*spacing/target
    # NIfTI/H5 spacing often round-trips through float32. Do not add a full
    # output voxel when a mathematically integral extent differs by a few ULPs.
    tolerance = 8*np.finfo(np.float32).eps*np.maximum(1., np.abs(steps))
    steps = np.where(np.abs(steps-np.rint(steps)) <= tolerance, np.rint(steps), steps)
    return tuple((np.ceil(steps).astype(int)+1).tolist())


def resample_labels(labels, spacing, target=1.0, shape=None):
    array = np.asarray(labels)
    if array.ndim != 3:
        raise ValueError('Geometry labels must be 3-D')
    shape = grid_shape(array.shape, spacing, target) if shape is None else tuple(shape)
    if tuple(array.shape) == shape and np.allclose(spacing, target, rtol=0, atol=1e-7):
        return array.astype(np.int16, copy=True)
    return affine_transform(array, np.diag(float(target)/np.asarray(spacing)),
                            output_shape=shape, order=0, mode='constant', cval=0,
                            prefilter=False).astype(np.int16)


def save_geometry_cache(path, labels, native, native_spacing, origin, source, temporal_labels=None):
    """Write a portable 3-D cache with a receipt for the native segmentation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(schema=SCHEMA, spacing_mm=[1., 1., 1.], origin_mm=np.asarray(origin).tolist(),
                    native_spacing_mm=np.asarray(native_spacing).tolist(),
                    native_shape=list(np.asarray(native).shape), native_sha256=segmentation_digest(native),
                    source=str(source), geometry_sha256=segmentation_digest(labels),
                    temporal_reduction='majority_vote_lowest_label_ties')
    arrays = dict(labels=np.asarray(labels, dtype=np.int16))
    if temporal_labels is not None:
        temporal = np.asarray(temporal_labels, dtype=np.int16)
        if temporal.ndim != 4 or temporal.shape[:3] != arrays['labels'].shape:
            raise ValueError('Temporal geometry must be XYZT on the same geometry grid')
        arrays['labels_4d'] = temporal
        metadata.update(temporal_shape=list(temporal.shape), temporal_sha256=segmentation_digest(temporal))
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp.npz')
    try:
        with open(temporary, 'xb') as stream:
            np.savez_compressed(stream, **arrays, metadata=np.asarray(json.dumps(metadata)))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return dict(metadata, path=str(path))


def read_geometry_cache(path, dataset=None, *, temporal=False):
    if dataset:
        import h5py
        with h5py.File(path, 'r') as file:
            data = file[dataset]
            values = file[dataset+'_4d'] if temporal and dataset+'_4d' in file else data
            return np.asarray(values, dtype=np.int16), json.loads(data.attrs['geometry_json'])
    with np.load(path, allow_pickle=False) as data:
        return data['labels_4d'] if temporal and 'labels_4d' in data else data['labels'], json.loads(str(data['metadata']))


def prepare_segmentation_geometry(ws, *, display=False):
    """Prepare the shared 1-mm topology/display copy, not the measurement ROI.

    Original source labels and measured fields stay on their acquisition grid.
    A model cache is accepted only when its source receipt still matches.
    """
    from .preprocess import majority_vote_labels_3d
    from .label_resampling import resample_label_linear
    working = None
    if display:
        working = (ws.segmentation.working_labels_4d if ws.segmentation.working_labels_4d is not None
                   else ws.segmentation.working_labels_3d)
    native = working if working is not None else ws.segmask_raw
    if native is None:
        return None, np.ones(3), {'source': 'unavailable'}
    native = np.asarray(native, dtype=np.int16)
    provenance = ws.get_active_segmentation_provenance()
    candidates = []
    if provenance.get('geometry_cache_path'):
        candidates.append((provenance['geometry_cache_path'], None))
    metadata = provenance.get('metadata', {})
    if metadata.get('geometry_cache_dataset'):
        candidates.append((ws.paths.flow_path, metadata['geometry_cache_dataset']))
    signature = (id(native), tuple(native.shape),
                 int(ws.segmentation.revision), tuple(np.asarray(ws.resolution)),
                 tuple(np.asarray(ws.origin)), tuple(candidates), working is not None)
    cache_name = '_segmentation_preview_grid_cache' if working is not None else '_segmentation_geometry_cache'
    previous = getattr(ws, cache_name, None)
    if previous is not None and previous[0] == signature:
        labels, spacing, receipt, temporal = previous[1]
        return labels, spacing, dict(receipt, memory_cache_hit=True)
    labels, temporal, receipt = None, None, None
    if working is not None:
        # In-app editing operates directly on the shared 1-mm display grid.
        temporal = native
        labels = majority_vote_labels_3d(temporal)
        receipt = dict(schema=SCHEMA, source='working_1mm_labels', cache_hit=False)
    digest = None
    rejection = []
    restored_labels = getattr(ws, 'segmentation_labels_1mm_3d', None)
    restored_info = getattr(ws, 'segmentation_geometry', {})
    if working is None and restored_labels is not None and restored_info.get('native_sha256'):
        digest = segmentation_digest(native)
        if (restored_info['native_sha256'] == digest
                and tuple(restored_labels.shape) == grid_shape(native.shape, ws.resolution)
                and np.allclose(restored_info.get('native_spacing_mm', ws.resolution), ws.resolution)
                and np.allclose(restored_info.get('origin_mm', ws.origin), ws.origin)):
            labels = restored_labels
            temporal = getattr(ws, 'segmentation_labels_1mm', None)
            receipt = dict(restored_info, snapshot_cache_hit=True)
    if working is None and labels is None:
        for path, dataset in candidates:
            try:
                candidate, info = read_geometry_cache(path, dataset)
                digest = digest or segmentation_digest(native)
                expected = grid_shape(native.shape, ws.resolution, 1.)
                if (info.get('schema') != SCHEMA or info.get('native_sha256') != digest
                        or tuple(candidate.shape) != expected
                        or not np.allclose(info['native_spacing_mm'], ws.resolution)
                        or not np.allclose(info['origin_mm'], ws.origin)):
                    raise ValueError('receipt does not match active segmentation/geometry')
                if info.get('geometry_sha256') and info['geometry_sha256'] != segmentation_digest(candidate):
                    raise ValueError('geometry cache checksum mismatch')
                temporal_candidate, _ = read_geometry_cache(path, dataset, temporal=True)
                if temporal_candidate.ndim == 4:
                    expected_time = native.shape[3] if native.ndim == 4 else 1
                    if temporal_candidate.shape != expected+(expected_time,):
                        raise ValueError('temporal geometry shape does not match active segmentation')
                    if info.get('temporal_sha256') and info['temporal_sha256'] != segmentation_digest(temporal_candidate):
                        raise ValueError('temporal geometry checksum mismatch')
                    temporal = temporal_candidate
                elif native.ndim == 3:
                    temporal = candidate
                elif np.all(native == native[..., :1]):
                    temporal = np.broadcast_to(candidate[..., None], candidate.shape+(native.shape[3],))
                labels, receipt = candidate, dict(info, cache_hit=True)
                break
            except (OSError, KeyError, ValueError, TypeError) as error:
                rejection.append(str(error))
    if labels is None:
        temporal, resampling = resample_label_linear(native, ws.resolution, return_info=True)
        labels = majority_vote_labels_3d(temporal)
        receipt = dict(resampling, schema=SCHEMA, source='native_labels_linear_argmax',
                       origin_mm=np.asarray(ws.origin).tolist(), cache_hit=False)
    elif temporal is None:
        # Old 3-D-only caches retain exact model geometry for topology. Their
        # missing per-phase geometry cannot be reconstructed as model logits;
        # use the label-linear fallback and state that limitation explicitly.
        temporal, resampling = resample_label_linear(native, ws.resolution, return_info=True)
        receipt['temporal_fallback'] = dict(resampling, reason='legacy_model_cache_has_no_temporal_labels')
    receipt['temporal_source'] = ('model_geometry_cache' if receipt.get('cache_hit') and 'temporal_fallback' not in receipt
                                  else 'per_label_linear_argmax')
    receipt['rejected_caches'] = rejection
    receipt['shape'] = list(labels.shape)
    receipt.setdefault('native_sha256', digest or segmentation_digest(native))
    receipt.setdefault('native_spacing_mm', np.asarray(ws.resolution).tolist())
    receipt.setdefault('origin_mm', np.asarray(ws.origin).tolist())
    spacing = np.ones(3)
    setattr(ws, cache_name, (signature, (labels, spacing, receipt, temporal)))
    if working is None:
        ws.segmentation_labels_1mm = temporal
        ws.segmentation_labels_1mm_3d = labels
        ws.segmentation_geometry = dict(receipt)
    return labels, spacing, receipt


def segmentation_geometry_data(ws, *, display=False):
    """Return XYZ[T] geometry labels, sharing the validated prepared array."""
    prepare_segmentation_geometry(ws, display=display)
    preview = display and (ws.segmentation.working_labels_4d is not None or ws.segmentation.working_labels_3d is not None)
    cached = getattr(ws, '_segmentation_preview_grid_cache' if preview else '_segmentation_geometry_cache', None)
    return None if cached is None else cached[1][3]


def prepare_skeleton_geometry(ws):
    """Skeleton and segmentation display share the same physical 1-mm grid."""
    labels, spacing, receipt = prepare_segmentation_geometry(ws)
    ws.skeleton_params.geometry_spacing_mm = 1.
    processed = getattr(ws, 'segmentation_analysis_labels_3d', None)
    if processed is not None and processed.shape == labels.shape:
        labels = processed
    return labels, spacing, receipt


def labels_on_flow_grid(labels, ws):
    """Project a geometry mask to acquired velocity voxel centers when needed.

    Supplied fine geometry needs this adapter for native-field algorithms.
    Native input segmentations use their own cells directly for measurement.
    """
    source = np.asarray(labels)
    field = ws.flow_raw if ws.flow_raw is not None else ws.segmask_raw
    shape = tuple(np.asarray(field).shape[:3])
    if source.shape[:3] == shape and np.allclose(ws.resolution, 1.):
        return source.copy()
    matrix = np.diag(np.asarray(ws.resolution, float))
    if source.ndim == 3:
        return affine_transform(source, matrix, output_shape=shape, order=0, mode='constant', cval=0, prefilter=False)
    output = np.empty(shape+(source.shape[3],), source.dtype)
    for frame in range(source.shape[3]):
        output[..., frame] = affine_transform(source[..., frame], matrix, output_shape=shape,
                                              order=0, mode='constant', cval=0, prefilter=False)
    return output


def bind_segmentation_geometry(ws, source, labels, provenance, directory):
    """Keep imported/edited 1-mm labels exact and bind their native adapter."""
    from .preprocess import majority_vote_labels_3d
    fine = np.asarray(labels, dtype=np.int16)
    native = labels_on_flow_grid(fine, ws)
    path = Path(directory) / ('segmentation_geometry_' + uuid.uuid4().hex + '.npz')
    save_geometry_cache(path, majority_vote_labels_3d(fine), native, ws.resolution,
                        ws.origin, source, temporal_labels=fine if fine.ndim == 4 else None)
    info = dict(provenance, geometry_cache_path=str(path), geometry_source='imported_or_edited_1mm')
    ws.set_segmentation_source(source, native, provenance=info)
    return native


def field_on_geometry_grid(field, ws, shape):
    """Temporary continuous image interpolation for external editing features."""
    source = np.asarray(field)
    trailing = source.shape[3:]
    output = np.empty(tuple(shape)+trailing, dtype=np.float32)
    for index in np.ndindex(trailing):
        output[(slice(None),)*3+index] = affine_transform(
            source[(slice(None),)*3+index], np.diag(1./np.asarray(ws.resolution)),
            output_shape=shape, order=1, mode='nearest', prefilter=False)
    return output


_GEOMETRY_ARRAY_FIELDS = ('segmentation_labels_1mm', 'segmentation_labels_1mm_3d',
    'segmentation_analysis_labels', 'segmentation_analysis_labels_3d',
    'segmentation_analysis_binary', 'segmentation_analysis_mask', 'geometry_branch_labels')
_GROUP_GEOMETRY_FIELDS = ('geometry_labels_3d', 'geometry_binary', 'geometry_mask_3d',
                         'geometry_clean_mask_3d', 'geometry_branch_labels')


def pack_geometry_arrays(ws):
    """Portable compressed labels without expanding fine 4-D arrays into JSON."""
    import base64, io
    arrays = {name: getattr(ws, name) for name in _GEOMETRY_ARRAY_FIELDS if getattr(ws, name, None) is not None}
    for index, name in enumerate(ws.group_order):
        for field in _GROUP_GEOMETRY_FIELDS:
            value = ws.multilabel_groups.get(name, {}).get(field)
            if value is not None:
                arrays[f'group_{index}__{field}'] = value
    if not arrays:
        return None
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return base64.b64encode(buffer.getvalue()).decode('ascii')


def restore_geometry_arrays(ws, payload):
    import base64, io
    ws._segmentation_geometry_cache = None
    ws._segmentation_preview_grid_cache = None
    for field in _GEOMETRY_ARRAY_FIELDS:
        setattr(ws, field, None)
    if not payload:
        return
    with np.load(io.BytesIO(base64.b64decode(payload)), allow_pickle=False) as data:
        for field in _GEOMETRY_ARRAY_FIELDS:
            if field in data:
                setattr(ws, field, data[field])
        for index, name in enumerate(ws.group_order):
            state = ws.multilabel_groups.get(name, {})
            for field in _GROUP_GEOMETRY_FIELDS:
                key = f'group_{index}__{field}'
                if key in data:
                    state[field] = data[key]
    # Older snapshots used the topology upsample for quantitative ROIs. Keep
    # their source/model cache and graph geometry, but require a fresh source-
    # specific preprocessing pass before calculating new measurements.
    if ws.segmentation_geometry.get('quantitative_policy') != QUANTITATIVE_POLICY:
        for field in ('segmentation_analysis_labels', 'segmentation_analysis_labels_3d',
                      'segmentation_analysis_binary', 'segmentation_analysis_mask'):
            setattr(ws, field, None)
