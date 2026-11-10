"""Difference of existing plane means, with a shared pressure-gauge check."""

import numpy as np
from scipy.ndimage import generate_binary_structure, label

from ...task_control import check_cancelled
from ..surfaces import _build_branch_grid
from .sampling import (_build_mask_phase_lookup, _build_plane_support_mesh_cache,
    _add_labeled_plane_support_meshes, _get_plane_metric_slice_spec,
    _sample_field_from_slice_spec, _target_label_for_plane)


def compute_plane_pressure_difference(planes, metrics, mask, spacing, origin, pressure, support,
                                      *, branch_labels=None, labels3=None, labels4=None,
                                      field_spacing=None):
    """Return first-minus-second in Pa; unavailable/incomparable phases are null.

    Geometry uses the same temporally resolved ROI and cut-cell sampler as the
    saved plane means. Component IDs are sampled with nearest-neighbour lookup,
    and every contributing sample must use one shared connected pressure gauge.
    """
    if len(planes) != 2 or len(metrics) != 2:
        raise ValueError('Select exactly two planes.')
    pressure = np.asarray(pressure)
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim == 3:
        mask = mask[..., None]
    if pressure.ndim != 4 or mask.shape[3] != pressure.shape[3]:
        raise ValueError('Pressure and segmentation phases must match.')
    support = np.asarray(support, dtype=bool)
    if support.ndim == 3:
        support = support[..., None]
    if support.shape[:3] != pressure.shape[:3] or support.shape[3] not in (1, pressure.shape[3]):
        raise ValueError('Valid pressure support is unavailable or does not match pressure.')
    means = [np.asarray(m.get('relative_pressure_mean_Pa_t', []), dtype=float).reshape(-1) for m in metrics]
    nt = pressure.shape[3]
    if any(values.size != nt for values in means):
        raise ValueError('Compute current relative-pressure plane metrics first.')
    spacing, origin = np.asarray(spacing, float), np.asarray(origin, float)
    grid_origin = origin - .5 * spacing if field_spacing is not None else origin
    lookup = _build_mask_phase_lookup(mask)
    meshes = _build_plane_support_mesh_cache(mask, lookup, spacing, grid_origin)
    _add_labeled_plane_support_meshes(meshes, mask, labels4 if labels4 is not None else labels3,
                                     planes, spacing, grid_origin)
    branch_grid = _build_branch_grid(branch_labels, spacing, grid_origin)
    targets = [_target_label_for_plane(p, branch_labels, spacing, origin) for p in planes]
    caches = [{}, {}]
    differences, reasons = [], []
    for t in range(nt):
        check_cancelled()
        missing = []
        for i, metric in enumerate(metrics):
            if np.isfinite(means[i][t]):
                continue
            index = metric.get('plane_index', i)
            valid = metric.get('sample_valid_t', [])
            counts = metric.get('relative_pressure_valid_cell_count_t', [])
            if len(valid) > t and not valid[t]:
                detail = 'ROI is empty or has no finite velocity samples in this phase'
            elif len(counts) > t and not counts[t]:
                detail = 'ROI has no valid pressure support in this phase'
            else:
                detail = 'relative-pressure samples are unavailable; update plane metrics'
            missing.append(f'Plane {index}: {detail}')
        difference, reason = None, '; '.join(missing) or 'No comparable supported pressure samples'
        if np.isfinite(means[0][t]) and np.isfinite(means[1][t]):
            valid_support = support[..., min(t, support.shape[3]-1)] & np.isfinite(pressure[..., t])
            components, _ = label(valid_support, structure=generate_binary_structure(3, 1))
            gauges = []
            for i, plane in enumerate(planes):
                spec = _get_plane_metric_slice_spec(caches[i], mask, lookup, meshes, plane, t,
                    spacing, origin, branch_grid=branch_grid, target_label=targets[i],
                    segmentation_labels_3d=labels3, segmentation_labels_4d=labels4,
                    voxel_centered=field_spacing is not None)
                if spec is None:
                    gauges.append(set())
                    continue
                def sample(array, name):
                    return np.asarray(_sample_field_from_slice_spec(array, mask.shape[:3], name, spec,
                        field_spacing=field_spacing, mask_spacing=spacing, origin=origin)).reshape(-1)
                values = sample(pressure[..., t], 'relative_pressure')
                supported = sample(valid_support, 'pressure_support').astype(bool)
                ids = sample(components, 'component_support')
                valid = supported & np.isfinite(values) & (np.asarray(spec['areas']) > 0)
                gauges.append(set(int(v) for v in ids[valid]))
            if len(gauges[0]) == 1 and gauges[0] == gauges[1] and 0 not in gauges[0]:
                difference = float(means[0][t] - means[1][t])
                reason = ''
            elif gauges[0] and gauges[1]:
                reason = 'Planes do not share one connected pressure reference'
        differences.append(difference)
        reasons.append(reason)
    finite = [v for v in differences if v is not None]
    return {'pressure_difference_Pa_t': differences,
            'valid_t': [v is not None for v in differences], 'reason_t': reasons,
            'mean_Pa': float(np.mean(finite)) if len(finite) == nt else None,
            'peak_abs_Pa': float(np.max(np.abs(finite))) if finite else None}
