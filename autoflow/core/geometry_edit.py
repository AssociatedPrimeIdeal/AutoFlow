"""Geometry edit transactions and downstream reconstruction.

Large input/derived volume arrays are shared read-only in snapshots. Topology
and metadata are copied; no workspace mutation is published before success.
"""
import copy
import json
import os
import numpy as np

from .models import GraphData, ObjectKind, StepId
from .plane_state import invalidate_planes, refresh_plane_consistency
from ..algorithms.paths import _project_point_to_path
from ..task_control import check_cancelled


FIELDS = ('skeleton_points', 'skeleton_mask', 'graph', 'graph_hierarchy',
          'branch_labels', 'geometry_branch_labels', 'centerline_paths',
          'centerline_node_paths', 'centerline_paths_smooth', 'path_info',
          'forks', 'planes', 'multilabel_groups', 'scene_objects',
          'selected_path_index', 'pipeline', 'derived', 'pathline_cache',
          'pathline_seed_cache', 'active_pathline_plane_indices',
          'pathline_colors', 'analysis_revision')


def _array_memo(value, memo, visited):
    if id(value) in visited:
        return
    visited.add(id(value))
    if isinstance(value, np.ndarray):
        memo[id(value)] = value
    elif isinstance(value, dict):
        for child in value.values():
            _array_memo(child, memo, visited)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _array_memo(child, memo, visited)
    elif hasattr(value, '__dataclass_fields__'):
        _array_memo(vars(value), memo, visited)
    elif hasattr(value, 'IsA'):
        memo[id(value)] = value  # Cached VTK datasets are never edited here.


def geometry_snapshot(ws):
    values = {name: getattr(ws, name) for name in FIELDS}
    memo = {}
    _array_memo(values, memo, set())
    return dict(case_identity=(id(ws), str(ws.paths.flow_path)),
                values=copy.deepcopy(values, memo))


def restore_geometry_snapshot(ws, snapshot):
    if snapshot['case_identity'] != (id(ws), str(ws.paths.flow_path)):
        raise ValueError('Geometry history belongs to a different case')
    memo = {}
    _array_memo(snapshot['values'], memo, set())
    for name, value in copy.deepcopy(snapshot['values'], memo).items():
        setattr(ws, name, value)


def _manual(plane):
    placement = plane.placement or {}
    return (int(plane.path_index) < 0 or placement.get('manual_geometry')
            or placement.get('geometry_source', placement.get('source')) in {'manual', 'imported'}
            or int(plane.geometry_revision) > 0)


def _slot(plane):
    p = plane.placement or {}
    return (plane.group_name, int(plane.segmentation_label), p.get('schema'),
            p.get('placement_index'), p.get('position'), p.get('fraction'),
            p.get('requested_anchor', p.get('anchor')))


def _geometry_key(plane):
    return (plane.group_name, int(plane.segmentation_label),
            tuple(np.round(plane.center, 8)), tuple(np.round(plane.normal, 8)))


def _associate_manual_plane(ws, plane):
    candidates = []
    paths = ws.centerline_paths_smooth or ws.centerline_paths
    for index, path in enumerate(paths):
        if len(path) < 2:
            continue
        group = str(ws.path_info[index].get('group_name', ''))
        if plane.group_name and group != plane.group_name:
            continue
        projected, _, _, distance, _, _ = _project_point_to_path(path, plane.center)
        candidates.append((float(np.linalg.norm(projected-plane.center)), index, float(distance), group))
    plane.path_index = -1
    plane.distance = 0.
    if candidates:
        candidates.sort()
        distance, index, arc, group = candidates[0]
        # Association is metadata only: preserve the exact manual center/normal.
        if len(candidates) == 1 or candidates[1][0]-distance > 1e-6:
            plane.path_index, plane.distance, plane.group_name = index, arc, group
            plane.placement['association_distance_mm'] = distance
    plane.placement['manual_geometry'] = True
    plane.placement['association_status'] = 'review'  # User geometry survives independently of the regenerated path.


def prepare_geometry_edit(ws, engine, mode, group_name, points, edges=None):
    """Build a replacement topology without changing the live workspace/files."""
    if mode not in {'skeleton', 'graph'}:
        raise ValueError('Expected skeleton or graph editing')
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if not np.all(np.isfinite(points)):
        raise ValueError('Geometry coordinates must be finite')
    if group_name not in ws.multilabel_groups:
        raise ValueError('Select a vessel group before editing geometry')
    staged = copy.copy(ws)
    snapshot = geometry_snapshot(ws)
    for name, value in snapshot['values'].items():
        setattr(staged, name, value)
    # Engine metadata/preprocessing must already belong to the loaded case.
    staged.paths = copy.copy(ws.paths)
    old_planes = list(staged.planes)
    group = staged.multilabel_groups[group_name]
    if mode == 'skeleton':
        group['skeleton_points'] = points.copy()
        spacing = np.asarray(staged.skeleton_spacing, dtype=float)
        shape = np.asarray(staged.skeleton_labels).shape
        mask = np.zeros(shape, dtype=bool)
        voxels = np.rint(points/spacing).astype(int)
        valid = np.all((voxels >= 0) & (voxels < np.asarray(shape)), axis=1)
        if len(voxels[valid]):
            mask[tuple(voxels[valid].T)] = True
        group['skeleton_mask'] = mask
        sets = [g['skeleton_points'] for g in staged.multilabel_groups.values() if g.get('skeleton_points') is not None]
        staged.skeleton_points = np.vstack(sets) if sets else np.empty((0, 3))
        masks = [g['skeleton_mask'] for g in staged.multilabel_groups.values() if g.get('skeleton_mask') is not None]
        staged.skeleton_mask = np.logical_or.reduce(masks) if masks else mask
        rebuild = {group_name}
    else:
        edges = np.asarray(edges if edges is not None else [], dtype=int).reshape(-1, 2)
        if edges.size and (np.any(edges < 0) or np.any(edges >= len(points)) or np.any(edges[:, 0] == edges[:, 1])):
            raise ValueError('Graph edges must reference two distinct existing nodes')
        edges = np.asarray(sorted({tuple(sorted(map(int, e))) for e in edges}), dtype=int).reshape(-1, 2)
        group['graph'] = GraphData(points=points.copy(), edges=edges)
        group['manual_graph_edit'] = True
        rebuild = set()
    result = engine._step_generate_graph(staged, rebuild_groups=rebuild, write_output=False)
    if not result.success or result.skipped:
        raise RuntimeError(result.message)
    if old_planes:
        automatic_groups = {p.group_name for p in old_planes if not _manual(p)
                            or (p.placement.get('schema') and p.placement.get('geometry_source', p.placement.get('source')) != 'imported')}
        result = engine._step_generate_planes(staged, write_output=False, generate_missing_graph=False, use_position_file=False)
        if not result.success or result.skipped:
            raise RuntimeError(result.message)
        staged.planes = [p for p in staged.planes if p.group_name in automatic_groups]
        # Restore unchanged planes/measurements by geometry, independent of reindexed paths.
        old_by_geometry = {_geometry_key(p): p for p in old_planes if not _manual(p)}
        for index, plane in enumerate(staged.planes):
            previous = old_by_geometry.get(_geometry_key(plane))
            if previous is not None and previous.group_name != group_name:
                retained = copy.deepcopy(previous)
                retained.path_index, retained.distance = plane.path_index, plane.distance
                staged.planes[index] = retained
        for previous in old_planes:
            if not _manual(previous):
                continue
            manual = copy.deepcopy(previous)
            # A moved generated plane replaces its unique generated slot.
            if previous.placement.get('schema') and previous.placement.get('geometry_source', previous.placement.get('source')) != 'imported':
                matches = [i for i, p in enumerate(staged.planes) if _slot(p) == _slot(previous)]
                if len(matches) == 1:
                    staged.planes.pop(matches[0])
            _associate_manual_plane(staged, manual)
            staged.planes.append(manual)
        plane_qc = copy.deepcopy(staged.derived.plane_qc)
        affected = [p.plane_id for p in staged.planes if p.group_name in {'', group_name}]
        invalidate_planes(staged, affected, 'Centerline geometry changed', geometry=False)
        staged.derived.plane_metrics = [dict(p.metrics or {}) for p in staged.planes]
        plane_qc.update(refresh_plane_consistency(staged))
        staged.derived.plane_qc = plane_qc
        staged.derived.plane_qc['actual_plane_count'] = len(staged.planes)
        staged.derived.plane_qc['manual_planes_preserved'] = sum(_manual(p) for p in staged.planes)
    for name, state in staged.multilabel_groups.items():
        state['planes'] = [p for p in staged.planes if p.group_name == name]
        indices = [i for i, p in enumerate(staged.planes) if p.group_name == name]
        state['plane_index_offset'] = min(indices, default=0)
    staged.remove_objects_by_prefix('plane_')
    for index, plane in enumerate(staged.planes):
        staged.add_object(name=f'plane {index}', kind=ObjectKind.PLANE,
            data_key=engine._plane_data_key('plane', staged, index), group_name=plane.group_name,
            color=staged.skeleton_params.scene_color_for_group(plane.group_name, 'plane') if plane.group_name else 'yellow',
            visible=True, opacity=1., line_width=2)
    staged.pipeline.mark_done(StepId.EDIT_SKELETON if mode == 'skeleton' else StepId.EDIT_GRAPH, skipped=False)
    staged.graph_hierarchy['manual_edit'] = dict(
        mode=mode, group_name=group_name,
        automatic_graph_cleanup_reapplied=mode == 'skeleton',
        node_count=len(staged.multilabel_groups[group_name]['graph'].points),
        edge_count=len(staged.multilabel_groups[group_name]['graph'].edges))
    staged.analysis_revision = int(ws.analysis_revision)+1
    check_cancelled()
    return {name: getattr(staged, name) for name in FIELDS}


def publish_geometry_edit(ws, values):
    for name in FIELDS:
        setattr(ws, name, values[name])


def persist_geometry_edit(ws, engine):
    output = engine._output_dir(ws)
    with open(os.path.join(output, 'graph_hierarchy.json'), 'w', encoding='utf-8') as handle:
        json.dump(ws.graph_hierarchy, handle, indent=2)
    engine._save_planes_json(ws)
    with open(os.path.join(output, 'plane_qc.json'), 'w', encoding='utf-8') as handle:
        json.dump(ws.derived.plane_qc, handle, indent=2)
