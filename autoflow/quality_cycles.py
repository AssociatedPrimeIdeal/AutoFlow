"""Read-only warnings for residual undirected cycles in the current graph."""
import hashlib
import json
import networkx as nx
import numpy as np


RETENTION_REASONS = {
    'preserve_component_or_endpoint': 'Removal would disconnect a physical component or lose an original endpoint.',
    'preserve_positive_label_coverage': 'Removal would erase all retained graph support for a positive label.',
    'flow_C_at_or_above_threshold': 'C is at or above the threshold; only strictly lower C is eligible.',
    'preserve_label_component_or_interface': 'Low C, but removal would violate protected component, endpoint or label-interface connectivity.',
    'cross_label_edge': 'Historical report: the old filter protected positive cross-label edges; regenerate the graph for the current policy.',
    'unknown_label_edge': 'An endpoint has no positive source label; this edge is outside cleanup.',
    'edge_crosses_label_boundary': 'The edge crosses another label/background between its endpoints and is protected.',
    'flow_signal_unavailable': 'No valid nonzero flow was available for the cleanup unit.',
    'cleanup_audit_unavailable': 'The saved cleanup audit does not explain this edge; regenerate the graph for a decision record.',
}


def _edge_key(left, right):
    return tuple(sorted((tuple(np.round(left, 9)), tuple(np.round(right, 9)))))


def _cleanup_evidence(workspace):
    """Index actual pre-pruning cleanup decisions by group and physical edge."""
    groups = {str(row.get('group_name', '')): row.get('cycle_cleanup', {})
              for row in (getattr(workspace, 'graph_hierarchy', {}) or {}).get('groups', [])}
    for name, state in (getattr(workspace, 'multilabel_groups', {}) or {}).items():
        if state.get('cycle_cleanup'):
            groups[str(name)] = state['cycle_cleanup']
    indexed = {}
    for group, report in groups.items():
        records = list(report.get('path_evaluations', []))
        # Older snapshots recorded only low-C paths rejected by the guard.
        if not records:
            records = [dict(row, decision='retained') for row in report.get('rejected_paths', [])]
        records += list(report.get('protected_edges', []))
        for index, record in enumerate(records):
            if record.get('decision') == 'removed':
                continue
            curve = np.asarray(record.get('points_mm', []), dtype=float).reshape(-1, 3)
            identifier = str(record.get('audit_id') or hashlib.sha256(
                json.dumps((group, record.get('reason'), sorted(map(tuple, curve.tolist())))).encode()).hexdigest()[:16])
            item = dict(unit_id=identifier, group_name=group, label_id=record.get('label'),
                node_labels=record.get('node_labels'),
                C=None if record.get('reason') == 'flow_signal_unavailable' else record.get('direction_confidence'),
                raw_C=record.get('direction_confidence') if record.get('reason') == 'flow_signal_unavailable' else None,
                signed_alignment=record.get('flow_alignment'),
                threshold=record.get('flow_threshold', report.get('flow_threshold')),
                reason=record.get('reason', 'cleanup_audit_unavailable'),
                decision=record.get('decision', 'retained'), stage=report.get('stage', 'before_terminal_pruning'),
                evidence_source='saved_cleanup_decision')
            item['explanation'] = RETENTION_REASONS.get(item['reason'], item['reason'])
            for left, right in zip(curve, curve[1:]):
                indexed.setdefault((group, _edge_key(left, right)), {})[identifier] = item
    return indexed


def _retention_units(points, closed, groups, indexed, edge_paths):
    units, missing = {}, 0
    for a, b in zip(closed, closed[1:]):
        evidence = {}
        key = _edge_key(points[a], points[b])
        for group in groups:
            evidence.update(indexed.get((group, key), {}))
        if not evidence:
            missing += 1
            continue
        for identifier, item in evidence.items():
            if identifier not in units:
                units[identifier] = dict(item, shared_cycle_edge_count=0, path_indices=set())
            units[identifier]['shared_cycle_edge_count'] += 1
            units[identifier]['path_indices'].update(edge_paths.get(tuple(sorted((a, b))), set()))
    if missing:
        units['unavailable'] = dict(unit_id='unavailable', C=None, threshold=None,
            reason='cleanup_audit_unavailable', explanation=RETENTION_REASONS['cleanup_audit_unavailable'],
            shared_cycle_edge_count=missing, path_indices=set(), evidence_source='unavailable')
    for item in units.values():
        item['path_indices'] = sorted(item['path_indices'])
    return sorted(units.values(), key=lambda item: (item['reason'], item['unit_id']))


def build_cycle_check(workspace, make_check):
    """Report an independent cycle basis, including split/rejoin flow DAGs."""
    graph = getattr(workspace, 'graph', None)
    points = np.asarray(getattr(graph, 'points', []), dtype=float).reshape(-1, 3)
    edges = np.asarray(getattr(graph, 'edges', []), dtype=int).reshape(-1, 2)
    if not len(points):
        return make_check('centerline.cycles', 'Centerline & Planes', 'not_run',
            'Residual graph cycles', 'Generate Graph has not produced a graph.',
            action='Generate the graph to evaluate residual undirected cycles.')
    finite = np.all(np.isfinite(points), axis=1)
    network = nx.Graph()
    network.add_nodes_from(np.flatnonzero(finite).tolist())
    network.add_edges_from((int(a), int(b)) for a, b in edges
        if 0 <= a < len(points) and 0 <= b < len(points) and a != b and finite[a] and finite[b])
    labels = getattr(workspace, 'segmentation_analysis_labels_3d', None)
    spacing = np.ones(3)
    if labels is None:
        labels = getattr(workspace, 'segmask_labels_3d', None)
        spacing = np.asarray(workspace.resolution, dtype=float)
    sampled = np.zeros(len(points), dtype=np.int64)
    if labels is not None:
        from .algorithms.graph import _sample_graph_labels
        sampled[finite] = _sample_graph_labels(points[finite], labels, spacing, np.zeros(3))
    names = {int(value): str(name) for name, value in workspace.skeleton_params.label_map.items()}
    names.update({int(value): str(name) for value, name in workspace.segmentation.label_names.items()})
    edge_paths = {}
    for path_id, nodes in enumerate(getattr(workspace, 'centerline_node_paths', []) or []):
        for left, right in zip(nodes, nodes[1:]):
            edge_paths.setdefault(tuple(sorted((int(left), int(right)))), set()).add(path_id)
    indexed = _cleanup_evidence(workspace)
    cycles = []
    for nodes in nx.cycle_basis(network):
        closed = nodes + nodes[:1]
        curve = points[closed]
        geometry = sorted(tuple(row) for row in points[nodes].tolist())
        cycle_id = hashlib.sha256(json.dumps(geometry).encode()).hexdigest()[:16]
        label_ids = sorted({int(value) for value in sampled[nodes] if int(value) > 0})
        path_ids = sorted(set().union(*(edge_paths.get(tuple(sorted((a, b))), set())
            for a, b in zip(closed, closed[1:]))))
        groups = sorted({str(workspace.path_info[index].get('group_name', '')) for index in path_ids
            if index < len(workspace.path_info) and workspace.path_info[index].get('group_name')})
        units = _retention_units(points, closed, groups, indexed, edge_paths)
        reasons = sorted({row['reason'] for row in units})
        context_C = None
        flow = getattr(workspace, 'flow_raw', None)
        if flow is not None and np.asarray(flow).ndim in (4, 5) and np.asarray(flow).shape[-1] == 3:
            from .algorithms.label_planes import _flow_direction
            from .algorithms.paths import inter_points
            state = (getattr(workspace, 'multilabel_groups', {}) or {}).get(groups[0], {}) if len(groups) == 1 else {}
            direction = _flow_direction(inter_points(curve, time=3), workspace.flow_raw,
                state.get('segmask_binary'), workspace.resolution, np.zeros(3))
            context_C = direction.get('flow_confidence')
        cycles.append(dict(cycle_id=cycle_id, node_indices=list(map(int, nodes)),
            edge_count=len(nodes), length_mm=float(np.linalg.norm(np.diff(curve, axis=0), axis=1).sum()),
            center_local_mm=points[nodes].mean(axis=0).tolist(), points_local_mm=curve.tolist(),
            path_indices=path_ids, label_ids=label_ids,
            label_names=[names.get(value, f'Label {value}') for value in label_ids], group_names=groups,
            cycle_C_context_only=context_C, cleanup_units=units, retention_reasons=reasons,
            cleanup_decision_scope='Per original single-label chain; whole-cycle C is context only, not a removal criterion.',
            label_scope='single_label' if len(label_ids) == 1 and np.all(sampled[nodes] > 0)
                else 'cross_label' if len(label_ids) > 1 else 'unknown'))
    cycles.sort(key=lambda row: row['cycle_id'])
    value = dict(cycle_rank=len(cycles), cycle_basis_count=len(cycles), cycles=cycles,
        detection='undirected_cycle_basis_of_current_graph', stage='after_graph_cleanup_and_terminal_pruning',
        single_label_count=sum(row['label_scope'] == 'single_label' for row in cycles),
        cross_label_count=sum(row['label_scope'] == 'cross_label' for row in cycles),
        unknown_label_count=sum(row['label_scope'] == 'unknown' for row in cycles),
        interpretation='Independent basis cycles, not a count of all possible closed walks. '
            'A flow-oriented split/rejoin DAG can still contain an undirected cycle. '
            'A warning requests review and does not label an anatomical ring as an error.')
    check = make_check('centerline.cycles', 'Centerline & Planes', 'warn' if cycles else 'pass',
        'Residual graph cycles', f'{len(cycles)} independent undirected cycle(s) remain in the current graph.',
        value=value, threshold='cycle_rank == 0; residual cycles require review',
        action='Inspect the highlighted cycle and its segmentation, paths and flow before accepting plane placement.'
            if cycles else '')
    check['details'] = [make_check('centerline.cycles:'+row['cycle_id'], 'Centerline & Planes', 'warn',
        'Graph cycle · '+(', '.join(row['label_names']) or 'unknown label'),
        f"{row['length_mm']:.2f} mm; {row['edge_count']} edges; paths {row['path_indices']}; {row['label_scope']}; "
        f"retained: {', '.join(row['retention_reasons'])}.",
        value=row, threshold='residual undirected cycle', action=check['action']) for row in cycles]
    return check
