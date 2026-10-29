"""Filter by flow first, then re-segment the surviving graph for node pruning."""

import hashlib
import numpy as np

from ..core.models import GraphData
from ..task_control import check_cancelled
from .graph import graph_to_networkx
from .graph_cycles import _chains
from .flow_alignment import branch_flow_evidence


def _filter_stage(graph, name, rule, judge, apply):
    net = graph_to_networkx(graph)
    points = np.asarray(graph.points, dtype=float)
    records = []
    incident = {node: [] for node, degree in net.degree() if degree >= 3}
    rejected = set()
    for chain in _chains(net, set()):
        check_cancelled()
        if tuple(points[chain[0]]) > tuple(points[chain[-1]]):
            chain = chain[::-1]
        curve = points[np.asarray(chain)]
        terminal = net.degree(chain[0]) == 1 or net.degree(chain[-1]) == 1
        reject, reason, fields = judge(curve, chain, terminal)
        assert name != 'nodes' or not reject or terminal
        geometry = np.round(curve, 8).tolist()
        identity = hashlib.sha256(str(geometry).encode()).hexdigest()[:12]
        records.append(dict(branch_id=identity, nodes=list(map(int, chain)), terminal=bool(terminal),
            points_mm=curve.tolist(), **fields,
            decision='removed' if reject and apply else 'candidate' if reject else 'retained',
            reason=reason))
        for node in (chain[0], chain[-1]):
            if node in incident:
                incident[node].append(dict(branch_id=identity))
        if reject:
            rejected.update(tuple(sorted(edge)) for edge in zip(chain, chain[1:]))
    result = graph
    if rejected:
        edges = [(a, b) for a, b in net.edges() if tuple(sorted((a, b))) not in rejected]
        used = sorted({node for edge in edges for node in edge})
        lookup = {node: i for i, node in enumerate(used)}
        result = GraphData(points=points[used].copy(),
            edges=np.asarray([[lookup[a], lookup[b]] for a, b in edges], dtype=int).reshape(-1, 2))
    report = dict(name=name, rule=rule, terminal_only=name == 'nodes', branches=records,
        junctions=[dict(point_mm=points[node].tolist(), degree=net.degree(node), arms=arms)
                   for node, arms in sorted(incident.items(), key=lambda item: tuple(points[item[0]]))],
        candidate_count=sum(b['decision'] in {'candidate', 'removed'} for b in records),
        would_remove_edge_count=len(rejected), removed_edge_count=len(rejected) if apply else 0)
    return result, report


def filter_graph_branches(graph, *, flow=None, flow_spacing=(1, 1, 1),
                          temporal_mask=None, mask_spacing=None, origin=(0, 0, 0),
                          apply=False, threshold=.3, min_node_count=5):
    """Filter all arms by C first, then prune only short terminal arms.

    The C pass evaluates every arm, irrespective of terminal status, and
    rejects C_axis < threshold OR C_phase < threshold. Recompute chains and
    endpoint degrees on the surviving graph. Only arms with at least one
    degree-one endpoint qualify for the node pass; internal connections and
    closed loops are protected there. Count unique nodes including endpoints.
    Each pass runs once; the node pass never samples flow. Zero disables it.

    Audit mode simulates both passes for its report, returns the original
    graph and never mutates it. The report contains both stage decisions and
    a compact intermediate_graph for review. No main route is selected.
    """
    if not isinstance(apply, (bool, np.bool_)):
        raise ValueError('apply must be boolean')
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('threshold must be finite and between zero and one')
    if (isinstance(min_node_count, (bool, np.bool_)) or
            not isinstance(min_node_count, (int, np.integer)) or min_node_count < 0):
        raise ValueError('min_node_count must be a nonnegative integer')

    def judge_flow(curve, chain, terminal):
        evidence = branch_flow_evidence(curve, flow, spacing=flow_spacing,
            origin=origin, mask=temporal_mask, mask_spacing=mask_spacing)
        axis, phase = evidence['axis_alignment'], evidence['phase_alignment']
        reject = (axis is not None and axis < threshold) or (phase is not None and phase < threshold)
        reason = 'below_threshold' if reject else 'scores_unavailable' if axis is None or phase is None else 'above_threshold'
        return reject, reason, dict(flow=evidence)

    def judge_nodes(curve, chain, terminal):
        count = len(set(chain))
        reject = terminal and count < min_node_count
        reason = 'connection_preserved' if not terminal else 'short_arm' if reject else 'enough_nodes'
        return reject, reason, dict(node_count=count)

    after_flow, flow_report = _filter_stage(graph, 'flow',
        f'C_axis < {threshold} OR C_phase < {threshold}', judge_flow, apply)
    final, node_report = _filter_stage(after_flow, 'nodes',
        f'terminal AND node_count < {min_node_count}', judge_nodes, apply)
    report = dict(schema='autoflow.branch_sequential_filter.v3', experimental=True, terminal_only_stages=['nodes'],
        applied=bool(apply), threshold=float(threshold), min_node_count=int(min_node_count),
        rule='C filter on all arms -> re-segment and recheck endpoints -> terminal node-count filter',
        stage_order=['flow', 'nodes'], computed_scores=['axis_alignment', 'phase_alignment'],
        stages=dict(flow=flow_report, nodes=node_report),
        intermediate_graph=dict(points=np.asarray(after_flow.points).tolist(),
                                edges=np.asarray(after_flow.edges, dtype=int).reshape(-1, 2).tolist()),
        candidate_count=flow_report['candidate_count']+node_report['candidate_count'],
        removed_edge_count=len(graph.edges)-len(final.edges) if apply else 0,
        no_main_route_search=True)
    return (final if apply else graph), report
