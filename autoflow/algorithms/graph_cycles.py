"""Filter flow-inconsistent paths wholly inside one positive source label.

This runs before terminal pruning. The default connectivity guard retains
every original within-label component, its endpoints and label interfaces.
Cross-label and unknown-node edges are never candidates for removal. A low
alignment score alone is not a guarantee that a path is an anatomical error.
"""

import networkx as nx
import numpy as np

from ..core.models import GraphData, validate_cycle_cleanup_parameters
from .graph import _edge_is_in_label, _sample_graph_labels, graph_to_networkx


def _rank(network):
    return network.number_of_edges() - network.number_of_nodes() + nx.number_connected_components(network)


def _chains(network, ports):
    """Compress degree-2 runs, splitting at every protected interface port."""
    stops = set(ports) | {node for node, degree in network.degree() if degree != 2}
    seen, result = set(), []

    def walk(start, following):
        chain = [start, following]
        seen.add(tuple(sorted((start, following))))
        previous, current = start, following
        while current not in stops:
            others = sorted(node for node in network[current] if node != previous)
            if not others:
                break
            nxt = others[0]
            edge = tuple(sorted((current, nxt)))
            if edge in seen:
                break
            seen.add(edge)
            chain.append(nxt)
            previous, current = current, nxt
        return chain

    for start in sorted(stops & set(network)):
        for following in sorted(network[start]):
            if tuple(sorted((start, following))) not in seen:
                result.append(walk(start, following))
    for left, right in sorted(tuple(sorted(edge)) for edge in network.edges()):
        if (left, right) not in seen:
            result.append(walk(left, right))
    return result


def _connectivity_preserved(original_components, remaining, required_nodes, interface_ports):
    kept_nodes = set(remaining)
    if not required_nodes.issubset(kept_nodes):
        return False
    for component in original_components:
        kept = component & kept_nodes
        if not kept or not nx.is_connected(remaining.subgraph(kept)):
            return False
        # A singleton port can still connect to protected cross-label edges.
        # A lone terminal endpoint without any retained edge is not a vessel
        # component and would disappear during graph compaction.
        if len(kept) == 1 and not kept.issubset(interface_ports):
            return False
    return True


def remove_intralabel_cycles(graph, segmentation_labels, spacing=(1, 1, 1), *,
                            support_mask=None, flow=None, temporal_mask=None, origin=(0, 0, 0),
                            flow_threshold=0.3, preserve_connectivity=True):
    """Return the graph after guarded Flow-C filtering and its audit report.

    C is the absolute signed velocity/tangent alignment integrated over physical
    arc length and all valid cardiac phases, shared with path direction grading.
    Candidates with C strictly below the threshold are considered from lowest
    C upwards; canonical physical geometry resolves ties. Radius is not used.
    Missing/zero valid flow is skipped, and reverse-aligned paths are retained.

    Turning connectivity protection off permits a low-C path to split or remove
    a label component, including terminal paths. Cross-label/unknown edges stay
    outside the removal scope regardless of that setting. ``support_mask`` is
    accepted for compatibility; this method does not use lumen radius.
    """
    threshold, preserve_connectivity = validate_cycle_cleanup_parameters(flow_threshold, preserve_connectivity)
    network = graph_to_networkx(graph)
    report = {"schema": "autoflow.intralabel_cycle_cleanup.v2", "status": "unchanged",
        "algorithm": "flow_alignment_filter", "flow_threshold": threshold,
        "preserve_connectivity": preserve_connectivity,
        "stage": "before_terminal_pruning", "policy": "positive_single_label_only",
        "cycle_rank_before": int(_rank(network)), "cycle_rank_after": int(_rank(network)),
        "protected_edge_count": 0, "unknown_node_count": 0, "labels": [],
        "removed_paths": [], "rejected_paths": [], "removed_edge_count": 0,
        "evaluated_path_count": 0, "candidate_count": 0, "flow_unavailable_path_count": 0,
        "original_endpoints_preserved": True, "interface_ports_preserved": True,
        "label_component_connectivity_preserved": True}
    if segmentation_labels is None:
        report.update(status="skipped", reason="label_volume_unavailable")
        return graph, report
    labels = np.asarray(segmentation_labels)
    if labels.ndim != 3:
        raise ValueError("Single-label cycle cleanup requires a 3-D label volume")
    points = np.asarray(graph.points, dtype=float)
    spacing = np.asarray(spacing, dtype=float).reshape(3)
    if not len(points) or not network.number_of_edges():
        return graph, report
    node_labels = _sample_graph_labels(points, labels, spacing, origin)
    report["unknown_node_count"] = int(np.count_nonzero(node_labels <= 0))
    owned, protected, ports = {}, set(), set()
    for left, right in sorted(tuple(sorted(edge)) for edge in network.edges()):
        value = int(node_labels[left])
        if (value > 0 and value == node_labels[right] and
                _edge_is_in_label(points[[left, right]], value, labels, spacing, origin)):
            owned.setdefault(value, nx.Graph()).add_edge(left, right)
        else:
            protected.add((left, right))
            ports.update((left, right))
    report["protected_edge_count"] = len(protected)
    removed_edges = set()
    from .label_planes import _flow_direction
    from .paths import inter_points

    for value, local in sorted(owned.items()):
        components = list(nx.connected_components(local))
        required = (ports & set(local)) | {node for node, degree in local.degree() if degree == 1}
        current = local.copy()
        label_report = {"label": value, "cycle_rank_before": int(_rank(local)),
            "component_count": len(components), "removed_path_count": 0, "rejected_path_count": 0}
        candidates = []
        for chain in _chains(local, ports):
            report["evaluated_path_count"] += 1
            curve = points[np.asarray(chain, dtype=int)]
            direction = _flow_direction(inter_points(curve, time=3), flow, temporal_mask, spacing, origin)
            confidence = direction["flow_confidence"]
            if not direction["flow_signal_available"] or confidence is None or not np.isfinite(confidence):
                report["flow_unavailable_path_count"] += 1
                continue
            if confidence >= threshold:
                continue
            geometry = tuple(map(tuple, np.round(curve, 9)))
            geometry = min(geometry, tuple(reversed(geometry)))
            candidates.append((round(float(confidence), 12), geometry, chain, direction))
        report["candidate_count"] += len(candidates)
        for _, _, chain, direction in sorted(candidates, key=lambda item: item[:2]):
            edges = [tuple(sorted(edge)) for edge in zip(chain, chain[1:])]
            trial = current.copy()
            trial.remove_edges_from(edges)
            retain = required if preserve_connectivity else ports
            trial.remove_nodes_from([node for node, degree in trial.degree() if degree == 0 and node not in retain])
            curve = points[np.asarray(chain, dtype=int)]
            record = {"label": value, "removal_unit": "flow_inconsistent_path",
                "input_node_indices": list(map(int, chain)), "points_mm": curve.tolist(),
                "length_mm": float(np.linalg.norm(np.diff(curve, axis=0), axis=1).sum()),
                "direction_status": direction["flow_direction_status"],
                "direction_confidence": float(direction["flow_confidence"]),
                "flow_alignment": float(direction["flow_alignment"])}
            if preserve_connectivity and not _connectivity_preserved(components, trial, required, ports):
                report["rejected_paths"].append(dict(record, reason="preserve_label_component_or_interface"))
                label_report["rejected_path_count"] += 1
                continue
            current = trial
            removed_edges.update(edges)
            report["removed_paths"].append(record)
            label_report["removed_path_count"] += 1
        label_report["cycle_rank_after"] = int(_rank(current))
        label_report["component_connectivity_preserved"] = _connectivity_preserved(components, current, required, ports)
        report["label_component_connectivity_preserved"] &= label_report["component_connectivity_preserved"]
        report["labels"].append(label_report)
    if not removed_edges:
        if report["evaluated_path_count"] and report["flow_unavailable_path_count"] == report["evaluated_path_count"]:
            report.update(status="skipped", reason="flow_signal_unavailable")
        return graph, report
    retained = sorted(tuple(sorted(edge)) for edge in network.edges() if tuple(sorted(edge)) not in removed_edges)
    used = sorted({node for edge in retained for node in edge} | {node for node, degree in network.degree() if degree == 0})
    node_map = {node: index for index, node in enumerate(used)}
    cleaned = GraphData(points=points[used].copy(),
        edges=np.asarray([[node_map[left], node_map[right]] for left, right in retained], dtype=int).reshape(-1, 2))
    endpoints = {node for node, degree in network.degree() if degree == 1}
    report.update(original_endpoints_preserved=endpoints.issubset(used), interface_ports_preserved=ports.issubset(used))
    if not protected.issubset(retained) or not ports.issubset(used):
        raise RuntimeError("Flow filtering changed a protected label interface")
    if preserve_connectivity and (not endpoints.issubset(used) or
            not report["label_component_connectivity_preserved"] or
            nx.number_connected_components(network) != nx.number_connected_components(graph_to_networkx(cleaned))):
        raise RuntimeError("Flow filtering changed a protected endpoint or component")
    report.update(status="applied", removed_edge_count=len(removed_edges),
        cycle_rank_after=int(_rank(graph_to_networkx(cleaned))),
        input_point_count=len(points), output_point_count=len(used))
    return cleaned, report
