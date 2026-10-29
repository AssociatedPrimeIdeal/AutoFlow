"""Flow filtering across positive source labels, with physical topology guards."""

import hashlib
import json
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



def _chain_trial_preserves_connectivity(chain, remaining, required_nodes, interface_ports,
                                        original_components):
    """Check a degree-2 chain removal using its surviving attachment nodes.

    Previously accepted trials preserve each original component. Interior
    chain nodes have no other incident edges, so only the two attachments
    can become disconnected. Keep the full component check for an unexpected
    protected interior, and audit every complete label with it afterwards.
    """
    if required_nodes.intersection(chain[1:-1]):
        return _connectivity_preserved(original_components, remaining, required_nodes, interface_ports)
    attachments = {node for node in (chain[0], chain[-1]) if node in remaining}
    if len(attachments) == 2:
        left, right = attachments
        return nx.has_path(remaining, left, right)
    if len(attachments) == 1:
        node = next(iter(attachments))
        return remaining.degree(node) > 0 or node in interface_ports
    return False


def remove_intralabel_cycles(graph, segmentation_labels, spacing=(1, 1, 1), *,
                            support_mask=None, flow=None, temporal_mask=None, origin=(0, 0, 0),
                            flow_threshold=0.3, preserve_connectivity=True, flow_spacing=None):
    """Filter low-alignment physical chains, including positive cross-label edges.

    The historical function name is retained for callers. Unknown-label edges
    remain outside filtering. The default guard preserves physical components,
    original endpoints and positive-label presence, rather than every label
    interface. Radius routing is a separate subsequent stage.
    """
    threshold, preserve_connectivity = validate_cycle_cleanup_parameters(flow_threshold, preserve_connectivity)
    network = graph_to_networkx(graph)
    report = {"schema": "autoflow.flow_cycle_cleanup.v3", "status": "unchanged",
        "algorithm": "flow_alignment_filter", "flow_threshold": threshold,
        "preserve_connectivity": preserve_connectivity, "stage": "before_terminal_pruning",
        "policy": "all_positive_labels", "cross_label_protection": False,
        "cycle_rank_before": int(_rank(network)), "cycle_rank_after": int(_rank(network)),
        "protected_edge_count": 0, "unknown_node_count": 0, "labels": [],
        "removed_paths": [], "rejected_paths": [], "path_evaluations": [], "protected_edges": [],
        "removed_edge_count": 0, "cross_label_candidate_edge_count": 0, "cross_label_removed_edge_count": 0,
        "evaluated_path_count": 0, "candidate_count": 0, "flow_unavailable_path_count": 0,
        "original_endpoints_preserved": True, "interface_ports_preserved": True,
        "component_connectivity_preserved": True, "label_component_connectivity_preserved": True,
        "label_presence_preserved": True}
    if segmentation_labels is None:
        report.update(status="skipped", reason="label_volume_unavailable")
        return graph, report
    labels = np.asarray(segmentation_labels)
    if labels.ndim != 3:
        raise ValueError("Flow cleanup requires a 3-D label volume")
    points = np.asarray(graph.points, dtype=float)
    spacing = np.asarray(spacing, dtype=float).reshape(3)
    if not len(points) or not network.number_of_edges():
        return graph, report
    node_labels = _sample_graph_labels(points, labels, spacing, origin)
    report["unknown_node_count"] = int(np.count_nonzero(node_labels <= 0))
    local, protected, ports, label_ports = nx.Graph(), set(), set(), set()
    for left, right in sorted(tuple(sorted(edge)) for edge in network.edges()):
        a, b = int(node_labels[left]), int(node_labels[right])
        if a > 0 and b > 0:
            local.add_edge(left, right)
            if a != b:
                report["cross_label_candidate_edge_count"] += 1
                label_ports.update((left, right))
        else:
            protected.add((left, right));ports.update((left, right))
            report["protected_edges"].append({"points_mm": points[[left, right]].tolist(),
                "node_labels": [a, b], "decision": "retained", "reason": "unknown_label_edge"})
    report["protected_edge_count"] = len(protected)
    from .label_planes import _flow_direction
    from .paths import inter_points
    components = list(nx.connected_components(local))
    required = (ports & set(local)) | {n for n, degree in network.degree() if degree == 1 and n in local}
    values = {int(node_labels[n]) for n in local}
    current, candidates, removed_edges = local.copy(), [], set()
    local_report = {"label": next(iter(values)) if len(values) == 1 else 0, "source_labels": sorted(values),
        "cycle_rank_before": int(_rank(local)), "component_count": len(components),
        "removed_path_count": 0, "rejected_path_count": 0}
    for chain in _chains(local, ports):
        report["evaluated_path_count"] += 1
        curve = points[np.asarray(chain, dtype=int)]
        direction = _flow_direction(inter_points(curve, time=3), flow, temporal_mask,
            spacing if flow_spacing is None else flow_spacing, origin)
        confidence = direction["flow_confidence"]
        geometry = tuple(map(tuple, np.round(curve, 9)));geometry = min(geometry, tuple(reversed(geometry)))
        chain_labels = sorted({int(node_labels[n]) for n in chain})
        record = {"audit_id": hashlib.sha256(json.dumps((chain_labels, geometry)).encode()).hexdigest()[:16],
            "label": chain_labels[0] if len(chain_labels) == 1 else 0, "source_labels": chain_labels,
            "cross_label": len(chain_labels) > 1, "removal_unit": "flow_inconsistent_path",
            "input_node_indices": list(map(int, chain)), "points_mm": curve.tolist(),
            "length_mm": float(np.linalg.norm(np.diff(curve, axis=0), axis=1).sum()),
            "direction_status": direction["flow_direction_status"],
            "direction_confidence": float(confidence) if confidence is not None and np.isfinite(confidence) else None,
            "flow_alignment": float(direction["flow_alignment"]) if direction["flow_alignment"] is not None else None,
            "flow_threshold": threshold, "decision": "retained"}
        report["path_evaluations"].append(record)
        if not direction["flow_signal_available"] or confidence is None or not np.isfinite(confidence):
            report["flow_unavailable_path_count"] += 1;record["reason"] = "flow_signal_unavailable";continue
        if confidence >= threshold:
            record["reason"] = "flow_C_at_or_above_threshold";continue
        candidates.append((round(float(confidence), 12), geometry, chain, record))
    report["candidate_count"] = len(candidates)
    for _, _, chain, record in sorted(candidates, key=lambda item: item[:2]):
        edges = [tuple(sorted(edge)) for edge in zip(chain, chain[1:])]
        edge_data = [(a,b,dict(current[a][b])) for a,b in edges if current.has_edge(a,b)]
        current.remove_edges_from(edges)
        retain = required if preserve_connectivity else ports
        isolated = [n for n in set(chain) if n in current and current.degree(n)==0 and n not in retain]
        node_data = [(n,dict(current.nodes[n])) for n in isolated];current.remove_nodes_from(isolated)
        connected = _chain_trial_preserves_connectivity(chain,current,required,ports,components)
        labels_retained = {int(node_labels[n]) for n in current}
        reason = ("preserve_component_or_endpoint" if not connected else
                  "preserve_positive_label_coverage" if values-labels_retained else "")
        if preserve_connectivity and reason:
            current.add_nodes_from(node_data);current.add_edges_from(edge_data)
            record["reason"]=reason;report["rejected_paths"].append(dict(record));local_report["rejected_path_count"]+=1
            continue
        record.update(decision="removed",reason="flow_C_below_threshold")
        removed_edges.update(edges);report["removed_paths"].append(record);local_report["removed_path_count"]+=1
    local_report.update(cycle_rank_after=int(_rank(current)),
        component_connectivity_preserved=_connectivity_preserved(components,current,required,ports))
    report["labels"].append(local_report)
    if not removed_edges:
        if report["evaluated_path_count"] and report["flow_unavailable_path_count"]==report["evaluated_path_count"]:
            report.update(status="skipped",reason="flow_signal_unavailable")
        return graph,report
    retained = sorted(tuple(sorted(e)) for e in network.edges() if tuple(sorted(e)) not in removed_edges)
    used = sorted({n for e in retained for n in e}|{n for n,d in network.degree() if d==0})
    lookup={n:i for i,n in enumerate(used)}
    cleaned=GraphData(points=points[used].copy(),edges=np.asarray([[lookup[a],lookup[b]] for a,b in retained],int).reshape(-1,2))
    terminals={n for n,d in network.degree() if d==1}
    report.update(original_endpoints_preserved=terminals.issubset(used),interface_ports_preserved=label_ports.issubset(used),
        component_connectivity_preserved=bool(local_report["component_connectivity_preserved"]),
        label_presence_preserved=values.issubset({int(node_labels[n]) for n in used}))
    report["label_component_connectivity_preserved"] = all(
        bool(set(part) & set(current)) and nx.is_connected(current.subgraph(set(part) & set(current)))
        for v in values for part in nx.connected_components(local.subgraph([n for n in local if node_labels[n] == v])))
    if not protected.issubset(retained) or not ports.issubset(used):
        raise RuntimeError("Flow filtering changed an unknown-label edge")
    if preserve_connectivity and (not terminals.issubset(used) or not report["component_connectivity_preserved"] or not report["label_presence_preserved"]):
        raise RuntimeError("Flow filtering violated a component, endpoint or label-coverage guard")
    report.update(status="applied",removed_edge_count=len(removed_edges),
        cross_label_removed_edge_count=int(sum(node_labels[a]!=node_labels[b] for a,b in removed_edges)),
        cycle_rank_after=int(_rank(graph_to_networkx(cleaned))),input_point_count=len(points),output_point_count=len(used))
    return cleaned,report
