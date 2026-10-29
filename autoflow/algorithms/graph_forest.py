"""Endpoint-constrained lumen-radius routing on existing graph edges."""

import networkx as nx
import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates

from ..core.models import GraphData, validate_radius_tree_policy
from ..task_control import check_cancelled
from .graph import _sample_graph_labels, graph_to_networkx


def route_endpoint_radius_forest(graph, segmentation_labels, spacing=(1, 1, 1), *,
                                 policy="auto", label_map=None):
    """Route terminal anchors without moving points or adding graph edges.

    All labels use the same eligibility constraints, without anatomical exemptions.
    Cyclic components use the union of shortest paths from a largest-
    radius terminal to every original terminal. The root is numerical, not a
    physiological inlet. Components without two terminals, or whose positive
    label coverage would be lost, remain unchanged with an explicit reason.
    """
    policy = validate_radius_tree_policy(policy)
    net = graph_to_networkx(graph)
    points = np.asarray(graph.points, dtype=float).reshape(-1, 3)
    spacing = np.asarray(spacing, dtype=float).reshape(3)
    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("Radius-tree spacing must be positive and finite")
    rank = lambda g: int(g.number_of_edges() - g.number_of_nodes() + nx.number_connected_components(g))
    report = {"schema": "autoflow.endpoint_radius_forest.v1", "algorithm": "endpoint_radius_shortest_paths",
              "policy": policy, "status": "unchanged", "stage": "after_flow_before_terminal_pruning",
              "cycle_rank_before": rank(net), "cycle_rank_after": rank(net), "components": [],
              "removed_edge_count": 0, "cross_label_removed_edge_count": 0,
              "endpoint_coordinates_preserved": True, "component_connectivity_preserved": True,
              "label_presence_preserved": True, "no_new_edges": True, "extent_limit_applied": False}
    if policy == "preserve":
        report.update(status="preserved", reason="explicit_preserve_policy")
        return graph, report
    if segmentation_labels is None:
        report.update(status="skipped", reason="label_volume_unavailable")
        return graph, report
    labels = np.asarray(segmentation_labels)
    if labels.ndim != 3:
        raise ValueError("Radius-tree routing requires a 3-D label volume")
    node_labels = _sample_graph_labels(points, labels, spacing)
    if not len(points) or report["cycle_rank_before"] == 0:
        return graph, report
    kept_nodes, kept_edges = set(), set()
    key = lambda n: tuple(np.round(points[n], 9))
    floor = .5 * float(spacing.min())
    for component in sorted(nx.connected_components(net), key=lambda nodes: min(key(n) for n in nodes)):
        check_cancelled()
        local = net.subgraph(component)
        terminals = {n for n in component if local.degree(n) == 1}
        record = {"input_nodes": len(component), "terminal_count": len(terminals),
                  "cycle_rank_before": rank(local), "status": "retained"}
        selected = local
        if rank(local) == 0:
            record["reason"] = "already_acyclic"
        elif len(terminals) < 2:
            record["reason"] = "fewer_than_two_terminal_anchors"
        elif len({key(n) for n in component}) != len(component):
            record["reason"] = "coincident_graph_points"
        else:
            nodes = sorted(component, key=key)
            ijk = points[nodes] / spacing
            lo = np.maximum(np.floor(ijk.min(0)).astype(int) - 2, 0)
            hi = np.minimum(np.ceil(ijk.max(0)).astype(int) + 3, labels.shape)
            values = set(int(node_labels[n]) for n in component if node_labels[n] > 0)
            crop = np.isin(labels[tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))], sorted(values))
            if not crop.size or not np.any(crop):
                record["reason"] = "lumen_support_unavailable"
            else:
                radii = distance_transform_edt(np.pad(crop, 1), sampling=spacing)
                sampled = map_coordinates(radii, (ijk - lo + 1).T, order=1, mode="constant", cval=0)
                radius = dict(zip(nodes, sampled))
                root = min(terminals, key=lambda n: (-round(float(radius[n]), 9), key(n)))
                canonical = nx.Graph()
                canonical.add_nodes_from(nodes)
                for a, b in sorted(local.edges(), key=lambda e: tuple(sorted((key(e[0]), key(e[1]))))):
                    cost = max(float(np.linalg.norm(points[a] - points[b])), 1e-9) / (floor + .5 * (radius[a] + radius[b])) ** 2
                    canonical.add_edge(a, b, weight=cost)
                routes = nx.single_source_dijkstra_path(canonical, root, weight="weight")
                candidate = nx.Graph()
                for terminal in sorted(terminals, key=key):
                    candidate.add_edges_from(zip(routes[terminal], routes[terminal][1:]))
                retained = {int(node_labels[n]) for n in candidate if node_labels[n] > 0}
                if values - retained:
                    record.update(reason="positive_label_coverage_guard", missing_labels=sorted(values - retained))
                else:
                    assert nx.is_tree(candidate) and terminals.issubset(candidate)
                    selected = candidate
                    record.update(status="routed", root_mm=points[root].tolist(),
                                  cost="length_mm / (half_spacing_mm + mean_radius_mm)^2")
        kept_nodes.update(selected.nodes())
        kept_edges.update(tuple(sorted(e)) for e in selected.edges())
        record.update(output_nodes=selected.number_of_nodes(), cycle_rank_after=rank(selected))
        report["components"].append(record)
    original_edges = {tuple(sorted(e)) for e in net.edges()}
    removed = original_edges - kept_edges
    terminals = {n for n, degree in net.degree() if degree == 1}
    assert terminals <= kept_nodes and kept_edges <= original_edges
    if not removed:
        return graph, report
    used = sorted(kept_nodes)
    lookup = {n: i for i, n in enumerate(used)}
    result = GraphData(points=points[used].copy(),
        edges=np.asarray([[lookup[a], lookup[b]] for a, b in sorted(kept_edges)], dtype=int).reshape(-1, 2))
    assert nx.number_connected_components(net) == nx.number_connected_components(graph_to_networkx(result))
    report.update(status="applied", removed_edge_count=len(removed), cycle_rank_after=rank(graph_to_networkx(result)),
                  cross_label_removed_edge_count=int(sum(node_labels[a] != node_labels[b] for a, b in removed)))
    return result, report
