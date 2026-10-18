"""Detect compact junction clusters and route their label-owned interfaces.

This stage follows isolated triangle contraction and precedes Flow-C filtering.
It changes only existing same-positive-label edges within compact clusters of
adjacent degree-three-or-higher nodes. It does not move points, add edges, or
infer that a source anatomical connection is false from a loop count alone.
"""

import networkx as nx
import numpy as np

from ..core.models import GraphData, validate_junction_cluster_parameters
from .graph import _edge_is_in_label, _sample_graph_labels, graph_to_networkx


def _rank(network):
    return int(network.number_of_edges() - network.number_of_nodes()
               + nx.number_connected_components(network))


def _components_preserved(original, current):
    retained = set(current)
    for component in original:
        remaining = component & retained
        if not remaining or not nx.is_connected(current.subgraph(remaining)):
            return False
    return True


def simplify_junction_clusters(graph, segmentation_labels, spacing=(1, 1, 1), *,
                               origin=(0, 0, 0), enabled=True,
                               max_extent_voxels=8.0, min_cycle_rank=2):
    """Return the graph after conservative local junction routing and its audit.

    Candidates are connected components of the induced degree >= 3 node graph,
    with >= 3 nodes, the configured internal cycle rank, and a bounded extent in
    voxel coordinates. Every cyclic same-label edge component inside a candidate
    is routed independently. All boundary ports, label-interface ports and old
    endpoints must survive. Mixed/unknown edges are protected, so mixed clusters
    can retain cycles even when their individual label-owned patches are trees.
    """
    enabled, max_extent_voxels, min_cycle_rank = validate_junction_cluster_parameters(
        enabled, max_extent_voxels, min_cycle_rank)
    points = np.asarray(graph.points, dtype=float).reshape(-1, 3)
    spacing = np.asarray(spacing, dtype=float).reshape(3)
    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("Junction-cluster spacing must contain three positive finite values")
    if not np.all(np.isfinite(points)):
        raise ValueError("Junction-cluster points must be finite")
    network = graph_to_networkx(graph)
    report = {"schema": "autoflow.junction_cluster_cleanup.v1",
        "algorithm": "junction_cluster_shortest_paths", "status": "unchanged",
        "stage": "after_triangle_contraction_before_flow_filter",
        "enabled": enabled, "max_extent_voxels": max_extent_voxels,
        "min_cycle_rank": min_cycle_rank, "minimum_node_degree": 3,
        "cycle_rank_before": _rank(network), "cycle_rank_after": _rank(network),
        "detected_cluster_count": 0, "candidate_count": 0, "simplified_cluster_count": 0,
        "removed_edge_count": 0, "removed_node_count": 0,
        "protected_edge_count": 0, "unknown_node_count": 0,
        "original_endpoints_preserved": True, "interface_ports_preserved": True,
        "component_connectivity_preserved": True, "label_component_connectivity_preserved": True,
        "boundary_ports_preserved": True, "outside_edges_preserved": True,
        "clusters": [], "removed_edges": []}
    if not enabled:
        report.update(status="disabled", reason="disabled_by_configuration")
        return graph, report
    if segmentation_labels is None:
        report.update(status="skipped", reason="label_volume_unavailable")
        return graph, report
    labels = np.asarray(segmentation_labels)
    if labels.ndim != 3:
        raise ValueError("Junction-cluster cleanup requires a 3-D label volume")
    if not len(points) or not network.number_of_edges():
        return graph, report
    node_labels = _sample_graph_labels(points, labels, spacing, origin)
    report["unknown_node_count"] = int(np.count_nonzero(node_labels <= 0))
    original = network.copy()
    original_components = list(nx.connected_components(original))
    original_endpoints = {n for n, degree in original.degree() if degree == 1}

    def key(node):
        return tuple(np.round(points[node], 12))

    def edge_key(edge):
        return tuple(sorted((key(edge[0]), key(edge[1]))))

    owned, protected, interface_ports = {}, set(), set()
    for left, right in sorted(original.edges(), key=edge_key):
        value = int(node_labels[left])
        if (value > 0 and value == node_labels[right]
                and _edge_is_in_label(points[[left, right]], value, labels, spacing, origin)):
            owned.setdefault(value, nx.Graph()).add_edge(left, right)
        else:
            protected.add(tuple(sorted((int(left), int(right)))))
            interface_ports.update((int(left), int(right)))
    label_components = {value: list(nx.connected_components(local)) for value, local in owned.items()}
    report["protected_edge_count"] = len(protected)
    high_degree = original.subgraph([n for n, degree in original.degree() if degree >= 3])
    clusters = [set(component) for component in nx.connected_components(high_degree) if len(component) >= 3]
    clusters.sort(key=lambda component: tuple(sorted(key(node) for node in component)))
    removed = set()
    boundary_ports, cluster_edges = set(), set()
    for cluster_id, component in enumerate(clusters):
        nodes = sorted(component, key=key)
        patch = original.subgraph(nodes)
        extent = np.ptp(points[nodes] / spacing, axis=0)
        boundary = {n for n in component if any(other not in component for other in original[n])}
        boundary_ports.update(boundary)
        cluster_edges.update(tuple(sorted((int(a), int(b)))) for a, b in patch.edges())
        values, counts = np.unique(node_labels[nodes], return_counts=True)
        record = {"cluster_id": cluster_id, "status": "skipped",
            "input_node_indices": list(map(int, nodes)), "points_mm": points[nodes].tolist(),
            "input_node_count": len(nodes), "input_edge_count": patch.number_of_edges(),
            "cycle_rank_before": _rank(patch), "cycle_rank_after": _rank(patch),
            "extent_voxels": extent.tolist(),
            "bounds_mm": [points[nodes].min(axis=0).tolist(), points[nodes].max(axis=0).tolist()],
            "source_labels": {str(int(v)): int(c) for v, c in zip(values, counts)},
            "boundary_port_indices": list(map(int, sorted(boundary, key=key))),
            "boundary_port_points_mm": points[sorted(boundary, key=key)].tolist(),
            "removed_edge_count": 0, "label_patches": []}
        report["clusters"].append(record)
        report["detected_cluster_count"] += 1
        if record["cycle_rank_before"] < min_cycle_rank:
            record["reason"] = "cycle_rank_below_minimum"
            continue
        if np.any(extent > max_extent_voxels + 1e-10):
            record["reason"] = "extent_exceeds_limit"
            continue
        if len({key(node) for node in component}) != len(component):
            record["reason"] = "coincident_graph_points"
            continue
        report["candidate_count"] += 1
        for value, local in sorted(owned.items()):
            local_patch = local.subgraph(component)
            parts = list(nx.connected_components(local_patch))
            parts.sort(key=lambda part: tuple(sorted(key(node) for node in part)))
            for part in parts:
                inner = local_patch.subgraph(part)
                if _rank(inner) == 0:
                    continue
                part = set(part)
                ports = {node for node in part if any(other not in part for other in network[node])}
                required = ports | (original_endpoints & part)
                detail = {"label": int(value), "status": "skipped",
                    "input_node_indices": list(map(int, sorted(part, key=key))),
                    "cycle_rank_before": _rank(inner), "cycle_rank_after": _rank(inner),
                    "required_node_indices": list(map(int, sorted(required, key=key))),
                    "interface_points_mm": points[sorted(required, key=key)].tolist(),
                    "removed_edge_count": 0}
                record["label_patches"].append(detail)
                if len(required) < 2:
                    detail["reason"] = "fewer_than_two_preserved_ports"
                    continue
                centre = points[sorted(part, key=key)].mean(axis=0)
                hub = min(part, key=lambda node: (float(np.linalg.norm(points[node] - centre)), key(node)))
                # Canonical insertion order makes equal-length shortest paths
                # deterministic under input node/edge permutations.
                canonical = nx.Graph()
                canonical.add_nodes_from(sorted(part, key=key))
                for left, right in sorted(inner.edges(), key=edge_key):
                    left, right = sorted((left, right), key=key)
                    weight = float(np.linalg.norm(points[left] - points[right]))
                    canonical.add_edge(left, right, weight=weight)
                paths = nx.single_source_dijkstra_path(canonical, hub, weight="weight")
                keep = nx.Graph()
                for node in sorted(required, key=key):
                    keep.add_edges_from(zip(paths[node], paths[node][1:]))
                # The chosen root is not itself a required anatomical port.
                # Remove an unnecessary root spur if shortest routes share a stem.
                while True:
                    leaves = [node for node, degree in keep.degree() if degree == 1 and node not in required]
                    if not leaves:
                        break
                    keep.remove_nodes_from(leaves)
                if not required.issubset(keep) or not nx.is_tree(keep):
                    raise RuntimeError("Junction routing did not retain a connected interface tree")
                retained = {tuple(sorted((int(a), int(b)))) for a, b in keep.edges()}
                cut = {tuple(sorted((int(a), int(b)))) for a, b in inner.edges()} - retained
                network.remove_edges_from(cut)
                network.remove_nodes_from([node for node in part if network.degree(node) == 0 and node not in required])
                removed.update(cut)
                detail.update(status="simplified", hub_input_node_index=int(hub),
                    hub_mm=points[hub].tolist(), removed_edge_count=len(cut), cycle_rank_after=0)
                for left, right in sorted(cut, key=edge_key):
                    report["removed_edges"].append({"cluster_id": cluster_id, "label": int(value),
                        "input_node_indices": [left, right], "points_mm": points[[left, right]].tolist()})
                record["removed_edge_count"] += len(cut)
        record["cycle_rank_after"] = _rank(network.subgraph(component & set(network)))
        if record["removed_edge_count"]:
            record["status"] = "simplified"
            report["simplified_cluster_count"] += 1
        else:
            record["reason"] = "no_eligible_label_owned_patch"
    retained_edges = {tuple(sorted((int(a), int(b)))) for a, b in network.edges()}
    report.update(original_endpoints_preserved=original_endpoints.issubset(network),
        interface_ports_preserved=interface_ports.issubset(network),
        boundary_ports_preserved=boundary_ports.issubset(network),
        outside_edges_preserved=({tuple(sorted((int(a), int(b)))) for a, b in original.edges()} - cluster_edges).issubset(retained_edges),
        component_connectivity_preserved=_components_preserved(original_components, network),
        label_component_connectivity_preserved=all(
            _components_preserved(label_components[value], local.edge_subgraph(
                [edge for edge in local.edges() if tuple(sorted(edge)) not in removed]))
            for value, local in owned.items()))
    if (not protected.issubset(retained_edges) or not all(report[name] for name in (
            "original_endpoints_preserved", "interface_ports_preserved",
            "boundary_ports_preserved", "outside_edges_preserved",
            "component_connectivity_preserved", "label_component_connectivity_preserved"))):
        raise RuntimeError("Junction cleanup changed a protected interface, endpoint or component")
    if not removed:
        return graph, report
    used = sorted(network.nodes())
    lookup = {node: index for index, node in enumerate(used)}
    edges = np.asarray([[lookup[a], lookup[b]] for a, b in sorted(retained_edges)], dtype=int).reshape(-1, 2)
    cleaned = GraphData(points=points[used].copy(), edges=edges)
    report.update(status="simplified", cycle_rank_after=_rank(network),
        removed_edge_count=len(removed), removed_node_count=len(points) - len(used),
        retained_input_node_indices=list(map(int, used)))
    return cleaned, report
