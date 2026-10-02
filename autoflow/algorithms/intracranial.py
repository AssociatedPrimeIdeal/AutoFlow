"""Topology checks for the optional intracranial Willis ring overlay.

The check is deliberately conservative.  It uses the generated centerline
graph of the combined anterior and vertebrobasilar masks, and only reports a
ring when a graph component contains a cycle and samples labels from both
arterial groups.  It does not alter the source segmentation or the ordinary
group/path metrics.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from scipy.ndimage import label as ndi_label
from scipy.spatial import cKDTree

from .graph import build_graph_from_points
from .skeleton import generate_skeleton_from_mask3d


def _node_label_values(points, labels, spacing):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    labels = np.asarray(labels)
    spacing = np.asarray(spacing, dtype=float).reshape(3)
    if len(points) == 0 or labels.ndim != 3:
        return np.zeros((len(points),), dtype=np.int16)
    voxels = np.rint(points / (spacing.reshape(1, 3) + 1e-12)).astype(int)
    for axis in range(3):
        voxels[:, axis] = np.clip(voxels[:, axis], 0, labels.shape[axis] - 1)
    values = labels[voxels[:, 0], voxels[:, 1], voxels[:, 2]].astype(np.int16, copy=False)
    # Morphological preprocessing can move a centerline point by one voxel.
    # Fill only background samples from the closest positive source label.
    missing = np.flatnonzero(values <= 0)
    if len(missing):
        source = np.argwhere(labels > 0).astype(float)
        if len(source):
            tree = cKDTree(source * spacing.reshape(1, 3))
            _, nearest = tree.query(points[missing], k=1)
            nearest_voxels = source[np.asarray(nearest, dtype=int)].astype(int)
            values[missing] = labels[tuple(nearest_voxels.T)]
    return values


def detect_willis_ring(
    label_mask_3d,
    spacing=(1.0, 1.0, 1.0),
    anterior_label_values=(),
    posterior_label_values=(),
    candidate_mask=None,
):
    """Return a conservative topology report for a Willis ring candidate.

    The returned ``points`` and ``edges`` are a derived overlay only.  They
    are never written into the normal group graph or used for plane metrics.
    """
    labels = np.asarray(label_mask_3d)
    if labels.ndim != 3:
        return {"status": "indeterminate", "reason": "labels_not_3d", "points": np.empty((0, 3)), "edges": np.empty((0, 2), dtype=int)}
    anterior = {int(value) for value in anterior_label_values if int(value) > 0}
    posterior = {int(value) for value in posterior_label_values if int(value) > 0}
    if not anterior or not posterior:
        return {"status": "indeterminate", "reason": "arterial_groups_unavailable", "points": np.empty((0, 3)), "edges": np.empty((0, 2), dtype=int)}
    mask = np.asarray(candidate_mask, dtype=bool) if candidate_mask is not None else np.isin(labels, sorted(anterior | posterior))
    if mask.shape != labels.shape or not np.any(mask):
        return {"status": "indeterminate", "reason": "arterial_mask_empty", "points": np.empty((0, 3)), "edges": np.empty((0, 2), dtype=int)}

    # Build each mask component independently.  ``build_graph_from_points``
    # links nearby skeleton samples, so combining disconnected components
    # before graph construction could turn a near miss into a false ring.
    component_labels, component_count = ndi_label(mask, structure=np.ones((3, 3, 3), dtype=bool))
    point_chunks = []
    edge_chunks = []
    point_offset = 0
    for component_id in range(1, int(component_count) + 1):
        component_mask = component_labels == int(component_id)
        component_points, _ = generate_skeleton_from_mask3d(component_mask, spacing)
        if len(component_points) == 0:
            continue
        component_graph = build_graph_from_points(component_points, spacing)
        point_chunks.append(np.asarray(component_points, dtype=float))
        if len(component_graph.edges):
            edge_chunks.append(np.asarray(component_graph.edges, dtype=int) + int(point_offset))
        point_offset += len(component_points)
    points = np.vstack(point_chunks) if point_chunks else np.empty((0, 3), dtype=float)
    if len(points) < 3:
        return {"status": "not_detected", "reason": "insufficient_skeleton", "points": points, "edges": np.empty((0, 2), dtype=int), "cycle_rank": 0}
    edges = np.vstack(edge_chunks) if edge_chunks else np.empty((0, 2), dtype=int)
    network = nx.Graph()
    network.add_nodes_from(range(len(points)))
    network.add_edges_from((int(left), int(right)) for left, right in edges)
    node_labels = _node_label_values(points, labels, spacing)
    ring_nodes = set()
    ring_edges = []
    cycle_rank = 0
    component_reports = []
    for component_id, nodes in enumerate(nx.connected_components(network)):
        node_list = sorted(int(node) for node in nodes)
        if not node_list:
            continue
        # ``network.edges(nodes)`` reports each internal edge once for an
        # undirected graph.
        edge_count = sum(1 for left, right in network.edges(node_list) if int(left) in nodes and int(right) in nodes)
        rank = max(0, int(edge_count) - len(node_list) + 1)
        labels_here = {int(value) for value in node_labels[node_list] if int(value) > 0}
        has_anterior = bool(labels_here & anterior)
        has_posterior = bool(labels_here & posterior)
        component_reports.append({"component_id": int(component_id), "nodes": len(node_list), "edges": int(edge_count), "cycle_rank": int(rank), "has_anterior": has_anterior, "has_posterior": has_posterior})
        if rank > 0 and has_anterior and has_posterior:
            cycle_rank += rank
            ring_nodes.update(node_list)
            ring_edges.extend((int(left), int(right)) for left, right in network.edges(node_list) if int(left) in nodes and int(right) in nodes)

    if ring_nodes and ring_edges:
        member_labels = sorted({int(node_labels[node]) for node in ring_nodes if int(node_labels[node]) > 0})
        return {
            "status": "detected",
            "reason": "mixed_anterior_posterior_cycle",
            "confidence": 0.9,
            "cycle_rank": int(cycle_rank),
            "member_labels": member_labels,
            "component_reports": component_reports,
            "points": np.asarray(points, dtype=float),
            "edges": np.asarray(ring_edges, dtype=int).reshape(-1, 2),
        }
    return {
        "status": "not_detected",
        "reason": "no_mixed_arterial_cycle",
        "confidence": 0.0,
        "cycle_rank": 0,
        "member_labels": [],
        "component_reports": component_reports,
        "points": np.asarray(points, dtype=float),
        "edges": np.empty((0, 2), dtype=int),
    }
