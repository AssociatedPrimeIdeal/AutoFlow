import networkx as nx
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

from ..core.models import GraphData, validate_graph_connectivity


def build_graph_from_points(points, spacing, connectivity=26):
    """Connect voxel-center skeleton points by face (6) or face/edge/corner (26)."""
    connectivity = validate_graph_connectivity(connectivity)
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    spacing = np.asarray(spacing, dtype=float).reshape(3)
    if len(points) == 0:
        return GraphData()
    scaled = points / spacing
    tree = cKDTree(scaled)
    pairs = tree.query_pairs(r=(1. if connectivity == 6 else np.sqrt(3)) * (1.0 + 1e-11), output_type="ndarray")
    if connectivity == 6 and len(pairs):
        delta = np.abs(scaled[pairs[:, 0]] - scaled[pairs[:, 1]])
        pairs = pairs[(np.count_nonzero(delta > 1e-7, axis=1) == 1)
                      & np.isclose(np.max(delta, axis=1), 1., rtol=0, atol=1e-7)]
    edges = np.asarray(pairs, dtype=int).reshape(-1, 2) if len(pairs) > 0 else np.empty((0, 2), dtype=int)
    graph = GraphData(points=points, edges=edges)
    graph = remove_triangle_cycles(graph)
    return graph


def remove_triangle_cycles(graph):
    """Contract isolated triangle junctions to their physical-coordinate centroid.

    Keep external edges by remapping their endpoints, without another spatial
    neighbor search. Overlapping triangles, closed triangular rings and patches
    whose contraction would merge distinct external edges are left unchanged.
    The historical function name is retained for existing callers.
    """
    if len(graph.edges) == 0:
        return graph
    G = graph_to_networkx(graph)
    triangles = []
    membership = np.zeros(len(graph.points), dtype=int)
    for left in G:
        for middle in G[left]:
            if middle <= left:
                continue
            for right in G[left].keys() & G[middle].keys():
                if right > middle:
                    triangle = (left, middle, right)
                    triangles.append(triangle)
                    membership[list(triangle)] += 1

    junctions = []
    for triangle in sorted(triangles):
        if np.any(membership[list(triangle)] != 1):
            continue
        external = set().union(*(G[node] for node in triangle)) - set(triangle)
        if len(external) >= 3:
            junctions.append(triangle)
    if not junctions:
        return graph

    representative = np.arange(len(graph.points))
    for triangle in junctions:
        representative[list(triangle)] = triangle[0]
    edges = np.asarray(list(G.edges()), dtype=int).reshape(-1, 2)
    mapped = np.sort(representative[edges], axis=1)
    external_edges = mapped[mapped[:, 0] != mapped[:, 1]]
    unique_edges, counts = np.unique(external_edges, axis=0, return_counts=True)
    # GraphData has no parallel-edge identity. Keep both patches intact if
    # contraction would collapse separate external connections into one edge.
    unsafe = set(unique_edges[counts > 1].ravel())
    junctions = [triangle for triangle in junctions if triangle[0] not in unsafe]
    if not junctions:
        return graph

    points = np.asarray(graph.points, dtype=float).copy()
    keep = np.ones(len(points), dtype=bool)
    representative = np.arange(len(points))
    for triangle in junctions:
        points[triangle[0]] = np.mean(points[list(triangle)], axis=0)
        representative[list(triangle)] = triangle[0]
        keep[list(triangle[1:])] = False
    compact_index = np.cumsum(keep) - 1
    new_edges = np.sort(compact_index[representative[edges]], axis=1)
    # Only the triangle's internal edges become self-edges under this mapping.
    new_edges = new_edges[new_edges[:, 0] != new_edges[:, 1]]
    return GraphData(points=points[keep], edges=np.unique(new_edges, axis=0))


def remove_short_terminal_branches(graph, min_edge_points=5, *, min_edge_count=None):
    """Remove short terminal branches from a skeleton graph.

    The graph built from voxel skeleton points contains one graph edge per
    pair of neighbouring skeleton points.  This helper therefore measures a
    terminal branch by the number of consecutive graph edges from a degree-1
    endpoint to the next junction (or to the other endpoint of a simple
    component).  Junction-to-junction links are preserved.  ``min_edge_count``
    is accepted as a descriptive alias for callers that use edge terminology.

    After removal, retained nodes are compacted and reindexed.  The original
    skeleton mask and points remain available separately for visualization.
    """
    if min_edge_count is not None:
        min_edge_points = min_edge_count
    try:
        threshold = int(min_edge_points)
    except (TypeError, ValueError):
        threshold = 5
    threshold = max(0, threshold)
    if threshold <= 1 or len(getattr(graph, "edges", [])) == 0:
        return graph

    G = graph_to_networkx(graph)
    if G.number_of_edges() == 0:
        return graph

    degree = dict(G.degree())
    endpoints = sorted(int(node) for node, value in degree.items() if int(value) == 1)
    visited_edges = set()
    edges_to_remove = set()

    for start in endpoints:
        for neighbor in sorted(G.neighbors(start)):
            first_edge = tuple(sorted((int(start), int(neighbor))))
            if first_edge in visited_edges:
                continue
            path_edges = []
            previous = int(start)
            current = int(neighbor)
            visited_edges.add(first_edge)
            path_edges.append(first_edge)

            # Walk until the next endpoint, a junction, or a malformed graph
            # boundary.  Degree-2 nodes are the interior of one logical arm.
            while int(degree.get(current, 0)) == 2:
                neighbors = sorted(int(value) for value in G.neighbors(current))
                next_nodes = [value for value in neighbors if value != previous]
                if not next_nodes:
                    break
                nxt = int(next_nodes[0])
                edge = tuple(sorted((current, nxt)))
                if edge in visited_edges:
                    break
                path_edges.append(edge)
                visited_edges.add(edge)
                previous, current = current, nxt

            if len(path_edges) < threshold:
                edges_to_remove.update(path_edges)

    if not edges_to_remove:
        return graph
    retained_edges = [
        (int(edge[0]), int(edge[1]))
        for edge in np.asarray(graph.edges, dtype=int).reshape(-1, 2)
        if tuple(sorted((int(edge[0]), int(edge[1])))) not in edges_to_remove
    ]
    if not retained_edges:
        return GraphData()
    used_nodes = sorted({node for edge in retained_edges for node in edge})
    node_map = {old: new for new, old in enumerate(used_nodes)}
    new_edges = np.asarray(
        [[node_map[int(left)], node_map[int(right)]] for left, right in retained_edges],
        dtype=int,
    ).reshape(-1, 2)
    new_points = np.asarray(graph.points, dtype=float)[np.asarray(used_nodes, dtype=int)]
    return GraphData(points=new_points, edges=new_edges)


def graph_to_networkx(graph):
    G = nx.Graph()
    for i in range(len(graph.points)):
        G.add_node(i)
    for e in np.asarray(graph.edges, dtype=int):
        if len(e) == 2:
            G.add_edge(int(e[0]), int(e[1]))
    return G


def graph_to_polydata(points, edges):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    poly = pv.PolyData(points)
    if len(edges) == 0:
        return poly
    cells = np.empty((len(edges), 3), dtype=np.int64)
    cells[:, 0] = 2
    cells[:, 1] = edges[:, 0]
    cells[:, 2] = edges[:, 1]
    poly.lines = cells.ravel()
    return poly
