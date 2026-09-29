import networkx as nx
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

from ..core.models import GraphData


def build_graph_from_points(points, spacing):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    spacing = np.asarray(spacing, dtype=float).reshape(3)
    if len(points) == 0:
        return GraphData()
    scaled = points / spacing
    tree = cKDTree(scaled)
    pairs = tree.query_pairs(r=np.sqrt(3) * (1.0 + 1e-11), output_type="ndarray")
    edges = np.asarray(pairs, dtype=int).reshape(-1, 2) if len(pairs) > 0 else np.empty((0, 2), dtype=int)
    graph = GraphData(points=points, edges=edges)
    graph = remove_triangle_cycles(graph)
    return graph


def remove_triangle_cycles(graph):
    if len(graph.edges) == 0:
        return graph
    G = nx.Graph()
    for i in range(len(graph.points)):
        G.add_node(i)
    for e in graph.edges:
        G.add_edge(int(e[0]), int(e[1]))
    edges_to_remove = set()
    for n in list(G.nodes()):
        neighbors = list(G.neighbors(n))
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                a, b = neighbors[i], neighbors[j]
                if G.has_edge(a, b):
                    tri = tuple(sorted([n, a, b]))
                    e_candidates = [
                        (tri[0], tri[1]),
                        (tri[0], tri[2]),
                        (tri[1], tri[2]),
                    ]
                    best_edge = None
                    best_deg_sum = -1
                    for ea, eb in e_candidates:
                        if (ea, eb) not in edges_to_remove and (eb, ea) not in edges_to_remove:
                            ds = G.degree(ea) + G.degree(eb)
                            if ds > best_deg_sum:
                                best_deg_sum = ds
                                best_edge = (ea, eb)
                    if best_edge is not None:
                        edges_to_remove.add(best_edge)
    for ea, eb in edges_to_remove:
        if G.has_edge(ea, eb):
            if nx.is_connected(G):
                G_test = G.copy()
                G_test.remove_edge(ea, eb)
                if nx.is_connected(G_test):
                    G.remove_edge(ea, eb)
            else:
                comp = nx.node_connected_component(G, ea)
                if eb in comp:
                    G_test = G.copy()
                    G_test.remove_edge(ea, eb)
                    if eb in nx.node_connected_component(G_test, ea):
                        G.remove_edge(ea, eb)
    new_edges = np.array(list(G.edges()), dtype=int).reshape(-1, 2) if G.number_of_edges() > 0 else np.empty((0, 2), dtype=int)
    return GraphData(points=graph.points.copy(), edges=new_edges)


def remove_short_terminal_branches(graph, min_edge_points=3, *, min_edge_count=None):
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
        threshold = 3
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
