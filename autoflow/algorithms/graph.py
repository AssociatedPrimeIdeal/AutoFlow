import networkx as nx
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

from ..models import GraphData


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
