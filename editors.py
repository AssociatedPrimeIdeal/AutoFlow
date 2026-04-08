import numpy as np
from models import PlaneData, GraphData


class SkeletonEditor:
    def __init__(self, workspace):
        self.workspace = workspace

    def remove_points_by_index(self, indices):
        if self.workspace.skeleton_points is None:
            return
        pts = np.asarray(self.workspace.skeleton_points)
        mask = np.ones(len(pts), dtype=bool)
        mask[np.asarray(indices, dtype=int)] = False
        self.workspace.skeleton_points = pts[mask]

    def append_points(self, points):
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        if self.workspace.skeleton_points is None or len(self.workspace.skeleton_points) == 0:
            self.workspace.skeleton_points = pts
        else:
            self.workspace.skeleton_points = np.vstack([self.workspace.skeleton_points, pts])

    def replace_points(self, points):
        self.workspace.skeleton_points = np.asarray(points, dtype=float).reshape(-1, 3)


class GraphEditor:
    def __init__(self, workspace):
        self.workspace = workspace

    def remove_edges_by_index(self, indices):
        edges = np.asarray(self.workspace.graph.edges, dtype=int)
        if len(edges) == 0:
            return
        mask = np.ones(len(edges), dtype=bool)
        mask[np.asarray(indices, dtype=int)] = False
        self.workspace.graph = GraphData(
            points=self.workspace.graph.points.copy(), edges=edges[mask])

    def append_edges(self, edges):
        e = np.asarray(edges, dtype=int).reshape(-1, 2)
        if len(self.workspace.graph.edges) == 0:
            new_edges = e
        else:
            new_edges = np.vstack([self.workspace.graph.edges, e])
        self.workspace.graph = GraphData(
            points=self.workspace.graph.points.copy(), edges=new_edges)

    def remove_nodes_by_index(self, indices):
        points = np.asarray(self.workspace.graph.points, dtype=float)
        edges = np.asarray(self.workspace.graph.edges, dtype=int)
        rm = set(int(i) for i in indices)
        keep_idx = [i for i in range(len(points)) if i not in rm]
        remap = {old: new for new, old in enumerate(keep_idx)}
        new_points = points[keep_idx]
        new_edges = []
        for a, b in edges:
            if int(a) in remap and int(b) in remap:
                new_edges.append([remap[int(a)], remap[int(b)]])
        self.workspace.graph = GraphData(
            points=new_points,
            edges=np.asarray(new_edges, dtype=int).reshape(-1, 2) if new_edges else np.empty((0, 2), dtype=int),
        )


class PlaneEditor:
    def __init__(self, workspace):
        self.workspace = workspace

    def add_plane(self, center, normal, label=1, path_index=0, distance=0.0):
        n = np.asarray(normal, dtype=float).reshape(3)
        n = n / (np.linalg.norm(n) + 1e-12)
        self.workspace.planes.append(PlaneData(
            center=np.asarray(center, dtype=float).reshape(3),
            normal=n,
            label=int(label),
            path_index=int(path_index),
            distance=float(distance),
        ))

    def remove_planes_by_index(self, indices):
        rm = set(int(i) for i in indices)
        self.workspace.planes = [p for i, p in enumerate(self.workspace.planes) if i not in rm]

    def update_plane(self, index, center=None, normal=None, label=None):
        p = self.workspace.planes[int(index)]
        if center is not None:
            p.center = np.asarray(center, dtype=float).reshape(3)
        if normal is not None:
            n = np.asarray(normal, dtype=float).reshape(3)
            p.normal = n / (np.linalg.norm(n) + 1e-12)
        if label is not None:
            p.label = int(label)

    def replace_planes(self, planes):
        out = []
        for p in planes:
            n = np.asarray(p["normal"], dtype=float).reshape(3)
            n = n / (np.linalg.norm(n) + 1e-12)
            out.append(PlaneData(
                center=np.asarray(p["center"], dtype=float).reshape(3),
                normal=n,
                label=int(p.get("label", 1)),
                path_index=int(p.get("path_index", 0)),
                distance=float(p.get("distance", 0.0)),
            ))
        self.workspace.planes = out
