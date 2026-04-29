import numpy as np
from scipy.spatial import cKDTree

from .graph import graph_to_networkx
from .paths import _vector_orientation_text, inter_points


def _orient_node_paths_by_flow(node_paths, graph_points, flow_xyzt3=None, segmask_binary_4d=None,
                               spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0),
                               confidence_eps=0.05):
    if flow_xyzt3 is None or segmask_binary_4d is None:
        return [list(map(int, p)) for p in node_paths]
    flow = np.asarray(flow_xyzt3, dtype=float)
    mask = np.asarray(segmask_binary_4d, dtype=bool)
    spacing = np.asarray(spacing, dtype=float).reshape(3)
    origin = np.asarray(origin, dtype=float).reshape(3)
    if mask.ndim == 3:
        mask = np.repeat(mask[..., np.newaxis], flow.shape[3], axis=3)
    out = []
    for nodes in node_paths:
        nodes = list(map(int, nodes))
        if len(nodes) < 2:
            out.append(nodes)
            continue
        pts = np.asarray(graph_points[nodes], dtype=float)
        vox = np.rint((pts - origin.reshape(1, 3)) / (spacing.reshape(1, 3) + 1e-12)).astype(int)
        for k in range(3):
            vox[:, k] = np.clip(vox[:, k], 0, flow.shape[k] - 1)
        vec = flow[vox[:, 0], vox[:, 1], vox[:, 2], :, :]
        m = mask[vox[:, 0], vox[:, 1], vox[:, 2], :]
        flow_per_pt = np.sum(vec * m[..., None], axis=1)
        seg_dl = pts[1:] - pts[:-1]
        flow_mid = 0.5 * (flow_per_pt[1:] + flow_per_pt[:-1])
        dots = np.sum(flow_mid * seg_dl, axis=1)
        score = float(np.sum(dots))
        norm_den = float(np.sum(
            np.linalg.norm(flow_mid, axis=1) * np.linalg.norm(seg_dl, axis=1)))
        confidence = score / (norm_den + 1e-12)

        reverse = False
        if abs(confidence) >= float(confidence_eps):
            reverse = (confidence < 0.0)
        else:
            geom_dir = pts[-1] - pts[0]
            mean_flow_global = flow_per_pt.sum(axis=0)
            if (np.linalg.norm(geom_dir) > 1e-12
                    and np.linalg.norm(mean_flow_global) > 1e-12
                    and np.dot(mean_flow_global, geom_dir) < 0):
                reverse = True
        if reverse:
            nodes = nodes[::-1]
        out.append(nodes)
    return out


def find_path_forks(node_paths, node_points):
    forks = []
    seen = set()
    for i, path in enumerate(node_paths):
        if len(path) == 0:
            continue
        last_node = int(path[-1])
        fork = {"left": [int(i)], "right": [], "crosspoint": np.asarray(node_points[last_node], dtype=float).tolist(), "node": int(last_node)}
        found = False
        for j, other in enumerate(node_paths):
            if i == j or len(other) == 0:
                continue
            if last_node == int(other[0]):
                fork["right"].append(int(j))
                found = True
            if last_node == int(other[-1]):
                fork["left"].append(int(j))
                found = True
        if found:
            left_sorted = tuple(sorted(set(fork["left"])))
            right_sorted = tuple(sorted(set(fork["right"])))
            key = (left_sorted, right_sorted, int(last_node))
            if key not in seen:
                seen.add(key)
                fork["left"] = list(left_sorted)
                fork["right"] = list(right_sorted)
                forks.append(fork)
    return forks


def build_path_info(node_paths, graph_points, forks=None):
    path_to_forks = {}
    path_to_roles = {}
    path_to_incoming = {}
    path_to_outgoing = {}
    for fork_id, fork in enumerate(forks or []):
        left = [int(pid) for pid in fork.get("left", [])]
        right = [int(pid) for pid in fork.get("right", [])]
        for pid in left:
            path_to_forks.setdefault(pid, []).append(int(fork_id))
            path_to_roles.setdefault(pid, []).append({"fork_id": int(fork_id), "role": "incoming"})
            path_to_incoming.setdefault(pid, set()).update(x for x in left if x != pid)
            path_to_outgoing.setdefault(pid, set()).update(right)
        for pid in right:
            path_to_forks.setdefault(pid, []).append(int(fork_id))
            path_to_roles.setdefault(pid, []).append({"fork_id": int(fork_id), "role": "outgoing"})
            path_to_incoming.setdefault(pid, set()).update(left)
            path_to_outgoing.setdefault(pid, set()).update(x for x in right if x != pid)
    infos = []
    for i, nodes in enumerate(node_paths):
        pts = np.asarray(graph_points[nodes], dtype=float) if len(nodes) else np.empty((0, 3), dtype=float)
        d = pts[-1] - pts[0] if len(pts) >= 2 else np.zeros(3, dtype=float)
        nd = d / (np.linalg.norm(d) + 1e-12) if np.linalg.norm(d) > 0 else np.zeros(3, dtype=float)
        incoming_ids = sorted(int(x) for x in path_to_incoming.get(int(i), set()) if int(x) != int(i))
        outgoing_ids = sorted(int(x) for x in path_to_outgoing.get(int(i), set()) if int(x) != int(i))
        infos.append({
            "path_index": int(i),
            "start_node": int(nodes[0]) if len(nodes) else -1,
            "end_node": int(nodes[-1]) if len(nodes) else -1,
            "start_point": pts[0].tolist() if len(pts) else [0.0, 0.0, 0.0],
            "end_point": pts[-1].tolist() if len(pts) else [0.0, 0.0, 0.0],
            "direction_vector": nd.tolist(),
            "direction_text": _vector_orientation_text(nd),
            "fork_ids": path_to_forks.get(int(i), []),
            "fork_roles": path_to_roles.get(int(i), []),
            "incoming_path_ids": incoming_ids,
            "outgoing_path_ids": outgoing_ids,
        })
    return infos


def segment_vessels_from_graph_and_mask(segmask_3d, graph, resolution, flow_xyzt3=None,
                                        segmask_binary_4d=None, origin=(0, 0, 0)):
    mask3d = np.asarray(segmask_3d, dtype=bool)
    G = graph_to_networkx(graph)
    if G.number_of_nodes() == 0:
        return mask3d.astype(np.int16), [], [], [], []

    degree = dict(G.degree())
    endpoints = [n for n, d in degree.items() if d == 1]
    branch_nodes = [n for n, d in degree.items() if d >= 3]
    keynodes = set(endpoints + branch_nodes)

    node_paths = []
    visited_edges = set()

    for start in keynodes:
        for curr in G.neighbors(start):
            edge0 = tuple(sorted((start, curr)))
            if edge0 in visited_edges:
                continue

            path = [start, curr]
            visited_edges.add(edge0)
            prev = start

            while curr not in keynodes:
                nbrs = list(G.neighbors(curr))
                if len(nbrs) != 2:
                    break
                nxt = nbrs[0] if nbrs[0] != prev else nbrs[1]
                edge = tuple(sorted((curr, nxt)))
                if edge in visited_edges:
                    break
                path.append(nxt)
                visited_edges.add(edge)
                prev, curr = curr, nxt

            if curr in keynodes:
                node_paths.append(path)
    node_paths = _orient_node_paths_by_flow(
        node_paths, np.asarray(graph.points, dtype=float),
        flow_xyzt3=flow_xyzt3, segmask_binary_4d=segmask_binary_4d,
        spacing=resolution, origin=origin,
    )
    point_paths = [np.asarray(graph.points[np.asarray(nodes, dtype=int)], dtype=float) for nodes in node_paths]
    forks = find_path_forks(node_paths, np.asarray(graph.points, dtype=float))
    path_info = build_path_info(node_paths, np.asarray(graph.points, dtype=float), forks)

    labels = np.zeros(mask3d.shape, dtype=np.int16)
    if len(point_paths) == 0:
        labels[mask3d > 0] = 1
        return labels, [], [], path_info, forks

    all_pts_list = []
    all_ids_list = []
    for i, p in enumerate(point_paths):
        fine = inter_points(p, time=10)
        all_pts_list.append(fine)
        all_ids_list.extend([i] * len(fine))
    tree = cKDTree(np.vstack(all_pts_list))
    all_ids = np.array(all_ids_list, dtype=int)

    idx_mask = np.argwhere(mask3d > 0)
    if len(idx_mask) > 0:
        world = idx_mask.astype(float) * np.asarray(resolution, dtype=float).reshape(1, 3)
        _, nearest = tree.query(world)
        for k, idx in enumerate(idx_mask):
            labels[tuple(idx)] = all_ids[nearest[k]] + 1
    return labels, point_paths, node_paths, path_info, forks
