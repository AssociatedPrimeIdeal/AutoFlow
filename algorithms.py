import os

import numpy as np
import pyvista as pv
import networkx as nx
import h5py
from scipy.ndimage import binary_closing, binary_opening, gaussian_filter, label
from scipy.spatial import cKDTree
from scipy.signal import savgol_filter
from skimage.morphology import skeletonize

from models import PlaneData, GraphData, SkeletonParams


def _axis_pair(a):
    mp = {
        "LR": ("LR", "RL"), "RL": ("LR", "RL"),
        "AP": ("AP", "PA"), "PA": ("AP", "PA"),
        "HF": ("HF", "FH"), "FH": ("HF", "FH"),
    }
    a = a.upper()
    if a not in mp:
        raise ValueError(a)
    return mp[a][0]


def _need_flip(curr_label, target_label):
    c, t = curr_label.upper(), target_label.upper()
    if _axis_pair(c) != _axis_pair(t):
        raise ValueError(f"{c} vs {t}")
    return c != t


def _permute_spatial(arr, curr_order, target_order, spatial_axes=(0, 1, 2)):
    curr_order = [x.upper() for x in curr_order]
    target_order = [x.upper() for x in target_order]
    cb = [_axis_pair(x) for x in curr_order]
    tb = [_axis_pair(x) for x in target_order]
    src_pos = [cb.index(x) for x in tb]
    axes = list(range(arr.ndim))
    new_spatial = [spatial_axes[p] for p in src_pos]
    for k, ax in enumerate(spatial_axes):
        axes[ax] = new_spatial[k]
    return np.transpose(arr, axes), src_pos


def _flip_axes(arr, axes_to_flip):
    for ax in axes_to_flip:
        arr = np.flip(arr, axis=ax)
    return arr


def reorient(mag, flow, segmask, venc, resolution, spatial_order, venc_order,
             target_spatial_order, target_venc_order, return_velocity=False):
    spatial_order = [s.upper() for s in spatial_order]
    venc_order = [v.upper() for v in venc_order]
    target_spatial_order = [s.upper() for s in target_spatial_order]
    target_venc_order = [v.upper() for v in target_venc_order]
    resolution = np.asarray(resolution, dtype=np.float32)
    venc = np.asarray(venc, dtype=np.float32)
    if venc.ndim == 0:
        venc = np.full(3, float(venc), dtype=np.float32)

    mag_r, src_pos_mag = _permute_spatial(mag, spatial_order, target_spatial_order, spatial_axes=(0, 1, 2))
    flow_r, src_pos_flow = _permute_spatial(flow, spatial_order, target_spatial_order, spatial_axes=(0, 1, 2))
    seg_r, src_pos_seg = _permute_spatial(segmask, spatial_order, target_spatial_order, spatial_axes=(0, 1, 2))

    cb = [_axis_pair(s) for s in spatial_order]
    tb = [_axis_pair(s) for s in target_spatial_order]
    res_perm = np.array([cb.index(x) for x in tb], dtype=int)
    resolution_r = resolution[res_perm]

    flip_mag = [i for i in range(3) if _need_flip(spatial_order[src_pos_mag[i]], target_spatial_order[i])]
    flip_flow = [i for i in range(3) if _need_flip(spatial_order[src_pos_flow[i]], target_spatial_order[i])]
    flip_seg = [i for i in range(3) if _need_flip(spatial_order[src_pos_seg[i]], target_spatial_order[i])]
    mag_r = _flip_axes(mag_r, flip_mag)
    flow_r = _flip_axes(flow_r, flip_flow)
    seg_r = _flip_axes(seg_r, flip_seg)

    vb = [_axis_pair(v) for v in venc_order]
    tb2 = [_axis_pair(v) for v in target_venc_order]
    comp_perm = np.array([vb.index(x) for x in tb2], dtype=int)

    flow_r = flow_r[..., comp_perm]
    venc_r = venc[comp_perm]

    sign3 = np.array([(-1.0 if _need_flip(venc_order[comp_perm[i]], target_venc_order[i]) else 1.0)
                      for i in range(3)], dtype=np.float32)
    flow_r = flow_r * sign3.reshape((1, 1, 1, 1, -1))

    if return_velocity:
        flow_r = (flow_r / np.pi) * venc_r.reshape((1, 1, 1, 1, -1))

    mag_max = np.max(np.abs(mag_r))
    if mag_max > 0:
        mag_r = mag_r / mag_max

    return flow_r, mag_r, seg_r, venc_r, resolution_r


def _ensure_flow_mag_time_and_segmask(flow, mag, segmask):
    flow = np.asarray(flow)
    mag = np.asarray(mag)
    segmask = np.asarray(segmask)
    if flow.ndim == 4 and flow.shape[-1] == 3:
        flow = flow[..., np.newaxis, :]
    if flow.ndim != 5 or flow.shape[-1] != 3:
        raise ValueError(f"flow must be XYZTV or XYZV with 3 components, got {flow.shape}")
    nt = int(flow.shape[3])
    if mag.ndim == 3:
        mag = np.repeat(mag[..., np.newaxis], nt, axis=3)
    elif mag.ndim == 4 and mag.shape[3] == 1 and nt > 1:
        mag = np.repeat(mag, nt, axis=3)
    elif mag.ndim != 4:
        raise ValueError(f"mag must be XYZT or XYZ, got {mag.shape}")
    if mag.shape[3] != nt:
        if mag.shape[3] == 1:
            mag = np.repeat(mag, nt, axis=3)
        else:
            raise ValueError(f"mag time dimension {mag.shape[3]} does not match flow {nt}")
    if segmask.ndim == 3:
        segmask = np.repeat(segmask[..., np.newaxis], nt, axis=3)
    elif segmask.ndim == 4 and segmask.shape[3] == 1 and nt > 1:
        segmask = np.repeat(segmask, nt, axis=3)
    elif segmask.ndim != 4:
        raise ValueError(f"segmask must be XYZT or XYZ, got {segmask.shape}")
    if segmask.shape[3] != nt:
        if segmask.shape[3] == 1:
            segmask = np.repeat(segmask, nt, axis=3)
        else:
            raise ValueError(f"segmask time dimension {segmask.shape[3]} does not match flow {nt}")
    return (
        np.ascontiguousarray(flow, dtype=np.float32),
        np.ascontiguousarray(mag, dtype=np.float32),
        np.ascontiguousarray(segmask, dtype=np.int16),
    )


def _reorient_spatial_only(arr, spatial_order, target_spatial_order):
    arr_r, src_pos = _permute_spatial(arr, spatial_order, target_spatial_order, spatial_axes=(0, 1, 2))
    flip_axes = [i for i in range(3) if _need_flip(spatial_order[src_pos[i]], target_spatial_order[i])]
    return _flip_axes(arr_r, flip_axes)


def _compute_spatial_bbox(mask, pad=0):
    m = np.asarray(mask, dtype=bool)
    if m.ndim > 3:
        m = np.any(m, axis=tuple(range(3, m.ndim)))
    if not np.any(m):
        return tuple(slice(0, int(m.shape[i])) for i in range(3))
    idx = np.argwhere(m)
    lo = np.maximum(idx.min(axis=0) - int(pad), 0)
    hi = np.minimum(idx.max(axis=0) + int(pad) + 1, np.array(m.shape[:3], dtype=int))
    return tuple(slice(int(lo[i]), int(hi[i])) for i in range(3))


def _target_bbox_to_source_slices(shape_raw, spatial_order, target_spatial_order, bbox_target):
    spatial_order = [str(x).upper() for x in spatial_order]
    target_spatial_order = [str(x).upper() for x in target_spatial_order]
    cb = [_axis_pair(s) for s in spatial_order]
    tb = [_axis_pair(s) for s in target_spatial_order]
    src_pos = [cb.index(x) for x in tb]
    out = [slice(0, int(shape_raw[i])) for i in range(3)]
    for target_axis, raw_axis in enumerate(src_pos):
        s = int(bbox_target[target_axis].start)
        e = int(bbox_target[target_axis].stop)
        if _need_flip(spatial_order[raw_axis], target_spatial_order[target_axis]):
            out[raw_axis] = slice(int(shape_raw[raw_axis]) - e, int(shape_raw[raw_axis]) - s)
        else:
            out[raw_axis] = slice(s, e)
    return tuple(out)

def _sigma_from_complex(img_complex, venc):
    venc = np.asarray(venc, dtype=np.float32)
    if venc.ndim == 0:
        venc = np.full(3, float(venc), dtype=np.float32)
    ref = np.abs(img_complex[..., 0]).astype(np.float32)
    enc = np.abs(img_complex[..., 1:4]).astype(np.float32)
    kv = np.pi / venc.reshape((1, 1, 1, 1, 3))
    ratio = ref[..., None] / np.clip(enc, 1e-12, None)
    ratio = np.clip(ratio, 1.0, None)
    sigma = np.sqrt(2.0 * np.log(ratio)) / kv
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    return sigma.astype(np.float32)

def _reorient_component_abs(arr, spatial_order, target_spatial_order, venc_order, target_venc_order):
    spatial_order = [s.upper() for s in spatial_order]
    venc_order = [v.upper() for v in venc_order]
    target_spatial_order = [s.upper() for s in target_spatial_order]
    target_venc_order = [v.upper() for v in target_venc_order]

    arr_r, src_pos = _permute_spatial(arr, spatial_order, target_spatial_order, spatial_axes=(0, 1, 2))
    flip_axes = [i for i in range(3) if _need_flip(spatial_order[src_pos[i]], target_spatial_order[i])]
    arr_r = _flip_axes(arr_r, flip_axes)

    vb = [_axis_pair(v) for v in venc_order]
    tb = [_axis_pair(v) for v in target_venc_order]
    comp_perm = np.array([vb.index(x) for x in tb], dtype=int)

    return arr_r[..., comp_perm]
def load_h5_data(path):
    target_spatial_order = ("LR", "AP", "FH")
    target_venc_order = ("LR", "AP", "FH")
    with h5py.File(path, "r") as g:
        if "img_complex" not in g or "segmask" not in g:
            raise ValueError(f"h5 must contain img_complex and segmask: {path}")
        VENC = g["VENC"][:] if "VENC" in g else np.array([150, 150, 150], dtype=float)
        resolution = g["Resolution"][:] if "Resolution" in g else np.array([1, 1, 1], dtype=float)
        origin = np.array([0.0, 0.0, 0.0], dtype=float)
        rr = float(g["RR"][()]) if "RR" in g else 1000.0
        spatial_order = g["SpatialOrder"][:].astype(str) if "SpatialOrder" in g else np.array(["FH", "AP", "LR"])
        venc_order = g["VENCOrder"][:].astype(str) if "VENCOrder" in g else np.array(["FH", "AP", "LR"])

        segmask_ds = g["segmask"]
        segmask_full = segmask_ds[:].astype(np.int16)
        segmask_target = _reorient_spatial_only(segmask_full, spatial_order, target_spatial_order)
        bbox_target = _compute_spatial_bbox(segmask_target, pad=2)
        src_slices = _target_bbox_to_source_slices(segmask_full.shape[:3], spatial_order, target_spatial_order, bbox_target)
        segmask = segmask_ds[src_slices + (slice(None),) * (segmask_ds.ndim - 3)].astype(np.int16)

        img_ds = g["img_complex"]
        img_complex = img_ds[src_slices + (slice(None),) * (img_ds.ndim - 3)]

        mag = np.abs(img_complex[..., 0]).astype(np.float32)
        flow_raw = np.angle(img_complex[..., 1:4] * np.conj(img_complex[..., 0][..., None])).astype(np.float32)
        sigma_raw = _sigma_from_complex(img_complex, VENC)

        flow, mag_out, seg_r, venc_new, res_new = reorient(
            mag, flow_raw, segmask, venc=VENC, resolution=resolution,
            spatial_order=spatial_order, venc_order=venc_order,
            target_spatial_order=target_spatial_order,
            target_venc_order=target_venc_order,
            return_velocity=True,
        )
        sigma = _reorient_component_abs(
            sigma_raw,
            spatial_order=spatial_order,
            target_spatial_order=target_spatial_order,
            venc_order=venc_order,
            target_venc_order=target_venc_order,
        ).astype(np.float32)

        flow, mag_out, seg_r = _ensure_flow_mag_time_and_segmask(flow, mag_out, seg_r)
        tke_array = (0.5 * 1060.0 * np.sum((sigma / 100.0) ** 2, axis=-1)).astype(np.float32)

        return {
            "flow": flow,
            "mag": mag_out,
            "segmask": seg_r,
            "resolution": np.asarray(res_new, dtype=float),
            "origin": origin,
            "venc": np.asarray(venc_new, dtype=float),
            "rr": float(rr),
            "sigma": sigma,
            "tke_array": tke_array,
        }

def filter_segmask_labels(segmask_raw, labels_to_keep=None, labels_to_remove=None):
    return np.asarray(segmask_raw).copy()


def merge_segmask_to_3d(segmask_binary_4d):
    seg = np.asarray(segmask_binary_4d, dtype=bool)
    if seg.ndim == 3:
        return seg
    return np.any(seg, axis=3)


def _connected_components(mask, connectivity=1):
    m = np.asarray(mask, dtype=bool)
    if not np.any(m):
        return []
    struct = np.ones((3, 3, 3), dtype=bool) if connectivity == 2 else None
    lab, n = label(m, structure=struct)
    return [(int(i), lab == int(i)) for i in range(1, int(n) + 1)]


def remove_small_cc_from_binary_mask(segmask_binary, resolution, min_cc_volume_mm3):
    seg = np.asarray(segmask_binary, dtype=bool).copy()
    if seg.ndim == 3:
        mask_3d = seg
    elif seg.ndim == 4:
        mask_3d = np.any(seg, axis=3)
    else:
        return seg
    resolution = np.asarray(resolution, dtype=float).reshape(3)
    voxel_volume = float(np.prod(resolution))
    remove_mask = np.zeros_like(mask_3d, dtype=bool)
    for _, cc in _connected_components(mask_3d):
        n_voxels = int(np.sum(cc))
        volume_mm3 = n_voxels * voxel_volume
        if volume_mm3 < min_cc_volume_mm3:
            remove_mask |= cc
    if np.any(remove_mask):
        if seg.ndim == 4:
            seg[remove_mask] = False
        else:
            seg[remove_mask] = False
    return seg


def _component_bbox(mask):
    idx = np.argwhere(np.asarray(mask, dtype=bool))
    if len(idx) == 0:
        return None
    lo = idx.min(axis=0)
    hi = idx.max(axis=0) + 1
    return tuple(slice(int(lo[k]), int(hi[k])) for k in range(3))


def _preprocess_single_component(mask_3d, params):
    m = np.asarray(mask_3d, dtype=bool).copy()
    if not np.any(m):
        return m
    if params.do_closing:
        m = binary_closing(m).astype(bool)
    if params.do_opening:
        m = binary_opening(m).astype(bool)
    if params.gaussian_enabled and params.gaussian_sigma > 0:
        m = gaussian_filter(m.astype(float), sigma=params.gaussian_sigma) > 0.5
    return m.astype(bool)


def preprocess_mask_for_skeleton(mask_3d, params=None, resolution=None):
    if params is None:
        params = SkeletonParams()
    m = np.asarray(mask_3d, dtype=bool).copy()
    if not np.any(m):
        return m
    if resolution is None:
        resolution = np.array([1.0, 1.0, 1.0])
    resolution = np.asarray(resolution, dtype=float).reshape(3)
    voxel_volume = float(np.prod(resolution))
    out = np.zeros_like(m, dtype=bool)
    for _, cc in _connected_components(m):
        if params.remove_small_cc:
            n_voxels = int(np.sum(cc))
            volume_mm3 = n_voxels * voxel_volume
            if volume_mm3 < params.min_cc_volume_mm3:
                continue
        bbox = _component_bbox(cc)
        if bbox is None:
            continue
        local = cc[bbox]
        proc = _preprocess_single_component(local, params)
        if np.any(proc):
            out[bbox] |= proc
    return out.astype(bool)


def largest_connected_component(mask, connectivity=1):
    m = np.asarray(mask).astype(bool)
    if not m.any():
        return m
    struct = np.ones((3, 3, 3), dtype=bool) if connectivity == 2 else None
    lab, n = label(m, structure=struct)
    if n <= 1:
        return m
    cnt = np.bincount(lab.ravel())
    cnt[0] = 0
    return lab == cnt.argmax()


def generate_skeleton_from_mask3d(mask3d, resolution):
    mask3d = np.asarray(mask3d, dtype=bool)
    skel = np.zeros_like(mask3d, dtype=bool)
    for _, cc in _connected_components(mask3d):
        bbox = _component_bbox(cc)
        if bbox is None:
            continue
        local = cc[bbox]
        if not np.any(local):
            continue
        local_skel = skeletonize(local).astype(bool)
        if np.any(local_skel):
            skel[bbox] |= local_skel
    pts = np.argwhere(skel > 0).astype(float) * np.asarray(resolution, dtype=float).reshape(1, 3)
    return pts, mask3d


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


def build_multilabel_surface(labels_3d, spacing, origin=(0, 0, 0)):
    labels = np.asarray(labels_3d, dtype=np.int32)
    if not np.any(labels > 0):
        return None
    grid = pv.ImageData()
    grid.dimensions = np.array(labels.shape) + 1
    grid.spacing = tuple(float(x) for x in np.asarray(spacing).reshape(-1)[:3])
    grid.origin = tuple(float(x) for x in np.asarray(origin).reshape(-1)[:3])
    grid.cell_data["label"] = labels.flatten(order="F")
    threshed = grid.threshold(0.5, scalars="label")
    if threshed.n_cells == 0:
        return None
    surf = threshed.extract_surface()
    return surf


def build_multilabel_surface_t(labels_4d, t, spacing, origin=(0, 0, 0)):
    return build_multilabel_surface(
        np.asarray(labels_4d)[..., int(t)], spacing, origin)


def build_binary_surface_t(mask_4d, t, spacing, origin=(0, 0, 0)):
    mask_xyz = np.asarray(mask_4d)[..., int(t)] > 0
    return build_surface_from_mask3d(mask_xyz, spacing, origin)


def build_surface_from_mask3d(mask_xyz, spacing, origin=(0, 0, 0), smooth_iter=1000):
    mask_xyz = np.asarray(mask_xyz) > 0
    grid = pv.ImageData()
    grid.dimensions = mask_xyz.shape
    grid.spacing = tuple(float(x) for x in np.asarray(spacing).reshape(-1)[:3])
    grid.origin = tuple(float(x) for x in np.asarray(origin).reshape(-1)[:3])
    grid.point_data["values"] = mask_xyz.astype(np.float32).ravel(order="F")
    th = grid.threshold(0.1, scalars="values")
    surf = th.extract_surface()
    if smooth_iter > 0 and surf.n_points > 0:
        surf = surf.smooth(n_iter=smooth_iter)
    return surf


def _vector_orientation_text(vec):
    vec = np.asarray(vec, dtype=float).reshape(3)
    n = np.linalg.norm(vec)
    if n <= 1e-12:
        return ""
    v = vec / n
    axis_names = ["LR", "AP", "HF"]
    idx = int(np.argmax(np.abs(v)))
    sign = "+" if v[idx] >= 0 else "-"
    return f"{axis_names[idx]}{sign}"


def _orient_node_paths_by_flow(node_paths, graph_points, flow_xyzt3=None, segmask_binary_4d=None,
                               spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
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
        geom_dir = pts[-1] - pts[0]
        if np.linalg.norm(geom_dir) <= 1e-12:
            out.append(nodes)
            continue
        vox = np.rint((pts - origin.reshape(1, 3)) / (spacing.reshape(1, 3) + 1e-12)).astype(int)
        for k in range(3):
            vox[:, k] = np.clip(vox[:, k], 0, flow.shape[k] - 1)
        vec = flow[vox[:, 0], vox[:, 1], vox[:, 2], :, :]
        m = mask[vox[:, 0], vox[:, 1], vox[:, 2], :]
        vec = vec * m[..., None]
        mean_flow = np.sum(vec, axis=(0, 1))
        if np.linalg.norm(mean_flow) > 1e-12 and np.dot(mean_flow, geom_dir) < 0:
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


def smooth_path_savgol(points, window=15, polyorder=3):
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    n = len(pts)
    if n <= polyorder:
        return pts.copy()
    w = int(window)
    if w % 2 == 0:
        w += 1
    w = min(w, n if n % 2 == 1 else max(1, n - 1))
    if w <= polyorder:
        return pts.copy()
    out = np.zeros_like(pts, dtype=float)
    for dim in range(3):
        out[:, dim] = savgol_filter(pts[:, dim], w, polyorder)
    out[0] = pts[0]
    out[-1] = pts[-1]
    return out


def inter_points(points, time=100):
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(pts) <= 2:
        return pts
    x = np.arange(len(pts))
    fine_x = np.linspace(0, len(pts) - 1, num=max(len(pts) * time, len(pts)))
    return np.stack([np.interp(fine_x, x, pts[:, k]) for k in range(3)], axis=1)


def generate_planes_from_paths(
    paths,
    cross_section_distance=20.0,
    start_distance=5.0,
    end_distance=0.0,
    smoothing_window=15,
    smoothing_polyorder=3,
    inter_time=9,
    use_center_plane=True,
):
    planes = []
    smooth_paths = []

    for path_i, path in enumerate(paths):
        path = inter_points(path, time=inter_time)
        if len(path) < 2:
            smooth_paths.append(path)
            continue

        path_smooth = smooth_path_savgol(path, window=smoothing_window, polyorder=smoothing_polyorder)
        smooth_paths.append(path_smooth)
        if len(path_smooth) < 2:
            continue

        seglens = np.linalg.norm(np.diff(path_smooth, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seglens)])
        total = cum[-1]

        if use_center_plane:
            target = total * 0.5
            j = int(np.searchsorted(cum, target, side="right") - 1)
            j = min(max(0, j), len(path_smooth) - 2)
            seg = path_smooth[j + 1] - path_smooth[j]
            seglen = np.linalg.norm(seg) + 1e-12
            alpha = (target - cum[j]) / seglen
            center = path_smooth[j] + alpha * seg
            i0 = max(0, j - 1)
            i1 = min(len(path_smooth) - 1, j + 2)
            tangent = path_smooth[i1] - path_smooth[i0]
            normal = tangent / (np.linalg.norm(tangent) + 1e-12)
            planes.append(
                PlaneData(
                    center=center,
                    normal=normal,
                    label=path_i + 1,
                    path_index=path_i,
                    distance=float(target),
                )
            )
            continue

        effective = total - float(end_distance)

        if effective <= float(start_distance):
            mid = len(path_smooth) // 2
            i0 = max(0, mid - 1)
            i1 = min(len(path_smooth) - 1, mid + 1)
            d = path_smooth[i1] - path_smooth[i0]
            n = d / (np.linalg.norm(d) + 1e-12)
            planes.append(
                PlaneData(
                    center=path_smooth[mid],
                    normal=n,
                    label=path_i + 1,
                    path_index=path_i,
                    distance=float(cum[mid]),
                )
            )
            continue

        planes_before = len(planes)
        for target in np.arange(float(start_distance), effective + 1e-8, float(cross_section_distance)):
            j = int(np.searchsorted(cum, target, side="right") - 1)
            j = min(max(0, j), len(path_smooth) - 2)
            seg = path_smooth[j + 1] - path_smooth[j]
            seglen = np.linalg.norm(seg) + 1e-12
            alpha = (target - cum[j]) / seglen
            center = path_smooth[j] + alpha * seg
            i0 = max(0, j - 1)
            i1 = min(len(path_smooth) - 1, j + 2)
            tangent = path_smooth[i1] - path_smooth[i0]
            normal = tangent / (np.linalg.norm(tangent) + 1e-12)
            planes.append(
                PlaneData(
                    center=center,
                    normal=normal,
                    label=path_i + 1,
                    path_index=path_i,
                    distance=float(target),
                )
            )

        if len(planes) == planes_before and float(cross_section_distance) > total > 0:
            target_mid = total * 0.5
            j = int(np.searchsorted(cum, target_mid, side="right") - 1)
            j = min(max(0, j), len(path_smooth) - 2)
            seg = path_smooth[j + 1] - path_smooth[j]
            seglen = np.linalg.norm(seg) + 1e-12
            alpha = (target_mid - cum[j]) / seglen
            center = path_smooth[j] + alpha * seg
            i0 = max(0, j - 1)
            i1 = min(len(path_smooth) - 1, j + 2)
            tangent = path_smooth[i1] - path_smooth[i0]
            normal = tangent / (np.linalg.norm(tangent) + 1e-12)
            planes.append(
                PlaneData(
                    center=center,
                    normal=normal,
                    label=path_i + 1,
                    path_index=path_i,
                    distance=float(target_mid),
                )
            )

    return planes, smooth_paths


def generate_seed_points(mask_3d, spacing, origin, ratio=0.02, rng_seed=0,
                         min_seeds=50):
    idx = np.argwhere(np.asarray(mask_3d, dtype=bool))
    K = len(idx)
    if K == 0:
        return np.empty((0, 3), dtype=float)
    n = int(min(K, max(int(min_seeds), int(round(K * float(ratio))))))
    rng = np.random.default_rng(int(rng_seed))
    pick = idx[rng.choice(K, size=n, replace=False)]
    sp = np.asarray(spacing, dtype=float).reshape(1, 3)
    org = np.asarray(origin, dtype=float).reshape(1, 3)
    return (org + pick.astype(float) * sp).astype(float)


def generate_streamlines_at_t(flow_xyzt3, t, seeds, spacing, origin, mask_3d=None,
                              max_steps=2000, terminal_speed=0.01,
                              seed_ratio=0.02, min_seeds=50, rng_seed=0):
    flow_t = np.asarray(flow_xyzt3[..., int(t), :], dtype=np.float32)
    velocity = create_uniform_vector(
        flow_t[..., 0] / 100.0,
        flow_t[..., 1] / 100.0,
        flow_t[..., 2] / 100.0,
        spacing, origin=origin)
    if mask_3d is None:
        return None
    mesh = create_uniform_grid(np.asarray(mask_3d) > 0, spacing, origin=origin)
    mesh = mesh.threshold(0.1)
    if mesh.n_points == 0:
        return None
    volume = mesh.sample(velocity)
    volume.set_active_scalars("Velocity")
    if volume.points.shape[0] == 0:
        return None
    if seeds is None or len(seeds) == 0:
        seeds = generate_seed_points(
            mask_3d,
            spacing,
            origin,
            ratio=seed_ratio,
            rng_seed=int(rng_seed) + int(t),
            min_seeds=min_seeds,
        )
    source = np.asarray(seeds, dtype=float)
    if source.size == 0:
        return None
    sl = volume.streamlines_from_source(
        vectors="vector",
        source=source,
        integrator_type=4,
        max_steps=int(max_steps),
        terminal_speed=float(terminal_speed),
        compute_vorticity=False,
    )
    if sl is None or sl.n_points == 0:
        return None
    return sl


def _build_branch_grid(branch_labels_3d, spacing, origin):
    if branch_labels_3d is None:
        return None
    branch_labels_3d = np.asarray(branch_labels_3d, dtype=np.int16)
    branch_grid = pv.ImageData()
    branch_grid.dimensions = np.array(branch_labels_3d.shape) + 1
    branch_grid.spacing = tuple(np.asarray(spacing, dtype=float).reshape(-1)[:3])
    branch_grid.origin = tuple(np.asarray(origin, dtype=float).reshape(-1)[:3])
    branch_grid.cell_data["branch_id"] = branch_labels_3d.reshape(-1, order="F")
    return branch_grid


def _largest_region_by_area(poly):
    if poly is None or poly.n_cells == 0:
        return None
    poly = poly.compute_cell_sizes(area=True)
    conn = poly.connectivity()
    if conn.n_cells == 0 or "RegionId" not in conn.cell_data:
        return poly
    region_ids = np.asarray(conn.cell_data["RegionId"])
    areas = np.asarray(conn.cell_data["Area"]) if "Area" in conn.cell_data else np.ones(conn.n_cells, dtype=float)
    best_region = None
    best_area = -1.0
    for rid in np.unique(region_ids):
        s = float(np.sum(areas[region_ids == rid]))
        if s > best_area:
            best_area = s
            best_region = int(rid)
    if best_region is None:
        return None
    out = conn.extract_cells(np.where(region_ids == best_region)[0])
    if out is None or out.n_cells == 0:
        return None
    return out.compute_cell_sizes(area=True)


def extract_plane_cross_section(mask_xyz, plane, spacing, origin, branch_grid=None, target_label=None):
    mask_xyz = np.asarray(mask_xyz, dtype=bool)
    if not np.any(mask_xyz):
        return None
    mesh = create_uniform_grid(mask_xyz, spacing, origin=origin)
    mesh = mesh.threshold(0.1)
    if mesh.n_cells == 0:
        return None
    pg = mesh.slice(normal=np.asarray(plane.normal, dtype=float), origin=np.asarray(plane.center, dtype=float))
    if pg is None or pg.n_cells == 0:
        return None
    pg = pg.compute_cell_sizes(area=True)
    if branch_grid is not None and target_label is not None and int(target_label) > 0:
        centers = pg.cell_centers().sample(branch_grid)
        bid = np.asarray(centers.point_data.get("branch_id", []))
        if len(bid) == 0:
            return None
        keep = np.where(bid == int(target_label))[0]
        if len(keep) == 0:
            return None
        pg = pg.extract_cells(keep)
        if pg is None or pg.n_cells == 0:
            return None
        pg = pg.compute_cell_sizes(area=True)
    return _largest_region_by_area(pg)


def _flow_grid_for_t(flow_t, spacing, origin):
    flow_t = np.asarray(flow_t, dtype=np.float32)
    grid = pv.ImageData()
    grid.dimensions = np.array(flow_t.shape[:3]) + 1
    grid.spacing = tuple(np.asarray(spacing, dtype=float).reshape(-1)[:3])
    grid.origin = tuple(np.asarray(origin, dtype=float).reshape(-1)[:3])
    grid.cell_data["flow"] = flow_t.reshape(-1, flow_t.shape[-1], order="F")
    return grid


def _extract_plane_flow_region(mask_xyz, flow_t, plane, spacing, origin, branch_grid=None, target_label=None):
    mask_xyz = np.asarray(mask_xyz, dtype=bool)
    if not np.any(mask_xyz):
        return None
    grid = create_uniform_grid(mask_xyz, spacing, origin=origin, name="mask")
    grid.cell_data["flow"] = np.asarray(flow_t, dtype=np.float32).reshape(-1, 3, order="F")
    mesh = grid.threshold(0.1, scalars="mask")
    if mesh is None or mesh.n_cells == 0:
        return None
    pg = mesh.slice(normal=np.asarray(plane.normal, dtype=float), origin=np.asarray(plane.center, dtype=float))
    if pg is None or pg.n_cells == 0:
        return None
    pg = pg.compute_cell_sizes(area=True)
    if branch_grid is not None and target_label is not None and int(target_label) > 0:
        centers = pg.cell_centers().sample(branch_grid)
        bid = np.asarray(centers.point_data.get("branch_id", []))
        if len(bid) == 0:
            return None
        keep = np.where(bid == int(target_label))[0]
        if len(keep) == 0:
            return None
        pg = pg.extract_cells(keep)
        if pg is None or pg.n_cells == 0:
            return None
        pg = pg.compute_cell_sizes(area=True)
    return _largest_region_by_area(pg)


def generate_streamlines_from_plane_at_t(flow_xyzt3, t, plane, spacing, origin,
                                          mask_3d=None, max_steps=2000,
                                          terminal_speed=0.01,
                                          seed_ratio=0.02, min_seeds=50,
                                          rng_seed=0,
                                          branch_labels_3d=None):
    flow_t = np.asarray(flow_xyzt3[..., int(t), :], dtype=np.float32)
    velocity = create_uniform_vector(
        flow_t[..., 0] / 100.0,
        flow_t[..., 1] / 100.0,
        flow_t[..., 2] / 100.0,
        spacing, origin=origin)
    if mask_3d is None:
        return None
    mesh = create_uniform_grid(np.asarray(mask_3d) > 0, spacing, origin=origin)
    mesh = mesh.threshold(0.1)
    if mesh.n_points == 0:
        return None
    volume = mesh.sample(velocity)
    volume.set_active_scalars("Velocity")
    if volume.points.shape[0] == 0:
        return None
    branch_grid = _build_branch_grid(branch_labels_3d, spacing, origin)
    target_label = None
    if branch_labels_3d is not None:
        target_label = int(getattr(plane, "label", 0) or 0)
        if target_label <= 0:
            ijk = np.rint((np.asarray(plane.center, dtype=float) - np.asarray(origin, dtype=float).reshape(3)) / (np.asarray(spacing, dtype=float).reshape(3) + 1e-12)).astype(int)
            ijk = np.clip(ijk, 0, np.array(np.asarray(branch_labels_3d).shape) - 1)
            target_label = int(np.asarray(branch_labels_3d)[ijk[0], ijk[1], ijk[2]])
    pg = extract_plane_cross_section(mask_3d, plane, spacing, origin, branch_grid=branch_grid, target_label=target_label)
    if pg is None or pg.n_cells == 0:
        return None
    seeds = pg.cell_centers().points
    if len(seeds) == 0:
        return None
    n_seeds = int(max(int(min_seeds), int(np.ceil(len(seeds) * float(seed_ratio)))))
    n_seeds = max(1, min(int(n_seeds), len(seeds)))
    if len(seeds) > n_seeds:
        rng = np.random.default_rng(int(rng_seed) + 104729 * int(getattr(plane, "path_index", 0)) + int(t))
        idx = rng.choice(len(seeds), size=n_seeds, replace=False)
        seeds = seeds[np.sort(idx)]
    sl = volume.streamlines_from_source(
        vectors="vector",
        source=seeds,
        integrator_type=4,
        max_steps=int(max_steps),
        terminal_speed=float(terminal_speed),
        compute_vorticity=False,
    )
    if sl is None or sl.n_points == 0:
        return None
    return sl


def create_vector_volume_from_flow(flow_xyz3, spacing, origin=(0, 0, 0), scale=1.0):
    flow_xyz3 = np.asarray(flow_xyz3, dtype=np.float32)
    nx_, ny_, nz_, _ = flow_xyz3.shape
    grid = pv.ImageData(
        dimensions=(nx_, ny_, nz_),
        spacing=tuple(np.asarray(spacing, dtype=float).reshape(-1)[:3].tolist()),
        origin=tuple(np.asarray(origin, dtype=float).reshape(-1)[:3].tolist()),
    )
    vec = (flow_xyz3 * float(scale)).reshape(-1, 3, order="F")
    grid.point_data["vector"] = vec
    grid.point_data["speed"] = np.linalg.norm(vec, axis=1)
    grid.set_active_vectors("vector")
    return grid


def create_uniform_grid(mask, spacing, origin=(0, 0, 0), name="mask"):
    mesh = pv.ImageData()
    mesh.dimensions = np.array(mask.shape) + 1
    mesh.spacing = tuple(np.asarray(spacing, dtype=float).reshape(-1)[:3])
    mesh.origin = tuple(np.asarray(origin, dtype=float).reshape(-1)[:3])
    mesh.cell_data[name] = np.asarray(mask).flatten(order="F")
    return mesh


def create_uniform_vector(u, v, w, spacing, origin=(0, 0, 0)):
    vel = np.sqrt(u**2 + v**2 + w**2)
    mesh = pv.ImageData()
    mesh.dimensions = np.array(u.shape) + 1
    mesh.spacing = tuple(np.asarray(spacing, dtype=float).reshape(-1)[:3])
    mesh.origin = tuple(np.asarray(origin, dtype=float).reshape(-1)[:3])
    mesh.cell_data["u"] = u.flatten(order="F")
    mesh.cell_data["v"] = v.flatten(order="F")
    mesh.cell_data["w"] = w.flatten(order="F")
    mesh.cell_data["vector"] = np.stack([u.flatten(order="F"), v.flatten(order="F"), w.flatten(order="F")], axis=1)
    mesh.cell_data["Velocity"] = vel.flatten(order="F")
    mesh.set_active_scalars("Velocity")
    return mesh


def extract_vectors(polydata):
    return np.stack((polydata["u"], polydata["v"], polydata["w"]), axis=-1)


def get_orthogonal_vectors(vectors, point_normals):
    c = np.sum(vectors * point_normals, axis=1)
    return c[:, None] * point_normals, vectors - c[:, None] * point_normals


def get_vector_magnitude(vectors):
    return np.sqrt(np.sum(vectors * vectors, axis=1))


def calculate_gradient(pc0_tangent_mag, pc1_tangent_mag, pc2_tangent_mag, inward_distance, use_parabolic=True):
    x = np.array([0, 1, 2], dtype=float) * float(inward_distance)
    y = np.stack((pc0_tangent_mag, pc1_tangent_mag, pc2_tangent_mag), axis=1).T
    z = np.polynomial.polynomial.polyfit(x, y, len(x) - 1)
    x_new = np.linspace(x[0], x[-1], len(x) * 5) if use_parabolic else x
    y_new = np.polynomial.polynomial.polyval(x_new, z)
    return np.gradient(y_new, x_new, axis=1)[:, 0]


def cal_wss_from_surf(surf, velocity, viscosity=4.0, inward_distance=0.6,
                      parabolic_fitting=True, no_slip_condition=True):
    surf.compute_normals(point_normals=True, cell_normals=True, inplace=True, flip_normals=True)
    pc0 = pv.PolyData(surf.points).sample(velocity)
    pc1 = pv.PolyData(pc0.points + float(inward_distance) * surf.point_normals).sample(velocity)
    pc2 = pv.PolyData(pc0.points + 2.0 * float(inward_distance) * surf.point_normals).sample(velocity)

    if no_slip_condition:
        t0 = np.zeros(len(pc0.points))
    else:
        _, tang0 = get_orthogonal_vectors(extract_vectors(pc0), surf.point_normals)
        t0 = get_vector_magnitude(tang0)

    _, tang1 = get_orthogonal_vectors(extract_vectors(pc1), surf.point_normals)
    t1 = get_vector_magnitude(tang1)
    _, tang2 = get_orthogonal_vectors(extract_vectors(pc2), surf.point_normals)
    t2 = get_vector_magnitude(tang2)
    c = np.sum(tang1 * tang2, axis=1).clip(-1, 1)
    t2 = c * t2

    surf["wss"] = calculate_gradient(t0, t1, t2, inward_distance, use_parabolic=parabolic_fitting) * float(viscosity)
    surf["wss_vectors"] = tang1
    return surf


def summarize_internal_consistency(plane_metrics, path_info=None, forks=None):
    by_path = {}
    for metric in plane_metrics:
        pidx = int(metric.get("path_index", -1))
        by_path.setdefault(pidx, []).append(abs(float(metric.get("netflow_mL_beat", 0.0))))
    path_ic = {}
    by_path_mean = {}
    for pidx, values in by_path.items():
        arr = np.abs(np.asarray(values, dtype=float))
        mu = float(np.mean(arr)) if len(arr) else 0.0
        by_path_mean[pidx] = mu
        if len(arr) <= 1:
            ic = 1.0
        elif mu <= 1e-12:
            ic = 1.0 if float(np.max(arr)) <= 1e-12 else 0.0
        else:
            ic = 1.0 - float(np.mean(np.abs(arr - mu)) / mu)
        path_ic[str(int(pidx))] = float(np.clip(ic, 0.0, 1.0))
    fork_items = []
    fork_ic = {}
    for fork_id, fork in enumerate(forks or []):
        left = [int(x) for x in fork.get("left", [])]
        right = [int(x) for x in fork.get("right", [])]
        sum_left = float(np.sum([abs(by_path_mean.get(x, 0.0)) for x in left]))
        sum_right = float(np.sum([abs(by_path_mean.get(x, 0.0)) for x in right]))
        denom = sum_left + sum_right
        if denom <= 1e-12:
            ic = 1.0
        else:
            ic = 1.0 - 2.0 * abs(sum_left - sum_right) / denom
        ic = float(np.clip(ic, 0.0, 1.0))
        fork_ic[str(int(fork_id))] = ic
        item = {
            "fork_id": int(fork_id),
            "left": left,
            "right": right,
            "crosspoint": fork.get("crosspoint", [0.0, 0.0, 0.0]),
            "node": int(fork.get("node", -1)),
            "ic": ic,
        }
        if path_info is not None:
            item["left_dirs"] = [path_info[x].get("direction_text", "") for x in left if 0 <= x < len(path_info)]
            item["right_dirs"] = [path_info[x].get("direction_text", "") for x in right if 0 <= x < len(path_info)]
        fork_items.append(item)
    return {"path_ic": path_ic, "fork_ic": fork_ic, "forks": fork_items}


def apply_internal_consistency_to_metrics(plane_metrics, path_info=None, forks=None):
    metrics = [dict(metric) for metric in plane_metrics]
    qc = summarize_internal_consistency(metrics, path_info=path_info, forks=forks)
    for metric in metrics:
        pidx = str(int(metric.get("path_index", -1)))
        metric["path_ic"] = float(qc["path_ic"].get(pidx, 1.0))
        rel = []
        for fork in qc.get("forks", []):
            pid = int(metric.get("path_index", -1))
            if pid in fork.get("left", []) or pid in fork.get("right", []):
                role = "incoming" if pid in fork.get("left", []) else "outgoing"
                rel.append({"fork_id": int(fork.get("fork_id", -1)), "role": role, "ic": float(fork.get("ic", 1.0))})
        metric["fork_ic"] = rel
    return metrics, qc


def compute_plane_metrics(flow_xyzt3, segmask_binary_4d, spacing, origin, planes, RR=1000.0,
                          branch_labels_3d=None, path_info=None, forks=None, return_qc=False):
    flow = np.asarray(flow_xyzt3, dtype=float)
    mask = np.asarray(segmask_binary_4d, dtype=bool)
    spacing = np.asarray(spacing, dtype=float).reshape(-1)[:3]
    origin = np.asarray(origin, dtype=float).reshape(-1)[:3]
    if flow.ndim != 5 or flow.shape[-1] != 3:
        raise ValueError(f"flow must be XYZTV, got {flow.shape}")
    if mask.ndim == 3:
        mask = np.repeat(mask[..., np.newaxis], flow.shape[3], axis=3)
    elif mask.ndim == 4 and mask.shape[3] == 1 and flow.shape[3] > 1:
        mask = np.repeat(mask, flow.shape[3], axis=3)
    if mask.shape[3] != flow.shape[3]:
        raise ValueError(f"mask time dimension {mask.shape[3]} does not match flow {flow.shape[3]}")
    Nt = int(flow.shape[3])
    branch_grid = _build_branch_grid(branch_labels_3d, spacing, origin)
    mask_static = all(np.array_equal(mask[..., 0], mask[..., t]) for t in range(1, Nt))
    results = []

    for plane in planes:
        normal = np.asarray(plane.normal, dtype=float).reshape(3)
        normal = normal / (np.linalg.norm(normal) + 1e-12)
        target_label = None
        if branch_labels_3d is not None:
            target_label = int(getattr(plane, "label", 0) or 0)
            if target_label <= 0:
                ijk = np.rint((np.asarray(plane.center, dtype=float).reshape(3) - origin) / (spacing + 1e-12)).astype(int)
                ijk = np.clip(ijk, 0, np.array(np.asarray(branch_labels_3d).shape) - 1)
                target_label = int(np.asarray(branch_labels_3d)[ijk[0], ijk[1], ijk[2]])
        mask_template = mask[..., 0] if mask_static else None
        peakv = 0.0
        flowrate = []
        area = []
        meanv_t = []
        for t in range(Nt):
            mask_t = mask_template if mask_template is not None else mask[..., t]
            pg = _extract_plane_flow_region(
                mask_t, flow[..., t, :], plane, spacing, origin,
                branch_grid=branch_grid, target_label=target_label,
            )
            if pg is None or pg.n_cells == 0:
                flowrate.append(0.0)
                meanv_t.append(0.0)
                area.append(0.0)
                continue
            if "flow" not in pg.cell_data and "flow" in pg.point_data:
                pg = pg.point_data_to_cell_data(pass_point_data=True)
            vec = np.asarray(pg.cell_data.get("flow", []), dtype=float)
            if vec.ndim != 2 or vec.shape[1] != 3 or len(vec) != pg.n_cells:
                ca0 = np.asarray(pg.cell_data.get("Area", []), dtype=float)
                flowrate.append(0.0)
                meanv_t.append(0.0)
                area.append(float(np.sum(ca0)) if len(ca0) else 0.0)
                continue
            ca = np.asarray(pg.cell_data.get("Area", np.ones(pg.n_cells, dtype=float)), dtype=float).reshape(-1)
            if len(ca) != len(vec):
                ca = np.ones(len(vec), dtype=float)
            proj = np.dot(vec, normal)
            area_t = float(np.sum(ca)) if len(ca) else 0.0
            fr = float(np.sum(proj * ca) / 100.0) if area_t > 0.0 else 0.0
            mv = float(np.sum(proj * ca) / area_t) if area_t > 0.0 else 0.0
            peakv = max(peakv, float(np.max(np.abs(proj))) if len(proj) else 0.0)
            flowrate.append(fr)
            meanv_t.append(mv)
            area.append(area_t)
        metric = {
            "center": np.asarray(plane.center, dtype=float).tolist(),
            "normal": normal.tolist(),
            "label": int(plane.label),
            "path_index": int(plane.path_index),
            "distance": float(plane.distance),
            "target_branch_label": int(target_label) if target_label is not None else 0,
            "peakv_cm_s": float(peakv),
            "flowrate_mL_s": [float(x) for x in flowrate],
            "netflow_mL_beat": float(abs(np.mean(flowrate)) * RR / 1000.0) if len(flowrate) else 0.0,
            "meanv_cm_s": float(np.mean(meanv_t)) if len(meanv_t) else 0.0,
            "meanv_cm_s_t": [float(x) for x in meanv_t],
            "area_mm2": [float(x) for x in area],
        }
        if path_info is not None and 0 <= int(plane.path_index) < len(path_info):
            info = path_info[int(plane.path_index)]
            metric["path_direction"] = info.get("direction_text", "")
            metric["path_start_point"] = info.get("start_point", [0.0, 0.0, 0.0])
            metric["path_end_point"] = info.get("end_point", [0.0, 0.0, 0.0])
            metric["path_fork_ids"] = info.get("fork_ids", [])
            metric["path_fork_roles"] = info.get("fork_roles", [])
        results.append(metric)

    results, qc = apply_internal_consistency_to_metrics(results, path_info=path_info, forks=forks)
    if return_qc:
        return results, qc
    return results


def compute_derived_metrics(mask4d, flow, spacing, origin=(0, 0, 0),
                            smoothing_iteration=200, viscosity=4.0,
                            inward_distance=0.6, parabolic_fitting=True,
                            no_slip_condition=True, step_size=5,
                            tube_radius=0.1, rho=1060.0,
                            save_pixelwise=False, tke_array=None, sigma=None):
    mask4d = np.asarray(mask4d, dtype=bool)
    flow = np.asarray(flow, dtype=float)
    if mask4d.ndim == 3:
        mask4d = mask4d[..., np.newaxis]
    Nt = int(mask4d.shape[-1])

    if tke_array is None:
        if sigma is None:
            raise ValueError("tke_array or sigma is required")
        tke_array = 0.5 * float(rho) * np.sum((np.asarray(sigma, dtype=float) / 100.0) ** 2, axis=-1)

    tke_array = np.asarray(tke_array, dtype=np.float32)
    if tke_array.ndim == 3:
        tke_array = np.repeat(tke_array[..., None], Nt, axis=3)
    elif tke_array.ndim == 4 and tke_array.shape[3] == 1 and Nt > 1:
        tke_array = np.repeat(tke_array, Nt, axis=3)
    elif tke_array.ndim != 4:
        raise ValueError(f"tke_array must be XYZ or XYZT, got {tke_array.shape}")
    if tke_array.shape[3] != Nt:
        raise ValueError(f"tke_array time dimension {tke_array.shape[3]} does not match mask {Nt}")

    tke_array = tke_array * mask4d.astype(np.float32)
    tke_peak = np.max(tke_array, axis=3)

    TKE = create_uniform_grid(tke_peak, spacing, origin=origin, name="TKE")
    mesh_union = create_uniform_grid(np.max(mask4d > 0, axis=-1), spacing, origin=origin)
    mesh_union = mesh_union.threshold(0.1)
    TKE = mesh_union.sample(TKE)

    spacing = np.asarray(spacing, dtype=float).reshape(3)
    origin = np.asarray(origin, dtype=float).reshape(3)
    wss_volume = np.zeros(mask4d.shape, dtype=np.float32)
    surfs = []
    for showt in range(Nt):
        velocity = create_uniform_vector(
            flow[..., showt, 0] / 100.0, flow[..., showt, 1] / 100.0,
            flow[..., showt, 2] / 100.0, spacing, origin=origin)
        mesh = create_uniform_grid(mask4d[..., showt] > 0, spacing, origin=origin)
        mesh = mesh.threshold(0.1)
        surf = mesh.extract_surface().smooth(n_iter=int(smoothing_iteration))
        surf = cal_wss_from_surf(surf, velocity, viscosity=viscosity,
                                 inward_distance=inward_distance,
                                 parabolic_fitting=parabolic_fitting,
                                 no_slip_condition=no_slip_condition)
        surfs.append(surf)
        if surf.n_points > 0 and "wss" in surf.point_data:
            pts = np.asarray(surf.points, dtype=float)
            vals = np.asarray(surf.point_data["wss"], dtype=np.float32)
            vox = np.rint((pts - origin.reshape(1, 3)) / (spacing.reshape(1, 3) + 1e-12)).astype(int)
            for k in range(3):
                vox[:, k] = np.clip(vox[:, k], 0, mask4d.shape[k] - 1)
            flat = np.ravel_multi_index((vox[:, 0], vox[:, 1], vox[:, 2]), mask4d.shape[:3])
            tgt = wss_volume[..., showt].reshape(-1)
            np.maximum.at(tgt, flat, vals)

    result = {
        "wss_surfaces": surfs,
        "wss_volume": wss_volume,
        "tke_volume": TKE,
        "tke_array": tke_array,
        "tke_peak": np.asarray(tke_peak, dtype=np.float32),
        "streamlines": [],
        "tube_radius": float(tube_radius),
    }
    if save_pixelwise:
        result["pixelwise_export"] = {
            "wss": np.asarray(wss_volume, dtype=np.float32),
            "tke": np.asarray(tke_peak, dtype=np.float32),
            "tke_time": np.asarray(tke_array, dtype=np.float32),
            "spacing": np.asarray(spacing, dtype=np.float32),
            "origin": np.asarray(origin, dtype=np.float32),
        }
    else:
        result["pixelwise_export"] = {}
    return result
def _compute_single_plane_metric(args):
    flow, mask, spacing, origin, plane, Nt, RR, branch_grid, target_label, path_info = args
    normal = np.asarray(plane.normal, dtype=float).reshape(3)
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    mask_static = all(np.array_equal(mask[..., 0], mask[..., t]) for t in range(1, Nt))
    mask_template = mask[..., 0] if mask_static else None
    peakv = 0.0
    flowrate = []
    area = []
    meanv_t = []
    for t in range(Nt):
        mask_t = mask_template if mask_template is not None else mask[..., t]
        pg = _extract_plane_flow_region(
            mask_t, flow[..., t, :], plane, spacing, origin,
            branch_grid=branch_grid, target_label=target_label,
        )
        if pg is None or pg.n_cells == 0:
            flowrate.append(0.0)
            meanv_t.append(0.0)
            area.append(0.0)
            continue
        if "flow" not in pg.cell_data and "flow" in pg.point_data:
            pg = pg.point_data_to_cell_data(pass_point_data=True)
        vec = np.asarray(pg.cell_data.get("flow", []), dtype=float)
        if vec.ndim != 2 or vec.shape[1] != 3 or len(vec) != pg.n_cells:
            ca0 = np.asarray(pg.cell_data.get("Area", []), dtype=float)
            flowrate.append(0.0)
            meanv_t.append(0.0)
            area.append(float(np.sum(ca0)) if len(ca0) else 0.0)
            continue
        ca = np.asarray(pg.cell_data.get("Area", np.ones(pg.n_cells, dtype=float)), dtype=float).reshape(-1)
        if len(ca) != len(vec):
            ca = np.ones(len(vec), dtype=float)
        proj = np.dot(vec, normal)
        area_t = float(np.sum(ca)) if len(ca) else 0.0
        fr = float(np.sum(proj * ca) / 100.0) if area_t > 0.0 else 0.0
        mv = float(np.sum(proj * ca) / area_t) if area_t > 0.0 else 0.0
        peakv = max(peakv, float(np.max(np.abs(proj))) if len(proj) else 0.0)
        flowrate.append(fr)
        meanv_t.append(mv)
        area.append(area_t)
    metric = {
        "center": np.asarray(plane.center, dtype=float).tolist(),
        "normal": normal.tolist(),
        "label": int(plane.label),
        "path_index": int(plane.path_index),
        "distance": float(plane.distance),
        "target_branch_label": int(target_label) if target_label is not None else 0,
        "peakv_cm_s": float(peakv),
        "flowrate_mL_s": [float(x) for x in flowrate],
        "netflow_mL_beat": float(abs(np.mean(flowrate)) * RR / 1000.0) if len(flowrate) else 0.0,
        "meanv_cm_s": float(np.mean(meanv_t)) if len(meanv_t) else 0.0,
        "meanv_cm_s_t": [float(x) for x in meanv_t],
        "area_mm2": [float(x) for x in area],
    }
    if path_info is not None and 0 <= int(plane.path_index) < len(path_info):
        info = path_info[int(plane.path_index)]
        metric["path_direction"] = info.get("direction_text", "")
        metric["path_start_point"] = info.get("start_point", [0.0, 0.0, 0.0])
        metric["path_end_point"] = info.get("end_point", [0.0, 0.0, 0.0])
        metric["path_fork_ids"] = info.get("fork_ids", [])
        metric["path_fork_roles"] = info.get("fork_roles", [])
    return metric


def compute_plane_metrics_multithread(flow_xyzt3, segmask_binary_4d, spacing, origin, planes, RR=1000.0,
                                       branch_labels_3d=None, path_info=None, forks=None, return_qc=False,
                                       max_workers=None):
    from concurrent.futures import ThreadPoolExecutor
    flow = np.asarray(flow_xyzt3, dtype=float)
    mask = np.asarray(segmask_binary_4d, dtype=bool)
    spacing = np.asarray(spacing, dtype=float).reshape(-1)[:3]
    origin = np.asarray(origin, dtype=float).reshape(-1)[:3]
    if flow.ndim != 5 or flow.shape[-1] != 3:
        raise ValueError(f"flow must be XYZTV, got {flow.shape}")
    if mask.ndim == 3:
        mask = np.repeat(mask[..., np.newaxis], flow.shape[3], axis=3)
    elif mask.ndim == 4 and mask.shape[3] == 1 and flow.shape[3] > 1:
        mask = np.repeat(mask, flow.shape[3], axis=3)
    Nt = int(flow.shape[3])
    branch_grid = _build_branch_grid(branch_labels_3d, spacing, origin)
    args_list = []
    for plane in planes:
        target_label = None
        if branch_labels_3d is not None:
            target_label = int(getattr(plane, "label", 0) or 0)
            if target_label <= 0:
                ijk = np.rint((np.asarray(plane.center, dtype=float).reshape(3) - origin) / (spacing + 1e-12)).astype(int)
                ijk = np.clip(ijk, 0, np.array(np.asarray(branch_labels_3d).shape) - 1)
                target_label = int(np.asarray(branch_labels_3d)[ijk[0], ijk[1], ijk[2]])
        args_list.append((flow, mask, spacing, origin, plane, Nt, RR, branch_grid, target_label, path_info))
    if max_workers is None:
        import os as _os
        max_workers = min(len(planes), max(1, _os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(_compute_single_plane_metric, args_list))
    results, qc = apply_internal_consistency_to_metrics(results, path_info=path_info, forks=forks)
    if return_qc:
        return results, qc
    return results


def load_metrics_as_table(metrics_json_path, qc_json_path=None):
    import json as _json
    with open(metrics_json_path, "r", encoding="utf-8") as f:
        metrics = _json.load(f)
    scalar_keys = [
        "label", "path_index", "distance", "target_branch_label",
        "peakv_cm_s", "netflow_mL_beat", "meanv_cm_s", "path_ic",
        "path_direction",
    ]
    table_rows = []
    for i, m in enumerate(metrics):
        row = {"plane_index": i}
        for k in scalar_keys:
            if k in m:
                row[k] = m[k]
        row["center_x"] = m["center"][0] if "center" in m else None
        row["center_y"] = m["center"][1] if "center" in m else None
        row["center_z"] = m["center"][2] if "center" in m else None
        Nt = len(m.get("flowrate_mL_s", []))
        for t in range(Nt):
            row[f"flowrate_t{t}"] = m["flowrate_mL_s"][t]
        for t in range(len(m.get("area_mm2", []))):
            row[f"area_t{t}"] = m["area_mm2"][t]
        for t in range(len(m.get("meanv_cm_s_t", []))):
            row[f"meanv_t{t}"] = m["meanv_cm_s_t"][t]
        fork_ic = m.get("fork_ic", [])
        for fi, fic in enumerate(fork_ic):
            row[f"fork{fi}_id"] = fic.get("fork_id", -1)
            row[f"fork{fi}_role"] = fic.get("role", "")
            row[f"fork{fi}_ic"] = fic.get("ic", 1.0)
        table_rows.append(row)
    qc_data = None
    if qc_json_path is not None:
        with open(qc_json_path, "r", encoding="utf-8") as f:
            qc_data = _json.load(f)
    return table_rows, metrics, qc_data