import inspect

import numpy as np
import pyvista as pv


_EXTRACT_SURFACE_SUPPORTS_ALGORITHM = (
    "algorithm" in inspect.signature(pv.UnstructuredGrid.extract_surface).parameters
)


def _extract_surface(dataset):
    if _EXTRACT_SURFACE_SUPPORTS_ALGORITHM:
        return dataset.extract_surface(algorithm="dataset_surface")
    return dataset.extract_surface()


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
    surf = _extract_surface(threshed)
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
    surf = _extract_surface(th)
    if smooth_iter > 0 and surf.n_points > 0:
        surf = surf.smooth(n_iter=smooth_iter)
    return surf



def build_cell_mask_surface(mask_xyz, spacing, origin=(0, 0, 0), *, smooth_iter=80):
    mask_xyz = np.asarray(mask_xyz, dtype=bool)
    if not np.any(mask_xyz):
        return None
    mask_grid = create_uniform_grid(mask_xyz.astype(np.uint8), spacing, origin=origin, name="mask")
    support = mask_grid.threshold(0.1, scalars="mask")
    if support is None or support.n_cells == 0:
        return None

    surface = _extract_surface(support)
    if surface is None or surface.n_points == 0:
        return None
    if int(smooth_iter) > 0:
        surface = surface.triangulate().smooth(n_iter=int(smooth_iter))
    return surface


def sample_volume_on_existing_surface(field_xyz, surface, spacing, origin=(0, 0, 0), *, name="field"):
    field_xyz = np.asarray(field_xyz, dtype=np.float32)
    if field_xyz.ndim != 3:
        raise ValueError(f"{name} must be XYZ, got {field_xyz.shape}")
    if surface is None or surface.n_points == 0:
        return None

    scalar_grid = create_uniform_grid(field_xyz, spacing, origin=origin, name=name)
    scalar_grid = scalar_grid.cell_data_to_point_data(pass_cell_data=False)
    sampled = surface.sample(scalar_grid)
    if sampled is None or sampled.n_points == 0:
        return None
    return sampled


def sample_volume_on_surface(field_xyz, mask_xyz, spacing, origin=(0, 0, 0), *, name="field", smooth_iter=80):
    field_xyz = np.asarray(field_xyz, dtype=np.float32)
    mask_xyz = np.asarray(mask_xyz, dtype=bool)
    if field_xyz.ndim != 3:
        raise ValueError(f"{name} must be XYZ, got {field_xyz.shape}")
    if field_xyz.shape != mask_xyz.shape:
        raise ValueError(f"{name} shape {field_xyz.shape} does not match mask {mask_xyz.shape}")
    surface = build_cell_mask_surface(mask_xyz, spacing, origin=origin, smooth_iter=smooth_iter)
    return sample_volume_on_existing_surface(field_xyz, surface, spacing, origin=origin, name=name)


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


def _select_connected_region(poly, ref_point=None):
    if poly is None or poly.n_cells == 0:
        return None
    poly = poly.compute_cell_sizes(area=True)
    conn = poly.connectivity()
    if conn.n_cells == 0 or "RegionId" not in conn.cell_data:
        return poly
    region_ids = np.asarray(conn.cell_data["RegionId"]).copy()
    areas = np.asarray(conn.cell_data["Area"]).copy() if "Area" in conn.cell_data else np.ones(conn.n_cells, dtype=float)
    centers = None
    ref = None
    if ref_point is not None:
        ref = np.asarray(ref_point, dtype=float).reshape(1, 3)
        cc = conn.cell_centers()
        centers = np.asarray(cc.points, dtype=float).copy() if cc is not None and cc.n_points == conn.n_cells else None

    best_region = None
    best_key = None
    for rid in np.unique(region_ids):
        mask = region_ids == rid
        s = float(np.sum(areas[mask]))
        if centers is not None and ref is not None and np.any(mask):
            region_centers = centers[mask]
            d = float(np.min(np.linalg.norm(region_centers - ref, axis=1)))
            key = (d, -s, int(rid))
        else:
            key = (0.0, -s, int(rid))
        if best_key is None or key < best_key:
            best_key = key
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
    # PlaneData stores local physical coordinates.  VTK meshes include the
    # image origin, so convert the plane center at the VTK boundary.
    plane_center_world = (
        np.asarray(plane.center, dtype=float).reshape(3)
        + np.asarray(origin, dtype=float).reshape(3)
    )
    pg = mesh.slice(normal=np.asarray(plane.normal, dtype=float), origin=plane_center_world)
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
    return _select_connected_region(pg, ref_point=plane_center_world)


def _flow_grid_for_t(flow_t, spacing, origin):
    flow_t = np.asarray(flow_t, dtype=np.float32)
    grid = pv.ImageData()
    grid.dimensions = np.array(flow_t.shape[:3]) + 1
    grid.spacing = tuple(np.asarray(spacing, dtype=float).reshape(-1)[:3])
    grid.origin = tuple(np.asarray(origin, dtype=float).reshape(-1)[:3])
    grid.cell_data["flow"] = flow_t.reshape(-1, flow_t.shape[-1], order="F")
    return grid


def create_uniform_field_grid(field, spacing, origin=(0, 0, 0), name="field"):
    field = np.asarray(field)
    if field.ndim < 3:
        raise ValueError(f"{name} must have at least 3 spatial dimensions, got {field.shape}")
    mesh = pv.ImageData()
    mesh.dimensions = np.array(field.shape[:3]) + 1
    mesh.spacing = tuple(np.asarray(spacing, dtype=float).reshape(-1)[:3])
    mesh.origin = tuple(np.asarray(origin, dtype=float).reshape(-1)[:3])
    if field.ndim == 3:
        mesh.cell_data[name] = field.flatten(order="F")
    else:
        mesh.cell_data[name] = field.reshape(-1, field.shape[-1], order="F")
    return mesh


def _extract_plane_flow_region(mask_xyz, flow_t, plane, spacing, origin, branch_grid=None, target_label=None):
    mask_xyz = np.asarray(mask_xyz, dtype=bool)
    if not np.any(mask_xyz):
        return None
    grid = create_uniform_grid(mask_xyz, spacing, origin=origin, name="mask")
    grid.cell_data["flow"] = np.asarray(flow_t, dtype=np.float32).reshape(-1, 3, order="F")
    mesh = grid.threshold(0.1, scalars="mask")
    if mesh is None or mesh.n_cells == 0:
        return None
    plane_center_world = (
        np.asarray(plane.center, dtype=float).reshape(3)
        + np.asarray(origin, dtype=float).reshape(3)
    )
    pg = mesh.slice(normal=np.asarray(plane.normal, dtype=float), origin=plane_center_world)
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
    return _select_connected_region(pg, ref_point=plane_center_world)


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
