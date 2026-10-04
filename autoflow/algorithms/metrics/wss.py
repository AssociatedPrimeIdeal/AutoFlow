"""Wall shear stress from tangential inward-normal velocity derivatives."""

import numpy as np
import pyvista as pv
from ...task_control import report_progress
from ..surfaces import _extract_surface, create_uniform_grid, create_uniform_vector

from ._common import _ensure_mask4d
from .sampling import _build_mask_phase_lookup


def extract_vectors(polydata):
    return np.stack((polydata["u"], polydata["v"], polydata["w"]), axis=-1)


def get_orthogonal_vectors(vectors, point_normals):
    c = np.sum(vectors * point_normals, axis=1)
    return c[:, None] * point_normals, vectors - c[:, None] * point_normals


def get_vector_magnitude(vectors):
    return np.sqrt(np.sum(vectors * vectors, axis=1))


def align_tangential_samples(reference_vectors, target_vectors):
    reference = np.asarray(reference_vectors, dtype=float)
    target = np.asarray(target_vectors, dtype=float)
    ref_mag = get_vector_magnitude(reference)
    tgt_mag = get_vector_magnitude(target)
    denom = np.maximum(ref_mag * tgt_mag, 1e-12)
    direction = np.sum(reference * target, axis=1) / denom
    return np.clip(direction, -1.0, 1.0) * tgt_mag


def resolve_wss_inward_distance(spacing, inward_distance=None):
    spacing_mm = np.asarray(spacing, dtype=float).reshape(3)
    if not np.all(np.isfinite(spacing_mm)) or np.any(spacing_mm <= 0):
        raise ValueError("WSS spacing must be finite and positive (mm)")
    if inward_distance is None:
        inward_distance = float(np.min(spacing_mm))
    distance = float(inward_distance)
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError("WSS inward distance must be finite and positive (mm)")
    return distance


def calculate_gradient(pc0_tangent_mag, pc1_tangent_mag, pc2_tangent_mag, inward_distance, use_parabolic=True):
    """Derivative at zero of equidistant scalar or vector wall samples."""
    distance = float(inward_distance)
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError("WSS inward distance must be finite and positive (mm)")
    v0, v1, v2 = (np.asarray(v, dtype=float) for v in
                  (pc0_tangent_mag, pc1_tangent_mag, pc2_tangent_mag))
    if use_parabolic:
        return (-3.0 * v0 + 4.0 * v1 - v2) / (2.0 * distance)
    return (v1 - v0) / distance


def cal_wss_from_surf(surf, velocity, viscosity=4.0, inward_distance=0.6,
                      parabolic_fitting=True, no_slip_condition=True, support_grid=None):
    """Compute tangential WSS vectors (Pa) using inward-normal derivatives.

    Cell-centred velocity is first interpolated to points for continuous probing.
    Invalid probes or wall-normal segments leaving the lumen produce NaNs.
    """
    distance = float(inward_distance)
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError("WSS inward distance must be finite and positive (mm)")
    viscosity = float(viscosity)
    if not np.isfinite(viscosity) or viscosity < 0:
        raise ValueError("WSS viscosity must be finite and nonnegative (mPa s)")
    if not all(key in velocity.point_data for key in ("u", "v", "w")):
        velocity = velocity.cell_data_to_point_data(pass_cell_data=False)
    surf.compute_normals(point_normals=True, cell_normals=True, inplace=True,
                         consistent_normals=True, auto_orient_normals=(surf.n_open_edges == 0),
                         flip_normals=True)
    normals = np.asarray(surf.point_normals, dtype=float)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    pc0 = pv.PolyData(surf.points).sample(velocity)
    pc1 = pv.PolyData(pc0.points + distance * normals).sample(velocity)
    pc2 = pv.PolyData(pc0.points + 2.0 * distance * normals).sample(velocity)

    if no_slip_condition:
        tang0 = np.zeros((len(pc0.points), 3), dtype=float)
    else:
        _, tang0 = get_orthogonal_vectors(extract_vectors(pc0), normals)

    _, tang1 = get_orthogonal_vectors(extract_vectors(pc1), normals)
    _, tang2 = get_orthogonal_vectors(extract_vectors(pc2), normals)
    required = [pc1] + ([pc2] if parabolic_fitting else [])
    if not no_slip_condition:
        required.append(pc0)
    valid = np.all(np.isfinite(normals), axis=1) & (np.linalg.norm(normals, axis=1) > 0.5)
    for probe in required:
        valid &= np.asarray(probe["vtkValidPointMask"], dtype=bool)
        valid &= np.all(np.isfinite(extract_vectors(probe)), axis=1)
    if support_grid is not None:
        # Check intermediate locations too: endpoints alone can miss a crossing
        # through background into another branch or the opposite vessel wall.
        fractions = (0.25, 0.5, 1.0, 1.5, 2.0) if parabolic_fitting else (0.25, 0.5, 1.0)
        for fraction in fractions:
            probe = pv.PolyData(pc0.points + fraction * distance * normals).sample(support_grid)
            valid &= np.asarray(probe["vtkValidPointMask"], dtype=bool)
            valid &= np.asarray(probe["wss_lumen"], dtype=float) > 0.5
    # mPa s * (m/s)/mm is numerically Pa; keep signed vector components.
    vectors = calculate_gradient(tang0, tang1, tang2, distance,
                                 use_parabolic=parabolic_fitting) * viscosity
    vectors[~valid] = np.nan
    surf["wss_vectors"] = vectors
    surf["wss"] = get_vector_magnitude(vectors)
    surf["wss_valid"] = valid.astype(np.uint8)
    return surf


def compute_wss_metrics(mask4d, flow, spacing, origin=(0, 0, 0),
                        smoothing_iteration=200, viscosity=4.0,
                        inward_distance=None, parabolic_fitting=True,
                        no_slip_condition=True):
    mask4d = _ensure_mask4d(mask4d)
    flow = np.asarray(flow)
    if flow.ndim != 5 or flow.shape[-1] != 3:
        raise ValueError(f"flow must be XYZTV, got {flow.shape}")
    if flow.shape[:4] != mask4d.shape:
        raise ValueError(f"flow shape {flow.shape[:4]} does not match mask {mask4d.shape}")

    spacing = np.asarray(spacing, dtype=float).reshape(3)
    inward_distance = resolve_wss_inward_distance(spacing, inward_distance)
    origin = np.asarray(origin, dtype=float).reshape(3)
    wss_volume = np.zeros(mask4d.shape, dtype=np.float32)
    mask_phase_lookup = _build_mask_phase_lookup(mask4d)
    surface_cache = {}
    support_cache = {}
    surfs = []
    for showt in range(int(mask4d.shape[-1])):
        report_progress({"stage": "wss", "current": showt, "total": mask4d.shape[3],
                         "message": f"Computing WSS phase {showt + 1}/{mask4d.shape[3]}"})
        rep_t = int(mask_phase_lookup[showt])
        if rep_t not in surface_cache:
            support_cache[rep_t] = create_uniform_grid(
                (mask4d[..., rep_t] > 0).astype(np.uint8), spacing, origin=origin, name="wss_lumen")
            mesh = support_cache[rep_t]
            mesh = mesh.threshold(0.1)
            if mesh is None or mesh.n_cells == 0:
                surface_cache[rep_t] = None
                surfs.append(None)
                continue
            base_surface = _extract_surface(mesh)
            if int(smoothing_iteration) > 0:
                base_surface = base_surface.smooth_taubin(n_iter=int(smoothing_iteration), pass_band=0.1)
            surface_cache[rep_t] = base_surface
        base_surface = surface_cache[rep_t]
        if base_surface is None:
            surfs.append(None)
            continue
        # Preserve double-precision sampling without duplicating the whole
        # multi-phase velocity field just to process one phase at a time.
        flow_t = np.asarray(flow[..., showt, :], dtype=float)
        velocity = create_uniform_vector(
            flow_t[..., 0] / 100.0, flow_t[..., 1] / 100.0,
            flow_t[..., 2] / 100.0, spacing, origin=origin)
        # cal_wss_from_surf writes phase-specific point data, so each phase
        # gets a cheap geometry copy while the expensive surface preparation
        # remains shared for identical masks.
        surf = base_surface.copy(deep=True)
        surf = cal_wss_from_surf(surf, velocity, viscosity=viscosity,
                                 inward_distance=inward_distance,
                                  parabolic_fitting=parabolic_fitting,
                                  no_slip_condition=no_slip_condition,
                                  support_grid=support_cache[rep_t])
        surfs.append(surf)
        if surf.n_points > 0 and "wss" in surf.point_data:
            pts = np.asarray(surf.points, dtype=float)
            vals = np.asarray(surf.point_data["wss"], dtype=np.float32)
            vox = np.rint((pts - origin.reshape(1, 3)) / (spacing.reshape(1, 3) + 1e-12)).astype(int)
            for k in range(3):
                vox[:, k] = np.clip(vox[:, k], 0, mask4d.shape[k] - 1)
            flat = np.ravel_multi_index((vox[:, 0], vox[:, 1], vox[:, 2]), mask4d.shape[:3])
            tgt = wss_volume[..., showt].reshape(-1)
            valid = np.isfinite(vals)
            tgt[flat[~valid]] = np.nan
            np.fmax.at(tgt, flat[valid], vals[valid])

    return {
        "wss_surfaces": surfs,
        "wss_volume": wss_volume,
    }
