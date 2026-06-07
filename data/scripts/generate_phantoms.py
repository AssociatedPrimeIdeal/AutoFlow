"""Generate the legacy complex S/U/Y phantoms used by AutoFlow."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
TRUTH_GROUP_NAME = "truth"


def make_straight_tube_centerline(length_mm=80.0, axis="z", n_points=400):
    s = np.linspace(-0.5 * length_mm, 0.5 * length_mm, n_points, dtype=np.float32)
    zeros = np.zeros_like(s)
    if axis == "z":
        centerline = np.stack([zeros, zeros, s], axis=1)
    elif axis == "y":
        centerline = np.stack([zeros, s, zeros], axis=1)
    elif axis == "x":
        centerline = np.stack([s, zeros, zeros], axis=1)
    else:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis}")
    return centerline.astype(np.float32)


def make_u_tube_centerline(
    leg_spacing_mm=30.0,
    leg_height_mm=40.0,
    y_center_mm=0.0,
    n_leg=100,
    n_arc=160,
):
    bend_radius_mm = leg_spacing_mm / 2.0
    x_left = -bend_radius_mm
    x_right = bend_radius_mm
    z_join = 0.0
    z_top = leg_height_mm

    z1 = np.linspace(z_top, z_join, n_leg, endpoint=False, dtype=np.float32)
    x1 = np.full_like(z1, x_left)
    y1 = np.full_like(z1, y_center_mm)

    theta = np.linspace(np.pi, 2.0 * np.pi, n_arc, endpoint=False, dtype=np.float32)
    x2 = bend_radius_mm * np.cos(theta)
    z2 = bend_radius_mm * np.sin(theta) + z_join
    y2 = np.full_like(x2, y_center_mm)

    z3 = np.linspace(z_join, z_top, n_leg, dtype=np.float32)
    x3 = np.full_like(z3, x_right)
    y3 = np.full_like(z3, y_center_mm)

    return np.stack(
        [
            np.concatenate([x1, x2, x3]),
            np.concatenate([y1, y2, y3]),
            np.concatenate([z1, z2, z3]),
        ],
        axis=1,
    ).astype(np.float32)


def make_y_tube_centerlines(
    stem_height_mm=40.0,
    branch_length_mm=35.0,
    branch_angle_deg=45.0,
    y_center_mm=0.0,
    n_stem=120,
    n_branch=100,
):
    z_bif = 0.0
    z_bottom = -stem_height_mm
    ang = np.deg2rad(branch_angle_deg).astype(np.float32)

    z0 = np.linspace(z_bottom, z_bif, n_stem, dtype=np.float32)
    x0 = np.zeros_like(z0)
    y0 = np.full_like(z0, y_center_mm)
    stem = np.stack([x0, y0, z0], axis=1).astype(np.float32)

    s = np.linspace(0.0, branch_length_mm, n_branch, dtype=np.float32)

    xl = -s * np.sin(ang)
    zl = z_bif + s * np.cos(ang)
    yl = np.full_like(xl, y_center_mm)
    left = np.stack([xl, yl, zl], axis=1).astype(np.float32)

    xr = s * np.sin(ang)
    zr = z_bif + s * np.cos(ang)
    yr = np.full_like(xr, y_center_mm)
    right = np.stack([xr, yr, zr], axis=1).astype(np.float32)

    return [stem, left, right]


def closest_polyline_distance_and_tangent(x_grid, y_grid, z_grid, polyline):
    best_d2 = np.full(x_grid.shape, np.inf, dtype=np.float32)
    best_tangent = np.zeros(x_grid.shape + (3,), dtype=np.float32)

    for idx in range(len(polyline) - 1):
        p0 = polyline[idx].astype(np.float32)
        p1 = polyline[idx + 1].astype(np.float32)
        vec = p1 - p0
        vv = float(np.dot(vec, vec))
        if vv < 1e-8:
            continue

        proj = ((x_grid - p0[0]) * vec[0] + (y_grid - p0[1]) * vec[1] + (z_grid - p0[2]) * vec[2]) / vv
        proj = np.clip(proj, 0.0, 1.0)

        cx = p0[0] + proj * vec[0]
        cy = p0[1] + proj * vec[1]
        cz = p0[2] + proj * vec[2]

        d2 = (x_grid - cx) ** 2 + (y_grid - cy) ** 2 + (z_grid - cz) ** 2
        update = d2 < best_d2
        if np.any(update):
            best_d2[update] = d2[update]
            tangent = vec / np.sqrt(vv)
            best_tangent[..., 0][update] = tangent[0]
            best_tangent[..., 1][update] = tangent[1]
            best_tangent[..., 2][update] = tangent[2]

    return best_d2, best_tangent


def closest_multiline_distance_and_tangent(x_grid, y_grid, z_grid, polylines):
    best_d2 = np.full(x_grid.shape, np.inf, dtype=np.float32)
    best_tangent = np.zeros(x_grid.shape + (3,), dtype=np.float32)
    best_line_id = np.full(x_grid.shape, -1, dtype=np.int16)

    for line_id, polyline in enumerate(polylines):
        for idx in range(len(polyline) - 1):
            p0 = polyline[idx].astype(np.float32)
            p1 = polyline[idx + 1].astype(np.float32)
            vec = p1 - p0
            vv = float(np.dot(vec, vec))
            if vv < 1e-8:
                continue

            proj = ((x_grid - p0[0]) * vec[0] + (y_grid - p0[1]) * vec[1] + (z_grid - p0[2]) * vec[2]) / vv
            proj = np.clip(proj, 0.0, 1.0)

            cx = p0[0] + proj * vec[0]
            cy = p0[1] + proj * vec[1]
            cz = p0[2] + proj * vec[2]

            d2 = (x_grid - cx) ** 2 + (y_grid - cy) ** 2 + (z_grid - cz) ** 2
            update = d2 < best_d2
            if np.any(update):
                best_d2[update] = d2[update]
                tangent = vec / np.sqrt(vv)
                best_tangent[..., 0][update] = tangent[0]
                best_tangent[..., 1][update] = tangent[1]
                best_tangent[..., 2][update] = tangent[2]
                best_line_id[update] = line_id

    return best_d2, best_tangent, best_line_id


def generate_periodic_waveform(nt, base=0.15, amp=0.85):
    if nt <= 1:
        return np.array([1.0], dtype=np.float32)
    phase = np.arange(nt, dtype=np.float32) / float(nt)
    return (base + amp * (np.sin(np.pi * phase) ** 2)).astype(np.float32)


def centered_origin(nx, ny, nz, resolution):
    dx_mm, dy_mm, dz_mm = map(float, resolution)
    return np.array(
        [
            -0.5 * (nx - 1) * dx_mm,
            -0.5 * (ny - 1) * dy_mm,
            -0.5 * (nz - 1) * dz_mm,
        ],
        dtype=np.float32,
    )


def build_grid(nx, ny, nz, resolution, origin=None):
    if origin is None:
        origin = centered_origin(nx, ny, nz, resolution)
    else:
        origin = np.asarray(origin, dtype=np.float32)

    x = origin[0] + np.arange(nx, dtype=np.float32) * float(resolution[0])
    y = origin[1] + np.arange(ny, dtype=np.float32) * float(resolution[1])
    z = origin[2] + np.arange(nz, dtype=np.float32) * float(resolution[2])
    x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing="ij")
    return origin, x_grid, y_grid, z_grid


def serialize_centerlines(centerlines_mm):
    counts = np.asarray([len(np.asarray(line)) for line in centerlines_mm], dtype=np.int32)
    max_points = int(counts.max()) if counts.size else 0
    points = np.zeros((len(centerlines_mm), max_points, 3), dtype=np.float32)
    for idx, line in enumerate(centerlines_mm):
        arr = np.asarray(line, dtype=np.float32)
        points[idx, : arr.shape[0], :] = arr
    return points, counts


def build_truth_payload(
    centerlines_mm,
    vel_cm_s,
    mag_xyzt,
    segmask_xyzt,
    resolution,
    origin_mm,
    venc_cm_s,
    rr_ms,
    spatial_order,
    venc_order,
    flow_rate_ml_s,
    flow_ml_per_beat,
    path_scale_values,
    path_id_xyz,
    radial_profile_xyz,
    extra_fields=None,
):
    centerline_points_mm, centerline_point_counts = serialize_centerlines(centerlines_mm)
    path_scale_values = np.asarray(path_scale_values, dtype=np.float32)
    flow_rate_ml_s = np.asarray(flow_rate_ml_s, dtype=np.float32)
    rr_ms = float(rr_ms)
    nt = int(mag_xyzt.shape[3])
    time_ms = (np.arange(nt, dtype=np.float32) / float(nt) * rr_ms).astype(np.float32)
    time_s = time_ms / 1000.0

    truth = {
        "flow_xyzt3_cm_s": np.asarray(vel_cm_s, dtype=np.float32),
        "flow_xyzt3_m_s": np.asarray(vel_cm_s, dtype=np.float32) / 100.0,
        "mag_xyzt": np.asarray(mag_xyzt, dtype=np.float32),
        "segmentation_xyzt": np.asarray(segmask_xyzt, dtype=np.int16),
        "tube_mask_xyz": np.asarray(segmask_xyzt[..., 0], dtype=np.int16),
        "resolution_mm": np.asarray(resolution, dtype=np.float32),
        "origin_mm": np.asarray(origin_mm, dtype=np.float32),
        "venc_cm_s": np.asarray(venc_cm_s, dtype=np.float32),
        "rr_ms": np.asarray(rr_ms, dtype=np.float32),
        "time_ms": time_ms,
        "time_s": time_s,
        "flow_rate_ml_s": flow_rate_ml_s,
        "flow_ml_per_beat": np.asarray(flow_ml_per_beat, dtype=np.float32),
        "centerline_paths_mm": centerline_points_mm,
        "centerline_path_point_counts": centerline_point_counts,
        "path_scale_values": path_scale_values,
        "path_flow_rate_ml_s": path_scale_values[:, None] * flow_rate_ml_s[None, :],
        "path_flow_ml_per_beat": path_scale_values * float(flow_ml_per_beat),
        "spatial_order": np.asarray(spatial_order, dtype="S8"),
        "venc_order": np.asarray(venc_order, dtype="S8"),
        "path_id_xyz": np.asarray(path_id_xyz, dtype=np.int16),
        "radial_profile_xyz": np.asarray(radial_profile_xyz, dtype=np.float32),
    }
    if extra_fields:
        for key, value in extra_fields.items():
            truth[key] = np.asarray(value)
    return truth


def save_truth_h5(group: h5py.Group, truth: dict) -> None:
    if TRUTH_GROUP_NAME in group:
        del group[TRUTH_GROUP_NAME]
    truth_group = group.create_group(TRUTH_GROUP_NAME)
    for key, value in truth.items():
        truth_group.create_dataset(key, data=np.asarray(value))


def write_legacy_complex_h5(
    out_path,
    vel_cm_s,
    mag,
    segmask_xyzt,
    resolution,
    rr_ms,
    spatial_order,
    venc_order,
    flow_rate_ml_s,
    flow_ml_per_beat,
    truth=None,
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    venc = np.ceil(np.full(3, np.max(np.abs(vel_cm_s)), dtype=np.float32)).astype(np.float32)
    phase = np.pi * vel_cm_s / venc.reshape(1, 1, 1, 1, 3)
    phase = np.clip(phase, -np.pi, np.pi).astype(np.float32)

    img_complex = np.empty(mag.shape + (4,), dtype=np.complex64)
    img_complex[..., 0] = mag.astype(np.complex64)
    img_complex[..., 1:4] = (mag[..., None] * np.exp(1j * phase)).astype(np.complex64)

    time_ms = (np.arange(mag.shape[3], dtype=np.float32) / float(mag.shape[3]) * rr_ms).astype(np.float32)

    with h5py.File(out_path, "w") as handle:
        handle.create_dataset("img_complex", data=img_complex, compression="gzip")
        handle.create_dataset("segmask", data=np.asarray(segmask_xyzt, dtype=np.int16), compression="gzip")
        handle.create_dataset("VENC", data=venc)
        handle.create_dataset("Resolution", data=np.asarray(resolution, dtype=np.float32))
        handle.create_dataset("Origin", data=np.asarray([0.0, 0.0, 0.0], dtype=np.float32))
        handle.create_dataset("RR", data=np.asarray(rr_ms, dtype=np.float32))
        handle.create_dataset("SpatialOrder", data=np.asarray(spatial_order, dtype="S8"))
        handle.create_dataset("VENCOrder", data=np.asarray(venc_order, dtype="S8"))
        handle.create_dataset("flow_rate_ml_s", data=np.asarray(flow_rate_ml_s, dtype=np.float32))
        handle.create_dataset("flow_ml_per_beat", data=np.asarray(flow_ml_per_beat, dtype=np.float32))
        handle.create_dataset("time_ms", data=time_ms)
        if truth is not None:
            save_truth_h5(handle, truth)

    return {
        "out_path": str(out_path),
        "img_complex_shape": img_complex.shape,
        "segmask_shape": np.asarray(segmask_xyzt).shape,
        "resolution_mm": np.asarray(resolution, dtype=np.float32),
        "venc_cm_s": venc,
        "rr_ms": float(rr_ms),
        "flow_ml_per_beat": float(flow_ml_per_beat),
    }


def generate_straight_tube_phantom_h5(
    out_path=DATA_DIR / "phantom_S.h5",
    nx=95,
    ny=63,
    nz=95,
    nt=20,
    resolution=(1.0, 1.0, 1.0),
    origin=None,
    rr_ms=900.0,
    spatial_order=("LR", "AP", "FH"),
    venc_order=("LR", "AP", "FH"),
    tube_radius_mm=5.0,
    tube_length_mm=80.0,
    tube_axis="z",
    peak_centerline_cm_s=80.0,
    background_mag=25.0,
    vessel_mag=180.0,
):
    origin_mm, x_grid, y_grid, z_grid = build_grid(nx, ny, nz, resolution, origin=origin)
    centerline = make_straight_tube_centerline(length_mm=tube_length_mm, axis=tube_axis, n_points=400)
    d2_mm2, tangent = closest_polyline_distance_and_tangent(x_grid, y_grid, z_grid, centerline)
    d_mm = np.sqrt(d2_mm2)
    tube_mask_3d = d_mm <= tube_radius_mm
    radial_profile = np.clip(1.0 - (d_mm ** 2) / float(tube_radius_mm ** 2), 0.0, 1.0).astype(np.float32)

    waveform = generate_periodic_waveform(nt)
    u_max_peak_m_s = peak_centerline_cm_s * 1e-2
    u_max_t = u_max_peak_m_s * waveform
    u_mean_t = 0.5 * u_max_t

    speed_m_s = radial_profile[..., None] * u_max_t[None, None, None, :]
    vel_m_s = tangent[..., None, :] * speed_m_s[..., None]
    vel_cm_s = (vel_m_s.astype(np.float32) * 100.0).astype(np.float32)

    mag = np.full((nx, ny, nz, nt), background_mag, dtype=np.float32)
    for tidx in range(nt):
        mag_t = np.full((nx, ny, nz), background_mag, dtype=np.float32)
        mag_t[tube_mask_3d] = vessel_mag - 0.15 * speed_m_s[..., tidx][tube_mask_3d] * 100.0
        mag[..., tidx] = np.clip(mag_t, 0.0, None)

    segmask_xyzt = np.repeat(tube_mask_3d[..., None], nt, axis=3).astype(np.int16)
    flow_rate_m3_s = np.pi * (tube_radius_mm * 1e-3) ** 2 * u_mean_t
    flow_rate_ml_s = (flow_rate_m3_s * 1e6).astype(np.float32)
    flow_ml_per_beat = np.float32(flow_rate_ml_s.mean() * (rr_ms * 1e-3))
    path_id_xyz = np.where(tube_mask_3d, 0, -1).astype(np.int16)
    truth = build_truth_payload(
        centerlines_mm=[centerline],
        vel_cm_s=vel_cm_s,
        mag_xyzt=mag,
        segmask_xyzt=segmask_xyzt,
        resolution=resolution,
        origin_mm=origin_mm,
        venc_cm_s=np.ceil(np.full(3, np.max(np.abs(vel_cm_s)), dtype=np.float32)).astype(np.float32),
        rr_ms=rr_ms,
        spatial_order=spatial_order,
        venc_order=venc_order,
        flow_rate_ml_s=flow_rate_ml_s,
        flow_ml_per_beat=flow_ml_per_beat,
        path_scale_values=[1.0],
        path_id_xyz=path_id_xyz,
        radial_profile_xyz=radial_profile,
        extra_fields={
            "tube_radius_mm": np.asarray(tube_radius_mm, dtype=np.float32),
            "tube_length_mm": np.asarray(tube_length_mm, dtype=np.float32),
            "peak_centerline_cm_s": np.asarray(peak_centerline_cm_s, dtype=np.float32),
        },
    )

    info = write_legacy_complex_h5(
        out_path=out_path,
        vel_cm_s=vel_cm_s,
        mag=mag,
        segmask_xyzt=segmask_xyzt,
        resolution=resolution,
        rr_ms=rr_ms,
        spatial_order=spatial_order,
        venc_order=venc_order,
        flow_rate_ml_s=flow_rate_ml_s,
        flow_ml_per_beat=flow_ml_per_beat,
        truth=truth,
    )
    info["origin_mm"] = origin_mm
    info["tube_length_m"] = float(tube_length_mm * 1e-3)
    return info


def generate_u_tube_phantom_h5(
    out_path=DATA_DIR / "phantom_U.h5",
    nx=95,
    ny=63,
    nz=95,
    nt=20,
    resolution=(1.0, 1.0, 1.0),
    origin=None,
    rr_ms=900.0,
    spatial_order=("LR", "AP", "FH"),
    venc_order=("LR", "AP", "FH"),
    tube_radius_mm=5.0,
    leg_spacing_mm=30.0,
    leg_height_mm=40.0,
    peak_centerline_cm_s=80.0,
    background_mag=25.0,
    vessel_mag=180.0,
):
    origin_mm, x_grid, y_grid, z_grid = build_grid(nx, ny, nz, resolution, origin=origin)
    centerline = make_u_tube_centerline(
        leg_spacing_mm=leg_spacing_mm,
        leg_height_mm=leg_height_mm,
        y_center_mm=0.0,
        n_leg=100,
        n_arc=160,
    )
    d2_mm2, tangent = closest_polyline_distance_and_tangent(x_grid, y_grid, z_grid, centerline)
    d_mm = np.sqrt(d2_mm2)
    tube_mask_3d = d_mm <= tube_radius_mm
    radial_profile = np.clip(1.0 - (d_mm ** 2) / float(tube_radius_mm ** 2), 0.0, 1.0).astype(np.float32)

    waveform = generate_periodic_waveform(nt)
    u_max_peak_m_s = peak_centerline_cm_s * 1e-2
    u_max_t = u_max_peak_m_s * waveform
    u_mean_t = 0.5 * u_max_t

    speed_m_s = radial_profile[..., None] * u_max_t[None, None, None, :]
    vel_m_s = tangent[..., None, :] * speed_m_s[..., None]
    vel_cm_s = (vel_m_s.astype(np.float32) * 100.0).astype(np.float32)

    mag = np.full((nx, ny, nz, nt), background_mag, dtype=np.float32)
    for tidx in range(nt):
        mag_t = np.full((nx, ny, nz), background_mag, dtype=np.float32)
        mag_t[tube_mask_3d] = vessel_mag - 0.15 * speed_m_s[..., tidx][tube_mask_3d] * 100.0
        mag[..., tidx] = np.clip(mag_t, 0.0, None)

    segmask_xyzt = np.repeat(tube_mask_3d[..., None], nt, axis=3).astype(np.int16)
    flow_rate_m3_s = np.pi * (tube_radius_mm * 1e-3) ** 2 * u_mean_t
    flow_rate_ml_s = (flow_rate_m3_s * 1e6).astype(np.float32)
    flow_ml_per_beat = np.float32(flow_rate_ml_s.mean() * (rr_ms * 1e-3))
    path_id_xyz = np.where(tube_mask_3d, 0, -1).astype(np.int16)
    truth = build_truth_payload(
        centerlines_mm=[centerline],
        vel_cm_s=vel_cm_s,
        mag_xyzt=mag,
        segmask_xyzt=segmask_xyzt,
        resolution=resolution,
        origin_mm=origin_mm,
        venc_cm_s=np.ceil(np.full(3, np.max(np.abs(vel_cm_s)), dtype=np.float32)).astype(np.float32),
        rr_ms=rr_ms,
        spatial_order=spatial_order,
        venc_order=venc_order,
        flow_rate_ml_s=flow_rate_ml_s,
        flow_ml_per_beat=flow_ml_per_beat,
        path_scale_values=[1.0],
        path_id_xyz=path_id_xyz,
        radial_profile_xyz=radial_profile,
        extra_fields={
            "tube_radius_mm": np.asarray(tube_radius_mm, dtype=np.float32),
            "leg_spacing_mm": np.asarray(leg_spacing_mm, dtype=np.float32),
            "leg_height_mm": np.asarray(leg_height_mm, dtype=np.float32),
            "peak_centerline_cm_s": np.asarray(peak_centerline_cm_s, dtype=np.float32),
        },
    )

    info = write_legacy_complex_h5(
        out_path=out_path,
        vel_cm_s=vel_cm_s,
        mag=mag,
        segmask_xyzt=segmask_xyzt,
        resolution=resolution,
        rr_ms=rr_ms,
        spatial_order=spatial_order,
        venc_order=venc_order,
        flow_rate_ml_s=flow_rate_ml_s,
        flow_ml_per_beat=flow_ml_per_beat,
        truth=truth,
    )
    info["origin_mm"] = origin_mm
    info["tube_length_m"] = float((2.0 * leg_height_mm + 0.5 * np.pi * leg_spacing_mm) * 1e-3)
    return info


def generate_y_tube_phantom_h5(
    out_path=DATA_DIR / "phantom_Y.h5",
    nx=95,
    ny=63,
    nz=95,
    nt=20,
    resolution=(1.0, 1.0, 1.0),
    origin=None,
    rr_ms=900.0,
    spatial_order=("FH", "AP", "LR"),
    venc_order=("FH", "AP", "LR"),
    tube_radius_mm=5.0,
    stem_height_mm=37.0,
    branch_length_mm=35.0,
    branch_angle_deg=45.0,
    peak_centerline_cm_s=80.0,
    background_mag=25.0,
    vessel_mag=180.0,
):
    origin_mm, x_grid, y_grid, z_grid = build_grid(nx, ny, nz, resolution, origin=origin)
    centerlines = make_y_tube_centerlines(
        stem_height_mm=stem_height_mm,
        branch_length_mm=branch_length_mm,
        branch_angle_deg=branch_angle_deg,
        y_center_mm=0.0,
        n_stem=120,
        n_branch=100,
    )
    d2_mm2, tangent, line_id = closest_multiline_distance_and_tangent(x_grid, y_grid, z_grid, centerlines)
    d_mm = np.sqrt(d2_mm2)
    tube_mask_3d = d_mm <= tube_radius_mm
    radial_profile = np.clip(1.0 - (d_mm ** 2) / float(tube_radius_mm ** 2), 0.0, 1.0).astype(np.float32)

    waveform = generate_periodic_waveform(nt)
    u_max_peak_m_s = peak_centerline_cm_s * 1e-2
    u_max_t = u_max_peak_m_s * waveform
    u_mean_t = 0.5 * u_max_t

    branch_scale = np.ones(x_grid.shape, dtype=np.float32)
    branch_scale[line_id == 1] = 0.5
    branch_scale[line_id == 2] = 0.5

    speed_m_s = radial_profile[..., None] * u_max_t[None, None, None, :] * branch_scale[..., None]
    vel_m_s = tangent[..., None, :] * speed_m_s[..., None]
    vel_cm_s = (vel_m_s.astype(np.float32) * 100.0).astype(np.float32)

    mag = np.full((nx, ny, nz, nt), background_mag, dtype=np.float32)
    for tidx in range(nt):
        mag_t = np.full((nx, ny, nz), background_mag, dtype=np.float32)
        mag_t[tube_mask_3d] = vessel_mag - 0.15 * speed_m_s[..., tidx][tube_mask_3d] * 100.0
        mag[..., tidx] = np.clip(mag_t, 0.0, None)

    segmask_xyzt = np.repeat(tube_mask_3d[..., None], nt, axis=3).astype(np.int16)
    flow_rate_m3_s = np.pi * (tube_radius_mm * 1e-3) ** 2 * u_mean_t
    flow_rate_ml_s = (flow_rate_m3_s * 1e6).astype(np.float32)
    flow_ml_per_beat = np.float32(flow_rate_ml_s.mean() * (rr_ms * 1e-3))
    path_id_xyz = np.where(tube_mask_3d, line_id, -1).astype(np.int16)
    truth = build_truth_payload(
        centerlines_mm=centerlines,
        vel_cm_s=vel_cm_s,
        mag_xyzt=mag,
        segmask_xyzt=segmask_xyzt,
        resolution=resolution,
        origin_mm=origin_mm,
        venc_cm_s=np.ceil(np.full(3, np.max(np.abs(vel_cm_s)), dtype=np.float32)).astype(np.float32),
        rr_ms=rr_ms,
        spatial_order=spatial_order,
        venc_order=venc_order,
        flow_rate_ml_s=flow_rate_ml_s,
        flow_ml_per_beat=flow_ml_per_beat,
        path_scale_values=[1.0, 0.5, 0.5],
        path_id_xyz=path_id_xyz,
        radial_profile_xyz=radial_profile,
        extra_fields={
            "tube_radius_mm": np.asarray(tube_radius_mm, dtype=np.float32),
            "stem_height_mm": np.asarray(stem_height_mm, dtype=np.float32),
            "branch_length_mm": np.asarray(branch_length_mm, dtype=np.float32),
            "branch_angle_deg": np.asarray(branch_angle_deg, dtype=np.float32),
            "peak_centerline_cm_s": np.asarray(peak_centerline_cm_s, dtype=np.float32),
        },
    )

    info = write_legacy_complex_h5(
        out_path=out_path,
        vel_cm_s=vel_cm_s,
        mag=mag,
        segmask_xyzt=segmask_xyzt,
        resolution=resolution,
        rr_ms=rr_ms,
        spatial_order=spatial_order,
        venc_order=venc_order,
        flow_rate_ml_s=flow_rate_ml_s,
        flow_ml_per_beat=flow_ml_per_beat,
        truth=truth,
    )
    info["origin_mm"] = origin_mm
    info["tube_length_m"] = float((stem_height_mm + 2.0 * branch_length_mm) * 1e-3)
    return info


PHANTOM_BUILDERS = {
    "S": generate_straight_tube_phantom_h5,
    "U": generate_u_tube_phantom_h5,
    "Y": generate_y_tube_phantom_h5,
}


def normalize_names(names):
    requested = list(names) if names else ["all"]
    if "all" in requested:
        return ["S", "U", "Y"]
    return list(dict.fromkeys(requested))


def generate_selected(names):
    infos = {}
    for name in normalize_names(names):
        infos[name] = PHANTOM_BUILDERS[name]()
    return infos


def summary_lines(name, info):
    yield "=" * 60
    yield f"Phantom {name} Summary"
    yield "=" * 60
    yield f"output:           {info['out_path']}"
    yield f"resolution:       {info['resolution_mm'].tolist()} mm"
    yield f"VENC:             {info['venc_cm_s'].tolist()} cm/s"
    yield f"RR interval:      {info['rr_ms']:.1f} ms"
    yield f"Tube length:      {info['tube_length_m'] * 100.0:.1f} cm"
    yield f"Flow per beat:    {info['flow_ml_per_beat']:.2f} mL/beat"
    yield f"Truth group:      /{TRUTH_GROUP_NAME}"


def main():
    parser = argparse.ArgumentParser(description="Regenerate AutoFlow phantoms S/U/Y.")
    parser.add_argument(
        "names",
        nargs="*",
        choices=["S", "U", "Y", "all"],
        default=["all"],
        help="Which phantom(s) to generate. Default: all.",
    )
    args = parser.parse_args()

    selected = normalize_names(args.names)
    infos = generate_selected(selected)
    for name in selected:
        for line in summary_lines(name, infos[name]):
            print(line)


if __name__ == "__main__":
    main()
