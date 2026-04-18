import sys
import os
import glob
import json
import traceback
import copy

import numpy as np
import pyvista as pv
import imageio.v2 as imageio
from PIL import Image

from models import Workspace, StepId, PlaneData
from pipeline import PipelineEngine
from algorithms import (
    load_metrics_as_table,
    compute_derived_metrics,
    create_uniform_grid,
    generate_seed_points,
    generate_streamlines_at_t,
)

WINDOW_SIZE = (1600, 1200)


def load_metrics_from_output(out_dir):
    metrics_path = os.path.join(out_dir, "plane_metrics.json")
    qc_path = os.path.join(out_dir, "plane_qc.json")
    if not os.path.isfile(metrics_path):
        return None, None, None
    qc_p = qc_path if os.path.isfile(qc_path) else None
    table_rows, raw_metrics, qc_data = load_metrics_as_table(metrics_path, qc_p)
    return table_rows, raw_metrics, qc_data


def print_metrics_summary(table_rows):
    if not table_rows:
        print("  No metrics to summarize.")
        return
    print(f"  {'Plane':>6} {'Path':>5} {'Net Flow(mL/beat)':>18} {'Peak Velocity(cm/s)':>20} {'Mean Velocity(cm/s)':>20} {'IC':>6}")
    print(f"  {'-'*6} {'-'*5}  {'-'*18} {'-'*20} {'-'*20} {'-'*6}")
    for row in table_rows:
        pidx = row.get("plane_index", "?")
        path = row.get("path_index", "?")
        nf = row.get("netflow_mL_beat", 0.0)
        pv_ = row.get("peakv_cm_s", 0.0)
        mv = row.get("meanv_cm_s", 0.0)
        ic = row.get("path_ic", 1.0)
        print(f"  {pidx:>6} {path:>5} {nf:>18.4f} {pv_:>20.3f} {mv:>20.3f} {ic:>6.3f}")


def _normalize(v):
    arr = np.asarray(v, dtype=float).reshape(3)
    n = np.linalg.norm(arr)
    if n <= 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return arr / n


def _path_cumdist(path):
    pts = np.asarray(path, dtype=float).reshape(-1, 3)
    if len(pts) <= 1:
        return np.zeros(len(pts), dtype=float)
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])


def _path_polydata(path_world):
    pts = np.asarray(path_world, dtype=float).reshape(-1, 3)
    if len(pts) == 0:
        return None
    poly = pv.PolyData(pts)
    if len(pts) >= 2:
        cells = np.empty((len(pts) - 1, 3), dtype=np.int64)
        cells[:, 0] = 2
        cells[:, 1] = np.arange(len(pts) - 1)
        cells[:, 2] = np.arange(1, len(pts))
        poly.lines = cells.ravel()
    return poly


def _path_label_anchor(path_world, all_paths_world=None):
    pts = np.asarray(path_world, dtype=float).reshape(-1, 3)
    if len(pts) == 0:
        return None
    mid_idx = len(pts) // 2
    anchor = pts[mid_idx].copy()
    if len(pts) >= 2:
        i0 = max(0, mid_idx - 1)
        i1 = min(len(pts) - 1, mid_idx + 1)
        tangent = pts[i1] - pts[i0]
        tangent = _normalize(tangent)
        ref = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(np.dot(tangent, ref)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0], dtype=float)
        normal = np.cross(tangent, ref)
        normal = _normalize(normal)
        if all_paths_world:
            all_pts = []
            for p in all_paths_world:
                arr = np.asarray(p, dtype=float).reshape(-1, 3)
                if len(arr) > 0:
                    all_pts.append(arr)
            if all_pts:
                all_pts = np.concatenate(all_pts, axis=0)
                bmin = all_pts.min(axis=0)
                bmax = all_pts.max(axis=0)
                diag = np.linalg.norm(bmax - bmin)
                offset = max(2.0, 0.02 * diag)
                anchor = anchor + normal * offset
    return anchor


def _plane_mesh(center_world, normal, size):
    return pv.Plane(
        center=np.asarray(center_world, dtype=float).reshape(3),
        direction=_normalize(normal),
        i_size=float(size),
        j_size=float(size),
        i_resolution=1,
        j_resolution=1,
    )


def _scalar_bar_args(title, bar_cfg=None):
    cfg = {
        "title": title,
        "vertical": True,
        "position_x": 0.86,
        "position_y": 0.1,
        "height": 0.8,
        "width": 0.08,
        "title_font_size": 18,
        "label_font_size": 14,
        "n_labels": 5,
        "fmt": "%.3g",
    }
    if bar_cfg:
        cfg.update(bar_cfg)
    return cfg


def _ensure_offscreen():
    try:
        if hasattr(pv, "start_xvfb") and not os.environ.get("DISPLAY"):
            pv.start_xvfb()
    except Exception:
        pass


def _make_plotter(window_size=WINDOW_SIZE):
    _ensure_offscreen()
    p = pv.Plotter(off_screen=True, window_size=window_size)
    p.set_background("white")
    return p


def _write_video(frames, out_path, fps=24):
    if not frames:
        return None
    out_path = os.path.splitext(out_path)[0] + ".mp4"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        with imageio.get_writer(out_path, fps=fps, codec="libx264", macro_block_size=None) as writer:
            for frame in frames:
                writer.append_data(np.asarray(frame))
        return out_path
    except Exception:
        gif_path = os.path.splitext(out_path)[0] + ".gif"
        imageio.mimsave(gif_path, [np.asarray(frame) for frame in frames], duration=1.0 / max(int(fps), 1))
        return gif_path


def _surface_center_radius(poly):
    if poly is None or poly.n_points == 0:
        return np.zeros(3, dtype=float), 100.0
    b = np.array(poly.bounds, dtype=float).reshape(3, 2)
    center = b.mean(axis=1)
    extent = np.maximum(b[:, 1] - b[:, 0], 1.0)
    radius = float(max(np.linalg.norm(extent) * 1.2, 50.0))
    return center, radius


CAMERA_PRESETS = {
    "iso": (35.0, 25.0),
    "iso_back": (215.0, 25.0),
    "right": (0.0, 0.0),
    "left": (180.0, 0.0),
    "anterior": (270.0, 0.0),
    "posterior": (90.0, 0.0),
    "superior": (0.0, 89.9),
    "inferior": (0.0, -89.9),
}


def _resolve_view(view):
    if view is None:
        return CAMERA_PRESETS["iso"]
    if isinstance(view, str):
        if view not in CAMERA_PRESETS:
            raise ValueError(f"unknown camera preset: {view}, options: {list(CAMERA_PRESETS)}")
        return CAMERA_PRESETS[view]
    az, el = view
    return float(az), float(el)


def _camera_from_scene(poly, azimuth_deg=35.0, elevation_deg=25.0, distance_scale=1.0):
    center, radius = _surface_center_radius(poly)
    radius = radius * float(distance_scale)
    az = np.deg2rad(float(azimuth_deg))
    el = np.deg2rad(float(elevation_deg))
    pos = center + np.array([
        radius * np.cos(el) * np.cos(az),
        radius * np.cos(el) * np.sin(az),
        radius * np.sin(el),
    ], dtype=float)
    if abs(elevation_deg) > 80.0:
        up = (0.0, 1.0, 0.0)
    else:
        up = (0.0, 0.0, 1.0)
    return [tuple(pos.tolist()), tuple(center.tolist()), up]


def _orbit_camera(poly, azimuth_deg, elevation_deg=25.0, distance_scale=1.0):
    return _camera_from_scene(poly, azimuth_deg, elevation_deg, distance_scale)


def _camera_from_view(poly, view, distance_scale=1.0):
    az, el = _resolve_view(view)
    return _camera_from_scene(poly, az, el, distance_scale)

def _time_and_azimuth(frame_idx, rotation_frames, n_time, time_repeat=1):
    rotation_frames = int(max(rotation_frames, 1))
    n_time = int(max(n_time, 1))
    time_repeat = int(max(time_repeat, 1))

    t = (frame_idx // time_repeat) % n_time
    az = 360.0 * (frame_idx % rotation_frames) / rotation_frames
    return t, az
def _build_union_surface(ws, smoothing_iteration=200):
    if ws.segmask_binary is not None:
        mask3d = np.any(np.asarray(ws.segmask_binary, dtype=bool), axis=3)
    else:
        mask3d = np.asarray(ws.segmask_3d, dtype=bool)
    mesh = create_uniform_grid(mask3d, ws.resolution, origin=ws.origin)
    mesh = mesh.threshold(0.1)
    if mesh is None or mesh.n_cells == 0:
        return None, None
    surf = mesh.extract_surface()
    if surf is not None and surf.n_points > 0 and int(smoothing_iteration) > 0:
        surf = surf.smooth(n_iter=int(smoothing_iteration))
    return mesh, surf


def _plane_size_from_surface(surf):
    if surf is None or surf.n_points == 0:
        return 25.0
    b = np.array(surf.bounds, dtype=float).reshape(3, 2)
    extent = b[:, 1] - b[:, 0]
    return float(max(12.0, 0.12 * np.max(extent)))


def _make_plane_payload(ws, source_path=""):
    origin = np.asarray(ws.origin, dtype=float).reshape(3)
    payload = {
        "source": source_path,
        "origin": origin.tolist(),
        "resolution": np.asarray(ws.resolution, dtype=float).reshape(3).tolist(),
        "planes": [],
    }
    for i, plane in enumerate(ws.planes):
        center_local = np.asarray(plane.center, dtype=float).reshape(3)
        payload["planes"].append({
            "plane_index": int(i),
            "center": center_local.tolist(),
            "center_world": (center_local + origin).tolist(),
            "normal": _normalize(plane.normal).tolist(),
            "label": int(plane.label),
            "path_index": int(plane.path_index),
            "distance": float(plane.distance),
        })
    return payload


def save_plane_positions(ws, out_path, source_path=""):
    payload = _make_plane_payload(ws, source_path=source_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def load_plane_positions(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "planes" in payload:
        return payload["planes"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Invalid plane position file: {path}")


def _nearest_path_info(center_world, paths_world):
    best_dist = np.inf
    best_path_idx = -1
    best_point_idx = -1
    best_distance = 0.0
    for path_idx, path in enumerate(paths_world):
        pts = np.asarray(path, dtype=float).reshape(-1, 3)
        if len(pts) == 0:
            continue
        d = np.linalg.norm(pts - center_world.reshape(1, 3), axis=1)
        point_idx = int(np.argmin(d))
        dist = float(d[point_idx])
        if dist < best_dist:
            cum = _path_cumdist(pts)
            best_dist = dist
            best_path_idx = int(path_idx)
            best_point_idx = point_idx
            best_distance = float(cum[point_idx]) if len(cum) > point_idx else 0.0
    return best_path_idx, best_point_idx, best_dist, best_distance


def _path_tangent(path_world, point_idx):
    pts = np.asarray(path_world, dtype=float).reshape(-1, 3)
    if len(pts) == 0:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    i0 = max(0, int(point_idx) - 1)
    i1 = min(len(pts) - 1, int(point_idx) + 1)
    if i1 == i0:
        i1 = min(len(pts) - 1, i0 + 1)
    tangent = pts[i1] - pts[i0]
    return _normalize(tangent)


def project_planes_to_workspace(plane_items, ws):
    origin = np.asarray(ws.origin, dtype=float).reshape(3)
    paths_local = ws.centerline_paths_smooth if len(ws.centerline_paths_smooth) > 0 else ws.centerline_paths
    paths_world = [np.asarray(path, dtype=float).reshape(-1, 3) + origin.reshape(1, 3) for path in paths_local]
    planes = []
    for i, item in enumerate(plane_items):
        if "center_world" in item:
            center_world = np.asarray(item["center_world"], dtype=float).reshape(3)
        elif "center" in item:
            center_world = np.asarray(item["center"], dtype=float).reshape(3)
        else:
            continue
        normal = _normalize(item.get("normal", [1.0, 0.0, 0.0]))
        path_index = int(item.get("path_index", -1))
        distance = float(item.get("distance", 0.0))
        if paths_world:
            nearest_path_idx, nearest_point_idx, _, nearest_distance = _nearest_path_info(center_world, paths_world)
            if nearest_path_idx >= 0:
                path_index = int(nearest_path_idx)
                distance = float(nearest_distance)
                if np.linalg.norm(normal) <= 1e-12:
                    normal = _path_tangent(paths_world[path_index], nearest_point_idx)
        if path_index < 0:
            path_index = 0
        planes.append(PlaneData(
            center=center_world - origin,
            normal=_normalize(normal),
            label=int(path_index) + 1,
            path_index=int(path_index),
            distance=float(distance),
        ))
    return planes


def resolve_reuse_plane_file(reuse_spec, case_name):
    if not reuse_spec:
        return ""
    if os.path.isfile(reuse_spec):
        return reuse_spec
    if os.path.isdir(reuse_spec):
        candidates = [
            os.path.join(reuse_spec, case_name, "plane_positions.json"),
            os.path.join(reuse_spec, case_name, "planes.json"),
            os.path.join(reuse_spec, "plane_positions.json"),
            os.path.join(reuse_spec, "planes.json"),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
    return reuse_spec


def render_plane_rotation_video(
    ws,
    out_dir,
    fps=24,
    n_frames=180,
    smoothing_iteration=200,
    elevation_deg=0.0,
    distance_scale=1.0,
    add_plane_idx=False,
    add_path_idx=False,
):
    _, surf = _build_union_surface(ws, smoothing_iteration=smoothing_iteration)
    if surf is None or surf.n_points == 0:
        return None

    plane_size = _plane_size_from_surface(surf)
    origin = np.asarray(ws.origin, dtype=float).reshape(3)
    p = _make_plotter()
    p.add_mesh(surf, opacity=0.18, color="white")

    paths_world = []
    for path in ws.centerline_paths_smooth:
        path_world = np.asarray(path, dtype=float) + origin.reshape(1, 3)
        paths_world.append(path_world)
        poly = _path_polydata(path_world)
        if poly is not None and poly.n_points > 0:
            p.add_mesh(poly, color="deepskyblue", line_width=5, render_lines_as_tubes=True)

    centers = []
    plane_labels = []
    for i, plane in enumerate(ws.planes):
        center_world = np.asarray(plane.center, dtype=float).reshape(3) + origin
        pm = _plane_mesh(center_world, plane.normal, plane_size)
        p.add_mesh(pm, color="yellow", opacity=0.75, show_edges=True, edge_color="black", line_width=2)
        centers.append(center_world)
        plane_labels.append(f"Plane {i}")

    if add_plane_idx and len(centers) > 0:
        p.add_point_labels(
            np.asarray(centers, dtype=float),
            plane_labels,
            font_size=28,
            bold=True,
            text_color="black",
            fill_shape=True,
            shape="rounded_rect",
            shape_color="yellow",
            shape_opacity=0.85,
            margin=5,
            always_visible=True,
        )

    if add_path_idx and len(paths_world) > 0:
        path_label_points = []
        path_label_texts = []
        offsets = [
            np.array([0, 0, 0]),
            np.array([3, 0, 0]),
            np.array([-3, 0, 0]),
            np.array([0, 3, 0]),
            np.array([0, -3, 0]),
        ]
        frac_choices = [0.25, 0.5, 0.75, 0.35, 0.65]

        for idx, path_world in enumerate(paths_world):
            if path_world is None or len(path_world) == 0:
                continue
            n = len(path_world)
            frac = frac_choices[idx % len(frac_choices)]
            k = min(max(int(frac * (n - 1)), 0), n - 1)
            anchor = np.asarray(path_world[k], dtype=float) + offsets[idx % len(offsets)]
            path_label_points.append(anchor)
            path_label_texts.append(f"Branch {idx}")

        if len(path_label_points) > 0:
            p.add_point_labels(
                np.asarray(path_label_points, dtype=float),
                path_label_texts,
                font_size=20,
                bold=True,
                text_color="black",
                fill_shape=True,
                shape="rounded_rect",
                shape_color="deepskyblue",
                shape_opacity=0.85,
                margin=2,
                always_visible=True,
            )

    frames = []
    for frame_idx in range(int(max(n_frames, 1))):
        az = 360.0 * frame_idx / max(n_frames, 1)
        p.camera_position = _orbit_camera(surf, az, elevation_deg, distance_scale)
        p.add_text(
            f"Rotating {frame_idx + 1}/{int(max(n_frames, 1))}",
            position="upper_left",
            font_size=14,
            color="black",
            name="frame_text",
        )
        p.render()
        frames.append(np.asarray(p.screenshot(return_img=True)))
        try:
            p.remove_actor("frame_text")
        except Exception:
            pass
    p.close()
    return _write_video(frames, os.path.join(out_dir, "planes_rotate.mp4"), fps=fps)


def render_wss_video(
    ws,
    out_dir,
    fps=24,
    smoothing_iteration=200,
    view="iso",
    distance_scale=1.0,
    wss_clim=None,
    wss_bar_cfg=None,
    rotate=False,
    rotation_frames=None,
    elevation_deg=None,
    time_repeat=1
):
    if not ws.derived.wss_surfaces:
        return None

    _, context_surf = _build_union_surface(ws, smoothing_iteration=smoothing_iteration)
    if context_surf is None or context_surf.n_points == 0:
        return None

    wss_max = 0.0
    for surf in ws.derived.wss_surfaces:
        if surf is not None and surf.n_points > 0 and "wss" in surf.point_data:
            vals = np.asarray(surf.point_data["wss"], dtype=float)
            if vals.size:
                wss_max = max(wss_max, float(np.nanmax(vals)))
    wss_max = max(wss_max, 1e-6)
    clim = wss_clim if wss_clim is not None else (0.0, wss_max)

    az0, el0 = _resolve_view(view)
    if elevation_deg is None:
        elevation_deg = el0

    n_time = int(max(ws.time_count(), 1))
    if rotate:
        base_frames = n_time * int(max(time_repeat, 1))
        if rotation_frames is not None:
            total_frames = max(int(rotation_frames), base_frames)
        else:
            total_frames = base_frames
    else:
        total_frames = n_time * int(max(time_repeat, 1))

    p = _make_plotter()
    frames = []

    for frame_idx in range(total_frames):
        if rotate:
            t, az = _time_and_azimuth(
                frame_idx,
                rotation_frames=rotation_frames if rotation_frames is not None else total_frames,
                n_time=n_time,
                time_repeat=time_repeat,
            )
            camera_position = _orbit_camera(context_surf, az, elevation_deg, distance_scale)
        else:
            t = min(frame_idx, n_time - 1)
            camera_position = _camera_from_view(context_surf, view, distance_scale)

        p.clear()
        p.set_background("white")
        p.add_mesh(context_surf, opacity=0.08, color="white")

        surf = ws.derived.wss_surfaces[min(max(0, t), len(ws.derived.wss_surfaces) - 1)]
        if surf is not None and surf.n_points > 0 and "wss" in surf.point_data:
            p.add_mesh(
                surf,
                scalars="wss",
                cmap="jet",
                clim=clim,
                show_scalar_bar=True,
                scalar_bar_args=_scalar_bar_args("WSS (Pa)", wss_bar_cfg),
            )

        if rotate:
            txt = f"t={t} | rot {frame_idx + 1}/{total_frames}"
        else:
            txt = f"t={t}"

        p.add_text(txt, position="upper_left", font_size=14, color="black")
        p.camera_position = camera_position
        p.render()
        frames.append(np.asarray(p.screenshot(return_img=True)))

    p.close()
    suffix = "rotate" if rotate else "video"
    return _write_video(frames, os.path.join(out_dir, f"wss_{suffix}.mp4"), fps=fps)

def _streamline_speed_max(ws):
    if ws.flow_raw is None:
        return 1e-6
    speed = np.linalg.norm(np.asarray(ws.flow_raw, dtype=float) / 100.0, axis=-1)
    if ws.segmask_binary is not None and np.any(ws.segmask_binary):
        vals = speed[np.asarray(ws.segmask_binary, dtype=bool)]
        if vals.size:
            return max(float(np.nanmax(vals)), 1e-6)
    return max(float(np.nanmax(speed)), 1e-6)


def _ensure_streamline_scalars(sl):
    if sl is None:
        return sl
    if "Velocity" in sl.point_data or "Velocity" in sl.cell_data:
        return sl
    if "vector" in sl.point_data:
        sl.point_data["Velocity"] = np.linalg.norm(np.asarray(sl.point_data["vector"], dtype=float), axis=1)
        return sl
    if "vector" in sl.cell_data:
        sl.cell_data["Velocity"] = np.linalg.norm(np.asarray(sl.cell_data["vector"], dtype=float), axis=1)
        return sl
    return sl


def render_streamlines_video(
    ws,
    out_dir,
    fps=24,
    smoothing_iteration=200,
    view="iso",
    distance_scale=1.0,
    streamline_clim=None,
    streamline_bar_cfg=None,
    rotate=False,
    rotation_frames=None,
    elevation_deg=None,
    time_repeat=1
):
    if ws.flow_raw is None or ws.segmask_binary is None or ws.segmask_3d is None:
        return None

    mesh, surf = _build_union_surface(ws, smoothing_iteration=smoothing_iteration)
    if mesh is None or surf is None or surf.n_points == 0:
        return None

    seeds = generate_seed_points(
        ws.segmask_3d,
        ws.resolution,
        ws.origin,
        ratio=ws.streamline_params.seed_ratio,
        rng_seed=ws.streamline_params.rng_seed,
        min_seeds=50,
    )

    v_max = _streamline_speed_max(ws)
    clim = streamline_clim if streamline_clim is not None else (0.0, v_max)

    az0, el0 = _resolve_view(view)
    if elevation_deg is None:
        elevation_deg = el0

    n_time = int(max(ws.time_count(), 1))
    if rotate:
        base_frames = n_time * int(max(time_repeat, 1))
        if rotation_frames is not None:
            total_frames = max(int(rotation_frames), base_frames)
        else:
            total_frames = base_frames
    else:
        total_frames = n_time * int(max(time_repeat, 1))

    p = _make_plotter()
    frames = []

    for frame_idx in range(total_frames):
        if rotate:
            t, az = _time_and_azimuth(
                frame_idx,
                rotation_frames=rotation_frames if rotation_frames is not None else total_frames,
                n_time=n_time,
                time_repeat=time_repeat,
            )
            camera_position = _orbit_camera(surf, az, elevation_deg, distance_scale)
        else:
            t = min(frame_idx, n_time - 1)
            camera_position = _camera_from_view(surf, view, distance_scale)

        mask_t = np.asarray(
            ws.segmask_binary[..., min(max(0, t), ws.segmask_binary.shape[3] - 1)],
            dtype=bool,
        )

        sl = generate_streamlines_at_t(
            ws.flow_raw,
            t,
            seeds,
            ws.resolution,
            ws.origin,
            mask_3d=mask_t,
            max_steps=ws.streamline_params.max_steps,
            terminal_speed=ws.streamline_params.terminal_speed,
            seed_ratio=ws.streamline_params.seed_ratio,
            min_seeds=50,
            rng_seed=ws.streamline_params.rng_seed,
        )
        sl = _ensure_streamline_scalars(sl)

        p.clear()
        p.set_background("white")
        p.add_mesh(surf, opacity=0.18, color="lightgray")

        if sl is not None and sl.n_points > 0:
            p.add_mesh(
                sl,
                scalars="Velocity",
                cmap="turbo",
                clim=clim,
                show_scalar_bar=True,
                scalar_bar_args=_scalar_bar_args("Velocity (m/s)", streamline_bar_cfg),
                render_lines_as_tubes=True,
                line_width=3,
            )

        if rotate:
            txt = f"t={t} | rot {frame_idx + 1}/{total_frames}"
        else:
            txt = f"t={t}"

        p.add_text(txt, position="upper_left", font_size=14, color="black")
        p.camera_position = camera_position
        p.render()
        frames.append(np.asarray(p.screenshot(return_img=True)))

    p.close()
    suffix = "rotate" if rotate else "video"
    return _write_video(frames, os.path.join(out_dir, f"streamlines_{suffix}.mp4"), fps=fps)

def _tke_max(ws):
    if ws.derived.tke_array is not None:
        return max(float(np.nanmax(np.asarray(ws.derived.tke_array, dtype=float))), 1e-6)
    tke_mesh = ws.derived.tke_volume
    if tke_mesh is None:
        return 1e-6
    if "TKE" in tke_mesh.point_data:
        return max(float(np.nanmax(np.asarray(tke_mesh.point_data["TKE"], dtype=float))), 1e-6)
    if "TKE" in tke_mesh.cell_data:
        return max(float(np.nanmax(np.asarray(tke_mesh.cell_data["TKE"], dtype=float))), 1e-6)
    return 1e-6

def render_tke_video(
    ws,
    out_dir,
    fps=24,
    smoothing_iteration=200,
    view="iso",
    distance_scale=1.0,
    tke_clim=None,
    tke_bar_cfg=None,
    rotate=False,
    rotation_frames=None,
    elevation_deg=None,
    time_repeat=1
):
    if ws.derived.tke_array is None and ws.derived.tke_volume is None:
        return None

    _, surf = _build_union_surface(ws, smoothing_iteration=smoothing_iteration)
    if surf is None or surf.n_points == 0:
        return None

    tke_max = _tke_max(ws)
    clim = tke_clim if tke_clim is not None else (0.0, tke_max)

    az0, el0 = _resolve_view(view)
    if elevation_deg is None:
        elevation_deg = el0

    n_time = int(max(ws.time_count(), 1))
    if rotate:
        base_frames = n_time * int(max(time_repeat, 1))
        if rotation_frames is not None:
            total_frames = max(int(rotation_frames), base_frames)
        else:
            total_frames = base_frames
    else:
        total_frames = n_time * int(max(time_repeat, 1))

    p = _make_plotter()
    frames = []

    for frame_idx in range(total_frames):
        if rotate:
            t, az = _time_and_azimuth(
                frame_idx,
                rotation_frames=rotation_frames if rotation_frames is not None else total_frames,
                n_time=n_time,
                time_repeat=time_repeat,
            )
            camera_position = _orbit_camera(surf, az, elevation_deg, distance_scale)
        else:
            t = min(frame_idx, n_time - 1)
            camera_position = _camera_from_view(surf, view, distance_scale)

        p.clear()
        p.set_background("white")
        p.add_mesh(surf, opacity=0.08, color="white")

        if ws.derived.tke_array is not None:
            arr = np.asarray(ws.derived.tke_array, dtype=np.float32)
            if arr.ndim == 4:
                vol_t = arr[..., min(max(0, t), arr.shape[3] - 1)]
            else:
                vol_t = arr
            tke_mesh = create_uniform_grid(vol_t, ws.resolution, origin=ws.origin, name="TKE")
            mesh_union = create_uniform_grid(
                np.max(ws.segmask_binary > 0, axis=-1),
                ws.resolution,
                origin=ws.origin,
            )
            mesh_union = mesh_union.threshold(0.1)
            tke_mesh = mesh_union.sample(tke_mesh)
            p.add_mesh(
                tke_mesh,
                scalars="TKE",
                cmap="hot",
                clim=clim,
                show_scalar_bar=True,
                scalar_bar_args=_scalar_bar_args("TKE (J/m³)", tke_bar_cfg),
            )
        else:
            p.add_mesh(
                ws.derived.tke_volume,
                scalars="TKE",
                cmap="hot",
                clim=clim,
                show_scalar_bar=True,
                scalar_bar_args=_scalar_bar_args("TKE (J/m³)", tke_bar_cfg),
            )

        if rotate:
            txt = f"t={t} | rot {frame_idx + 1}/{total_frames}"
        else:
            txt = f"t={t}"

        p.add_text(txt, position="upper_left", font_size=14, color="black")
        p.camera_position = camera_position
        p.render()
        frames.append(np.asarray(p.screenshot(return_img=True)))

    p.close()
    suffix = "rotate" if rotate else "video"
    return _write_video(frames, os.path.join(out_dir, f"tke_{suffix}.mp4"), fps=fps)
def _format_path_group(v):
    if v is None:
        return "?"
    if isinstance(v, dict):
        if "paths" in v:
            v = v["paths"]
        elif "path_indices" in v:
            v = v["path_indices"]
    if isinstance(v, (int, np.integer)):
        return f"Branch{int(v)}"
    if isinstance(v, str):
        return v
    try:
        vals = list(v)
    except Exception:
        return str(v)
    if not vals:
        return "-"
    out = []
    for x in vals:
        if isinstance(x, (int, np.integer)):
            out.append(f"Branch{int(x)}")
        else:
            out.append(str(x))
    return "+".join(out)


def _fork_side_text(fork):
    if not isinstance(fork, dict):
        return "left=?", "right=?"

    left = (
        fork.get("left")
        or fork.get("left_paths")
        or fork.get("left_path_indices")
        or fork.get("in_paths")
        or fork.get("in_path_indices")
    )
    right = (
        fork.get("right")
        or fork.get("right_paths")
        or fork.get("right_path_indices")
        or fork.get("out_paths")
        or fork.get("out_path_indices")
    )

    return f"{_format_path_group(left)}", f"{_format_path_group(right)}"


def print_qc_summary(qc_data, forks=None):
    if not qc_data:
        print("  No QC results to summarize.")
        return

    items = []
    if isinstance(qc_data, list):
        items = qc_data
    elif isinstance(qc_data, dict):
        if isinstance(qc_data.get("forks"), list):
            items = qc_data["forks"]
        elif isinstance(qc_data.get("fork_qc"), list):
            items = qc_data["fork_qc"]
        else:
            for k, v in qc_data.items():
                if isinstance(v, dict):
                    row = dict(v)
                    row.setdefault("fork_index", k)
                    items.append(row)

    rows = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        fork_idx = item.get("fork_index", item.get("fork_id", item.get("fork", i)))
        ic = item.get("internal_consistency", item.get("ic", item.get("path_ic", np.nan)))

        fork_obj = None
        if isinstance(forks, list):
            try:
                if isinstance(fork_idx, (int, np.integer)) and 0 <= int(fork_idx) < len(forks):
                    fork_obj = forks[int(fork_idx)]
                elif i < len(forks):
                    fork_obj = forks[i]
            except Exception:
                pass

        left_txt, right_txt = _fork_side_text(fork_obj)
        rows.append((fork_idx, ic, left_txt, right_txt))

    if not rows:
        print("  No fork-level QC results found.")
        print(json.dumps(qc_data, ensure_ascii=False, indent=2))
        return

    print(f"  {'Fork':>6} {'Internal Consistency':>24} {'Left':>24} {'Right':>24}")
    print(f"  {'-'*6} {'-'*24} {'-'*24} {'-'*24}")
    for fork_idx, ic, left_txt, right_txt in rows:
        ic_str = f"{float(ic):.6f}" if np.isfinite(ic) else "nan"
        print(f"  {str(fork_idx):>6} {ic_str:>24} {left_txt:>24} {right_txt:>24}")


def process_single(
    h5_path,
    out_dir,
    workspace=None,
    skip_derived=False,
    skip_plane_metrics=False,
    use_multithread=False,
    reuse_planes_path="",
    fps=24,
    plane_rotation_frames=180,
    rotate_dynamic_video=False,
    dynamic_rotation_frames=180,
    dynamic_rotation_elevation_deg=None,
    make_plane_video=True,
    make_wss_video=True,
    make_streamlines_video=True,
    make_tke_video=True,
    camera_view="iso",
    camera_distance_scale=1.0,
    add_plane_idx=False,
    add_path_idx=False,
    wss_clim=None,
    wss_bar_cfg=None,
    tke_clim=None,
    tke_bar_cfg=None,
    streamline_clim=None,
    streamline_bar_cfg=None,
    dynamic_time_repeat=1,
):
    print(f"\n{'=' * 60}")
    print(f"Processing: {h5_path}")
    print(f"Output dir: {out_dir}")
    print(f"{'=' * 60}")

    os.makedirs(out_dir, exist_ok=True)
    ws = copy.deepcopy(workspace) if workspace is not None else Workspace()
    ws.paths.segmask_path = h5_path
    ws.paths.flow_path = h5_path
    ws.paths.output_dir = out_dir
    ws.derived_params.use_multithread = use_multithread
    engine = PipelineEngine()
    logger = lambda msg: print(f"  [LOG] {msg}")

    import time as _time
    _t_total_start = _time.time()

    _t0 = _time.time()
    print("[1/7] Loading data...")
    engine.load_data(ws, logger)
    print(f"  -> Step 1 took {_time.time() - _t0:.2f}s")

    _t0 = _time.time()
    print("[2/7] Generate Skeleton...")
    r = engine.run_step(ws, StepId.GENERATE_SKELETON, logger)
    print(f"  -> {r.message}")
    print(f"  -> Step 2 took {_time.time() - _t0:.2f}s")

    _t0 = _time.time()
    print("[3/7] Generate Graph (+ branches/forks)...")
    r = engine.run_step(ws, StepId.GENERATE_GRAPH, logger)
    print(f"  -> {r.message}")
    print(f"  -> Step 3 took {_time.time() - _t0:.2f}s")

    _t0 = _time.time()
    print("[4/7] Generate Planes...")
    r = engine.run_step(ws, StepId.GENERATE_PLANES, logger)
    print(f"  -> {r.message}")
    print(f"  -> Step 4 took {_time.time() - _t0:.2f}s")

    _t0 = _time.time()
    if reuse_planes_path:
        print(f"[5/7] Reuse Plane Positions: {reuse_planes_path}")
        plane_items = load_plane_positions(reuse_planes_path)
        ws.planes = project_planes_to_workspace(plane_items, ws)
        planes_json = engine._save_planes_json(ws)
        print(f"  -> Reused {len(ws.planes)} planes saved={planes_json}")
    else:
        print("[5/7] Use generated planes")
    print(f"  -> Step 5 took {_time.time() - _t0:.2f}s")

    _t0 = _time.time()
    if skip_plane_metrics:
        print("[6/7] Skipped plane metrics")
    else:
        print("[6/7] Calculate & Save Metrics...")
        _, _, metric_msg = engine._compute_plane_metrics_internal(
            ws,
            save=True,
            use_multithread=use_multithread,
        )
        print(f"  -> {metric_msg}")
        try:
            engine._save_planes_json(ws)
        except Exception:
            pass
    print(f"  -> Step 6 took {_time.time() - _t0:.2f}s")

    _t0 = _time.time()
    pixelwise_result = {}
    if not skip_derived:
        print("[7/7] Compute Derived Metrics (WSS/TKE)...")
        dp = ws.derived_params
        engine.preprocess(ws)
        loaded_tke = ws.derived.tke_array

        result = compute_derived_metrics(
            flow=ws.flow_raw * ws.segmask_binary[..., None],
            mask4d=ws.segmask_binary,
            spacing=ws.resolution,
            origin=ws.origin,
            smoothing_iteration=dp.smoothing_iteration,
            viscosity=dp.viscosity,
            inward_distance=dp.inward_distance,
            parabolic_fitting=dp.parabolic_fitting,
            no_slip_condition=dp.no_slip_condition,
            step_size=dp.step_size,
            tube_radius=dp.tube_radius,
            rho=dp.rho,
            save_pixelwise=True,
            tke_array=loaded_tke,
        )
        ws.derived.wss_surfaces = result["wss_surfaces"]
        ws.derived.wss_volume = result.get("wss_volume")
        ws.derived.tke_volume = result["tke_volume"]
        ws.derived.tke_array = result.get("tke_array")
        ws.derived.pixelwise_export = result.get("pixelwise_export", {})
        pixelwise_result = ws.derived.pixelwise_export
        pixel_path = os.path.join(out_dir, "derived_metrics_pixelwise.npz")
        if pixelwise_result:
            np.savez_compressed(pixel_path, **pixelwise_result)
            print(f"  -> Saved pixelwise: {pixel_path}")
        ws.pipeline.mark_done(StepId.COMPUTE_DERIVED_METRICS)
        print(f"  -> Derived: Nt={len(ws.derived.wss_surfaces)}")
    else:
        print("[7/7] Skipped derived metrics (WSS/TKE)")
    print(f"  -> Step 7 took {_time.time() - _t0:.2f}s")
    total_time_sec = _time.time() - _t_total_start
    print(f"  => Total pipeline took {total_time_sec:.2f}s")

    plane_positions_path = save_plane_positions(ws, os.path.join(out_dir, "plane_positions.json"), source_path=h5_path)
    print(f"Plane positions saved: {plane_positions_path}")

    video_paths = {}
    if make_plane_video:
        try:
            video_paths["planes"] = render_plane_rotation_video(
                ws,
                out_dir,
                fps=fps,
                n_frames=plane_rotation_frames,
                smoothing_iteration=ws.derived_params.smoothing_iteration,
                distance_scale=camera_distance_scale,
                add_plane_idx=add_plane_idx,
                add_path_idx=add_path_idx,
            )
            if video_paths["planes"]:
                print(f"Plane video saved: {video_paths['planes']}")
        except Exception:
            print("[WARN] Plane video failed")
            print(traceback.format_exc())
            video_paths["planes"] = ""

    if make_streamlines_video:
        try:
            video_paths["streamlines"] = render_streamlines_video(
                ws,
                out_dir,
                fps=fps,
                smoothing_iteration=ws.derived_params.smoothing_iteration,
                view=camera_view,
                distance_scale=camera_distance_scale,
                streamline_clim=streamline_clim,
                streamline_bar_cfg=streamline_bar_cfg,
                rotate=rotate_dynamic_video,
                rotation_frames=dynamic_rotation_frames,
                elevation_deg=dynamic_rotation_elevation_deg,
                time_repeat=dynamic_time_repeat,
            )
            if video_paths["streamlines"]:
                print(f"Streamlines video saved: {video_paths['streamlines']}")
        except Exception:
            print("[WARN] Streamlines video failed")
            print(traceback.format_exc())
            video_paths["streamlines"] = ""

    if not skip_derived and make_wss_video:
        try:
            video_paths["wss"] = render_wss_video(
                ws,
                out_dir,
                fps=fps,
                smoothing_iteration=ws.derived_params.smoothing_iteration,
                view=camera_view,
                distance_scale=camera_distance_scale,
                wss_clim=wss_clim,
                wss_bar_cfg=wss_bar_cfg,
                rotate=rotate_dynamic_video,
                rotation_frames=dynamic_rotation_frames,
                elevation_deg=dynamic_rotation_elevation_deg,
                time_repeat=dynamic_time_repeat,
            )
            if video_paths["wss"]:
                print(f"WSS video saved: {video_paths['wss']}")
        except Exception:
            print("[WARN] WSS video failed")
            print(traceback.format_exc())
            video_paths["wss"] = ""

    if not skip_derived and make_tke_video:
        try:
            video_paths["tke"] = render_tke_video(
                ws,
                out_dir,
                fps=fps,
                smoothing_iteration=ws.derived_params.smoothing_iteration,
                view=camera_view,
                distance_scale=camera_distance_scale,
                tke_clim=tke_clim,
                tke_bar_cfg=tke_bar_cfg,
                rotate=rotate_dynamic_video,
                rotation_frames=dynamic_rotation_frames,
                elevation_deg=dynamic_rotation_elevation_deg,
                time_repeat=dynamic_time_repeat,
            )
            if video_paths["tke"]:
                print(f"TKE video saved: {video_paths['tke']}")
        except Exception:
            print("[WARN] TKE video failed")
            print(traceback.format_exc())
            video_paths["tke"] = ""
    table_rows, raw_metrics, qc_data = (None, None, None)
    if not skip_plane_metrics:
        table_rows, raw_metrics, qc_data = load_metrics_from_output(out_dir)

    if table_rows:
        print("\n  === Plane Metrics Summary ===")
        print_metrics_summary(table_rows)

    if qc_data:
        print("\n  === Fork QC Summary ===")
        print_qc_summary(qc_data, ws.forks)

    summary = {
        "input": h5_path,
        "output_dir": out_dir,
        "resolution": ws.resolution.tolist(),
        "origin": np.asarray(ws.origin, dtype=float).reshape(3).tolist(),
        "rr": ws.rr,
        "total_time_sec": float(total_time_sec),
        "n_planes": len(ws.planes),
        "n_skeleton_pts": len(ws.skeleton_points) if ws.skeleton_points is not None else 0,
        "n_graph_nodes": len(ws.graph.points),
        "n_graph_edges": len(ws.graph.edges),
        "n_paths": len(ws.centerline_paths_smooth),
        "n_forks": len(ws.forks),
        "path_info": ws.path_info,
        "forks": ws.forks,
        "plane_metrics": ws.derived.plane_metrics,
        "plane_qc": ws.derived.plane_qc,
        "plane_positions_file": plane_positions_path,
        "reused_planes_file": reuse_planes_path,
        "videos": video_paths,
        "pixelwise_export": {k: list(np.asarray(v).shape) for k, v in pixelwise_result.items()} if pixelwise_result else {},
    }
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSummary saved: {summary_path}")
    return summary


def collect_h5_files(inputs):
    files = []
    for inp in inputs:
        if os.path.isfile(inp) and inp.lower().endswith((".h5", ".hdf5")):
            files.append(inp)
        elif os.path.isdir(inp):
            files.extend(sorted(glob.glob(os.path.join(inp, "**", "*.h5"), recursive=True)))
            files.extend(sorted(glob.glob(os.path.join(inp, "**", "*.hdf5"), recursive=True)))
    return sorted(dict.fromkeys(files))


def build_base_workspace():
    ws = Workspace()
    ws.plane_gen_params.use_center_plane = globals().get("USE_CENTER_PLANE", True)
    ws.plane_gen_params.cross_section_distance = globals().get("CROSS_SECTION_DIST", 5.0)
    ws.plane_gen_params.start_distance = globals().get("START_DIST", 5.0)
    ws.plane_gen_params.end_distance = globals().get("END_DIST", 0.0)
    ws.skeleton_params.remove_small_cc = globals().get("REMOVE_SMALL_CC", True)
    ws.skeleton_params.min_cc_volume_mm3 = globals().get("MIN_CC_VOLUME", 50.0)
    ws.streamline_params.max_steps = 2000
    ws.streamline_params.min_seeds = 50
    ws.streamline_params.seed_ratio = globals().get("SEED_RATIO", 50.0)
    ws.streamline_params.tube_radius = globals().get("TUBE_RADIUS", 0.05)
    return ws


def run_batch():
    inputs = globals().get("INPUT", None)
    if inputs is None:
        raise ValueError("INPUT is not defined.")
    dynamic_time_repeat = globals().get("DYNAMIC_TIME_REPEAT", 1)
    output_dir = globals().get("OUTPUT_DIR", "./batch_output")
    skip_derived = globals().get("SKIP_DERIVED", False)
    use_multithread = globals().get("USE_MULTITHREAD", True)
    reuse_planes = globals().get("REUSE_PLANES", "")
    fps = globals().get("FPS", 12)
    plane_rotation_frames = globals().get("PLANE_ROTATION_FRAMES", 180)
    make_plane_video = globals().get("MAKE_PLANE_VIDEO", True)
    make_wss_video = globals().get("MAKE_WSS_VIDEO", True)
    make_streamlines_video = globals().get("MAKE_STREAMLINES_VIDEO", True)
    make_tke_video = globals().get("MAKE_TKE_VIDEO", True)
    camera_view = globals().get("CAMERA_VIEW", "posterior")
    camera_distance_scale = globals().get("CAMERA_DISTANCE_SCALE", 1.5)
    skip_plane_metrics = globals().get("SKIP_PLANE_METRICS", False)
    rotate_dynamic_video = globals().get("ROTATE_DYNAMIC_VIDEO", False)
    dynamic_rotation_frames = globals().get("DYNAMIC_ROTATION_FRAMES", 180)
    dynamic_rotation_elevation_deg = globals().get("DYNAMIC_ROTATION_ELEVATION_DEG", None)
    add_plane_idx = globals().get("ADD_PLANE_IDX", False)
    add_path_idx = globals().get("ADD_PATH_IDX", False)

    wss_clim = globals().get("WSS_CLIM", (0, 5))
    wss_bar_cfg = globals().get(
        "WSS_BAR_CFG",
        {"position_x": 0.75, "position_y": 0.2, "height": 0.22, "width": 0.05, "title_font_size": 40, "label_font_size": 32},
    )
    tke_clim = globals().get("TKE_CLIM", (0, 2))
    tke_bar_cfg = globals().get(
        "TKE_BAR_CFG",
        {"position_x": 0.75, "position_y": 0.2, "height": 0.22, "width": 0.05, "title_font_size": 40, "label_font_size": 32},
    )
    streamline_clim = globals().get("STREAMLINE_CLIM", (0, 0.6))
    streamline_bar_cfg = globals().get(
        "STREAMLINE_BAR_CFG",
        {"position_x": 0.75, "position_y": 0.2, "height": 0.22, "width": 0.05, "title_font_size": 40, "label_font_size": 32},
    )

    h5_files = collect_h5_files(inputs)
    if not h5_files:
        print("No H5 files found.")
        return [], ""

    print(f"Found {len(h5_files)} file(s) to process.")
    base_ws = build_base_workspace()
    results = []
    case_out = ""

    for path in h5_files:
        name = os.path.splitext(os.path.basename(path))[0]
        case_out = os.path.join(output_dir, name)
        reuse_file = resolve_reuse_plane_file(reuse_planes, name)

        if reuse_planes and not os.path.isfile(reuse_file):
            results.append({"file": path, "status": "error", "error": f"reuse plane file not found: {reuse_planes}"})
            print(f"\n[ERROR] Reuse plane file not found: {reuse_planes}")
            continue

        try:
            summary = process_single(
                path,
                case_out,
                workspace=base_ws,
                skip_derived=skip_derived,
                use_multithread=use_multithread,
                reuse_planes_path=reuse_file,
                fps=fps,
                plane_rotation_frames=plane_rotation_frames,
                make_plane_video=make_plane_video,
                make_wss_video=make_wss_video,
                make_streamlines_video=make_streamlines_video,
                make_tke_video=make_tke_video,
                camera_view=camera_view,
                camera_distance_scale=camera_distance_scale,
                add_plane_idx=add_plane_idx,
                add_path_idx=add_path_idx,
                wss_clim=wss_clim,
                wss_bar_cfg=wss_bar_cfg,
                tke_clim=tke_clim,
                tke_bar_cfg=tke_bar_cfg,
                streamline_clim=streamline_clim,
                streamline_bar_cfg=streamline_bar_cfg,
                skip_plane_metrics=skip_plane_metrics,
                rotate_dynamic_video=rotate_dynamic_video,
                dynamic_rotation_frames=dynamic_rotation_frames,
                dynamic_rotation_elevation_deg=dynamic_rotation_elevation_deg,
                dynamic_time_repeat=dynamic_time_repeat,
            )
            results.append({"file": path, "status": "ok", "summary": summary})
        except Exception:
            print(f"\n[ERROR] Failed: {path}")
            print(traceback.format_exc())
            results.append({"file": path, "status": "error", "error": traceback.format_exc()})

    os.makedirs(output_dir, exist_ok=True)
    batch_report = os.path.join(output_dir, "batch_report.json")
    with open(batch_report, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    n_ok = sum(1 for r in results if r["status"] == "ok")

    times_sec = []
    for r in results:
        if r.get("status") == "ok":
            t = r.get("summary", {}).get("total_time_sec", None)
            if t is not None:
                times_sec.append(float(t))

    if times_sec:
        times_sec = np.asarray(times_sec, dtype=float)
        mean_sec = times_sec.mean()
        std_sec = times_sec.std(ddof=1) if len(times_sec) > 1 else 0.0
        time_text = f"{mean_sec:.2f} ± {std_sec:.2f} s"
        print(f"\nCase time: {time_text}")

        with open(os.path.join(output_dir, "time_summary.txt"), "w", encoding="utf-8") as f:
            f.write(f"n = {len(times_sec)}\n")
            f.write(f"mean_sec = {mean_sec:.6f}\n")
            f.write(f"std_sec = {std_sec:.6f}\n")
            f.write(f"formatted = {time_text}\n")
    print(f"\nDone: {n_ok}/{len(results)} succeeded. Report: {batch_report}")
    return results, case_out


def extract_frame(mp4_path, frame_index, out_png):
    reader = imageio.get_reader(mp4_path, format="ffmpeg")
    frame = reader.get_data(frame_index)
    reader.close()
    Image.fromarray(np.asarray(frame)).save(out_png, format="PNG", compress_level=0)
    print(f"Saved frame {frame_index} -> {out_png}")