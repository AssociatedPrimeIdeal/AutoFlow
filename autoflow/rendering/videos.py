import os

import imageio.v2 as imageio
import numpy as np
import pyvista as pv
from PIL import Image

from ..algorithms import create_uniform_grid, generate_seed_points, generate_streamlines_at_t

WINDOW_SIZE = (1600, 1200)
_OFFSCREEN_BOOTSTRAPPED = False

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


def _offscreen_mode():
    mode = str(os.environ.get("AUTOFLOW_OFFSCREEN_MODE", "local")).strip().lower()
    if mode in {"display", "x11", "onscreen"}:
        return "display"
    if mode in {"local", "headless", "xvfb"}:
        return "local"

    display = str(os.environ.get("DISPLAY", "")).strip().lower()
    if not display:
        return "local"
    if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_CLIENT") or os.environ.get("SSH_TTY"):
        if display.startswith("localhost:") or display.startswith("localhost/unix:") or display.startswith("127.0.0.1:"):
            return "local"
    return "display"


def _normalize(v):
    arr = np.asarray(v, dtype=float).reshape(3)
    n = np.linalg.norm(arr)
    if n <= 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return arr / n


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
    global _OFFSCREEN_BOOTSTRAPPED
    if _OFFSCREEN_BOOTSTRAPPED:
        return
    _OFFSCREEN_BOOTSTRAPPED = True
    os.environ["PYVISTA_OFF_SCREEN"] = "true"
    os.environ.pop("DISPLAY", None)
    try:
        if hasattr(pv, "start_xvfb"):
            pv.start_xvfb()
    except Exception:
        pass


def _make_plotter(window_size=WINDOW_SIZE):
    _ensure_offscreen()
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background("white")
    return plotter


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
    bounds = np.array(poly.bounds, dtype=float).reshape(3, 2)
    center = bounds.mean(axis=1)
    extent = np.maximum(bounds[:, 1] - bounds[:, 0], 1.0)
    radius = float(max(np.linalg.norm(extent) * 1.2, 50.0))
    return center, radius


def _resolve_view(view):
    if view is None:
        return CAMERA_PRESETS["iso"]
    if isinstance(view, str):
        if view not in CAMERA_PRESETS:
            raise ValueError(f"unknown camera preset: {view}, options: {list(CAMERA_PRESETS)}")
        return CAMERA_PRESETS[view]
    azimuth_deg, elevation_deg = view
    return float(azimuth_deg), float(elevation_deg)


def _camera_from_scene(poly, azimuth_deg=35.0, elevation_deg=25.0, distance_scale=1.0):
    center, radius = _surface_center_radius(poly)
    radius = radius * float(distance_scale)
    azimuth = np.deg2rad(float(azimuth_deg))
    elevation = np.deg2rad(float(elevation_deg))
    pos = center + np.array(
        [
            radius * np.cos(elevation) * np.cos(azimuth),
            radius * np.cos(elevation) * np.sin(azimuth),
            radius * np.sin(elevation),
        ],
        dtype=float,
    )
    if abs(elevation_deg) > 80.0:
        up = (0.0, 1.0, 0.0)
    else:
        up = (0.0, 0.0, 1.0)
    return [tuple(pos.tolist()), tuple(center.tolist()), up]


def _orbit_camera(poly, azimuth_deg, elevation_deg=25.0, distance_scale=1.0):
    return _camera_from_scene(poly, azimuth_deg, elevation_deg, distance_scale)


def _camera_from_view(poly, view, distance_scale=1.0):
    azimuth_deg, elevation_deg = _resolve_view(view)
    return _camera_from_scene(poly, azimuth_deg, elevation_deg, distance_scale)


def _time_and_azimuth(frame_idx, rotation_frames, n_time, time_repeat=1):
    rotation_frames = int(max(rotation_frames, 1))
    n_time = int(max(n_time, 1))
    time_repeat = int(max(time_repeat, 1))

    t = (frame_idx // time_repeat) % n_time
    azimuth_deg = 360.0 * (frame_idx % rotation_frames) / rotation_frames
    return t, azimuth_deg


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
    bounds = np.array(surf.bounds, dtype=float).reshape(3, 2)
    extent = bounds[:, 1] - bounds[:, 0]
    return float(max(12.0, 0.12 * np.max(extent)))


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
    plotter = _make_plotter()
    plotter.add_mesh(surf, opacity=0.18, color="white")

    paths_world = []
    for path in ws.centerline_paths_smooth:
        path_world = np.asarray(path, dtype=float) + origin.reshape(1, 3)
        paths_world.append(path_world)
        poly = _path_polydata(path_world)
        if poly is not None and poly.n_points > 0:
            plotter.add_mesh(poly, color="deepskyblue", line_width=5, render_lines_as_tubes=True)

    centers = []
    plane_labels = []
    for i, plane in enumerate(ws.planes):
        center_world = np.asarray(plane.center, dtype=float).reshape(3) + origin
        plane_mesh = _plane_mesh(center_world, plane.normal, plane_size)
        plotter.add_mesh(plane_mesh, color="yellow", opacity=0.75, show_edges=True, edge_color="black", line_width=2)
        centers.append(center_world)
        plane_labels.append(f"Plane {i}")

    if add_plane_idx and centers:
        plotter.add_point_labels(
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

    if add_path_idx and paths_world:
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
            n_points = len(path_world)
            frac = frac_choices[idx % len(frac_choices)]
            k = min(max(int(frac * (n_points - 1)), 0), n_points - 1)
            anchor = np.asarray(path_world[k], dtype=float) + offsets[idx % len(offsets)]
            path_label_points.append(anchor)
            path_label_texts.append(f"Branch {idx}")

        if path_label_points:
            plotter.add_point_labels(
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
        azimuth_deg = 360.0 * frame_idx / max(n_frames, 1)
        plotter.camera_position = _orbit_camera(surf, azimuth_deg, elevation_deg, distance_scale)
        plotter.add_text(
            f"Rotating {frame_idx + 1}/{int(max(n_frames, 1))}",
            position="upper_left",
            font_size=14,
            color="black",
            name="frame_text",
        )
        plotter.render()
        frames.append(np.asarray(plotter.screenshot(return_img=True)))
        try:
            plotter.remove_actor("frame_text")
        except Exception:
            pass
    plotter.close()
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
    time_repeat=1,
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

    _, default_elevation_deg = _resolve_view(view)
    if elevation_deg is None:
        elevation_deg = default_elevation_deg

    n_time = int(max(ws.time_count(), 1))
    if rotate:
        base_frames = n_time * int(max(time_repeat, 1))
        if rotation_frames is not None:
            total_frames = max(int(rotation_frames), base_frames)
        else:
            total_frames = base_frames
    else:
        total_frames = n_time * int(max(time_repeat, 1))

    plotter = _make_plotter()
    frames = []

    for frame_idx in range(total_frames):
        if rotate:
            t, azimuth_deg = _time_and_azimuth(
                frame_idx,
                rotation_frames=rotation_frames if rotation_frames is not None else total_frames,
                n_time=n_time,
                time_repeat=time_repeat,
            )
            camera_position = _orbit_camera(context_surf, azimuth_deg, elevation_deg, distance_scale)
        else:
            t = min(frame_idx, n_time - 1)
            camera_position = _camera_from_view(context_surf, view, distance_scale)

        plotter.clear()
        plotter.set_background("white")
        plotter.add_mesh(context_surf, opacity=0.08, color="white")

        surf = ws.derived.wss_surfaces[min(max(0, t), len(ws.derived.wss_surfaces) - 1)]
        if surf is not None and surf.n_points > 0 and "wss" in surf.point_data:
            plotter.add_mesh(
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

        plotter.add_text(txt, position="upper_left", font_size=14, color="black")
        plotter.camera_position = camera_position
        plotter.render()
        frames.append(np.asarray(plotter.screenshot(return_img=True)))

    plotter.close()
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
    time_repeat=1,
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
        min_seeds=ws.streamline_params.min_seeds,
    )

    v_max = _streamline_speed_max(ws)
    clim = streamline_clim if streamline_clim is not None else (0.0, v_max)

    _, default_elevation_deg = _resolve_view(view)
    if elevation_deg is None:
        elevation_deg = default_elevation_deg

    n_time = int(max(ws.time_count(), 1))
    if rotate:
        base_frames = n_time * int(max(time_repeat, 1))
        if rotation_frames is not None:
            total_frames = max(int(rotation_frames), base_frames)
        else:
            total_frames = base_frames
    else:
        total_frames = n_time * int(max(time_repeat, 1))

    plotter = _make_plotter()
    frames = []

    for frame_idx in range(total_frames):
        if rotate:
            t, azimuth_deg = _time_and_azimuth(
                frame_idx,
                rotation_frames=rotation_frames if rotation_frames is not None else total_frames,
                n_time=n_time,
                time_repeat=time_repeat,
            )
            camera_position = _orbit_camera(surf, azimuth_deg, elevation_deg, distance_scale)
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
            min_seeds=ws.streamline_params.min_seeds,
            rng_seed=ws.streamline_params.rng_seed,
        )
        sl = _ensure_streamline_scalars(sl)

        plotter.clear()
        plotter.set_background("white")
        plotter.add_mesh(surf, opacity=0.18, color="lightgray")

        if sl is not None and sl.n_points > 0:
            sl_show = sl
            render_lines_as_tubes = True
            if float(ws.streamline_params.tube_radius) > 0.0 and hasattr(sl, "tube"):
                sl_show = sl.tube(radius=float(ws.streamline_params.tube_radius))
                render_lines_as_tubes = False
            plotter.add_mesh(
                sl_show,
                scalars="Velocity",
                cmap="turbo",
                clim=clim,
                show_scalar_bar=True,
                scalar_bar_args=_scalar_bar_args("Velocity (m/s)", streamline_bar_cfg),
                render_lines_as_tubes=render_lines_as_tubes,
                line_width=3,
            )

        if rotate:
            txt = f"t={t} | rot {frame_idx + 1}/{total_frames}"
        else:
            txt = f"t={t}"

        plotter.add_text(txt, position="upper_left", font_size=14, color="black")
        plotter.camera_position = camera_position
        plotter.render()
        frames.append(np.asarray(plotter.screenshot(return_img=True)))

    plotter.close()
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
    time_repeat=1,
):
    if ws.derived.tke_array is None and ws.derived.tke_volume is None:
        return None

    _, surf = _build_union_surface(ws, smoothing_iteration=smoothing_iteration)
    if surf is None or surf.n_points == 0:
        return None

    tke_max = _tke_max(ws)
    clim = tke_clim if tke_clim is not None else (0.0, tke_max)

    _, default_elevation_deg = _resolve_view(view)
    if elevation_deg is None:
        elevation_deg = default_elevation_deg

    n_time = int(max(ws.time_count(), 1))
    if rotate:
        base_frames = n_time * int(max(time_repeat, 1))
        if rotation_frames is not None:
            total_frames = max(int(rotation_frames), base_frames)
        else:
            total_frames = base_frames
    else:
        total_frames = n_time * int(max(time_repeat, 1))

    plotter = _make_plotter()
    frames = []

    for frame_idx in range(total_frames):
        if rotate:
            t, azimuth_deg = _time_and_azimuth(
                frame_idx,
                rotation_frames=rotation_frames if rotation_frames is not None else total_frames,
                n_time=n_time,
                time_repeat=time_repeat,
            )
            camera_position = _orbit_camera(surf, azimuth_deg, elevation_deg, distance_scale)
        else:
            t = min(frame_idx, n_time - 1)
            camera_position = _camera_from_view(surf, view, distance_scale)

        plotter.clear()
        plotter.set_background("white")
        plotter.add_mesh(surf, opacity=0.08, color="white")

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
            plotter.add_mesh(
                tke_mesh,
                scalars="TKE",
                cmap="hot",
                clim=clim,
                show_scalar_bar=True,
                scalar_bar_args=_scalar_bar_args("TKE (J/m³)", tke_bar_cfg),
            )
        else:
            plotter.add_mesh(
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

        plotter.add_text(txt, position="upper_left", font_size=14, color="black")
        plotter.camera_position = camera_position
        plotter.render()
        frames.append(np.asarray(plotter.screenshot(return_img=True)))

    plotter.close()
    suffix = "rotate" if rotate else "video"
    return _write_video(frames, os.path.join(out_dir, f"tke_{suffix}.mp4"), fps=fps)


def extract_frame(mp4_path, frame_index, out_png):
    reader = imageio.get_reader(mp4_path, format="ffmpeg")
    frame = reader.get_data(frame_index)
    reader.close()
    Image.fromarray(np.asarray(frame)).save(out_png, format="PNG", compress_level=0)
    print(f"Saved frame {frame_index} -> {out_png}")
