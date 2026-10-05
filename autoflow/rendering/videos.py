from concurrent.futures import ThreadPoolExecutor
import os
import sys
import uuid
import stat
import inspect
from functools import wraps

import imageio.v2 as imageio
import numpy as np
import pyvista as pv
from PIL import Image
from ..task_control import TaskCancelled, check_cancelled, report_progress

from ..algorithms import (
    automatic_streamline_clim,
    build_cell_mask_surface,
    create_uniform_grid,
    generate_seed_points,
    generate_streamlines_at_t,
)
from ..config import DEFAULT_PLANE_VIDEO_CFG, BACKGROUND_COLOR, CONTEXT_COLOR
from .style import (
    configure_plotter, foreground_color, plane_display_size,
    metric_style, scalar_bar_args, scene_object, surface_style, configured_metric_style, render_style_settings,
    wss_display_surface, wss_scalar_range, metric_volume_kwargs, apply_metric_volume_opacity, metric_clim,
)
from .datasets import tke_display_mesh, display_support_surface, sample_display_field

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
    # GUI exports already own a live display-backed VTK context, so keep that
    # path instead of forcing a separate local EGL/Xvfb context.
    qtwidgets = sys.modules.get("PySide6.QtWidgets")
    if qtwidgets is not None:
        app_cls = getattr(qtwidgets, "QApplication", None)
        try:
            if app_cls is not None and app_cls.instance() is not None:
                return "display"
        except Exception:
            pass
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
    return scalar_bar_args(title, feature_cfg=bar_cfg)


def _video_background(ws):
    return (getattr(ws, "render_settings", {}) or {}).get("render_background_color") or BACKGROUND_COLOR


def _ensure_offscreen():
    global _OFFSCREEN_BOOTSTRAPPED
    if _OFFSCREEN_BOOTSTRAPPED:
        return
    _OFFSCREEN_BOOTSTRAPPED = True
    os.environ["PYVISTA_OFF_SCREEN"] = "true"
    if _offscreen_mode() == "local":
        os.environ.pop("DISPLAY", None)
        try:
            if hasattr(pv, "start_xvfb"):
                pv.start_xvfb()
        except Exception:
            pass


def _make_plotter(window_size=WINDOW_SIZE, background=BACKGROUND_COLOR, workspace=None):
    _ensure_offscreen()
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    configure_plotter(plotter, background, workspace=workspace)
    return plotter


def _resolve_window_size(window_size=None):
    if not isinstance(window_size, (list, tuple)) or len(window_size) < 2:
        return WINDOW_SIZE
    try:
        return (max(int(window_size[0]), 1), max(int(window_size[1]), 1))
    except Exception:
        return WINDOW_SIZE


def _scalar_bar_mesh_kwargs(show_scalar_bar, title, bar_cfg=None, ws=None):
    settings = getattr(ws, "render_settings", {}) or {}
    kwargs = {"show_scalar_bar": bool(show_scalar_bar) and bool(settings.get("shared_colorbar_show", True))}
    if kwargs["show_scalar_bar"]:
        kwargs["scalar_bar_args"] = scalar_bar_args(
            title, bar_cfg, settings.get("shared_colorbar_bar_cfg"), _video_background(ws), workspace=ws,
        )
    return kwargs


def _add_context_surfaces(plotter, ws, union_surface, smoothing_iteration):
    groups = _build_group_surfaces(ws, smoothing_iteration=smoothing_iteration)
    context = render_style_settings(ws)["context"]
    if groups:
        for name, surface, color in groups:
            obj = scene_object(ws, f"segmask_group_{name}")
            plotter.add_mesh(surface, color=obj.color if obj is not None else color,
                             opacity=obj.opacity if obj is not None else context["opacity"],
                             show_scalar_bar=False, **surface_style(workspace=ws))
    else:
        plotter.add_mesh(union_surface, color=context["color"], opacity=context["opacity"],
                         show_scalar_bar=False, **surface_style(workspace=ws))


def _write_video(frames, out_path, fps=24):
    """Encode a list or replayable frame factory with bounded frame memory."""
    if not callable(frames) and not frames:
        return None
    out_path = os.path.splitext(str(out_path))[0] + ".mp4"
    directory = os.path.dirname(out_path) or "."
    os.makedirs(directory, exist_ok=True)
    attempts = [{"codec": "libx264", "macro_block_size": None},
                {"codec": "mpeg4", "macro_block_size": None},
                {"macro_block_size": None}]
    last_error = None
    for writer_kwargs in attempts:
        check_cancelled()
        temporary = os.path.join(directory, f".autoflow_video_{uuid.uuid4().hex}.mp4")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        os.close(fd)
        iterator = iter(frames() if callable(frames) else frames)
        try:
            count = 0
            with imageio.get_writer(temporary, format="ffmpeg", fps=fps, **writer_kwargs) as writer:
                for frame in iterator:
                    check_cancelled()
                    writer.append_data(np.asarray(frame))
                    count += 1
            check_cancelled()
            if count == 0:
                return None
            if os.path.isfile(out_path):
                os.chmod(temporary, stat.S_IMODE(os.stat(out_path).st_mode))
            os.replace(temporary, out_path)
            return out_path
        except TaskCancelled:
            raise
        except Exception as exc:
            last_error = exc
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
            if os.path.exists(temporary):
                os.remove(temporary)
    raise RuntimeError(f"failed to write MP4 video: {out_path}") from last_error


def _stream_video(name):
    def decorate(function):
        signature = inspect.signature(function)
        @wraps(function)
        def render(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            suffix = "rotate" if bound.arguments.get("rotate", False) else "video"
            filename = "planes_rotate.mp4" if name == "planes" else f"{name}_{suffix}.mp4"
            return _write_video(lambda: function(*args, **kwargs),
                                os.path.join(bound.arguments["out_dir"], filename),
                                fps=bound.arguments["fps"])
        return render
    return decorate


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
    cache_key = (
        id(getattr(ws, "segmask_binary", None)),
        id(getattr(ws, "segmask_3d", None)),
        int(smoothing_iteration),
        tuple(np.asarray(ws.resolution, dtype=float).reshape(3)),
        tuple(np.asarray(ws.origin, dtype=float).reshape(3)),
    )
    cache = getattr(ws, "_video_union_surface_cache", {})
    if cache_key in cache:
        return cache[cache_key]
    if ws.segmask_binary is not None:
        mask3d = np.any(np.asarray(ws.segmask_binary, dtype=bool), axis=3)
    else:
        mask3d = np.asarray(ws.segmask_3d, dtype=bool)
    mesh = create_uniform_grid(mask3d, ws.resolution, origin=ws.origin)
    mesh = mesh.threshold(0.1)
    if mesh is None or mesh.n_cells == 0:
        return None, None
    try:
        surf = mesh.extract_surface(algorithm="dataset_surface")
    except TypeError as exc:
        # Older PyVista releases do not expose the ``algorithm`` kwarg.
        if "algorithm" not in str(exc):
            raise
        surf = mesh.extract_surface()
    if surf is not None and surf.n_points > 0 and int(smoothing_iteration) > 0:
        surf = surf.smooth(n_iter=int(smoothing_iteration))
    result = (mesh, surf)
    cache[cache_key] = result
    ws._video_union_surface_cache = cache
    return result


def _build_group_surfaces(ws, smoothing_iteration=200):
    """Build GUI-style per-label surfaces for grouped plane exports."""
    group_order = tuple(str(name) for name in (getattr(ws, "group_order", []) or []))
    if not group_order:
        return []
    cache_key = (
        id(getattr(ws, "segmask_3d", None)),
        group_order,
        int(smoothing_iteration),
        tuple(np.asarray(ws.resolution, dtype=float).reshape(3)),
        tuple(np.asarray(ws.origin, dtype=float).reshape(3)),
    )
    cache = getattr(ws, "_video_group_surface_cache", {})
    if cache_key in cache:
        return cache[cache_key]

    entries = []
    groups = getattr(ws, "multilabel_groups", {}) or {}
    for group_name in group_order:
        state = groups.get(group_name, {}) or {}
        mask = state.get("segmask_3d")
        if mask is None or not np.any(mask):
            continue
        surface = build_cell_mask_surface(
            np.asarray(mask, dtype=bool),
            ws.resolution,
            origin=ws.origin,
            smooth_iter=smoothing_iteration,
        )
        if surface is None or surface.n_points == 0:
            continue
        try:
            color = ws.skeleton_params.scene_color_for_group(group_name, "scene")
        except Exception:
            color = state.get("scene_color") or state.get("browser_color") or "lightgray"
        entries.append((group_name, surface, str(color or "lightgray")))

    cache[cache_key] = entries
    ws._video_group_surface_cache = cache
    return entries


def _pressure_support_surface(ws, support_source, t, smooth_iter=80):
    support_key = (
        id(support_source),
        tuple(int(x) for x in np.asarray(support_source).shape),
        tuple(np.asarray(ws.resolution, dtype=float).reshape(3)),
        tuple(np.asarray(ws.origin, dtype=float).reshape(3)),
        int(smooth_iter),
    )
    cache = getattr(ws, "_pressure_support_surface_cache", {})
    entry = cache.get(support_key)
    if entry is None:
        support = np.asarray(support_source, dtype=bool)
        if support.ndim == 4:
            representatives = {}
            lookup = []
            for tidx in range(int(support.shape[3])):
                token = np.ascontiguousarray(support[..., tidx]).tobytes()
                lookup.append(representatives.setdefault(token, int(tidx)))
        else:
            lookup = [0]
        entry = {"support": support, "lookup": lookup, "surfaces": {}}
        cache[support_key] = entry
        ws._pressure_support_surface_cache = cache

    support = entry["support"]
    lookup = entry["lookup"]
    tidx = min(max(0, int(t)), len(lookup) - 1)
    rep_t = int(lookup[tidx])
    surfaces = entry["surfaces"]
    if rep_t not in surfaces:
        support_t = support[..., rep_t] if support.ndim == 4 else support
        surfaces[rep_t] = display_support_surface(
            support_t,
            ws.resolution,
            origin=ws.origin,
            smooth_iter=smooth_iter,
        )
    return surfaces[rep_t]


def _path_group_name(ws, path_idx):
    idx = int(path_idx)
    path_info = list(getattr(ws, "path_info", []) or [])
    if 0 <= idx < len(path_info):
        info = path_info[idx]
        if isinstance(info, dict):
            group_name = str(info.get("group_name", "") or "")
            if group_name:
                return group_name
    for group_name in list(getattr(ws, "group_order", []) or []):
        group_state = dict(getattr(ws, "multilabel_groups", {}).get(group_name, {}) or {})
        start = int(group_state.get("path_index_offset", -1))
        local_paths = list(group_state.get("centerline_paths_smooth", []) or group_state.get("centerline_paths", []) or [])
        if start >= 0 and start <= idx < start + len(local_paths):
            return str(group_name)
    return ""


def _path_color(ws, path_idx):
    group_name = _path_group_name(ws, path_idx)
    if group_name and hasattr(ws, "skeleton_params"):
        try:
            return ws.skeleton_params.scene_color_for_group(group_name, "path")
        except Exception:
            pass
    return "deepskyblue"


def _plane_group_name(ws, plane):
    group_name = str(getattr(plane, "group_name", "") or "")
    if group_name:
        return group_name
    return _path_group_name(ws, getattr(plane, "path_index", -1))


def _plane_video_style(ws, group_name, plane_video_cfg, default_plane_size):
    cfg = plane_video_cfg if isinstance(plane_video_cfg, dict) else {}
    default_cfg = cfg.get("default", {}) if isinstance(cfg.get("default"), dict) else {}
    groups_cfg = cfg.get("groups", {}) if isinstance(cfg.get("groups"), dict) else {}
    group_cfg = groups_cfg.get(str(group_name), {}) if group_name and isinstance(groups_cfg.get(str(group_name)), dict) else {}
    merged = dict(default_cfg)
    merged.update(group_cfg)

    skeleton_color = str(merged.get("skeleton_color", "") or "")
    plane_color = str(merged.get("plane_color", "") or "")
    if group_name and hasattr(ws, "skeleton_params"):
        if not skeleton_color:
            try:
                skeleton_color = str(ws.skeleton_params.scene_color_for_group(group_name, "skeleton") or "")
            except Exception:
                skeleton_color = ""
        if not plane_color:
            try:
                plane_color = str(ws.skeleton_params.scene_color_for_group(group_name, "plane") or "")
            except Exception:
                plane_color = ""
    if not skeleton_color:
        skeleton_color = "#ff922b"
    if not plane_color:
        plane_color = "yellow"

    plane_size = merged.get("plane_size", None)
    try:
        plane_size = float(plane_size)
        if plane_size <= 0.0:
            plane_size = float(default_plane_size)
    except Exception:
        plane_size = float(default_plane_size)

    plane_opacity = merged.get("plane_opacity", 0.75)
    try:
        plane_opacity = float(plane_opacity)
    except Exception:
        plane_opacity = 0.75
    plane_opacity = max(0.0, min(1.0, plane_opacity))

    return {
        "skeleton_color": skeleton_color,
        "plane_color": plane_color,
        "plane_size": plane_size,
        "plane_opacity": plane_opacity,
    }


def _plane_label_style(plane_video_cfg):
    cfg = plane_video_cfg if isinstance(plane_video_cfg, dict) else {}
    default_cfg = DEFAULT_PLANE_VIDEO_CFG.get("label", {})
    label_cfg = cfg.get("label", {}) if isinstance(cfg.get("label"), dict) else {}
    merged = dict(default_cfg)
    merged.update(label_cfg)

    prefix_value = merged.get("prefix", "planeidx=")
    prefix = "" if prefix_value is None else str(prefix_value)

    try:
        font_size = int(merged.get("font_size", 28))
    except Exception:
        font_size = 28
    font_size = max(1, font_size)

    text_color = str(merged.get("text_color", "black") or "black")
    shape_color = str(merged.get("shape_color", "yellow") or "yellow")
    try:
        shape_opacity = float(merged.get("shape_opacity", 0.85))
    except Exception:
        shape_opacity = 0.85
    shape_opacity = max(0.0, min(1.0, shape_opacity))

    return {
        "prefix": prefix,
        "font_size": font_size,
        "text_color": text_color,
        "shape_color": shape_color,
        "shape_opacity": shape_opacity,
    }


@_stream_video("planes")
def render_plane_rotation_video(
    ws,
    out_dir,
    fps=24,
    n_frames=180,
    smoothing_iteration=200,
    elevation_deg=0.0,
    distance_scale=1.0,
    add_plane_idx=True,
    add_path_idx=False,
    plane_video_cfg=None,
    window_size=None,
):
    _, surf = _build_union_surface(ws, smoothing_iteration=smoothing_iteration)
    if surf is None or surf.n_points == 0:
        return None

    cfg = plane_video_cfg if isinstance(plane_video_cfg, dict) else {}
    base_cfg = {
        "show_skeleton": bool(cfg.get("show_skeleton", DEFAULT_PLANE_VIDEO_CFG.get("show_skeleton", True))),
        "skeleton_point_size": cfg.get("skeleton_point_size", DEFAULT_PLANE_VIDEO_CFG.get("skeleton_point_size", 10.0)),
        "label": dict(DEFAULT_PLANE_VIDEO_CFG.get("label", {})),
        "default": dict(DEFAULT_PLANE_VIDEO_CFG.get("default", {})),
        "groups": dict(cfg.get("groups", {})) if isinstance(cfg.get("groups"), dict) else {},
    }
    if isinstance(cfg.get("default"), dict):
        base_cfg["default"].update(cfg.get("default", {}))
    if isinstance(cfg.get("label"), dict):
        base_cfg["label"].update(cfg.get("label", {}))

    try:
        skeleton_point_size = float(base_cfg.get("skeleton_point_size", 10.0))
    except Exception:
        skeleton_point_size = 10.0
    skeleton_point_size = max(1.0, skeleton_point_size)

    label_style = _plane_label_style(base_cfg)

    default_plane_size = plane_display_size(ws)
    origin = np.asarray(ws.origin, dtype=float).reshape(3)
    plotter = _make_plotter(window_size=_resolve_window_size(window_size), background=_video_background(ws), workspace=ws)
    try:
        _add_context_surfaces(plotter, ws, surf, smoothing_iteration)

        if bool(base_cfg.get("show_skeleton", True)):
            if list(getattr(ws, "group_order", []) or []):
                for group_name in list(getattr(ws, "group_order", []) or []):
                    group_state = dict(getattr(ws, "multilabel_groups", {}).get(group_name, {}) or {})
                    pts = group_state.get("skeleton_points")
                    if pts is None or len(pts) == 0:
                        continue
                    style = _plane_video_style(ws, str(group_name), base_cfg, default_plane_size)
                    plotter.add_mesh(
                        pv.PolyData(np.asarray(pts, dtype=float) + origin.reshape(1, 3)),
                        color=style["skeleton_color"],
                        point_size=skeleton_point_size,
                        render_points_as_spheres=True,
                    )
            elif ws.skeleton_points is not None and len(ws.skeleton_points) > 0:
                style = _plane_video_style(ws, "", base_cfg, default_plane_size)
                plotter.add_mesh(
                    pv.PolyData(np.asarray(ws.skeleton_points, dtype=float) + origin.reshape(1, 3)),
                    color=style["skeleton_color"],
                    point_size=skeleton_point_size,
                    render_points_as_spheres=True,
                )

        paths_world = []
        for path_idx, path in enumerate(ws.centerline_paths_smooth):
            path_world = np.asarray(path, dtype=float) + origin.reshape(1, 3)
            paths_world.append(path_world)
            poly = _path_polydata(path_world)
            if poly is not None and poly.n_points > 0:
                plotter.add_mesh(
                    poly,
                    color=_path_color(ws, path_idx),
                    line_width=5,
                    render_lines_as_tubes=True,
                )

        centers = []
        plane_labels = []
        for i, plane in enumerate(ws.planes):
            style = _plane_video_style(ws, _plane_group_name(ws, plane), base_cfg, default_plane_size)
            center_world = np.asarray(plane.center, dtype=float).reshape(3) + origin
            plane_mesh = _plane_mesh(center_world, plane.normal, style["plane_size"])
            plotter.add_mesh(
                plane_mesh,
                color=style["plane_color"],
                opacity=style["plane_opacity"],
                style="wireframe", lighting=False,
                line_width=4,
            )
            centers.append(center_world)
            plane_labels.append(f'{label_style["prefix"]}{i}')

        if add_plane_idx and centers:
            plotter.add_point_labels(
                np.asarray(centers, dtype=float),
                plane_labels,
                font_size=label_style["font_size"],
                bold=True,
                text_color=label_style["text_color"],
                fill_shape=True,
                shape="rounded_rect",
                shape_color=label_style["shape_color"],
                shape_opacity=label_style["shape_opacity"],
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

        total_frames = int(max(n_frames, 1))
        for frame_idx in range(total_frames):
            check_cancelled()
            azimuth_deg = 360.0 * frame_idx / max(n_frames, 1)
            plotter.camera_position = _orbit_camera(surf, azimuth_deg, elevation_deg, distance_scale)
            plotter.add_text(
                f"Rotating {frame_idx + 1}/{int(max(n_frames, 1))}",
                position="upper_left",
                font_size=14,
                color=foreground_color(_video_background(ws), ws),
                name="frame_text",
            )
            plotter.render()
            check_cancelled()
            yield np.asarray(plotter.screenshot(return_img=True))
            report_progress({"stage": "video_frame", "current": frame_idx + 1, "total": total_frames,
                             "message": f"Rendered frame {frame_idx + 1}/{total_frames}"})
            try:
                plotter.remove_actor("frame_text")
            except Exception:
                pass
    finally:
        plotter.close()


@_stream_video("wss")
def render_wss_video(
    ws,
    out_dir,
    fps=24,
    smoothing_iteration=200,
    view="iso",
    distance_scale=1.0,
    wss_clim=None,
    wss_bar_cfg=None,
    show_scalar_bar=True,
    rotate=False,
    rotation_frames=None,
    elevation_deg=None,
    time_repeat=1,
    window_size=None,
):
    if not ws.derived.wss_surfaces:
        return None

    _, context_surf = _build_union_surface(ws, smoothing_iteration=smoothing_iteration)
    if context_surf is None or context_surf.n_points == 0:
        return None

    clim = metric_clim(ws, "wss_surface_live", wss_clim, wss_scalar_range(ws.derived.wss_surfaces))

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

    plotter = _make_plotter(window_size=_resolve_window_size(window_size), background=_video_background(ws), workspace=ws)
    try:
        wss_display_cache = {}
        wss_actor = None
        text_actor = plotter.add_text("", position="upper_left", font_size=14,
                                     color=foreground_color(_video_background(ws), ws))
        for frame_idx in range(total_frames):
            check_cancelled()
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

            phase = min(max(0, t), len(ws.derived.wss_surfaces) - 1)
            if phase not in wss_display_cache:
                wss_display_cache[phase] = wss_display_surface(ws.derived.wss_surfaces[phase])
            surface = wss_display_cache[phase]
            if surface is not None and surface.n_points > 0 and "wss" in surface.point_data:
                if wss_actor is None:
                    wss_actor = plotter.add_mesh(
                        surface, scalars="wss", clim=clim,
                        **metric_style(ws, "wss_surface_live"),
                        **_scalar_bar_mesh_kwargs(show_scalar_bar, "WSS (Pa)", wss_bar_cfg, ws=ws),
                    )
                else:
                    wss_actor.SetVisibility(1)
                    wss_actor.GetMapper().dataset = surface
                    wss_actor.GetMapper().Update()
            elif wss_actor is not None:
                wss_actor.SetVisibility(0)

            if rotate:
                txt = f"t={t} | rot {frame_idx + 1}/{total_frames}"
            else:
                txt = f"t={t}"

            text_actor.SetText(2, txt)
            plotter.camera_position = camera_position
            plotter.render()
            check_cancelled()
            yield np.asarray(plotter.screenshot(return_img=True))
            report_progress({"stage": "video_frame", "current": frame_idx + 1, "total": total_frames,
                             "message": f"Rendered frame {frame_idx + 1}/{total_frames}"})

    finally:
        plotter.close()


def _streamline_speed_max(ws):
    return automatic_streamline_clim(ws.flow_raw, ws.segmask_binary)[1]


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


@_stream_video("streamlines")
def render_streamlines_video(
    ws,
    out_dir,
    fps=24,
    smoothing_iteration=200,
    view="iso",
    distance_scale=1.0,
    streamline_clim=None,
    streamline_bar_cfg=None,
    show_scalar_bar=True,
    rotate=False,
    rotation_frames=None,
    elevation_deg=None,
    time_repeat=1,
    window_size=None,
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
    clim = metric_clim(ws, "streamlines_live", streamline_clim, (0.0, v_max))

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

    def _build_streamline(tidx):
        mask_t = np.asarray(
            ws.segmask_binary[..., min(max(0, tidx), ws.segmask_binary.shape[3] - 1)],
            dtype=bool,
        )
        return tidx, _ensure_streamline_scalars(
            generate_streamlines_at_t(
                ws.flow_raw,
                tidx,
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
        )

    worker_count = min(n_time, 8, max(1, os.cpu_count() or 1))
    if worker_count > 1:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            streamline_cache = dict(pool.map(_build_streamline, range(n_time)))
    else:
        streamline_cache = dict(_build_streamline(tidx) for tidx in range(n_time))

    plotter = _make_plotter(window_size=_resolve_window_size(window_size), background=_video_background(ws), workspace=ws)
    try:
        _add_context_surfaces(plotter, ws, surf, smoothing_iteration)
        streamline_actor = None
        text_actor = plotter.add_text("", position="upper_left", font_size=14, color=foreground_color(_video_background(ws), ws))

        for frame_idx in range(total_frames):
            check_cancelled()
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

            sl = streamline_cache[t]

            if sl is not None and sl.n_points > 0:
                sl_show = sl
                if sl_show is not None:
                    if streamline_actor is None:
                        streamline_actor = plotter.add_mesh(
                            sl_show,
                            scalars="Velocity",
                            clim=clim,
                            render_lines_as_tubes=True,
                            line_width=getattr(scene_object(ws, "streamlines_live"), "line_width", configured_metric_style(ws, "streamlines_live")["line_width"]),
                            **metric_style(ws, "streamlines_live"),
                            **_scalar_bar_mesh_kwargs(show_scalar_bar, "Velocity (m/s)", streamline_bar_cfg, ws=ws),
                        )
                    else:
                        streamline_actor.SetVisibility(1)
                        mapper = streamline_actor.GetMapper()
                        mapper.dataset = sl_show
                        mapper.Update()
            elif streamline_actor is not None:
                streamline_actor.SetVisibility(0)

            if rotate:
                txt = f"t={t} | rot {frame_idx + 1}/{total_frames}"
            else:
                txt = f"t={t}"

            try:
                text_actor.SetText(2, txt)
            except Exception:
                try:
                    text_actor.SetInput(txt)
                except Exception:
                    pass
            plotter.camera_position = camera_position
            plotter.render()
            check_cancelled()
            yield np.asarray(plotter.screenshot(return_img=True))
            report_progress({"stage": "video_frame", "current": frame_idx + 1, "total": total_frames,
                             "message": f"Rendered frame {frame_idx + 1}/{total_frames}"})

    finally:
        plotter.close()


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


@_stream_video("tke")
def render_tke_video(
    ws,
    out_dir,
    fps=24,
    smoothing_iteration=200,
    view="iso",
    distance_scale=1.0,
    tke_clim=None,
    tke_bar_cfg=None,
    show_scalar_bar=True,
    rotate=False,
    rotation_frames=None,
    elevation_deg=None,
    time_repeat=1,
    window_size=None,
):
    if ws.derived.tke_array is None and ws.derived.tke_volume is None:
        return None

    _, surf = _build_union_surface(ws, smoothing_iteration=smoothing_iteration)
    if surf is None or surf.n_points == 0:
        return None

    tke_max = _tke_max(ws)
    clim = metric_clim(ws, "tke_volume", tke_clim, (0.0, tke_max))

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

    plotter = _make_plotter(window_size=_resolve_window_size(window_size), background=_video_background(ws), workspace=ws)
    try:
        tke_mesh_cache = {}
        tke_actor = None
        text_actor = plotter.add_text("", position="upper_left", font_size=14,
                                     color=foreground_color(_video_background(ws), ws))

        for frame_idx in range(total_frames):
            check_cancelled()
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

            if t not in tke_mesh_cache:
                tke_mesh_cache[t] = tke_display_mesh(ws, t)
            tke_mesh = tke_mesh_cache[t]
            if tke_mesh is not None and tke_mesh.n_points > 0:
                if tke_actor is None:
                    tke_actor = plotter.add_volume(
                        tke_mesh, scalars="TKE",
                        **metric_volume_kwargs(ws, "tke_volume", clim),
                        mapper='fixed_point' if plotter.render_window.GetClassName() == 'vtkOSOpenGLRenderWindow' else 'smart',
                        **_scalar_bar_mesh_kwargs(show_scalar_bar, "TKE (J/m³)", tke_bar_cfg, ws=ws),
                    )
                else:
                    tke_actor.SetVisibility(1)
                    tke_actor.GetMapper().dataset = tke_mesh
                    tke_actor.GetMapper().Update()
                apply_metric_volume_opacity(tke_actor, clim, metric_style(ws, 'tke_volume')['opacity'], ws.resolution, workspace=ws)
            elif tke_actor is not None:
                tke_actor.SetVisibility(0)

            if rotate:
                txt = f"t={t} | rot {frame_idx + 1}/{total_frames}"
            else:
                txt = f"t={t}"

            text_actor.SetText(2, txt)
            plotter.camera_position = camera_position
            plotter.render()
            check_cancelled()
            yield np.asarray(plotter.screenshot(return_img=True))
            report_progress({"stage": "video_frame", "current": frame_idx + 1, "total": total_frames,
                             "message": f"Rendered frame {frame_idx + 1}/{total_frames}"})

    finally:
        plotter.close()


def _pressure_gradient_max(ws):
    arr = ws.derived.pressure_gradient_magnitude
    if arr is None:
        return 1e-6
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return 1e-6
    value = float(np.nanmax(arr))
    if not np.isfinite(value):
        return 1e-6
    return max(value, 1e-6)


def _relative_pressure_max(ws):
    arr = ws.derived.relative_pressure_array
    if arr is None:
        return 1e-6
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return 1e-6
    value = float(np.nanmax(np.abs(arr)))
    if not np.isfinite(value):
        return 1e-6
    return max(value, 1e-6)


@_stream_video("pressure_gradient")
def render_pressure_gradient_video(
    ws,
    out_dir,
    fps=24,
    smoothing_iteration=200,
    view="iso",
    distance_scale=1.0,
    pressure_gradient_clim=None,
    pressure_gradient_bar_cfg=None,
    show_scalar_bar=True,
    rotate=False,
    rotation_frames=None,
    elevation_deg=None,
    time_repeat=1,
    window_size=None,
):
    if ws.derived.pressure_gradient_magnitude is None:
        return None

    _, surf = _build_union_surface(ws, smoothing_iteration=smoothing_iteration)
    if surf is None or surf.n_points == 0:
        return None

    pg_arr = np.asarray(ws.derived.pressure_gradient_magnitude, dtype=np.float32)
    if pg_arr.ndim not in (3, 4):
        return None

    pg_max = _pressure_gradient_max(ws)
    fallback = tuple(ws.derived.pressure_gradient_display_clim) if ws.derived.pressure_gradient_display_clim is not None else (0.0, pg_max)
    clim = metric_clim(ws, "pressure_gradient_volume", pressure_gradient_clim, fallback)

    _, default_elevation_deg = _resolve_view(view)
    if elevation_deg is None:
        elevation_deg = default_elevation_deg

    n_time = int(max(ws.time_count(), 1 if pg_arr.ndim == 3 else pg_arr.shape[3]))
    if rotate:
        base_frames = n_time * int(max(time_repeat, 1))
        total_frames = max(int(rotation_frames), base_frames) if rotation_frames is not None else base_frames
    else:
        total_frames = n_time * int(max(time_repeat, 1))

    support_source = ws.derived.pressure_gradient_support_mask
    if support_source is None:
        support_source = ws.segmask_3d if ws.segmask_3d is not None else np.max(ws.segmask_binary > 0, axis=-1)
    support_arr = np.asarray(support_source, dtype=bool)
    plotter = _make_plotter(window_size=_resolve_window_size(window_size), background=_video_background(ws), workspace=ws)
    try:
        pg_mesh_cache = {}
        pg_actor = None
        text_actor = plotter.add_text("", position="upper_left", font_size=14, color=foreground_color(_video_background(ws), ws))

        for frame_idx in range(total_frames):
            check_cancelled()
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

            tidx = min(max(0, t), pg_arr.shape[3] - 1) if pg_arr.ndim == 4 else 0
            if tidx not in pg_mesh_cache:
                vol_t = pg_arr if pg_arr.ndim == 3 else pg_arr[..., tidx]
                support_t = support_arr if support_arr.ndim == 3 else support_arr[..., tidx]
                vol_t = np.where(support_t, vol_t, 0.0)
                support_surface = _pressure_support_surface(ws, support_source, tidx, smooth_iter=80)
                pg_mesh_cache[tidx] = sample_display_field(
                    vol_t, support_t, support_surface, ws.resolution, origin=ws.origin,
                    name="PressureGradient",
                )
            pg_mesh = pg_mesh_cache[tidx]
            if pg_mesh is not None and pg_mesh.n_points > 0:
                if pg_actor is None:
                    pg_actor = plotter.add_mesh(
                        pg_mesh,
                        scalars="PressureGradient",
                        clim=clim,
                        **metric_style(ws, "pressure_gradient_volume"),
                        **_scalar_bar_mesh_kwargs(show_scalar_bar, "|Pressure Grad| (Pa/m)", pressure_gradient_bar_cfg, ws=ws),
                    )
                else:
                    pg_actor.SetVisibility(1)
                    mapper = pg_actor.GetMapper()
                    mapper.dataset = pg_mesh
                    mapper.Update()
            elif pg_actor is not None:
                pg_actor.SetVisibility(0)

            txt = f"t={t} | rot {frame_idx + 1}/{total_frames}" if rotate else f"t={t}"
            try:
                text_actor.SetText(2, txt)
            except Exception:
                try:
                    text_actor.SetInput(txt)
                except Exception:
                    pass
            plotter.camera_position = camera_position
            plotter.render()
            check_cancelled()
            yield np.asarray(plotter.screenshot(return_img=True))
            report_progress({"stage": "video_frame", "current": frame_idx + 1, "total": total_frames,
                             "message": f"Rendered frame {frame_idx + 1}/{total_frames}"})

    finally:
        plotter.close()


@_stream_video("relative_pressure")
def render_relative_pressure_video(
    ws,
    out_dir,
    fps=24,
    smoothing_iteration=200,
    view="iso",
    distance_scale=1.0,
    relative_pressure_clim=None,
    relative_pressure_bar_cfg=None,
    show_scalar_bar=True,
    rotate=False,
    rotation_frames=None,
    elevation_deg=None,
    time_repeat=1,
    window_size=None,
):
    if ws.derived.relative_pressure_array is None:
        return None

    _, surf = _build_union_surface(ws, smoothing_iteration=smoothing_iteration)
    if surf is None or surf.n_points == 0:
        return None

    rp_arr = np.asarray(ws.derived.relative_pressure_array, dtype=np.float32)
    if rp_arr.ndim not in (3, 4):
        return None

    rp_max = _relative_pressure_max(ws)
    fallback = tuple(ws.derived.relative_pressure_display_clim) if ws.derived.relative_pressure_display_clim is not None else (-rp_max, rp_max)
    clim = metric_clim(ws, "relative_pressure_volume", relative_pressure_clim, fallback)

    _, default_elevation_deg = _resolve_view(view)
    if elevation_deg is None:
        elevation_deg = default_elevation_deg

    n_time = int(max(ws.time_count(), 1 if rp_arr.ndim == 3 else rp_arr.shape[3]))
    if rotate:
        base_frames = n_time * int(max(time_repeat, 1))
        total_frames = max(int(rotation_frames), base_frames) if rotation_frames is not None else base_frames
    else:
        total_frames = n_time * int(max(time_repeat, 1))

    support_source = ws.derived.pressure_gradient_support_mask
    if support_source is None:
        support_source = ws.segmask_3d if ws.segmask_3d is not None else np.max(ws.segmask_binary > 0, axis=-1)
    support_arr = np.asarray(support_source, dtype=bool)
    plotter = _make_plotter(window_size=_resolve_window_size(window_size), background=_video_background(ws), workspace=ws)
    try:
        rp_mesh_cache = {}
        rp_actor = None
        text_actor = plotter.add_text("", position="upper_left", font_size=14, color=foreground_color(_video_background(ws), ws))

        for frame_idx in range(total_frames):
            check_cancelled()
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

            tidx = min(max(0, t), rp_arr.shape[3] - 1) if rp_arr.ndim == 4 else 0
            if tidx not in rp_mesh_cache:
                vol_t = rp_arr if rp_arr.ndim == 3 else rp_arr[..., tidx]
                support_t = support_arr if support_arr.ndim == 3 else support_arr[..., tidx]
                vol_t = np.where(support_t, vol_t, 0.0)
                support_surface = _pressure_support_surface(ws, support_source, tidx, smooth_iter=80)
                rp_mesh_cache[tidx] = sample_display_field(
                    vol_t, support_t, support_surface, ws.resolution, origin=ws.origin,
                    name="RelativePressure",
                )
            rp_mesh = rp_mesh_cache[tidx]
            if rp_mesh is not None and rp_mesh.n_points > 0:
                if rp_actor is None:
                    rp_actor = plotter.add_mesh(
                        rp_mesh,
                        scalars="RelativePressure",
                        clim=clim,
                        **metric_style(ws, "relative_pressure_volume"),
                        **_scalar_bar_mesh_kwargs(show_scalar_bar, "Relative Pressure (Pa)", relative_pressure_bar_cfg, ws=ws),
                    )
                else:
                    rp_actor.SetVisibility(1)
                    mapper = rp_actor.GetMapper()
                    mapper.dataset = rp_mesh
                    mapper.Update()
            elif rp_actor is not None:
                rp_actor.SetVisibility(0)

            txt = f"t={t} | rot {frame_idx + 1}/{total_frames}" if rotate else f"t={t}"
            try:
                text_actor.SetText(2, txt)
            except Exception:
                try:
                    text_actor.SetInput(txt)
                except Exception:
                    pass
            plotter.camera_position = camera_position
            plotter.render()
            check_cancelled()
            yield np.asarray(plotter.screenshot(return_img=True))
            report_progress({"stage": "video_frame", "current": frame_idx + 1, "total": total_frames,
                             "message": f"Rendered frame {frame_idx + 1}/{total_frames}"})

    finally:
        plotter.close()


def extract_frame(mp4_path, frame_index, out_png):
    reader = imageio.get_reader(mp4_path, format="ffmpeg")
    frame = reader.get_data(frame_index)
    reader.close()
    Image.fromarray(np.asarray(frame)).save(out_png, format="PNG", compress_level=0)
    print(f"Saved frame {frame_index} -> {out_png}")
