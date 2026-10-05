"""Shared quantitative and anatomical styling for interactive and offline views."""

import numpy as np
import pyvista as pv


from copy import deepcopy
from ..config import BACKGROUND_COLOR, DEFAULT_RENDER_STYLE_CFG, DEFAULT_SCALAR_BAR_STYLE, METRIC_DATA_KEYS, DEFAULT_PLANE_RENDER_CFG


def render_style_settings(workspace=None):
    """Resolved JSON style travels with the workspace, including export jobs."""
    settings = getattr(workspace, "render_settings", {}) or {}
    return settings.get("render_style_cfg") or DEFAULT_RENDER_STYLE_CFG


def configured_metric_style(workspace, data_key):
    return deepcopy(render_style_settings(workspace)["metrics"][METRIC_DATA_KEYS[data_key]])


def foreground_color(background, workspace=None):
    rgb = np.asarray(pv.Color(background).float_rgb, dtype=float)
    key = "dark" if float(rgb @ [0.2126, 0.7152, 0.0722]) >= 0.5 else "light"
    return render_style_settings(workspace)["text"][key]


def configure_plotter(plotter, background=BACKGROUND_COLOR, workspace=None):
    plotter.set_background(background)
    mode = render_style_settings(workspace)["anti_aliasing"]
    # EGL/OSMesa use SSAA because VTK does not support FXAA on those backends.
    try:
        if mode == "none":
            plotter.disable_anti_aliasing()
        else:
            if mode == "auto":
                backend = plotter.render_window.GetClassName()
                mode = "ssaa" if "EGL" in backend or "OSOpenGL" in backend else "fxaa"
            plotter.enable_anti_aliasing(mode)
    except (AttributeError, TypeError, RuntimeError):
        pass



def plane_display_size(workspace=None):
    """Physical side length of the displayed square, independent of metric ROI."""
    settings = getattr(workspace, "render_settings", {}) or {}
    cfg = settings.get("plane_render_cfg", {}) or {}
    value = float(cfg.get("plane_size_mm", DEFAULT_PLANE_RENDER_CFG["plane_size_mm"]))
    if not np.isfinite(value) or not 1.0 <= value <= 200.0:
        raise ValueError("plane display size must be between 1 and 200 mm")
    return value

def surface_style(quantitative=False, workspace=None):
    key = "quantitative" if quantitative else "anatomical"
    return dict(render_style_settings(workspace)["surfaces"][key])


def scene_object(workspace, data_key):
    return next((obj for obj in getattr(workspace, "scene_objects", {}).values()
                 if obj.data_key == data_key), None)


def metric_style(workspace, data_key):
    obj = scene_object(workspace, data_key)
    configured = configured_metric_style(workspace, data_key)
    opacity = configured["opacity"]
    if data_key == "pressure_gradient_volume":
        opacity = workspace.derived_params.pressure_gradient_layer_opacity
    elif data_key == "relative_pressure_volume":
        opacity = workspace.derived_params.relative_pressure_layer_opacity
    return dict(surface_style(quantitative=True, workspace=workspace),
                cmap=(obj.cmap if obj is not None else None) or configured["cmap"],
                opacity=float(obj.opacity if obj is not None else opacity))


def scalar_bar_args(title, feature_cfg=None, shared_cfg=None, background=BACKGROUND_COLOR, workspace=None):
    # VTK's Arial font omits the superscript minus and lambda glyphs on some
    # backends. Keep reciprocal units readable rather than changing their sign.
    title = str(title or "").replace("s⁻¹", "1/s").replace("s⁻²", "1/s²")
    title = title.replace("Swirling Strength λci", "Swirling Strength")
    cfg = dict(DEFAULT_SCALAR_BAR_STYLE, title=title.replace(" (", "\n("))
    for override in (feature_cfg, shared_cfg):
        cfg.update({k: v for k, v in (override or {}).items() if k != "stack_gap"})
    cfg["width"] = min(max(float(cfg["width"]), 0.02), 0.4)
    cfg["height"] = min(max(float(cfg["height"]), 0.15), 0.95)
    cfg["position_x"] = min(max(float(cfg["position_x"]), 0.0), 0.98 - cfg["width"])
    cfg["position_y"] = min(max(float(cfg["position_y"]), 0.0), 0.98 - cfg["height"])
    cfg.setdefault("color", foreground_color(background, workspace))
    return cfg


def wss_display_surface(surface):
    """Respect invalid probes on a display copy without changing scientific data."""
    if surface is None or "wss" not in surface.point_data:
        return surface
    values = np.asarray(surface.point_data["wss"], dtype=float)
    valid = np.isfinite(values)
    if "wss_valid" in surface.point_data:
        valid &= np.asarray(surface.point_data["wss_valid"], dtype=bool)
    if np.all(valid):
        return surface
    display = surface.copy(deep=False)
    display.point_data["wss"] = np.where(valid, values, np.nan)
    return display


def wss_scalar_range(surfaces):
    maximum = 0.0
    for surface in surfaces or []:
        if surface is None or "wss" not in surface.point_data:
            continue
        values = np.asarray(surface.point_data["wss"], dtype=float)
        valid = np.isfinite(values)
        if "wss_valid" in surface.point_data:
            valid &= np.asarray(surface.point_data["wss_valid"], dtype=bool)
        if np.any(valid):
            maximum = max(maximum, float(np.max(values[valid])))
    return (0.0, maximum if maximum > 0.0 else 1.0)


def metric_volume_kwargs(workspace, data_key, clim):
    style = metric_style(workspace, data_key)
    volume = configured_metric_style(workspace, data_key)["volume"]
    alpha = float(np.clip(style['opacity'], 0.0, 1.0))
    positions, values = zip(*volume["opacity_points"])
    opacity = alpha * np.interp(np.linspace(0, 1, 256), positions, values)
    return dict(cmap=style['cmap'], clim=clim, opacity=np.rint(opacity * 255).astype(np.uint8),
                shade=volume['shade'], blending=volume['blending'],
                opacity_unit_distance=max(float(np.mean(workspace.resolution)), .1)
                * volume['opacity_unit_distance_scale'])


def apply_metric_volume_opacity(actor, clim, opacity, spacing, *, workspace=None, data_key="tke_volume"):
    # Keep quantitative colour bars opaque while the volume hides low energy.
    lookup = actor.GetMapper().lookup_table
    colors = lookup.values.copy()
    colors[:, 3] = 255
    lookup.values = colors
    volume = configured_metric_style(workspace, data_key)["volume"]
    prop = actor.GetProperty()
    transfer = prop.GetScalarOpacity(0)
    transfer.RemoveAllPoints()
    lo, hi = map(float, clim)
    width = max(hi - lo, 1e-6)
    alpha = float(np.clip(opacity, 0.0, 1.0))
    for fraction, value in volume["opacity_points"]:
        transfer.AddPoint(lo + width * fraction, alpha * value)
    transfer.AddPoint(0., 0.)
    if volume["interpolation"] == "nearest":
        prop.SetInterpolationTypeToNearest()
    else:
        prop.SetInterpolationTypeToLinear()
    prop.SetScalarOpacityUnitDistance(max(float(np.mean(spacing)), .1)
                                     * volume["opacity_unit_distance_scale"])


def metric_clim(workspace, data_key, explicit=None, fallback=(0.0, 1.0)):
    """Use explicit export limits, then the GUI scene limits, then config."""
    if explicit is not None:
        return tuple(explicit)
    obj = scene_object(workspace, data_key)
    if obj is not None and obj.clim is not None:
        return tuple(obj.clim)
    key = {'wss_surface_live': 'wss_clim', 'tke_volume': 'tke_clim',
           'pressure_gradient_volume': 'pressure_gradient_clim',
           'relative_pressure_volume': 'relative_pressure_clim',
           'streamlines_live': 'streamline_clim'}[data_key]
    configured = (getattr(workspace, 'render_settings', {}) or {}).get(key)
    return tuple(configured if configured is not None else fallback)
