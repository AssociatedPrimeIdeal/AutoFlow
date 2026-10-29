"""Reuse plane curve artists; phase changes only move the current sample."""

import hashlib
from functools import lru_cache
import numpy as np
from matplotlib import font_manager, ticker
from .theme import THEMES, theme_colors


PANEL_BACKGROUND = THEMES["dark"]["panel"]
CURVE_COLOR = THEMES["dark"]["curve"]
CURSOR_COLOR = THEMES["dark"]["cursor"]


def analysis_palette(window):
    return theme_colors(window)


@lru_cache(maxsize=8)
def _chart_font(family):
    available = {entry.name for entry in font_manager.fontManager.ttflist}
    return next((name for name in (family, "Noto Sans", "Arial", "DejaVu Sans")
                 if name in available), "DejaVu Sans")


class _UiScalarFormatter(ticker.ScalarFormatter):
    @staticmethod
    def fix_minus(text):
        # Some installed UI font files omit U+2212. Keep negative labels legible.
        return str(text).replace("\N{MINUS SIGN}", "-")


def style_analysis_figure(window):
    """Style GUI artists locally; preserve data, series colors and export plots."""
    colors = analysis_palette(window)
    figure = window.fig_pwv
    figure.set_facecolor(colors["panel"])
    family = _chart_font(window.font().family())
    for axes in figure.axes:
        axes.set_facecolor(colors["panel"])
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)
        for side in ("left", "bottom"):
            axes.spines[side].set_color(colors["spine"])
            axes.spines[side].set_linewidth(.65)
        axes.grid(False)
        axes.yaxis.grid(True, color=colors["grid"], linewidth=.6, alpha=.65)
        for axis in (axes.xaxis, axes.yaxis):
            formatter = axis.get_major_formatter()
            if isinstance(formatter, ticker.ScalarFormatter) and not isinstance(formatter, _UiScalarFormatter):
                axis.set_major_formatter(_UiScalarFormatter(useOffset=formatter.get_useOffset(),
                                                           useLocale=formatter.get_useLocale(), useMathText=False))
        if axes.get_xlabel().lower() == "cardiac phase":
            axes.xaxis.set_major_locator(ticker.MaxNLocator(nbins="auto", integer=True))
        axes.tick_params(axis="both", colors=colors["detail_label"], labelsize=11, width=.6, length=3, pad=5)
        labels = [axes.title, axes.xaxis.label, axes.yaxis.label,
                  *axes.get_xticklabels(), *axes.get_yticklabels(),
                  axes.xaxis.get_offset_text(), axes.yaxis.get_offset_text(), *axes.texts]
        for label in labels:
            label.set_fontfamily(family)
            label.set_fontweight("normal")
        axes.title.set_fontsize(12)
        axes.title.set_color(colors["title"])
        for label in (axes.xaxis.label, axes.yaxis.label):
            label.set_fontsize(11)
            label.set_color(colors["detail_label"])
        for label in axes.texts:
            if label.get_color() in ("black", "#000000"):
                label.set_color(colors["secondary"])
        legend = axes.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(colors["panel"])
            legend.get_frame().set_edgecolor("none")
            for label in legend.get_texts():
                label.set_fontfamily(family)
                label.set_fontsize(10)
                label.set_color(colors["detail_label"])
    figure.tight_layout(pad=1.6)


def update_plane_curve(window, index, metric, key, label, smooth):
    colors = analysis_palette(window)
    values = np.asarray(metric.get(key, []), dtype=float).reshape(-1).copy()
    valid = metric.get("sample_valid_t", [])
    if len(valid) == len(values):
        values[~np.asarray(valid, dtype=bool)] = np.nan
    token = (window.workspace.planes[index].plane_id, key, hashlib.sha256(values.tobytes()).digest(),
             int(window._flowrate_spline_order()))
    cache = getattr(window, "_plane_curve_artists", None)
    phase = int(np.clip(window.workspace.current_t, 0, max(0, len(values)-1)))
    if cache is not None and cache["token"] == token and cache["axes"] in window.fig_pwv.axes:
        cache["cursor"].set_xdata([phase, phase])
        cache["point"].set_offsets([[phase, values[phase]]])
    else:
        window.fig_pwv.clear()
        axes = window.fig_pwv.add_subplot(111)
        phases = np.arange(len(values), dtype=float)
        if str(key).startswith("flowrate_") and np.all(np.isfinite(values)):
            x, y = smooth(values, order=window._flowrate_spline_order())
            axes.plot(x, y, color=colors["curve"], linewidth=1.65)
            axes.plot(phases, values, color=colors["curve"], marker="o", linestyle="none", markersize=2.5,
                      markerfacecolor=colors["panel"], markeredgewidth=.8)
        else:
            axes.plot(phases, values, color=colors["curve"], marker="o", linewidth=1.65, markersize=2.5,
                      markerfacecolor=colors["panel"], markeredgewidth=.8)
        cursor = axes.axvline(phase, color=colors["cursor"], linestyle=(0, (2, 4)), linewidth=.8)
        point = axes.scatter([phase], [values[phase]], color=colors["curve"], s=24,
                             edgecolors=colors["panel"], linewidths=.9, zorder=4)
        axes.set(xlabel="Cardiac phase", ylabel=label)
        if len(values) > 1:
            axes.set_xlim(0, len(values)-1)
        style_analysis_figure(window)
        window._plane_curve_artists = {"token": token, "axes": axes, "cursor": cursor, "point": point, "values": values}
    window.canvas_pwv.draw_idle()
    return values


def move_phase_cursor(window):
    cache = getattr(window, "_plane_curve_artists", None)
    if cache is None or not window.canvas_pwv.isVisible() or cache["axes"] not in window.fig_pwv.axes:
        return
    values = cache["values"]
    phase = int(np.clip(window.workspace.current_t, 0, max(0, len(values)-1)))
    cache["cursor"].set_xdata([phase, phase])
    cache["point"].set_offsets([[phase, values[phase]]])
    window.canvas_pwv.draw_idle()
