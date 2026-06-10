import json
import os
import re
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np

from ..core.models import PwvParams, SkeletonParams
from .branch import segment_vessels_from_graph_and_mask
from .graph import build_graph_from_points, graph_to_networkx
from .metrics import compute_plane_metrics
from .paths import _vector_orientation_text
from .planes import generate_planes_from_paths
from .preprocess import largest_connected_component, majority_vote_labels_3d, preprocess_mask_for_skeleton
from .skeleton import generate_skeleton_from_mask3d


def _repeat_mask_to_time(mask_3d, time_count):
    mask = np.asarray(mask_3d, dtype=bool)
    nt = max(1, int(time_count))
    return np.repeat(mask[..., np.newaxis], nt, axis=3).astype(bool)


def _sanitize_name(name: str) -> str:
    token = re.sub(r"[^0-9A-Za-z._-]+", "_", str(name or "pwv")).strip("_")
    return token or "pwv"


def _path_length_mm(path) -> float:
    pts = np.asarray(path, dtype=float).reshape(-1, 3)
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def _smooth_signal(values, window, polyorder):
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    try:
        from scipy.signal import savgol_filter
    except Exception:
        return arr.copy()
    win = int(window)
    if win < 3:
        return arr.copy()
    if win % 2 == 0:
        win += 1
    if win > arr.size:
        win = arr.size if arr.size % 2 == 1 else max(1, arr.size - 1)
    if win < 3 or win <= int(polyorder):
        return arr.copy()
    try:
        return np.asarray(savgol_filter(arr, win, int(polyorder)), dtype=float)
    except Exception:
        return arr.copy()


def _wrap_time_to_reference_ms(time_ms, reference_ms, rr_ms):
    if time_ms is None or reference_ms is None:
        return time_ms
    rr = float(rr_ms)
    if rr <= 0.0:
        return float(time_ms)
    delta = float(time_ms) - float(reference_ms)
    while delta <= -0.5 * rr:
        delta += rr
    while delta > 0.5 * rr:
        delta -= rr
    return float(reference_ms) + float(delta)


def _upsample_cyclic_signal(values, factor):
    arr = np.asarray(values, dtype=float).reshape(-1)
    scale = max(1, int(factor))
    if arr.size < 2 or scale <= 1:
        return arr.copy()
    xp = np.arange(arr.size + 1, dtype=float)
    fp = np.concatenate([arr, arr[:1]])
    x_new = np.arange(arr.size * scale, dtype=float) / float(scale)
    return np.asarray(np.interp(x_new, xp, fp), dtype=float)


def _foot_search_state(values, *, allow_cycle_wrap=False):
    arr = np.asarray(values, dtype=float).reshape(-1)
    nt = int(arr.size)
    if nt < 3 or not np.any(np.isfinite(arr)):
        return None
    peak_idx = int(np.argmax(arr))
    if allow_cycle_wrap and nt >= 3:
        ext = np.concatenate([arr, arr])
        peak_idx_wrapped = int(peak_idx + nt)
        search_lo = int(peak_idx + 1)
        search_hi = int(peak_idx_wrapped + 1)
        if search_hi - search_lo >= 2:
            baseline_idx_wrapped = int(np.argmin(ext[search_lo:search_hi])) + search_lo
            if baseline_idx_wrapped < peak_idx_wrapped:
                return {
                    "signal": ext,
                    "peak_index": int(peak_idx_wrapped),
                    "baseline_index": int(baseline_idx_wrapped),
                    "cycle_wrapped": True,
                    "original_peak_index": int(peak_idx),
                    "original_length": int(nt),
                }
    if peak_idx <= 0:
        return None
    baseline_idx = int(np.argmin(arr[:peak_idx + 1]))
    if baseline_idx >= peak_idx:
        return None
    return {
        "signal": arr,
        "peak_index": int(peak_idx),
        "baseline_index": int(baseline_idx),
        "cycle_wrapped": False,
        "original_peak_index": int(peak_idx),
        "original_length": int(nt),
    }


def _annotate_cycle_indices(meta, original_length, cycle_wrapped):
    info = dict(meta or {})
    info["cycle_wrapped"] = bool(cycle_wrapped)
    info["cycle_search"] = "wrapped" if cycle_wrapped else "single_cycle"
    if original_length <= 0:
        return info
    for key in ("baseline_index", "peak_index", "slope_index", "threshold_crossing_index"):
        value = info.get(key)
        if value is None:
            continue
        idx = int(value)
        if cycle_wrapped:
            info[f"{key}_wrapped"] = int(idx)
            info[key] = int(idx % int(original_length))
        else:
            info[key] = int(idx)
    return info


def _threshold_foot_time_ms(smooth, times_ms, peak_idx, baseline_idx, threshold_percent):
    if peak_idx <= baseline_idx:
        return None, {}
    baseline = float(smooth[baseline_idx])
    peak_value = float(smooth[peak_idx])
    amp = peak_value - baseline
    if not np.isfinite(amp) or amp <= 1e-12:
        return None, {
            "baseline_index": int(baseline_idx),
            "peak_index": int(peak_idx),
            "baseline_value": baseline,
            "peak_value": peak_value,
        }
    frac = float(np.clip(float(threshold_percent) / 100.0, 0.0, 1.0))
    target = baseline + frac * amp
    seg = np.asarray(smooth[baseline_idx:peak_idx + 1], dtype=float)
    idx_rel = np.where(seg >= target)[0]
    if idx_rel.size == 0:
        return None, {
            "baseline_index": int(baseline_idx),
            "peak_index": int(peak_idx),
            "baseline_value": baseline,
            "peak_value": peak_value,
            "threshold_percent": float(threshold_percent),
            "threshold_value": float(target),
        }
    hi = int(idx_rel[0] + baseline_idx)
    lo = max(int(baseline_idx), hi - 1)
    if hi == lo:
        foot_time = float(times_ms[hi])
    else:
        y0 = float(smooth[lo])
        y1 = float(smooth[hi])
        if abs(y1 - y0) <= 1e-12:
            alpha = 0.0
        else:
            alpha = float(np.clip((target - y0) / (y1 - y0), 0.0, 1.0))
        foot_time = float(times_ms[lo] + alpha * (times_ms[hi] - times_ms[lo]))
    return foot_time, {
        "baseline_index": int(baseline_idx),
        "peak_index": int(peak_idx),
        "baseline_value": baseline,
        "peak_value": peak_value,
        "threshold_percent": float(threshold_percent),
        "threshold_value": float(target),
        "threshold_crossing_index": int(hi),
    }


def _tangent_foot_time_ms(smooth, times_ms, baseline_idx, peak_idx, dt_ms):
    if baseline_idx >= peak_idx:
        return None, {}
    deriv = np.gradient(smooth, dt_ms)
    search = deriv[baseline_idx:peak_idx + 1]
    if search.size == 0:
        return None, {}
    slope_idx = int(np.argmax(search)) + baseline_idx
    slope = float(deriv[slope_idx])
    if not np.isfinite(slope) or slope <= 1e-12:
        return None, {
            "baseline_index": int(baseline_idx),
            "peak_index": int(peak_idx),
        }
    baseline = float(smooth[baseline_idx])
    foot_time = float(times_ms[slope_idx] - (float(smooth[slope_idx]) - baseline) / slope)
    return foot_time, {
        "baseline_index": int(baseline_idx),
        "peak_index": int(peak_idx),
        "slope_index": int(slope_idx),
        "baseline_value": baseline,
        "slope_value": slope,
    }


def detect_waveform_foot_time_ms(waveform, rr_ms, window=5, polyorder=2, *, method="tangent", threshold_percent=10.0, allow_cycle_wrap=True):
    values = np.asarray(waveform, dtype=float).reshape(-1)
    if values.size < 3 or not np.any(np.isfinite(values)):
        return None, {}
    clean = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    smooth = _smooth_signal(clean, window, polyorder)
    nt = max(1, smooth.size)
    dt_ms = float(rr_ms) / float(nt)
    times_ms = np.arange(nt, dtype=float) * dt_ms
    search_state = _foot_search_state(smooth, allow_cycle_wrap=bool(allow_cycle_wrap) and float(rr_ms) > 0.0)
    if search_state is None:
        return None, {"times_ms": times_ms.tolist(), "smooth_waveform": smooth.tolist()}
    search_signal = np.asarray(search_state["signal"], dtype=float)
    search_times_ms = np.arange(search_signal.size, dtype=float) * dt_ms
    peak_idx = int(search_state["peak_index"])
    baseline_idx = int(search_state["baseline_index"])
    method_name = str(method or "tangent").strip().lower()
    if method_name == "threshold":
        foot_time, meta = _threshold_foot_time_ms(search_signal, search_times_ms, peak_idx, baseline_idx, threshold_percent)
    else:
        method_name = "tangent"
        foot_time, meta = _tangent_foot_time_ms(search_signal, search_times_ms, baseline_idx, peak_idx, dt_ms)
    meta = _annotate_cycle_indices(meta, int(search_state["original_length"]), bool(search_state["cycle_wrapped"]))
    if foot_time is None:
        return None, {
            "times_ms": times_ms.tolist(),
            "smooth_waveform": smooth.tolist(),
            "method": method_name,
            **meta,
        }
    clipped = False
    raw_foot_time = float(foot_time)
    rr = float(rr_ms)
    if allow_cycle_wrap and rr > 0.0:
        while foot_time < 0.0:
            foot_time += rr
        while foot_time >= rr:
            foot_time -= rr
    else:
        lo = float(times_ms[0])
        hi = float(times_ms[-1])
        clipped = bool(foot_time < lo or foot_time > hi)
        foot_time = float(np.clip(foot_time, lo, hi))
    return float(foot_time), {
        "times_ms": times_ms.tolist(),
        "smooth_waveform": smooth.tolist(),
        "method": method_name,
        "raw_time_to_foot_ms": raw_foot_time,
        "time_to_foot_ms": float(foot_time),
        "clipped_to_range": bool(clipped),
        "allow_cycle_wrap": bool(allow_cycle_wrap),
        **meta,
    }


def _waveform_segment_for_xcorr(waveform, *, window="full", allow_cycle_wrap=True):
    arr = np.asarray(waveform, dtype=float).reshape(-1)
    if arr.size < 3 or not np.any(np.isfinite(arr)):
        return np.empty(0, dtype=float), 0, arr, {}
    clean = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    mode = str(window or "full").strip().lower()
    if mode != "upstroke":
        return clean, 0, clean, {"window": mode, "segment_mode": "full"}
    search_state = _foot_search_state(clean, allow_cycle_wrap=bool(allow_cycle_wrap))
    if search_state is None:
        return clean, 0, clean, {"window": mode, "segment_mode": "full_fallback"}
    baseline_idx = int(search_state["baseline_index"])
    peak_idx = int(search_state["peak_index"])
    if baseline_idx >= peak_idx:
        return clean, 0, clean, {"window": mode, "segment_mode": "full_fallback"}
    seg = np.asarray(search_state["signal"][baseline_idx:peak_idx + 1], dtype=float)
    meta = {
        "window": mode,
        "segment_mode": "upstroke",
        **_annotate_cycle_indices(
            {"baseline_index": baseline_idx, "peak_index": peak_idx},
            int(search_state["original_length"]),
            bool(search_state["cycle_wrapped"]),
        ),
    }
    return seg, int(baseline_idx), clean, meta


def compute_cross_correlation_delay_ms(reference_waveform, target_waveform, rr_ms, *, window="full", allow_cycle_wrap=True, interp_factor=10):
    factor = max(1, int(interp_factor))
    ref_input = _upsample_cyclic_signal(reference_waveform, factor)
    tgt_input = _upsample_cyclic_signal(target_waveform, factor)
    ref_seg, ref_offset, ref_clean, ref_seg_meta = _waveform_segment_for_xcorr(
        ref_input, window=window, allow_cycle_wrap=allow_cycle_wrap
    )
    tgt_seg, tgt_offset, tgt_clean, tgt_seg_meta = _waveform_segment_for_xcorr(
        tgt_input, window=window, allow_cycle_wrap=allow_cycle_wrap
    )
    n = min(ref_seg.size, tgt_seg.size)
    if n < 3:
        return None, {}
    ref = np.asarray(ref_seg[:n], dtype=float)
    tgt = np.asarray(tgt_seg[:n], dtype=float)
    ref = ref - float(np.mean(ref))
    tgt = tgt - float(np.mean(tgt))
    ref_std = float(np.std(ref))
    tgt_std = float(np.std(tgt))
    if ref_std <= 1e-12 or tgt_std <= 1e-12:
        return None, {"window": str(window or "full")}
    ref = ref / ref_std
    tgt = tgt / tgt_std
    corr = np.correlate(tgt, ref, mode="full")
    lags = np.arange(-n + 1, n, dtype=float)
    best_idx = int(np.argmax(corr))
    best_lag = float(lags[best_idx])
    offset_lag = float(tgt_offset) - float(ref_offset)
    total_lag = float(offset_lag + best_lag)
    dt_ms = float(rr_ms) / float(max(1, len(ref_clean)))
    delay_ms = float(total_lag) * dt_ms
    raw_delay_ms = float(delay_ms)
    if allow_cycle_wrap and float(rr_ms) > 0.0:
        while delay_ms <= -0.5 * float(rr_ms):
            delay_ms += float(rr_ms)
        while delay_ms > 0.5 * float(rr_ms):
            delay_ms -= float(rr_ms)
    return float(delay_ms), {
        "window": str(window or "full"),
        "segment_length": int(n),
        "reference_offset_index": float(ref_offset) / float(factor),
        "target_offset_index": float(tgt_offset) / float(factor),
        "reference_offset_interp_index": int(ref_offset),
        "target_offset_interp_index": int(tgt_offset),
        "offset_lag_samples": float(offset_lag) / float(factor),
        "offset_lag_interp_samples": float(offset_lag),
        "lag_samples": float(best_lag) / float(factor),
        "lag_interp_samples": float(best_lag),
        "total_lag_samples": float(total_lag) / float(factor),
        "total_lag_interp_samples": float(total_lag),
        "raw_delay_ms": raw_delay_ms,
        "delay_ms": float(delay_ms),
        "peak_correlation": float(corr[best_idx]),
        "allow_cycle_wrap": bool(allow_cycle_wrap),
        "interp_factor": int(factor),
        "reference_segment": ref_seg_meta,
        "target_segment": tgt_seg_meta,
    }


def save_pwv_plot(result, out_path, color="#2b8a3e", fit_color="#f08c00", dpi=160):
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except Exception as exc:
        raise RuntimeError("PWV plotting requires matplotlib") from exc
    fig = Figure(figsize=(7.2, 7.2), dpi=int(dpi))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(211)
    ax_flow = fig.add_subplot(212)
    _plot_pwv_axes(ax, result, color=color, fit_color=fit_color)
    _plot_plane_flowrate_axes(ax_flow, result, color=color)
    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi))
    return out_path


def save_pwv_results(results, out_path):
    payload = {"results": results}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def _longest_path(paths):
    best_idx = -1
    best_len = -1.0
    best_path = None
    for idx, path in enumerate(list(paths or [])):
        plen = _path_length_mm(path)
        if plen > best_len:
            best_len = plen
            best_idx = int(idx)
            best_path = np.asarray(path, dtype=float)
    return best_idx, best_path, float(max(best_len, 0.0))


def _edge_weighted_graph(graph):
    G = graph_to_networkx(graph)
    pts = np.asarray(getattr(graph, "points", []), dtype=float).reshape(-1, 3)
    if G.number_of_nodes() == 0 or len(pts) == 0:
        return G
    for u, v in list(G.edges()):
        pu = np.asarray(pts[int(u)], dtype=float)
        pv = np.asarray(pts[int(v)], dtype=float)
        G[int(u)][int(v)]["weight"] = float(np.linalg.norm(pv - pu))
    return G


def _node_path_to_points(graph_points, node_path):
    pts = np.asarray(graph_points, dtype=float).reshape(-1, 3)
    node_ids = np.asarray(list(node_path or []), dtype=int).reshape(-1)
    if node_ids.size == 0:
        return np.empty((0, 3), dtype=float)
    return np.asarray(pts[node_ids], dtype=float)


def _path_info_from_points(path_points):
    pts = np.asarray(path_points, dtype=float).reshape(-1, 3)
    if len(pts) == 0:
        return {
            "path_index": 0,
            "start_node": -1,
            "end_node": -1,
            "start_point": [0.0, 0.0, 0.0],
            "end_point": [0.0, 0.0, 0.0],
            "direction_vector": [0.0, 0.0, 0.0],
            "direction_text": "",
            "fork_ids": [],
            "fork_roles": [],
            "incoming_path_ids": [],
            "outgoing_path_ids": [],
        }
    direction = pts[-1] - pts[0] if len(pts) >= 2 else np.zeros(3, dtype=float)
    norm = float(np.linalg.norm(direction))
    unit = direction / norm if norm > 1e-12 else np.zeros(3, dtype=float)
    return {
        "path_index": 0,
        "start_node": -1,
        "end_node": -1,
        "start_point": pts[0].tolist(),
        "end_point": pts[-1].tolist(),
        "direction_vector": unit.tolist(),
        "direction_text": _vector_orientation_text(unit),
        "fork_ids": [],
        "fork_roles": [],
        "incoming_path_ids": [],
        "outgoing_path_ids": [],
    }


def _orient_graph_path_by_flow(node_path, graph_points, flow_xyzt3=None, segmask_binary_4d=None,
                               spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    nodes = [int(x) for x in list(node_path or [])]
    if len(nodes) < 2:
        return nodes
    try:
        from .branch import _orient_node_paths_by_flow
        oriented = _orient_node_paths_by_flow(
            [nodes],
            np.asarray(graph_points, dtype=float),
            flow_xyzt3=flow_xyzt3,
            segmask_binary_4d=segmask_binary_4d,
            spacing=spacing,
            origin=origin,
        )
        if oriented and len(oriented[0]) == len(nodes):
            return [int(x) for x in oriented[0]]
    except Exception:
        pass
    return nodes


def _longest_endpoint_pair_path(graph, *, flow_xyzt3=None, segmask_binary_4d=None, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    G = _edge_weighted_graph(graph)
    pts = np.asarray(getattr(graph, "points", []), dtype=float).reshape(-1, 3)
    if G.number_of_nodes() == 0 or len(pts) == 0:
        return None
    endpoints = sorted(int(node) for node, deg in G.degree() if int(deg) == 1)
    if len(endpoints) < 2:
        return None
    best = None
    for i, src in enumerate(endpoints):
        for dst in endpoints[i + 1:]:
            try:
                node_path = nx.shortest_path(G, source=int(src), target=int(dst), weight="weight")
                length = float(nx.path_weight(G, node_path, weight="weight"))
            except Exception:
                continue
            if best is None or length > best["length_mm"]:
                best = {
                    "start_node": int(src),
                    "end_node": int(dst),
                    "length_mm": float(length),
                    "node_path": [int(x) for x in node_path],
                }
    if best is None:
        return None
    oriented_nodes = _orient_graph_path_by_flow(
        best["node_path"],
        pts,
        flow_xyzt3=flow_xyzt3,
        segmask_binary_4d=segmask_binary_4d,
        spacing=spacing,
        origin=origin,
    )
    oriented_points = _node_path_to_points(pts, oriented_nodes)
    if len(oriented_points) < 2:
        return None
    length_mm = _path_length_mm(oriented_points)
    start_node = int(oriented_nodes[0])
    end_node = int(oriented_nodes[-1])
    return {
        "start_node": start_node,
        "end_node": end_node,
        "length_mm": float(length_mm),
        "node_path": oriented_nodes,
        "path_points": oriented_points,
        "path_info": {
            **_path_info_from_points(oriented_points),
            "start_node": start_node,
            "end_node": end_node,
            "endpoint_pair": [start_node, end_node],
        },
    }


def _plot_pwv_axes(ax_pwv, result, *, color="#2b8a3e", fit_color="#f08c00"):
    ax_pwv.clear()
    ax_pwv.set_title(str(result.get("name", "PWV")))
    ax_pwv.set_xlabel("Slice Position (mm)")
    ax_pwv.set_ylabel(str(result.get("transit_time_label", "Arrival Time (ms)")))
    ax_pwv.grid(True, alpha=0.25)
    positions = np.asarray(result.get("position_mm", []), dtype=float).reshape(-1)
    feet = np.asarray(result.get("time_to_foot_ms", []), dtype=float).reshape(-1)
    if positions.size > 0 and feet.size > 0:
        ax_pwv.scatter(positions, feet, color=color, label="Planes")
        slope = result.get("fit_slope_ms_per_mm")
        intercept = result.get("fit_intercept_ms")
        if slope is not None and intercept is not None:
            xfit = np.linspace(float(np.min(positions)), float(np.max(positions)), 100)
            yfit = float(intercept) + float(slope) * xfit
            label = "Fit"
            pwv = result.get("pwv_m_s")
            if pwv is not None:
                label = f"Fit PWV={float(pwv):.3g} m/s"
            ax_pwv.plot(xfit, yfit, color=fit_color, linewidth=2.0, label=label)
            if pwv is not None:
                ax_pwv.text(
                    0.02,
                    0.98,
                    label,
                    transform=ax_pwv.transAxes,
                    va="top",
                    ha="left",
                    fontsize=10,
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
                )
        ax_pwv.legend(loc="best")
    else:
        ax_pwv.text(0.5, 0.5, str(result.get("message", "No valid PWV points")), ha="center", va="center", transform=ax_pwv.transAxes)


def _plot_plane_flowrate_axes(ax_flow, result, *, color="#2b8a3e"):
    ax_flow.clear()
    ax_flow.set_title("Plane Flowrate")
    ax_flow.set_xlabel("Cardiac Phase")
    ax_flow.set_ylabel("Flowrate (mL/s)")
    ax_flow.grid(True, alpha=0.25)
    plane_rows = list(result.get("planes", []) or [])
    drawn = 0
    for idx, row in enumerate(plane_rows):
        waveform = np.asarray(row.get("flowrate_mL_s", []), dtype=float).reshape(-1)
        if waveform.size == 0 or not np.any(np.isfinite(waveform)):
            continue
        xvals = np.arange(waveform.size, dtype=float)
        distance = row.get("distance_mm")
        label = f"{float(distance):.1f} mm" if distance is not None else f"plane {idx}"
        alpha = 0.9 if idx == 0 or idx == len(plane_rows) - 1 else 0.55
        linewidth = 1.8 if idx == 0 or idx == len(plane_rows) - 1 else 1.1
        ax_flow.plot(xvals, waveform, color=color, alpha=alpha, linewidth=linewidth, label=label)
        foot_ms = row.get("time_to_foot_ms")
        rr_ms = result.get("rr_ms")
        if foot_ms is not None and rr_ms is not None and waveform.size > 0 and float(rr_ms) > 0:
            foot_phase = float(foot_ms) * float(waveform.size) / float(rr_ms)
            ax_flow.axvline(foot_phase, color=color, alpha=0.12, linewidth=0.8)
        drawn += 1
    if drawn == 0:
        ax_flow.text(0.5, 0.5, "No plane flowrate waveforms", ha="center", va="center", transform=ax_flow.transAxes)
        return
    if drawn <= 8:
        ax_flow.legend(loc="best", fontsize=8)


def _fit_pwv(position_mm, foot_time_ms):
    xvals = np.asarray(position_mm, dtype=float).reshape(-1)
    yvals = np.asarray(foot_time_ms, dtype=float).reshape(-1)
    if xvals.size < 2 or yvals.size < 2:
        return None
    slope, intercept = np.polyfit(xvals, yvals, 1)
    slope = float(slope)
    intercept = float(intercept)
    if not np.isfinite(slope) or abs(slope) <= 1e-12:
        return None
    pred = slope * xvals + intercept
    ss_res = float(np.sum((yvals - pred) ** 2))
    ss_tot = float(np.sum((yvals - np.mean(yvals)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
    pwv = 1.0 / slope if slope > 0.0 else None
    return {
        "fit_slope_ms_per_mm": slope,
        "fit_intercept_ms": intercept,
        "fit_r2": float(r2),
        "pwv_m_s": None if pwv is None or not np.isfinite(pwv) else float(pwv),
    }


def compute_pwv_groups(
    flow_xyzt3,
    segmask_raw,
    spacing,
    origin,
    rr_ms,
    skeleton_params: SkeletonParams,
    pwv_params: PwvParams,
    *,
    out_dir="",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not bool(getattr(pwv_params, "enabled", False)):
        return [], []
    if segmask_raw is None or flow_xyzt3 is None:
        return [], []

    labels_3d = majority_vote_labels_3d(segmask_raw)
    flow = np.asarray(flow_xyzt3, dtype=np.float32)
    resolution = np.asarray(spacing, dtype=float).reshape(3)
    origin = np.asarray(origin, dtype=float).reshape(3)
    time_count = max(1, int(flow.shape[3]))
    results: List[Dict[str, Any]] = []
    scene_planes: List[Dict[str, Any]] = []
    out_dir = str(out_dir or "")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for group in list(getattr(pwv_params, "groups", []) or []):
        group_name = str(getattr(group, "name", "") or "pwv")
        labels = sorted({int(x) for x in list(getattr(group, "labels", []) or []) if int(x) != 0})
        result: Dict[str, Any] = {
            "name": group_name,
            "labels": labels,
            "enabled": True,
            "waveform_key": str(getattr(pwv_params, "waveform_key", "flowrate_mL_s") or "flowrate_mL_s"),
            "rr_ms": float(rr_ms),
        }
        if not labels:
            result.update({"status": "skipped", "message": "no labels configured"})
            results.append(result)
            continue
        group_mask = np.isin(labels_3d, labels)
        if not np.any(group_mask):
            result.update({"status": "skipped", "message": "group mask is empty"})
            results.append(result)
            continue
        group_mask = largest_connected_component(group_mask)
        if not np.any(group_mask):
            result.update({"status": "skipped", "message": "largest connected component is empty"})
            results.append(result)
            continue
        processed = preprocess_mask_for_skeleton(group_mask, skeleton_params, resolution=resolution)
        processed = largest_connected_component(processed)
        if not np.any(processed):
            result.update({"status": "skipped", "message": "preprocessed group mask is empty"})
            results.append(result)
            continue
        skeleton_points, _mask = generate_skeleton_from_mask3d(processed, resolution)
        skeleton_points = np.asarray(skeleton_points, dtype=float).reshape(-1, 3)
        if skeleton_points.size == 0:
            result.update({"status": "skipped", "message": "no skeleton points"})
            results.append(result)
            continue
        graph = build_graph_from_points(skeleton_points, resolution)
        group_binary = _repeat_mask_to_time(group_mask, time_count)
        local_flow = flow * group_binary[..., None]
        branch_labels, local_paths, _node_paths, local_path_info, _forks = segment_vessels_from_graph_and_mask(
            processed,
            graph,
            resolution,
            flow_xyzt3=local_flow,
            segmask_binary_4d=group_binary,
            origin=origin,
        )
        longest_graph_path = _longest_endpoint_pair_path(
            graph,
            flow_xyzt3=local_flow,
            segmask_binary_4d=group_binary,
            spacing=resolution,
            origin=origin,
        )
        if longest_graph_path is not None:
            longest_path = np.asarray(longest_graph_path["path_points"], dtype=float)
            longest_len = float(longest_graph_path["length_mm"])
            chosen_info = dict(longest_graph_path["path_info"])
            result["longest_path_length_mm"] = float(longest_len)
            result["longest_path_endpoint_pair"] = [int(x) for x in longest_graph_path["path_info"].get("endpoint_pair", [])]
            result["longest_path_method"] = "endpoint_pair_edge_weighted"
        else:
            longest_idx, longest_path, longest_len = _longest_path(local_paths)
            result["longest_path_length_mm"] = float(longest_len)
            result["longest_path_method"] = "segment_vessels_longest_path_fallback"
            chosen_info = dict(local_path_info[longest_idx]) if 0 <= longest_idx < len(local_path_info) else {"path_index": 0}
        if longest_path is None or longest_len <= 0.0:
            result.update({"status": "skipped", "message": "no centerline path found"})
            results.append(result)
            continue

        chosen_info["path_index"] = 0
        interval_mm = max(0.1, float(getattr(pwv_params, "plane_interval_mm", 10.0) or 10.0))
        pwv_planes, smooth_paths = generate_planes_from_paths(
            [longest_path],
            cross_section_distance=interval_mm,
            start_distance=float(getattr(pwv_params, "start_distance", 0.0) or 0.0),
            end_distance=float(getattr(pwv_params, "end_distance", 0.0) or 0.0),
            smoothing_window=int(getattr(pwv_params, "smoothing_window", 15) or 15) * int(getattr(pwv_params, "inter_time", 10) or 10),
            smoothing_polyorder=int(getattr(pwv_params, "smoothing_polyorder", 2) or 2),
            inter_time=int(getattr(pwv_params, "inter_time", 10) or 10),
            use_center_plane=False,
        )
        if not pwv_planes:
            result.update({"status": "skipped", "message": "no PWV planes generated"})
            results.append(result)
            continue

        for plane in pwv_planes:
            plane.path_index = 0
            plane.label = 0
            plane.group_name = group_name
        metrics = compute_plane_metrics(
            flow,
            group_binary,
            resolution,
            origin,
            pwv_planes,
            RR=float(rr_ms),
            branch_labels_3d=np.asarray(branch_labels, dtype=np.int16),
            path_info=[chosen_info],
            forks=[],
            paths=[np.asarray(smooth_paths[0] if smooth_paths else longest_path, dtype=float)],
            return_qc=False,
        )
        waveform_key = str(result["waveform_key"])
        transit_time_method = str(getattr(pwv_params, "transit_time_method", "foot_to_foot") or "foot_to_foot")
        foot_method = str(getattr(pwv_params, "foot_method", "tangent") or "tangent")
        xcorr_window = str(getattr(pwv_params, "xcorr_window", "full") or "full")
        xcorr_interp_factor = max(1, int(getattr(pwv_params, "xcorr_interp_factor", 10) or 10))
        allow_cycle_wrap = bool(getattr(pwv_params, "allow_cycle_wrap", True))
        result["transit_time_method"] = transit_time_method
        result["foot_method"] = foot_method
        result["xcorr_window"] = xcorr_window
        result["xcorr_interp_factor"] = xcorr_interp_factor
        result["allow_cycle_wrap"] = allow_cycle_wrap
        result["transit_time_label"] = "Cross-correlation Delay (ms)" if transit_time_method == "cross_correlation" else "Arrival Time (ms)"
        plane_rows: List[Dict[str, Any]] = []
        raw_arrival_times: List[Any] = []
        plane_positions_all: List[float] = []
        flowrate_waveforms: List[np.ndarray] = []
        for plane, metric in zip(pwv_planes, metrics):
            waveform = metric.get(waveform_key, [])
            foot_ms, foot_meta = detect_waveform_foot_time_ms(
                waveform,
                rr_ms=float(rr_ms),
                window=int(getattr(pwv_params, "foot_savgol_window", 5) or 5),
                polyorder=int(getattr(pwv_params, "foot_savgol_polyorder", 2) or 2),
                method=foot_method,
                threshold_percent=float(getattr(pwv_params, "foot_threshold_percent", 10.0) or 10.0),
                allow_cycle_wrap=allow_cycle_wrap,
            )
            flowrate_waveform = np.asarray(metric.get("flowrate_mL_s", []), dtype=float).reshape(-1)
            row = {
                "distance_mm": float(getattr(plane, "distance", 0.0)),
                "center": np.asarray(getattr(plane, "center", [0.0, 0.0, 0.0]), dtype=float).tolist(),
                "normal": np.asarray(getattr(plane, "normal", [1.0, 0.0, 0.0]), dtype=float).tolist(),
                "time_to_foot_ms": None if foot_ms is None else float(foot_ms),
                "arrival_time_ms": None,
                "waveform": [float(x) for x in list(np.asarray(waveform, dtype=float).reshape(-1))],
                "flowrate_mL_s": [float(x) for x in list(flowrate_waveform)],
                "foot_detection": foot_meta,
                "cross_correlation": {},
            }
            plane_rows.append(row)
            scene_planes.append(
                {
                    "group_name": group_name,
                    "distance_mm": float(getattr(plane, "distance", 0.0)),
                    "center": row["center"],
                    "normal": row["normal"],
                }
            )
            plane_positions_all.append(float(getattr(plane, "distance", 0.0)))
            raw_arrival_times.append(None if foot_ms is None else float(foot_ms))
            flowrate_waveforms.append(flowrate_waveform)

        if transit_time_method == "cross_correlation":
            arrival_times = [None] * len(plane_rows)
            valid_idxs = [idx for idx, wf in enumerate(flowrate_waveforms) if wf.size >= 3 and np.any(np.isfinite(wf))]
            if valid_idxs:
                anchor_idx = valid_idxs[0]
                arrival_times[anchor_idx] = 0.0
                plane_rows[anchor_idx]["arrival_time_ms"] = 0.0
                ref_waveform = flowrate_waveforms[anchor_idx]
                for idx in valid_idxs[1:]:
                    delay_ms, cc_meta = compute_cross_correlation_delay_ms(
                        ref_waveform,
                        flowrate_waveforms[idx],
                        float(rr_ms),
                        window=xcorr_window,
                        allow_cycle_wrap=allow_cycle_wrap,
                        interp_factor=xcorr_interp_factor,
                    )
                    plane_rows[idx]["cross_correlation"] = cc_meta
                    if delay_ms is None:
                        continue
                    arrival_times[idx] = float(delay_ms)
                    plane_rows[idx]["arrival_time_ms"] = float(delay_ms)
                plane_rows[anchor_idx]["cross_correlation"] = {
                    "window": xcorr_window,
                    "delay_ms": 0.0,
                    "is_anchor": True,
                    "interp_factor": xcorr_interp_factor,
                    "allow_cycle_wrap": allow_cycle_wrap,
                }
            valid_pairs = [(plane_positions_all[i], arrival_times[i]) for i in range(len(arrival_times)) if arrival_times[i] is not None]
        else:
            arrival_times = list(raw_arrival_times)
            valid_indices = [i for i, val in enumerate(arrival_times) if val is not None]
            if valid_indices:
                ref_time = float(arrival_times[valid_indices[0]])
                for i in valid_indices:
                    arrival_times[i] = _wrap_time_to_reference_ms(arrival_times[i], ref_time, float(rr_ms)) if allow_cycle_wrap else float(arrival_times[i])
                    plane_rows[i]["arrival_time_ms"] = float(arrival_times[i])
            valid_pairs = [(plane_positions_all[i], arrival_times[i]) for i in range(len(arrival_times)) if arrival_times[i] is not None]

        position_mm = [float(x) for x, _ in valid_pairs]
        arrival_time_ms = [float(y) for _, y in valid_pairs]
        result["plane_count"] = int(len(pwv_planes))
        result["valid_plane_count"] = int(len(position_mm))
        result["planes"] = plane_rows
        result["position_mm"] = [float(x) for x in position_mm]
        result["time_to_foot_ms"] = [float(x) for x in arrival_time_ms]
        result["arrival_time_ms"] = [float(x) for x in arrival_time_ms]
        minimum_valid = max(2, int(getattr(pwv_params, "minimum_valid_planes", 2) or 2))
        if len(position_mm) < minimum_valid:
            result.update({"status": "skipped", "message": "not enough valid PWV planes"})
        else:
            fit = _fit_pwv(position_mm, arrival_time_ms)
            if fit is None or fit.get("pwv_m_s") is None:
                result.update({"status": "skipped", "message": "PWV fit failed"})
            else:
                result.update(fit)
                result["status"] = "ok"
                result["message"] = f"PWV={float(fit['pwv_m_s']):.4f} m/s via {transit_time_method}"
        if out_dir:
            plot_path = os.path.join(out_dir, f"pwv_{_sanitize_name(group_name)}.png")
            try:
                result["plot_file"] = save_pwv_plot(
                    result,
                    plot_path,
                    color=str(getattr(pwv_params, "plot_color", "#2b8a3e") or "#2b8a3e"),
                    fit_color=str(getattr(pwv_params, "fit_color", "#f08c00") or "#f08c00"),
                    dpi=int(getattr(pwv_params, "plot_dpi", 160) or 160),
                )
            except Exception as exc:
                result["plot_file"] = ""
                result["plot_error"] = f"{type(exc).__name__}: {exc}"
        results.append(result)
    return results, scene_planes
