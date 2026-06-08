import json
import os
import re
from typing import Any, Dict, List, Tuple

import numpy as np

from ..core.models import PwvParams, SkeletonParams
from .branch import segment_vessels_from_graph_and_mask
from .graph import build_graph_from_points
from .metrics import compute_plane_metrics
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


def detect_waveform_foot_time_ms(waveform, rr_ms, window=5, polyorder=2):
    values = np.asarray(waveform, dtype=float).reshape(-1)
    if values.size < 3 or not np.any(np.isfinite(values)):
        return None, {}
    clean = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.any(clean > 0.0) and np.any(clean < 0.0):
        clean = -clean
    smooth = _smooth_signal(clean, window, polyorder)
    nt = max(1, smooth.size)
    dt_ms = float(rr_ms) / float(nt)
    times_ms = np.arange(nt, dtype=float) * dt_ms
    peak_idx = int(np.argmax(smooth))
    if peak_idx <= 0:
        return None, {"times_ms": times_ms.tolist(), "smooth_waveform": smooth.tolist()}
    baseline_idx = int(np.argmin(smooth[:peak_idx + 1]))
    if baseline_idx >= peak_idx:
        return None, {"times_ms": times_ms.tolist(), "smooth_waveform": smooth.tolist()}
    deriv = np.gradient(smooth, dt_ms)
    search = deriv[baseline_idx:peak_idx + 1]
    if search.size == 0:
        return None, {"times_ms": times_ms.tolist(), "smooth_waveform": smooth.tolist()}
    slope_idx = int(np.argmax(search)) + baseline_idx
    slope = float(deriv[slope_idx])
    if not np.isfinite(slope) or slope <= 1e-12:
        return None, {"times_ms": times_ms.tolist(), "smooth_waveform": smooth.tolist()}
    baseline = float(smooth[baseline_idx])
    foot_time = float(times_ms[slope_idx] - (float(smooth[slope_idx]) - baseline) / slope)
    foot_time = float(np.clip(foot_time, float(times_ms[0]), float(times_ms[-1])))
    return foot_time, {
        "times_ms": times_ms.tolist(),
        "smooth_waveform": smooth.tolist(),
        "baseline_index": int(baseline_idx),
        "peak_index": int(peak_idx),
        "slope_index": int(slope_idx),
        "baseline_value": baseline,
        "slope_value": slope,
    }


def save_pwv_plot(result, out_path, color="#2b8a3e", fit_color="#f08c00", dpi=160):
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except Exception as exc:
        raise RuntimeError("PWV plotting requires matplotlib") from exc
    fig = Figure(figsize=(6.0, 4.0), dpi=int(dpi))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    positions = np.asarray(result.get("position_mm", []), dtype=float)
    feet = np.asarray(result.get("time_to_foot_ms", []), dtype=float)
    ax.set_title(str(result.get("name", "PWV")))
    ax.set_xlabel("Slice Position (mm)")
    ax.set_ylabel("Time-to-Foot (ms)")
    if positions.size > 0 and feet.size > 0:
        ax.scatter(positions, feet, color=color, label="Planes")
        slope = result.get("fit_slope_ms_per_mm")
        intercept = result.get("fit_intercept_ms")
        if slope is not None and intercept is not None:
            xfit = np.linspace(float(np.min(positions)), float(np.max(positions)), 100)
            yfit = float(intercept) + float(slope) * xfit
            label = f"Fit PWV={float(result.get('pwv_m_s', float('nan'))):.3g} m/s"
            ax.plot(xfit, yfit, color=fit_color, linewidth=2.0, label=label)
            ax.text(
                0.02,
                0.98,
                label,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=10,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )
        ax.legend(loc="best")
    else:
        ax.text(0.5, 0.5, str(result.get("message", "No valid PWV points")), ha="center", va="center")
    ax.grid(True, alpha=0.25)
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
            "waveform_key": str(getattr(pwv_params, "waveform_key", "flowrate_signed_mL_s") or "flowrate_signed_mL_s"),
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
        longest_idx, longest_path, longest_len = _longest_path(local_paths)
        result["longest_path_length_mm"] = float(longest_len)
        if longest_idx < 0 or longest_path is None or longest_len <= 0.0:
            result.update({"status": "skipped", "message": "no centerline path found"})
            results.append(result)
            continue

        chosen_info = dict(local_path_info[longest_idx]) if 0 <= longest_idx < len(local_path_info) else {"path_index": 0}
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

        branch_label = int(longest_idx) + 1
        for plane in pwv_planes:
            plane.path_index = 0
            plane.label = branch_label
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
        position_mm: List[float] = []
        foot_times_ms: List[float] = []
        plane_rows: List[Dict[str, Any]] = []
        for plane, metric in zip(pwv_planes, metrics):
            waveform = metric.get(waveform_key, [])
            foot_ms, foot_meta = detect_waveform_foot_time_ms(
                waveform,
                rr_ms=float(rr_ms),
                window=int(getattr(pwv_params, "foot_savgol_window", 5) or 5),
                polyorder=int(getattr(pwv_params, "foot_savgol_polyorder", 2) or 2),
            )
            row = {
                "distance_mm": float(getattr(plane, "distance", 0.0)),
                "center": np.asarray(getattr(plane, "center", [0.0, 0.0, 0.0]), dtype=float).tolist(),
                "normal": np.asarray(getattr(plane, "normal", [1.0, 0.0, 0.0]), dtype=float).tolist(),
                "time_to_foot_ms": None if foot_ms is None else float(foot_ms),
                "waveform": [float(x) for x in list(np.asarray(waveform, dtype=float).reshape(-1))],
                "foot_detection": foot_meta,
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
            if foot_ms is None:
                continue
            position_mm.append(float(getattr(plane, "distance", 0.0)))
            foot_times_ms.append(float(foot_ms))
        result["plane_count"] = int(len(pwv_planes))
        result["valid_plane_count"] = int(len(position_mm))
        result["planes"] = plane_rows
        result["position_mm"] = [float(x) for x in position_mm]
        result["time_to_foot_ms"] = [float(x) for x in foot_times_ms]
        minimum_valid = max(2, int(getattr(pwv_params, "minimum_valid_planes", 2) or 2))
        if len(position_mm) < minimum_valid:
            result.update({"status": "skipped", "message": "not enough valid PWV planes"})
        else:
            fit = _fit_pwv(position_mm, foot_times_ms)
            if fit is None or fit.get("pwv_m_s") is None:
                result.update({"status": "skipped", "message": "PWV fit failed"})
            else:
                result.update(fit)
                result["status"] = "ok"
                result["message"] = f"PWV={float(fit['pwv_m_s']):.4f} m/s"
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
