import numpy as np

from ..core.models import PlaneData
from .paths import inter_points, smooth_path_savgol


def _build_plane(path_smooth, cum, target, path_i):
    j = int(np.searchsorted(cum, target, side="right") - 1)
    j = min(max(0, j), len(path_smooth) - 2)
    seg = path_smooth[j + 1] - path_smooth[j]
    seglen = np.linalg.norm(seg) + 1e-12
    alpha = (float(target) - float(cum[j])) / seglen
    center = path_smooth[j] + alpha * seg
    i0 = max(0, j - 1)
    i1 = min(len(path_smooth) - 1, j + 2)
    tangent = path_smooth[i1] - path_smooth[i0]
    normal = tangent / (np.linalg.norm(tangent) + 1e-12)
    return PlaneData(
        center=center,
        normal=normal,
        label=path_i + 1,
        path_index=path_i,
        distance=float(target),
    )


def _fallback_mid_plane(path_smooth, cum, total, path_i):
    if len(path_smooth) < 2:
        return None
    target = float(total) * 0.5
    return _build_plane(path_smooth, cum, target, path_i)


def _effective_interval(total, start_distance, end_distance):
    start = max(0.0, float(start_distance))
    end = max(0.0, float(end_distance))
    lo = min(start, float(total))
    hi = max(lo, float(total) - end)
    return lo, hi


def _target_distances(total, *, plane_mode, plane_count, cross_section_distance, start_distance, end_distance, anchor, anchor_offset_mm):
    total = float(total)
    if total <= 0.0:
        return []

    lo, hi = _effective_interval(total, start_distance, end_distance)
    mode = str(plane_mode or "count").strip().lower()

    if mode == "distance":
        step = max(float(cross_section_distance), 1e-6)
        if hi <= lo:
            return [float(np.clip(0.5 * (lo + hi), 0.0, total))]
        targets = np.arange(lo, hi + 1e-8, step, dtype=float).tolist()
        return [float(np.clip(x, 0.0, total)) for x in targets] or [float(np.clip(0.5 * (lo + hi), 0.0, total))]

    if mode == "anchored_offset":
        offset = max(float(anchor_offset_mm), 0.0)
        which = str(anchor or "end").strip().lower()
        if which == "start":
            target = lo + offset
        else:
            target = hi - offset
        return [float(np.clip(target, lo, hi if hi >= lo else total))]

    count = max(1, int(plane_count) or 1)
    if hi <= lo:
        return [float(np.clip(0.5 * (lo + hi), 0.0, total))]
    if count == 1:
        return [float(np.clip(0.5 * (lo + hi), 0.0, total))]
    return [float(x) for x in np.linspace(lo, hi, num=count)]


def generate_planes_from_paths(
    paths,
    cross_section_distance=5.0,
    start_distance=0.0,
    end_distance=0.0,
    smoothing_window=15,
    smoothing_polyorder=3,
    inter_time=9,
    use_center_plane=None,
    plane_mode="count",
    plane_count=1,
    anchor="end",
    anchor_offset_mm=5.0,
):
    if use_center_plane is not None:
        plane_mode = "count" if bool(use_center_plane) else "distance"
        if bool(use_center_plane):
            plane_count = 1

    planes = []
    smooth_paths = []

    for path_i, path in enumerate(paths):
        path = inter_points(path, time=inter_time)
        if len(path) < 2:
            smooth_paths.append(path)
            continue

        path_smooth = smooth_path_savgol(path, window=smoothing_window, polyorder=smoothing_polyorder)
        smooth_paths.append(path_smooth)
        if len(path_smooth) < 2:
            continue

        seglens = np.linalg.norm(np.diff(path_smooth, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seglens)])
        total = float(cum[-1])

        targets = _target_distances(
            total,
            plane_mode=plane_mode,
            plane_count=plane_count,
            cross_section_distance=cross_section_distance,
            start_distance=start_distance,
            end_distance=end_distance,
            anchor=anchor,
            anchor_offset_mm=anchor_offset_mm,
        )
        if not targets:
            fallback = _fallback_mid_plane(path_smooth, cum, total, path_i)
            if fallback is not None:
                planes.append(fallback)
            continue

        built = 0
        for target in targets:
            planes.append(_build_plane(path_smooth, cum, target, path_i))
            built += 1

        if built == 0:
            fallback = _fallback_mid_plane(path_smooth, cum, total, path_i)
            if fallback is not None:
                planes.append(fallback)

    return planes, smooth_paths
