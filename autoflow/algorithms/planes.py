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


def _sample_path_labels(path, labels, spacing, origin):
    """Sample a 3-D/4-D segmentation on a physical centerline path."""
    volume = np.asarray(labels)
    if volume.ndim == 4:
        # Temporal labels are reduced by the same majority rule as preprocessing.
        values = np.unique(volume)
        counts = np.stack([(volume == value).sum(axis=3) for value in values], axis=-1)
        winners = np.argmax(counts, axis=-1)
        volume3d = np.zeros(volume.shape[:3], dtype=volume.dtype)
        for i, value in enumerate(values):
            volume3d[winners == i] = value
        volume = volume3d
    if volume.ndim != 3 or len(path) == 0:
        return np.zeros(len(path), dtype=np.int16)
    pts = np.asarray(path, dtype=float).reshape(-1, 3)
    spacing = np.asarray(spacing, dtype=float).reshape(3)
    origin = np.asarray(origin, dtype=float).reshape(3)
    ijk = np.rint((pts - origin) / (spacing + 1e-12)).astype(int)
    for axis in range(3):
        ijk[:, axis] = np.clip(ijk[:, axis], 0, volume.shape[axis] - 1)
    return np.asarray(volume[ijk[:, 0], ijk[:, 1], ijk[:, 2]], dtype=np.int16)


def _label_runs(path, labels, spacing, origin, neighborhood=1):
    """Return contiguous non-zero label runs as (label, i0, i1, length_mm)."""
    pts = np.asarray(path, dtype=float).reshape(-1, 3)
    sampled = _sample_path_labels(pts, labels, spacing, origin)
    if len(sampled) == 0:
        return sampled, []
    # A small local majority vote suppresses isolated voxel-label flicker.
    radius = max(0, int(neighborhood))
    if radius:
        smooth = sampled.copy()
        for i in range(len(sampled)):
            lo, hi = max(0, i - radius), min(len(sampled), i + radius + 1)
            vals, counts = np.unique(sampled[lo:hi][sampled[lo:hi] > 0], return_counts=True)
            if len(vals):
                smooth[i] = vals[int(np.argmax(counts))]
        sampled = smooth
    seglen = np.linalg.norm(np.diff(pts, axis=0), axis=1) if len(pts) > 1 else np.empty(0)
    runs = []
    start = 0
    for i in range(1, len(sampled) + 1):
        if i == len(sampled) or int(sampled[i]) != int(sampled[start]):
            label = int(sampled[start])
            if label > 0:
                # Include the segment leaving the final sample in the run.
                length = float(np.sum(seglen[start:min(i, len(seglen))]))
                if i - start > 1 and length <= 1e-12:
                    length = float(np.linalg.norm(pts[i - 1] - pts[start]))
                runs.append((label, start, i - 1, max(length, 0.0)))
            start = i
    return sampled, runs


def filter_paths_by_segmentation(paths, segmentation_labels, spacing=(1, 1, 1), origin=(0, 0, 0),
                                 path_info=None, forks=None, inter_time=10):
    """Clip each graph path to its topology-aware owner segmentation label.

    The owner is selected from contiguous physical-length runs.  A path attached to
    one fork keeps the run on the free endpoint side; otherwise the longest run wins.
    This avoids classifying a short branch from a junction by a few voxels of the
    parent vessel label.
    """
    if segmentation_labels is None:
        return list(paths), [{"segmentation_filter": False} for _ in paths]
    infos = list(path_info or [])
    filtered, metadata = [], []
    for idx, raw_path in enumerate(paths):
        path = np.asarray(raw_path, dtype=float).reshape(-1, 3)
        if len(path) < 2:
            filtered.append(path)
            metadata.append({"segmentation_filter": True, "owner_label": 0,
                             "owner_label_source": "insufficient_path", "owner_label_confidence": 0.0,
                             "original_path_length_mm": 0.0, "retained_label_length_mm": 0.0})
            continue
        dense = inter_points(path, time=max(2, int(inter_time)))
        sampled, runs = _label_runs(dense, segmentation_labels, spacing, origin)
        original_length = float(np.sum(np.linalg.norm(np.diff(dense, axis=0), axis=1)))
        if not runs:
            # If labels do not overlap this path (e.g. a binary mask or a
            # registration mismatch), retain geometry and mark the owner as
            # unknown so the pipeline remains usable.
            filtered.append(path)
            metadata.append({"segmentation_filter": True, "owner_label": 0,
                             "owner_label_source": "no_label_run", "owner_label_confidence": 0.0,
                             "original_path_length_mm": original_length, "retained_label_length_mm": 0.0})
            continue
        info = infos[idx] if idx < len(infos) and isinstance(infos[idx], dict) else {}
        roles = {str(item.get("role", "")) for item in info.get("fork_roles", []) if isinstance(item, dict)}
        junction_endpoint = str(info.get("junction_endpoint", "") or "").strip().lower()
        # Incoming paths end at a fork; outgoing paths start at one.  The
        # topology-only pre-pass does not have reliable roles yet, so use the
        # endpoint attached to a degree-based junction as a direction-neutral
        # free-end hint.
        if not roles and junction_endpoint == "end":
            candidates = [r for r in runs if r[1] <= len(dense) * 0.5]
            source = "free_start_of_topology_junction"
        elif not roles and junction_endpoint == "start":
            candidates = [r for r in runs if r[2] >= len(dense) * 0.5]
            source = "free_end_of_topology_junction"
        elif "incoming" in roles and "outgoing" not in roles:
            candidates = [r for r in runs if r[1] <= len(dense) * 0.5]
            source = "free_start_of_incoming"
        elif "outgoing" in roles and "incoming" not in roles:
            candidates = [r for r in runs if r[2] >= len(dense) * 0.5]
            source = "free_end_of_outgoing"
        else:
            candidates = list(runs)
            source = "longest_contiguous_run"
        if not candidates:
            candidates = list(runs)
            source = "longest_contiguous_run_fallback"
        owner = max(candidates, key=lambda r: r[3])
        label, i0, i1, retained_length = owner
        # Keep only samples belonging to the selected owner label.
        lo = max(0, i0)
        hi = min(len(dense) - 1, i1)
        clipped = dense[lo:hi + 1]
        if len(clipped) < 2:
            clipped = path
        filtered.append(np.asarray(clipped, dtype=float))
        total_run_length = float(sum(r[3] for r in runs))
        confidence = float(retained_length / (total_run_length + 1e-12)) if total_run_length > 0 else 0.0
        metadata.append({"segmentation_filter": True, "owner_label": int(label),
                         "owner_label_source": source, "owner_label_confidence": confidence,
                         "original_path_length_mm": original_length,
                         "retained_label_length_mm": float(retained_length)})
    return filtered, metadata


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


def _spacing_targets(lo, hi, spacing, count):
    if hi <= lo:
        return [float(0.5 * (lo + hi))]
    targets = np.arange(lo, hi + 1e-8, max(float(spacing), 1e-6), dtype=float).tolist()
    if int(count) != -1:
        targets = targets[:max(1, int(count))]
    return [float(value) for value in targets] or [float(0.5 * (lo + hi))]


def _junction_anchor_endpoint(path, fork_points, fallback_anchor):
    """Choose the path endpoint occupied by a known graph junction."""
    pts = np.asarray(path, dtype=float).reshape(-1, 3)
    fallback = 0.0 if str(fallback_anchor or "end").strip().lower() == "start" else 1.0
    if len(pts) < 2:
        return fallback
    candidates = []
    for point in fork_points or []:
        try:
            point = np.asarray(point, dtype=float).reshape(3)
        except (TypeError, ValueError):
            continue
        candidates.append((float(np.linalg.norm(pts[0] - point)), 0.0))
        candidates.append((float(np.linalg.norm(pts[-1] - point)), 1.0))
    if not candidates:
        return fallback
    distance, endpoint = min(candidates, key=lambda item: item[0])
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    steps = steps[steps > 1e-12]
    tolerance = max(2.0, 2.0 * float(np.median(steps)) if steps.size else 2.0)
    return endpoint if distance <= tolerance else fallback


def _target_distances(total, *, plane_mode, plane_count, cross_section_distance,
                      start_distance, end_distance, anchor, anchor_offset_mm,
                      direction="toward_end", spacing_mode="distance", spacing_ratio=0.25,
                      junction_endpoint=None):
    total = float(total)
    if total <= 0.0:
        return []

    lo, hi = _effective_interval(total, start_distance, end_distance)
    mode = str(plane_mode or "fixed_step").strip().lower()

    count = int(plane_count)
    if count != -1:
        count = max(1, count)
    spacing = max(float(cross_section_distance), 1e-6)
    if mode in {"fixed_step", "step", "fixed-spacing", "fixed_fraction"} and str(spacing_mode or "distance").strip().lower() in {"fraction", "ratio", "proportion"}:
        spacing = max(total * float(spacing_ratio), 1e-6)

    # Legacy modes remain accepted for workspace compatibility.
    if mode == "distance":
        return [float(np.clip(x, 0.0, total)) for x in _spacing_targets(lo, hi, spacing, count)]

    if mode == "anchored_offset":
        offset = max(float(anchor_offset_mm), 0.0)
        at_start = (
            float(junction_endpoint) <= 0.5
            if junction_endpoint is not None
            else str(anchor or "end").strip().lower() == "start"
        )
        anchor_distance = 0.0 if at_start else total
        direction = 1.0 if at_start else -1.0
        targets = []
        limit = None if count == -1 else count
        index = 0
        while limit is None or index < limit:
            target = anchor_distance + direction * (offset + index * spacing)
            if target < lo - 1e-8 or target > hi + 1e-8:
                break
            targets.append(float(np.clip(target, lo, hi)))
            index += 1
        if targets:
            return targets
        return [float(np.clip(anchor_distance + direction * offset, lo, hi if hi >= lo else total))]

    if mode in {"fixed_step", "step", "fixed-spacing", "fixed_fraction"}:
        anchor_name = str(anchor or "start").strip().lower()
        if anchor_name == "center":
            anchor_distance = 0.5 * total
        elif anchor_name == "end":
            anchor_distance = total
        elif anchor_name == "junction":
            at_start = float(junction_endpoint) <= 0.5 if junction_endpoint is not None else False
            anchor_distance = 0.0 if at_start else total
        else:
            anchor_distance = 0.0
        direction_name = str(direction or "toward_end").strip().lower()
        if direction_name in {"both", "both_sides", "symmetric", "symmetrical"}:
            if count == -1:
                count = max(1, int(np.floor(total / spacing)) + 1)
            n = max(1, count)
            if n % 2:
                offsets = np.arange(-(n // 2), n // 2 + 1, dtype=float) * spacing
            else:
                offsets = (np.arange(n, dtype=float) - (n / 2.0 - 0.5)) * spacing
            return [float(x) for x in (anchor_distance + offsets)
                    if lo - 1e-8 <= x <= hi + 1e-8]
        sign = -1.0 if direction_name in {"toward_start", "start", "left"} else 1.0
        if anchor_name == "end" and direction_name in {"toward_end", "end"}:
            sign = -1.0
        n = max(1, count if count != -1 else int(np.floor(total / spacing)) + 1)
        targets = [anchor_distance + sign * float(i) * spacing for i in range(n)]
        return [float(x) for x in targets if lo - 1e-8 <= x <= hi + 1e-8]

    if count == -1:
        return [float(np.clip(x, 0.0, total)) for x in _spacing_targets(lo, hi, spacing, count)]
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
    plane_mode="fixed_step",
    plane_count=3,
    anchor="center",
    anchor_offset_mm=5.0,
    direction="both",
    spacing_mode="fraction",
    spacing_ratio=0.25,
    segmentation_labels=None,
    segmentation_filter=True,
    path_info=None,
    segmentation_spacing=(1, 1, 1),
    segmentation_origin=(0, 0, 0),
    return_qc=False,
    fork_points=None,
    forks=None,
):
    if use_center_plane is not None:
        plane_mode = "count" if bool(use_center_plane) else "distance"
        if bool(use_center_plane):
            plane_count = 1

    planes = []
    smooth_paths = []
    filter_qc = []

    paths_input = list(paths)
    if segmentation_filter:
        paths_input, filter_qc = filter_paths_by_segmentation(
            paths_input, segmentation_labels, spacing=segmentation_spacing,
            origin=segmentation_origin, path_info=path_info, forks=forks, inter_time=inter_time)
    else:
        filter_qc = [{"segmentation_filter": False} for _ in paths_input]

    for path_i, path in enumerate(paths_input):
        # Segmentation filtering already samples the path at ``inter_time``;
        # resampling it again would silently change the effective smoothing scale.
        path = (np.asarray(path, dtype=float).reshape(-1, 3)
                if segmentation_filter and segmentation_labels is not None
                else inter_points(path, time=inter_time))
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
            direction=direction,
            spacing_mode=spacing_mode,
            spacing_ratio=spacing_ratio,
            junction_endpoint=_junction_anchor_endpoint(path_smooth, fork_points, anchor),
        )
        if not targets:
            fallback = _fallback_mid_plane(path_smooth, cum, total, path_i)
            if fallback is not None:
                if path_i < len(filter_qc):
                    fallback.segmentation_label = int(filter_qc[path_i].get("owner_label", 0) or 0)
                planes.append(fallback)
            continue

        built = 0
        for target in targets:
            plane = _build_plane(path_smooth, cum, target, path_i)
            if path_i < len(filter_qc):
                plane.segmentation_label = int(filter_qc[path_i].get("owner_label", 0) or 0)
            planes.append(plane)
            built += 1

        if built == 0:
            fallback = _fallback_mid_plane(path_smooth, cum, total, path_i)
            if fallback is not None:
                if path_i < len(filter_qc):
                    fallback.segmentation_label = int(filter_qc[path_i].get("owner_label", 0) or 0)
                planes.append(fallback)

    if return_qc:
        return planes, smooth_paths, filter_qc
    return planes, smooth_paths
