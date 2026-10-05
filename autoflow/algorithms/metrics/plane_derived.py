"""Per-plane sampling and summaries of available derived fields."""

import os
import numpy as np
from ...task_control import check_cancelled, task_scope
from ..surfaces import _build_branch_grid

from ._common import _ensure_mask4d
from ._parallel import (
    _PlaneProcessToken,
    _mark_plane_process_progress,
    _wait_plane_processes,
)
from .sampling import (
    _build_mask_phase_lookup,
    _build_plane_support_mesh_cache,
    _get_cached_plane_slice_spec,
    _plane_roi_point_mask,
    _sample_field_from_slice_spec,
    _target_label_for_plane,
)
from .tke import _prepare_tke_array


def _nanpercentile_safe(values, q):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0
    return float(np.nanpercentile(finite, q))


def _weighted_mean(values, weights):
    vals = np.asarray(values, dtype=float).reshape(-1)
    wts = np.asarray(weights, dtype=float).reshape(-1)
    if vals.size == 0 or wts.size != vals.size:
        return 0.0
    finite = np.isfinite(vals) & np.isfinite(wts) & (wts > 0)
    if not np.any(finite):
        return 0.0
    vals = vals[finite]
    wts = wts[finite]
    denom = float(np.sum(wts))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(vals * wts) / denom)


def _append_summary(metric, prefix, series):
    arr = np.asarray(series, dtype=float).reshape(-1)
    metric[f"{prefix}_t"] = [float(x) for x in arr.tolist()]
    metric[prefix] = float(np.mean(arr)) if arr.size else 0.0


def _available_pressure_statistics(values, areas, absolute=False):
    values = np.asarray(values, dtype=float).reshape(-1)
    areas = np.asarray(areas, dtype=float).reshape(-1)
    if values.size != areas.size:
        return None, None, None
    valid = np.isfinite(values) & np.isfinite(areas) & (areas > 0)
    if not np.any(valid):
        return None, None, None
    values, areas = values[valid], areas[valid]
    magnitude = np.abs(values) if absolute else values
    return _weighted_mean(values, areas), float(np.max(magnitude)), _nanpercentile_safe(magnitude, 95)


def _append_available_summary(metric, prefix, series):
    values = np.asarray(series, dtype=float)
    valid = np.isfinite(values)
    metric[f"{prefix}_t"] = [float(x) if ok else None for x, ok in zip(values, valid)]
    metric[prefix] = float(np.mean(values[valid])) if np.any(valid) else None


def summarize_plane_derived_metrics(plane, mask4d, spacing, origin, branch_labels_3d=None,
                                    tke_array=None, pressure_gradient_array=None,
                                    relative_pressure_array=None, wss_surfaces=None,
                                    branch_grid=None, mask_phase_lookup=None,
                                    support_mesh_cache=None, pressure_gradient_support_mask=None):
    mask4d = _ensure_mask4d(mask4d)
    spacing = np.asarray(spacing, dtype=float).reshape(3)
    origin = np.asarray(origin, dtype=float).reshape(3)
    Nt = int(mask4d.shape[3])
    branch_grid = branch_grid if branch_grid is not None else _build_branch_grid(branch_labels_3d, spacing, origin)
    target_label = _target_label_for_plane(plane, branch_labels_3d, spacing, origin)
    normal = np.asarray(plane.normal, dtype=float).reshape(3)
    normal = normal / (np.linalg.norm(normal) + 1e-12)

    summary = {}
    pixelwise = {"timepoints": []}

    has_tke = tke_array is not None
    has_pressure_gradient = pressure_gradient_array is not None
    has_relative_pressure = relative_pressure_array is not None
    has_wss = bool(wss_surfaces)
    needs_volume_slice = has_tke or has_pressure_gradient or has_relative_pressure

    if has_tke:
        tke_array = _prepare_tke_array(mask4d, tke_array=tke_array, sigma=None)
    if has_pressure_gradient:
        pressure_gradient_array = np.asarray(pressure_gradient_array, dtype=np.float32)
        if pressure_gradient_array.ndim != 5 or pressure_gradient_array.shape[-1] != 3:
            raise ValueError(f"pressure_gradient_array must be XYZTV, got {pressure_gradient_array.shape}")
    if has_relative_pressure:
        relative_pressure_array = np.asarray(relative_pressure_array, dtype=np.float32)
        if relative_pressure_array.ndim != 4:
            raise ValueError(f"relative_pressure_array must be XYZT, got {relative_pressure_array.shape}")
    pressure_support = None
    if pressure_gradient_support_mask is not None:
        pressure_support = _ensure_mask4d(pressure_gradient_support_mask)
        if pressure_support.shape[:3] != mask4d.shape[:3] or pressure_support.shape[3] not in (1, Nt):
            raise ValueError("pressure_gradient_support_mask must match the spatial mask and have one or all phases")

    if mask_phase_lookup is None:
        mask_phase_lookup = _build_mask_phase_lookup(mask4d) if needs_volume_slice else []
    if support_mesh_cache is None and needs_volume_slice:
        support_mesh_cache = _build_plane_support_mesh_cache(mask4d, mask_phase_lookup, spacing, origin)
    slice_cache = {}

    tke_mean_t = []
    tke_peak_t = []
    tke_p95_t = []
    pg_mag_mean_t = []
    pg_mag_peak_t = []
    pg_mag_p95_t = []
    pg_normal_mean_t = []
    pg_normal_peak_t = []
    pg_normal_p95_t = []
    rp_mean_t = []
    rp_peak_t = []
    rp_p95_t = []
    pg_valid_count_t = []
    rp_valid_count_t = []
    wss_mean_t = []
    wss_peak_t = []
    wss_p95_t = []

    for tidx in range(Nt):
        entry = {"time_index": int(tidx)}

        tke_series_vals = np.array([], dtype=float)
        pg_mag_vals = np.array([], dtype=float)
        pg_normal_vals = np.array([], dtype=float)
        rp_vals = np.array([], dtype=float)
        areas = np.array([], dtype=float)
        slice_spec = None
        if needs_volume_slice:
            rep_t = int(mask_phase_lookup[tidx])
            slice_spec = _get_cached_plane_slice_spec(
                slice_cache,
                (rep_t, tidx) if getattr(plane, "roi_edit_operations", {}) else rep_t,
                mask4d[..., rep_t],
                plane,
                spacing,
                origin,
                branch_grid=branch_grid,
                target_label=target_label,
                select_connected=False,
                support_mesh=support_mesh_cache.get(rep_t) if support_mesh_cache is not None else None,
                frame_index=tidx,
            )

        sampled_pressure_support = None
        if pressure_support is not None and slice_spec is not None:
            sampled_pressure_support = np.asarray(_sample_field_from_slice_spec(
                pressure_support[..., min(tidx, pressure_support.shape[3] - 1)],
                mask4d.shape[:3], "pressure_support", slice_spec), dtype=bool).reshape(-1)

        if has_tke and slice_spec is not None:
            tke_series_vals = np.asarray(
                _sample_field_from_slice_spec(tke_array[..., tidx], mask4d.shape[:3], "tke", slice_spec),
                dtype=float,
            ).reshape(-1)
            if tke_series_vals.size:
                areas = np.asarray(slice_spec["areas"], dtype=float).reshape(-1)
                entry["cell_area_mm2"] = areas.astype(np.float32)
                entry["tke_J_m3"] = tke_series_vals.astype(np.float32)
                entry["lumen_mask"] = np.ones_like(tke_series_vals, dtype=np.uint8)

        if has_pressure_gradient and slice_spec is not None:
            vec = np.asarray(
                _sample_field_from_slice_spec(
                    pressure_gradient_array[..., tidx, :],
                    mask4d.shape[:3],
                    "pressure_gradient",
                    slice_spec,
                ),
                dtype=float,
            )
            if vec.ndim == 2 and vec.shape[1] == 3:
                if areas.size == 0:
                    areas = np.asarray(slice_spec["areas"], dtype=float).reshape(-1)
                    entry.setdefault("cell_area_mm2", areas.astype(np.float32))
                    entry.setdefault("lumen_mask", np.ones(len(areas), dtype=np.uint8))
                valid = np.all(np.isfinite(vec), axis=1)
                if sampled_pressure_support is not None:
                    valid &= sampled_pressure_support
                vec = np.where(valid[:, None], vec, np.nan)
                entry["pressure_gradient_valid"] = valid.astype(np.uint8)
                pg_mag_vals = np.linalg.norm(vec, axis=1)
                pg_normal_vals = np.dot(vec, normal)
                entry["pressure_gradient_mag_Pa_m"] = pg_mag_vals.astype(np.float32)
                entry["pressure_gradient_normal_Pa_m"] = pg_normal_vals.astype(np.float32)
                entry["pressure_gradient_vec_Pa_m"] = vec.astype(np.float32)

        if has_relative_pressure and slice_spec is not None:
            vals = np.asarray(
                _sample_field_from_slice_spec(
                    relative_pressure_array[..., tidx],
                    mask4d.shape[:3],
                    "relative_pressure",
                    slice_spec,
                ),
                dtype=float,
            ).reshape(-1)
            if vals.size:
                if areas.size == 0:
                    areas = np.asarray(slice_spec["areas"], dtype=float).reshape(-1)
                    entry.setdefault("cell_area_mm2", areas.astype(np.float32))
                    entry.setdefault("lumen_mask", np.ones(len(areas), dtype=np.uint8))
                valid = np.isfinite(vals)
                if sampled_pressure_support is not None:
                    valid &= sampled_pressure_support
                rp_vals = np.where(valid, vals, np.nan)
                entry["relative_pressure_valid"] = valid.astype(np.uint8)
                entry["relative_pressure_Pa"] = rp_vals.astype(np.float32)

        if tke_series_vals.size:
            tke_mean_t.append(_weighted_mean(tke_series_vals, areas))
            tke_peak_t.append(float(np.max(tke_series_vals)))
            tke_p95_t.append(_nanpercentile_safe(tke_series_vals, 95.0))
        else:
            tke_mean_t.append(0.0)
            tke_peak_t.append(0.0)
            tke_p95_t.append(0.0)

        pg_stats = _available_pressure_statistics(pg_mag_vals, areas)
        for series, value in zip((pg_mag_mean_t, pg_mag_peak_t, pg_mag_p95_t), pg_stats):
            series.append(value)
        normal_stats = _available_pressure_statistics(pg_normal_vals, areas, absolute=True)
        for series, value in zip((pg_normal_mean_t, pg_normal_peak_t, pg_normal_p95_t), normal_stats):
            series.append(value)
        rp_stats = _available_pressure_statistics(rp_vals, areas, absolute=True)
        for series, value in zip((rp_mean_t, rp_peak_t, rp_p95_t), rp_stats):
            series.append(value)
        pg_valid_count_t.append(int(np.count_nonzero(np.isfinite(pg_mag_vals))))
        rp_valid_count_t.append(int(np.count_nonzero(np.isfinite(rp_vals))))

        wss_vals = np.array([], dtype=float)
        surf = None if not has_wss or tidx >= len(wss_surfaces) else wss_surfaces[tidx]
        if surf is not None and getattr(surf, "n_points", 0) > 0:
            try:
                wall = surf.slice(
                    normal=normal,
                    origin=np.asarray(plane.center, dtype=float).reshape(3) + origin,
                )
            except Exception:
                wall = None
            if wall is not None and getattr(wall, "n_points", 0) > 0:
                if "wss" not in wall.point_data and "wss" in wall.cell_data:
                    vals = np.asarray(wall.cell_data.get("wss", []), dtype=float).reshape(-1)
                    sample_points = np.asarray(wall.cell_centers().points, dtype=float)
                else:
                    vals = np.asarray(wall.point_data.get("wss", []), dtype=float).reshape(-1)
                    sample_points = np.asarray(wall.points, dtype=float)
                if len(vals) == len(sample_points):
                    vals = vals[_plane_roi_point_mask(sample_points, plane, origin)]
                wss_vals = vals[np.isfinite(vals)]
        if wss_vals.size:
            wss_mean_t.append(float(np.mean(wss_vals)))
            wss_peak_t.append(float(np.max(wss_vals)))
            wss_p95_t.append(_nanpercentile_safe(wss_vals, 95.0))
        else:
            wss_mean_t.append(0.0)
            wss_peak_t.append(0.0)
            wss_p95_t.append(0.0)

        pixelwise["timepoints"].append(entry)

    if has_tke:
        _append_summary(summary, "tke_mean_J_m3", tke_mean_t)
        _append_summary(summary, "tke_peak_J_m3", tke_peak_t)
        _append_summary(summary, "tke_p95_J_m3", tke_p95_t)
    if has_pressure_gradient:
        summary["pressure_gradient_valid_cell_count_t"] = pg_valid_count_t
        _append_available_summary(summary, "pressure_gradient_mag_mean_Pa_m", pg_mag_mean_t)
        _append_available_summary(summary, "pressure_gradient_mag_peak_Pa_m", pg_mag_peak_t)
        _append_available_summary(summary, "pressure_gradient_mag_p95_Pa_m", pg_mag_p95_t)
        _append_available_summary(summary, "pressure_gradient_normal_mean_Pa_m", pg_normal_mean_t)
        _append_available_summary(summary, "pressure_gradient_normal_peak_Pa_m", pg_normal_peak_t)
        _append_available_summary(summary, "pressure_gradient_normal_p95_Pa_m", pg_normal_p95_t)
    if has_relative_pressure:
        summary["relative_pressure_valid_cell_count_t"] = rp_valid_count_t
        _append_available_summary(summary, "relative_pressure_mean_Pa", rp_mean_t)
        _append_available_summary(summary, "relative_pressure_peak_Pa", rp_peak_t)
        _append_available_summary(summary, "relative_pressure_p95_Pa", rp_p95_t)
    if has_wss:
        _append_summary(summary, "wss_wall_mean_Pa", wss_mean_t)
        _append_summary(summary, "wss_wall_peak_Pa", wss_peak_t)
        _append_summary(summary, "wss_wall_p95_Pa", wss_p95_t)
    return summary, pixelwise


def _augment_plane_metrics_serial(plane_metrics, planes, mask4d, spacing, origin, branch_labels_3d=None,
                                 tke_array=None, pressure_gradient_array=None,
                                 relative_pressure_array=None, wss_surfaces=None, progress_callback=None,
                                 pressure_gradient_support_mask=None):
    mask4d = _ensure_mask4d(mask4d)
    shared_branch_grid = _build_branch_grid(branch_labels_3d, spacing, origin)
    mask_phase_lookup = _build_mask_phase_lookup(mask4d)
    support_mesh_cache = _build_plane_support_mesh_cache(mask4d, mask_phase_lookup, spacing, origin)
    metrics = [dict(m) for m in plane_metrics]
    pixelwise = []
    for idx, metric in enumerate(metrics):
        check_cancelled()
        if idx >= len(planes):
            pixelwise.append({"plane_index": int(idx), "timepoints": []})
            continue
        summary, payload = summarize_plane_derived_metrics(
            planes[idx], mask4d, spacing, origin, branch_labels_3d=branch_labels_3d,
            tke_array=tke_array, pressure_gradient_array=pressure_gradient_array,
            relative_pressure_array=relative_pressure_array,
            wss_surfaces=wss_surfaces,
            branch_grid=shared_branch_grid,
            mask_phase_lookup=mask_phase_lookup,
            support_mesh_cache=support_mesh_cache,
            pressure_gradient_support_mask=pressure_gradient_support_mask,
        )
        metric.update(summary)
        payload["plane_index"] = int(idx)
        payload["center"] = np.asarray(planes[idx].center, dtype=float).reshape(3).tolist()
        payload["normal"] = np.asarray(planes[idx].normal, dtype=float).reshape(3).tolist()
        payload["label"] = int(getattr(planes[idx], "label", 0) or 0)
        payload["path_index"] = int(getattr(planes[idx], "path_index", -1))
        pixelwise.append(payload)
        if progress_callback is not None:
            progress_callback({"stage": "plane_derived", "current": idx + 1, "total": len(metrics),
                               "message": f"Sampled derived plane metrics ({idx + 1}/{len(metrics)})"})
    return metrics, pixelwise


def _augment_plane_metrics_chunk(indexed_metrics, planes, mask4d, spacing, origin, branch_labels_3d,
                                 tke_array, pressure_gradient_array, relative_pressure_array, wss_surfaces,
                                 progress_path, pressure_gradient_support_mask):
    indices = [index for index, _metric in indexed_metrics]
    local_planes = [planes[index] for index in indices]
    with task_scope(_PlaneProcessToken(progress_path)):
        metrics, payloads = _augment_plane_metrics_serial(
            [metric for _index, metric in indexed_metrics], local_planes, mask4d, spacing, origin,
            branch_labels_3d, tke_array, pressure_gradient_array, relative_pressure_array, wss_surfaces,
            progress_callback=lambda _payload: _mark_plane_process_progress(progress_path),
            pressure_gradient_support_mask=pressure_gradient_support_mask,
        )
    for index, payload in zip(indices, payloads):
        payload["plane_index"] = int(index)
    return list(zip(indices, metrics, payloads))


def augment_plane_metrics_with_derived(plane_metrics, planes, mask4d, spacing, origin, branch_labels_3d=None,
                                       tke_array=None, pressure_gradient_array=None,
                                       relative_pressure_array=None, wss_surfaces=None, *,
                                       use_multithread=False, max_workers=None, progress_callback=None,
                                       pressure_gradient_support_mask=None):
    mask4d = _ensure_mask4d(mask4d)
    count = min(len(plane_metrics), len(planes))
    if max_workers is None:
        max_workers = min(4, count, max(1, os.cpu_count() or 1)) if use_multithread and count * mask4d.shape[3] >= 1920 else 1
    if int(max_workers) <= 1 or count != len(plane_metrics):
        return _augment_plane_metrics_serial(
            plane_metrics, planes, mask4d, spacing, origin, branch_labels_3d,
            tke_array, pressure_gradient_array, relative_pressure_array, wss_surfaces, progress_callback,
            pressure_gradient_support_mask,
        )
    import tempfile
    from joblib import Parallel, delayed
    workers = min(int(max_workers), count)
    indexed = list(enumerate(plane_metrics))
    with tempfile.TemporaryDirectory(prefix="autoflow_plane_derived_") as progress_dir:
        progress_path = os.path.join(progress_dir, "progress.log")
        open(progress_path, "ab").close()
        def calculate():
            return Parallel(n_jobs=workers, backend="loky", max_nbytes="10M", mmap_mode="r")(
                delayed(_augment_plane_metrics_chunk)(
                    indexed[offset::workers], planes, mask4d, spacing, origin, branch_labels_3d,
                    tke_array, pressure_gradient_array, relative_pressure_array, wss_surfaces, progress_path,
                    pressure_gradient_support_mask,
                ) for offset in range(workers)
            )
        chunks = _wait_plane_processes(calculate, progress_path, count, progress_callback, stage="plane_derived")
    metrics = [None] * count
    payloads = [None] * count
    for chunk in chunks:
        for index, metric, payload in chunk:
            metrics[index] = metric
            payloads[index] = payload
    return metrics, payloads
