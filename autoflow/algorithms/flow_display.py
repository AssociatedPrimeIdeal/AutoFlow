"""Presentation of integrated trajectories without altering scientific caches."""

import numpy as np
import pyvista as pv


def pathline_color_style(workspace, plane_index):
    """One consistent style for engine-created and interactively-created paths."""
    params = workspace.streamline_params
    mode = params.pathline_color_mode
    result = {"color": workspace.pathline_color_for_plane(plane_index),
              "scalars": None, "show_scalar_bar": False}
    if int(plane_index) in workspace.pathline_colors or mode not in {"velocity", "age"}:
        return result
    if mode == "age":
        clim, scalar, title = (0., max(float(workspace.rr), 1e-6)), "ParticleAge_ms", "Particle age (ms)"
    else:
        from .streamlines import automatic_streamline_clim
        mask = workspace.segmask_binary if params.pathline_seed_region == "segmentation" else None
        key = (id(workspace.flow_raw), id(mask), params.pathline_seed_region)
        cache = getattr(workspace, "_pathline_velocity_limits", {})
        if key not in cache:
            cache = {key: automatic_streamline_clim(workspace.flow_raw, mask)}
            workspace._pathline_velocity_limits = cache
        clim = workspace.render_settings.get("streamline_clim") or cache[key]
        scalar, title = "Velocity", "Velocity (m/s)"
    result.update(scalars=scalar, cmap="turbo", clim=clim, show_scalar_bar=True,
                  scalar_bar_title=title, scalar_bar_cfg=dict(workspace.render_settings.get("streamline_bar_cfg", {})))
    return result


def _point_subset(mesh, indices):
    if not len(indices):
        return pv.PolyData()
    result = pv.PolyData(np.asarray(mesh.points)[indices])
    for name, values in mesh.point_data.items():
        result.point_data[name] = np.asarray(values)[indices]
    for name, values in mesh.field_data.items():
        result.field_data[name] = np.asarray(values)
    result._flow_display_source = mesh
    return result


def clip_flow_lines(mesh, minimum_cm_s=0., *, phase_min=None, phase_max=None):
    """Interpolate cut endpoints and preserve disjoint visible line pieces."""
    minimum = float(minimum_cm_s)
    if not np.isfinite(minimum) or minimum < 0:
        raise ValueError("Minimum display speed must be finite and nonnegative")
    if mesh is None or mesh.n_points == 0:
        return mesh
    tests = []
    if minimum > 0 and "Velocity" in mesh.point_data:
        tests.append((np.asarray(mesh["Velocity"], float), 1., minimum / 100.))
    if "PathlinePhase" in mesh.point_data:
        values = np.asarray(mesh["PathlinePhase"], float)
        if phase_min is not None:
            tests.append((values, 1., float(phase_min)))
        if phase_max is not None:
            tests.append((values, -1., -float(phase_max)))
    if not tests:
        return mesh
    point_keep = np.ones(mesh.n_points, bool)
    for values, sign, boundary in tests:
        point_keep &= np.isfinite(values) & (values * sign >= boundary - 1e-12)
    if np.all(point_keep):
        return mesh
    points = np.asarray(mesh.points, float)
    packed = np.asarray(mesh.lines, np.int64).reshape(-1)
    if not len(packed):
        return _point_subset(mesh, np.flatnonzero(point_keep))
    starts, ends, breaks = [], [], []
    cursor = 0
    while cursor < len(packed):
        count = int(packed[cursor]); indices = packed[cursor+1:cursor+count+1]
        cursor += count + 1
        if count >= 2:
            starts.extend(indices[:-1]); ends.extend(indices[1:])
            breaks.extend([True] + [False] * (count - 2))
    if not starts:
        return _point_subset(mesh, np.flatnonzero(point_keep))
    starts, ends = np.asarray(starts), np.asarray(ends)
    lo, hi = np.zeros(len(starts)), np.ones(len(starts))
    keep = np.all(np.isfinite(points[starts]), axis=1) & np.all(np.isfinite(points[ends]), axis=1)
    for values, sign, boundary in tests:
        first, second = values[starts]*sign, values[ends]*sign
        finite = np.isfinite(first) & np.isfinite(second)
        keep &= finite & ((first >= boundary) | (second >= boundary))
        delta = second - first
        fraction = np.divide(boundary-first, delta, out=np.zeros_like(delta), where=finite & (delta != 0))
        lo = np.maximum(lo, np.where(first < boundary, fraction, 0.))
        hi = np.minimum(hi, np.where(second < boundary, fraction, 1.))
    keep &= hi - lo > 1e-12
    selected = np.flatnonzero(keep)
    if not len(selected):
        return _point_subset(mesh, np.flatnonzero(point_keep))
    split = np.asarray(breaks, bool)[selected[1:]] | (np.diff(selected) != 1)
    split |= (hi[selected[:-1]] < 1.) | (lo[selected[1:]] > 0.)
    groups = np.split(selected, np.flatnonzero(split) + 1)
    left, right, weights, lines = [], [], [], []
    offset = 0
    for group in groups:
        segments = np.r_[group, group[-1]]
        left.append(starts[segments]); right.append(ends[segments])
        weights.append(np.r_[lo[group], hi[group[-1]]])
        count = len(group) + 1
        lines.append(np.r_[count, np.arange(offset, offset + count)])
        offset += count
    left, right, weights = np.concatenate(left), np.concatenate(right), np.concatenate(weights)
    result = pv.PolyData(points[left] + weights[:, None]*(points[right]-points[left]), lines=np.concatenate(lines))
    for name, values in mesh.point_data.items():
        values = np.asarray(values)
        if np.issubdtype(values.dtype, np.floating):
            fraction = weights.reshape((-1,) + (1,)*(values.ndim-1))
            result.point_data[name] = values[left] + fraction*(values[right]-values[left])
        else:
            result.point_data[name] = values[np.where(weights < .5, left, right)]
    for name, values in mesh.field_data.items():
        result.field_data[name] = np.asarray(values)
    if mesh.active_scalars_name in result.point_data:
        result.set_active_scalars(mesh.active_scalars_name)
    if mesh.active_vectors_name in result.point_data:
        result.set_active_vectors(mesh.active_vectors_name)
    result._flow_display_source = mesh
    return result


def pathline_display(mesh, phase, *, style="history", trail_duration_ms=100., rr_ms=1000., minimum_cm_s=0.):
    """Reveal full history, particle heads, or heads with finite-age tails."""
    if mesh is None or "PathlinePhase" not in mesh.point_data:
        return clip_flow_lines(mesh, minimum_cm_s)
    target = float(np.clip(phase, 0., 1.))
    lower = None
    if style == "trails":
        lower = max(0., target - max(float(trail_duration_ms), 0.) / max(float(rr_ms), 1e-6))
    visible = clip_flow_lines(mesh, minimum_cm_s, phase_min=lower, phase_max=target)
    if visible is None or not visible.n_points:
        return visible
    visible = visible.copy(deep=False)
    phases = np.asarray(visible["PathlinePhase"], float)
    heads = np.flatnonzero(np.isclose(phases, target, atol=1e-9, rtol=0))
    if style == "particles":
        visible = _point_subset(visible, heads)
    elif style == "trails":
        visible.verts = np.column_stack((np.ones(len(heads), np.int64), heads)).reshape(-1)
    elif style != "history":
        raise ValueError(f"Unknown pathline display style: {style}")
    visible.point_data["ParticleAge_ms"] = np.asarray(visible["PathlinePhase"], float) * float(rr_ms)
    visible._flow_display_source = mesh
    return visible
