"""Rotating QC views of computed scalar fields and recorded trajectories."""

import os
import numpy as np
import pyvista as pv

from ..task_control import check_cancelled, report_progress
from ..algorithms.flow_display import pathline_display, pathline_color_style
from . import videos
from .video_layout import VideoOverlay


FIELDS = {
    'velocity': (None, 'Velocity magnitude', 'cm/s'),
    'vorticity': ('vorticity_magnitude', 'Vorticity magnitude', '1/s'),
    'q_criterion': ('q_criterion_array', 'Q-criterion', '1/s²'),
    'swirling_strength': ('swirling_strength_array', 'Swirling strength', '1/s'),
    'pcmra': (None, 'PC-MRA', 'a.u.'),
}


def _field_at(ws, field, phase):
    if field == 'velocity':
        return np.linalg.norm(np.asarray(ws.flow_raw[..., phase, :], np.float32), axis=-1)
    source = ws.pcmra_array if field == 'pcmra' else getattr(ws.derived, FIELDS[field][0])
    return np.asarray(source[..., min(phase, source.shape[3] - 1)] if source.ndim == 4 else source, np.float32)


def render_qc_field_video(ws, out_dir, *, field, fps=12, smoothing_iteration=80,
                          view='iso', distance_scale=1.5, rotate=True, rotation_frames=180,
                          elevation_deg=10., time_repeat=3, window_size=None):
    title, unit = FIELDS[field][1:]
    if field == 'velocity' and ws.flow_raw is None or field == 'pcmra' and ws.pcmra_array is None:
        return None
    count = max(1, ws.time_count())
    # One fixed range for all phases. Display limits never change source arrays.
    low, high = np.inf, -np.inf
    for t in range(count):
        check_cancelled()
        values = _field_at(ws, field, t)
        valid = values[np.isfinite(values)]
        if valid.size:
            low, high = min(low, float(valid.min())), max(high, float(valid.max()))
    if not np.isfinite(low):
        return None
    if field == 'q_criterion':
        limit = max(abs(low), abs(high), 1e-9);clim = (-limit, limit)
    else:
        clim = (min(0., low), max(high, 1e-9))
    total = max(int(rotation_frames or 0), count * max(1, int(time_repeat))) if rotate else count * max(1, int(time_repeat))
    start = np.asarray(ws.origin, float)
    end = start + np.asarray(ws.flow_raw.shape[:3] if ws.flow_raw is not None else _field_at(ws, field, 0).shape) * np.asarray(ws.resolution)
    camera_surface = pv.Box(bounds=tuple(np.column_stack((start, end)).reshape(-1)))

    def frames():
        plotter = videos._make_plotter(window_size=videos._resolve_window_size(window_size), background=videos._video_background(ws), workspace=ws)
        overlay = VideoOverlay(plotter, ws, title)
        actor = None
        current = None
        try:
            for index in range(total):
                check_cancelled()
                phase, azimuth = videos._time_and_azimuth(index, rotation_frames=rotation_frames or total,
                    n_time=count, time_repeat=max(1, int(time_repeat))) if rotate else (videos._time_index(index, count, time_repeat), 0.)
                if phase != current:
                    volume = _field_at(ws, field, phase)
                    support = ws.derived.vortex_support_mask if field in {'vorticity', 'q_criterion', 'swirling_strength'} else ws.segmask_binary
                    if support is not None:
                        mask = np.asarray(support[..., min(phase, support.shape[3]-1)] if support.ndim == 4 else support, bool)
                    else:
                        mask = np.ones(volume.shape, bool)
                    data = np.where(mask & np.isfinite(volume), volume, 0.)
                    grid = pv.ImageData(dimensions=np.array(volume.shape) + 2,
                        spacing=ws.resolution, origin=np.asarray(ws.origin) - .5 * np.asarray(ws.resolution))
                    grid.point_data['value'] = np.pad(data, 1).ravel(order='F')
                    if actor is None:
                        actor = plotter.add_volume(grid, scalars='value', clim=clim,
                            cmap='coolwarm' if field == 'q_criterion' else 'turbo',
                            opacity=[.4, 0., .4] if field == 'q_criterion' else [0., .12, .5],
                            scalar_bar_args={'title': f'{title} ({unit})'},
                            mapper='fixed_point' if plotter.render_window.GetClassName() == 'vtkOSOpenGLRenderWindow' else 'smart')
                    else:
                        actor.mapper.dataset = grid
                        actor.mapper.Update()
                    current = phase
                plotter.camera_position = videos._orbit_camera(camera_surface, azimuth, elevation_deg or 0., distance_scale) if rotate else videos._camera_from_view(camera_surface, view, distance_scale)
                overlay.update(frame_index=index, total_frames=total, phase_index=phase, phase_count=count,
                               azimuth_deg=azimuth if rotate else None)
                plotter.render()
                yield np.asarray(plotter.screenshot(return_img=True))
                report_progress({'stage': 'video_frame', 'current': index+1, 'total': total, 'message': f'Rendered {title} frame {index+1}/{total}'})
        finally:
            plotter.close()
    return videos._write_video(frames, os.path.join(str(out_dir), f'{field}_rotate.mp4'), fps=fps)


def render_qc_pathline_video(ws, out_dir, *, fps=12, smoothing_iteration=80,
                             view='iso', distance_scale=1.5, rotate=True, rotation_frames=180,
                             elevation_deg=10., time_repeat=3, window_size=None):
    tracks = [(i, launch, mesh) for i, cache in ws.pathline_cache.items() for launch, mesh in cache.items()
              if mesh is not None and mesh.n_points]
    if not tracks:
        return None
    count = max(1, ws.time_count())
    total = max(int(rotation_frames or 0), count*max(1, int(time_repeat)))
    points = np.vstack([mesh.points for _, _, mesh in tracks])
    camera_surface = pv.PolyData(points)
    def frames():
        plotter = videos._make_plotter(window_size=videos._resolve_window_size(window_size), background=videos._video_background(ws), workspace=ws)
        overlay = VideoOverlay(plotter, ws, 'Recorded pathlines')
        actors = []
        try:
            for index in range(total):
                check_cancelled()
                phase, azimuth = videos._time_and_azimuth(index, rotation_frames=rotation_frames or total, n_time=count, time_repeat=time_repeat)
                for actor in actors:
                    plotter.remove_actor(actor, render=False)
                actors.clear()
                for plane, launch, full in tracks:
                    mesh = pathline_display(full, ((phase-launch)%count)/count,
                        style=ws.streamline_params.pathline_display_style,
                        trail_duration_ms=ws.streamline_params.pathline_trail_duration_ms,
                        rr_ms=ws.rr, minimum_cm_s=ws.streamline_params.pathline_display_min_speed_cm_s) if 'PathlinePhase' in full.point_data else full
                    if mesh is None or not mesh.n_points:
                        continue
                    style = pathline_color_style(ws, plane)
                    style.pop('scalar_bar_title', None);style.pop('scalar_bar_cfg', None)
                    actors.append(plotter.add_mesh(mesh, line_width=2, render=False, **style))
                plotter.camera_position = videos._orbit_camera(camera_surface, azimuth, elevation_deg or 0., distance_scale)
                overlay.update(frame_index=index, total_frames=total, phase_index=phase, phase_count=count, azimuth_deg=azimuth)
                plotter.render()
                yield np.asarray(plotter.screenshot(return_img=True))
        finally:
            plotter.close()
    return videos._write_video(frames, os.path.join(str(out_dir), 'pathlines_rotate.mp4'), fps=fps)
