"""Capture the actual main 3D scene across cardiac phases on the GUI thread."""

import copy
import os
from pathlib import Path
import tempfile
import time
import numpy as np
import imageio.v2 as imageio
from PySide6 import QtCore, QtWidgets

from ..task_control import TaskCancelled, check_cancelled
from ..core.output import publish_video
from .video_options import MovieCapture, validate_resolution


def export_current_view(window, filename, *, fps, cycles=1, resolution=None, rotate=False,
                        rotation_axis='screen_vertical', rotation_angle=360., rotation_speed=30.,
                        hide_orientation_marker=False, cancelled=lambda: False, progress=None):
    if not np.isfinite(fps) or fps <= 0 or int(cycles) < 1:
        raise ValueError('FPS and cycle count must be positive')
    resolution = validate_resolution(resolution)
    ws = window.workspace
    count = max(1, ws.time_count())
    start = int(ws.current_t)
    camera = copy.deepcopy(window.plotter.camera_position)
    source_camera = getattr(window.plotter, 'camera', None)
    camera_state = source_camera.copy() if source_camera is not None else None
    def restore_camera():
        if camera_state is not None:
            window.plotter.camera.DeepCopy(camera_state)
        else:
            window.plotter.camera_position = camera
    frame_count = count * int(cycles)
    axis = None
    if rotate:
        if not np.isfinite(rotation_angle) or rotation_angle == 0 or not np.isfinite(rotation_speed) or rotation_speed <= 0:
            raise ValueError('Rotation angle must be finite/nonzero and speed must be positive')
        position, focal, up = (np.asarray(value, dtype=float) for value in camera)
        direction = focal-position
        direction /= max(np.linalg.norm(direction), 1e-12)
        up = up-direction*np.dot(up, direction)
        up /= max(np.linalg.norm(up), 1e-12)
        axes = dict(screen_vertical=up, screen_horizontal=np.cross(direction, up))
        signs = np.asarray(getattr(window.scene, '_display_axis_signs', (1., 1., 1.)))
        axes.update({name: np.eye(3)[i]*signs for i, name in enumerate(('lr', 'ap', 'fh'))})
        if rotation_axis not in axes:
            raise ValueError('Unknown rotation axis: ' + str(rotation_axis))
        axis = np.asarray(axes[rotation_axis], dtype=float)
        if not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-10:
            raise ValueError('The starting camera has no valid rotation axis')
        axis /= np.linalg.norm(axis)
        frame_count = max(2, int(np.ceil(abs(rotation_angle)/rotation_speed*fps))+1)
    marker = getattr(getattr(window.scene, '_orientation_marker', None), 'widget', None)
    marker_enabled = marker.GetEnabled() if marker is not None else None
    playing = window._play_timer.isActive()
    preview = bool(getattr(window.scene, '_interaction_preview_active', False))
    target = Path(filename)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.current-view-', suffix='.mp4', dir=target.parent)
    os.close(descriptor)
    window._on_pause()
    window.scene.set_interaction_preview(False, render=False)
    window._scene_export_running = True
    phases = []
    capture = None
    writer = None
    encoded_size = None
    try:
        check_cancelled()
        if cancelled():
            raise TaskCancelled("Current-view export cancelled")
        if hide_orientation_marker and marker is not None:
            marker.SetEnabled(False)
        capture = MovieCapture(window.plotter, resolution)
        with imageio.get_writer(temporary, format='ffmpeg', fps=float(fps), codec='libx264',
                               pixelformat='yuv420p', macro_block_size=2, quality=None,
                               output_params=['-crf', '18', '-preset', 'medium', '-movflags', '+faststart']) as writer:
            for index in range(frame_count):
                check_cancelled()
                if cancelled():
                    raise TaskCancelled('Current-view export cancelled')
                phase = (start + index) % count
                ws.current_t = phase
                window.scene.update_time(phase)
                if ws.streamline_active and any(obj.visible and obj.data_key == 'streamlines_live' for obj in ws.scene_objects.values()):
                    from .streamline_jobs import streamline_requests
                    manager = streamline_requests(ws)
                    manager.request(phase)
                    deadline = time.monotonic() + 180
                    while phase not in ws.streamline_cache:
                        QtWidgets.QApplication.processEvents()
                        check_cancelled()
                        if cancelled():
                            raise TaskCancelled('Current-view export cancelled')
                        if any(key[1] == phase for key in manager.errors) or time.monotonic() > deadline:
                            raise RuntimeError(f'Streamline phase {phase} could not be prepared; see Log')
                        time.sleep(.01)
                    window.scene.update_time(phase)
                restore_camera()
                if rotate:
                    angle = np.radians(rotation_angle*index/(frame_count-1))
                    def turn(vector):
                        return vector*np.cos(angle)+np.cross(axis, vector)*np.sin(angle)+axis*np.dot(axis, vector)*(1-np.cos(angle))
                    rotated_position = focal+turn(position-focal)
                    rotated_up = turn(up)
                    active_camera = getattr(window.plotter, 'camera', None)
                    if active_camera is not None:
                        active_camera.SetPosition(*rotated_position)
                        active_camera.SetViewUp(*rotated_up)
                        active_camera.OrthogonalizeViewUp()
                        bounds = getattr(window.scene, '_visible_scene_bounds', lambda: None)()
                        if bounds is not None:
                            window.plotter.renderer.ResetCameraClippingRange(bounds)
                        else:
                            window.plotter.renderer.ResetCameraClippingRange()
                    else:
                        window.plotter.camera_position = [tuple(rotated_position), tuple(focal), tuple(rotated_up)]
                window.scene.flush_render()
                pixels = np.asarray(capture.image())
                if pixels.ndim != 3 or pixels.shape[2] < 3:
                    raise RuntimeError('The current 3D view did not return an RGB frame')
                # Pad only the outermost pixels for codecs requiring even sizes.
                pixels = np.ascontiguousarray(pixels[..., :3])
                pixels = np.pad(pixels, ((0, pixels.shape[0] % 2), (0, pixels.shape[1] % 2), (0, 0)), mode='edge')
                encoded_size = [int(pixels.shape[1]), int(pixels.shape[0])]
                writer.append_data(pixels)
                phases.append(phase)
                if progress:
                    progress(index + 1, frame_count)
                QtWidgets.QApplication.processEvents()
        check_cancelled()
        if cancelled():
            raise TaskCancelled('Current-view export cancelled')
        publish_video(temporary, target)
        return {'path': str(target), 'fps': float(fps), 'cycles': None if rotate else int(cycles), 'frames': len(phases),
                'phases': phases, 'camera': str(camera), 'phase_start': start,
                'resolution': encoded_size, 'source_resolution': capture.source_size,
                'rotate': bool(rotate), 'rotation_axis': rotation_axis if rotate else None,
                'rotation_axis_world': axis.tolist() if axis is not None else None,
                'rotation_angle': float(rotation_angle) if rotate else 0.,
                'rotation_speed': float(rotation_speed) if rotate else 0.,
                'duration_seconds': len(phases)/float(fps),
                'hide_orientation_marker': bool(hide_orientation_marker),
                'visible_layers': [obj.data_key for obj in ws.scene_objects.values() if obj.visible]}
    finally:
        try:
            # imageio's writer context intentionally skips close on an
            # exception. Reap the codec before deleting a cancelled movie.
            if writer is not None:
                writer.close()
        finally:
            if capture is not None:
                capture.close()
            window._scene_export_running = False
            ws.current_t = start
            window.scene.update_time(start)
            restore_camera()
            if marker is not None and hide_orientation_marker:
                marker.SetEnabled(marker_enabled)
            window.scene.set_interaction_preview(preview, render=False)
            window.scene.flush_render()
            window._refresh_timeline()
            window.ortho_viewer.refresh()
            if playing:
                window._on_play()
            if os.path.exists(temporary):
                os.unlink(temporary)
