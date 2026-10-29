"""Measure native/SSH 3D rendering, frame capture and Qt presentation separately.

Manual benchmark; does not modify its H5 input or run the analysis pipeline.
Run with QT_QPA_PLATFORM=offscreen for a control, or xcb over forwarded X11.
Presentation timings include QApplication.sync(), not physical monitor latency.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=20)
    parser.add_argument('--sizes', nargs='+', default=['640x420', '960x630', '1400x900'])
    parser.add_argument('--scenes', nargs='+', choices=['surface', 'volume', 'slices'],
                        default=['surface', 'volume', 'slices'])
    parser.add_argument('--paint-format', choices=['default', 'rgb32'], default='default')
    parser.add_argument('--backend', choices=['remote', 'native'], default='remote',
                        help='Compare the SSH image widget with native Qt/VTK (use VirtualGL on VNC).')
    parser.add_argument('--full-repaint', action='store_true', help='Control: repaint the full widget every frame.')
    parser.add_argument('--compare', action='store_true', help='Run full/dirty/dirty/full controls at each size.')
    args = parser.parse_args()

    if args.backend == 'remote':
        from autoflow.ui.launcher import _configure_ssh_rendering
        _configure_ssh_rendering()
    import h5py
    import numpy as np
    import pyvista as pv
    from PySide6 import QtCore, QtGui, QtTest, QtWidgets
    from autoflow.core.models import Workspace
    from autoflow.ui import remote_plotter
    from autoflow.ui.remote_plotter import RemotePlotter
    from autoflow.ui.viewer import SceneController

    if args.backend == 'native':
        from pyvistaqt import QtInteractor
        plotter_class = QtInteractor
    else:
        plotter_class = RemotePlotter

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    current_full_repaint = args.full_repaint
    changed_frame_region = remote_plotter._changed_frame_region
    remote_plotter._changed_frame_region = lambda *values: (
        None if current_full_repaint else changed_frame_region(*values))
    class MeasuredPlotter(plotter_class):
        paint_count = 0
        paint_pixels = 0

        def paintEvent(self, event):
            self.paint_count += 1
            self.paint_pixels += sum(rect.width() * rect.height() for rect in event.region())
            super().paintEvent(event)

    with h5py.File(args.input, 'r') as source:
        mag = np.abs(source['img_complex'][:, :, :, 0, 0]).astype(np.float32)
        seg = np.asarray(source['segmask'][:, :, :, 0]) > 0
        spacing = np.asarray(source['Resolution'], dtype=float)
    grid = pv.ImageData(dimensions=seg.shape, spacing=spacing)
    grid.point_data['mask'] = seg.astype(np.float32).ravel(order='F')
    surface = grid.contour([0.5], scalars='mask')
    output = dict(host=platform.node(), qt_platform=app.platformName(),
                  display=os.environ.get('DISPLAY'), input=str(args.input),
                  volume_shape=list(mag.shape), surface_cells=surface.n_cells,
                  backend=args.backend, paint_format=args.paint_format, compare=args.compare,
                  full_repaint=args.full_repaint, frames=args.frames, cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(output, indent=2), encoding='utf-8')

    for kind in args.scenes:
        ws = Workspace()
        ws.mag_raw = mag[..., None]
        ws.resolution = spacing
        plotter = MeasuredPlotter()
        plotter.setWindowTitle('AutoFlow X11 performance test')
        plotter.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.BypassWindowManagerHint)
        plotter.resize(640, 420)
        plotter.show()
        app.processEvents()
        scene = SceneController(plotter, ws, lambda _: None)
        scene.initialize()
        if kind == 'surface':
            plotter.add_mesh(surface, color='#df765b', smooth_shading=True)
        else:
            obj = scene.add_content_render(3)
            if kind == 'slices':
                scene.update_content_render(obj.uid, mode='slices', handles=False)
        plotter.reset_camera()
        plotter.render()
        if args.backend == 'remote':
            plotter._capture_frame()
        output['render_window'] = plotter.render_window.GetClassName()
        output['opengl'] = [line.strip() for line in plotter.render_window.ReportCapabilities().splitlines()
                            if 'vendor string' in line or 'renderer string' in line]
        base_camera = [tuple(vector) for vector in plotter.camera_position]
        policies = [True, False, False, True] if args.compare else [args.full_repaint]
        cases = [(size, policy, index) for size in args.sizes for index, policy in enumerate(policies)]
        for size_text, current_full_repaint, run_index in cases:
            width, height = map(int, size_text.split('x'))
            scene.set_interaction_preview(False)
            plotter.setFixedSize(width, height)
            QtWidgets.QApplication.sync()
            QtTest.QTest.qWait(100)
            app.processEvents()
            if args.backend == 'remote':
                plotter._resize_timer.stop()
                plotter._resize_render_window()
            scene.set_interaction_preview(True)
            renderer_plotter = plotter._plotter if args.backend == 'remote' else plotter
            renderer_plotter.camera_position = base_camera
            timings = []
            renders = [0]
            observer = plotter.render_window.AddObserver('EndEvent', lambda *_: renders.__setitem__(0, renders[0] + 1))
            for index in range(args.frames + 4):
                before_paint = plotter.paint_count
                before_pixels = plotter.paint_pixels
                begin = time.perf_counter()
                plotter.camera.Azimuth(2.0)
                renderer_plotter.render()
                rendered = time.perf_counter()
                before_capture = renders[0]
                if args.backend == 'remote':
                    plotter._frame_timer.stop()
                    plotter._capture_frame()
                    if args.paint_format == 'rgb32':
                        plotter._frame = plotter._frame.convertToFormat(QtGui.QImage.Format_RGB32)
                captured = time.perf_counter()
                app.processEvents()
                QtWidgets.QApplication.sync()
                presented = time.perf_counter()
                if index >= 4:
                    timings.append([1000 * (rendered - begin), 1000 * (captured - rendered),
                                    1000 * (presented - captured), 1000 * (presented - begin),
                                    renders[0] - before_capture, plotter.paint_count - before_paint,
                                    plotter.paint_pixels - before_pixels])
            plotter.render_window.RemoveObserver(observer)
            values = np.asarray(timings)
            record = dict(scene=kind, repaint=('native' if args.backend == 'native' else
                                             'full' if current_full_repaint else 'dirty'), run=run_index,
                          requested_size=[width, height],
                          widget_size=[plotter.width(), plotter.height()], render_size=list(plotter.window_size),
                          median_ms=dict(zip(['render', 'capture', 'present_sync', 'total'],
                                             np.median(values[:, :4], axis=0).tolist())),
                          p95_total_ms=float(np.percentile(values[:, 3], 95)),
                          extra_renders_during_capture=int(values[:, 4].sum()),
                          paints_per_frame=float(np.mean(values[:, 5])),
                          mean_repaint_fraction=float(np.mean(values[:, 6]) / (plotter.width() * plotter.height())),
                          samples_ms=values[:, :4].tolist())
            output['cases'].append(record)
            save()
            print(json.dumps({key: value for key, value in record.items() if key != 'samples_ms'}), flush=True)
        scene._close_interaction_rendering()
        plotter.close()
        plotter.deleteLater()
        app.processEvents()
    save()


if __name__ == '__main__':
    main()
