"""Current-view movie resolution controls and real VTK rendering at that size."""

import numpy as np
from PySide6 import QtWidgets
from .parameter_help import bind_help


class CurrentViewVideoDialog(QtWidgets.QDialog):
    def __init__(self, parent, fps, *, rotate=False):
        super().__init__(parent)
        self.setWindowTitle('Current 3D View Video')
        form = QtWidgets.QFormLayout(self)
        self.fps = QtWidgets.QDoubleSpinBox()
        self.fps.setRange(.1, 120)
        self.fps.setDecimals(2)
        self.fps.setValue(fps)
        self.cycles = QtWidgets.QSpinBox()
        self.cycles.setRange(1, 100)
        self.cycles.setValue(3)
        form.addRow('Frames per second', self.fps)
        bind_help(form, self.fps, 'Encoded playback frames per second. Capture uses the visible 3D layers with the current camera, colours and ranges.')
        form.addRow('Cardiac cycles', self.cycles)
        bind_help(form, self.cycles, 'Number of complete cardiac cycles to record, starting at the displayed phase. Camera, phase and playback state are restored afterwards.')
        self.camera_mode = QtWidgets.QComboBox()
        self.camera_mode.addItem('Fixed camera', False)
        self.camera_mode.addItem('Rotate from current view', True)
        self.camera_mode.setCurrentIndex(int(rotate))
        form.addRow('Camera', self.camera_mode)
        self.rotation_axis = QtWidgets.QComboBox()
        for label, axis in [('Screen vertical', 'screen_vertical'), ('Screen horizontal', 'screen_horizontal'),
                             ('Patient LR', 'lr'), ('Patient AP', 'ap'), ('Patient FH', 'fh')]:
            self.rotation_axis.addItem(label, axis)
        form.addRow('Rotation axis', self.rotation_axis)
        self.rotation_angle = QtWidgets.QDoubleSpinBox()
        self.rotation_angle.setRange(1, 3600)
        self.rotation_angle.setValue(360)
        self.rotation_angle.setSuffix('°')
        form.addRow('Angle', self.rotation_angle)
        self.rotation_speed = QtWidgets.QDoubleSpinBox()
        self.rotation_speed.setRange(1, 360)
        self.rotation_speed.setValue(30)
        self.rotation_speed.setSuffix(' °/s')
        form.addRow('Rotation speed', self.rotation_speed)
        self.rotation_direction = QtWidgets.QComboBox()
        self.rotation_direction.addItem('Forward', 1)
        self.rotation_direction.addItem('Reverse', -1)
        form.addRow('Direction', self.rotation_direction)
        self.duration = QtWidgets.QLabel()
        form.addRow('Duration', self.duration)
        self.hide_orientation = QtWidgets.QCheckBox('Hide lower-left orientation marker')
        self.hide_orientation.setChecked(False)
        form.addRow(self.hide_orientation)
        bind_help(form, self.rotation_axis, 'Screen axes are taken from the starting camera and remain fixed during the orbit. Patient axes follow the displayed LR/AP/FH directions. The focal point and zoom are retained.')
        bind_help(form, self.rotation_speed, 'Degrees per second of encoded playback. Rotation duration comes from angle and speed; cardiac phases repeat from the displayed phase throughout the orbit.')
        self._form = form
        self.camera_mode.currentIndexChanged.connect(self._update_motion)
        for control in (self.rotation_angle, self.rotation_speed, self.fps):
            control.valueChanged.connect(self._update_motion)
        self._update_motion()
        self.resolution = QtWidgets.QComboBox()
        for label, size in (('Current view size', None), ('1080p (1920 × 1080)', (1920, 1080)),
                            ('1440p (2560 × 1440)', (2560, 1440)),
                            ('4K (3840 × 2160)', (3840, 2160)), ('Custom', 'custom')):
            self.resolution.addItem(label, size)
        self.resolution.setCurrentIndex(1)
        form.addRow('Resolution', self.resolution)
        bind_help(form, self.resolution, 'Render new pixels at the selected dimensions, rather than enlarging a screenshot. '
            'The scene keeps its proportions and fits the output frame. A different aspect ratio can add space. '
            'Camera, text and viewport size are restored after completion or cancellation.')
        self.dimensions = QtWidgets.QWidget()
        dimensions = QtWidgets.QHBoxLayout(self.dimensions)
        dimensions.setContentsMargins(0, 0, 0, 0)
        self.width, self.height = QtWidgets.QSpinBox(), QtWidgets.QSpinBox()
        for spin, initial in ((self.width, 1920), (self.height, 1080)):
            spin.setRange(64, 8192)
            spin.setSingleStep(2)
            spin.setValue(initial)
            dimensions.addWidget(spin)
        dimensions.insertWidget(1, QtWidgets.QLabel('×'))
        form.addRow('Width × height (px)', self.dimensions)
        bind_help(form, self.dimensions, 'Custom output dimensions in pixels. Odd dimensions are padded by one outer edge pixel for MP4 encoding.')
        self.dimensions.setEnabled(False)
        self.resolution.currentIndexChanged.connect(
            lambda _: self.dimensions.setEnabled(self.resolution.currentData() == 'custom'))
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _update_motion(self, *_):
        rotating = bool(self.camera_mode.currentData())
        self.cycles.setEnabled(not rotating)
        for control in (self.rotation_axis, self.rotation_angle, self.rotation_speed, self.rotation_direction, self.duration):
            self._form.setRowVisible(control, rotating)
        from math import ceil
        frames = ceil(self.rotation_angle.value()/self.rotation_speed.value()*self.fps.value())+1
        self.duration.setText(f'{frames/self.fps.value():.1f} s · {frames} frames')

    def motion_options(self):
        return dict(rotate=bool(self.camera_mode.currentData()), rotation_axis=self.rotation_axis.currentData(),
            rotation_angle=self.rotation_angle.value()*self.rotation_direction.currentData(),
            rotation_speed=self.rotation_speed.value(), hide_orientation_marker=self.hide_orientation.isChecked())

    def output_resolution(self):
        value = self.resolution.currentData()
        return (self.width.value(), self.height.value()) if value == 'custom' else value


def validate_resolution(resolution):
    if resolution is None:
        return None
    values = np.asarray(resolution, dtype=float)
    if (values.shape != (2,) or not np.all(np.isfinite(values)) or
            np.any(values < 1) or np.any(values > 8192) or np.any(values != np.floor(values))):
        raise ValueError('Resolution must contain two integer pixel dimensions from 1 to 8192')
    return tuple(int(value) for value in values)


class MovieCapture:
    def __init__(self, plotter, resolution):
        self.plotter = plotter
        self.resolution = validate_resolution(resolution)
        self.source_size = None
        self._fonts = []
        window = getattr(plotter, 'render_window', None)
        if window is not None:
            self.source_size = tuple(window.GetSize())
        if self.resolution is None:
            return
        if not self.source_size or min(self.source_size) <= 0:
            raise ValueError('The 3D renderer has no valid source size')
        factor = min(target/source for target, source in zip(self.resolution, self.source_size))
        # VTK text sizes are pixels, so scale them with the fitted source view.
        seen = set()
        for renderer in window.GetRenderers():
            for actor in renderer.GetViewProps():
                for getter in ('GetTextProperty', 'GetLabelTextProperty', 'GetTitleTextProperty', 'GetAnnotationTextProperty'):
                    if not callable(getattr(actor, getter, None)):
                        continue
                    try:
                        prop = getattr(actor, getter)()
                    except TypeError:  # Some axes getters need an axis index.
                        continue
                    if prop is None or prop.GetAddressAsString('') in seen:
                        continue
                    seen.add(prop.GetAddressAsString(''))
                    size = prop.GetFontSize()
                    self._fonts.append((prop, size))
                    prop.SetFontSize(max(1, round(size*factor)))

    def image(self):
        if self.resolution is None:
            return self.plotter.screenshot(return_img=True)
        camera = self.plotter.camera
        before = camera.copy()
        source_aspect = self.source_size[0]/self.source_size[1]
        target_aspect = self.resolution[0]/self.resolution[1]
        try:
            # Expand the view if the new frame would otherwise crop an edge.
            if camera.GetParallelProjection():
                camera.SetParallelScale(camera.GetParallelScale()*max(1., source_aspect/target_aspect))
            else:
                fit = max(1., target_aspect/source_aspect) if camera.GetUseHorizontalViewAngle() else max(1., source_aspect/target_aspect)
                camera.SetViewAngle(np.degrees(2*np.arctan(np.tan(np.radians(camera.GetViewAngle())/2)*fit)))
            return self.plotter.screenshot(return_img=True, window_size=self.resolution, scale=1)
        finally:
            camera.DeepCopy(before)

    def close(self):
        for prop, size in self._fonts:
            prop.SetFontSize(size)
        if self.resolution is not None and self.source_size:
            self.plotter.render_window.SetSize(*self.source_size)
