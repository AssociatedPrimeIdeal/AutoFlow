"""Live, reversible layout controls for the shared 3D colorbar."""

import copy
from PySide6 import QtCore, QtWidgets
from ..config import DEFAULT_SCALAR_BAR_STYLE


class PercentControl(QtWidgets.QWidget):
    valueChanged = QtCore.Signal(float)

    def __init__(self, minimum, maximum, parent=None):
        super().__init__(parent)
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.spin = QtWidgets.QDoubleSpinBox()
        self.spin.setDecimals(1)
        self.spin.setSuffix(' %')
        self.spin.setMinimumWidth(90)
        self.setRange(minimum, maximum)
        row.addWidget(self.slider, 1)
        row.addWidget(self.spin)
        self.slider.valueChanged.connect(lambda value: self.spin.setValue(value/10))
        self.spin.valueChanged.connect(self._changed)

    def _changed(self, value):
        blocker = QtCore.QSignalBlocker(self.slider)
        self.slider.setValue(round(value*10))
        del blocker
        self.valueChanged.emit(value)

    def setRange(self, minimum, maximum):
        self.slider.setRange(round(minimum*10), round(maximum*10))
        self.spin.setRange(minimum, maximum)

    def setValue(self, value):
        self.spin.setValue(value)

    def value(self):
        return self.spin.value()


class ColorbarLayoutDialog(QtWidgets.QDialog):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle('3D Colorbar Layout')
        self.setMinimumWidth(430)
        self._loading = False
        settings = window.workspace.render_settings
        self._before = {key: copy.deepcopy(settings[key]) for key in
                        ('shared_colorbar_bar_cfg', 'shared_colorbar_show') if key in settings}
        self.cfg = dict(DEFAULT_SCALAR_BAR_STYLE)
        self.cfg.update(settings.get('shared_colorbar_bar_cfg', {}))
        form = QtWidgets.QFormLayout(self)
        self.presets = QtWidgets.QComboBox()
        for label, config in [('Custom', None),
                ('Bottom', (False, .25, .045, .65, .12)),
                ('Top', (False, .25, .84, .65, .12)),
                ('Right', (True, .86, .18, .11, .66)),
                ('Left', (True, .04, .18, .11, .66))]:
            self.presets.addItem(label, config)
        form.addRow('Position', self.presets)
        self.orientation = QtWidgets.QComboBox()
        self.orientation.addItem('Horizontal', False)
        self.orientation.addItem('Vertical', True)
        form.addRow('Orientation', self.orientation)
        self.length = PercentControl(15, 95)
        self.thickness = PercentControl(2, 40)
        self.x = PercentControl(0, 98)
        self.y = PercentControl(0, 98)
        for label, control in [('Length', self.length), ('Thickness', self.thickness),
                               ('Horizontal position', self.x), ('Vertical position', self.y)]:
            form.addRow(label, control)
            control.valueChanged.connect(self._changed)
        self.title_font, self.label_font = QtWidgets.QSpinBox(), QtWidgets.QSpinBox()
        for label, control in [('Title size (px)', self.title_font), ('Label size (px)', self.label_font)]:
            control.setRange(1, 100)
            form.addRow(label, control)
            control.valueChanged.connect(self._changed)
        self.show_bar = QtWidgets.QCheckBox('Show colorbar')
        self.show_bar.setChecked(settings.get('shared_colorbar_show', True))
        form.addRow(self.show_bar)
        self.show_bar.toggled.connect(self._changed)
        self._populate()
        self.presets.currentIndexChanged.connect(self._preset)
        self.orientation.currentIndexChanged.connect(self._orientation)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok |
            QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.RestoreDefaults)
        buttons.button(QtWidgets.QDialogButtonBox.Ok).setText('Apply')
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QtWidgets.QDialogButtonBox.RestoreDefaults).clicked.connect(self._reset)
        form.addRow(buttons)

    def _populate(self):
        self._loading = True
        vertical = bool(self.cfg.get('vertical', False))
        self.orientation.setCurrentIndex(int(vertical))
        self.length.setValue(100*self.cfg['height' if vertical else 'width'])
        self.thickness.setValue(100*self.cfg['width' if vertical else 'height'])
        self.x.setRange(0, max(0, 98-100*self.cfg['width']))
        self.y.setRange(0, max(0, 98-100*self.cfg['height']))
        self.x.setValue(100*self.cfg['position_x'])
        self.y.setValue(100*self.cfg['position_y'])
        self.title_font.setValue(self.cfg['title_font_size'])
        self.label_font.setValue(self.cfg['label_font_size'])
        self._loading = False

    def _preview(self):
        settings = self.window.workspace.render_settings
        settings['shared_colorbar_bar_cfg'] = dict(self.cfg)
        settings['shared_colorbar_show'] = self.show_bar.isChecked()
        self.window.scene._refresh_shared_scalar_bar(render=True)

    def _changed(self, *_):
        if self._loading:
            return
        self._loading = True
        vertical = bool(self.orientation.currentData())
        width, height = ((self.thickness.value(), self.length.value()) if vertical else
                         (self.length.value(), self.thickness.value()))
        self.x.setRange(0, max(0, 98-width))
        self.y.setRange(0, max(0, 98-height))
        self.cfg.update(vertical=vertical, width=width/100, height=height/100,
            position_x=self.x.value()/100, position_y=self.y.value()/100,
            title_font_size=self.title_font.value(), label_font_size=self.label_font.value())
        self.presets.setCurrentIndex(0)
        self._loading = False
        self._preview()

    def _preset(self, *_):
        if self._loading or self.presets.currentData() is None:
            return
        vertical, x, y, width, height = self.presets.currentData()
        self.cfg.update(vertical=vertical, position_x=x, position_y=y, width=width, height=height)
        self._populate()
        self._preview()

    def _orientation(self, *_):
        if self._loading:
            return
        self._changed()

    def _reset(self):
        self.cfg = dict(DEFAULT_SCALAR_BAR_STYLE)
        self._populate()
        self.show_bar.setChecked(True)
        self._preview()

    def reject(self):
        settings = self.window.workspace.render_settings
        for key in ('shared_colorbar_bar_cfg', 'shared_colorbar_show'):
            if key in self._before:
                settings[key] = copy.deepcopy(self._before[key])
            else:
                settings.pop(key, None)
        self.window.scene._refresh_shared_scalar_bar(render=True)
        super().reject()
