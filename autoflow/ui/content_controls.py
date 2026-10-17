"""Compact properties for the selected Browser Content render."""

from PySide6 import QtCore, QtWidgets


class ContentRenderControls(QtWidgets.QGroupBox):
    settingsChanged = QtCore.Signal(dict)
    positionChanged = QtCore.Signal(int, float)
    interactionChanged = QtCore.Signal(bool)

    def __init__(self, parent=None):
        super().__init__("3D Volume", parent)
        self._loading = False
        form = QtWidgets.QFormLayout(self)
        self.mode = QtWidgets.QComboBox()
        for label, value in (("Volume", "composite"), ("Maximum intensity", "maximum"),
                             ("Orthogonal planes", "slices")):
            self.mode.addItem(label, value)
        self.mode.currentIndexChanged.connect(self._changed)
        form.addRow("Render", self.mode)
        self.show_planes = QtWidgets.QPushButton("Show 3 orthogonal planes")
        self.show_planes.clicked.connect(self._show_all_planes)
        form.addRow(self.show_planes)
        self.show_planes.setToolTip("Show all three LR/AP/FH cuts and their 3D drag handles")
        self.cmap = QtWidgets.QComboBox()
        self.cmap.addItems(["gray", "turbo", "RdBu_r", "magma", "viridis", "Reds"])
        self.cmap.currentTextChanged.connect(self._changed)
        form.addRow("Colormap", self.cmap)
        self.planes = []
        self.positions = []
        for axis, label in enumerate(("LR", "AP", "FH")):
            row = QtWidgets.QHBoxLayout()
            checkbox = QtWidgets.QCheckBox(label)
            checkbox.setToolTip("Open this cut directly from Volume/MIP; in cut mode, show or hide it")
            checkbox.toggled.connect(lambda checked, a=axis: self._axis_toggled(a, checked))
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setRange(0, 1000)
            slider.setToolTip(f"{label} slice position; moving it from Volume/MIP opens this cut")
            slider.valueChanged.connect(lambda value, a=axis: self._position_changed(a, value))
            slider.sliderPressed.connect(lambda: self.interactionChanged.emit(True))
            slider.sliderReleased.connect(lambda: self.interactionChanged.emit(False))
            row.addWidget(checkbox)
            row.addWidget(slider, 1)
            self.planes.append(checkbox)
            self.positions.append(slider)
            form.addRow(row)
        self.handles = QtWidgets.QCheckBox("Drag colored axes in 3D")
        self.handles.toggled.connect(self._changed)
        form.addRow(self.handles)
        self.threshold = QtWidgets.QDoubleSpinBox()
        self.threshold.setRange(0, 0.99)
        self.threshold.setSingleStep(0.01)
        self.threshold.valueChanged.connect(self._changed)
        form.addRow("Transparent below", self.threshold)
        self.shade = QtWidgets.QCheckBox("Volume shading")
        self.shade.toggled.connect(self._changed)
        form.addRow(self.shade)
        self.hide()

    def set_object(self, obj, settings=None):
        self._loading = True
        self.setVisible(obj is not None)
        if obj is not None:
            self.setTitle("Selected 3D Volume")
            self.mode.setCurrentIndex(max(0, self.mode.findData(settings["mode"])))
            if self.cmap.findText(obj.cmap) < 0:
                self.cmap.addItem(obj.cmap)
            self.cmap.setCurrentText(obj.cmap)
            for axis in range(3):
                self.planes[axis].setChecked(settings["planes"][axis])
                self.positions[axis].setValue(round(1000 * settings["positions"][axis]))
            self.handles.setChecked(settings["handles"])
            self.threshold.setValue(settings["threshold"])
            self.shade.setChecked(settings["shade"])
        self._loading = False
        self._sync_enabled()

    def _sync_enabled(self):
        slices = self.mode.currentData() == "slices"
        for check, slider in zip(self.planes, self.positions):
            check.setEnabled(True)
            slider.setEnabled(not slices or check.isChecked())
        self.handles.setEnabled(slices)
        self.shade.setEnabled(not slices)

    def _activate_planes(self, axis=None):
        self._loading = True
        self.mode.setCurrentIndex(self.mode.findData("slices"))
        for index, check in enumerate(self.planes):
            check.setChecked(axis is None or index == axis)
        self.handles.setChecked(True)
        self._loading = False
        self._changed()

    def _show_all_planes(self):
        self._activate_planes()

    def _axis_toggled(self, axis, checked):
        if self._loading:
            return
        if self.mode.currentData() != "slices":
            self._activate_planes(axis)
        else:
            self._changed()

    def _changed(self, *_):
        self._sync_enabled()
        if self._loading:
            return
        if not any(check.isChecked() for check in self.planes):
            # Keep at least one plane so a checked object remains visible.
            self.planes[0].blockSignals(True)
            self.planes[0].setChecked(True)
            self.planes[0].blockSignals(False)
            self._sync_enabled()
        self.settingsChanged.emit(dict(mode=self.mode.currentData(), cmap=self.cmap.currentText(),
                                       planes=[check.isChecked() for check in self.planes],
                                       handles=self.handles.isChecked(), threshold=self.threshold.value(),
                                       shade=self.shade.isChecked()))

    def _position_changed(self, axis, value):
        if not self._loading:
            if self.mode.currentData() != "slices":
                self._activate_planes(axis)
            self.positionChanged.emit(axis, value / 1000.0)

    def set_positions(self, positions):
        self._loading = True
        for slider, value in zip(self.positions, positions):
            slider.setValue(round(1000 * value))
        self._loading = False
