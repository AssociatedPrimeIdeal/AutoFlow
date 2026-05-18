from PyQt5 import QtCore, QtWidgets


SOURCE_LABELS = {
    "original": "Original",
    "imported": "Imported",
    "threshold": "Threshold",
    "auto": "Auto",
}


class SegmentationConfigDialog(QtWidgets.QDialog):
    def __init__(self, workspace, parent=None):
        super().__init__(parent)
        self.workspace = workspace
        self.setWindowTitle("Configure Segmentation")
        self.resize(520, 420)
        self._build_ui()
        self._load_from_workspace()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        mode_row = QtWidgets.QHBoxLayout()
        self.combo_mode = QtWidgets.QComboBox()
        self.combo_mode.addItems(["input", "threshold", "auto"])
        self.combo_mode.currentTextChanged.connect(self._sync_mode)
        mode_row.addWidget(QtWidgets.QLabel("Mode"))
        mode_row.addWidget(self.combo_mode, 1)
        layout.addLayout(mode_row)

        self.stack = QtWidgets.QStackedWidget()
        layout.addWidget(self.stack, 1)

        self.stack.addWidget(self._build_input_page())
        self.stack.addWidget(self._build_threshold_page())
        self.stack.addWidget(self._build_auto_page())

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _build_input_page(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        self.radio_input_original = QtWidgets.QRadioButton("Use original segmentation")
        self.radio_input_import = QtWidgets.QRadioButton("Import external segmentation")
        self.radio_input_original.setChecked(True)
        layout.addWidget(self.radio_input_original)
        layout.addWidget(self.radio_input_import)
        row = QtWidgets.QHBoxLayout()
        self.edit_import_path = QtWidgets.QLineEdit()
        self.btn_browse_import = QtWidgets.QPushButton("Browse...")
        self.btn_browse_import.clicked.connect(self._browse_import)
        row.addWidget(self.edit_import_path, 1)
        row.addWidget(self.btn_browse_import)
        layout.addLayout(row)
        layout.addStretch(1)
        return page

    def _build_threshold_page(self):
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        self.combo_threshold_scalar = QtWidgets.QComboBox()
        self.combo_threshold_scalar.addItems(["mag", "pcmra", "pcmra_std"])
        self.combo_threshold_mode = QtWidgets.QComboBox()
        self.combo_threshold_mode.addItems(["auto", "manual"])
        self.combo_threshold_mode.currentTextChanged.connect(self._sync_threshold_mode)
        self.spin_threshold_value = QtWidgets.QDoubleSpinBox()
        self.spin_threshold_value.setDecimals(6)
        self.spin_threshold_value.setRange(-1e9, 1e9)
        self.spin_threshold_value.setSingleStep(0.05)
        self.chk_keep_largest_cc = QtWidgets.QCheckBox()
        self.chk_threshold_closing = QtWidgets.QCheckBox()
        self.chk_threshold_opening = QtWidgets.QCheckBox()
        self.spin_min_cc_volume = QtWidgets.QDoubleSpinBox()
        self.spin_min_cc_volume.setDecimals(3)
        self.spin_min_cc_volume.setRange(0.0, 1e9)
        self.spin_min_cc_volume.setSuffix(" mm^3")
        form.addRow("Scalar", self.combo_threshold_scalar)
        form.addRow("Threshold", self.combo_threshold_mode)
        form.addRow("Manual Value", self.spin_threshold_value)
        form.addRow("Keep Largest CC", self.chk_keep_largest_cc)
        form.addRow("Closing", self.chk_threshold_closing)
        form.addRow("Opening", self.chk_threshold_opening)
        form.addRow("Min Component Volume", self.spin_min_cc_volume)
        return page

    def _build_auto_page(self):
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        self.edit_auto_backend = QtWidgets.QLineEdit()
        self.edit_auto_model = QtWidgets.QLineEdit()
        self.edit_auto_checkpoint = QtWidgets.QLineEdit()
        self.btn_browse_checkpoint = QtWidgets.QPushButton("Browse...")
        self.btn_browse_checkpoint.clicked.connect(self._browse_checkpoint)
        checkpoint_row = QtWidgets.QHBoxLayout()
        checkpoint_row.addWidget(self.edit_auto_checkpoint, 1)
        checkpoint_row.addWidget(self.btn_browse_checkpoint)
        checkpoint_widget = QtWidgets.QWidget()
        checkpoint_widget.setLayout(checkpoint_row)
        self.edit_auto_device = QtWidgets.QLineEdit()
        self.edit_auto_label_map = QtWidgets.QPlainTextEdit()
        self.edit_auto_label_map.setPlaceholderText('{"aorta": 1}')
        self.edit_auto_label_map.setMaximumHeight(90)
        note = QtWidgets.QLabel("Auto backend config is wired in the GUI, but inference is not implemented in this branch.")
        note.setWordWrap(True)
        form.addRow("Backend", self.edit_auto_backend)
        form.addRow("Model", self.edit_auto_model)
        form.addRow("Checkpoint", checkpoint_widget)
        form.addRow("Device", self.edit_auto_device)
        form.addRow("Label Map", self.edit_auto_label_map)
        form.addRow("", note)
        return page

    def _browse_import(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import Segmentation",
            self.edit_import_path.text().strip(),
            "Segmentation (*.h5 *.hdf5 *.npy *.npz);;All (*)",
        )
        if path:
            self.edit_import_path.setText(path)
            self.radio_input_import.setChecked(True)

    def _browse_checkpoint(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Checkpoint", "", "All (*)")
        if path:
            self.edit_auto_checkpoint.setText(path)

    def _sync_mode(self, mode):
        index = {"input": 0, "threshold": 1, "auto": 2}.get(str(mode), 0)
        self.stack.setCurrentIndex(index)

    def _sync_threshold_mode(self, mode):
        self.spin_threshold_value.setEnabled(str(mode) == "manual")

    def _load_from_workspace(self):
        seg = self.workspace.segmentation
        self.combo_mode.setCurrentText(seg.mode if seg.mode in ("input", "threshold", "auto") else "input")
        self.radio_input_original.setChecked(seg.input_source != "imported")
        self.radio_input_import.setChecked(seg.input_source == "imported")
        self.edit_import_path.setText(seg.import_path)
        self.combo_threshold_scalar.setCurrentText(seg.threshold_scalar if seg.threshold_scalar in ("mag", "pcmra", "pcmra_std") else "pcmra")
        self.combo_threshold_mode.setCurrentText("auto" if seg.threshold_value == "auto" else "manual")
        if seg.threshold_value != "auto":
            try:
                self.spin_threshold_value.setValue(float(seg.threshold_value))
            except Exception:
                self.spin_threshold_value.setValue(0.0)
        self.chk_keep_largest_cc.setChecked(bool(seg.threshold_keep_largest_cc))
        self.chk_threshold_closing.setChecked(bool(seg.threshold_closing))
        self.chk_threshold_opening.setChecked(bool(seg.threshold_opening))
        self.spin_min_cc_volume.setValue(float(seg.threshold_min_component_volume_mm3))
        self.edit_auto_backend.setText(seg.auto_backend)
        self.edit_auto_model.setText(seg.auto_model)
        self.edit_auto_checkpoint.setText(seg.auto_checkpoint)
        self.edit_auto_device.setText(seg.auto_device)
        self.edit_auto_label_map.setPlainText(seg.auto_label_map)
        self._sync_mode(self.combo_mode.currentText())
        self._sync_threshold_mode(self.combo_threshold_mode.currentText())

    def values(self):
        return {
            "mode": self.combo_mode.currentText(),
            "input_source": "imported" if self.radio_input_import.isChecked() else "original",
            "import_path": self.edit_import_path.text().strip(),
            "threshold_scalar": self.combo_threshold_scalar.currentText(),
            "threshold_value": "auto" if self.combo_threshold_mode.currentText() == "auto" else float(self.spin_threshold_value.value()),
            "threshold_keep_largest_cc": self.chk_keep_largest_cc.isChecked(),
            "threshold_closing": self.chk_threshold_closing.isChecked(),
            "threshold_opening": self.chk_threshold_opening.isChecked(),
            "threshold_min_component_volume_mm3": float(self.spin_min_cc_volume.value()),
            "auto_backend": self.edit_auto_backend.text().strip(),
            "auto_model": self.edit_auto_model.text().strip(),
            "auto_checkpoint": self.edit_auto_checkpoint.text().strip(),
            "auto_device": self.edit_auto_device.text().strip(),
            "auto_label_map": self.edit_auto_label_map.toPlainText().strip(),
        }


class SegmentationDock(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        source_group = QtWidgets.QGroupBox("Source")
        source_form = QtWidgets.QFormLayout(source_group)
        self.combo_source = QtWidgets.QComboBox()
        self.check_visible = QtWidgets.QCheckBox()
        self.check_visible.setChecked(True)
        self.slider_opacity = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_opacity.setRange(0, 100)
        self.slider_opacity.setValue(35)
        self.text_provenance = QtWidgets.QPlainTextEdit()
        self.text_provenance.setReadOnly(True)
        self.text_provenance.setMaximumHeight(100)
        source_form.addRow("Active Source", self.combo_source)
        source_form.addRow("Visible", self.check_visible)
        source_form.addRow("Opacity", self.slider_opacity)
        source_form.addRow("Provenance", self.text_provenance)
        layout.addWidget(source_group)

        labels_group = QtWidgets.QGroupBox("Labels")
        labels_layout = QtWidgets.QVBoxLayout(labels_group)
        self.table_labels = QtWidgets.QTableWidget(0, 4)
        self.table_labels.setHorizontalHeaderLabels(["Label", "Name", "Voxels", "Color"])
        self.table_labels.horizontalHeader().setStretchLastSection(True)
        self.table_labels.verticalHeader().setVisible(False)
        self.table_labels.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table_labels.setEditTriggers(
            QtWidgets.QAbstractItemView.DoubleClicked
            | QtWidgets.QAbstractItemView.EditKeyPressed
            | QtWidgets.QAbstractItemView.SelectedClicked
        )
        labels_layout.addWidget(self.table_labels)
        layout.addWidget(labels_group, 1)

        active_group = QtWidgets.QGroupBox("Active Label")
        active_form = QtWidgets.QFormLayout(active_group)
        self.spin_active_label = QtWidgets.QSpinBox()
        self.spin_active_label.setRange(1, 999)
        self.edit_active_name = QtWidgets.QLineEdit()
        self.btn_active_color = QtWidgets.QPushButton("Change...")
        active_form.addRow("Label ID", self.spin_active_label)
        active_form.addRow("Name", self.edit_active_name)
        active_form.addRow("Color", self.btn_active_color)
        layout.addWidget(active_group)

        tools_group = QtWidgets.QGroupBox("Tools")
        tools_layout = QtWidgets.QGridLayout(tools_group)
        self.btn_tool_brush = QtWidgets.QPushButton("Brush")
        self.btn_tool_erase = QtWidgets.QPushButton("Erase")
        self.btn_tool_fill = QtWidgets.QPushButton("Fill")
        self.btn_tool_relabel = QtWidgets.QPushButton("Relabel")
        self.tool_buttons = [self.btn_tool_brush, self.btn_tool_erase, self.btn_tool_fill, self.btn_tool_relabel]
        for btn in self.tool_buttons:
            btn.setCheckable(True)
        self.btn_tool_brush.setChecked(True)
        self.spin_brush_radius = QtWidgets.QSpinBox()
        self.spin_brush_radius.setRange(1, 50)
        self.spin_brush_radius.setValue(3)
        self.chk_edit_all_timepoints = QtWidgets.QCheckBox("Edit Across All Frames (3D)")
        self.chk_edit_all_timepoints.setChecked(True)
        tools_layout.addWidget(self.btn_tool_brush, 0, 0)
        tools_layout.addWidget(self.btn_tool_erase, 0, 1)
        tools_layout.addWidget(self.btn_tool_fill, 1, 0)
        tools_layout.addWidget(self.btn_tool_relabel, 1, 1)
        tools_layout.addWidget(QtWidgets.QLabel("Brush Radius"), 2, 0)
        tools_layout.addWidget(self.spin_brush_radius, 2, 1)
        tools_layout.addWidget(self.chk_edit_all_timepoints, 3, 0, 1, 2)
        layout.addWidget(tools_group)

        actions_group = QtWidgets.QGroupBox("Edit")
        actions_layout = QtWidgets.QGridLayout(actions_group)
        self.btn_undo = QtWidgets.QPushButton("Undo")
        self.btn_redo = QtWidgets.QPushButton("Redo")
        self.btn_apply = QtWidgets.QPushButton("Apply")
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        actions_layout.addWidget(self.btn_undo, 0, 0)
        actions_layout.addWidget(self.btn_redo, 0, 1)
        actions_layout.addWidget(self.btn_apply, 1, 0)
        actions_layout.addWidget(self.btn_cancel, 1, 1)
        layout.addWidget(actions_group)

        self.label_status = QtWidgets.QLabel("No active segmentation")
        self.label_status.setWordWrap(True)
        layout.addWidget(self.label_status)
        layout.addStretch(1)
