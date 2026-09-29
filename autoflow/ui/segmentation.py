from PySide6 import QtCore, QtWidgets


SOURCE_LABELS = {
    "original": "Original",
    "imported": "Imported",
    "threshold": "Threshold",
    "auto": "Auto",
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
        self.slider_opacity.setValue(15)
        self.text_provenance = QtWidgets.QPlainTextEdit()
        self.text_provenance.setReadOnly(True)
        self.text_provenance.setMaximumHeight(100)
        source_actions = QtWidgets.QWidget()
        source_actions_layout = QtWidgets.QHBoxLayout(source_actions)
        source_actions_layout.setContentsMargins(0, 0, 0, 0)
        source_actions_layout.setSpacing(4)
        self.btn_import = QtWidgets.QPushButton("Import...")
        self.btn_save = QtWidgets.QPushButton("Save...")
        self.btn_import.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogOpenButton)
        )
        self.btn_save.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton)
        )
        source_actions_layout.addWidget(self.btn_import)
        source_actions_layout.addWidget(self.btn_save)
        source_form.addRow("Active Source", self.combo_source)
        source_form.addRow("Visible", self.check_visible)
        source_form.addRow("Opacity", self.slider_opacity)
        source_form.addRow("Provenance", self.text_provenance)
        source_form.addRow("Actions", source_actions)
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

        actions_group = QtWidgets.QGroupBox("Edit")
        actions_layout = QtWidgets.QVBoxLayout(actions_group)
        self.btn_run_auto = QtWidgets.QPushButton("Run Automatic Segmentation")
        self.btn_run_auto.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPlay)
        )
        self.btn_run_auto.setProperty("role", "primary")
        self.btn_external_editor = QtWidgets.QPushButton("Open in SpatioTemporal Labeler")
        self.btn_external_editor.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_FileDialogDetailedView)
        )
        self.btn_external_editor.setProperty("role", "primary")
        self.btn_external_editor.setToolTip(
            "Export magnitude, flow components, PC-MRA, and segmentation, then open the optional external editor."
        )
        actions_layout.addWidget(self.btn_run_auto)
        actions_layout.addWidget(self.btn_external_editor)
        layout.addWidget(actions_group)

        cleanup_group = QtWidgets.QGroupBox("4D Connected Components")
        cleanup_form = QtWidgets.QFormLayout(cleanup_group)
        self.check_cleanup_4d = QtWidgets.QCheckBox("Enable cleanup")
        self.combo_cleanup_4d_mode = QtWidgets.QComboBox()
        self.combo_cleanup_4d_mode.addItem("Remove below volume", "absolute")
        self.combo_cleanup_4d_mode.addItem("Keep largest per label", "largest")
        self.spin_cleanup_4d_volume = QtWidgets.QDoubleSpinBox()
        self.spin_cleanup_4d_volume.setRange(0.0, 1e9)
        self.spin_cleanup_4d_volume.setDecimals(2)
        self.spin_cleanup_4d_volume.setSuffix(" mm^3")
        self.btn_apply_cleanup_4d = QtWidgets.QPushButton("Apply / Rebuild")
        cleanup_form.addRow("Mode", self.combo_cleanup_4d_mode)
        cleanup_form.addRow("Minimum volume", self.spin_cleanup_4d_volume)
        cleanup_form.addRow("", self.check_cleanup_4d)
        cleanup_form.addRow("", self.btn_apply_cleanup_4d)
        layout.addWidget(cleanup_group)

        self.label_status = QtWidgets.QLabel("No active segmentation")
        self.label_status.setWordWrap(True)
        layout.addWidget(self.label_status)
        layout.addStretch(1)
