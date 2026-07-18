from PyQt5 import QtCore, QtWidgets


class H5CaseSelectDialog(QtWidgets.QDialog):
    def __init__(self, cases, parent=None):
        super().__init__(parent)
        self._cases = list(cases or [])
        self._selected_case = None
        self.setWindowTitle("Select H5 Case")
        self.resize(860, 360)
        self._build_ui()
        self._populate_cases()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        intro = QtWidgets.QLabel(
            "Select the H5 data-group path to load when one file contains multiple supported cases."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        layout.addWidget(splitter, 1)

        case_group = QtWidgets.QGroupBox("Cases")
        case_layout = QtWidgets.QVBoxLayout(case_group)
        self.list_cases = QtWidgets.QListWidget()
        self.list_cases.currentRowChanged.connect(self._on_case_changed)
        case_layout.addWidget(self.list_cases)
        splitter.addWidget(case_group)

        info_group = QtWidgets.QGroupBox("Selected Case")
        form = QtWidgets.QFormLayout(info_group)
        self.label_case_info = QtWidgets.QLabel("-")
        self.label_case_info.setWordWrap(True)
        self.label_case_group = QtWidgets.QLabel("-")
        self.label_case_output = QtWidgets.QLabel("-")
        form.addRow("Case", self.label_case_info)
        form.addRow("Data Group", self.label_case_group)
        form.addRow("Output Name", self.label_case_output)
        splitter.addWidget(info_group)
        splitter.setSizes([380, 480])

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _populate_cases(self):
        self.list_cases.clear()
        for case in self._cases:
            self.list_cases.addItem(case.display_name or case.output_name or case.input_path)
        if self._cases:
            self.list_cases.setCurrentRow(0)

    def _on_case_changed(self, row):
        if row < 0 or row >= len(self._cases):
            self._selected_case = None
            self.label_case_info.setText("-")
            self.label_case_group.setText("-")
            self.label_case_output.setText("-")
            return
        case = self._cases[row]
        self._selected_case = case
        self.label_case_info.setText(case.display_name or case.input_path)
        self.label_case_group.setText(str(case.source_group or "<root>"))
        self.label_case_output.setText(case.output_name or case.input_path)

    def selected_case(self):
        return self._selected_case

    def accept(self):
        if self._selected_case is None:
            QtWidgets.QMessageBox.warning(self, "No Case Selected", "Select an H5 case before continuing.")
            return
        super().accept()


class DicomImportDialog(QtWidgets.QDialog):
    def __init__(self, cases, preview_loader, parent=None):
        super().__init__(parent)
        self._cases = list(cases or [])
        self._preview_loader = preview_loader
        self._preview_cache = {}
        self._selected_case = None
        self.setWindowTitle("Confirm DICOM Import")
        self.resize(860, 420)
        self._build_ui()
        self._populate_cases()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        intro = QtWidgets.QLabel(
            "Select a scanned DICOM case, review the detected load parameters, adjust them if needed, and confirm before loading."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        layout.addWidget(splitter, 1)

        case_group = QtWidgets.QGroupBox("Cases")
        case_layout = QtWidgets.QVBoxLayout(case_group)
        self.list_cases = QtWidgets.QListWidget()
        self.list_cases.currentRowChanged.connect(self._on_case_changed)
        case_layout.addWidget(self.list_cases)
        splitter.addWidget(case_group)

        form_group = QtWidgets.QGroupBox("Detected Parameters")
        form = QtWidgets.QFormLayout(form_group)
        self.label_case_info = QtWidgets.QLabel("-")
        self.label_case_info.setWordWrap(True)
        self.edit_matrix_size = QtWidgets.QLineEdit()
        self.edit_matrix_size.setReadOnly(True)
        self.edit_resolution = QtWidgets.QLineEdit()
        self.edit_venc = QtWidgets.QLineEdit()
        self.edit_spatial_order = QtWidgets.QLineEdit()
        self.edit_venc_order = QtWidgets.QLineEdit()
        self.edit_rr = QtWidgets.QLineEdit()
        self.label_preview_status = QtWidgets.QLabel("")
        self.label_preview_status.setWordWrap(True)
        form.addRow("Case", self.label_case_info)
        form.addRow("Matrix Size XYZT", self.edit_matrix_size)
        form.addRow("Resolution XYZ", self.edit_resolution)
        form.addRow("VENC XYZ", self.edit_venc)
        form.addRow("Spatial Order", self.edit_spatial_order)
        form.addRow("VENC Order", self.edit_venc_order)
        form.addRow("RR (ms)", self.edit_rr)
        form.addRow("Status", self.label_preview_status)
        splitter.addWidget(form_group)
        splitter.setSizes([300, 560])

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _populate_cases(self):
        self.list_cases.clear()
        for case in self._cases:
            self.list_cases.addItem(case.display_name or case.output_name or case.input_path)
        if self._cases:
            self.list_cases.setCurrentRow(0)

    def _format_triplet(self, values):
        return ", ".join(f"{float(x):.6g}" for x in list(values)[:3])

    def _format_labels(self, values):
        return ", ".join(str(x).upper() for x in list(values)[:3])

    def _format_matrix_size(self, values):
        vals = [int(float(x)) for x in list(values)[:4]]
        return " x ".join(str(v) for v in vals)

    def _preview_key(self, case):
        return (str(case.input_path), str(case.source_group or ""))

    def _preview_for_case(self, case):
        key = self._preview_key(case)
        if key not in self._preview_cache:
            self._preview_cache[key] = dict(self._preview_loader(case))
        return self._preview_cache[key]

    def _set_preview_fields(self, preview):
        self.edit_matrix_size.setText(self._format_matrix_size(preview.get("matrix_size", [1, 1, 1, 1])))
        self.edit_resolution.setText(self._format_triplet(preview.get("resolution", [1.0, 1.0, 1.0])))
        self.edit_venc.setText(self._format_triplet(preview.get("venc", [150.0, 150.0, 150.0])))
        self.edit_spatial_order.setText(self._format_labels(preview.get("spatial_order", ["LR", "AP", "FH"])))
        self.edit_venc_order.setText(self._format_labels(preview.get("venc_order", ["LR", "AP", "FH"])))
        self.edit_rr.setText(f"{float(preview.get('rr', 1000.0)):.6g}")

    def _on_case_changed(self, row):
        if row < 0 or row >= len(self._cases):
            self._selected_case = None
            self.label_case_info.setText("-")
            self.label_preview_status.setText("")
            return
        case = self._cases[row]
        self._selected_case = case
        try:
            preview = self._preview_for_case(case)
        except Exception as exc:
            self.label_case_info.setText(case.display_name or case.input_path)
            self.edit_matrix_size.clear()
            self.edit_resolution.clear()
            self.edit_venc.clear()
            self.edit_spatial_order.clear()
            self.edit_venc_order.clear()
            self.edit_rr.clear()
            self.label_preview_status.setText(f"Preview failed: {type(exc).__name__}: {exc}")
            return
        info = [
            case.display_name or case.input_path,
            f"files={int(preview.get('file_count', 0))}",
            f"group={preview.get('group_kind', '-')}",
        ]
        self.label_case_info.setText(" | ".join(info))
        self._set_preview_fields(preview)
        self.label_preview_status.setText("Preview loaded. Edit values if the detected parameters need correction.")

    def _parse_float_triplet(self, text, field_name):
        values = []
        for token in str(text or "").replace(";", ",").split(","):
            token = token.strip()
            if token:
                try:
                    values.append(float(token))
                except ValueError as exc:
                    raise ValueError(f"{field_name} must contain numeric values") from exc
        if len(values) == 1:
            values *= 3
        if len(values) != 3:
            raise ValueError(f"{field_name} must contain exactly 3 values")
        return values

    def _parse_label_triplet(self, text, field_name):
        values = [token.strip().upper() for token in str(text or "").replace(";", ",").split(",") if token.strip()]
        if len(values) != 3:
            raise ValueError(f"{field_name} must contain exactly 3 labels")
        return values

    def selected_case(self):
        return self._selected_case

    def parameter_overrides(self):
        return {
            "resolution": self._parse_float_triplet(self.edit_resolution.text(), "Resolution"),
            "venc": self._parse_float_triplet(self.edit_venc.text(), "VENC"),
            "spatial_order": self._parse_label_triplet(self.edit_spatial_order.text(), "Spatial order"),
            "venc_order": self._parse_label_triplet(self.edit_venc_order.text(), "VENC order"),
            "rr": float(self.edit_rr.text().strip()),
        }

    def accept(self):
        if self._selected_case is None:
            QtWidgets.QMessageBox.warning(self, "No Case Selected", "Select a DICOM case before continuing.")
            return
        if self._preview_key(self._selected_case) not in self._preview_cache:
            QtWidgets.QMessageBox.warning(self, "Preview Missing", "Review a valid DICOM preview before continuing.")
            return
        try:
            overrides = self.parameter_overrides()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid Parameters", str(exc))
            return
        if overrides["rr"] <= 0:
            QtWidgets.QMessageBox.warning(self, "Invalid RR", "RR must be a positive value in milliseconds.")
            return
        super().accept()
