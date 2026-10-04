"""Case-group and dual-VENC selection for H5 inputs."""

from PySide6 import QtCore, QtWidgets


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


class DualVencSelectDialog(QtWidgets.QDialog):
    """Choose which source a legacy dual-VENC H5 should expose as flow."""

    def __init__(self, lv_venc=None, hv_venc=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Dual-VENC Source")
        self.resize(520, 280)
        self._buttons = {}

        layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "This H5 contains low- and high-VENC channels. Choose the velocity source for this load."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        group = QtWidgets.QGroupBox("Flow source")
        group_layout = QtWidgets.QVBoxLayout(group)
        options = (
            ("lv", "LV (low VENC)", "Use the low-VENC velocity and VENC values."),
            ("hv", "HV (high VENC)", "Use the high-VENC velocity and VENC values."),
            ("dv", "DV (dual-VENC reconstruction)", "Combine LV and HV with the dual-VENC alias reconstruction."),
        )
        for mode, title, description in options:
            button = QtWidgets.QRadioButton(title)
            button.setToolTip(description)
            self._buttons[mode] = button
            group_layout.addWidget(button)
            detail = QtWidgets.QLabel(description)
            detail.setIndent(24)
            detail.setStyleSheet("color: palette(mid);")
            group_layout.addWidget(detail)
        self._buttons["dv"].setChecked(True)
        layout.addWidget(group)

        venc_text = []
        if lv_venc is not None:
            venc_text.append("LV VENC: " + ", ".join(f"{float(x):.6g}" for x in list(lv_venc)[:3]))
        if hv_venc is not None:
            venc_text.append("HV VENC: " + ", ".join(f"{float(x):.6g}" for x in list(hv_venc)[:3]))
        if venc_text:
            label = QtWidgets.QLabel(" | ".join(venc_text))
            label.setWordWrap(True)
            layout.addWidget(label)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_mode(self):
        for mode, button in self._buttons.items():
            if button.isChecked():
                return mode
        return "dv"
