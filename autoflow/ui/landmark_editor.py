"""Structured anatomical placement controls shared by the plane settings table."""

import copy

from PySide6 import QtCore, QtWidgets

from ..plane_rules import normalize_label_plane_rules


ANCHORS = (
    ("Observed label connections (all arms)", "label_connections"),
    ("Longest continuous label interval", "label_longest"),
    ("Label-local path level", "label_level"), ("Highest label-local path level", "label_terminal"),
    ("MPA / LPA / RPA junction", "pulmonary_split"), ("SMV / SV / PV confluence", "portal_confluence"),
    ("First PV split after confluence", "pv_first_split"), ("Common PV route", "portal_trunk"),
    ("LVA / RVA / BA confluence", "vertebral_confluence"), ("BA / LPCA / RPCA junction", "pca_split"),
    ("Left ICA / MCA / ACA junction", "left_carotid_split"), ("Right ICA / MCA / ACA junction", "right_carotid_split"),
    ("SSS / StrS / LTS / RTS confluence", "sinus_confluence"), ("Aortic side-branch origin", "aortic_origin"),
    ("ARCH route from AAO to DAO", "arch_route"),
)
ANCHOR_TITLES = {value: title for title, value in ANCHORS}
PATH_CHOICES = (("Require one path", "unique"), ("Longest label interval at this level", "longest"),
                ("All paths at this level", "all"))


def placement_summary(rule):
    if rule.get("plane_mode") != "landmark":
        return "Legacy layout: " + str(rule.get("plane_mode", "fixed_step"))
    rows = []
    for spec in rule.get("placements", []):
        title = ANCHOR_TITLES.get(spec["anchor"], spec["anchor"])
        if spec["anchor"] == "label_level":
            title += f" L{spec.get('level', 1)}"
        if spec["anchor"] in {"label_level", "label_terminal"}:
            title += "; " + dict((value, label) for label, value in PATH_CHOICES)[spec.get("path_selection", "unique")]
        position = ("inside label +" + f"{spec.get('offset_mm', 5):g} mm" if spec["position"] == "near" else
                    ", ".join(f"{100 * float(x):.6g}%" for x in spec["fractions"]))
        rows.append(title + " → " + position)
        if spec.get("when_connected_to"):
            rows[-1] += "; when connected to " + ", ".join(spec["when_connected_to"]) + f"; otherwise L{spec['otherwise_level']}"
        if spec.get("fallback"):
            rows[-1] += "; no connection: " + placement_summary({"plane_mode":"landmark", "plane_count":1,
                                                               "placements":[spec["fallback"]]})
    return ("Disabled; " if rule.get("plane_count") == 0 else "") + " | ".join(rows)


class LandmarkPlanDialog(QtWidgets.QDialog):
    def __init__(self, name, rule, parent=None, *, allow_fallback=True):
        super().__init__(parent)
        self.name, self.original = str(name), copy.deepcopy(rule)
        self.rule = None
        self.allow_fallback = allow_fallback
        self.setWindowTitle(self.name + " — landmark placement plan")
        self.resize(1320, 740)
        layout = QtWidgets.QVBoxLayout(self)
        self.enabled = QtWidgets.QCheckBox("Enable measurements for " + self.name)
        self.enabled.setObjectName("landmark_enabled")
        self.enabled.setChecked(rule.get("plane_count", 0) > 0)
        layout.addWidget(self.enabled)
        explanation = QtWidgets.QLabel(
            "Each row defines a connection or path property and positions inside the target label.\n"
            "Observed connections retain every real arm. Fallback applies only when no connection exists; unsupported sections are skipped.")
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        self.table = QtWidgets.QTableWidget(0, 8)
        self.table.setObjectName("landmark_placements")
        self.table.setHorizontalHeaderLabels(["Anchor", "Label level", "Paths at level", "Position", "Offset (mm)", "Positions (%)", "No connection", "Level condition"])
        self.table.setColumnHidden(6, not allow_fallback)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(5, QtWidgets.QHeaderView.Stretch)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        layout.addWidget(self.table, 1)
        actions = QtWidgets.QHBoxLayout()
        add = QtWidgets.QPushButton("Add placement")
        add.setObjectName("add_landmark_placement")
        remove = QtWidgets.QPushButton("Remove selected")
        remove.setObjectName("remove_landmark_placement")
        add.clicked.connect(lambda: self.add_placement({"anchor": "label_level", "level": 1,
            "position": "fractions", "fractions": [.5]}))
        remove.clicked.connect(self.remove_selected)
        actions.addWidget(add)
        actions.addWidget(remove)
        actions.addStretch()
        self.count_label = QtWidgets.QLabel()
        actions.addWidget(self.count_label)
        layout.addLayout(actions)
        self.preview = QtWidgets.QLabel()
        self.preview.setWordWrap(True)
        self.preview.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        layout.addWidget(self.preview)
        advanced = QtWidgets.QGroupBox("Filters and smoothing (inherit shared settings unless overridden)")
        form = QtWidgets.QFormLayout(advanced)
        self.overrides = {}
        for key, title in (("segmentation_filter", "Branch support"), ("fov_boundary_filter", "Exclude FOV-truncated sections")):
            combo = QtWidgets.QComboBox()
            for text, data in (("Inherit", None), ("Enabled", True), ("Disabled", False)):
                combo.addItem(text, data)
            combo.setCurrentIndex(combo.findData(rule.get(key)))
            combo.setObjectName("landmark_" + key)
            self.overrides[key] = combo
            form.addRow(title, combo)
        for key, title, minimum, maximum in (
            ("fov_boundary_max_fraction", "Maximum FOV contact fraction", -1., 1.),
            ("smoothing_window", "Path smoothing window", 0, 100001),
            ("smoothing_polyorder", "Path smoothing order", -1, 100),
            ("inter_time", "Path interpolation factor", 0, 1000),
        ):
            control = QtWidgets.QDoubleSpinBox() if isinstance(minimum, float) else QtWidgets.QSpinBox()
            control.setRange(minimum, maximum)
            control.setSpecialValueText("Inherit")
            if isinstance(control, QtWidgets.QDoubleSpinBox):
                control.setDecimals(4)
                control.setSingleStep(.01)
            control.setValue(rule[key] if rule.get(key) is not None else minimum)
            control.setObjectName("landmark_" + key)
            self.overrides[key] = control
            form.addRow(title, control)
        layout.addWidget(advanced)
        self.error = QtWidgets.QLabel()
        self.error.setWordWrap(True)
        self.error.setStyleSheet("color: #b42318")
        layout.addWidget(self.error)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        for spec in rule.get("placements", []):
            self.add_placement(spec)
        self.enabled.toggled.connect(self.refresh)
        self.refresh()

    def add_placement(self, spec):
        row = self.table.rowCount()
        self.table.insertRow(row)
        anchor = QtWidgets.QComboBox()
        for title, value in ANCHORS:
            if not self.allow_fallback and value == "label_connections":
                continue
            anchor.addItem(title, value)
        anchor.setCurrentIndex(anchor.findData(spec["anchor"]))
        anchor.setMinimumWidth(220)
        level = QtWidgets.QSpinBox()
        level.setRange(1, 10000)
        level.setValue(spec.get("level", 1))
        paths = QtWidgets.QComboBox()
        for title, value in PATH_CHOICES:
            paths.addItem(title, value)
        paths.setCurrentIndex(paths.findData(spec.get("path_selection", "unique")))
        position = QtWidgets.QComboBox()
        position.addItem("Offset inside label", "near")
        position.addItem("Fractions of label interval", "fractions")
        position.setCurrentIndex(position.findData(spec["position"]))
        offset = QtWidgets.QDoubleSpinBox()
        offset.setDecimals(3)
        offset.setRange(0, 10000)
        offset.setValue(spec.get("offset_mm", 5.))
        fractions = QtWidgets.QLineEdit(", ".join(f"{100 * float(x):.15g}" for x in spec.get("fractions", [.5])))
        fractions.setToolTip("Comma-separated percentages strictly between 0 and 100, for example 20, 50, 80")
        fractions.setProperty("source_text", fractions.text())
        fractions.setProperty("source_fractions", list(spec.get("fractions", [.5])))
        for column, control in enumerate((anchor, level, paths, position, offset, fractions)):
            self.table.setCellWidget(row, column, control)
            signal = (control.currentIndexChanged if isinstance(control, QtWidgets.QComboBox) else
                      control.textChanged if isinstance(control, QtWidgets.QLineEdit) else control.valueChanged)
            signal.connect(self.refresh)
        fallback = QtWidgets.QPushButton("Edit fallback…" if spec.get("fallback") else "No fallback")
        fallback.setProperty("fallback_spec", copy.deepcopy(spec.get("fallback")))
        fallback.clicked.connect(lambda *_args, widget=fallback: self.edit_fallback(widget))
        self.table.setCellWidget(row, 6, fallback)
        condition = QtWidgets.QPushButton("Edit condition…" if spec.get("when_connected_to") else "Always this level")
        condition.setProperty("connected_labels", list(spec.get("when_connected_to", [])))
        condition.setProperty("otherwise_level", spec.get("otherwise_level", 1))
        condition.clicked.connect(lambda *_args, widget=condition: self.edit_condition(widget))
        self.table.setCellWidget(row, 7, condition)
        self.table.selectRow(row)
        self.refresh()

    def edit_fallback(self, button):
        spec = button.property("fallback_spec") or {"anchor":"label_longest", "position":"fractions", "fractions":[.5]}
        count = len(spec["fractions"]) if spec["position"] == "fractions" else 1
        dialog = LandmarkPlanDialog(self.name + " fallback", {"plane_mode":"landmark", "plane_count":count,
                                    "placements":[spec]}, self, allow_fallback=False)
        try:
            if dialog.exec() != QtWidgets.QDialog.Accepted:
                return
            fallback = dialog.rule["placements"]
            if dialog.rule["plane_count"] and len(fallback) != 1:
                QtWidgets.QMessageBox.warning(self, "Fallback", "Use exactly one fallback placement.")
                return
            value = fallback[0] if dialog.rule["plane_count"] else None
            button.setProperty("fallback_spec", copy.deepcopy(value))
            button.setText("Edit fallback…" if value else "No fallback")
            self.refresh()
        finally:
            dialog.deleteLater()

    def edit_condition(self, button):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Conditional label-local level")
        form = QtWidgets.QFormLayout(dialog)
        note = QtWidgets.QLabel("Use the selected L level when all listed labels actually connect to this label in the same physical component. Leave blank to always use it.")
        note.setWordWrap(True)
        form.addRow(note)
        labels = QtWidgets.QLineEdit(", ".join(button.property("connected_labels") or []))
        other = QtWidgets.QSpinBox(); other.setRange(1,10000); other.setValue(button.property("otherwise_level") or 1)
        form.addRow("Connected labels (comma separated)", labels)
        form.addRow("Otherwise use L", other)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject); form.addRow(buttons)
        try:
            if dialog.exec() == QtWidgets.QDialog.Accepted:
                values = [value.strip() for value in labels.text().replace("，",",").split(",") if value.strip()]
                button.setProperty("connected_labels", values)
                button.setProperty("otherwise_level", other.value())
                button.setText("Edit condition…" if values else "Always this level")
                self.refresh()
        finally:
            dialog.deleteLater()

    def remove_selected(self):
        rows = {index.row() for index in self.table.selectedIndexes()}
        if not rows and self.table.currentRow() >= 0:
            rows = {self.table.currentRow()}
        for row in sorted(rows, reverse=True):
            self.table.removeRow(row)
        self.refresh()

    def placements(self):
        placements = []
        for row in range(self.table.rowCount()):
            anchor, level, paths, position, offset, fractions = [self.table.cellWidget(row, i) for i in range(6)]
            spec = {"anchor": anchor.currentData(), "position": position.currentData()}
            if spec["anchor"] == "label_level":
                spec["level"] = level.value()
            if spec["anchor"] in {"label_level", "label_terminal"}:
                choice = paths.currentData()
                if choice != "unique":
                    spec["path_selection"] = choice
            fallback = self.table.cellWidget(row, 6)
            if self.allow_fallback and spec["anchor"] == "label_connections" and fallback.property("fallback_spec"):
                spec["fallback"] = copy.deepcopy(fallback.property("fallback_spec"))
            condition = self.table.cellWidget(row, 7)
            if spec["anchor"] == "label_level" and condition.property("connected_labels"):
                spec["when_connected_to"] = list(condition.property("connected_labels"))
                spec["otherwise_level"] = int(condition.property("otherwise_level"))
            if spec["position"] == "near":
                spec["offset_mm"] = offset.value()
            else:
                text = fractions.text().strip()
                if text == fractions.property("source_text"):
                    spec["fractions"] = list(fractions.property("source_fractions"))
                else:
                    spec["fractions"] = [float(part.strip().removesuffix("%")) / 100 for part in text.replace("，", ",").split(",")]
            placements.append(spec)
        return placements

    def refresh(self, *_args):
        for row in range(self.table.rowCount()):
            anchor, level, paths, position, offset, fractions = [self.table.cellWidget(row, i) for i in range(6)]
            if fractions is None:
                continue
            level.setEnabled(anchor.currentData() == "label_level")
            paths.setEnabled(anchor.currentData() in {"label_level", "label_terminal"})
            fallback = self.table.cellWidget(row, 6)
            condition = self.table.cellWidget(row, 7)
            if fallback is not None:
                fallback.setEnabled(self.allow_fallback and anchor.currentData() == "label_connections")
            if condition is not None:
                condition.setEnabled(anchor.currentData() == "label_level")
            offset.setEnabled(position.currentData() == "near")
            fractions.setEnabled(position.currentData() == "fractions")
        try:
            placements = self.placements()
            count = sum(len(spec["fractions"]) if spec["position"] == "fractions" else 1 for spec in placements)
            self.count_label.setText(f"Declared positions: {count}")
            self.preview.setText(placement_summary({"plane_mode": "landmark", "plane_count": count if self.enabled.isChecked() else 0,
                                                     "placements": placements}))
        except (TypeError, ValueError):
            self.count_label.setText("Invalid percentages")

    def accept(self):
        try:
            rule = copy.deepcopy(self.original)
            rule.pop("main_path", None)
            rule["plane_mode"] = "landmark"
            rule["placements"] = self.placements()
            rule["plane_count"] = (sum(len(spec["fractions"]) if spec["position"] == "fractions" else 1
                                       for spec in rule["placements"]) if self.enabled.isChecked() else 0)
            for key, control in self.overrides.items():
                rule[key] = (control.currentData() if isinstance(control, QtWidgets.QComboBox) else
                             None if control.value() == control.minimum() else control.value())
            self.rule = normalize_label_plane_rules({self.name: rule})[self.name]
        except (ValueError, TypeError) as exc:
            self.error.setText(str(exc))
            return
        super().accept()
