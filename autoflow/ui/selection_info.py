"""Compact, selectable plane/path information in the Browser workspace."""

from PySide6 import QtCore, QtGui, QtWidgets


def consistency_rows(workspace, path_index, metric=None):
    """Reuse the existing QC scores; unavailable scores remain undefined."""
    import math
    qc = getattr(workspace.derived, "plane_qc", {}) or {}
    metric = metric or {}
    path_scores = qc.get("path_ic", {}) or {}
    score = metric.get("path_ic", path_scores.get(str(path_index), path_scores.get(path_index)))
    def display(value):
        return f"{float(value):.3f}" if value is not None and math.isfinite(float(value)) else "undefined"
    rows = [("Path IC", display(score), "")]
    related = list(metric.get("fork_ic", []) or [])
    if not related:
        for item in qc.get("forks", []) or []:
            members = list(item.get("left", []) or []) + list(item.get("right", []) or [])
            if path_index in [int(value) for value in members]:
                related.append(item)
    for item in related:
        rows.append((f"Fork {int(item.get('fork_id', -1))} IC", display(item.get("ic")), ""))
    return rows


class SelectionInfoPanel(QtWidgets.QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("selectionInfo")
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        self._plain_text = ""
        self._copy_action = QtGui.QAction("Copy details", self)
        self._copy_action.triggered.connect(lambda: QtWidgets.QApplication.clipboard().setText(self._plain_text))
        self._row_labels = []
        self._row_keys = None
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 7, 6, 8)
        layout.setSpacing(5)
        self.title = self._label("title")
        self.subtitle = self._label("secondary")
        self.header_layout = QtWidgets.QHBoxLayout()
        self.header_layout.setContentsMargins(0, 0, 0, 0)
        self.header_layout.addWidget(self.title, 1)
        layout.addLayout(self.header_layout)
        layout.addWidget(self.subtitle)
        self.rows = QtWidgets.QWidget(self)
        self.grid = QtWidgets.QGridLayout(self.rows)
        self.grid.setContentsMargins(0, 6, 0, 3)
        self.grid.setHorizontalSpacing(9)
        self.grid.setVerticalSpacing(8)
        self.grid.setColumnStretch(1, 1)
        layout.addWidget(self.rows)
        self.note = self._label("secondary")
        layout.addWidget(self.note)
        self.addAction(self._copy_action)
        self.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)

    def _label(self, role):
        label = QtWidgets.QLabel(self)
        label.setProperty("infoRole", role)
        label.setTextFormat(QtCore.Qt.PlainText)
        label.setWordWrap(True)
        label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse | QtCore.Qt.TextSelectableByKeyboard)
        label.setMinimumWidth(0)
        label.addAction(self._copy_action)
        label.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)
        return label

    def setPlainText(self, text):
        self._plain_text = str(text)
        self.title.hide()
        self.subtitle.hide()
        self.rows.hide()
        self.note.setText(self._plain_text)
        self.note.setVisible(bool(text))

    def toPlainText(self):
        return self._plain_text

    def set_summary(self, title, subtitle, rows, notes, *, plain_text):
        self._plain_text = plain_text
        self.title.setText(title)
        self.title.show()
        self.subtitle.setText(subtitle)
        self.subtitle.setVisible(bool(subtitle))
        keys = tuple((label, bool(unit)) for label, _value, unit in rows)
        if keys != self._row_keys:
            while self.grid.count():
                item = self.grid.takeAt(0)
                if item.widget() is not None:
                    item.widget().deleteLater()
            self._row_labels.clear()
            for index, (name, _value, unit) in enumerate(rows):
                name_label = self._label("secondary")
                value_label = self._label("value")
                unit_label = self._label("unit") if unit else None
                name_label.setText(name)
                name_label.setWordWrap(False)
                value_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                if unit_label is not None:
                    unit_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
                    unit_label.setWordWrap(False)
                self.grid.addWidget(name_label, index, 0, alignment=QtCore.Qt.AlignTop)
                self.grid.addWidget(value_label, index, 1, 1, 1 if unit else 2)
                if unit:
                    self.grid.addWidget(unit_label, index, 2)
                self._row_labels.append((value_label, unit_label))
            self._row_keys = keys
        for (value_label, unit_label), (_name, value, unit) in zip(self._row_labels, rows):
            value_label.setText(str(value))
            if unit_label is not None:
                unit_label.setText(unit)
        self.rows.setVisible(bool(rows))
        self.note.setText(notes or "")
        self.note.setVisible(bool(notes))
