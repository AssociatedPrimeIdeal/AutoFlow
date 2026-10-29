"""Current measurement and compact statistics above the analysis plot."""

from PySide6 import QtCore, QtGui, QtWidgets


class MeasurementSummary(QtWidgets.QFrame):
    parametersRequested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("measurementSummary")
        self._text = ""
        self._base_text = ""
        self._pinned_keys = None
        self.pinned_labels = []
        self.parameters_action = QtGui.QAction("Parameters…", self)
        self.parameters_action.triggered.connect(lambda _checked=False: self.parametersRequested.emit())
        self.copy_action = QtGui.QAction("Copy summary", self)
        self.copy_action.triggered.connect(lambda: QtWidgets.QApplication.clipboard().setText(self._text))
        self.addActions([self.parameters_action, self.copy_action])
        self.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(9, 9, 9, 6)
        layout.setSpacing(7)
        self.heading = self._label("heading")
        self.context = self._label("secondary")
        layout.addWidget(self.heading)
        layout.addWidget(self.context)
        self.reading = QtWidgets.QWidget(self)
        reading_layout = QtWidgets.QHBoxLayout(self.reading)
        reading_layout.setContentsMargins(0, 2, 0, 2)
        reading_layout.setSpacing(7)
        self.value = self._label("reading")
        self.unit = self._label("secondary")
        reading_layout.addWidget(self.value)
        reading_layout.addWidget(self.unit, alignment=QtCore.Qt.AlignBottom)
        reading_layout.addStretch(1)
        layout.addWidget(self.reading)
        self.statistics = QtWidgets.QWidget(self)
        stats_layout = QtWidgets.QGridLayout(self.statistics)
        stats_layout.setContentsMargins(0, 6, 0, 3)
        stats_layout.setHorizontalSpacing(18)
        stats_layout.setVerticalSpacing(4)
        self.stat_labels = []
        for column in range(3):
            name = self._label("secondary")
            value = self._label("statValue")
            stats_layout.addWidget(name, 0, column)
            stats_layout.addWidget(value, 1, column)
            stats_layout.setColumnStretch(column, 1)
            self.stat_labels.append((name, value))
        layout.addWidget(self.statistics)
        self.pinned = QtWidgets.QWidget(self)
        pinned_layout = QtWidgets.QVBoxLayout(self.pinned)
        pinned_layout.setContentsMargins(0, 10, 0, 3)
        pinned_layout.setSpacing(9)
        self.pinned_heading = self._label("secondary")
        pinned_layout.addWidget(self.pinned_heading)
        self.pinned_grid = QtWidgets.QGridLayout()
        self.pinned_grid.setContentsMargins(0, 0, 0, 0)
        self.pinned_grid.setHorizontalSpacing(18)
        self.pinned_grid.setVerticalSpacing(12)
        self.pinned_grid.setColumnStretch(0, 1)
        self.pinned_grid.setColumnStretch(1, 1)
        pinned_layout.addLayout(self.pinned_grid)
        layout.addWidget(self.pinned)
        self.note = self._label("secondary")
        layout.addWidget(self.note)
        self.setText("No analysis available.")

    def _label(self, role):
        label = QtWidgets.QLabel(self)
        label.setProperty("measurementRole", role)
        label.setTextFormat(QtCore.Qt.PlainText)
        label.setWordWrap(True)
        label.setMinimumWidth(0)
        label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse | QtCore.Qt.TextSelectableByKeyboard)
        label.addActions([self.parameters_action, self.copy_action])
        label.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)
        return label

    def setText(self, text):
        self._text = self._base_text = str(text)
        for widget in (self.heading, self.context, self.reading, self.statistics, self.pinned):
            widget.hide()
        self.note.setText(self._text)
        self.note.setVisible(bool(text))

    def text(self):
        return self._text

    def set_measurement(self, heading, context, value, unit, statistics, *, notes="", plain_text=""):
        self._text = self._base_text = plain_text or "\n".join([heading, context, f"{value} {unit}", *[f"{name}: {number}" for name, number in statistics], notes])
        self.heading.setText(heading)
        self.heading.setVisible(bool(heading))
        self.context.setText(context)
        self.context.setVisible(bool(context))
        self.value.setText(value)
        self.unit.setText(unit)
        self.unit.setVisible(bool(unit))
        self.reading.show()
        for index, (name_label, value_label) in enumerate(self.stat_labels):
            populated = index < len(statistics)
            if populated:
                name, number = statistics[index]
                name_label.setText(name)
                value_label.setText(number)
            name_label.setVisible(populated)
            value_label.setVisible(populated)
        self.statistics.setVisible(bool(statistics))
        self.note.setText(notes)
        self.note.setVisible(bool(notes))

    def set_pinned_metrics(self, heading, rows):
        keys = tuple((name, unit) for name, _value, unit in rows)
        if keys != self._pinned_keys:
            while self.pinned_grid.count():
                item = self.pinned_grid.takeAt(0)
                if item.widget() is not None:
                    item.widget().hide()
                    item.widget().deleteLater()
            self.pinned_labels.clear()
            for index, (name, _value, unit) in enumerate(rows):
                cell = QtWidgets.QWidget(self)
                cell_layout = QtWidgets.QVBoxLayout(cell)
                cell_layout.setContentsMargins(0, 0, 0, 0)
                cell_layout.setSpacing(4)
                name_label = self._label("secondary")
                name_label.setText(name)
                cell_layout.addWidget(name_label)
                reading = QtWidgets.QHBoxLayout()
                reading.setContentsMargins(0, 0, 0, 0)
                reading.setSpacing(5)
                value_label = self._label("statValue")
                unit_label = self._label("secondary")
                unit_label.setText(unit)
                reading.addWidget(value_label)
                reading.addWidget(unit_label)
                reading.addStretch(1)
                cell_layout.addLayout(reading)
                self.pinned_grid.addWidget(cell, index // 2, index % 2)
                self.pinned_labels.append(value_label)
            self._pinned_keys = keys
        for label, (_name, value, _unit) in zip(self.pinned_labels, rows):
            label.setText(value)
        self.pinned_heading.setText(heading)
        self.pinned.setVisible(bool(rows))
        self.pinned_grid.activate()
        self.pinned.layout().activate()
        self._text = self._base_text
        if rows:
            self._text += "\n" + heading + "\n" + "\n".join(f"{name}: {value} {unit}" for name, value, unit in rows)
