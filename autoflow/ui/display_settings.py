"""Shared, serializable controls for slice display."""

import copy
import json

from PySide6 import QtCore, QtGui, QtWidgets
from .parameter_help import bind_help, compact_label, parameter_help

SHARED_FIELDS = {
    ("contours", "smoothing"), ("contours", "minimum_points"), ("contours", "snap_distance"),
    ("view", "linked_zoom"), ("view", "order"), ("guides", "values"),
    ("colorbars", "background"), ("colorbars", "width"), ("colorbars", "font_size"),
}

def display_defaults():
    stream = dict(enabled=False, color_mode="speed", quantity="total", cmap="jet",
                  color="#00e5ff", range_mode="auto", low=0.0, high=100.0,
                  opacity=0.85, line_width=1.5, arrows=True, arrow_size=1.0,
                  arrow_spacing=8.0, min_length=0.0)
    return {
        "images": {}, "views": {},
        "image": dict(cmap="auto", range_mode="slice", low=0.0, high=100.0, interpolation="nearest"),
        "segmentation": dict(visible=True, mode="fill", labels="", colors={}, line_width=1.5),
        "contours": dict(visible=True, roi_color="#00f5d4", manual_color="#ffd43b",
                         line_width=1.8, opacity=1.0, dashed=False),
        "noise": dict(color="#ff0000"),
        "planar": dict(stream, density=1.0, terminal_speed=1.0, max_steps=200, max_length=0.0),
        "projection": dict(stream, color="#ffb347", fraction=1.0, slab=3.0),
        "guides": dict(crosshairs=True, orientation=True, values=True, plane_intersection=True,
                       crosshair_color="#ff5f5f", plane_color="#ffd43b", line_width=1.0),
        "view": dict(background="#050809", zoom=0.5, linked_zoom=False,
                     order="axial,coronal,sagittal"),
        "colorbars": dict(background=True, velocity=True, width=18, font_size=10),
    }


def merge_display(base, overrides):
    result = copy.deepcopy(base)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_display(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def saved_display_defaults():
    try:
        raw = QtCore.QSettings("AutoFlow", "AutoFlow").value("viewer/display_defaults", "{}")
        value = json.loads(str(raw))
        return merge_display(display_defaults(), value if isinstance(value, dict) else {})
    except (ValueError, TypeError):
        return display_defaults()


def view_display(settings, name=None, content=None):
    result = merge_display(display_defaults(), settings)
    if content is not None:
        result["image"] = merge_display(result["image"], result["images"].get(str(content), {}))
    if name:
        result = merge_display(result, result["views"].get(name, {}))
        if content is not None:
            result["image"] = merge_display(result["image"], result["images"].get(str(content), {}))
    return result


class DisplaySettingsPanel(QtWidgets.QWidget):
    settingsChanged = QtCore.Signal(dict)

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self._state = merge_display(display_defaults(), settings)
        self._content = 3
        self._controls = {}
        self._flow_rows = {}
        self.setMinimumWidth(310)
        layout = QtWidgets.QVBoxLayout(self)
        self.target = QtWidgets.QComboBox()
        for label, key in (("All slice views", ""), ("Axial / U × V", "axial"),
                           ("Coronal / V × N", "coronal"), ("Sagittal / U × N", "sagittal")):
            self.target.addItem(label, key)
        self.target.currentIndexChanged.connect(self._sync)
        layout.addWidget(self.target)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        layout.addWidget(self.tabs, 1)
        self._forms = {}
        for title in ("Image", "Overlays", "Flow", "View"):
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            body = QtWidgets.QWidget()
            form = QtWidgets.QFormLayout(body)
            form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
            scroll.setWidget(body)
            self.tabs.addTab(scroll, title)
            self._forms[title] = form
        self._build_fields()
        self.validation = QtWidgets.QLabel()
        self.validation.setWordWrap(True)
        layout.addWidget(self.validation)
        buttons = QtWidgets.QHBoxLayout()
        reset = QtWidgets.QPushButton("Restore defaults")
        save = QtWidgets.QPushButton("Save defaults")
        reset.clicked.connect(self._reset)
        save.clicked.connect(self._save)
        buttons.addWidget(reset)
        buttons.addWidget(save)
        layout.addLayout(buttons)
        self._sync()

    def _heading(self, tab, text):
        label = QtWidgets.QLabel(text)
        font = label.font()
        font.setBold(True)
        label.setFont(font)
        self._forms[tab].addRow(label)

    def _field(self, tab, section, key, label, kind, options=None, limits=None, suffix=""):
        if kind == "bool":
            widget = QtWidgets.QCheckBox()
            widget.toggled.connect(lambda _value: self._changed(section, key))
        elif kind == "choice":
            widget = QtWidgets.QComboBox()
            for text, value in options:
                widget.addItem(text, value)
            widget.currentIndexChanged.connect(lambda _value: self._changed(section, key))
        elif kind in {"float", "int"}:
            widget = QtWidgets.QDoubleSpinBox() if kind == "float" else QtWidgets.QSpinBox()
            widget.setRange(*limits)
            if kind == "float":
                widget.setDecimals(3)
                widget.setSingleStep(0.1)
            widget.setSuffix(suffix)
            widget.setKeyboardTracking(False)
            widget.valueChanged.connect(lambda _value: self._changed(section, key))
        elif kind == "color":
            widget = QtWidgets.QPushButton()
            widget.clicked.connect(lambda: self._pick_color(section, key))
        else:
            widget = QtWidgets.QLineEdit()
            widget.editingFinished.connect(lambda: self._changed(section, key))
        self._controls[(section, key)] = (widget, kind)
        text = parameter_help(section+"."+key, label)
        if section == "image":
            text += "\n\nImage styles are remembered separately for each Content source."
        if (section, key) in SHARED_FIELDS:
            text += "\n\nShared by all three slice views."
        self._forms[tab].addRow(compact_label(label), widget)
        bind_help(self._forms[tab], widget, text)
        if tab == "Flow":
            self._flow_rows[(section, key)] = widget
        return widget

    def _build_fields(self):
        cmap = [(name, name) for name in ("jet", "turbo", "viridis", "plasma", "gray", "inferno", "RdBu_r")]
        f = self._field
        f("Image", "image", "cmap", "Colormap", "choice", [("Content default", "auto")] + cmap)
        f("Image", "image", "range_mode", "Range", "choice", [("Auto per slice", "slice"), ("Auto current phase", "frame"), ("Lock across phases", "all"), ("Manual", "manual")])
        f("Image", "image", "low", "Lower limit", "float", limits=(-1e9, 1e9))
        f("Image", "image", "high", "Upper limit", "float", limits=(-1e9, 1e9))
        f("Image", "image", "interpolation", "Interpolation", "choice", [("Nearest", "nearest"), ("Linear", "linear")])
        self._heading("Overlays", "Segmentation")
        f("Overlays", "segmentation", "visible", "Visible", "bool")
        f("Overlays", "segmentation", "opacity", "Opacity", "float", limits=(0, 1))
        f("Overlays", "segmentation", "mode", "Display", "choice", [("Fill", "fill"), ("Boundary", "outline"), ("Fill + boundary", "both")])
        f("Overlays", "segmentation", "labels", "Labels (empty = all)", "text")
        f("Overlays", "segmentation", "line_width", "Boundary width", "float", limits=(0.25, 8), suffix=" px")
        self.label_id = QtWidgets.QSpinBox()
        self.label_id.setRange(1, 65535)
        label_color = QtWidgets.QPushButton("Choose label color…")
        label_color.clicked.connect(self._pick_label_color)
        self._forms["Overlays"].addRow("Label ID", self.label_id)
        self._forms["Overlays"].addRow(label_color)
        self._heading("Overlays", "Contours / ROI")
        for key, label, kind in (("visible", "Visible", "bool"), ("roi_color", "Metric ROI", "color"), ("manual_color", "Manual ROI", "color"), ("dashed", "Dashed", "bool")):
            f("Overlays", "contours", key, label, kind)
        f("Overlays", "contours", "line_width", "Line width", "float", limits=(0.25, 8), suffix=" px")
        f("Overlays", "contours", "opacity", "Opacity", "float", limits=(0, 1))
        f("Overlays", "contours", "smoothing", "Display smoothing", "float", limits=(0, 5), suffix=" px")
        f("Overlays", "contours", "minimum_points", "Minimum contour points", "int", limits=(0, 100))
        f("Overlays", "contours", "snap_distance", "Endpoint snap (0 = auto)", "float", limits=(0, 50), suffix=" mm")
        self._heading("Overlays", "Noise mask")
        f("Overlays", "noise", "visible", "Visible", "bool")
        f("Overlays", "noise", "color", "Color", "color")
        f("Overlays", "noise", "opacity", "Opacity", "float", limits=(0, 1))
        self.flow_family = QtWidgets.QComboBox()
        self.flow_family.addItem("2D streamlines", "planar")
        self.flow_family.addItem("3D projection", "projection")
        self.flow_family.currentIndexChanged.connect(self._sync_flow_rows)
        self._forms["Flow"].addRow("Overlay", self.flow_family)
        self.flow_detail_tabs = QtWidgets.QTabBar()
        self.flow_detail_tabs.addTab("Basic")
        self.flow_detail_tabs.addTab("Advanced")
        self.flow_detail_tabs.currentChanged.connect(self._sync_flow_rows)
        self._forms["Flow"].addRow(self.flow_detail_tabs)
        for section, title in (("planar", "2D streamlines"), ("projection", "3D projection")):
            f("Flow", section, "enabled", "Visible", "bool")
            f("Flow", section, "color_mode", "Color", "choice", [("By speed", "speed"), ("Fixed color", "fixed")])
            f("Flow", section, "quantity", "Speed quantity", "choice", [("Total speed (cm/s)", "total"), ("In-plane speed (cm/s)", "in_plane")])
            f("Flow", section, "cmap", "Colormap", "choice", cmap)
            f("Flow", section, "color", "Fixed color", "color")
            f("Flow", section, "range_mode", "Range", "choice", [("Auto across phases", "auto"), ("Manual", "manual")])
            f("Flow", section, "low", "Lower limit", "float", limits=(0, 1e6), suffix=" cm/s")
            f("Flow", section, "high", "Upper limit", "float", limits=(0, 1e6), suffix=" cm/s")
            f("Flow", section, "opacity", "Opacity", "float", limits=(0, 1))
            f("Flow", section, "line_width", "Line width", "float", limits=(0.25, 8), suffix=" px")
            f("Flow", section, "arrows", "Direction arrows", "bool")
            f("Flow", section, "arrow_size", "Arrow size", "float", limits=(0.1, 10), suffix=" mm")
            f("Flow", section, "arrow_spacing", "Arrow spacing", "float", limits=(1, 100), suffix=" mm")
            f("Flow", section, "min_length", "Minimum visible length", "float", limits=(0, 100), suffix=" mm")
        f("Flow", "planar", "density", "2D seed density", "float", limits=(0.25, 3))
        f("Flow", "planar", "terminal_speed", "2D minimum speed", "float", limits=(0, 1e4), suffix=" cm/s")
        f("Flow", "planar", "max_steps", "2D steps / direction", "int", limits=(1, 800))
        f("Flow", "planar", "max_length", "2D maximum length (0 = auto)", "float", limits=(0, 1000), suffix=" mm")
        f("Flow", "projection", "fraction", "3D display fraction", "float", limits=(0.01, 1))
        f("Flow", "projection", "slab", "Full slab thickness", "float", limits=(0.1, 100), suffix=" mm")
        self._heading("View", "View and guides")
        f("View", "view", "background", "Background", "color")
        f("View", "view", "zoom", "Initial FOV fraction", "float", limits=(0.1, 1))
        f("View", "view", "linked_zoom", "Link zoom / pan", "bool")
        f("View", "view", "order", "Stack order", "choice", [("Axial, Coronal, Sagittal", "axial,coronal,sagittal"), ("Coronal, Sagittal, Axial", "coronal,sagittal,axial"), ("Sagittal, Axial, Coronal", "sagittal,axial,coronal")])
        for key, label in (("crosshairs", "Crosshair"), ("orientation", "Orientation labels"), ("values", "Value / metric labels"), ("plane_intersection", "Plane intersection")):
            f("View", "guides", key, label, "bool")
        f("View", "guides", "crosshair_color", "Crosshair color", "color")
        f("View", "guides", "plane_color", "Plane intersection color", "color")
        f("View", "guides", "line_width", "Guide width", "float", limits=(0.25, 8), suffix=" px")
        self._heading("View", "Colorbars")
        f("View", "colorbars", "background", "Background colorbar", "bool")
        f("View", "colorbars", "velocity", "Flow colorbars", "bool")
        f("View", "colorbars", "width", "Bar width", "int", limits=(8, 40), suffix=" px")
        f("View", "colorbars", "font_size", "Tick font", "int", limits=(7, 20), suffix=" pt")
    def set_state(self, settings, content=3, *, has_magnitude=False, has_pcmra=False, has_noise=None, has_flow=None):
        self._state = merge_display(display_defaults(), settings)
        self._content = content
        for section in ("planar", "projection"):
            if has_flow is not None:
                self._controls[(section, "enabled")][0].setEnabled(has_flow)
        if has_noise is not None:
            self._controls[("noise", "visible")][0].setEnabled(has_noise)
        self._sync()

    def _sync(self, *_):
        state = view_display(self._state, self.target.currentData(), self._content)
        for (section, key), (widget, kind) in self._controls.items():
            value = self._state[section].get(key) if (section, key) in SHARED_FIELDS else state[section].get(key)
            if value is None:
                value = {"opacity": 0.35, "smoothing": 0.8, "minimum_points": 8, "snap_distance": 0.0}.get(key, False)
            widget.blockSignals(True)
            if kind == "bool": widget.setChecked(bool(value))
            elif kind == "choice": widget.setCurrentIndex(max(0, widget.findData(value)))
            elif kind in {"float", "int"}: widget.setValue(value)
            elif kind == "color":
                widget.setText(str(value))
                widget.setStyleSheet("border-left: 12px solid " + str(value) + ";")
            else: widget.setText(str(value))
            widget.blockSignals(False)
        for section in ("image", "planar", "projection"):
            cfg = state[section]
            speed = section not in {"planar", "projection"} or cfg["color_mode"] == "speed"
            for key in ("low", "high"):
                self._controls[(section, key)][0].setEnabled(speed and cfg["range_mode"] == "manual")
            if section in {"planar", "projection"}:
                self._controls[(section, "color")][0].setEnabled(not speed)
                for key in ("cmap", "quantity", "range_mode"):
                    self._controls[(section, key)][0].setEnabled(speed)
                for key in ("arrow_size", "arrow_spacing"):
                    self._controls[(section, key)][0].setEnabled(cfg["arrows"])
        self._sync_flow_rows()

    def _sync_flow_rows(self, *_):
        if not hasattr(self, "flow_family"):
            return
        family = self.flow_family.currentData()
        advanced = self.flow_detail_tabs.currentIndex() == 1
        details = {"arrow_size", "arrow_spacing", "min_length", "terminal_speed", "max_steps", "max_length"}
        cfg = view_display(self._state, self.target.currentData(), self._content)[family]
        for (section, key), widget in self._flow_rows.items():
            visible = section == family and ((key in details) == advanced)
            if key == "color":
                visible &= cfg["color_mode"] == "fixed"
            elif key in {"quantity", "cmap", "range_mode", "low", "high"}:
                visible &= cfg["color_mode"] == "speed"
                if key in {"low", "high"}:
                    visible &= cfg["range_mode"] == "manual"
            widget.setVisible(visible)
            self._forms["Flow"].labelForField(widget).setVisible(visible)

    def _write(self, section, key, value):
        target = self.target.currentData()
        state = self._state if not target or (section, key) in SHARED_FIELDS else self._state["views"].setdefault(target, {})
        if section == "image":
            state.setdefault("images", {}).setdefault(str(self._content), {})[key] = value
        else:
            state.setdefault(section, {})[key] = value
        effective = view_display(self._state, target, self._content)
        for family in ("image", "planar", "projection"):
            cfg = effective[family]
            if cfg["range_mode"] == "manual" and cfg["high"] <= cfg["low"] and (family not in {"planar", "projection"} or cfg["color_mode"] == "speed"):
                self.validation.setText("Upper limit must exceed lower limit.")
                return
        labels = effective["segmentation"]["labels"]
        if labels and any(not token.strip().isdigit() for token in labels.split(",")):
            self.validation.setText("Enter comma-separated positive label IDs.")
            return
        self.validation.clear()
        self._sync()
        self.settingsChanged.emit(copy.deepcopy(self._state))

    def _changed(self, section, key):
        widget, kind = self._controls[(section, key)]
        value = widget.isChecked() if kind == "bool" else widget.currentData() if kind == "choice" else widget.value() if kind in {"float", "int"} else widget.text()
        self._write(section, key, value)

    def _pick_color(self, section, key):
        widget = self._controls[(section, key)][0]
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(widget.text()), self)
        if color.isValid():
            self._write(section, key, color.name())
            self._sync()

    def _pick_label_color(self):
        state = view_display(self._state, self.target.currentData(), self._content)
        colors = dict(state["segmentation"]["colors"])
        key = str(self.label_id.value())
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(colors.get(key, "#ff6b6b")), self)
        if color.isValid():
            colors[key] = color.name()
            self._write("segmentation", "colors", colors)

    def _reset(self):
        self._state = display_defaults()
        self._sync()
        self.settingsChanged.emit(copy.deepcopy(self._state))

    def _save(self):
        state = copy.deepcopy(self._state)
        # Manual patient-specific windows belong to the saved workspace.
        for scope in [state] + list(state["views"].values()):
            for image in scope.get("images", {}).values():
                if image.get("range_mode") == "manual":
                    image["range_mode"] = "slice"
                image.pop("low", None)
                image.pop("high", None)
            for family in ("planar", "projection"):
                if family in scope:
                    scope[family]["enabled"] = False
                if scope.get(family, {}).get("range_mode") == "manual":
                    scope[family]["range_mode"] = "auto"
        state["planar"]["enabled"] = False
        state["projection"]["enabled"] = False
        state.setdefault("noise", {})["visible"] = False
        QtCore.QSettings("AutoFlow", "AutoFlow").setValue("viewer/display_defaults", json.dumps(state))
        self.validation.setText("Default styles saved; manual image windows remain case-specific.")
