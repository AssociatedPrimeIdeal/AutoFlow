from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .slice_view import PLANE_SPECS, SliceView, make_colormap, make_label_overlay


@dataclass
class EditCommand:
    indices: np.ndarray
    before: np.ndarray
    after: np.ndarray


class ManualSegmentationWindow(QtWidgets.QMainWindow):
    applied = QtCore.Signal(object, object, object, object)
    cancelled = QtCore.Signal()

    def __init__(
        self,
        *,
        magnitude,
        flow,
        labels,
        resolution,
        cursor,
        current_time,
        label_names,
        label_colors,
        active_label=1,
        brush_radius=3,
        edit_all_timepoints=True,
        parent=None,
    ):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.setWindowTitle("Manual Segmentation")
        self.resize(1500, 920)
        self.setMinimumSize(1040, 680)

        self.magnitude = None if magnitude is None else np.asarray(magnitude, dtype=np.float32)
        self.flow = None if flow is None else np.asarray(flow, dtype=np.float32)
        self.labels = self._normalize_labels(labels)
        values, counts = np.unique(self.labels, return_counts=True)
        self._label_counts = {
            int(value): int(count)
            for value, count in zip(values, counts)
            if int(value) > 0
        }
        self.resolution = np.asarray(resolution, dtype=float).reshape(-1)[:3]
        self.resolution = np.where(self.resolution > 0, self.resolution, 1.0)
        self.cursor = np.asarray(cursor, dtype=int).reshape(-1)[:3].copy()
        self.cursor = np.clip(self.cursor, 0, np.asarray(self.labels.shape[:3]) - 1)
        self.current_time = int(np.clip(current_time, 0, self.labels.shape[3] - 1))
        self.label_names = {str(key): str(value) for key, value in dict(label_names or {}).items()}
        self.label_colors = {str(key): str(value) for key, value in dict(label_colors or {}).items()}
        self.active_label = max(1, int(active_label))
        self.brush_diameter_mm = max(
            1.0,
            2.0 * max(1, int(brush_radius)) * float(np.min(self.resolution)),
        )
        self.tool = "brush"
        self._volume_cache = {}
        self._display_levels = None
        self._undo_stack = []
        self._redo_stack = []
        self._stroke_before = None
        self._dirty = False
        self._allow_close = False
        self._maximized_plane = None
        self._build_ui(bool(edit_all_timepoints))
        self._build_actions()
        self._ensure_active_label_metadata()
        self._refresh_label_list()
        self._refresh_views(reset_levels=True)

    def _normalize_labels(self, labels):
        array = np.asarray(labels, dtype=np.int16)
        if array.ndim == 3:
            time_count = 1
            if self.magnitude is not None and self.magnitude.ndim == 4:
                time_count = int(self.magnitude.shape[3])
            elif self.flow is not None and self.flow.ndim == 5:
                time_count = int(self.flow.shape[3])
            array = np.repeat(array[..., None], max(1, time_count), axis=3)
        if array.ndim != 4:
            raise ValueError(f"manual segmentation labels must be XYZT, got {array.shape}")
        return np.ascontiguousarray(array.copy())

    def _build_ui(self, edit_all_timepoints):
        root = QtWidgets.QWidget()
        root_layout = QtWidgets.QVBoxLayout(root)
        root_layout.setContentsMargins(7, 7, 7, 7)
        root_layout.setSpacing(6)
        self.setCentralWidget(root)

        source_row = QtWidgets.QHBoxLayout()
        source_row.addWidget(QtWidgets.QLabel("Image"))
        self.combo_content = QtWidgets.QComboBox()
        self.combo_content.addItems(
            [
                "PC-MRA",
                "Magnitude",
                "Speed (cm/s)",
                "Flow LR (cm/s)",
                "Flow AP (cm/s)",
                "Flow FH (cm/s)",
            ]
        )
        self.combo_content.currentIndexChanged.connect(self._content_changed)
        source_row.addWidget(self.combo_content, 1)
        source_row.addWidget(QtWidgets.QLabel("Overlay"))
        self.opacity_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(42)
        self.opacity_slider.setMaximumWidth(180)
        self.opacity_slider.valueChanged.connect(lambda _value: self._refresh_views())
        source_row.addWidget(self.opacity_slider)
        self.all_frames_check = QtWidgets.QCheckBox("All time frames")
        self.all_frames_check.setChecked(edit_all_timepoints)
        source_row.addWidget(self.all_frames_check)
        root_layout.addLayout(source_row)

        body_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        body_splitter.setChildrenCollapsible(False)
        root_layout.addWidget(body_splitter, 1)

        label_panel = QtWidgets.QWidget()
        label_panel.setObjectName("segmentationLabelPanel")
        label_panel.setMinimumWidth(190)
        label_panel.setMaximumWidth(260)
        label_layout = QtWidgets.QVBoxLayout(label_panel)
        label_layout.setContentsMargins(6, 6, 6, 6)
        label_layout.addWidget(QtWidgets.QLabel("Labels"))
        self.label_list = QtWidgets.QListWidget()
        self.label_list.currentItemChanged.connect(self._label_selected)
        label_layout.addWidget(self.label_list, 1)
        add_label_button = QtWidgets.QPushButton("Add Label")
        add_label_button.clicked.connect(self._add_label)
        label_layout.addWidget(add_label_button)
        label_layout.addWidget(QtWidgets.QLabel("Active label"))
        self.active_label_spin = QtWidgets.QSpinBox()
        self.active_label_spin.setRange(1, np.iinfo(np.int16).max)
        self.active_label_spin.setValue(self.active_label)
        self.active_label_spin.valueChanged.connect(self._active_label_changed)
        label_layout.addWidget(self.active_label_spin)
        self.label_name_edit = QtWidgets.QLineEdit()
        self.label_name_edit.editingFinished.connect(self._label_name_changed)
        label_layout.addWidget(self.label_name_edit)
        self.label_color_button = QtWidgets.QPushButton("Color")
        self.label_color_button.clicked.connect(self._choose_label_color)
        label_layout.addWidget(self.label_color_button)
        body_splitter.addWidget(label_panel)

        self.views_widget = QtWidgets.QWidget()
        self.views_layout = QtWidgets.QGridLayout(self.views_widget)
        self.views_layout.setContentsMargins(0, 0, 0, 0)
        self.views_layout.setSpacing(5)
        self.slice_views = {name: SliceView(name) for name in PLANE_SPECS}
        self.views_layout.addWidget(self.slice_views["axial"], 0, 0)
        self.views_layout.addWidget(self.slice_views["coronal"], 0, 1)
        self.views_layout.addWidget(self.slice_views["sagittal"], 1, 0)
        self.views_layout.setRowStretch(0, 1)
        self.views_layout.setRowStretch(1, 1)
        self.views_layout.setColumnStretch(0, 1)
        self.views_layout.setColumnStretch(1, 1)

        self.navigation_panel = QtWidgets.QWidget()
        navigation_layout = QtWidgets.QVBoxLayout(self.navigation_panel)
        navigation_layout.setContentsMargins(8, 8, 8, 8)
        navigation_layout.addWidget(QtWidgets.QLabel("Coordinates"))
        self.coordinate_labels = []
        for axis_name in ("LR", "AP", "FH"):
            label = QtWidgets.QLabel(f"{axis_name} 0")
            self.coordinate_labels.append(label)
            navigation_layout.addWidget(label)
        navigation_layout.addSpacing(8)
        navigation_layout.addWidget(QtWidgets.QLabel("Brush diameter"))
        self.brush_diameter_spin = QtWidgets.QDoubleSpinBox()
        self.brush_diameter_spin.setRange(1.0, 80.0)
        self.brush_diameter_spin.setDecimals(1)
        self.brush_diameter_spin.setSingleStep(1.0)
        self.brush_diameter_spin.setSuffix(" mm")
        self.brush_diameter_spin.setValue(self.brush_diameter_mm)
        self.brush_diameter_spin.valueChanged.connect(self._brush_diameter_changed)
        navigation_layout.addWidget(self.brush_diameter_spin)
        navigation_layout.addSpacing(8)
        navigation_layout.addWidget(QtWidgets.QLabel("Display range"))
        self.levels_label = QtWidgets.QLabel("-")
        self.levels_label.setWordWrap(True)
        navigation_layout.addWidget(self.levels_label)
        reset_view_button = QtWidgets.QPushButton("Reset Views")
        reset_view_button.clicked.connect(self._reset_views)
        navigation_layout.addWidget(reset_view_button)
        navigation_layout.addStretch(1)
        self.views_layout.addWidget(self.navigation_panel, 1, 1)
        body_splitter.addWidget(self.views_widget)
        body_splitter.setSizes([220, 1100])

        for view in self.slice_views.values():
            view.set_editable(True)
            view.hoverMoved.connect(self._view_hovered)
            view.cursorRequested.connect(self._cursor_requested)
            view.strokeStarted.connect(self._stroke_started)
            view.strokeMoved.connect(self._stroke_moved)
            view.strokeFinished.connect(self._stroke_finished)
            view.sliceStepRequested.connect(self._step_slice)
            view.brushSizeStepRequested.connect(self._step_brush_size)
            view.windowLevelDragged.connect(self._adjust_window_level)
            view.viewDoubleClicked.connect(self._toggle_maximized_view)

        time_row = QtWidgets.QHBoxLayout()
        self.previous_button = QtWidgets.QPushButton()
        self.previous_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaSkipBackward))
        self.previous_button.setToolTip("Previous frame")
        self.previous_button.clicked.connect(lambda: self._step_time(-1))
        self.next_button = QtWidgets.QPushButton()
        self.next_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaSkipForward))
        self.next_button.setToolTip("Next frame")
        self.next_button.clicked.connect(lambda: self._step_time(1))
        self.time_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.time_slider.setRange(0, self.labels.shape[3] - 1)
        self.time_slider.setValue(self.current_time)
        self.time_slider.valueChanged.connect(self._time_changed)
        self.time_label = QtWidgets.QLabel()
        self.time_label.setMinimumWidth(90)
        time_row.addWidget(self.previous_button)
        time_row.addWidget(self.next_button)
        time_row.addWidget(self.time_slider, 1)
        time_row.addWidget(self.time_label)
        root_layout.addLayout(time_row)

        button_box = QtWidgets.QDialogButtonBox()
        self.apply_button = button_box.addButton("Apply", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        self.cancel_button = button_box.addButton("Cancel", QtWidgets.QDialogButtonBox.ButtonRole.RejectRole)
        self.apply_button.setProperty("role", "primary")
        self.apply_button.clicked.connect(self._apply)
        self.cancel_button.clicked.connect(self._cancel)
        root_layout.addWidget(button_box)

        self.toolbar = self.addToolBar("Segmentation tools")
        self.toolbar.setMovable(False)
        self.tool_group = QtGui.QActionGroup(self)
        self.tool_group.setExclusive(True)
        self.tool_actions = {}
        for key, label, shortcut in (
            ("brush", "Brush", "B"),
            ("eraser", "Eraser", "E"),
            ("picker", "Picker", "I"),
        ):
            action = QtGui.QAction(label, self, checkable=True)
            action.setShortcut(QtGui.QKeySequence(shortcut))
            action.setToolTip(f"{label} ({shortcut})")
            action.triggered.connect(lambda checked, tool=key: self._set_tool(tool) if checked else None)
            self.tool_group.addAction(action)
            self.toolbar.addAction(action)
            self.tool_actions[key] = action
        self.tool_actions["brush"].setChecked(True)
        self.toolbar.addSeparator()
        self.undo_action = QtGui.QAction(
            self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_ArrowBack), "Undo", self
        )
        self.undo_action.setShortcut(QtGui.QKeySequence.StandardKey.Undo)
        self.undo_action.triggered.connect(self._undo)
        self.toolbar.addAction(self.undo_action)
        self.redo_action = QtGui.QAction(
            self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_ArrowForward), "Redo", self
        )
        self.redo_action.setShortcut(QtGui.QKeySequence.StandardKey.Redo)
        self.redo_action.triggered.connect(self._redo)
        self.toolbar.addAction(self.redo_action)

    def _build_actions(self):
        reset_action = QtGui.QAction(self)
        reset_action.setShortcut(QtGui.QKeySequence("R"))
        reset_action.triggered.connect(self._reset_views)
        self.addAction(reset_action)
        decrease_action = QtGui.QAction(self)
        decrease_action.setShortcut(QtGui.QKeySequence("["))
        decrease_action.triggered.connect(lambda: self._step_brush_size(-1))
        self.addAction(decrease_action)
        increase_action = QtGui.QAction(self)
        increase_action.setShortcut(QtGui.QKeySequence("]"))
        increase_action.triggered.connect(lambda: self._step_brush_size(1))
        self.addAction(increase_action)
        previous_action = QtGui.QAction(self)
        previous_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Left))
        previous_action.triggered.connect(lambda: self._step_time(-1))
        self.addAction(previous_action)
        next_action = QtGui.QAction(self)
        next_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Right))
        next_action.triggered.connect(lambda: self._step_time(1))
        self.addAction(next_action)

    def _current_volume(self):
        index = int(self.combo_content.currentIndex())
        key = (index, self.current_time)
        if key in self._volume_cache:
            return self._volume_cache[key]
        t = self.current_time
        mag_t = None if self.magnitude is None else np.asarray(self.magnitude[..., min(t, self.magnitude.shape[3] - 1)], dtype=np.float32)
        flow_t = None if self.flow is None else np.asarray(self.flow[..., min(t, self.flow.shape[3] - 1), :], dtype=np.float32)
        if index == 0 and mag_t is not None and flow_t is not None:
            volume = mag_t * np.sqrt(np.sum(np.square(flow_t, dtype=np.float32), axis=-1))
        elif index == 1 and mag_t is not None:
            volume = mag_t
        elif index == 2 and flow_t is not None:
            volume = np.sqrt(np.sum(np.square(flow_t, dtype=np.float32), axis=-1))
        elif index in (3, 4, 5) and flow_t is not None:
            volume = flow_t[..., index - 3]
        elif mag_t is not None:
            volume = mag_t
        else:
            volume = np.zeros(self.labels.shape[:3], dtype=np.float32)
        volume = np.asarray(volume, dtype=np.float32)
        if len(self._volume_cache) >= 18:
            self._volume_cache.clear()
        self._volume_cache[key] = volume
        return volume

    def _default_levels(self, volume):
        finite = np.asarray(volume)[np.isfinite(volume)]
        if finite.size == 0:
            return 0.0, 1.0
        if self.combo_content.currentIndex() >= 3:
            maximum = max(float(np.percentile(np.abs(finite), 99.5)), 1e-6)
            return -maximum, maximum
        low, high = np.percentile(finite, (0.5, 99.5))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low = float(np.min(finite))
            high = float(np.max(finite))
        if high <= low:
            high = low + 1.0
        return float(low), float(high)

    def _content_changed(self, _index):
        self._display_levels = None
        self._refresh_views(reset_levels=True)

    def _time_changed(self, value):
        self.current_time = int(value)
        self._display_levels = None
        self._refresh_views(reset_levels=True)

    def _step_time(self, delta):
        target = int(np.clip(self.current_time + int(delta), 0, self.labels.shape[3] - 1))
        self.time_slider.setValue(target)

    def _plane_slice(self, volume, plane):
        x, y, z = (int(value) for value in self.cursor)
        if plane == "axial":
            return volume[:, :, z], (x, y), z
        if plane == "coronal":
            return volume[:, y, :], (x, z), y
        return volume[x, :, :], (y, z), x

    def _refresh_views(self, reset_levels=False):
        volume = self._current_volume()
        if reset_levels or self._display_levels is None:
            self._display_levels = self._default_levels(volume)
        levels = tuple(float(value) for value in self._display_levels)
        cmap_name = "RdBu_r" if self.combo_content.currentIndex() >= 3 else "gray"
        cmap = make_colormap(cmap_name)
        labels_t = self.labels[..., self.current_time]
        colors = self._label_color_map()
        opacity = self.opacity_slider.value() / 100.0
        for plane, view in self.slice_views.items():
            spec = PLANE_SPECS[plane]
            image_slice, cursor_2d, fixed = self._plane_slice(volume, plane)
            label_slice, _, _ = self._plane_slice(labels_t, plane)
            overlay = make_label_overlay(label_slice, colors, opacity)
            view.set_slice(
                image_slice,
                overlay,
                (self.resolution[spec.horizontal_axis], self.resolution[spec.vertical_axis]),
                cursor_2d,
                fixed,
                cmap,
                levels,
            )
            color = self.label_colors.get(str(self.active_label), self._default_color(self.active_label))
            view.set_brush(self.brush_diameter_mm, color, True)
        for axis, name in enumerate(("LR", "AP", "FH")):
            self.coordinate_labels[axis].setText(
                f"{name} {int(self.cursor[axis]) + 1} / {self.labels.shape[axis]}"
            )
        self.time_label.setText(f"T {self.current_time + 1} / {self.labels.shape[3]}")
        self.levels_label.setText(f"{levels[0]:.5g} .. {levels[1]:.5g}")
        self.undo_action.setEnabled(bool(self._undo_stack))
        self.redo_action.setEnabled(bool(self._redo_stack))

    def _refresh_plane_overlay(self, plane):
        labels_t = self.labels[..., self.current_time]
        label_slice, _, _ = self._plane_slice(labels_t, plane)
        overlay = make_label_overlay(
            label_slice,
            self._label_color_map(),
            self.opacity_slider.value() / 100.0,
        )
        self.slice_views[plane].set_overlay(overlay)

    def _label_color_map(self):
        values = set(self._label_counts)
        values.add(int(self.active_label))
        for key in self.label_colors:
            try:
                value = int(key)
            except (TypeError, ValueError):
                continue
            if value > 0:
                values.add(value)
        return {
            value: self.label_colors.get(str(value), self._default_color(value))
            for value in values
        }

    @staticmethod
    def _default_color(label):
        palette = (
            "#ff6b6b",
            "#4dabf7",
            "#51cf66",
            "#ffd43b",
            "#f783ac",
            "#74c0fc",
            "#63e6be",
            "#ffa94d",
            "#b197fc",
            "#a9e34b",
        )
        return palette[(max(1, int(label)) - 1) % len(palette)]

    def _ensure_active_label_metadata(self):
        key = str(self.active_label)
        self.label_names.setdefault(key, f"Label {self.active_label}")
        self.label_colors.setdefault(key, self._default_color(self.active_label))
        self.label_name_edit.setText(self.label_names[key])
        color = self.label_colors[key]
        self.label_color_button.setText(color)
        self.label_color_button.setStyleSheet(f"background-color: {color}; color: #111111;")

    def _refresh_label_list(self):
        current = self.active_label
        count_map = dict(self._label_counts)
        count_map.setdefault(current, 0)
        self.label_list.blockSignals(True)
        self.label_list.clear()
        selected_item = None
        for label in sorted(count_map):
            key = str(label)
            self.label_names.setdefault(key, f"Label {label}")
            self.label_colors.setdefault(key, self._default_color(label))
            item = QtWidgets.QListWidgetItem(f"{label}  {self.label_names[key]}  ({count_map[label]})")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, label)
            swatch = QtGui.QPixmap(14, 14)
            swatch.fill(QtGui.QColor(self.label_colors[key]))
            item.setIcon(QtGui.QIcon(swatch))
            self.label_list.addItem(item)
            if label == current:
                selected_item = item
        if selected_item is not None:
            self.label_list.setCurrentItem(selected_item)
        self.label_list.blockSignals(False)
        self._ensure_active_label_metadata()

    def _label_selected(self, current, _previous):
        if current is None:
            return
        value = current.data(QtCore.Qt.ItemDataRole.UserRole)
        if value is not None:
            self.active_label_spin.setValue(int(value))

    def _active_label_changed(self, value):
        self.active_label = max(1, int(value))
        self._ensure_active_label_metadata()
        self._refresh_label_list()
        self._refresh_views()

    def _add_label(self):
        values = list(self._label_counts)
        value = max(values + [self.active_label]) + 1
        self.active_label_spin.setValue(value)
        self._refresh_label_list()

    def _label_name_changed(self):
        self.label_names[str(self.active_label)] = self.label_name_edit.text().strip() or f"Label {self.active_label}"
        self._refresh_label_list()

    def _choose_label_color(self):
        current = QtGui.QColor(self.label_colors.get(str(self.active_label), self._default_color(self.active_label)))
        color = QtWidgets.QColorDialog.getColor(current, self, "Select Label Color")
        if not color.isValid():
            return
        self.label_colors[str(self.active_label)] = color.name()
        self._ensure_active_label_metadata()
        self._refresh_label_list()
        self._refresh_views()

    def _set_tool(self, tool):
        self.tool = str(tool)
        self._refresh_views()

    def _brush_diameter_changed(self, value):
        self.brush_diameter_mm = max(1.0, float(value))
        for view in self.slice_views.values():
            color = self.label_colors.get(str(self.active_label), self._default_color(self.active_label))
            view.set_brush(self.brush_diameter_mm, color, True)

    def _step_brush_size(self, delta):
        self.brush_diameter_spin.setValue(
            float(
                np.clip(
                    self.brush_diameter_spin.value() + int(delta) * self.brush_diameter_spin.singleStep(),
                    self.brush_diameter_spin.minimum(),
                    self.brush_diameter_spin.maximum(),
                )
            )
        )

    def _cursor_requested(self, plane, h, v):
        spec = PLANE_SPECS[plane]
        self.cursor[spec.horizontal_axis] = int(h)
        self.cursor[spec.vertical_axis] = int(v)
        self._refresh_views()

    def _view_hovered(self, plane, h, v):
        if (
            QtWidgets.QApplication.keyboardModifiers()
            & QtCore.Qt.KeyboardModifier.ShiftModifier
            and QtWidgets.QApplication.mouseButtons() == QtCore.Qt.MouseButton.NoButton
        ):
            self._cursor_requested(plane, h, v)

    def _step_slice(self, plane, delta):
        axis = PLANE_SPECS[plane].fixed_axis
        self.cursor[axis] = int(
            np.clip(self.cursor[axis] + int(delta), 0, self.labels.shape[axis] - 1)
        )
        self._refresh_views()

    def _stroke_started(self, plane, h, v, temporary_erase):
        self._cursor_requested(plane, h, v)
        if self.tool == "picker":
            value = int(self.labels[tuple(self.cursor) + (self.current_time,)])
            if value > 0:
                self.active_label_spin.setValue(value)
            self._stroke_before = None
            return
        self._stroke_before = {}
        self._paint_at(plane, h, v, temporary_erase)

    def _stroke_moved(self, plane, h, v, temporary_erase):
        if self._stroke_before is None:
            return
        self._paint_at(plane, h, v, temporary_erase)

    def _update_label_counts(self, before, after):
        before_values, before_counts = np.unique(before, return_counts=True)
        after_values, after_counts = np.unique(after, return_counts=True)
        for value, count in zip(before_values, before_counts):
            label = int(value)
            if label > 0:
                self._label_counts[label] = self._label_counts.get(label, 0) - int(count)
        for value, count in zip(after_values, after_counts):
            label = int(value)
            if label > 0:
                self._label_counts[label] = self._label_counts.get(label, 0) + int(count)
        self._label_counts = {
            label: count for label, count in self._label_counts.items() if count > 0
        }

    def _stroke_finished(self, _plane):
        if not self._stroke_before:
            self._stroke_before = None
            return
        flat = self.labels.reshape(-1)
        indices = np.fromiter(self._stroke_before.keys(), dtype=np.int64)
        before = np.fromiter(self._stroke_before.values(), dtype=self.labels.dtype)
        after = flat[indices].copy()
        changed = before != after
        if np.any(changed):
            self._update_label_counts(before[changed], after[changed])
            self._undo_stack.append(EditCommand(indices[changed], before[changed], after[changed]))
            if len(self._undo_stack) > 64:
                self._undo_stack = self._undo_stack[-64:]
            self._redo_stack.clear()
            self._dirty = True
        self._stroke_before = None
        self._refresh_label_list()
        self._refresh_views()

    def _paint_at(self, plane, h, v, temporary_erase):
        spec = PLANE_SPECS[plane]
        radius = self.brush_diameter_mm / 2.0
        h_spacing = float(self.resolution[spec.horizontal_axis])
        v_spacing = float(self.resolution[spec.vertical_axis])
        h_radius = max(0, int(np.ceil(radius / h_spacing)))
        v_radius = max(0, int(np.ceil(radius / v_spacing)))
        h_values = np.arange(max(0, h - h_radius), min(self.labels.shape[spec.horizontal_axis], h + h_radius + 1))
        v_values = np.arange(max(0, v - v_radius), min(self.labels.shape[spec.vertical_axis], v + v_radius + 1))
        hh, vv = np.meshgrid(h_values, v_values, indexing="ij")
        selected = ((hh - h) * h_spacing) ** 2 + ((vv - v) * v_spacing) ** 2 <= radius**2
        if not np.any(selected):
            return
        spatial_count = int(np.count_nonzero(selected))
        coords = [np.empty(spatial_count, dtype=np.intp) for _ in range(3)]
        coords[spec.horizontal_axis] = hh[selected]
        coords[spec.vertical_axis] = vv[selected]
        coords[spec.fixed_axis].fill(int(self.cursor[spec.fixed_axis]))
        times = np.arange(self.labels.shape[3], dtype=np.intp) if self.all_frames_check.isChecked() else np.asarray([self.current_time], dtype=np.intp)
        x = np.repeat(coords[0], len(times))
        y = np.repeat(coords[1], len(times))
        z = np.repeat(coords[2], len(times))
        t = np.tile(times, spatial_count)
        indices = np.ravel_multi_index((x, y, z, t), self.labels.shape)
        flat = self.labels.reshape(-1)
        value = 0 if temporary_erase or self.tool == "eraser" else self.active_label
        changed_indices = indices[flat[indices] != value]
        if changed_indices.size == 0:
            return
        for index in changed_indices:
            key = int(index)
            if key not in self._stroke_before:
                self._stroke_before[key] = int(flat[key])
        flat[changed_indices] = int(value)
        self._refresh_plane_overlay(plane)

    def _undo(self):
        if not self._undo_stack:
            return
        command = self._undo_stack.pop()
        self.labels.reshape(-1)[command.indices] = command.before
        self._update_label_counts(command.after, command.before)
        self._redo_stack.append(command)
        self._dirty = True
        self._refresh_label_list()
        self._refresh_views()

    def _redo(self):
        if not self._redo_stack:
            return
        command = self._redo_stack.pop()
        self.labels.reshape(-1)[command.indices] = command.after
        self._update_label_counts(command.before, command.after)
        self._undo_stack.append(command)
        self._dirty = True
        self._refresh_label_list()
        self._refresh_views()

    def _adjust_window_level(self, delta_x, delta_y):
        low, high = self._display_levels or self._default_levels(self._current_volume())
        width = max(float(high - low), 1e-6)
        center = (float(low) + float(high)) / 2.0
        width *= float(np.exp(float(delta_x) * 0.012))
        center -= float(delta_y) * width * 0.004
        self._display_levels = (center - width / 2.0, center + width / 2.0)
        self._refresh_views()

    def _reset_views(self):
        self._display_levels = None
        for view in self.slice_views.values():
            view.reset_view()
        self._refresh_views(reset_levels=True)

    def _toggle_maximized_view(self, plane):
        if plane not in self.slice_views:
            return
        for view in self.slice_views.values():
            self.views_layout.removeWidget(view)
        self.views_layout.removeWidget(self.navigation_panel)

        if self._maximized_plane == plane:
            self._maximized_plane = None
            self.views_layout.addWidget(self.slice_views["axial"], 0, 0)
            self.views_layout.addWidget(self.slice_views["coronal"], 0, 1)
            self.views_layout.addWidget(self.slice_views["sagittal"], 1, 0)
            self.views_layout.addWidget(self.navigation_panel, 1, 1)
            self.navigation_panel.show()
            for view in self.slice_views.values():
                view.show()
        else:
            self._maximized_plane = plane
            for name, view in self.slice_views.items():
                view.setVisible(name == plane)
            self.navigation_panel.hide()
            self.views_layout.addWidget(self.slice_views[plane], 0, 0, 2, 2)

        self.views_layout.setRowStretch(0, 1)
        self.views_layout.setRowStretch(1, 1)
        self.views_layout.setColumnStretch(0, 1)
        self.views_layout.setColumnStretch(1, 1)
        self.views_layout.invalidate()
        self.views_layout.activate()
        self.views_widget.updateGeometry()

    def _apply(self):
        settings = {
            "brush_radius": max(1, int(round(self.brush_diameter_mm / (2.0 * float(np.min(self.resolution)))))),
            "edit_all_timepoints": self.all_frames_check.isChecked(),
            "active_label": self.active_label,
            "cursor": self.cursor.copy(),
            "current_time": self.current_time,
            "changed": bool(self._dirty),
        }
        self._allow_close = True
        self._dirty = False
        self.applied.emit(
            self.labels.copy(),
            dict(self.label_names),
            dict(self.label_colors),
            settings,
        )
        self.close()

    def _cancel(self):
        self._allow_close = True
        self.cancelled.emit()
        self.close()

    def closeEvent(self, event):
        if not self._allow_close and self._dirty:
            choice = QtWidgets.QMessageBox.warning(
                self,
                "Discard Segmentation Edits",
                "Discard the unapplied segmentation edits?",
                QtWidgets.QMessageBox.StandardButton.Discard
                | QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.Cancel,
            )
            if choice != QtWidgets.QMessageBox.StandardButton.Discard:
                event.ignore()
                return
        if not self._allow_close:
            self.cancelled.emit()
        event.accept()
