from ..config import METRIC_CMAPS, METRIC_DATA_KEYS
from ..rendering.style import configured_metric_style
import copy
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
from matplotlib.path import Path
from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    gaussian_filter,
    label as label_components,
    map_coordinates,
)
from skimage.measure import find_contours

from ..algorithms.metrics import _build_plane_slice_region
from ..algorithms.surfaces import _build_branch_grid, _extract_surface
from .contour_edit import new_contour_from_stroke, polygon_area, replace_contour_segment
from .slice_view import PLANE_SPECS, SliceView, make_colormap, make_label_overlay


class ReorderableSliceContainer(QtWidgets.QWidget):
    """Vertical host that lets users reorder slice viewers by dragging them."""

    orderChanged = QtCore.Signal(list)

    def __init__(self, views, order=None, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)
        self._views = dict(views)
        self._order = []
        self._drag_name = None
        self._drag_start = None
        for name, view in self._views.items():
            view.setProperty("slice_name", str(name))
            view.title_label.setProperty("slice_name", str(name))
            view.title_label.installEventFilter(self)
        self.set_order(order or list(self._views), emit=False)

    def order(self):
        return list(self._order)

    def set_order(self, order, *, emit=True):
        normalized = [str(name) for name in list(order or []) if str(name) in self._views]
        normalized.extend(name for name in self._views if name not in normalized)
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget() is not None:
                item.widget().setParent(self)
        self._order = normalized
        for name in normalized:
            self._layout.addWidget(self._views[name], 1)
        if emit:
            self.orderChanged.emit(self.order())

    def eventFilter(self, watched, event):
        if event.type() == QtCore.QEvent.Type.MouseButtonPress and event.button() == QtCore.Qt.MouseButton.LeftButton:
            name = watched.property("slice_name") or getattr(watched.parentWidget(), "property", lambda *_: None)("slice_name")
            if name:
                self._drag_name = str(name)
                self._drag_start = event.position().toPoint()
        elif event.type() == QtCore.QEvent.Type.MouseMove and event.buttons() & QtCore.Qt.MouseButton.LeftButton:
            if self._drag_name and self._drag_start is not None:
                if (event.position().toPoint() - self._drag_start).manhattanLength() >= QtWidgets.QApplication.startDragDistance():
                    drag_name = self._drag_name
                    self._drag_name = None
                    self._drag_start = None
                    drag = QtGui.QDrag(self)
                    mime = QtCore.QMimeData()
                    mime.setData("application/x-autoflow-slice", drag_name.encode("utf-8"))
                    drag.setMimeData(mime)
                    drag.exec(QtCore.Qt.DropAction.MoveAction)
                    return True
        elif event.type() in (QtCore.QEvent.Type.MouseButtonRelease, QtCore.QEvent.Type.Leave):
            self._drag_name = None
            self._drag_start = None
        return super().eventFilter(watched, event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat("application/x-autoflow-slice"):
            event.acceptProposedAction()

    def dropEvent(self, event):
        if not event.mimeData().hasFormat("application/x-autoflow-slice"):
            event.ignore()
            return
        name = bytes(event.mimeData().data("application/x-autoflow-slice")).decode("utf-8")
        if name not in self._order:
            event.ignore()
            return
        y = int(event.position().toPoint().y())
        target = len(self._order)
        for index, item_name in enumerate(self._order):
            if y < self._views[item_name].geometry().center().y():
                target = index
                break
        updated = list(self._order)
        updated.remove(name)
        updated.insert(min(target, len(updated)), name)
        self.set_order(updated)
        event.acceptProposedAction()


class OrthoViewer(QtWidgets.QWidget):
    timeStepRequested = QtCore.Signal(int)
    planeRoiChanged = QtCore.Signal(int)

    def __init__(self, workspace, parent=None):
        super().__init__(parent)
        self.workspace = workspace
        self._selected_plane_idx = None
        self._cache = {}
        self._playback_active = False
        self._pending_cursor = None
        self._cursor_refresh_timer = QtCore.QTimer(self)
        self._cursor_refresh_timer.setSingleShot(True)
        self._cursor_refresh_timer.setInterval(24)
        self._cursor_refresh_timer.timeout.connect(self._apply_pending_cursor)
        self._manual_levels = None
        self._current_volume = None
        self._current_title = ""
        self._updating_colorbar = False
        self._maximized_view = None
        self._slice_keys = {}
        self._colorbar_state = None
        self._noise_overlay_visible = False
        self._noise_overlay_opacity = 0.35
        self._plane_region_surface_cache = {}
        self._contour_stroke = []
        self._contour_preview = None
        self._edit_boundary = None
        self._edit_boundary_ready = False
        self._contour_display_override = None
        settings = QtCore.QSettings("AutoFlow", "AutoFlow")
        self._plane_display_smoothing_sigma = float(settings.value("plane/display_smoothing_sigma", 0.8))
        self._plane_display_smoothing_points = int(settings.value("plane/display_smoothing_points", 8))
        # Zero keeps the historical adaptive threshold; a positive value is a
        # fixed maximum endpoint-to-contour distance in millimetres.
        self._contour_snap_distance_mm = float(
            np.clip(float(settings.value("plane/contour_snap_distance_mm", 0.0)), 0.0, 50.0)
        )
        self._slice_default_view_fraction = float(
            np.clip(float(settings.value("plane/slice_default_view_fraction", 0.5)), 0.1, 1.0)
        )
        self._plane_roi_undo = []
        self._plane_roi_redo = []
        self._build_ui()

    def _cached(self, group, key, builder, max_items=24):
        bucket = self._cache.setdefault(group, {})
        if key in bucket:
            return bucket[key]
        value = builder()
        if len(bucket) >= max_items:
            bucket.clear()
        bucket[key] = value
        return value

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        ctrl = QtWidgets.QHBoxLayout()
        self.combo_content = QtWidgets.QComboBox()
        self._content_labels = [
            "Flow LR (cm/s)", "Flow AP (cm/s)", "Flow FH (cm/s)",
            "Magnitude", "PC-MRA", "Speed (cm/s)",
            "WSS (Pa)", "TKE (J/m³)",
            "Pressure Grad LR (Pa/m)", "Pressure Grad AP (Pa/m)", "Pressure Grad FH (Pa/m)", "|Pressure Grad| (Pa/m)",
            "Relative Pressure (Pa)",
            "Vorticity Magnitude (s⁻¹)", "Q-Criterion (s⁻²)", "Swirling Strength λci (s⁻¹)",
            "Corr Low LR (rad)", "Corr Low AP (rad)", "Corr Low FH (rad)",
            "Corr High LR (rad)", "Corr High AP (rad)", "Corr High FH (rad)",
            "Wrap Mask (any component)", "Wrap Count LR", "Wrap Count AP", "Wrap Count FH",
            "Unwrapped − Wrapped Speed (cm/s)", "Through-plane Flow (cm/s)",
        ]
        self.combo_content.setPlaceholderText("Load data to view content")
        self._refresh_content_choices()
        self.combo_content.currentIndexChanged.connect(self._on_content_changed)
        ctrl.addWidget(QtWidgets.QLabel("Content:"))
        ctrl.addWidget(self.combo_content, 1)
        ctrl.addWidget(QtWidgets.QLabel("Segmentation:"))
        self.slider_overlay_opacity = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_overlay_opacity.setRange(0, 100)
        self.slider_overlay_opacity.setFixedWidth(78)
        self.slider_overlay_opacity.setToolTip("Segmentation overlay opacity")
        self.slider_overlay_opacity.valueChanged.connect(self._on_overlay_opacity_changed)
        ctrl.addWidget(self.slider_overlay_opacity)
        self.label_overlay_opacity = QtWidgets.QLabel("35%")
        self.label_overlay_opacity.setMinimumWidth(34)
        ctrl.addWidget(self.label_overlay_opacity)
        self.btn_noise_overlay = QtWidgets.QCheckBox("Noise mask")
        self.btn_noise_overlay.setChecked(False)
        self.btn_noise_overlay.setToolTip("Show excluded voxels (weak signal or unstable velocity) in red on orthogonal slices")
        self.btn_noise_overlay.toggled.connect(self._on_noise_overlay_toggled)
        ctrl.addWidget(self.btn_noise_overlay)
        self.slider_noise_overlay_opacity = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_noise_overlay_opacity.setRange(0, 100)
        self.slider_noise_overlay_opacity.setFixedWidth(78)
        self.slider_noise_overlay_opacity.setToolTip("Noise mask overlay opacity")
        self.slider_noise_overlay_opacity.setValue(0)
        self.slider_noise_overlay_opacity.valueChanged.connect(self._on_noise_overlay_opacity_changed)
        ctrl.addWidget(self.slider_noise_overlay_opacity)
        self.label_noise_overlay_opacity = QtWidgets.QLabel("0%")
        self.label_noise_overlay_opacity.setMinimumWidth(34)
        ctrl.addWidget(self.label_noise_overlay_opacity)
        self.btn_reset_views = QtWidgets.QPushButton()
        self.btn_reset_views.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_BrowserReload)
        )
        self.btn_reset_views.setToolTip("Reset slice zoom and display range (R)")
        self.btn_reset_views.setProperty("role", "icon")
        self.btn_reset_views.clicked.connect(self._reset_views)
        ctrl.addWidget(self.btn_reset_views)
        self.btn_edit_contour = QtWidgets.QToolButton()
        self.btn_edit_contour.setText("Edit contour")
        self.btn_edit_contour.setCheckable(True)
        self.btn_edit_contour.setEnabled(False)
        self.btn_edit_contour.toggled.connect(self._toggle_plane_roi_editor)
        self.btn_undo_plane_roi = QtWidgets.QToolButton()
        self.btn_undo_plane_roi.setText("Undo")
        self.btn_undo_plane_roi.clicked.connect(self._undo_plane_roi)
        self.btn_redo_plane_roi = QtWidgets.QToolButton()
        self.btn_redo_plane_roi.setText("Redo")
        self.btn_redo_plane_roi.clicked.connect(self._redo_plane_roi)
        self.btn_reset_plane_roi = QtWidgets.QToolButton()
        self.btn_reset_plane_roi.setText("Reset frame")
        self.btn_reset_plane_roi.setToolTip("Remove contour edits from the current frame only")
        self.btn_reset_plane_roi.clicked.connect(self._reset_plane_roi)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self.contour_controls = QtWidgets.QWidget()
        contour_ctrl = QtWidgets.QHBoxLayout(self.contour_controls)
        contour_ctrl.setContentsMargins(0, 0, 0, 0)
        contour_ctrl.addWidget(QtWidgets.QLabel("Plane contour:"))
        contour_ctrl.addWidget(self.btn_edit_contour)
        contour_ctrl.addWidget(self.btn_undo_plane_roi)
        contour_ctrl.addWidget(self.btn_redo_plane_roi)
        contour_ctrl.addWidget(self.btn_reset_plane_roi)
        contour_ctrl.addStretch()
        self.contour_controls.hide()
        layout.addWidget(self.contour_controls)

        slider_layout = QtWidgets.QHBoxLayout()
        self.slider_x = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_y = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_z = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.label_x = QtWidgets.QLabel("LR:0")
        self.label_y = QtWidgets.QLabel("AP:0")
        self.label_z = QtWidgets.QLabel("FH:0")
        for lbl, sl in [(self.label_x, self.slider_x), (self.label_y, self.slider_y), (self.label_z, self.slider_z)]:
            sl.setRange(0, 0)
            sl.setTracking(False)
            sl.valueChanged.connect(self._on_slider_changed)
            slider_layout.addWidget(lbl)
            slider_layout.addWidget(sl)
        layout.addLayout(slider_layout)

        self.label_value = QtWidgets.QLabel("Voxel: -   Value: -")
        self.label_plane_metric = QtWidgets.QLabel("Plane metrics: -")
        self.label_value.setWordWrap(True)
        self.label_plane_metric.setWordWrap(True)
        layout.addWidget(self.label_value)
        layout.addWidget(self.label_plane_metric)
        view_row = QtWidgets.QHBoxLayout()
        view_row.setSpacing(4)
        self.slice_views = {name: SliceView(name, self) for name in PLANE_SPECS}
        for view in self.slice_views.values():
            view.set_default_view_fraction(self._slice_default_view_fraction)
        settings = QtCore.QSettings("AutoFlow", "AutoFlow")
        stored_order = str(settings.value("ortho/slice_order", "") or "")
        order = [name for name in stored_order.split(",") if name]
        self.view_grid = ReorderableSliceContainer(self.slice_views, order=order or list(self.slice_views))
        self.view_grid.orderChanged.connect(self._on_slice_order_changed)
        view_row.addWidget(self.view_grid, 1)

        self.colorbar_widget = pg.GraphicsLayoutWidget()
        self.colorbar_widget.setBackground("#050809")
        self.colorbar_widget.setMinimumWidth(62)
        self.colorbar_widget.setMaximumWidth(82)
        self.colorbar_item = pg.ColorBarItem(
            values=(0.0, 1.0),
            width=18,
            colorMap=make_colormap("gray"),
            interactive=True,
            colorMapMenu=False,
            pen="#d7e0e3",
        )
        self.colorbar_widget.addItem(self.colorbar_item)
        self.colorbar_item.sigLevelsChanged.connect(self._on_colorbar_levels_changed)
        view_row.addWidget(self.colorbar_widget)
        layout.addLayout(view_row, 1)

        for view in self.slice_views.values():
            view.cursorRequested.connect(self._on_slice_cursor_requested)
            view.hoverMoved.connect(self._on_slice_hovered)
            view.sliceStepRequested.connect(self._on_slice_step_requested)
            view.windowLevelDragged.connect(self._adjust_window_level)
            view.viewDoubleClicked.connect(self._toggle_maximized_view)
            view.contourStrokeStarted.connect(self._on_contour_stroke_started)
            view.contourStrokeMoved.connect(self._on_contour_stroke_moved)
            view.contourStrokeFinished.connect(self._on_contour_stroke_finished)

        self._reset_shortcut = QtGui.QShortcut(QtGui.QKeySequence("R"), self)
        self._reset_shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._reset_shortcut.activated.connect(self._reset_views)
        self._previous_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key.Key_Left), self
        )
        self._previous_shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._previous_shortcut.activated.connect(lambda: self.timeStepRequested.emit(-1))
        self._next_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key.Key_Right), self
        )
        self._next_shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._next_shortcut.activated.connect(lambda: self.timeStepRequested.emit(1))

    def _on_slice_order_changed(self, order):
        QtCore.QSettings("AutoFlow", "AutoFlow").setValue(
            "ortho/slice_order", ",".join(str(name) for name in order)
        )

    def set_segmentation_edit_handler(self, handler):
        # Segmentation editing is handled by the external Labeler exchange.
        # Keep this method as a compatibility no-op for older callers.
        return None

    def set_playback_active(self, active):
        self._playback_active = bool(active)

    def _plane_voxel(self, plane, h, v):
        shape = self._get_volume_shape()
        if shape is None:
            return None
        if self._selected_plane_idx is not None and 0 <= int(self._selected_plane_idx) < len(self.workspace.planes):
            selected = self.workspace.planes[int(self._selected_plane_idx)]
            normal = np.asarray(selected.normal, dtype=float).reshape(3)
            normal /= np.linalg.norm(normal) + 1e-12
            axis_u, axis_v = self._plane_axes(normal)
            bases = {
                "axial": (axis_u, axis_v),
                "coronal": (axis_v, normal),
                "sagittal": (axis_u, normal),
            }
            horizontal, vertical = bases[plane]
            view = self.slice_views[plane]
            offset_h = view._sample_origin[0] + int(h) * view._spacing[0]
            offset_v = view._sample_origin[1] + int(v) * view._spacing[1]
            point = (
                np.asarray(selected.center, dtype=float).reshape(3)
                + offset_h * horizontal
                + offset_v * vertical
            )
            voxel = np.rint(point / (self._get_resolution() + 1e-12)).astype(int)
            return tuple(int(np.clip(voxel[axis], 0, shape[axis] - 1)) for axis in range(3))
        cx, cy, cz = self.slider_x.value(), self.slider_y.value(), self.slider_z.value()
        if plane == "axial":
            return int(h), int(v), cz
        if plane == "coronal":
            return int(h), cy, int(v)
        if plane == "sagittal":
            return cx, int(h), int(v)
        return None

    def _on_slice_cursor_requested(self, plane, h, v):
        voxel = self._plane_voxel(plane, h, v)
        if voxel is not None:
            self._set_cursor(*voxel)

    def _on_slice_hovered(self, plane, h, v):
        voxel = self._plane_voxel(plane, h, v)
        if voxel is None or self._current_volume is None:
            return
        try:
            value = float(self._current_volume[voxel])
            self.label_value.setText(
                f"Voxel (LR, AP, FH): {tuple(int(x) for x in voxel)}   {self._current_title}: {value:.6g}"
            )
        except Exception:
            pass

    def _on_slice_step_requested(self, plane, delta):
        if self._selected_plane_idx is not None:
            return
        shape = self._get_volume_shape()
        if shape is None:
            return
        cursor = [self.slider_x.value(), self.slider_y.value(), self.slider_z.value()]
        axis = PLANE_SPECS[plane].fixed_axis
        cursor[axis] = int(np.clip(cursor[axis] + int(delta), 0, shape[axis] - 1))
        self._set_cursor(*cursor)

    def _set_cursor(self, x, y, z):
        self.slider_x.blockSignals(True)
        self.slider_y.blockSignals(True)
        self.slider_z.blockSignals(True)
        self.slider_x.setValue(int(x))
        self.slider_y.setValue(int(y))
        self.slider_z.setValue(int(z))
        self.slider_x.blockSignals(False)
        self.slider_y.blockSignals(False)
        self.slider_z.blockSignals(False)
        self._update_labels()
        self.workspace.ortho_cursor = np.array([int(x), int(y), int(z)], dtype=int)
        self.refresh(update_plane=False)

    def _plane_intersection_for_slice(self, plane, fixed_axis, fixed_index, shape, resolution):
        normal = np.asarray(plane.normal, dtype=float).reshape(3)
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-12:
            return None
        normal /= norm
        origin = np.asarray(getattr(self.workspace, "origin", np.zeros(3)), dtype=float).reshape(3)
        center = np.asarray(plane.center, dtype=float).reshape(3) + origin
        axes = [axis for axis in range(3) if axis != int(fixed_axis)]
        fixed_value = float(origin[int(fixed_axis)]) + float(fixed_index) * float(resolution[int(fixed_axis)])
        rhs = float(np.dot(normal, center)) - float(normal[int(fixed_axis)]) * fixed_value
        a, b = float(normal[axes[0]]), float(normal[axes[1]])
        if abs(a) + abs(b) < 1e-10:
            return None
        bounds = [
            (
                float(origin[axis]) - 0.5 * float(resolution[axis]),
                float(origin[axis]) + (int(shape[axis]) - 0.5) * float(resolution[axis]),
            )
            for axis in axes
        ]
        points = []
        for value in bounds[0]:
            if abs(b) > 1e-10:
                other = (rhs - a * value) / b
                if bounds[1][0] - 1e-6 <= other <= bounds[1][1] + 1e-6:
                    point = np.zeros(3, dtype=float)
                    point[int(fixed_axis)] = fixed_value
                    point[axes[0]] = value
                    point[axes[1]] = other
                    points.append(point)
        for value in bounds[1]:
            if abs(a) > 1e-10:
                other = (rhs - b * value) / a
                if bounds[0][0] - 1e-6 <= other <= bounds[0][1] + 1e-6:
                    point = np.zeros(3, dtype=float)
                    point[int(fixed_axis)] = fixed_value
                    point[axes[0]] = other
                    point[axes[1]] = value
                    points.append(point)
        unique = []
        for point in points:
            if not any(float(np.linalg.norm(point - old)) < 1e-5 for old in unique):
                unique.append(point)
        if len(unique) < 2:
            return None
        first, second = max(
            ((one, two) for i, one in enumerate(unique) for two in unique[i + 1:]),
            key=lambda pair: float(np.linalg.norm(pair[0] - pair[1])),
        )
        return [
            [float(first[axis] - origin[axis]) for axis in axes],
            [float(second[axis] - origin[axis]) for axis in axes],
        ]

    def _resample_basis(self, volume, center, axis_u, axis_v, fov_mm, spacing_mm, order=1, cell_centered=True):
        center = np.asarray(center, dtype=float).reshape(3)
        axis_u = np.asarray(axis_u, dtype=float).reshape(3)
        axis_v = np.asarray(axis_v, dtype=float).reshape(3)
        res = self._get_resolution()
        origin = np.asarray(getattr(self.workspace, "origin", np.zeros(3)), dtype=float).reshape(3)
        grid_key = (
            tuple(np.round(center, 5).tolist()),
            tuple(np.round(axis_u, 7).tolist()),
            tuple(np.round(axis_v, 7).tolist()),
            round(float(fov_mm), 3),
            round(float(spacing_mm), 3),
            tuple(np.round(res, 6).tolist()),
            tuple(np.round(origin, 6).tolist()),
            bool(cell_centered),
        )

        def _build_grid():
            count = int(np.clip(np.ceil(float(fov_mm) / max(float(spacing_mm), 1e-3)) + 1, 129, 257))
            axis = np.linspace(-float(fov_mm) / 2.0, float(fov_mm) / 2.0, count)
            gu, gv = np.meshgrid(axis, axis, indexing="ij")
            points = (
                center.reshape(1, 1, 3)
                + gu[..., None] * axis_u.reshape(1, 1, 3)
                + gv[..., None] * axis_v.reshape(1, 1, 3)
            )
            coords = (points + origin.reshape(1, 1, 3)) / (res.reshape(1, 1, 3) + 1e-12)
            if cell_centered:
                coords -= 0.5
            return tuple(coords[..., index].ravel() for index in range(3)), axis

        coords, axis = self._cached("basis_grids", grid_key, _build_grid, max_items=18)
        count = len(axis)
        sampled = map_coordinates(volume, coords, order=int(order), mode="constant", cval=0.0)
        return sampled.reshape(count, count), axis

    def _plane_orthogonal_geometry(self, plane, frame_index):
        center = np.asarray(plane.center, dtype=float).reshape(3)
        shape = self._get_volume_shape()
        resolution = self._get_resolution()
        if shape is None or resolution.size < 3:
            return center, 40.0
        lower = -0.5 * resolution[:3]
        upper = (np.asarray(shape[:3], dtype=float) - 0.5) * resolution[:3]
        corners = np.asarray(
            [
                [x, y, z]
                for x in (lower[0], upper[0])
                for y in (lower[1], upper[1])
                for z in (lower[2], upper[2])
            ],
            dtype=float,
        )
        normal = np.asarray(plane.normal, dtype=float).reshape(3)
        normal /= np.linalg.norm(normal) + 1e-12
        axis_u, axis_v = self._plane_axes(normal)
        axes = ((axis_u, axis_v), (axis_v, normal), (axis_u, normal))
        relative = corners - center.reshape(1, 3)
        half_fov = max(
            max(float(np.max(np.abs(relative @ axis))) for axis in pair)
            for pair in axes
        )
        return center, max(8.0, 2.0 * half_fov)

    def _selected_plane_contours(self, plane, frame_index, fov_mm, sample_spacing_mm):
        key = (
            int(self._selected_plane_idx),
            int(frame_index),
            round(float(fov_mm), 3),
            round(float(sample_spacing_mm), 3),
            float(self._plane_display_smoothing_sigma),
            int(self._plane_display_smoothing_points),
            id(self.workspace.segmask_binary),
            id(self.workspace.branch_labels),
            repr(getattr(plane, "roi_edit_operations", {}) or {}),
        )
        return self._cached(
            "plane_contours",
            key,
            lambda: self._build_selected_plane_contours(plane, frame_index, fov_mm, sample_spacing_mm),
            max_items=12,
        )

    def _build_selected_plane_contours(self, plane, frame_index, fov_mm, sample_spacing_mm):
        # A saved replace operation is already the exact user-authored metric
        # ROI. Do not rasterize it back through the 3-D mask and smooth it on
        # every refresh; that made the contour visibly change after Calculate
        # Metrics even though the numeric calculation used the original
        # polygon.
        operations = list(
            (getattr(plane, "roi_edit_operations", {}) or {}).get(str(int(frame_index)), []) or []
        )
        manual_polygon = None
        for operation in reversed(operations):
            if str(operation.get("mode", "")) != "replace":
                continue
            candidate = np.asarray(operation.get("polygon", []), dtype=float)
            if candidate.ndim == 2 and candidate.shape[1] == 2 and len(candidate) >= 3:
                manual_polygon = candidate
            break
        if manual_polygon is not None:
            contours = [(np.vstack((manual_polygon, manual_polygon[0])), "#00f5d4", 1.6, False)]
            legacy = np.asarray(getattr(plane, "roi_polygon_uv_mm", []) or [], dtype=float)
            if legacy.ndim == 2 and legacy.shape[1] == 2 and len(legacy) >= 3:
                contours.append((np.vstack((legacy, legacy[0])), "#ffd43b", 1.2, True))
            return contours

        center = np.asarray(plane.center, dtype=float).reshape(3)
        normal = np.asarray(plane.normal, dtype=float).reshape(3)
        normal /= np.linalg.norm(normal) + 1e-12
        region_key = (
            int(self._selected_plane_idx),
            int(frame_index),
            tuple(np.round(center, 4)),
            tuple(np.round(normal, 6)),
            id(self.workspace.segmask_binary),
            id(self.workspace.branch_labels),
            repr(getattr(plane, "roi_edit_operations", {}) or {}),
        )
        region = self._cached(
            "plane_region",
            region_key,
            lambda: self._effective_plane_region(plane, frame_index),
        )
        result = self._sample_smooth_region_mask(
            region,
            plane,
            fov_mm,
            sample_spacing_mm=sample_spacing_mm,
        )
        if result is None:
            result = self._sample_smooth_plane_mask(
                plane,
                frame_index,
                center,
                normal,
                fov_mm,
                sample_spacing_mm=sample_spacing_mm,
            )

        contours = []
        if result is not None:
            smooth, axis_mm = result
            if self._plane_display_smoothing_sigma > 0:
                smooth = gaussian_filter(
                    np.asarray(smooth, dtype=np.float32),
                    sigma=float(self._plane_display_smoothing_sigma),
                )
            indices = np.arange(len(axis_mm), dtype=float)
            for contour in find_contours(smooth, 0.5):
                if len(contour) < int(self._plane_display_smoothing_points):
                    continue
                points = np.column_stack(
                    (
                        np.interp(contour[:, 0], indices, axis_mm),
                        np.interp(contour[:, 1], indices, axis_mm),
                    )
                )
                contours.append((points, "#00f5d4", 1.6, False))

        polygon = np.asarray(getattr(plane, "roi_polygon_uv_mm", []) or [], dtype=float)
        if polygon.ndim == 2 and polygon.shape[1] == 2 and len(polygon) >= 3:
            contours.append((np.vstack((polygon, polygon[0])), "#ffd43b", 1.2, True))
        return contours

    @staticmethod
    def _auto_image_levels(image, fallback):
        values = np.asarray(image, dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return tuple(float(value) for value in fallback)
        fallback = tuple(float(value) for value in fallback)
        signed = fallback[0] < 0.0 < fallback[1]
        if signed:
            magnitudes = np.abs(finite)
            upper = float(np.percentile(magnitudes, 99.5))
            if not np.isfinite(upper) or upper <= 1e-8:
                upper = max(abs(fallback[0]), abs(fallback[1]), 1e-6)
            return (-upper, upper)
        scale = max(float(np.percentile(np.abs(finite), 99.0)), 1e-8)
        supported = finite[np.abs(finite) > scale * 1e-5]
        if supported.size < 8:
            supported = finite
        low, high = np.percentile(supported, (0.5, 99.5))
        if not np.isfinite(low) or not np.isfinite(high) or high - low <= 1e-8:
            return fallback
        return (float(low), float(high))

    def _cached_image_levels(self, image, fallback):
        fallback = tuple(float(value) for value in fallback)
        key = (self._array_cache_token(image), fallback)
        return self._cached(
            "image_levels",
            key,
            lambda: self._auto_image_levels(image, fallback),
            max_items=24,
        )

    @staticmethod
    def _array_cache_token(value):
        if value is None:
            return None
        array = np.asarray(value)
        pointer = int(array.__array_interface__["data"][0]) if array.size else 0
        return (pointer, tuple(int(item) for item in array.shape), tuple(int(item) for item in array.strides), array.dtype.str)

    def _plane_orthogonal_slices(self, vol, labels, noise_region, plane, center, fov_mm, spacing_mm):
        normal = np.asarray(plane.normal, dtype=float); normal /= np.linalg.norm(normal) + 1e-12
        center = np.asarray(center, dtype=float).reshape(3)
        key = (
            self._array_cache_token(vol),
            tuple(np.round(center, 5).tolist()),
            tuple(np.round(normal, 7).tolist()),
            round(float(fov_mm), 3),
            round(float(spacing_mm), 3),
            tuple(np.round(self._get_resolution(), 6).tolist()),
            tuple(np.round(np.asarray(getattr(self.workspace, "origin", np.zeros(3)), dtype=float), 6).tolist()),
            self._array_cache_token(noise_region),
        )

        def _build():
            u, v = self._plane_axes(normal)
            bases = [(u, v, "Plane U × V"), (v, normal, "Plane V × N"), (u, normal, "Plane U × N")]
            images = []
            for axis_u, axis_v, title in bases:
                image, _ = self._resample_basis(
                    vol, center, axis_u, axis_v, fov_mm, spacing_mm,
                    order=1, cell_centered=True,
                )
                images.append((image, title, axis_u, axis_v))
            return images

        images = self._cached("plane_slices", key, _build, max_items=8)
        out = []
        for image, title, axis_u, axis_v in images:
            overlay = None
            if labels is not None:
                sampled, _ = self._resample_basis(
                    labels, center, axis_u, axis_v, fov_mm, spacing_mm,
                    order=0, cell_centered=True,
                )
                overlay = np.rint(sampled).astype(np.int16)
            noise_overlay = None
            if noise_region is not None:
                sampled_noise, _ = self._resample_basis(
                    noise_region, center, axis_u, axis_v, fov_mm, spacing_mm,
                    order=0, cell_centered=True,
                )
                noise_overlay = sampled_noise >= 0.5
            out.append((image, overlay, noise_overlay, title))
        return out

    def update_slider_ranges(self):
        self._refresh_content_choices()
        shape = self._get_volume_shape()
        if shape is None:
            return
        self._slice_keys.clear()
        self._colorbar_state = None
        self.slider_x.blockSignals(True)
        self.slider_y.blockSignals(True)
        self.slider_z.blockSignals(True)
        self.slider_x.setRange(0, max(0, shape[0] - 1))
        self.slider_y.setRange(0, max(0, shape[1] - 1))
        self.slider_z.setRange(0, max(0, shape[2] - 1))
        self.slider_x.setValue(min(shape[0] // 2, self.slider_x.maximum()))
        self.slider_y.setValue(min(shape[1] // 2, self.slider_y.maximum()))
        self.slider_z.setValue(min(shape[2] // 2, self.slider_z.maximum()))
        self.slider_x.blockSignals(False)
        self.slider_y.blockSignals(False)
        self.slider_z.blockSignals(False)
        self._update_labels()
        self.workspace.ortho_cursor = np.array([self.slider_x.value(), self.slider_y.value(), self.slider_z.value()], dtype=int)
        self.refresh(update_plane=False)

    def _get_volume_shape(self):
        ws = self.workspace
        if ws.flow_raw is not None and ws.flow_raw.ndim == 5:
            return ws.flow_raw.shape[:3]
        if ws.mag_raw is not None and ws.mag_raw.ndim == 4:
            return ws.mag_raw.shape[:3]
        display_labels = self._display_labels_source()
        if display_labels is not None:
            return display_labels.shape[:3]
        if ws.segmask_3d is not None:
            return ws.segmask_3d.shape[:3]
        return None

    def _update_labels(self):
        self.label_x.setText(f"LR:{self.slider_x.value()}")
        self.label_y.setText(f"AP:{self.slider_y.value()}")
        self.label_z.setText(f"FH:{self.slider_z.value()}")

    def _on_slider_changed(self, _):
        self._update_labels()
        self._pending_cursor = np.array(
            [self.slider_x.value(), self.slider_y.value(), self.slider_z.value()],
            dtype=int,
        )
        # Coalesce rapid slider events so each drag only rebuilds the slices
        # after the user pauses briefly.  The displayed cursor labels remain
        # immediate while expensive image/overlay work is reduced.
        self._cursor_refresh_timer.start()

    def _apply_pending_cursor(self):
        if self._pending_cursor is None:
            return
        cursor = np.asarray(self._pending_cursor, dtype=int)
        self._pending_cursor = None
        self.workspace.ortho_cursor = cursor.copy()
        self.refresh()

    def _on_content_changed(self, _):
        self._manual_levels = None
        self.refresh(update_plane=False)

    def _on_plane_view_changed(self, _=None):
        self._cache.pop("plane_content", None)
        self._cache.pop("plane_region", None)
        self._slice_keys.clear()
        for view in self.slice_views.values():
            view.plane_line.hide()
        if self._get_volume_shape() is not None and not self._playback_active:
            self.refresh(update_plane=True)

    def _stop_plane_roi_editor(self):
        self._contour_stroke = []
        self._contour_preview = None
        self._edit_boundary = None
        self._edit_boundary_ready = False
        for view in self.slice_views.values():
            view.set_contour_editing(False)
            view.clear_contour_draft()
        if hasattr(self, "btn_edit_contour"):
            self.btn_edit_contour.blockSignals(True)
            self.btn_edit_contour.setChecked(False)
            self.btn_edit_contour.blockSignals(False)

    def _toggle_plane_roi_editor(self, checked):
        if not checked:
            self._stop_plane_roi_editor()
            return
        plane_idx = self._selected_plane_idx
        if plane_idx is None or not (0 <= int(plane_idx) < len(self.workspace.planes)):
            self._stop_plane_roi_editor()
            return

        for name, view in self.slice_views.items():
            view.set_contour_editing(name == "axial")
        self._edit_boundary = self._current_edit_boundary()
        self._edit_boundary_ready = True
        self._prewarm_contour_support_mesh(plane_idx, self._edit_boundary)

    def _prewarm_contour_support_mesh(self, plane_idx, boundary):
        """Warm the bounded geometry cache without computing a metric."""
        metrics = getattr(self.workspace.derived, "plane_metrics", []) or []
        if not (len(metrics) == len(self.workspace.planes) and bool(metrics[int(plane_idx)])):
            return
        polygon = np.asarray(boundary, dtype=float).reshape(-1, 2) if boundary is not None else np.empty((0, 2))
        if len(polygon) < 3 or self.workspace.segmask_binary is None:
            return
        try:
            mask = np.asarray(self.workspace.segmask_binary, dtype=bool)
            t = int(self.workspace.current_t)
            mask_t = mask[..., t] if mask.ndim == 4 else mask
            plane = copy.copy(self.workspace.planes[int(plane_idx)])
            plane.roi_edit_operations = {"0": [{"mode": "replace", "polygon": polygon.tolist()}]}
            _build_plane_slice_region(
                mask_t,
                plane,
                self.workspace.resolution,
                self.workspace.origin,
                select_connected=True,
                frame_index=0,
            )
        except Exception:
            return

    def _contour_spacing(self):
        return max(float(np.min(self._get_resolution())) * 0.75, 0.25)

    def _on_contour_stroke_started(self, plane_name, u, v):
        if plane_name != "axial" or self._selected_plane_idx is None:
            return
        self._contour_stroke = [(float(u), float(v))]
        self._contour_preview = None
        self.slice_views["axial"].set_contour_draft(self._contour_stroke)

    def _on_contour_stroke_moved(self, plane_name, u, v):
        if plane_name != "axial" or not self._contour_stroke:
            return
        point = np.asarray([float(u), float(v)], dtype=float)
        if np.linalg.norm(point - np.asarray(self._contour_stroke[-1])) < self._contour_spacing() * 0.25:
            return
        self._contour_stroke.append((float(u), float(v)))
        # Keep drag-time feedback lightweight. Show the live stroke as the
        # green preview; exact boundary intersection, smoothing, and
        # validation run once when the stroke is released.
        self.slice_views["axial"].set_contour_draft(
            self._contour_stroke,
            self._contour_stroke if len(self._contour_stroke) >= 3 else None,
        )

    def _on_contour_stroke_finished(self, plane_name):
        if plane_name != "axial" or not self._contour_stroke:
            return
        stroke = np.asarray(self._contour_stroke, dtype=float)
        self._contour_stroke = []
        contour, message = self._build_contour_edit(stroke, validate=True)
        self.slice_views["axial"].clear_contour_draft()
        if contour is None:
            self._edit_boundary = None
            self._edit_boundary_ready = False
            return
        self._commit_contour(contour, message)
        self._edit_boundary = None
        self._edit_boundary_ready = False

    def _current_edit_boundary(self):
        plane_idx = self._selected_plane_idx
        if plane_idx is None or not (0 <= int(plane_idx) < len(self.workspace.planes)):
            return None
        plane = self.workspace.planes[int(plane_idx)]
        frame = int(self.workspace.current_t)
        operations = getattr(plane, "roi_edit_operations", {}) or {}
        frame_ops = list(operations.get(str(frame), []) or [])
        for operation in reversed(frame_ops):
            if str(operation.get("mode", "")) == "replace":
                polygon = np.asarray(operation.get("polygon", []), dtype=float)
                if polygon.ndim == 2 and polygon.shape[1] == 2 and len(polygon) >= 3:
                    return polygon
        contours = self._selected_plane_contours(
            plane,
            frame,
            self._plane_orthogonal_geometry(plane, frame)[1],
            self._contour_spacing(),
        )
        candidates = [item[0] for item in contours if item[1] == "#00f5d4"]
        if not candidates:
            return None
        return max(candidates, key=lambda item: abs(polygon_area(item)))

    def _build_contour_edit(self, stroke, *, validate):
        boundary = self._edit_boundary
        if not self._edit_boundary_ready and self._selected_plane_idx is not None:
            boundary = self._current_edit_boundary()
        spacing = self._contour_spacing()
        if boundary is None:
            return new_contour_from_stroke(stroke, spacing, validate=validate)
        snap_distance = self._contour_snap_distance_mm
        if snap_distance <= 0.0:
            snap_distance = max(2.5 * spacing, 1.5)
        return replace_contour_segment(
            boundary,
            stroke,
            spacing,
            snap_distance=snap_distance,
            validate=validate,
        )

    def _commit_contour(self, contour, message):
        plane_idx = self._selected_plane_idx
        if plane_idx is None or not (0 <= int(plane_idx) < len(self.workspace.planes)):
            return
        plane = self.workspace.planes[int(plane_idx)]
        frame_key = str(int(self.workspace.current_t))
        # Keep the already-rendered contour list so the immediate post-edit
        # display can replace one polyline without rebuilding a 3-D mask
        # intersection. A normal refresh later still recomputes from data.
        # ``_edit_boundary`` is already the contour used to construct the
        # replacement stroke. Reusing it avoids another 3-D segmentation
        # intersection just to capture a display snapshot at mouse release.
        boundary = np.asarray(self._edit_boundary, dtype=float).reshape(-1, 2) if self._edit_boundary is not None else np.empty((0, 2))
        display_contours = []
        if len(boundary) >= 3:
            display_contours.append((boundary, "#00f5d4", 1.6, False))
        legacy_polygon = np.asarray(getattr(plane, "roi_polygon_uv_mm", []) or [], dtype=float)
        if legacy_polygon.ndim == 2 and legacy_polygon.shape[1] == 2 and len(legacy_polygon) >= 3:
            display_contours.append((np.vstack((legacy_polygon, legacy_polygon[0])), "#ffd43b", 1.2, True))
        self._contour_display_override = (
            int(plane_idx),
            int(self.workspace.current_t),
            display_contours,
            np.asarray(contour, dtype=float).copy(),
        )
        before = self._snapshot_plane_roi(plane)
        operations = {
            str(key): [dict(item) for item in value]
            for key, value in dict(getattr(plane, "roi_edit_operations", {}) or {}).items()
        }
        operations[frame_key] = [{"mode": "replace", "polygon": np.asarray(contour, dtype=float).tolist()}]
        plane.roi_edit_operations = operations
        self._plane_roi_undo.append((int(plane_idx), before))
        self._plane_roi_redo.clear()
        self._cache.pop("plane_region", None)
        self._cache.pop("plane_content", None)
        self._cache.pop("plane_contours", None)
        self.planeRoiChanged.emit(int(plane_idx))

    def _snapshot_plane_roi(self, plane):
        return {
            "roi_polygon_uv_mm": list(getattr(plane, "roi_polygon_uv_mm", []) or []),
            "roi_edit_operations": {
                str(k): [dict(item) for item in value]
                for k, value in dict(getattr(plane, "roi_edit_operations", {}) or {}).items()
            },
        }

    def _restore_plane_roi(self, plane_idx, snapshot):
        plane = self.workspace.planes[int(plane_idx)]
        plane.roi_polygon_uv_mm = list(snapshot.get("roi_polygon_uv_mm", []) or [])
        plane.roi_edit_operations = {
            str(k): [dict(item) for item in value]
            for k, value in dict(snapshot.get("roi_edit_operations", {}) or {}).items()
        }
        self._cache.pop("plane_region", None); self._cache.pop("plane_content", None); self._cache.pop("plane_contours", None)
        self._contour_display_override = None
        self.planeRoiChanged.emit(int(plane_idx))

    def _undo_plane_roi(self):
        if not self._plane_roi_undo:
            return
        plane_idx, snapshot = self._plane_roi_undo.pop()
        plane = self.workspace.planes[int(plane_idx)]
        self._plane_roi_redo.append((plane_idx, self._snapshot_plane_roi(plane)))
        self._restore_plane_roi(plane_idx, snapshot)

    def _redo_plane_roi(self):
        if not self._plane_roi_redo:
            return
        plane_idx, snapshot = self._plane_roi_redo.pop()
        plane = self.workspace.planes[int(plane_idx)]
        self._plane_roi_undo.append((plane_idx, self._snapshot_plane_roi(plane)))
        self._restore_plane_roi(plane_idx, snapshot)

    def _reset_plane_roi(self):
        plane_idx = self._selected_plane_idx
        if plane_idx is None or not (0 <= int(plane_idx) < len(self.workspace.planes)):
            return
        self._stop_plane_roi_editor()
        plane = self.workspace.planes[int(plane_idx)]
        frame_key = str(int(self.workspace.current_t))
        operations = dict(getattr(plane, "roi_edit_operations", {}) or {})
        if not operations.get(frame_key):
            return
        before = self._snapshot_plane_roi(plane)
        operations.pop(frame_key, None)
        plane.roi_edit_operations = operations
        self._plane_roi_undo.append((int(plane_idx), before)); self._plane_roi_redo.clear()
        self._cache.pop("plane_region", None)
        self._cache.pop("plane_content", None)
        self._cache.pop("plane_contours", None)
        self._contour_display_override = None
        self.planeRoiChanged.emit(int(plane_idx))

    def _on_overlay_opacity_changed(self, value):
        opacity = float(np.clip(float(value) / 100.0, 0.0, 1.0))
        self.workspace.segmentation.opacity = opacity
        self.label_overlay_opacity.setText(f"{int(value)}%")
        for view in self.slice_views.values():
            view.overlay_item.setOpacity(opacity)
        # Keep the 3-D segmentation surface in sync with the 2-D overlay.
        for obj in self.workspace.scene_objects.values():
            if str(getattr(obj, "data_key", "")) == "segmask_raw_surface":
                obj.opacity = opacity
                scene = getattr(self.parent(), "scene", None)
                if scene is not None:
                    scene.apply_object_properties(obj, render=False, refresh_scalar_bar=False)
        self.refresh(update_plane=False)

    def _on_noise_overlay_toggled(self, checked):
        self._noise_overlay_visible = bool(checked)
        if checked and self._noise_overlay_opacity <= 0.0:
            self._noise_overlay_opacity = 0.35
        self._slice_keys.clear()
        self.refresh(update_plane=False)
        self._apply_noise_overlay_opacity()

    def _on_noise_overlay_opacity_changed(self, value):
        self._noise_overlay_opacity = float(np.clip(float(value) / 100.0, 0.0, 1.0))
        self.btn_noise_overlay.setChecked(value > 0)
        self.label_noise_overlay_opacity.setText(f"{int(value)}%")
        self._apply_noise_overlay_opacity()

    def _apply_noise_overlay_opacity(self):
        for view in self.slice_views.values():
            view.noise_overlay_item.setOpacity(self._noise_overlay_opacity)

    def _sync_overlay_opacity_control(self):
        value = int(round(float(np.clip(getattr(self.workspace.segmentation, "opacity", 0.35), 0.0, 1.0)) * 100.0))
        self.slider_overlay_opacity.blockSignals(True)
        self.slider_overlay_opacity.setValue(value)
        self.slider_overlay_opacity.blockSignals(False)
        self.label_overlay_opacity.setText(f"{value}%")

    def _sync_noise_overlay_control(self):
        available = self._get_noise_region_3d() is not None
        self.btn_noise_overlay.blockSignals(True)
        self.btn_noise_overlay.setEnabled(available)
        if not available:
            self.btn_noise_overlay.setChecked(False)
            self._noise_overlay_visible = False
        self.btn_noise_overlay.blockSignals(False)
        self.slider_noise_overlay_opacity.setEnabled(available)
        self.slider_noise_overlay_opacity.blockSignals(True)
        value = int(round(self._noise_overlay_opacity * 100.0)) if self._noise_overlay_visible else 0
        self.slider_noise_overlay_opacity.setValue(value)
        self.slider_noise_overlay_opacity.blockSignals(False)
        self.label_noise_overlay_opacity.setText(f"{value}%")

    def _get_noise_region_3d(self):
        mask = getattr(self.workspace, "pcmra_render_mask", None)
        if mask is None:
            return None
        region = np.asarray(mask, dtype=bool)
        if region.ndim == 4:
            frame_index = min(max(0, int(self.workspace.current_t)), region.shape[3] - 1)
            region = region[..., frame_index]
        if region.ndim != 3:
            return None
        shape = self._get_volume_shape()
        if shape is None or tuple(region.shape) != tuple(shape):
            return None
        return ~region

    def _noise_region_overlay(self, region_2d):
        if not self._noise_overlay_visible or region_2d is None:
            return None
        region = np.asarray(region_2d, dtype=bool)
        rgba = np.zeros((*region.shape, 4), dtype=np.uint8)
        rgba[region] = (255, 0, 0, 255)
        return rgba

    def _available_content_keys(self):
        """Only expose acquired data and derived fields that actually exist."""
        ws = self.workspace
        available = set()
        if ws.flow_raw is not None:
            available.update((0, 1, 2, 5))
        if ws.mag_raw is not None:
            available.add(3)
        if ws.pcmra_array is not None:
            available.add(4)
        derived = ws.derived
        if derived.wss_volume is not None or any(surf is not None for surf in derived.wss_surfaces):
            available.add(6)
        if derived.tke_array is not None or derived.tke_volume is not None:
            available.add(7)
        if derived.pressure_gradient_array is not None:
            available.update(range(8, 12))
        for key, field in ((12, "relative_pressure_array"), (13, "vorticity_magnitude"),
                           (14, "q_criterion_array"), (15, "swirling_strength_array")):
            if getattr(derived, field) is not None:
                available.add(key)
        for first, field in ((16, "correction_raw"), (19, "correction_high_raw")):
            value = getattr(ws, field, None)
            if value is not None and np.asarray(value).ndim == 5 and np.asarray(value).shape[-1] == 3:
                available.update(range(first, first + 3))
        unwrap = ws.phase_unwrap_result or {}
        if unwrap.get("wrap_mask") is not None:
            available.add(22)
        if unwrap.get("wrap_count") is not None:
            available.update((23, 24, 25))
        if unwrap.get("flow_unwrapped") is not None and unwrap.get("flow_wrapped") is not None:
            available.add(26)
        idx = self._selected_plane_idx
        if ws.flow_raw is not None and idx is not None and 0 <= int(idx) < len(ws.planes):
            available.add(27)
        return sorted(available)

    def _refresh_content_choices(self):
        # Stable item data identifies a field after insertions/removals; row
        # indices must never be used as field IDs in this filtered menu.
        available = self._available_content_keys()
        combo = self.combo_content
        existing = [combo.itemData(index) for index in range(combo.count())]
        if existing == available:
            combo.setEnabled(bool(available))
            return
        selected = combo.currentData()
        if selected not in available:
            selected = 3 if 3 in available else (available[0] if available else None)
        combo.blockSignals(True)
        combo.clear()
        for key in available:
            combo.addItem(self._content_labels[key], key)
        combo.setCurrentIndex(combo.findData(selected) if selected is not None else -1)
        combo.setEnabled(bool(available))
        combo.blockSignals(False)
        self._manual_levels = None
        self._slice_keys.clear()

    def _reset_views(self):
        self._manual_levels = None
        for view in self.slice_views.values():
            view.reset_view()
        self.refresh(update_plane=False)

    def _toggle_maximized_view(self, plane):
        if self._maximized_view == plane:
            self._maximized_view = None
            for view in self.slice_views.values():
                view.show()
            self.colorbar_widget.setVisible(self._current_volume is not None)
            return
        self._maximized_view = plane
        for name, view in self.slice_views.items():
            view.setVisible(name == plane)
        self.colorbar_widget.hide()

    def set_selected_plane(self, idx):
        self._stop_plane_roi_editor()
        self._selected_plane_idx = idx
        has_plane = idx is not None and 0 <= int(idx) < len(self.workspace.planes)
        self.btn_edit_contour.setEnabled(has_plane)
        self.contour_controls.setVisible(has_plane)
        self._refresh_content_choices()
        self._slice_keys.clear()
        for view in self.slice_views.values():
            view.plane_line.hide()
        self._move_to_plane_center(idx)
        self.refresh()

    def _move_to_plane_center(self, idx):
        ws = self.workspace
        if idx is None or idx >= len(ws.planes):
            return
        plane = ws.planes[idx]
        res = self._get_resolution()
        center_vox = np.asarray(plane.center, dtype=float) / (res + 1e-12)
        shape = self._get_volume_shape()
        if shape is None:
            return
        self._set_cursor(
            int(np.clip(np.round(center_vox[0]), 0, shape[0] - 1)),
            int(np.clip(np.round(center_vox[1]), 0, shape[1] - 1)),
            int(np.clip(np.round(center_vox[2]), 0, shape[2] - 1)),
        )

    def _get_resolution(self):
        ws = self.workspace
        if ws.resolution is not None and len(ws.resolution) >= 3:
            r = np.asarray(ws.resolution, dtype=float).reshape(-1)[:3]
            return np.where(r > 0, r, 1.0)
        return np.array([1.0, 1.0, 1.0])

    def _scene_style(self, data_key, default_cmap, default_clim=None):
        if data_key in METRIC_DATA_KEYS:
            style = configured_metric_style(self.workspace, data_key)
            default_cmap = style["cmap"]
            default_clim = style.get("clim") or default_clim
        for obj in self.workspace.scene_objects.values():
            if obj.data_key == data_key:
                return obj.cmap or default_cmap, obj.clim if obj.clim else default_clim
        return default_cmap, default_clim

    def _get_wss_volume(self, t):
        ws = self.workspace
        if ws.derived.wss_volume is not None:
            cmap, clim = self._scene_style("wss_surface_live", METRIC_CMAPS["wss_surface_live"], None)
            tidx = min(max(0, int(t)), ws.derived.wss_volume.shape[3] - 1)
            return np.asarray(ws.derived.wss_volume[..., tidx], dtype=np.float32), "WSS (Pa)", {"cmap": cmap, "clim": clim}
        if not ws.derived.wss_surfaces:
            return None, "WSS (no data)", {"cmap": METRIC_CMAPS["wss_surface_live"], "clim": None}
        tidx = min(max(0, t), len(ws.derived.wss_surfaces) - 1)
        surf = ws.derived.wss_surfaces[tidx]
        if surf is None or "wss" not in surf.point_data:
            return None, "WSS (no data)", {"cmap": METRIC_CMAPS["wss_surface_live"], "clim": None}
        shape = self._get_volume_shape()
        if shape is None:
            return None, "WSS (no data)", {"cmap": METRIC_CMAPS["wss_surface_live"], "clim": None}
        res = self._get_resolution()
        key = (
            id(surf),
            tuple(int(x) for x in shape),
            tuple(np.round(res, 6).tolist()),
        )
        def _build():
            vol = np.zeros(shape, dtype=float)
            pts = np.asarray(surf.points, dtype=float)
            vals = np.asarray(surf.point_data["wss"], dtype=float)
            vox = np.rint(pts / (res.reshape(1, 3) + 1e-12)).astype(int)
            for k in range(3):
                vox[:, k] = np.clip(vox[:, k], 0, shape[k] - 1)
            flat = np.ravel_multi_index((vox[:, 0], vox[:, 1], vox[:, 2]), shape)
            tgt = vol.reshape(-1)
            np.maximum.at(tgt, flat, vals)
            return vol
        vol = self._cached("wss_volume", key, _build)
        cmap, clim = self._scene_style("wss_surface_live", METRIC_CMAPS["wss_surface_live"], None)
        return vol, "WSS (Pa)", {"cmap": cmap, "clim": clim}

    def _get_tke_volume(self, t):
        ws = self.workspace
        if ws.derived.tke_array is not None:
            arr = np.asarray(ws.derived.tke_array, dtype=np.float32)
            tidx = min(max(0, int(t)), arr.shape[3] - 1) if arr.ndim == 4 else 0
            mask_id = id(ws.segmask_binary) if ws.segmask_binary is not None else -1
            key = (id(ws.derived.tke_array), mask_id, int(tidx))
            def _build():
                if arr.ndim == 4:
                    vol = arr[..., tidx]
                else:
                    vol = arr
                if ws.segmask_binary is not None:
                    if ws.segmask_binary.ndim == 4:
                        mask_t = ws.segmask_binary[..., min(max(0, int(t)), ws.segmask_binary.shape[3] - 1)]
                    else:
                        mask_t = ws.segmask_binary
                    vol = np.asarray(vol, dtype=np.float32) * np.asarray(mask_t, dtype=np.float32)
                return np.asarray(vol, dtype=np.float32)
            vol = self._cached("tke_volume", key, _build)
            cmap, clim = self._scene_style("tke_volume", METRIC_CMAPS["tke_volume"], None)
            return vol, "TKE (J/m³)", {"cmap": cmap, "clim": clim}
        if ws.derived.tke_volume is None:
            return None, "TKE (no data)", {"cmap": METRIC_CMAPS["tke_volume"], "clim": None}
        shape = self._get_volume_shape()
        if shape is None:
            return None, "TKE (no data)", {"cmap": METRIC_CMAPS["tke_volume"], "clim": None}
        tke_mesh = ws.derived.tke_volume
        if "TKE" not in tke_mesh.point_data and "TKE" not in tke_mesh.cell_data:
            return None, "TKE (no data)", {"cmap": METRIC_CMAPS["tke_volume"], "clim": None}
        res = self._get_resolution()
        key = (
            id(tke_mesh),
            tuple(int(x) for x in shape),
            tuple(np.round(res, 6).tolist()),
        )
        def _build():
            vol = np.zeros(shape, dtype=float)
            if "TKE" in tke_mesh.cell_data:
                pts = tke_mesh.cell_centers().points
                vals = np.asarray(tke_mesh.cell_data["TKE"], dtype=float)
            else:
                pts = tke_mesh.points
                vals = np.asarray(tke_mesh.point_data["TKE"], dtype=float)
            vox = np.rint(pts / (res.reshape(1, 3) + 1e-12)).astype(int)
            for k in range(3):
                vox[:, k] = np.clip(vox[:, k], 0, shape[k] - 1)
            flat = np.ravel_multi_index((vox[:, 0], vox[:, 1], vox[:, 2]), shape)
            tgt = vol.reshape(-1)
            np.maximum.at(tgt, flat, vals)
            return vol
        vol = self._cached("tke_mesh_volume", key, _build)
        cmap, clim = self._scene_style("tke_volume", METRIC_CMAPS["tke_volume"], None)
        return vol, "TKE (J/m³)", {"cmap": cmap, "clim": clim}
    def _get_relative_pressure_volume(self, t):
        ws = self.workspace
        if ws.derived.relative_pressure_array is None:
            return None, "Relative Pressure (no data)", {"cmap": "RdBu_r", "clim": None}
        arr = np.asarray(ws.derived.relative_pressure_array, dtype=np.float32)
        tidx = min(max(0, int(t)), arr.shape[3] - 1)
        mask_id = id(ws.segmask_binary) if ws.segmask_binary is not None else -1
        key = (id(ws.derived.relative_pressure_array), mask_id, int(tidx))
        def _build():
            vol_t = np.asarray(arr[..., tidx], dtype=np.float32)
            if ws.segmask_binary is not None:
                mask_t = ws.segmask_binary[..., tidx] if ws.segmask_binary.ndim == 4 else ws.segmask_binary
                vol_t = np.where(np.asarray(mask_t, dtype=bool), vol_t, np.float32(0.0))
            return np.asarray(vol_t, dtype=np.float32)
        vol = self._cached("scalar_volume", ("relative_pressure",) + key, _build)
        cmap, clim = self._scene_style("relative_pressure_volume", "RdBu_r", None)
        if clim is None and ws.derived.relative_pressure_display_clim is not None:
            clim = tuple(ws.derived.relative_pressure_display_clim)
        if clim is None:
            finite = vol[np.isfinite(vol)]
            vmax = float(np.percentile(np.abs(finite), 99.0)) if finite.size else 1.0
            vmax = max(vmax, 1.0)
            clim = (-vmax, vmax)
        return vol, "Relative Pressure (Pa)", {"cmap": cmap, "clim": clim}

    def _get_pressure_gradient_volume(self, t, component=None):
        ws = self.workspace
        if ws.derived.pressure_gradient_array is None:
            return None, "Pressure Gradient (no data)", {"cmap": "magma", "clim": None}
        arr = np.asarray(ws.derived.pressure_gradient_array, dtype=np.float32)
        support_source = ws.derived.pressure_gradient_support_mask
        support = None if support_source is None else np.asarray(support_source, dtype=bool)
        support_id = id(support_source) if support_source is not None else -1
        tidx = min(max(0, int(t)), arr.shape[3] - 1)
        if component is None:
            if ws.derived.pressure_gradient_magnitude is None:
                vol = np.sqrt(np.sum(arr[..., tidx, :] ** 2, axis=-1))
            else:
                vol = np.asarray(ws.derived.pressure_gradient_magnitude[..., tidx], dtype=np.float32)
            if support is not None:
                key = (id(ws.derived.pressure_gradient_magnitude), support_id, int(tidx))
                vol = self._cached(
                    "scalar_volume",
                    ("pressure_gradient_magnitude",) + key,
                    lambda: np.where(support[..., tidx], vol, np.float32(0.0)).astype(np.float32, copy=False),
                )
            cmap, clim = self._scene_style("pressure_gradient_volume", "magma", None)
            if clim is None and ws.derived.pressure_gradient_display_clim is not None:
                clim = tuple(ws.derived.pressure_gradient_display_clim)
            return np.asarray(vol, dtype=np.float32), "|Pressure Grad| (Pa/m)", {"cmap": cmap, "clim": clim}
        vol = np.asarray(arr[..., tidx, int(component)], dtype=np.float32)
        if support is not None:
            key = (id(ws.derived.pressure_gradient_array), support_id, int(tidx), int(component))
            vol = self._cached(
                "scalar_volume",
                ("pressure_gradient_component",) + key,
                lambda: np.where(support[..., tidx], vol, np.float32(0.0)).astype(np.float32, copy=False),
            )
            finite = vol[support[..., tidx] & np.isfinite(vol)]
        else:
            finite = vol[np.isfinite(vol)]
        vmax = float(np.percentile(np.abs(finite), 99.0)) if finite.size else 1e-6
        vmax = max(vmax, 1e-6)
        direction = ("LR", "AP", "FH")[int(component)]
        return vol, f"Pressure Grad {direction} (Pa/m)", {"cmap": "RdBu_r", "clim": (-vmax, vmax)}

    def _get_vortex_volume(self, t, field):
        ws = self.workspace
        labels = {
            "vorticity_magnitude": ("Vorticity Magnitude (s⁻¹)", "turbo", False),
            "q_criterion": ("Q-Criterion (s⁻²)", "RdBu_r", True),
            "swirling_strength": ("Swirling Strength λci (s⁻¹)", "turbo", False),
        }
        title, cmap, signed = labels[field]
        source_name = {
            "vorticity_magnitude": "vorticity_magnitude",
            "q_criterion": "q_criterion_array",
            "swirling_strength": "swirling_strength_array",
        }[field]
        source = getattr(ws.derived, source_name, None)
        if source is None:
            return None, f"{title} (no data)", {"cmap": cmap, "clim": None}
        arr = np.asarray(source, dtype=np.float32)
        if arr.ndim != 4:
            return None, f"{title} (no data)", {"cmap": cmap, "clim": None}
        tidx = min(max(0, int(t)), arr.shape[3] - 1)
        support_source = ws.derived.vortex_support_mask
        support = None if support_source is None else np.asarray(support_source, dtype=bool)
        if support is not None and support.shape == arr.shape:
            key = (id(source), id(support_source), int(tidx))
            vol = self._cached(
                "scalar_volume",
                ("vortex", field) + key,
                lambda: np.where(support[..., tidx], arr[..., tidx], np.float32(0.0)).astype(np.float32, copy=False),
            )
            finite = vol[support[..., tidx] & np.isfinite(vol)]
        else:
            vol = np.asarray(arr[..., tidx], dtype=np.float32)
            finite = vol[np.isfinite(vol)]
        upper = float(np.percentile(np.abs(finite), 99.0)) if finite.size else 1.0
        upper = max(upper, 1e-6)
        clim = (-upper, upper) if signed else (0.0, upper)
        return np.asarray(vol, dtype=np.float32), title, {"cmap": cmap, "clim": clim}

    def _get_scalar_slice(self, t):
        ws = self.workspace
        content_idx = self.combo_content.currentData()
        if content_idx == 0 and ws.flow_raw is not None:
            vol = np.asarray(ws.flow_raw[..., t, 0], dtype=np.float32)
            vmax = self._cached("scalar_clim", ("flow", id(ws.flow_raw), int(t), 0), lambda: max(abs(float(np.nanmin(vol))), abs(float(np.nanmax(vol))), 1e-6))
            return vol, "Flow LR (cm/s)", {"cmap": "RdBu_r", "clim": (-vmax, vmax)}
        if content_idx == 1 and ws.flow_raw is not None:
            vol = np.asarray(ws.flow_raw[..., t, 1], dtype=np.float32)
            vmax = self._cached("scalar_clim", ("flow", id(ws.flow_raw), int(t), 1), lambda: max(abs(float(np.nanmin(vol))), abs(float(np.nanmax(vol))), 1e-6))
            return vol, "Flow AP (cm/s)", {"cmap": "RdBu_r", "clim": (-vmax, vmax)}
        if content_idx == 2 and ws.flow_raw is not None:
            vol = np.asarray(ws.flow_raw[..., t, 2], dtype=np.float32)
            vmax = self._cached("scalar_clim", ("flow", id(ws.flow_raw), int(t), 2), lambda: max(abs(float(np.nanmin(vol))), abs(float(np.nanmax(vol))), 1e-6))
            return vol, "Flow FH (cm/s)", {"cmap": "RdBu_r", "clim": (-vmax, vmax)}
        if content_idx == 3 and ws.mag_raw is not None:
            vol = np.asarray(ws.mag_raw[..., t], dtype=np.float32)
            clim = self._cached("scalar_clim", ("magnitude", id(ws.mag_raw), int(t)), lambda: (float(np.nanmin(vol)), float(np.nanmax(vol))))
            return vol, "Magnitude", {"cmap": "gray", "clim": clim}
        if content_idx == 4 and ws.pcmra_array is not None:
            pcmra = np.asarray(ws.pcmra_array)
            tidx = min(max(0, int(t)), pcmra.shape[3] - 1)
            vol = np.asarray(pcmra[..., tidx], dtype=np.float32)
            key = (id(ws.pcmra_array), tidx)
            clim = self._cached("scalar_clim", ("pcmra",) + key, lambda: (float(np.nanmin(vol)), float(np.nanmax(vol))))
            return vol, "PC-MRA", {"cmap": "gray", "clim": clim}
        if content_idx == 5 and ws.flow_raw is not None:
            key = (id(ws.flow_raw), int(t))
            vol = self._cached(
                "scalar_volume",
                ("speed",) + key,
                lambda: np.sqrt(np.sum(np.square(np.asarray(ws.flow_raw[..., t, :], dtype=np.float32), dtype=np.float32), axis=-1)),
            )
            vmax = self._cached("scalar_clim", ("speed",) + key, lambda: max(float(np.nanmax(vol)), 1.0))
            return np.asarray(vol, dtype=np.float32), "Speed (cm/s)", {"cmap": "turbo", "clim": (0.0, vmax)}
        if content_idx == 6:
            return self._get_wss_volume(t)
        if content_idx == 7:
            return self._get_tke_volume(t)
        if content_idx == 8:
            return self._get_pressure_gradient_volume(t, component=0)
        if content_idx == 9:
            return self._get_pressure_gradient_volume(t, component=1)
        if content_idx == 10:
            return self._get_pressure_gradient_volume(t, component=2)
        if content_idx == 11:
            return self._get_pressure_gradient_volume(t, component=None)
        if content_idx == 12:
            return self._get_relative_pressure_volume(t)
        if content_idx == 13:
            return self._get_vortex_volume(t, "vorticity_magnitude")
        if content_idx == 14:
            return self._get_vortex_volume(t, "q_criterion")
        if content_idx == 15:
            return self._get_vortex_volume(t, "swirling_strength")
        if content_idx in range(16, 22):
            is_high = content_idx >= 19
            corr = getattr(ws, "correction_high_raw" if is_high else "correction_raw", None)
            corr = None if corr is None else np.asarray(corr)
            if corr is not None and corr.ndim == 5 and corr.shape[-1] == 3:
                component = int(content_idx - (19 if is_high else 16))
                tidx = min(int(t), corr.shape[3] - 1)
                vol = np.asarray(corr[..., tidx, component], dtype=np.float32)
                vmax = self._cached(
                    "scalar_clim",
                    ("corr", id(corr), tidx, component),
                    lambda: max(
                        float(np.percentile(np.abs(vol[np.isfinite(vol)]), 99.0))
                        if np.isfinite(vol).any() else 0.0,
                        1e-6,
                    ),
                )
                direction = ("LR", "AP", "FH")[component]
                venc_name = "High" if is_high else "Low"
                return vol, f"Corr {venc_name} {direction} (rad)", {"cmap": "RdBu_r", "clim": (-vmax, vmax)}
        if content_idx == 22:
            result = getattr(ws, "phase_unwrap_result", {}) or {}
            mask = result.get("wrap_mask")
            if mask is None:
                return None, "Wrap Mask (no unwrapping)", {"cmap": "gray", "clim": (0, 1)}
            arr = np.asarray(mask)
            tidx = min(int(t), arr.shape[3] - 1)
            vol = self._cached(
                "scalar_volume", ("wrap_mask", id(mask), tidx),
                lambda: np.any(arr[..., tidx, :], axis=-1).astype(np.float32),
            )
            return vol, "Wrap Mask (any component)", {"cmap": "Reds", "clim": (0, 1)}
        if content_idx in {23, 24, 25}:
            result = getattr(ws, "phase_unwrap_result", {}) or {}
            count = result.get("wrap_count")
            comp = content_idx - 23
            if count is None:
                return None, "Wrap Count (no unwrapping)", {"cmap": "RdBu_r", "clim": (-1, 1)}
            arr = np.asarray(count)
            tidx = min(int(t), arr.shape[3] - 1)
            vol = arr[..., tidx, comp].astype(np.float32)
            lim = self._cached(
                "scalar_clim", ("wrap_count", id(count), tidx, comp),
                lambda: max(1.0, float(np.max(np.abs(vol)))),
            )
            direction = ("LR", "AP", "FH")[comp]
            return vol, f"Wrap Count {direction}", {"cmap": "RdBu_r", "clim": (-lim, lim)}
        if content_idx == 26:
            result = getattr(ws, "phase_unwrap_result", {}) or {}
            flow_u = result.get("flow_unwrapped")
            flow_w = result.get("flow_wrapped")
            if flow_u is None or flow_w is None:
                return None, "Unwrapped − Wrapped Speed (no unwrapping)", {"cmap": "gray", "clim": None}
            au, aw = np.asarray(flow_u), np.asarray(flow_w)
            tidx = min(int(t), au.shape[3] - 1)
            vol = self._cached(
                "scalar_volume", ("unwrap_delta", id(flow_u), id(flow_w), tidx),
                lambda: (
                    np.linalg.norm(au[..., tidx, :], axis=-1)
                    - np.linalg.norm(aw[..., tidx, :], axis=-1)
                ).astype(np.float32),
            )
            lim = self._cached(
                "scalar_clim", ("unwrap_delta", id(flow_u), id(flow_w), tidx),
                lambda: max(1e-6, float(np.nanmax(np.abs(vol)))),
            )
            return vol, "Unwrapped − Wrapped Speed (cm/s)", {"cmap": "RdBu_r", "clim": (-lim, lim)}
        if content_idx == 27 and ws.flow_raw is not None:
            plane_idx = self._selected_plane_idx
            if plane_idx is None or not (0 <= int(plane_idx) < len(ws.planes)):
                return None, "Through-plane Flow (select a plane)", {"cmap": "RdBu_r", "clim": None}
            plane = ws.planes[int(plane_idx)]
            normal = np.asarray(plane.normal, dtype=np.float32).reshape(3)
            normal /= np.linalg.norm(normal) + 1e-12
            key = (
                id(ws.flow_raw),
                int(t),
                int(plane_idx),
                tuple(np.round(normal, 7).tolist()),
            )
            vol = self._cached(
                "scalar_volume",
                ("through_plane",) + key,
                lambda: np.einsum(
                    "...c,c->...",
                    np.asarray(ws.flow_raw[..., t, :], dtype=np.float32),
                    normal,
                    dtype=np.float32,
                ).astype(np.float32, copy=False),
            )
            vmax = self._cached(
                "scalar_clim",
                ("through_plane",) + key,
                lambda: max(float(np.nanmax(np.abs(vol))), 1e-6),
            )
            return vol, "Through-plane Flow (cm/s)", {"cmap": "RdBu_r", "clim": (-vmax, vmax)}
        return None, "", {"cmap": "gray", "clim": None}

    def _display_labels_source(self):
        """Return active display labels without materialising 3-D repeats."""
        ws = self.workspace
        segmentation = getattr(ws, "segmentation", None)
        working_4d = getattr(segmentation, "working_labels_4d", None)
        if working_4d is not None:
            return np.asarray(working_4d, dtype=np.int16)
        working_3d = getattr(segmentation, "working_labels_3d", None)
        if working_3d is not None:
            return np.asarray(working_3d, dtype=np.int16)
        getter = getattr(ws, "get_active_segmentation", None)
        active = getter() if callable(getter) else getattr(ws, "segmask_raw", None)
        return None if active is None else np.asarray(active, dtype=np.int16)

    def _display_labels_for_frame(self, frame_index):
        display = self._display_labels_source()
        if display is None:
            return None
        if display.ndim == 4:
            index = min(max(0, int(frame_index)), display.shape[3] - 1)
            return display[..., index]
        return display

    def _get_mask_3d(self):
        ws = self.workspace
        display = self._display_labels_for_frame(ws.current_t)
        if display is not None:
            return display
        if ws.segmask_3d is not None:
            return ws.segmask_3d
        if ws.segmask_binary is not None and ws.segmask_binary.ndim == 4:
            key = (id(ws.segmask_binary), tuple(int(x) for x in ws.segmask_binary.shape))
            return self._cached("mask_3d", key, lambda: np.any(ws.segmask_binary, axis=3))
        return None

    def _label_color(self, label_id):
        palette = [
            "#ff6b6b", "#4dabf7", "#51cf66", "#ffd43b", "#f783ac",
            "#74c0fc", "#63e6be", "#ffa94d", "#b197fc", "#a9e34b",
        ]
        color = self.workspace.segmentation.label_colors.get(str(int(label_id)), "")
        if color:
            return color
        return palette[(max(1, int(label_id)) - 1) % len(palette)]

    def _label_overlay(self, labels_2d):
        seg_state = self.workspace.segmentation
        if not seg_state.visible or labels_2d is None:
            return None
        colors = {
            int(value): self._label_color(int(value))
            for value in np.unique(labels_2d)
            if int(value) > 0
        }
        return make_label_overlay(labels_2d, colors, seg_state.opacity)

    def _update_value_label(self, vol, title):
        if vol is None:
            self.label_value.setText("Voxel: -   Value: -")
            return
        x, y, z = self.slider_x.value(), self.slider_y.value(), self.slider_z.value()
        try:
            val = float(vol[x, y, z])
            self.label_value.setText(f"Voxel (LR, AP, FH): ({x}, {y}, {z})   {title}: {val:.6g}")
        except Exception:
            self.label_value.setText(f"Voxel (LR, AP, FH): ({x}, {y}, {z})   {title}: -")

    def _update_plane_metric_label(self):
        ws = self.workspace
        if self._selected_plane_idx is None or self._selected_plane_idx >= len(ws.planes):
            self.label_plane_metric.setText("Plane metrics: -")
            return
        plane = ws.planes[self._selected_plane_idx]
        metrics = plane.metrics or {}
        t = int(ws.current_t)
        fr = metrics.get("flowrate_mL_s", [])
        ar = metrics.get("area_mm2", [])
        mv = metrics.get("meanv_cm_s_t", [])
        cur_fr = float(fr[t]) if t < len(fr) else 0.0
        cur_ar = float(ar[t]) if t < len(ar) else 0.0
        cur_mv = float(mv[t]) if t < len(mv) else metrics.get("meanv_cm_s", 0.0)
        path_direction = metrics.get("path_direction", "")
        path_ic = metrics.get("path_ic", None)
        txt = f"Plane {self._selected_plane_idx}"
        if path_direction:
            txt += f" [{path_direction}]"
        txt += f"   t={t} Flow Rate={cur_fr:.4g} mL/s Area={cur_ar:.4g} mm² Mean Velocity={cur_mv:.4g} cm/s Peak Velocity={metrics.get('peakv_cm_s', 0.0):.4g} cm/s"
        if path_ic is not None:
            txt += f"   Path IC={float(path_ic):.3f}"
        self.label_plane_metric.setText(txt)

    def _on_colorbar_levels_changed(self):
        if self._updating_colorbar:
            return
        levels = self.colorbar_item.levels()
        if levels is None:
            return
        self._manual_levels = tuple(float(value) for value in levels)
        for view in self.slice_views.values():
            view.image_item.setLevels(self._manual_levels)

    def _adjust_window_level(self, delta_x, delta_y):
        if self._current_volume is None:
            return
        levels = self._manual_levels
        if levels is None:
            finite = self._current_volume[np.isfinite(self._current_volume)]
            if finite.size == 0:
                return
            levels = (float(np.min(finite)), float(np.max(finite)))
        low, high = levels
        width = max(float(high - low), 1e-6)
        center = (float(low) + float(high)) / 2.0
        width *= float(np.exp(float(delta_x) * 0.012))
        center -= float(delta_y) * width * 0.004
        self._manual_levels = (center - width / 2.0, center + width / 2.0)
        self._set_colorbar(make_colormap(self._current_cmap), self._manual_levels, self._current_title)
        for view in self.slice_views.values():
            view.image_item.setLevels(self._manual_levels)
        self._update_value_label(self._current_volume, self._current_title)

    def _set_colorbar(self, colormap, levels, title):
        state = (
            str(getattr(self, "_current_cmap", "gray")),
            tuple(float(value) for value in levels),
            str(title or ""),
        )
        if state == self._colorbar_state:
            return
        self._updating_colorbar = True
        try:
            self.colorbar_item.setColorMap(colormap)
            self.colorbar_item.setLevels(values=levels, update_items=False)
            self.colorbar_item.axis.setLabel(text=str(title or ""), color="#d7e0e3")
            self._colorbar_state = state
        finally:
            self._updating_colorbar = False

    def refresh(self, update_plane=True):
        self._refresh_content_choices()
        ws = self.workspace
        # Never carry an intersection from a previous plane, time frame, or
        # slice while the new geometry is being rebuilt.
        for view in self.slice_views.values():
            view.plane_line.hide()
        self._sync_overlay_opacity_control()
        self._sync_noise_overlay_control()
        t = int(ws.current_t)
        if update_plane:
            self._slice_keys.clear()
        shape = self._get_volume_shape()
        if shape is None:
            self._current_volume = None
            self.colorbar_widget.hide()
            for view in self.slice_views.values():
                view.clear()
            return

        cx, cy, cz = self.slider_x.value(), self.slider_y.value(), self.slider_z.value()
        vol, title, style = self._get_scalar_slice(t)
        labels_3d = self._get_mask_3d()
        noise_region_3d = self._get_noise_region_3d()
        res = self._get_resolution()
        self._current_volume = vol
        self._current_title = title
        if vol is not None:
            if self._maximized_view is None:
                self.colorbar_widget.show()
            self._current_cmap = style.get("cmap", "gray")
            cmap = make_colormap(self._current_cmap)
            clim = style.get("clim", None)
            if clim is None:
                clim = (float(np.nanmin(vol)), float(np.nanmax(vol)))
            if self._manual_levels is not None:
                clim = self._manual_levels
            clim = tuple(float(value) for value in clim)
            if self._manual_levels is None:
                clim = self._cached_image_levels(vol, clim)
            self._set_colorbar(cmap, clim, title)
            selected_plane = None
            if self._selected_plane_idx is not None and 0 <= int(self._selected_plane_idx) < len(ws.planes):
                selected_plane = ws.planes[int(self._selected_plane_idx)]
            orth_center = None
            orth_fov = None
            mode = "plane" if selected_plane is not None else "axis"
            if selected_plane is not None:
                display_labels = self._display_labels_for_frame(t)
                orth_center, orth_fov = self._plane_orthogonal_geometry(selected_plane, t)
                orthogonal = self._plane_orthogonal_slices(
                    vol, display_labels, noise_region_3d, selected_plane, center=orth_center, fov_mm=orth_fov,
                    spacing_mm=max(float(np.min(res)) * 0.75, 0.25),
                )
                orth_spacing = max(float(orth_fov) / max(int(orthogonal[0][0].shape[0]) - 1, 1), 1e-3)
                orth_extent = (
                    -0.5 * float(orth_fov) - 0.5 * orth_spacing,
                    0.5 * float(orth_fov) + 0.5 * orth_spacing,
                    -0.5 * float(orth_fov) - 0.5 * orth_spacing,
                    0.5 * float(orth_fov) + 0.5 * orth_spacing,
                )
                slices = {
                    name: (
                        item[0],
                        item[1],
                        item[2],
                        (item[0].shape[0] // 2, item[0].shape[1] // 2),
                        0,
                        (orth_spacing, orth_spacing),
                        item[3],
                        orth_extent,
                        (0.0, 0.0),
                    )
                    for name, item in zip(("axial", "coronal", "sagittal"), orthogonal)
                }
            else:
                plane_centers = {
                    "axial": None,
                    "coronal": None,
                    "sagittal": None,
                }
                if selected_plane is not None:
                    plane_center = np.asarray(selected_plane.center, dtype=float).reshape(3)
                    plane_centers = {
                        "axial": (plane_center[0], plane_center[1]),
                        "coronal": (plane_center[0], plane_center[2]),
                        "sagittal": (plane_center[1], plane_center[2]),
                    }
                slices = {
                    "axial": (vol[:, :, cz], None if labels_3d is None else labels_3d[:, :, cz], None if noise_region_3d is None else noise_region_3d[:, :, cz], (cx, cy), cz, (res[0], res[1]), "Axial", None, plane_centers["axial"]),
                    "coronal": (vol[:, cy, :], None if labels_3d is None else labels_3d[:, cy, :], None if noise_region_3d is None else noise_region_3d[:, cy, :], (cx, cz), cy, (res[0], res[2]), "Coronal", None, plane_centers["coronal"]),
                    "sagittal": (vol[cx, :, :], None if labels_3d is None else labels_3d[cx, :, :], None if noise_region_3d is None else noise_region_3d[cx, :, :], (cy, cz), cx, (res[1], res[2]), "Sagittal", None, plane_centers["sagittal"]),
                }
            for plane, (image, labels, noise_region, cursor, fixed, view_spacing, view_title, extent, view_center) in slices.items():
                spec = PLANE_SPECS[plane]
                slice_key = (
                    mode, self.combo_content.currentData(), int(t), int(fixed),
                    None if selected_plane is None else tuple(np.round(np.asarray(selected_plane.center), 4)),
                    None if selected_plane is None else tuple(np.round(np.asarray(selected_plane.normal), 6)),
                    None if orth_center is None else tuple(np.round(orth_center, 4)),
                    None if orth_fov is None else round(float(orth_fov), 3),
                    id(ws.pcmra_render_mask), bool(self._noise_overlay_visible),
                )
                previous_key = self._slice_keys.get(plane)
                view_levels = clim if self._manual_levels is not None else self._cached_image_levels(image, clim)
                noise_overlay = self._noise_region_overlay(noise_region)
                if previous_key == slice_key:
                    self.slice_views[plane].update_cursor(cursor, fixed, view_levels)
                elif (
                    self._playback_active
                    and previous_key is not None
                    and previous_key[0] == slice_key[0]
                    and previous_key[2] == slice_key[2]
                ):
                    self.slice_views[plane].update_image(
                        image,
                        self._label_overlay(labels),
                        cursor,
                        fixed,
                        view_levels,
                        noise_overlay=noise_overlay,
                    )
                    self._slice_keys[plane] = slice_key
                else:
                    self.slice_views[plane].set_slice(
                        image,
                        self._label_overlay(labels),
                        view_spacing,
                        cursor,
                        fixed,
                        cmap,
                        view_levels,
                        extent=extent,
                        view_center=view_center,
                        noise_overlay=noise_overlay,
                    )
                    self._slice_keys[plane] = slice_key
                if mode == "plane" and selected_plane is not None:
                    self.slice_views[plane].set_title(view_title, "selected plane")
                    plane_axes = {
                        "axial": (("-U", "+U"), ("-V", "+V")),
                        "coronal": (("-V", "+V"), ("-N", "+N")),
                        "sagittal": (("-U", "+U"), ("-N", "+N")),
                    }
                    self.slice_views[plane].set_orientation_labels(*plane_axes[plane])
                else:
                    self.slice_views[plane].set_title(view_title, f"{spec.fixed_name} {int(fixed) + 1}")
                    self.slice_views[plane].set_orientation_labels()
                self.slice_views[plane].set_plane_intersection(
                    None if selected_plane is None or mode == "plane" else self._plane_intersection_for_slice(
                        selected_plane, spec.fixed_axis, fixed, shape, res
                    )
                )
                if not self._playback_active:
                    if mode == "plane" and plane == "axial":
                        self.slice_views[plane].set_contours(
                            self._selected_plane_contours(
                                selected_plane,
                                t,
                                orth_fov,
                                max(float(np.min(res)) * 0.75, 0.25),
                            )
                        )
                    else:
                        self.slice_views[plane].clear_contours()
        else:
            self.colorbar_widget.hide()
            self._colorbar_state = None
            for view in self.slice_views.values():
                view.clear()

        self._update_value_label(vol, title)
        self._update_plane_metric_label()

    def refresh_plane_contours(self):
        """Refresh only contour artists after an ROI edit.

        ROI edits do not change scalar images, slice geometry, cursor, or
        color limits. Rebuilding all three ImageItems adds avoidable Qt work
        to every contour commit on large 4-D volumes.
        """
        plane_idx = self._selected_plane_idx
        if plane_idx is None or not (0 <= int(plane_idx) < len(self.workspace.planes)):
            return
        try:
            plane = self.workspace.planes[int(plane_idx)]
            t = int(self.workspace.current_t)
            override = self._contour_display_override
            if override is not None and override[0] == int(plane_idx) and override[1] == t:
                display_contours = override[2]
                polygon = np.asarray(override[3], dtype=float).reshape(-1, 2)
                if display_contours is not None and len(polygon) >= 3:
                    updated = list(display_contours)
                    cyan = [i for i, item in enumerate(updated) if item[1] == "#00f5d4"]
                    if cyan:
                        replace_index = max(cyan, key=lambda i: abs(polygon_area(updated[i][0])))
                        updated[replace_index] = (polygon, "#00f5d4", 1.6, False)
                    else:
                        updated.append((polygon, "#00f5d4", 1.6, False))
                    self.slice_views["axial"].set_contours(updated)
                    self._contour_display_override = None
                    return
            fov_mm = self._plane_orthogonal_geometry(plane, t)[1]
            spacing_mm = max(float(np.min(self._get_resolution())) * 0.75, 0.25)
            contours = self._selected_plane_contours(plane, t, fov_mm, spacing_mm)
            self.slice_views["axial"].set_contours(contours)
        except Exception:
            # Presentation-only failure must not invalidate the saved ROI.
            return

    @staticmethod
    def _plane_axes(normal):
        normal = np.asarray(normal, dtype=float)
        normal = normal / (np.linalg.norm(normal) + 1e-12)
        reference = np.eye(3, dtype=float)[int(np.argmin(np.abs(normal)))]
        u = np.cross(reference, normal)
        u = u / (np.linalg.norm(u) + 1e-12)
        v = np.cross(normal, u)
        v = v / (np.linalg.norm(v) + 1e-12)
        return u, v

    def _resample_oblique(
        self,
        volume_3d,
        center_mm,
        normal,
        fov_mm,
        sample_spacing_mm,
        *,
        interpolation_order=1,
        cell_centered=False,
    ):
        spacing = self._get_resolution()
        u, v = self._plane_axes(normal)
        sample_count = int(np.clip(np.ceil(float(fov_mm) / max(float(sample_spacing_mm), 1e-3)) + 1, 129, 257))
        axis_mm = np.linspace(-0.5 * float(fov_mm), 0.5 * float(fov_mm), sample_count)
        grid_u, grid_v = np.meshgrid(axis_mm, axis_mm, indexing="ij")
        points_mm = (
            np.asarray(center_mm, dtype=float).reshape(1, 1, 3)
            + grid_u[..., None] * u.reshape(1, 1, 3)
            + grid_v[..., None] * v.reshape(1, 1, 3)
        )
        origin = np.asarray(getattr(self.workspace, "origin", np.zeros(3)), dtype=float)
        coords = (points_mm + origin.reshape(1, 1, 3)) / (spacing.reshape(1, 1, 3) + 1e-12)
        if cell_centered:
            coords -= 0.5
        sampled = map_coordinates(
            volume_3d,
            [coords[..., 0].ravel(), coords[..., 1].ravel(), coords[..., 2].ravel()],
            order=int(interpolation_order),
            mode="constant",
            cval=0.0,
        )
        return sampled.reshape(sample_count, sample_count), axis_mm, u, v

    def _effective_plane_region(self, plane, t):
        ws = self.workspace
        mask = ws.segmask_binary
        if mask is None:
            return None
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim == 4:
            mask_t = mask[..., min(max(0, int(t)), mask.shape[3] - 1)]
        elif mask.ndim == 3:
            mask_t = mask
        else:
            return None
        plane_seg_label = int(getattr(plane, "segmentation_label", 0) or 0)
        labels = getattr(ws, "segmask_labels", None)
        if plane_seg_label > 0 and labels is not None:
            labels = np.asarray(labels)
            if labels.ndim == 4:
                labels = labels[..., min(max(0, int(t)), labels.shape[3] - 1)]
            if labels.shape == mask_t.shape:
                mask_t = mask_t & (labels == plane_seg_label)
        return _build_plane_slice_region(
            mask_t,
            plane,
            ws.resolution,
            ws.origin,
            select_connected=True,
            frame_index=int(t),
        )

    def _segmentation_plane_region(self, plane, t):
        ws = self.workspace
        mask = self._display_labels_for_frame(t)
        if mask is None:
            mask = ws.segmask_binary
        if mask is None:
            return None
        mask = np.asarray(mask)
        if mask.ndim == 4:
            mask = mask[..., min(max(0, int(t)), mask.shape[3] - 1)]
        if mask.ndim != 3:
            return None
        mask = mask != 0
        plane_seg_label = int(getattr(plane, "segmentation_label", 0) or 0)
        labels = getattr(ws, "segmask_labels", None)
        if plane_seg_label > 0 and labels is not None:
            labels = np.asarray(labels)
            if labels.ndim == 4:
                labels = labels[..., min(max(0, int(t)), labels.shape[3] - 1)]
            if labels.shape == mask.shape:
                mask = mask & (labels == plane_seg_label)
        return _build_plane_slice_region(
            mask,
            plane,
            ws.resolution,
            ws.origin,
            select_connected=True,
            frame_index=int(t),
        )

    def _plane_region_coordinates(self, region, plane):
        if region is None or getattr(region, "n_points", 0) == 0:
            return None
        normal = np.asarray(plane.normal, dtype=float)
        u, v = self._plane_axes(normal)
        center_world = np.asarray(plane.center, dtype=float) + np.asarray(self.workspace.origin, dtype=float)
        relative = np.asarray(region.points, dtype=float) - center_world.reshape(1, 3)
        return np.dot(relative, u), np.dot(relative, v), u, v

    def _region_surface(self, region):
        if region is None or getattr(region, "n_points", 0) == 0:
            return None
        cache_key = id(region)
        cached = self._plane_region_surface_cache.get(cache_key)
        if cached is not None and cached[0] is region:
            return cached[1]
        if getattr(region, "faces", None) is not None:
            surface = region
        else:
            try:
                surface = _extract_surface(region)
            except Exception:
                return None
        if surface is None or surface.n_points == 0:
            return None
        if len(self._plane_region_surface_cache) >= 24:
            self._plane_region_surface_cache.pop(next(iter(self._plane_region_surface_cache)))
        self._plane_region_surface_cache[cache_key] = (region, surface)
        return surface

    def _draw_effective_region_fill(self, region, plane, *, color="#00f5d4", alpha=0.18):
        surface = self._region_surface(region)
        projected = self._plane_region_coordinates(surface, plane)
        if projected is None:
            return False
        coord_u, coord_v, _u, _v = projected
        raw_faces = getattr(surface, "faces", None)
        if raw_faces is None:
            return False
        faces = np.asarray(raw_faces, dtype=np.int64).reshape(-1)
        if faces.size == 0:
            return False
        polygons = []
        cursor = 0
        while cursor < len(faces):
            count = int(faces[cursor])
            point_ids = faces[cursor + 1:cursor + 1 + count]
            cursor += count + 1
            if count < 3 or np.any(point_ids < 0) or np.any(point_ids >= len(coord_u)):
                continue
            polygons.append(np.column_stack((coord_u[point_ids], coord_v[point_ids])))
        if not polygons:
            return False
        self.ax_plane.add_collection(
            PolyCollection(
                polygons,
                closed=True,
                facecolors=color,
                edgecolors="none",
                linewidths=0.0,
                alpha=float(alpha),
                zorder=1.5,
            )
        )
        return True

    def _draw_effective_region_outline(self, region, plane, *, color="#00f5d4", linewidth=1.8):
        surface = self._region_surface(region)
        projected = self._plane_region_coordinates(surface, plane)
        if projected is None:
            return False
        coord_u, coord_v, _u, _v = projected
        try:
            edges = surface.extract_feature_edges(
                boundary_edges=True,
                feature_edges=False,
                manifold_edges=False,
                non_manifold_edges=False,
            )
            edge_projected = self._plane_region_coordinates(edges, plane)
            if edge_projected is None:
                return
            edge_u, edge_v, _u, _v = edge_projected
            lines = np.asarray(edges.lines, dtype=np.int64)
            cursor = 0
            plotted = False
            while cursor < len(lines):
                count = int(lines[cursor])
                point_ids = lines[cursor + 1:cursor + 1 + count]
                if count >= 2:
                    self.ax_plane.plot(
                        edge_u[point_ids],
                        edge_v[point_ids],
                        color=color,
                        linewidth=linewidth,
                        solid_capstyle="round",
                        zorder=3,
                    )
                    plotted = True
                cursor += count + 1
            return plotted
        except Exception:
            self.ax_plane.scatter(coord_u, coord_v, s=2, color=color, zorder=3)
            return True

    def _effective_plane_mask(self, plane, t):
        ws = self.workspace
        mask = ws.segmask_binary
        if mask is None:
            return None
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim == 4:
            mask = mask[..., min(max(0, int(t)), mask.shape[3] - 1)]
        if mask.ndim != 3:
            return None
        plane_seg_label = int(getattr(plane, "segmentation_label", 0) or 0)
        labels = getattr(ws, "segmask_labels_3d", None)
        if plane_seg_label > 0 and labels is not None:
            labels = np.asarray(labels)
            if labels.ndim == 4:
                labels = labels[..., 0]
            if labels.shape == mask.shape:
                mask = mask & (labels == plane_seg_label)
        return mask

    def _segmentation_plane_labels(self, t):
        ws = self.workspace
        source = getattr(ws, "segmask_raw", None)
        if source is not None:
            source = np.asarray(source)
            mask = source[..., min(max(0, int(t)), source.shape[3] - 1)] if source.ndim == 4 else source
        else:
            mask = self._display_labels_for_frame(t)
        if mask is None:
            mask = getattr(ws, "segmask_binary", None)
            if mask is not None:
                mask = np.asarray(mask)
                if mask.ndim == 4:
                    mask = mask[..., min(max(0, int(t)), mask.shape[3] - 1)]
        if mask is None or np.asarray(mask).ndim != 3:
            return None
        return np.asarray(mask, dtype=np.int16)

    def _segmentation_plane_mask(self, t):
        labels = self._segmentation_plane_labels(t)
        return None if labels is None else np.asarray(labels != 0, dtype=np.float32)

    def _sample_segmentation_overlay(self, t, center_mm, normal, fov_mm, sample_spacing_mm):
        mask = self._segmentation_plane_mask(t)
        if mask is None or not np.any(mask):
            return None
        sampled, axis_mm, _u, _v = self._resample_oblique(
            mask, center_mm, normal, fov_mm, sample_spacing_mm
        )
        grid_spacing_mm = float(abs(axis_mm[-1] - axis_mm[0]) / max(len(axis_mm) - 1, 1))
        smooth_sigma = max(1.0, 0.45 / max(grid_spacing_mm, 1e-6))
        return gaussian_filter(np.clip(sampled, 0.0, 1.0), sigma=smooth_sigma), axis_mm

    def _sample_raw_segmentation_plane(self, plane, t, center_mm, normal, fov_mm, sample_spacing_mm):
        labels = self._segmentation_plane_labels(t)
        if labels is None or not np.any(labels > 0):
            return None
        plane_seg_label = int(getattr(plane, "segmentation_label", 0) or 0)
        if plane_seg_label > 0:
            labels = np.where(labels == plane_seg_label, labels, 0)
            if not np.any(labels > 0):
                return None
        sampled, axis_mm, _u, _v = self._resample_oblique(
            labels,
            center_mm,
            normal,
            fov_mm,
            sample_spacing_mm,
            interpolation_order=0,
            cell_centered=True,
        )
        return np.asarray(np.rint(sampled), dtype=np.int16), axis_mm

    def _draw_multilabel_overlay(self, labels, axis_mm, alpha=0.30):
        labels = np.asarray(labels, dtype=np.int16)
        if labels.ndim != 2 or not np.any(labels > 0):
            return False
        rgba = np.zeros((*labels.shape, 4), dtype=np.float32)
        for label_id in np.unique(labels):
            label_id = int(label_id)
            if label_id <= 0:
                continue
            color = QtGui.QColor(self._label_color(label_id))
            if not color.isValid():
                continue
            selected = labels == label_id
            rgba[selected, :3] = (color.redF(), color.greenF(), color.blueF())
            rgba[selected, 3] = float(alpha)
        self.ax_plane.imshow(
            rgba.transpose(1, 0, 2), origin="lower", aspect="equal",
            extent=[float(axis_mm[0]), float(axis_mm[-1]), float(axis_mm[0]), float(axis_mm[-1])],
            interpolation="nearest", zorder=2.2,
        )
        return True

    def _draw_multilabel_contours(self, labels, axis_mm, linewidth=1.0):
        plotted = False
        labels = np.asarray(labels, dtype=np.int16)
        for label_id in np.unique(labels):
            label_id = int(label_id)
            if label_id <= 0:
                continue
            color = self._label_color(label_id)
            for contour in find_contours((labels == label_id).astype(np.float32), 0.5):
                if len(contour) < 4:
                    continue
                self.ax_plane.plot(
                    np.interp(contour[:, 0], np.arange(len(axis_mm), dtype=float), axis_mm),
                    np.interp(contour[:, 1], np.arange(len(axis_mm), dtype=float), axis_mm),
                    color=color, linewidth=linewidth, solid_capstyle="round", zorder=3,
                )
                plotted = True
        return plotted

    def _sample_smooth_plane_mask(self, plane, t, center_mm, normal, fov_mm, sample_spacing_mm):
        mask = self._effective_plane_mask(plane, t)
        if mask is None or not np.any(mask):
            return None
        sampled, axis_mm, _u, _v = self._resample_oblique(
            mask.astype(np.float32), center_mm, normal, fov_mm, sample_spacing_mm
        )
        binary = sampled > 0.45
        components, component_count = label_components(binary)
        if component_count:
            center_index = np.array([binary.shape[0] // 2, binary.shape[1] // 2], dtype=int)
            center_component = int(components[tuple(center_index)])
            if center_component <= 0:
                labels = np.arange(1, component_count + 1, dtype=int)
                centroids = np.asarray(
                    [np.argwhere(components == label_id).mean(axis=0) for label_id in labels],
                    dtype=float,
                )
                center_component = int(labels[np.argmin(np.linalg.norm(centroids - center_index, axis=1))])
            binary = components == center_component
        binary = binary_fill_holes(binary)
        grid_spacing_mm = float(abs(axis_mm[-1] - axis_mm[0]) / max(len(axis_mm) - 1, 1))
        close_iterations = max(1, int(round(0.45 / max(grid_spacing_mm, 1e-6))))
        binary = binary_closing(binary, iterations=close_iterations)
        smooth_sigma = max(0.8, 1.0 / max(grid_spacing_mm, 1e-6))
        return gaussian_filter(binary.astype(np.float32), sigma=smooth_sigma), axis_mm

    def _sample_smooth_region_mask(self, region, plane, fov_mm, sample_spacing_mm):
        surface = self._region_surface(region)
        projected = self._plane_region_coordinates(surface, plane)
        if projected is None:
            return None
        coord_u, coord_v, _u, _v = projected
        raw_faces = getattr(surface, "faces", None)
        if raw_faces is None:
            return None
        faces = np.asarray(raw_faces, dtype=np.int64).reshape(-1)
        if faces.size == 0:
            return None
        sample_count = int(
            np.clip(
                np.ceil(float(fov_mm) / max(min(float(sample_spacing_mm), 0.25), 1e-3)) + 1,
                129,
                257,
            )
        )
        axis_mm = np.linspace(-0.5 * float(fov_mm), 0.5 * float(fov_mm), sample_count)
        binary = np.zeros((sample_count, sample_count), dtype=bool)
        cursor = 0
        while cursor < len(faces):
            count = int(faces[cursor])
            point_ids = faces[cursor + 1:cursor + 1 + count]
            cursor += count + 1
            if count < 3 or np.any(point_ids < 0) or np.any(point_ids >= len(coord_u)):
                continue
            polygon = np.column_stack((coord_u[point_ids], coord_v[point_ids]))
            lo_u = max(0, int(np.searchsorted(axis_mm, np.min(polygon[:, 0]), side="left")) - 1)
            hi_u = min(sample_count, int(np.searchsorted(axis_mm, np.max(polygon[:, 0]), side="right")) + 1)
            lo_v = max(0, int(np.searchsorted(axis_mm, np.min(polygon[:, 1]), side="left")) - 1)
            hi_v = min(sample_count, int(np.searchsorted(axis_mm, np.max(polygon[:, 1]), side="right")) + 1)
            if lo_u >= hi_u or lo_v >= hi_v:
                continue
            grid_u, grid_v = np.meshgrid(axis_mm[lo_u:hi_u], axis_mm[lo_v:hi_v], indexing="ij")
            inside = Path(polygon).contains_points(np.column_stack((grid_u.ravel(), grid_v.ravel())))
            binary[lo_u:hi_u, lo_v:hi_v] |= inside.reshape(grid_u.shape)
        if not np.any(binary):
            return None
        binary = binary_fill_holes(binary)
        grid_spacing_mm = float(abs(axis_mm[-1] - axis_mm[0]) / max(len(axis_mm) - 1, 1))
        smooth_sigma = max(0.8, 1.0 / max(grid_spacing_mm, 1e-6))
        return gaussian_filter(binary.astype(np.float32), sigma=smooth_sigma), axis_mm

    def _draw_smooth_contours(self, ax, smooth, axis_mm, *, color="#00f5d4", linewidth=1.8):
        plotted = 0
        sigma = float(self._plane_display_smoothing_sigma)
        if sigma > 0:
            smooth = gaussian_filter(np.asarray(smooth, dtype=np.float32), sigma=sigma)
        min_points = int(self._plane_display_smoothing_points)
        contours = find_contours(smooth, 0.5)
        for contour in contours:
            if len(contour) < min_points:
                continue
            contour_u = np.interp(contour[:, 0], np.arange(len(axis_mm), dtype=float), axis_mm)
            contour_v = np.interp(contour[:, 1], np.arange(len(axis_mm), dtype=float), axis_mm)
            ax.plot(
                contour_u,
                contour_v,
                color=color,
                linewidth=linewidth,
                solid_capstyle="round",
                solid_joinstyle="round",
            )
            plotted += 1
        return plotted > 0

    def _draw_smooth_mask_outline(self, plane, t, center_mm, normal, fov_mm, sample_spacing_mm):
        smooth_result = self._sample_smooth_plane_mask(
            plane, t, center_mm, normal, fov_mm, sample_spacing_mm
        )
        if smooth_result is None:
            return False
        smooth, axis_mm = smooth_result
        return self._draw_smooth_contours(self.ax_plane, smooth, axis_mm)

    def _set_plane_status(self, text):
        self.label_plane_status.setText(str(text))

    def open_plane_display_settings(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Ortho Viewer Display")
        form = QtWidgets.QFormLayout(dialog)
        slice_zoom = QtWidgets.QDoubleSpinBox()
        slice_zoom.setRange(10.0, 100.0)
        slice_zoom.setDecimals(0)
        slice_zoom.setSingleStep(5.0)
        slice_zoom.setValue(100.0 * self._slice_default_view_fraction)
        slice_zoom.setSuffix(" % of full FOV")
        slice_zoom.setToolTip(
            "Initial view range for all three orthogonal viewers; 50% shows a 2x view-only zoom"
        )
        sigma = QtWidgets.QDoubleSpinBox()
        sigma.setRange(0.0, 5.0)
        sigma.setDecimals(2)
        sigma.setSingleStep(0.1)
        sigma.setValue(float(self._plane_display_smoothing_sigma))
        sigma.setSuffix(" px")
        points = QtWidgets.QSpinBox()
        points.setRange(0, 100)
        points.setValue(int(self._plane_display_smoothing_points))
        snap_distance = QtWidgets.QDoubleSpinBox()
        snap_distance.setRange(0.0, 50.0)
        snap_distance.setDecimals(2)
        snap_distance.setSingleStep(0.25)
        snap_distance.setValue(float(self._contour_snap_distance_mm))
        snap_distance.setSuffix(" mm")
        snap_distance.setSpecialValueText("Auto")
        snap_distance.setToolTip(
            "Maximum distance for an open contour endpoint; Auto uses 2.5x voxel spacing with a 1.5 mm minimum"
        )
        form.addRow("Three-view default FOV", slice_zoom)
        form.addRow("Contour smoothing", sigma)
        form.addRow("Minimum contour points", points)
        form.addRow("Endpoint snap distance", snap_distance)
        hint = QtWidgets.QLabel("Display settings do not change metric areas or masks. Endpoint snap distance controls open contour edits.")
        hint.setWordWrap(True)
        form.addRow(hint)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        def apply_preview():
            self._slice_default_view_fraction = float(slice_zoom.value()) / 100.0
            self._plane_display_smoothing_sigma = float(sigma.value())
            self._plane_display_smoothing_points = int(points.value())
            self._contour_snap_distance_mm = float(snap_distance.value())
            for view in self.slice_views.values():
                view.set_default_view_fraction(self._slice_default_view_fraction)
                view.reset_view()
            self._slice_keys.clear()
            self.refresh(update_plane=True)
        slice_zoom.valueChanged.connect(lambda _value: apply_preview())
        sigma.valueChanged.connect(lambda _value: apply_preview())
        points.valueChanged.connect(lambda _value: apply_preview())
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        apply_preview()
        settings = QtCore.QSettings("AutoFlow", "AutoFlow")
        settings.setValue("plane/slice_default_view_fraction", self._slice_default_view_fraction)
        settings.setValue("plane/display_smoothing_sigma", self._plane_display_smoothing_sigma)
        settings.setValue("plane/display_smoothing_points", self._plane_display_smoothing_points)
        settings.setValue("plane/contour_snap_distance_mm", self._contour_snap_distance_mm)

    def _draw_manual_roi_outline(self, plane):
        polygon = np.asarray(getattr(plane, "roi_polygon_uv_mm", []) or [], dtype=float)
        if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
            return False
        closed = np.vstack((polygon, polygon[0]))
        self.ax_plane.plot(closed[:, 0], closed[:, 1], "--", color="#ffd43b", linewidth=1.2)
        return True

    def _draw_plane_flow(self, t):
        # Kept as a compatibility no-op for callers from older GUI code. The
        # selected-plane view is now rendered by the three SliceView widgets.
        return
        # Legacy Matplotlib rendering code below is intentionally unreachable.
        self.ax_plane.clear()
        self.ax_plane.set_facecolor("black")
        self.ax_plane.set_axis_off()
        ws = self.workspace
        if self._selected_plane_idx is None or self._selected_plane_idx >= len(ws.planes):
            self._set_plane_status("Select a plane to inspect the cross-section")
            return
        content = str(self.combo_plane_content.currentData() or "through_plane")
        if content in {"through_plane", "pcmra", "speed"} and ws.flow_raw is None:
            self._set_plane_status("Selected Plane · no flow data")
            return
        if content in {"pcmra", "magnitude"} and ws.mag_raw is None:
            self._set_plane_status("Selected Plane · no magnitude data")
            return
        plane = ws.planes[self._selected_plane_idx]
        res = self._get_resolution()
        center_mm = np.asarray(plane.center, dtype=float)
        normal = np.asarray(plane.normal, dtype=float)
        normal = normal / (np.linalg.norm(normal) + 1e-12)
        region_key = (
            int(self._selected_plane_idx), int(t),
            tuple(np.round(center_mm, 4).tolist()), tuple(np.round(normal, 6).tolist()),
            id(ws.segmask_binary), id(ws.branch_labels), id(getattr(ws, "segmask_labels_3d", None)),
            repr(getattr(plane, "roi_edit_operations", {}) or {}),
        )
        region = self._cached("plane_region", region_key, lambda: self._effective_plane_region(plane, t))
        projected = self._plane_region_coordinates(region, plane)
        auto_fov = 30.0
        if projected is not None:
            coord_u, coord_v, _u, _v = projected
            radius = max(float(np.max(np.abs(coord_u))), float(np.max(np.abs(coord_v))), 1.0)
            auto_fov = max(8.0, 2.0 * radius + 4.0 * float(np.min(res)))
        fov_mm = auto_fov
        plane_key = (
            int(self._selected_plane_idx),
            int(t),
            content,
            tuple(np.round(center_mm, 4).tolist()),
            tuple(np.round(normal, 6).tolist()),
            round(float(fov_mm), 3),
            id(ws.flow_raw), id(ws.mag_raw), id(getattr(ws, "segmask_raw", None)),
            id(ws.segmask_binary), id(getattr(ws, "segmask_labels_3d", None)),
            id(getattr(getattr(ws, "segmentation", None), "working_labels_4d", None)),
            id(getattr(getattr(ws, "segmentation", None), "working_labels_3d", None)),
            repr(getattr(plane, "roi_edit_operations", {}) or {}),
        )
        def _build_plane_content():
            if content == "pcmra":
                flow_t = np.asarray(ws.flow_raw[..., t, :], dtype=np.float32)
                volume = np.asarray(ws.mag_raw[..., t], dtype=np.float32) * np.linalg.norm(flow_t, axis=-1)
                title, cmap, symmetric = "PC-MRA", "gray", False
            elif content == "magnitude":
                volume = np.asarray(ws.mag_raw[..., t], dtype=np.float32)
                title, cmap, symmetric = "Magnitude", "gray", False
            elif content == "speed":
                volume = np.linalg.norm(np.asarray(ws.flow_raw[..., t, :], dtype=np.float32), axis=-1)
                title, cmap, symmetric = "Speed (cm/s)", "turbo", False
            else:
                flow_t = np.asarray(ws.flow_raw[..., t, :], dtype=np.float32)
                volume = np.dot(flow_t, normal)
                title, cmap, symmetric = "Through-plane flow (cm/s)", "RdBu_r", True
            sampled, axis_mm, _u, _v = self._resample_oblique(
                volume,
                center_mm,
                normal,
                fov_mm,
                sample_spacing_mm=max(float(np.min(res)) * 0.75, 0.25),
                interpolation_order=1,
                cell_centered=True,
            )
            display_alpha = None
            segmentation_region = self._segmentation_plane_region(plane, t)
            sample_spacing_mm = max(float(np.min(res)) * 0.75, 0.25)
            effective_result = self._sample_smooth_region_mask(
                region,
                plane,
                fov_mm,
                sample_spacing_mm=sample_spacing_mm,
            )
            if effective_result is None:
                effective_result = self._sample_smooth_plane_mask(
                    plane,
                    t,
                    center_mm,
                    normal,
                    fov_mm,
                    sample_spacing_mm=sample_spacing_mm,
                )
            segmentation_result = self._sample_smooth_region_mask(
                segmentation_region,
                plane,
                fov_mm,
                sample_spacing_mm=sample_spacing_mm,
            )
            raw_segmentation_result = self._sample_raw_segmentation_plane(
                plane,
                t,
                center_mm,
                normal,
                fov_mm,
                sample_spacing_mm=sample_spacing_mm,
            )
            smooth_mask = None if effective_result is None else effective_result[0]
            smooth_axis_mm = None if effective_result is None else effective_result[1]
            segmentation_smooth_mask = None if segmentation_result is None else segmentation_result[0]
            segmentation_axis_mm = None if segmentation_result is None else segmentation_result[1]
            raw_segmentation_mask = None if raw_segmentation_result is None else raw_segmentation_result[0]
            raw_segmentation_axis_mm = None if raw_segmentation_result is None else raw_segmentation_result[1]
            return (
                sampled,
                display_alpha,
                axis_mm,
                title,
                cmap,
                symmetric,
                smooth_mask,
                smooth_axis_mm,
                segmentation_region,
                segmentation_smooth_mask,
                segmentation_axis_mm,
                raw_segmentation_mask,
                raw_segmentation_axis_mm,
            )
        (
            sl,
            display_alpha,
            axis_mm,
            content_title,
            cmap,
            symmetric,
            smooth_mask,
            smooth_axis_mm,
            segmentation_region,
            segmentation_smooth_mask,
            segmentation_axis_mm,
            raw_segmentation_mask,
            raw_segmentation_axis_mm,
        ) = self._cached("plane_content", plane_key, _build_plane_content)
        finite = sl[np.isfinite(sl)]
        if finite.size:
            if symmetric:
                vmax = max(float(np.percentile(np.abs(finite), 99.0)), 1e-6)
                vmin = -vmax
            else:
                vmin, vmax = float(np.percentile(finite, 1.0)), float(np.percentile(finite, 99.0))
                if vmax <= vmin:
                    vmax = vmin + 1e-6
        else:
            vmin, vmax = (0.0, 1.0)
        extent = [float(axis_mm[0]), float(axis_mm[-1]), float(axis_mm[0]), float(axis_mm[-1])]
        display_cmap = colormaps.get_cmap(cmap).copy()
        display_cmap.set_bad("#050809")
        self.ax_plane.imshow(sl.T, origin="lower", cmap=display_cmap, vmin=vmin, vmax=vmax, alpha=None if display_alpha is None else display_alpha.T, aspect="equal", extent=extent, interpolation="nearest")
        if raw_segmentation_mask is not None and self.check_plane_raw_mask_overlay.isChecked():
            self._draw_multilabel_overlay(raw_segmentation_mask, raw_segmentation_axis_mm)
        if region is None and smooth_mask is not None:
            roi_overlay = np.ma.masked_less(smooth_mask.T, 0.45)
            roi_cmap = ListedColormap(["#00f5d4"])
            roi_cmap.set_bad((0.0, 0.0, 0.0, 0.0))
            roi_alpha = np.clip((smooth_mask.T - 0.45) / 0.35, 0.0, 1.0) * 0.22
            self.ax_plane.imshow(
                roi_overlay,
                origin="lower",
                cmap=roi_cmap,
                alpha=roi_alpha,
                aspect="equal",
                extent=extent,
                interpolation="bilinear",
            )
        self.ax_plane.plot(0.0, 0.0, "+", color="#ff4d6d", markersize=9, markeredgewidth=1.8)
        if smooth_mask is not None:
            self._draw_smooth_contours(
                self.ax_plane, smooth_mask, smooth_axis_mm, color="#00f5d4", linewidth=1.6
            )
        has_manual_roi = self._draw_manual_roi_outline(plane)
        metrics = plane.metrics or {}
        txt = (
            f"Plane {self._selected_plane_idx} · t={int(t)} · {content_title} · FOV {fov_mm:.1f} mm\n"
            "CYAN = actual metric ROI · RED = selected raw seg label"
        )
        if has_manual_roi:
            txt += " · yellow = manual limit"
        if metrics:
            fr = metrics.get("flowrate_mL_s", [])
            ar = metrics.get("area_mm2", [])
            flow_txt = float(fr[t]) if t < len(fr) else 0.0
            area_txt = float(ar[t]) if t < len(ar) else 0.0
            txt += f" · Flow={flow_txt:.4g} mL/s · Area={area_txt:.4g} mm²"
        self._set_plane_status(txt)
        self.ax_plane.set_xlim(extent[0], extent[1])
        self.ax_plane.set_ylim(extent[2], extent[3])

    def reset_state(self):
        self._stop_plane_roi_editor()
        self._selected_plane_idx = None
        self._cache.clear()
        self._plane_region_surface_cache.clear()
        self._manual_levels = None
        self._current_volume = None
        self._current_title = ""
        self._noise_overlay_visible = False
        self._noise_overlay_opacity = 0.35
        self._slice_keys.clear()
        self._colorbar_state = None
        self._plane_roi_undo.clear()
        self._plane_roi_redo.clear()
        self.label_value.setText("Voxel: -   Value: -")
        self.label_plane_metric.setText("Plane metrics: -")
        self.contour_controls.hide()
        self.btn_edit_contour.setEnabled(False)
        self.btn_noise_overlay.blockSignals(True)
        self.btn_noise_overlay.setChecked(False)
        self.btn_noise_overlay.blockSignals(False)
        self.slider_noise_overlay_opacity.blockSignals(True)
        self.slider_noise_overlay_opacity.setValue(0)
        self.slider_noise_overlay_opacity.blockSignals(False)
        self.label_noise_overlay_opacity.setText("0%")
        self._refresh_content_choices()
        for view in self.slice_views.values():
            view.clear()
