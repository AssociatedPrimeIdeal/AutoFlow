from ..rendering.content import ScalarContentSource
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

from ..algorithms.metrics import _build_plane_slice_region, _resolve_plane_roi_frame, _target_label_for_plane
from ..algorithms.surfaces import _build_branch_grid, _extract_surface
from .contour_edit import new_contour_from_stroke, polygon_area, replace_contour_segment
from .slice_view import PLANE_SPECS, SliceView, make_colormap, make_label_overlay
from ..algorithms.slice_streamlines import trace_planar_streamlines, project_streamline_slab, limit_polyline_length
from ..algorithms.streamlines import generate_streamlines_at_t
from .viewer_display import OrthoDisplayMixin


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


class OrthoViewer(OrthoDisplayMixin, QtWidgets.QWidget):
    timeStepRequested = QtCore.Signal(int)
    planeRoiChanged = QtCore.Signal(int)
    contentRenderRequested = QtCore.Signal(int)

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
        self._streamline_cache = {}
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
        self._plane_roi_workspace = workspace
        self._plane_roi_planes = tuple(workspace.planes)
        self._initialize_display_settings()
        self._build_ui()
        self._sync_display_legacy_controls()

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
        self.combo_content.setMinimumWidth(140)
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
        self.combo_content.currentIndexChanged.connect(self._on_content_changed)
        ctrl.addWidget(QtWidgets.QLabel("Content:"))
        ctrl.addWidget(self.combo_content, 1)
        self.btn_render_content = QtWidgets.QPushButton("3D Render")
        self.btn_render_content.setToolTip("Add this Content as a 3D Volume in Browser")
        self.btn_render_content.clicked.connect(lambda: self.contentRenderRequested.emit(int(self.combo_content.currentData())))
        ctrl.addWidget(self.btn_render_content)
        self._refresh_content_choices()
        self.btn_segmentation_overlay = QtWidgets.QCheckBox("Seg")
        self.btn_segmentation_overlay.setChecked(self.display_state()["segmentation"]["visible"])
        self.btn_segmentation_overlay.setToolTip("Show 2D segmentation overlay")
        self.btn_segmentation_overlay.toggled.connect(self._on_segmentation_display_toggled)
        ctrl.addWidget(self.btn_segmentation_overlay)
        self.slider_overlay_opacity = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_overlay_opacity.setRange(0, 100)
        self.slider_overlay_opacity.setFixedWidth(78)
        self.slider_overlay_opacity.setToolTip("Segmentation overlay opacity")
        self.slider_overlay_opacity.valueChanged.connect(self._on_overlay_opacity_changed)
        self.slider_overlay_opacity.setParent(self)
        self.slider_overlay_opacity.hide()
        self.label_overlay_opacity = QtWidgets.QLabel("35%")
        self.label_overlay_opacity.setMinimumWidth(34)
        self.label_overlay_opacity.setParent(self)
        self.label_overlay_opacity.hide()
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
        self.slider_noise_overlay_opacity.setParent(self)
        self.slider_noise_overlay_opacity.hide()
        self.label_noise_overlay_opacity = QtWidgets.QLabel("0%")
        self.label_noise_overlay_opacity.setMinimumWidth(34)
        self.label_noise_overlay_opacity.setParent(self)
        self.label_noise_overlay_opacity.hide()
        self.btn_reset_views = QtWidgets.QPushButton()
        self.btn_reset_views.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_BrowserReload)
        )
        self.btn_reset_views.setToolTip("Reset slice zoom and display range (R)")
        self.btn_reset_views.setProperty("role", "icon")
        self.btn_reset_views.clicked.connect(self._reset_views)
        ctrl.addWidget(self.btn_reset_views)
        self.btn_display_settings = QtWidgets.QToolButton()
        self.btn_display_settings.setText("Display")
        self.btn_display_settings.setToolTip("Viewer Display Settings")
        self.btn_display_settings.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogDetailedView))
        self.btn_display_settings.clicked.connect(self.open_display_settings)
        ctrl.addWidget(self.btn_display_settings)
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

        streamline_controls = QtWidgets.QGridLayout()
        self.check_planar_streamlines = QtWidgets.QCheckBox("2D streamlines")
        self.check_planar_streamlines.setToolTip("Integrate in-plane velocity; configure speed or fixed colors in Display → Flow")
        self.check_projected_streamlines = QtWidgets.QCheckBox("3D projection")
        self.check_projected_streamlines.setToolTip("Clip generated 3D streamlines to a nearby slab, then project onto the slice")
        self.label_streamline_status = QtWidgets.QLabel()
        self.label_streamline_status.setWordWrap(True)
        streamline_controls.addWidget(self.check_planar_streamlines, 0, 0, 1, 2)
        streamline_controls.addWidget(self.check_projected_streamlines, 0, 2, 1, 2)
        streamline_controls.addWidget(self.label_streamline_status, 0, 4, 1, 2)
        self.spin_streamline_density = QtWidgets.QDoubleSpinBox()
        self.spin_streamline_density.setRange(0.25, 3.0)
        self.spin_streamline_density.setSingleStep(0.25)
        self.spin_streamline_density.setValue(1.0)
        self.spin_streamline_density.setToolTip("2D seed density; 3D projection uses the existing 3D seeds")
        self.spin_streamline_slab = QtWidgets.QDoubleSpinBox()
        self.spin_streamline_slab.setRange(0.1, 100.0)
        self.spin_streamline_slab.setSingleStep(0.5)
        self.spin_streamline_slab.setValue(3.0)
        self.spin_streamline_slab.setSuffix(" mm")
        self.spin_streamline_slab.setToolTip("Full slab thickness, centered on each displayed slice")
        self.slider_streamline_opacity = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_streamline_opacity.setRange(0, 100)
        self.slider_streamline_opacity.setValue(85)
        self.slider_streamline_opacity.setToolTip("Opacity of both streamline overlays")
        # Legacy controls remain callable; the unified dock edits details.
        for widget in (self.spin_streamline_density, self.spin_streamline_slab, self.slider_streamline_opacity):
            widget.setParent(self)
            widget.hide()
        layout.addLayout(streamline_controls)
        for check in (self.check_planar_streamlines, self.check_projected_streamlines):
            check.toggled.connect(self._on_streamline_display_changed)
        self.spin_streamline_density.valueChanged.connect(self._on_streamline_display_changed)
        self.spin_streamline_slab.valueChanged.connect(self._on_streamline_display_changed)
        self.slider_streamline_opacity.valueChanged.connect(self._on_streamline_opacity_changed)
        self._sync_streamline_controls()

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
        stored_order = str(self.display_state()["view"]["order"])
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
        self._velocity_colorbar_layout = QtWidgets.QGridLayout()
        layout.addLayout(self._velocity_colorbar_layout)

        for view in self.slice_views.values():
            view.cursorRequested.connect(self._on_slice_cursor_requested)
            view.view_box.sigRangeChanged.connect(lambda *_args, name=view.spec.name: self._link_slice_range(name))
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
        self.workspace.render_settings["viewer_display"]["view"]["order"] = ",".join(order)
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
            # Plane centers are local physical coordinates. World origin is
            # added only when comparing against VTK streamline points.
            coords = points / (res.reshape(1, 1, 3) + 1e-12)
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
            id(self.workspace.segmask_labels),
            tuple(np.asarray(plane.center, dtype=float)),
            tuple(np.asarray(plane.normal, dtype=float)),
            int(getattr(plane, "segmentation_label", 0) or 0),
            repr(getattr(plane, "roi_polygon_uv_mm", []) or []),
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
        def replacement(frame):
            operations = (getattr(plane, "roi_edit_operations", {}) or {}).get(str(int(frame)), []) or []
            for operation in reversed(operations):
                if str(operation.get("mode", "")) != "replace":
                    continue
                candidate = np.asarray(operation.get("polygon", []), dtype=float)
                if candidate.ndim == 2 and candidate.shape[1] == 2 and len(candidate) >= 3:
                    return candidate
                break
            return None

        manual_polygon = replacement(frame_index)
        if manual_polygon is None:
            _region, source = self._effective_plane_region(plane, frame_index, return_source=True)
            if source is not None and int(source) != int(frame_index):
                manual_polygon = replacement(source)
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
            id(self.workspace.segmask_labels),
            tuple(np.asarray(plane.center, dtype=float)),
            tuple(np.asarray(plane.normal, dtype=float)),
            int(getattr(plane, "segmentation_label", 0) or 0),
            repr(getattr(plane, "roi_polygon_uv_mm", []) or []),
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
        contours = []
        if result is not None:
            smooth, axis_mm = result
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

        if not contours:
            # A valid small ROI can vanish at the 0.5 isovalue after display
            # smoothing, fall below the minimum point count, or lie between
            # raster samples. Its actual cut-cell boundary remains drawable.
            contours.extend(
                (points, "#00f5d4", 1.6, False)
                for points in self._plane_region_boundary_contours(region, plane)
            )

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
        cfg = self.display_state()["image"]
        self._manual_levels = (cfg["low"], cfg["high"]) if cfg["range_mode"] == "manual" else None
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
        self._sync_plane_roi_history()
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

    def _sync_plane_roi_history(self):
        """Invalidate index-based history when its case or plane objects change."""
        planes = tuple(self.workspace.planes)
        previous = self._plane_roi_planes
        if self._plane_roi_workspace is self.workspace and len(previous) == len(planes) and all(
            old is new for old, new in zip(previous, planes)
        ):
            return
        self._stop_plane_roi_editor()
        self._plane_roi_undo.clear()
        self._plane_roi_redo.clear()
        self._plane_roi_workspace = self.workspace
        self._plane_roi_planes = planes
        self._selected_plane_idx = None
        self._contour_display_override = None
        for key in ("plane_region", "plane_content", "plane_contours"):
            self._cache.pop(key, None)
        self._slice_keys.clear()

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
        self._sync_plane_roi_history()
        if not self._plane_roi_undo:
            return
        plane_idx, snapshot = self._plane_roi_undo.pop()
        plane = self.workspace.planes[int(plane_idx)]
        self._plane_roi_redo.append((plane_idx, self._snapshot_plane_roi(plane)))
        self._restore_plane_roi(plane_idx, snapshot)

    def _redo_plane_roi(self):
        self._sync_plane_roi_history()
        if not self._plane_roi_redo:
            return
        plane_idx, snapshot = self._plane_roi_redo.pop()
        plane = self.workspace.planes[int(plane_idx)]
        self._plane_roi_undo.append((plane_idx, self._snapshot_plane_roi(plane)))
        self._restore_plane_roi(plane_idx, snapshot)

    def _reset_plane_roi(self):
        self._sync_plane_roi_history()
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
        self.workspace.render_settings["viewer_display"]["segmentation"]["opacity"] = opacity
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
        self.workspace.render_settings["viewer_display"]["noise"]["visible"] = bool(checked)
        if checked and self._noise_overlay_opacity <= 0.0:
            self._noise_overlay_opacity = 0.35
        self.workspace.render_settings["viewer_display"]["noise"]["opacity"] = self._noise_overlay_opacity
        self._slice_keys.clear()
        self.refresh(update_plane=False)
        self._apply_noise_overlay_opacity()

    def _on_noise_overlay_opacity_changed(self, value):
        self._noise_overlay_opacity = float(np.clip(float(value) / 100.0, 0.0, 1.0))
        self.workspace.render_settings["viewer_display"]["noise"]["opacity"] = self._noise_overlay_opacity
        self.btn_noise_overlay.setChecked(value > 0)
        self.label_noise_overlay_opacity.setText(f"{int(value)}%")
        self._apply_noise_overlay_opacity()

    def _apply_noise_overlay_opacity(self):
        for view in self.slice_views.values():
            view.noise_overlay_item.setOpacity(self._noise_overlay_opacity)

    def _sync_overlay_opacity_control(self):
        self.workspace.render_settings["viewer_display"]["segmentation"]["opacity"] = self.workspace.segmentation.opacity
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

    def _on_streamline_display_changed(self, *_):
        state = self.workspace.render_settings["viewer_display"]
        state["planar"].update(enabled=self.check_planar_streamlines.isChecked(), density=self.spin_streamline_density.value())
        state["projection"].update(enabled=self.check_projected_streamlines.isChecked(), slab=self.spin_streamline_slab.value())
        self.refresh(update_plane=False)

    def _on_streamline_opacity_changed(self, value):
        for mode in ("planar", "projection"):
            self.workspace.render_settings["viewer_display"][mode]["opacity"] = float(value) / 100.0
        for view in self.slice_views.values():
            for items in view.streamline_items.values():
                for item in items:
                    item.setOpacity(float(value) / 100.0)

    def _sync_streamline_controls(self):
        available = self.workspace.flow_raw is not None
        planar = available and any(self.display_state(name)["planar"]["enabled"] for name in PLANE_SPECS)
        projection = available and any(self.display_state(name)["projection"]["enabled"] for name in PLANE_SPECS)
        self.check_planar_streamlines.setEnabled(available)
        self.check_projected_streamlines.setEnabled(available)
        self.spin_streamline_density.setEnabled(planar)
        self.spin_streamline_slab.setEnabled(projection)
        self.slider_streamline_opacity.setEnabled(planar or projection)
        if projection and not self.workspace.streamline_active:
            text = "Generate Streamlines first"
        elif (planar or projection) and self._get_mask_3d() is None:
            text = "No segmentation: 2D uses the finite flow domain"
        else:
            text = "" if available else "Flow unavailable"
        self.label_streamline_status.setText(text)

    def _projection_streamline_mesh(self, t):
        ws = self.workspace
        if not ws.streamline_active or ws.flow_raw is None:
            return None
        if t in ws.streamline_cache:
            return ws.streamline_cache[t]
        mask = None
        if ws.segmask_binary is not None:
            binary = np.asarray(ws.segmask_binary)
            mask = binary[..., min(t, binary.shape[3] - 1)] if binary.ndim == 4 else binary
        if mask is None:
            mask = self._get_mask_3d()
        if mask is None:
            return None
        params = ws.streamline_params
        mesh = generate_streamlines_at_t(
            ws.flow_raw, t, ws.streamline_seeds, ws.resolution, ws.origin,
            mask_3d=np.asarray(mask, dtype=bool), max_steps=params.max_steps,
            terminal_speed=params.terminal_speed, seed_ratio=params.seed_ratio,
            min_seeds=params.min_seeds, rng_seed=params.rng_seed,
        )
        ws.streamline_cache[t] = mesh
        return mesh

    def _update_slice_streamlines(self, name, t, labels, selected_plane, center, fov_mm):
        view = self.slice_views[name]
        ws = self.workspace
        cfg = self.display_state(name)
        planar_cfg, projection_cfg = cfg["planar"], cfg["projection"]
        planar = planar_cfg["enabled"] and ws.flow_raw is not None
        projection = projection_cfg["enabled"] and ws.flow_raw is not None
        view._flow_clims = {}
        if not planar:
            view.set_streamlines("planar", None)
        if not projection:
            view.set_streamlines("projection", None)
        if not (planar or projection):
            return
        res = self._get_resolution()
        origin = np.asarray(ws.origin, dtype=float).reshape(3)
        frame = min(max(0, int(t)), ws.flow_raw.shape[3] - 1)
        if selected_plane is not None:
            normal = np.asarray(selected_plane.normal, dtype=float).copy()
            normal /= np.linalg.norm(normal) + 1e-12
            u, v = self._plane_axes(normal)
            horizontal, vertical = {"axial": (u, v), "coronal": (v, normal), "sagittal": (u, normal)}[name]
            world_center = origin + np.asarray(center, dtype=float)
        else:
            spec = PLANE_SPECS[name]
            horizontal, vertical = np.eye(3)[[spec.horizontal_axis, spec.vertical_axis]]
            fixed = (self.slider_x.value(), self.slider_y.value(), self.slider_z.value())[spec.fixed_axis]
            # Axis images place array sample i at i*spacing. VTK's velocity
            # grid stores that sample at origin + (i+0.5)*spacing.
            world_center = origin + 0.5 * res
            world_center[spec.fixed_axis] += fixed * res[spec.fixed_axis]
        geometry_key = (
            name, tuple(world_center), tuple(horizontal), tuple(vertical),
            tuple(view._shape), tuple(view._spacing), tuple(view._sample_origin),
        )
        if planar:
            key = (
                "planar", self._array_cache_token(ws.flow_raw), frame, geometry_key,
                None if labels is None else hash(np.asarray(labels > 0).tobytes()),
                planar_cfg["density"], planar_cfg["terminal_speed"], planar_cfg["max_steps"], planar_cfg["max_length"],
            )
            if key not in self._streamline_cache:
                flow = ws.flow_raw[..., frame, :]
                if selected_plane is not None:
                    sample_spacing = max(float(np.min(res)) * 0.75, 0.25)
                    components = [self._resample_basis(
                        flow[..., component], center, horizontal, vertical, fov_mm,
                        sample_spacing, order=1, cell_centered=True,
                    )[0] for component in range(3)]
                    samples = np.stack(components, axis=-1)
                else:
                    samples = np.take(flow, fixed, axis=spec.fixed_axis)
                vectors = np.stack((samples @ horizontal, samples @ vertical), axis=-1)
                curves = trace_planar_streamlines(
                    vectors, view._spacing, mask=None if labels is None else labels > 0,
                    offset=view._sample_origin, density=planar_cfg["density"],
                    terminal_speed=planar_cfg["terminal_speed"], max_steps=planar_cfg["max_steps"],
                )
                if planar_cfg["max_length"] > 0:
                    curves = [limit_polyline_length(curve, planar_cfg["max_length"]) for curve in curves]
                if len(self._streamline_cache) >= 24:
                    self._streamline_cache.pop(next(iter(self._streamline_cache)))
                self._streamline_cache[key] = curves
            curves = self._streamline_cache[key]
            values = self._planar_curve_values(curves, frame, world_center, horizontal, vertical, planar_cfg["quantity"])
            clim = self._streamline_clim(planar_cfg, horizontal, vertical)
            view._flow_clims["planar"] = clim
            view.set_streamlines("planar", curves, planar_cfg["opacity"], values=values, style=planar_cfg, clim=clim)
        if projection:
            mesh = self._projection_streamline_mesh(frame)
            key = ("projection", id(mesh), frame, geometry_key, projection_cfg["slab"], projection_cfg["fraction"], projection_cfg["quantity"])
            cached = self._streamline_cache.get(key)
            # Retain the mesh with the cache entry so a replaced VTK object's
            # Python id cannot accidentally reuse old projected geometry.
            if cached is None or cached[0] is not mesh:
                curves, vectors = project_streamline_slab(
                    mesh, world_center, horizontal, vertical, projection_cfg["slab"], return_vectors=True,
                    fraction=projection_cfg["fraction"],
                )
                basis = np.column_stack((horizontal, vertical))
                values = [np.linalg.norm(v @ basis if projection_cfg["quantity"] == "in_plane" else v, axis=-1) * 100.0 for v in vectors]
                if len(self._streamline_cache) >= 24:
                    self._streamline_cache.pop(next(iter(self._streamline_cache)))
                cached = self._streamline_cache[key] = (mesh, curves, values)
            clim = self._streamline_clim(projection_cfg, horizontal, vertical)
            view._flow_clims["projection"] = clim
            view.set_streamlines("projection", cached[1], projection_cfg["opacity"], values=cached[2], style=projection_cfg, clim=clim)

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

    def _noise_region_overlay(self, region_2d, name=None):
        cfg = self.display_state(name)["noise"]
        override = self.workspace.render_settings["viewer_display"].get("views", {}).get(name, {}).get("noise", {})
        visible = override.get("visible", self._noise_overlay_visible)
        if not visible or region_2d is None:
            return None
        region = np.asarray(region_2d, dtype=bool)
        rgba = np.zeros((*region.shape, 4), dtype=np.uint8)
        rgba[region] = QtGui.QColor(cfg["color"]).getRgb()
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
        self.btn_render_content.setEnabled(bool(available))
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
        self._sync_plane_roi_history()
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

    def _get_scalar_slice(self, t):
        if not hasattr(self, "_content_source") or self._content_source.workspace is not self.workspace:
            self._content_source = ScalarContentSource(self.workspace)
        self._content_source._cache = self._cache
        self._content_source._selected_plane_idx = self._selected_plane_idx
        return self._content_source.scalar_volume(t, self.combo_content.currentData())

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

    def _filtered_display_labels(self, labels_2d, name=None):
        if labels_2d is None:
            return None
        cfg = self.display_state(name)["segmentation"]
        if not cfg["labels"].strip():
            return labels_2d
        allowed = [int(v.strip()) for v in cfg["labels"].split(",") if v.strip().isdigit()]
        return np.where(np.isin(labels_2d, allowed), labels_2d, 0)

    def _label_overlay(self, labels_2d, name=None):
        seg_state = self.workspace.segmentation
        cfg = self.display_state(name)["segmentation"]
        if not seg_state.visible or not cfg["visible"] or labels_2d is None or cfg["mode"] == "outline":
            return None
        labels_2d = self._filtered_display_labels(labels_2d, name)
        colors = {
            int(value): cfg["colors"].get(str(int(value)), self._label_color(int(value)))
            for value in np.unique(labels_2d)
            if int(value) > 0
        }
        return make_label_overlay(labels_2d, colors, cfg.get("opacity", seg_state.opacity))

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
        self._remember_manual_image_levels()
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
        self._remember_manual_image_levels()
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
        self._sync_plane_roi_history()
        self._refresh_content_choices()
        ws = self.workspace
        # Never carry an intersection from a previous plane, time frame, or
        # slice while the new geometry is being rebuilt.
        for view in self.slice_views.values():
            view.plane_line.hide()
        self._sync_overlay_opacity_control()
        self._sync_noise_overlay_control()
        self._sync_streamline_controls()
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
            configured_cmap = self.display_state()["image"]["cmap"]
            self._current_cmap = style.get("cmap", "gray") if configured_cmap == "auto" else configured_cmap
            cmap = make_colormap(self._current_cmap)
            clim = style.get("clim", None)
            if clim is None:
                clim = (float(np.nanmin(vol)), float(np.nanmax(vol)))
            if self._manual_levels is not None:
                clim = self._manual_levels
            clim = tuple(float(value) for value in clim)
            if self._manual_levels is None:
                clim = self._cached_image_levels(vol, clim)
            if self.display_state()["image"]["range_mode"] != "slice":
                clim = self._image_display_range(vol, vol, clim, None)
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
                # Time and fixed-index changes replace pixels; only presentation
                # geometry/content changes require a full set_slice operation.
                slice_key = (
                    mode, self.combo_content.currentData(), tuple(image.shape),
                    tuple(view_spacing), None if extent is None else tuple(extent),
                    None if view_center is None else tuple(view_center),
                    None if selected_plane is None else tuple(np.asarray(selected_plane.center)),
                    None if selected_plane is None else tuple(np.asarray(selected_plane.normal)),
                )
                previous_key = self._slice_keys.get(plane)
                view_levels = self._image_display_range(image, vol, clim, plane)
                noise_overlay = self._noise_region_overlay(noise_region, plane)
                if previous_key == slice_key:
                    self.slice_views[plane].update_image(
                        image, self._label_overlay(labels, plane), cursor, fixed, view_levels,
                        noise_overlay=noise_overlay,
                    )
                else:
                    self.slice_views[plane].set_slice(
                        image, self._label_overlay(labels, plane), view_spacing, cursor, fixed,
                        cmap, view_levels, extent=extent, view_center=view_center,
                        noise_overlay=noise_overlay,
                    )
                    self._slice_keys[plane] = slice_key
                self._update_slice_streamlines(plane, t, labels, selected_plane, orth_center, orth_fov)
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
                self._style_slice(plane)
                seg_cfg = self.display_state(plane)["segmentation"]
                boundary_labels = self._filtered_display_labels(labels, plane) if seg_cfg["visible"] and seg_cfg["mode"] in {"outline", "both"} else None
                colors = {int(v): seg_cfg["colors"].get(str(int(v)), self._label_color(int(v))) for v in np.unique(boundary_labels) if v > 0} if boundary_labels is not None else {}
                self.slice_views[plane].set_segmentation_boundaries(boundary_labels, colors, seg_cfg["line_width"], seg_cfg.get("opacity", self.workspace.segmentation.opacity))
        else:
            self.colorbar_widget.hide()
            self._colorbar_state = None
            for view in self.slice_views.values():
                view.clear()

        self._update_value_label(vol, title)
        self._update_plane_metric_label()
        self._sync_velocity_colorbars()

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
        coords = points_mm / (spacing.reshape(1, 1, 3) + 1e-12)
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

    def _effective_plane_region(self, plane, t, *, return_source=False):
        ws = self.workspace
        mask = ws.segmask_binary
        if mask is None:
            return (None, None) if return_source else None
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim not in (3, 4):
            return (None, None) if return_source else None
        plane_seg_label = int(getattr(plane, "segmentation_label", 0) or 0)
        labels = getattr(ws, "segmask_labels", None)
        key = (id(plane), id(ws.segmask_binary), id(labels), id(ws.branch_labels),
               tuple(plane.center), tuple(plane.normal), tuple(ws.resolution), tuple(ws.origin),
               int(plane.label), plane_seg_label,
               repr(getattr(plane, "roi_polygon_uv_mm", []) or []),
               repr(getattr(plane, "roi_edit_operations", {}) or {}))
        history = self._cached("plane_roi_history", key, dict, max_items=8)
        branch_grid = self._cached(
            "plane_branch_grid", (id(ws.branch_labels), tuple(ws.resolution), tuple(ws.origin)),
            lambda: _build_branch_grid(ws.branch_labels, ws.resolution, ws.origin), max_items=1,
        )
        target = _target_label_for_plane(plane, ws.branch_labels, np.asarray(ws.resolution), ws.origin)

        def build(frame):
            mask_t = mask[..., min(frame, mask.shape[3] - 1)] if mask.ndim == 4 else mask
            if plane_seg_label > 0 and labels is not None:
                labels_t = np.asarray(labels)
                if labels_t.ndim == 4:
                    labels_t = labels_t[..., min(frame, labels_t.shape[3] - 1)]
                if labels_t.shape == mask_t.shape:
                    mask_t = mask_t & (labels_t == plane_seg_label)
            return _build_plane_slice_region(
                mask_t, plane, ws.resolution, ws.origin,
                branch_grid=branch_grid, target_label=target,
                select_connected=True, frame_index=frame,
            )

        region, source = _resolve_plane_roi_frame(history, plane, max(0, int(t)), build)
        return (region, source) if return_source else region

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

    def _plane_region_boundary_contours(self, region, plane):
        """Return ordered ROI boundaries without rasterizing or smoothing."""
        surface = self._region_surface(region)
        if surface is None:
            return []
        edges = surface.extract_feature_edges(
            boundary_edges=True,
            feature_edges=False,
            manifold_edges=False,
            non_manifold_edges=False,
        ).strip()
        projected = self._plane_region_coordinates(edges, plane)
        if projected is None:
            return []
        coord_u, coord_v, _u, _v = projected
        points = np.column_stack((coord_u, coord_v))
        lines = np.asarray(edges.lines, dtype=np.int64)
        contours = []
        cursor = 0
        while cursor < len(lines):
            count = int(lines[cursor])
            point_ids = lines[cursor + 1:cursor + 1 + count]
            if count >= 3:
                contours.append(points[point_ids])
            cursor += count + 1
        return contours

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

    def _smooth_plane_display_mask(self, binary, axis_mm):
        """Cap display smoothing to 10% of the ROI's equivalent diameter."""
        binary = np.asarray(binary, dtype=bool)
        values = binary.astype(np.float32)
        area_pixels = int(np.count_nonzero(binary))
        if area_pixels == 0:
            return values
        grid_spacing_mm = float(abs(axis_mm[-1] - axis_mm[0]) / max(len(axis_mm) - 1, 1))
        base_sigma = max(0.8, 1.0 / max(grid_spacing_mm, 1e-6))
        extra_sigma = max(0.0, float(self._plane_display_smoothing_sigma))
        diameter_pixels = 2.0 * np.sqrt(area_pixels / np.pi)
        sigma = min(float(np.hypot(base_sigma, extra_sigma)), 0.1 * diameter_pixels)
        return gaussian_filter(values, sigma=sigma)

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
        return self._smooth_plane_display_mask(binary, axis_mm), axis_mm

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
            # Include shared voxel edges. Path's default boundary membership
            # is undefined; gaps there split small ROIs when smoothing is mild.
            if polygon_area(polygon) < 0.0:
                polygon = polygon[::-1]
            inside = Path(polygon).contains_points(
                np.column_stack((grid_u.ravel(), grid_v.ravel())), radius=1e-7,
            )
            binary[lo_u:hi_u, lo_v:hi_v] |= inside.reshape(grid_u.shape)
        if not np.any(binary):
            return None
        binary = binary_fill_holes(binary)
        return self._smooth_plane_display_mask(binary, axis_mm), axis_mm

    def _draw_smooth_contours(self, ax, smooth, axis_mm, *, color="#00f5d4", linewidth=1.8):
        plotted = 0
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
        return self.open_display_settings()

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
        from .display_settings import merge_display, saved_display_defaults
        self.workspace.render_settings["viewer_display"] = merge_display(
            saved_display_defaults(), self.workspace.render_settings.get("viewer_display", {}))
        self._stop_plane_roi_editor()
        self._selected_plane_idx = None
        self._cache.clear()
        self._plane_region_surface_cache.clear()
        self._manual_levels = None
        self._current_volume = None
        self._current_title = ""
        self._noise_overlay_visible = False
        self._noise_overlay_opacity = 0.35
        self._streamline_cache.clear()
        self._streamline_values_cache.clear()
        state = self.workspace.render_settings.get("viewer_display", {})
        for mode in ("planar", "projection"):
            state.setdefault(mode, {})["enabled"] = False
        state.setdefault("noise", {})["visible"] = False
        for check in (self.check_planar_streamlines, self.check_projected_streamlines):
            check.blockSignals(True)
            check.setChecked(False)
            check.blockSignals(False)
        self._sync_streamline_controls()
        self._slice_keys.clear()
        self._colorbar_state = None
        self._plane_roi_undo.clear()
        self._plane_roi_redo.clear()
        self._plane_roi_workspace = self.workspace
        self._plane_roi_planes = tuple(self.workspace.planes)
        self._contour_display_override = None
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
        self._sync_display_legacy_controls()
        self._sync_display_panel()
        self._refresh_content_choices()
        for view in self.slice_views.values():
            view.clear()
