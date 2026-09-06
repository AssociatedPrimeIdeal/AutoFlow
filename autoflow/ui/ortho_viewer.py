import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.ndimage import map_coordinates

from .slice_view import PLANE_SPECS, SliceView, make_colormap, make_label_overlay


class OrthoViewer(QtWidgets.QWidget):
    timeStepRequested = QtCore.Signal(int)

    def __init__(self, workspace, parent=None):
        super().__init__(parent)
        self.workspace = workspace
        self._selected_plane_idx = None
        self._cache = {}
        self._playback_active = False
        self._manual_levels = None
        self._current_volume = None
        self._current_title = ""
        self._updating_colorbar = False
        self._maximized_view = None
        self._slice_keys = {}
        self._colorbar_state = None
        self._correction_content_available = None
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
        self.combo_content.addItems([
            "Flow LR (cm/s)", "Flow AP (cm/s)", "Flow FH (cm/s)",
            "Magnitude", "PC-MRA", "Speed (cm/s)",
            "WSS (Pa)", "TKE (J/m³)",
            "Pressure Grad LR (Pa/m)", "Pressure Grad AP (Pa/m)", "Pressure Grad FH (Pa/m)", "|Pressure Grad| (Pa/m)",
            "Relative Pressure (Pa)",
            "Vorticity Magnitude (s⁻¹)", "Q-Criterion (s⁻²)", "Swirling Strength λci (s⁻¹)",
            "Corr Low LR (rad)", "Corr Low AP (rad)", "Corr Low FH (rad)",
            "Corr High LR (rad)", "Corr High AP (rad)", "Corr High FH (rad)"
            , "Wrap Mask (any component)", "Wrap Count LR", "Wrap Count AP", "Wrap Count FH",
            "Unwrapped − Wrapped Speed (cm/s)"
        ])
        self.combo_content.setCurrentIndex(4)
        self.combo_content.currentIndexChanged.connect(self._on_content_changed)
        ctrl.addWidget(QtWidgets.QLabel("Content:"))
        ctrl.addWidget(self.combo_content, 1)
        ctrl.addWidget(QtWidgets.QLabel("Overlay:"))
        self.slider_overlay_opacity = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_overlay_opacity.setRange(0, 100)
        self.slider_overlay_opacity.setFixedWidth(78)
        self.slider_overlay_opacity.setToolTip("Segmentation overlay opacity")
        self.slider_overlay_opacity.valueChanged.connect(self._on_overlay_opacity_changed)
        ctrl.addWidget(self.slider_overlay_opacity)
        self.label_overlay_opacity = QtWidgets.QLabel("35%")
        self.label_overlay_opacity.setMinimumWidth(34)
        ctrl.addWidget(self.label_overlay_opacity)
        self.btn_reset_views = QtWidgets.QPushButton()
        self.btn_reset_views.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_BrowserReload)
        )
        self.btn_reset_views.setToolTip("Reset slice zoom and display range (R)")
        self.btn_reset_views.setProperty("role", "icon")
        self.btn_reset_views.clicked.connect(self._reset_views)
        ctrl.addWidget(self.btn_reset_views)
        ctrl.addStretch()
        layout.addLayout(ctrl)

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
        self.view_grid = QtWidgets.QWidget()
        self.view_grid_layout = QtWidgets.QGridLayout(self.view_grid)
        self.view_grid_layout.setContentsMargins(0, 0, 0, 0)
        self.view_grid_layout.setSpacing(4)
        self.slice_views = {name: SliceView(name, self) for name in PLANE_SPECS}
        self.view_grid_layout.addWidget(self.slice_views["axial"], 0, 0)
        self.view_grid_layout.addWidget(self.slice_views["coronal"], 0, 1)
        self.view_grid_layout.addWidget(self.slice_views["sagittal"], 1, 0)

        self.plane_panel = QtWidgets.QFrame()
        self.plane_panel.setObjectName("sliceView")
        plane_layout = QtWidgets.QVBoxLayout(self.plane_panel)
        plane_layout.setContentsMargins(0, 0, 0, 0)
        plane_layout.setSpacing(0)
        plane_header = QtWidgets.QWidget()
        plane_header.setObjectName("sliceHeader")
        plane_header_layout = QtWidgets.QHBoxLayout(plane_header)
        plane_header_layout.setContentsMargins(7, 3, 7, 3)
        plane_title = QtWidgets.QLabel("Selected Plane")
        plane_title.setObjectName("sliceTitle")
        plane_header_layout.addWidget(plane_title)
        plane_layout.addWidget(plane_header)
        self.fig = Figure(figsize=(3.2, 3.2), dpi=80, facecolor="#050809")
        self.canvas = FigureCanvas(self.fig)
        self.ax_plane = self.fig.add_subplot(1, 1, 1)
        self.ax_plane.set_facecolor("#050809")
        self.ax_plane.set_xticks([])
        self.ax_plane.set_yticks([])
        self.fig.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.03)
        plane_layout.addWidget(self.canvas, 1)
        self.view_grid_layout.addWidget(self.plane_panel, 1, 1)
        self.view_grid_layout.setRowStretch(0, 1)
        self.view_grid_layout.setRowStretch(1, 1)
        self.view_grid_layout.setColumnStretch(0, 1)
        self.view_grid_layout.setColumnStretch(1, 1)
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

    def update_slider_ranges(self):
        self._refresh_correction_content_availability()
        self._refresh_phase_unwrap_content_availability()
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
        display_labels = ws.segmentation_display_4d()
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
        self.workspace.ortho_cursor = np.array([self.slider_x.value(), self.slider_y.value(), self.slider_z.value()], dtype=int)
        self.refresh()

    def _on_content_changed(self, _):
        self._manual_levels = None
        self.refresh(update_plane=False)

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

    def _sync_overlay_opacity_control(self):
        value = int(round(float(np.clip(getattr(self.workspace.segmentation, "opacity", 0.35), 0.0, 1.0)) * 100.0))
        self.slider_overlay_opacity.blockSignals(True)
        self.slider_overlay_opacity.setValue(value)
        self.slider_overlay_opacity.blockSignals(False)
        self.label_overlay_opacity.setText(f"{value}%")

    def _refresh_correction_content_availability(self):
        def valid(field):
            value = getattr(self.workspace, field, None)
            return bool(value is not None and np.asarray(value).ndim == 5 and np.asarray(value).shape[-1] == 3)

        low_available = valid("correction_raw")
        high_available = valid("correction_high_raw")
        availability = (low_available, high_available)
        if availability == self._correction_content_available:
            return
        self._correction_content_available = availability
        model = self.combo_content.model()
        for index, available, name in (
            (16, low_available, "low-VENC"),
            (17, low_available, "low-VENC"),
            (18, low_available, "low-VENC"),
            (19, high_available, "high-VENC"),
            (20, high_available, "high-VENC"),
            (21, high_available, "high-VENC"),
        ):
            item = model.item(index) if hasattr(model, "item") else None
            if item is not None:
                item.setEnabled(available)
            tooltip = f"{name} background-phase correction field." if available else "Unavailable: enable background phase correction and reload the input."
            self.combo_content.setItemData(index, tooltip, QtCore.Qt.ToolTipRole)
        if not self.combo_content.model().item(int(self.combo_content.currentIndex())).isEnabled():
            self.combo_content.setCurrentIndex(4)

    def _refresh_phase_unwrap_content_availability(self):
        available = "wrap_mask" in (getattr(self.workspace, "phase_unwrap_result", {}) or {})
        model = self.combo_content.model()
        for index in range(22, 27):
            item = model.item(index) if hasattr(model, "item") else None
            if item is not None:
                item.setEnabled(bool(available))
            self.combo_content.setItemData(index, "Estimated wrap diagnostics from the selected unwrapping run." if available else "Unavailable: run phase unwrapping first.", QtCore.Qt.ToolTipRole)
        if not model.item(int(self.combo_content.currentIndex())).isEnabled():
            self.combo_content.setCurrentIndex(4)

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
            self.plane_panel.show()
            self.colorbar_widget.setVisible(self._current_volume is not None)
            return
        self._maximized_view = plane
        for name, view in self.slice_views.items():
            view.setVisible(name == plane)
        self.plane_panel.hide()
        self.colorbar_widget.hide()

    def set_selected_plane(self, idx):
        self._selected_plane_idx = idx
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
        for obj in self.workspace.scene_objects.values():
            if obj.data_key == data_key:
                return obj.cmap or default_cmap, obj.clim if obj.clim else default_clim
        return default_cmap, default_clim

    def _get_wss_volume(self, t):
        ws = self.workspace
        if ws.derived.wss_volume is not None:
            cmap, clim = self._scene_style("wss_surface_live", "jet", None)
            tidx = min(max(0, int(t)), ws.derived.wss_volume.shape[3] - 1)
            return np.asarray(ws.derived.wss_volume[..., tidx], dtype=np.float32), "WSS (Pa)", {"cmap": cmap, "clim": clim}
        if not ws.derived.wss_surfaces:
            return None, "WSS (no data)", {"cmap": "jet", "clim": None}
        tidx = min(max(0, t), len(ws.derived.wss_surfaces) - 1)
        surf = ws.derived.wss_surfaces[tidx]
        if surf is None or "wss" not in surf.point_data:
            return None, "WSS (no data)", {"cmap": "jet", "clim": None}
        shape = self._get_volume_shape()
        if shape is None:
            return None, "WSS (no data)", {"cmap": "jet", "clim": None}
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
        cmap, clim = self._scene_style("wss_surface_live", "jet", None)
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
            cmap, clim = self._scene_style("tke_volume", "hot", None)
            return vol, "TKE (J/m³)", {"cmap": cmap, "clim": clim}
        if ws.derived.tke_volume is None:
            return None, "TKE (no data)", {"cmap": "hot", "clim": None}
        shape = self._get_volume_shape()
        if shape is None:
            return None, "TKE (no data)", {"cmap": "hot", "clim": None}
        tke_mesh = ws.derived.tke_volume
        if "TKE" not in tke_mesh.point_data and "TKE" not in tke_mesh.cell_data:
            return None, "TKE (no data)", {"cmap": "hot", "clim": None}
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
        cmap, clim = self._scene_style("tke_volume", "hot", None)
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
        content_idx = self.combo_content.currentIndex()
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
        if content_idx == 4 and ws.mag_raw is not None and ws.flow_raw is not None:
            key = (id(ws.mag_raw), id(ws.flow_raw), int(t))
            def _build():
                flow_t = np.asarray(ws.flow_raw[..., t, :], dtype=np.float32)
                speed = np.sqrt(np.sum(np.square(flow_t, dtype=np.float32), axis=-1))
                return np.asarray(ws.mag_raw[..., t], dtype=np.float32) * speed
            vol = self._cached("scalar_volume", ("pcmra",) + key, _build)
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
                vol = np.asarray(corr[..., min(int(t), corr.shape[3] - 1), component], dtype=np.float32)
                finite = vol[np.isfinite(vol)]
                vmax = max(float(np.percentile(np.abs(finite), 99.0)) if finite.size else 0.0, 1e-6)
                direction = ("LR", "AP", "FH")[component]
                venc_name = "High" if is_high else "Low"
                return vol, f"Corr {venc_name} {direction} (rad)", {"cmap": "RdBu_r", "clim": (-vmax, vmax)}
        if content_idx == 22:
            result = getattr(ws, "phase_unwrap_result", {}) or {}
            mask = result.get("wrap_mask")
            if mask is None:
                return None, "Wrap Mask (no unwrapping)", {"cmap": "gray", "clim": (0, 1)}
            vol = np.any(np.asarray(mask)[..., min(int(t), np.asarray(mask).shape[3]-1), :], axis=-1).astype(np.float32)
            return vol, "Wrap Mask (any component)", {"cmap": "Reds", "clim": (0, 1)}
        if content_idx in {23, 24, 25}:
            result = getattr(ws, "phase_unwrap_result", {}) or {}
            count = result.get("wrap_count")
            comp = content_idx - 23
            if count is None:
                return None, "Wrap Count (no unwrapping)", {"cmap": "RdBu_r", "clim": (-1, 1)}
            arr = np.asarray(count)
            vol = arr[..., min(int(t), arr.shape[3]-1), comp].astype(np.float32)
            lim = max(1.0, float(np.max(np.abs(vol))))
            direction = ("LR", "AP", "FH")[comp]
            return vol, f"Wrap Count {direction}", {"cmap": "RdBu_r", "clim": (-lim, lim)}
        if content_idx == 26:
            result = getattr(ws, "phase_unwrap_result", {}) or {}
            flow_u = result.get("flow_unwrapped")
            flow_w = result.get("flow_wrapped")
            if flow_u is None or flow_w is None:
                return None, "Unwrapped − Wrapped Speed (no unwrapping)", {"cmap": "gray", "clim": None}
            du = np.linalg.norm(np.asarray(flow_u)[..., min(int(t), np.asarray(flow_u).shape[3]-1), :], axis=-1)
            dw = np.linalg.norm(np.asarray(flow_w)[..., min(int(t), np.asarray(flow_w).shape[3]-1), :], axis=-1)
            vol = (du - dw).astype(np.float32)
            lim = max(1e-6, float(np.nanmax(np.abs(vol))))
            return vol, "Unwrapped − Wrapped Speed (cm/s)", {"cmap": "RdBu_r", "clim": (-lim, lim)}
        return None, "", {"cmap": "gray", "clim": None}

    def _get_mask_3d(self):
        ws = self.workspace
        display = ws.segmentation_display_4d()
        if display is not None:
            display = np.asarray(display, dtype=np.int16)
            if display.ndim == 4:
                t = min(max(0, int(ws.current_t)), max(0, display.shape[3] - 1))
                return display[..., t]
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
        self._refresh_correction_content_availability()
        self._refresh_phase_unwrap_content_availability()
        ws = self.workspace
        self._sync_overlay_opacity_control()
        t = int(ws.current_t)
        if update_plane:
            self._slice_keys.clear()
        shape = self._get_volume_shape()
        if shape is None:
            self._current_volume = None
            self.colorbar_widget.hide()
            for view in self.slice_views.values():
                view.clear()
            self.ax_plane.clear()
            self.ax_plane.set_facecolor("#050809")
            self.canvas.draw_idle()
            return

        cx, cy, cz = self.slider_x.value(), self.slider_y.value(), self.slider_z.value()
        vol, title, style = self._get_scalar_slice(t)
        labels_3d = self._get_mask_3d()
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
            self._set_colorbar(cmap, clim, title)
            slices = {
                "axial": (vol[:, :, cz], None if labels_3d is None else labels_3d[:, :, cz], (cx, cy), cz),
                "coronal": (vol[:, cy, :], None if labels_3d is None else labels_3d[:, cy, :], (cx, cz), cy),
                "sagittal": (vol[cx, :, :], None if labels_3d is None else labels_3d[cx, :, :], (cy, cz), cx),
            }
            for plane, (image, labels, cursor, fixed) in slices.items():
                spec = PLANE_SPECS[plane]
                slice_key = (int(self.combo_content.currentIndex()), int(t), int(fixed))
                previous_key = self._slice_keys.get(plane)
                if previous_key == slice_key:
                    self.slice_views[plane].update_cursor(cursor, fixed, clim)
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
                        clim,
                    )
                    self._slice_keys[plane] = slice_key
                else:
                    self.slice_views[plane].set_slice(
                        image,
                        self._label_overlay(labels),
                        (res[spec.horizontal_axis], res[spec.vertical_axis]),
                        cursor,
                        fixed,
                        cmap,
                        clim,
                    )
                    self._slice_keys[plane] = slice_key
        else:
            self.colorbar_widget.hide()
            self._colorbar_state = None
            for view in self.slice_views.values():
                view.clear()

        if update_plane and not self._playback_active:
            self._draw_plane_flow(t)
            self.canvas.draw_idle()
        self._update_value_label(vol, title)
        self._update_plane_metric_label()

    def _resample_oblique(self, volume_3d, center_vox, normal, half_size=30):
        normal = np.asarray(normal, dtype=float)
        normal = normal / (np.linalg.norm(normal) + 1e-12)
        up_hint = np.array([0.0, 1.0, 0.0]) if abs(normal[2]) > max(abs(normal[0]), abs(normal[1])) else np.array([0.0, 0.0, 1.0])
        u = np.cross(normal, up_hint)
        u = u / (np.linalg.norm(u) + 1e-12)
        v = np.cross(normal, u)
        v = v / (np.linalg.norm(v) + 1e-12)
        ii = np.arange(-half_size, half_size + 1, dtype=float)
        jj = np.arange(-half_size, half_size + 1, dtype=float)
        gi, gj = np.meshgrid(ii, jj, indexing="ij")
        coords = center_vox.reshape(1, 1, 3) + gi[..., None] * u.reshape(1, 1, 3) + gj[..., None] * v.reshape(1, 1, 3)
        sampled = map_coordinates(volume_3d, [coords[..., 0].ravel(), coords[..., 1].ravel(), coords[..., 2].ravel()], order=1, mode="constant", cval=0.0)
        return sampled.reshape(len(ii), len(jj))

    def _draw_plane_flow(self, t):
        self.ax_plane.clear()
        self.ax_plane.set_facecolor("black")
        self.ax_plane.set_xticks([])
        self.ax_plane.set_yticks([])
        ws = self.workspace
        if self._selected_plane_idx is None or self._selected_plane_idx >= len(ws.planes):
            self.ax_plane.set_title("Plane Through-Plane Velocity (select a plane)", color="white", fontsize=8)
            return
        if ws.flow_raw is None:
            self.ax_plane.set_title("Plane Through-Plane Velocity (no flow data)", color="white", fontsize=8)
            return
        plane = ws.planes[self._selected_plane_idx]
        res = self._get_resolution()
        center_vox = np.asarray(plane.center, dtype=float) / (res + 1e-12)
        normal = np.asarray(plane.normal, dtype=float)
        normal = normal / (np.linalg.norm(normal) + 1e-12)
        flow_t = ws.flow_raw[..., t, :]
        shape = self._get_volume_shape()
        half_size = max(10, min(shape) // 2)
        plane_key = (
            int(self._selected_plane_idx),
            int(t),
            tuple(np.round(center_vox, 4).tolist()),
            tuple(np.round(normal, 6).tolist()),
            int(half_size),
            id(ws.flow_raw),
        )
        def _build_plane_flow():
            proj = flow_t[..., 0] * normal[0] + flow_t[..., 1] * normal[1] + flow_t[..., 2] * normal[2]
            return self._resample_oblique(proj, center_vox, normal, half_size=half_size)
        sl = self._cached("plane_flow", plane_key, _build_plane_flow)
        vmax = max(abs(np.nanmin(sl)), abs(np.nanmax(sl)), 1e-6)
        self.ax_plane.imshow(sl.T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect=1.0)
        self.ax_plane.plot(half_size, half_size, "r+", markersize=10, markeredgewidth=2)
        mask_3d = self._get_mask_3d()
        if mask_3d is not None:
            mask_key = (
                int(self._selected_plane_idx),
                tuple(np.round(center_vox, 4).tolist()),
                tuple(np.round(normal, 6).tolist()),
                int(half_size),
                id(mask_3d),
            )
            m_sl = self._cached(
                "plane_mask",
                mask_key,
                lambda: self._resample_oblique(mask_3d.astype(float), center_vox, normal, half_size=half_size),
            )
            try:
                self.ax_plane.contour(
                    m_sl.T,
                    levels=[0.5],
                    colors=[self._label_color(self.workspace.segmentation.active_label)],
                    linewidths=0.7,
                    origin="lower",
                )
            except Exception:
                pass
        metrics = plane.metrics or {}
        txt = f"Plane {self._selected_plane_idx} Through-Plane Velocity [{-vmax:.2f}, {vmax:.2f}] cm/s"
        if metrics:
            fr = metrics.get("flowrate_mL_s", [])
            ar = metrics.get("area_mm2", [])
            flow_txt = float(fr[t]) if t < len(fr) else 0.0
            area_txt = float(ar[t]) if t < len(ar) else 0.0
            txt += f"\nFlow Rate={flow_txt:.4g} mL/s Area={area_txt:.4g} mm²"
        self.ax_plane.set_title(txt, color="white", fontsize=7)

    def reset_state(self):
        self._selected_plane_idx = None
        self._cache.clear()
        self._manual_levels = None
        self._current_volume = None
        self._current_title = ""
        self._slice_keys.clear()
        self._colorbar_state = None
        self.label_value.setText("Voxel: -   Value: -")
        self.label_plane_metric.setText("Plane metrics: -")
        for view in self.slice_views.values():
            view.clear()
        self.ax_plane.clear()
        self.ax_plane.set_facecolor("#050809")
        self.ax_plane.set_xticks([])
        self.ax_plane.set_yticks([])
        self.canvas.draw_idle()
