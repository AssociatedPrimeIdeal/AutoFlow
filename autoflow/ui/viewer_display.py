"""OrthoViewer's display state, dock and layer styling."""

import copy
import numpy as np
from scipy.ndimage import map_coordinates
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from .display_settings import DisplaySettingsPanel, merge_display, saved_display_defaults, view_display
from .slice_view import make_colormap


class OrthoDisplayMixin:
    def _initialize_display_settings(self):
        incoming = self.workspace.render_settings.get("viewer_display", {})
        state = merge_display(saved_display_defaults(), incoming)
        state["segmentation"].setdefault("opacity", float(self.workspace.segmentation.opacity))
        state["contours"].setdefault("smoothing", self._plane_display_smoothing_sigma)
        state["contours"].setdefault("minimum_points", self._plane_display_smoothing_points)
        state["contours"].setdefault("snap_distance", self._contour_snap_distance_mm)
        state["noise"].setdefault("visible", False)
        state["noise"].setdefault("opacity", 0.35)
        if "view" not in incoming:
            order = str(QtCore.QSettings("AutoFlow", "AutoFlow").value("ortho/slice_order", "") or "")
            if set(order.split(",")) == {"axial", "coronal", "sagittal"}:
                state["view"]["order"] = order
        self.workspace.render_settings["viewer_display"] = state
        self._display_dock = None
        self._display_panel = None
        self._streamline_values_cache = {}
        self._velocity_bars = []
        self._linking_views = False

    def display_state(self, name=None):
        return view_display(self.workspace.render_settings.get("viewer_display", {}), name,
                            self.combo_content.currentData() if hasattr(self, "combo_content") else None)

    def open_display_settings(self):
        if self._display_dock is None:
            window = self.window()
            self._display_dock = QtWidgets.QDockWidget("Viewer Display Settings", window)
            self._display_dock.setObjectName("viewerDisplaySettingsDock")
            self._display_panel = DisplaySettingsPanel(self.workspace.render_settings["viewer_display"])
            self._display_panel.settingsChanged.connect(self.apply_display_settings)
            self._display_dock.setWidget(self._display_panel)
            if isinstance(window, QtWidgets.QMainWindow):
                window.addDockWidget(QtCore.Qt.RightDockWidgetArea, self._display_dock)
            else:
                self._display_dock.setFloating(True)
        self._sync_display_panel()
        self._display_dock.show()
        self._display_dock.raise_()

    def _sync_display_panel(self):
        if self._display_panel is not None:
            self._display_panel.set_state(
                self.workspace.render_settings.get("viewer_display", {}), self.combo_content.currentData(),
                has_magnitude=self.workspace.mag_raw is not None, has_pcmra=self.workspace.pcmra_array is not None,
                has_noise=self.workspace.pcmra_render_mask is not None, has_flow=self.workspace.flow_raw is not None,
            )

    def apply_display_settings(self, state):
        previous = self.display_state()
        old_zoom = {name: view._default_view_fraction for name, view in self.slice_views.items()}
        state = merge_display(saved_display_defaults(), state)
        self.workspace.render_settings["viewer_display"] = state
        self._sync_display_legacy_controls()
        cfg = self.display_state()
        self._plane_display_smoothing_sigma = cfg["contours"].get("smoothing", 0.8)
        self._plane_display_smoothing_points = cfg["contours"].get("minimum_points", 8)
        self._contour_snap_distance_mm = cfg["contours"].get("snap_distance", 0.0)
        if previous["contours"] != cfg["contours"]:
            self._cache.pop("plane_contours", None)
        if previous["view"]["order"] != cfg["view"]["order"]:
            self.view_grid.set_order(cfg["view"]["order"].split(","), emit=False)
        for name, view in self.slice_views.items():
            zoom = self.display_state(name)["view"]["zoom"]
            view.set_default_view_fraction(zoom)
            if old_zoom[name] != zoom:
                view.reset_view()
        self._manual_levels = (cfg["image"]["low"], cfg["image"]["high"]) if cfg["image"]["range_mode"] == "manual" else None
        self.refresh(update_plane=False)

    def _sync_display_legacy_controls(self):
        cfg = self.display_state()
        self._slice_default_view_fraction = cfg["view"]["zoom"]
        self._plane_display_smoothing_sigma = cfg["contours"].get("smoothing", self._plane_display_smoothing_sigma)
        self._plane_display_smoothing_points = cfg["contours"].get("minimum_points", self._plane_display_smoothing_points)
        self._contour_snap_distance_mm = cfg["contours"].get("snap_distance", self._contour_snap_distance_mm)
        for name, view in self.slice_views.items():
            view.set_default_view_fraction(self.display_state(name)["view"]["zoom"])
        mapping = (
            (self.btn_segmentation_overlay, cfg["segmentation"]["visible"]),
            (self.btn_noise_overlay, cfg["noise"].get("visible", False)),
            (self.check_planar_streamlines, cfg["planar"]["enabled"]),
            (self.check_projected_streamlines, cfg["projection"]["enabled"]),
            (self.spin_streamline_density, cfg["planar"]["density"]),
            (self.spin_streamline_slab, cfg["projection"]["slab"]),
            (self.slider_overlay_opacity, int(100 * cfg["segmentation"].get("opacity", 0.35))),
        )
        for widget, value in mapping:
            widget.blockSignals(True)
            if isinstance(widget, QtWidgets.QCheckBox): widget.setChecked(value)
            else: widget.setValue(value)
            widget.blockSignals(False)
        self.workspace.segmentation.opacity = cfg["segmentation"].get("opacity", 0.35)
        self._noise_overlay_visible = cfg["noise"].get("visible", False)
        self._noise_overlay_opacity = cfg["noise"].get("opacity", 0.35)

    def _on_segmentation_display_toggled(self, visible):
        self.workspace.render_settings["viewer_display"]["segmentation"]["visible"] = bool(visible)
        self.refresh(update_plane=False)

    def _remember_manual_image_levels(self):
        if self._manual_levels is not None:
            state = self.workspace.render_settings["viewer_display"]
            cfg = state.setdefault("images", {}).setdefault(str(self.combo_content.currentData()), {})
            cfg.update(range_mode="manual", low=self._manual_levels[0], high=self._manual_levels[1])
            self._sync_display_panel()

    def _image_display_range(self, image, volume, fallback, name):
        cfg = self.display_state(name)["image"]
        if cfg["range_mode"] == "manual":
            return float(cfg["low"]), float(cfg["high"])
        if self._manual_levels is not None:
            return self._manual_levels
        if cfg["range_mode"] == "slice":
            return self._cached_image_levels(image, fallback)
        if cfg["range_mode"] == "frame":
            return self._cached_image_levels(volume, fallback)
        ws = self.workspace
        key = ("locked_display", self.combo_content.currentData(), id(ws.mag_raw), id(ws.pcmra_array),
               id(ws.flow_raw), id(ws.derived))
        def build():
            # Visit every cardiac phase; only the foreground display range is
            # retained. Do not materialize a second full 4D scalar volume.
            count = max(1, ws.flow_raw.shape[3] if ws.flow_raw is not None else
                        ws.mag_raw.shape[3] if ws.mag_raw is not None and ws.mag_raw.ndim == 4 else 1)
            ranges = []
            for t in range(count):
                values, _, style = self._get_scalar_slice(t)
                if values is not None:
                    ranges.append(self._cached_image_levels(values, style.get("clim") or fallback))
            return (min(x[0] for x in ranges), max(x[1] for x in ranges)) if ranges else fallback
        return self._cached("locked_display", key, build, max_items=6)

    def _streamline_clim(self, cfg, horizontal, vertical):
        if cfg["range_mode"] == "manual":
            return float(cfg["low"]), float(cfg["high"])
        ws = self.workspace
        basis_key = (tuple(horizontal), tuple(vertical)) if cfg["quantity"] == "in_plane" else ()
        key = (id(ws.flow_raw), id(ws.segmask_binary), cfg["quantity"], basis_key)
        def build():
            flow = np.asarray(ws.flow_raw)
            mask = ws.segmask_binary
            if mask is not None and np.asarray(mask).shape == flow.shape[:4]:
                vectors = flow[np.asarray(mask, dtype=bool)]
            elif mask is not None and np.asarray(mask).shape == flow.shape[:3]:
                vectors = flow[np.asarray(mask, dtype=bool)].reshape(-1, 3)
            else:
                vectors = flow.reshape(-1, 3)
            vectors = vectors[::max(1, len(vectors) // 1000000)]
            if cfg["quantity"] == "in_plane":
                vectors = vectors @ np.column_stack((horizontal, vertical))
            speed = np.linalg.norm(vectors, axis=-1)
            speed = speed[np.isfinite(speed)]
            high = float(np.percentile(speed, 99)) if speed.size else 1.0
            return 0.0, max(high, 1e-6)
        return self._cached("flow_display_range", key, build, max_items=12)

    def _planar_curve_values(self, curves, frame, world_center, horizontal, vertical, quantity):
        key = (id(curves), id(self.workspace.flow_raw), frame, quantity)
        found = self._streamline_values_cache.get(key)
        if found is not None and found[0] is curves:
            return found[1]
        if not curves:
            return []
        points = np.vstack(curves)
        world = world_center + points[:, 0, None] * horizontal + points[:, 1, None] * vertical
        coords = ((world - self.workspace.origin) / self._get_resolution() - 0.5).T
        flow = self.workspace.flow_raw[..., frame, :]
        vectors = np.column_stack([map_coordinates(flow[..., c], coords, order=1, mode="nearest") for c in range(3)])
        if quantity == "in_plane":
            vectors = vectors @ np.column_stack((horizontal, vertical))
        speed = np.linalg.norm(vectors, axis=-1)
        values = list(np.split(speed, np.cumsum([len(c) for c in curves])[:-1]))
        if len(self._streamline_values_cache) >= 24:
            self._streamline_values_cache.pop(next(iter(self._streamline_values_cache)))
        self._streamline_values_cache[key] = (curves, values)
        return values

    def _style_slice(self, name):
        view = self.slice_views[name]
        cfg = self.display_state(name)
        guides = cfg["guides"]
        view.plot.setBackground(cfg["view"]["background"])
        for line in (view.vertical_line, view.horizontal_line):
            line.setVisible(guides["crosshairs"])
            line.setPen(pg.mkPen(guides["crosshair_color"], width=guides["line_width"]))
        for item in view.orientation_items.values():
            item.setVisible(guides["orientation"])
        view.plane_line.setPen(pg.mkPen(guides["plane_color"], width=guides["line_width"], style=QtCore.Qt.DashLine))
        if not guides["plane_intersection"]:
            view.plane_line.hide()
        view.image_item.smooth = cfg["image"]["interpolation"] == "linear"
        view.image_item.update()
        cmap = self._current_cmap if cfg["image"]["cmap"] == "auto" else cfg["image"]["cmap"]
        view.image_item.setLookupTable(make_colormap(cmap).getLookupTable(nPts=256, alpha=True))
        view.noise_overlay_item.setOpacity(cfg["noise"].get("opacity", self._noise_overlay_opacity))
        if not cfg["noise"].get("visible", self._noise_overlay_visible):
            view.noise_overlay_item.hide()
        for item in view.contour_items:
            color = cfg["contours"]["manual_color"] if getattr(item, "_contour_role", "roi") == "manual" else cfg["contours"]["roi_color"]
            item.setPen(pg.mkPen(color, width=cfg["contours"]["line_width"],
                                style=QtCore.Qt.DashLine if cfg["contours"]["dashed"] else QtCore.Qt.SolidLine))
            item.setOpacity(cfg["contours"]["opacity"])
            item.setVisible(cfg["contours"]["visible"])

    def _sync_velocity_colorbars(self):
        specs = []
        cfg = self.display_state()
        font_size = cfg["colorbars"]["font_size"]
        if getattr(self, "_display_font_size", None) != font_size:
            self._display_font_size = font_size
            self._display_axis_font = QtGui.QFont("DejaVu Sans", font_size)
            self.colorbar_item.axis.setStyle(tickFont=self._display_axis_font)
        for name, view in self.slice_views.items():
            state = self.display_state(name)
            if not state["colorbars"]["velocity"]:
                continue
            for mode in ("planar", "projection"):
                style = state[mode]
                clim = getattr(view, "_flow_clims", {}).get(mode)
                if style["enabled"] and style["color_mode"] == "speed" and clim is not None:
                    # Same quantity, colormap and range share one legend.
                    spec = (style["quantity"], style["cmap"], tuple(clim))
                    if spec not in specs:
                        specs.append(spec)
        old = getattr(self, "_velocity_bar_specs", None)
        bar_style = (cfg["colorbars"]["width"], font_size, cfg["view"]["background"])
        if getattr(self, "_velocity_bar_style", None) != bar_style:
            old = None
            self._velocity_bar_style = bar_style
        if old != specs:
            for widget in self._velocity_bars:
                self._velocity_colorbar_layout.removeWidget(widget)
                widget.deleteLater()
            self._velocity_bars.clear()
            for index, (quantity, cmap, clim) in enumerate(specs):
                widget = pg.GraphicsLayoutWidget()
                widget.setFixedHeight(cfg["colorbars"]["width"] + font_size * 2 + 28)
                widget.setBackground(cfg["view"]["background"])
                bar = pg.ColorBarItem(values=clim, width=cfg["colorbars"]["width"], colorMap=make_colormap(cmap),
                                     interactive=False, orientation="horizontal")
                bar.axis.setLabel("Total speed (cm/s)" if quantity == "total" else "In-plane speed (cm/s)")
                bar.axis.setStyle(tickFont=self._display_axis_font)
                widget.addItem(bar)
                self._velocity_colorbar_layout.addWidget(widget, index // 2, index % 2)
                self._velocity_bars.append(widget)
            self._velocity_bar_specs = specs
        self.colorbar_widget.setVisible(cfg["colorbars"]["background"] and self._current_volume is not None and self._maximized_view is None)
        self.colorbar_item.layout.setColumnFixedWidth(1, cfg["colorbars"]["width"])
        self.colorbar_widget.setFixedWidth(cfg["colorbars"]["width"] + max(64, font_size * 6))
        self.label_value.setVisible(cfg["guides"]["values"])
        self.label_plane_metric.setVisible(cfg["guides"]["values"])
        self._sync_display_panel()

    def _link_slice_range(self, name):
        if self._linking_views or not self.display_state()["view"]["linked_zoom"]:
            return
        self._linking_views = True
        try:
            source = self.slice_views[name].view_box.viewRange()
            for other, view in self.slice_views.items():
                if other != name:
                    view.view_box.setRange(xRange=source[0], yRange=source[1], padding=0)
        finally:
            self._linking_views = False
