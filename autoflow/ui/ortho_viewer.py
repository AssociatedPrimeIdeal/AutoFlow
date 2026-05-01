import numpy as np
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.ndimage import map_coordinates


class OrthoViewer(QtWidgets.QWidget):
    def __init__(self, workspace, parent=None):
        super().__init__(parent)
        self.workspace = workspace
        self._selected_plane_idx = None
        self._scalar_cbar = None
        self._cache = {}
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
        layout.setSpacing(2)

        ctrl = QtWidgets.QHBoxLayout()
        self.combo_content = QtWidgets.QComboBox()
        self.combo_content.addItems([
            "Flow X (cm/s)", "Flow Y (cm/s)", "Flow Z (cm/s)",
            "Magnitude", "PC-MRA", "Speed (cm/s)",
            "WSS (Pa)", "TKE (J/m³)"
        ])
        self.combo_content.setCurrentIndex(4)
        self.combo_content.currentIndexChanged.connect(self._on_content_changed)
        ctrl.addWidget(QtWidgets.QLabel("Content:"))
        ctrl.addWidget(self.combo_content)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        slider_layout = QtWidgets.QHBoxLayout()
        self.slider_x = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_y = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_z = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.label_x = QtWidgets.QLabel("X:0")
        self.label_y = QtWidgets.QLabel("Y:0")
        self.label_z = QtWidgets.QLabel("Z:0")
        for lbl, sl in [(self.label_x, self.slider_x), (self.label_y, self.slider_y), (self.label_z, self.slider_z)]:
            sl.setRange(0, 0)
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

        self.fig = Figure(figsize=(6.2, 6.6), dpi=80, facecolor="black")
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMinimumSize(300, 300)
        self.ax_ax = self.fig.add_subplot(2, 2, 1)
        self.ax_cor = self.fig.add_subplot(2, 2, 2)
        self.ax_sag = self.fig.add_subplot(2, 2, 3)
        self.ax_plane = self.fig.add_subplot(2, 2, 4)
        for ax in [self.ax_ax, self.ax_cor, self.ax_sag, self.ax_plane]:
            ax.set_facecolor("black")
            ax.tick_params(colors="white", labelsize=6)
            ax.set_xticks([])
            ax.set_yticks([])
        self.fig.subplots_adjust(left=0.03, right=0.96, top=0.96, bottom=0.03, wspace=0.14, hspace=0.24)
        layout.addWidget(self.canvas, 1)

        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_click)

    def _remove_colorbar(self):
        if self._scalar_cbar is not None:
            try:
                self._scalar_cbar.remove()
            except Exception:
                pass
        self._scalar_cbar = None

    def _on_scroll(self, event):
        if event.inaxes is None:
            return
        ax = event.inaxes
        factor = 0.8 if event.button == "up" else 1.25
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xdata = event.xdata if event.xdata is not None else (xlim[0] + xlim[1]) / 2
        ydata = event.ydata if event.ydata is not None else (ylim[0] + ylim[1]) / 2
        new_w = (xlim[1] - xlim[0]) * factor
        new_h = (ylim[1] - ylim[0]) * factor
        ax.set_xlim(xdata - new_w / 2, xdata + new_w / 2)
        ax.set_ylim(ydata - new_h / 2, ydata + new_h / 2)
        self.canvas.draw_idle()

    def _on_click(self, event):
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        shape = self._get_volume_shape()
        if shape is None:
            return
        x = int(np.clip(np.round(event.xdata), 0, shape[0] - 1))
        y = int(np.clip(np.round(event.ydata), 0, max(shape[1], shape[2]) - 1))
        cx, cy, cz = self.slider_x.value(), self.slider_y.value(), self.slider_z.value()
        if event.inaxes == self.ax_ax:
            self._set_cursor(x, int(np.clip(np.round(event.ydata), 0, shape[1] - 1)), cz)
        elif event.inaxes == self.ax_cor:
            self._set_cursor(x, cy, int(np.clip(np.round(event.ydata), 0, shape[2] - 1)))
        elif event.inaxes == self.ax_sag:
            self._set_cursor(cx, int(np.clip(np.round(event.xdata), 0, shape[1] - 1)), int(np.clip(np.round(event.ydata), 0, shape[2] - 1)))

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
        self.refresh()

    def update_slider_ranges(self):
        shape = self._get_volume_shape()
        if shape is None:
            return
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
        self.refresh()

    def _get_volume_shape(self):
        ws = self.workspace
        if ws.flow_raw is not None and ws.flow_raw.ndim == 5:
            return ws.flow_raw.shape[:3]
        if ws.mag_raw is not None and ws.mag_raw.ndim == 4:
            return ws.mag_raw.shape[:3]
        if ws.segmask_3d is not None:
            return ws.segmask_3d.shape[:3]
        return None

    def _update_labels(self):
        self.label_x.setText(f"X:{self.slider_x.value()}")
        self.label_y.setText(f"Y:{self.slider_y.value()}")
        self.label_z.setText(f"Z:{self.slider_z.value()}")

    def _on_slider_changed(self, _):
        self._update_labels()
        self.workspace.ortho_cursor = np.array([self.slider_x.value(), self.slider_y.value(), self.slider_z.value()], dtype=int)
        self.refresh()

    def _on_content_changed(self, _):
        self.refresh()

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
            return np.asarray(ws.derived.wss_volume[..., tidx], dtype=float), "WSS (Pa)", {"cmap": cmap, "clim": clim}
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
            arr = np.asarray(ws.derived.tke_array, dtype=float)
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
                    vol = np.asarray(vol, dtype=float) * np.asarray(mask_t, dtype=float)
                return np.asarray(vol, dtype=float)
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
    def _get_scalar_slice(self, t):
        ws = self.workspace
        content_idx = self.combo_content.currentIndex()
        if content_idx == 0 and ws.flow_raw is not None:
            vol = np.asarray(ws.flow_raw[..., t, 0], dtype=float)
            vmax = max(abs(np.nanmin(vol)), abs(np.nanmax(vol)), 1e-6)
            return vol, "Flow X (cm/s)", {"cmap": "RdBu_r", "clim": (-vmax, vmax)}
        if content_idx == 1 and ws.flow_raw is not None:
            vol = np.asarray(ws.flow_raw[..., t, 1], dtype=float)
            vmax = max(abs(np.nanmin(vol)), abs(np.nanmax(vol)), 1e-6)
            return vol, "Flow Y (cm/s)", {"cmap": "RdBu_r", "clim": (-vmax, vmax)}
        if content_idx == 2 and ws.flow_raw is not None:
            vol = np.asarray(ws.flow_raw[..., t, 2], dtype=float)
            vmax = max(abs(np.nanmin(vol)), abs(np.nanmax(vol)), 1e-6)
            return vol, "Flow Z (cm/s)", {"cmap": "RdBu_r", "clim": (-vmax, vmax)}
        if content_idx == 3 and ws.mag_raw is not None:
            vol = np.asarray(ws.mag_raw[..., t], dtype=float)
            return vol, "Magnitude", {"cmap": "gray", "clim": (float(np.nanmin(vol)), float(np.nanmax(vol)))}
        if content_idx == 4 and ws.mag_raw is not None and ws.flow_raw is not None:
            key = (id(ws.mag_raw), id(ws.flow_raw), int(t))
            def _build():
                speed = np.sqrt(np.sum(ws.flow_raw[..., t, :] ** 2, axis=-1))
                return np.asarray(ws.mag_raw[..., t], dtype=float) * np.asarray(speed, dtype=float)
            vol = self._cached("scalar_volume", ("pcmra",) + key, _build)
            return vol, "PC-MRA", {"cmap": "gray", "clim": (float(np.nanmin(vol)), float(np.nanmax(vol)))}
        if content_idx == 5 and ws.flow_raw is not None:
            key = (id(ws.flow_raw), int(t))
            vol = self._cached(
                "scalar_volume",
                ("speed",) + key,
                lambda: np.sqrt(np.sum(ws.flow_raw[..., t, :] ** 2, axis=-1)),
            )
            return np.asarray(vol, dtype=float), "Speed (cm/s)", {"cmap": "turbo", "clim": (0.0, float(np.nanmax(vol)) if np.nanmax(vol) > 0 else 1.0)}
        if content_idx == 6:
            return self._get_wss_volume(t)
        if content_idx == 7:
            return self._get_tke_volume(t)
        return None, "", {"cmap": "gray", "clim": None}

    def _get_mask_3d(self):
        ws = self.workspace
        if ws.segmask_3d is not None:
            return ws.segmask_3d
        if ws.segmask_binary is not None and ws.segmask_binary.ndim == 4:
            key = (id(ws.segmask_binary), tuple(int(x) for x in ws.segmask_binary.shape))
            return self._cached("mask_3d", key, lambda: np.any(ws.segmask_binary, axis=3))
        return None

    def _update_value_label(self, vol, title):
        if vol is None:
            self.label_value.setText("Voxel: -   Value: -")
            return
        x, y, z = self.slider_x.value(), self.slider_y.value(), self.slider_z.value()
        try:
            val = float(vol[x, y, z])
            self.label_value.setText(f"Voxel: ({x}, {y}, {z})   {title}: {val:.6g}")
        except Exception:
            self.label_value.setText(f"Voxel: ({x}, {y}, {z})   {title}: -")

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

    def refresh(self):
        ws = self.workspace
        t = int(ws.current_t)
        shape = self._get_volume_shape()
        if shape is None:
            self._remove_colorbar()
            self.canvas.draw_idle()
            return

        cx, cy, cz = self.slider_x.value(), self.slider_y.value(), self.slider_z.value()
        vol, title, style = self._get_scalar_slice(t)
        mask_3d = self._get_mask_3d()
        res = self._get_resolution()

        for ax in [self.ax_ax, self.ax_cor, self.ax_sag]:
            ax.clear()
            ax.set_facecolor("black")
            ax.set_xticks([])
            ax.set_yticks([])

        self._remove_colorbar()
        im = None
        if vol is not None:
            cmap = style.get("cmap", "gray")
            clim = style.get("clim", None)
            if clim is None:
                clim = (float(np.nanmin(vol)), float(np.nanmax(vol)))
            axial = vol[:, :, cz]
            im = self.ax_ax.imshow(axial.T, origin="lower", cmap=cmap, vmin=clim[0], vmax=clim[1], aspect=float(res[1] / res[0]))
            self.ax_ax.axhline(cy, color="lime", linewidth=0.5, alpha=0.5)
            self.ax_ax.axvline(cx, color="lime", linewidth=0.5, alpha=0.5)
            self.ax_ax.plot(cx, cy, "r+", markersize=8, markeredgewidth=1.5)
            self.ax_ax.set_title(f"Axial Z={cz}", color="white", fontsize=8)
            if mask_3d is not None and cz < mask_3d.shape[2]:
                try:
                    self.ax_ax.contour(mask_3d[:, :, cz].astype(float).T, levels=[0.5], colors="cyan", linewidths=0.5, origin="lower")
                except Exception:
                    pass

            coronal = vol[:, cy, :]
            self.ax_cor.imshow(coronal.T, origin="lower", cmap=cmap, vmin=clim[0], vmax=clim[1], aspect=float(res[2] / res[0]))
            self.ax_cor.axhline(cz, color="lime", linewidth=0.5, alpha=0.5)
            self.ax_cor.axvline(cx, color="lime", linewidth=0.5, alpha=0.5)
            self.ax_cor.plot(cx, cz, "r+", markersize=8, markeredgewidth=1.5)
            self.ax_cor.set_title(f"Coronal Y={cy}", color="white", fontsize=8)
            if mask_3d is not None and cy < mask_3d.shape[1]:
                try:
                    self.ax_cor.contour(mask_3d[:, cy, :].astype(float).T, levels=[0.5], colors="cyan", linewidths=0.5, origin="lower")
                except Exception:
                    pass

            sagittal = vol[cx, :, :]
            self.ax_sag.imshow(sagittal.T, origin="lower", cmap=cmap, vmin=clim[0], vmax=clim[1], aspect=float(res[2] / res[1]))
            self.ax_sag.axhline(cz, color="lime", linewidth=0.5, alpha=0.5)
            self.ax_sag.axvline(cy, color="lime", linewidth=0.5, alpha=0.5)
            self.ax_sag.plot(cy, cz, "r+", markersize=8, markeredgewidth=1.5)
            self.ax_sag.set_title(f"Sagittal X={cx}", color="white", fontsize=8)
            if mask_3d is not None and cx < mask_3d.shape[0]:
                try:
                    self.ax_sag.contour(mask_3d[cx, :, :].astype(float).T, levels=[0.5], colors="cyan", linewidths=0.5, origin="lower")
                except Exception:
                    pass

            if self.combo_content.currentIndex() in (6, 7):
                self._scalar_cbar = self.fig.colorbar(im, ax=[self.ax_ax, self.ax_cor, self.ax_sag], fraction=0.025, pad=0.01)
                self._scalar_cbar.ax.tick_params(labelsize=6, colors="white")
                self._scalar_cbar.set_label(title, color="white", fontsize=7)
                try:
                    self._scalar_cbar.outline.set_edgecolor("white")
                except Exception:
                    pass

        self._draw_plane_flow(t)
        self._update_value_label(vol, title)
        self._update_plane_metric_label()
        self.fig.subplots_adjust(left=0.03, right=0.96, top=0.96, bottom=0.03, wspace=0.14, hspace=0.24)
        self.canvas.draw_idle()

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
                self.ax_plane.contour(m_sl.T, levels=[0.5], colors="cyan", linewidths=0.5, origin="lower")
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
        self._remove_colorbar()
        self._cache.clear()
        self.label_value.setText("Voxel: -   Value: -")
        self.label_plane_metric.setText("Plane metrics: -")
        for ax in [self.ax_ax, self.ax_cor, self.ax_sag, self.ax_plane]:
            ax.clear()
            ax.set_facecolor("black")
            ax.set_xticks([])
            ax.set_yticks([])
        self.canvas.draw_idle()
