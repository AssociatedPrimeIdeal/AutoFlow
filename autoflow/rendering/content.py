"""Scalar Content sources shared by the slice and live 3D viewers."""

import numpy as np
from ..config import METRIC_CMAPS, METRIC_DATA_KEYS
from .style import configured_metric_style


class ScalarContentSource:
    def __init__(self, workspace):
        self.workspace = workspace
        self._cache = {}
        self._selected_plane_idx = None

    def _cached(self, group, key, builder, max_items=24):
        bucket = self._cache.setdefault(group, {})
        if key not in bucket:
            if len(bucket) >= max_items:
                bucket.clear()
            bucket[key] = builder()
        return bucket[key]

    def _get_volume_shape(self):
        for source in (self.workspace.mag_raw, self.workspace.flow_raw, self.workspace.segmask_raw):
            if source is not None:
                return np.asarray(source).shape[:3]
        return None

    def _get_resolution(self):
        res = np.asarray(self.workspace.resolution, dtype=float).reshape(-1)[:3]
        return np.where(res > 0, res, 1.0)

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

    def scalar_volume(self, t, content_idx):
        ws = self.workspace
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
            mag = np.asarray(ws.mag_raw)
            vol = np.asarray(mag[..., min(int(t), mag.shape[3] - 1)] if mag.ndim == 4 else mag, dtype=np.float32)
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

