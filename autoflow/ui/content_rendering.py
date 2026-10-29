"""Browser-owned scalar volumes and draggable orthogonal slice axes."""

import numpy as np
import pyvista as pv
from vtkmodules.vtkInteractionWidgets import vtkHandleWidget, vtkPointHandleRepresentation3D

from ..core.models import ObjectKind
from ..rendering.content import ScalarContentSource


def content_data_key(content):
    return {3: "magnitude_volume", 4: "pcmra_volume"}.get(int(content), f"content_volume_{int(content)}")


class ContentRenderingMixin:
    def _initialize_content_rendering(self):
        self._content_source = ScalarContentSource(self.workspace)
        self._content_widgets = []
        self._content_axis_actors = []
        self._content_widget_uid = None
        self._content_selected_uid = None
        self._content_position_callback = None
        self._content_widget_dragging = False
        self._content_slice_templates = {}

    def _ensure_content_source(self):
        """Input loading replaces Workspace; never retain a previous case's source."""
        if self._content_source.workspace is not self.workspace:
            self._content_source = ScalarContentSource(self.workspace)
            self._content_slice_templates.clear()
        return self._content_source

    @staticmethod
    def is_content_render(obj):
        key = str(getattr(obj, "data_key", ""))
        return key in {"magnitude_volume", "pcmra_volume"} or key.startswith("content_volume_")

    def content_render_settings(self, data_key):
        content = {"magnitude_volume": 3, "pcmra_volume": 4}.get(data_key)
        if content is None:
            content = int(data_key.rsplit("_", 1)[-1])
        defaults = dict(content=content, mode="composite", planes=[True, True, True],
                        positions=[0.5, 0.5, 0.5], handles=True, threshold=0.05, shade=False,
                        plane_index=None)
        return self.workspace.render_settings.setdefault("content_renders", {}).setdefault(data_key, defaults)

    def add_content_render(self, content, *, plane_index=None):
        """Create or reveal one Browser layer per Content; repeated clicks reuse it."""
        self._ensure_content_source()
        key = content_data_key(content)
        cfg = self.content_render_settings(key)
        self._content_source._selected_plane_idx = plane_index
        values, title, style = self._content_source.scalar_volume(self.workspace.current_t, int(content))
        if values is None:
            self.logger("Selected Content has no volume to render")
            return None
        cfg["plane_index"] = plane_index
        obj = next((obj for obj in self.workspace.scene_objects.values() if obj.data_key == key), None)
        if obj is None:
            uid = self.workspace.add_object(name=f"{title} · 3D Volume", kind=ObjectKind.AUX,
                                            data_key=key, visible=True, opacity=0.85,
                                            scalars=title, cmap=style["cmap"], dynamic=True)
            obj = self.workspace.scene_objects[uid]
        obj.visible = True
        obj.scalars = title
        self.invalidate_cache(key)
        self.readd_object(obj, refresh_scalar_bar=False)
        self.select_content_render(obj.uid)
        getattr(self, "request_render", self.plotter.render)()
        return obj

    def update_content_render(self, uid, **settings):
        obj = self.workspace.scene_objects.get(uid)
        if obj is None or not self.is_content_render(obj):
            return
        self.content_render_settings(obj.data_key).update(settings)
        if "cmap" in settings:
            obj.cmap = settings["cmap"]
        self._clear_content_widgets()
        self.readd_object(obj, refresh_scalar_bar=False)
        self._sync_content_widgets()
        getattr(self, "request_render", self.plotter.render)()

    def _content_slice_dataset(self, key, grid):
        cfg = self.content_render_settings(key)
        if cfg["mode"] != "slices":
            return grid
        # The padded grid's interior samples are the original voxel centers.
        shape = np.asarray(grid.dimensions, dtype=int) - 2
        spacing = np.asarray(self.workspace.resolution, dtype=float)
        origin = np.asarray(self.workspace.origin, dtype=float)
        center = origin + (0.5 + np.asarray(cfg["positions"]) * np.maximum(shape - 1, 0)) * spacing
        name = grid.active_scalars_name
        dims = tuple(grid.dimensions)
        values = np.asarray(grid.point_data[name]).reshape(dims, order="F")
        points, cells, scalars = [], [], []
        offset = 0
        for axis, enabled in enumerate(cfg["planes"]):
            if not enabled:
                continue
            key = (dims, tuple(grid.spacing), tuple(grid.origin), axis)
            template = self._content_slice_templates.get(key)
            if template is None:
                others = [index for index in range(3) if index != axis]
                a, b = np.meshgrid(np.arange(dims[others[0]]), np.arange(dims[others[1]]), indexing="ij")
                coords = np.zeros((a.size, 3), dtype=float)
                for indices, other in ((a, others[0]), (b, others[1])):
                    coords[:, other] = grid.origin[other] + indices.ravel(order="F") * grid.spacing[other]
                width, height = a.shape
                i = (np.arange(width - 1)[:, None] + width * np.arange(height - 1)[None, :]).ravel()
                faces = np.column_stack((np.full(i.size, 4), i, i + 1, i + 1 + width, i + width))
                template = coords, faces
                if len(self._content_slice_templates) >= 6:
                    self._content_slice_templates.clear()
                self._content_slice_templates[key] = template
            coords, faces = template[0].copy(), template[1].copy()
            coords[:, axis] = center[axis]
            faces[:, 1:] += offset
            offset += len(coords)
            # Axis-aligned sampling only touches the two neighboring 2D arrays;
            # vtkCutter would scan the full 3D grid for every mouse movement.
            position = 1 + float(cfg["positions"][axis]) * (dims[axis] - 3)
            lower = int(np.floor(position))
            upper = min(lower + 1, dims[axis] - 2)
            weight = np.float32(position - lower)
            index = [slice(None)] * 3
            index[axis] = lower
            cut = values[tuple(index)]
            if weight > 0:
                index[axis] = upper
                cut = (1 - weight) * cut + weight * values[tuple(index)]
            points.append(coords)
            cells.append(faces.ravel())
            scalars.append(cut.ravel(order="F"))
        if not points:
            return None
        result = pv.PolyData(np.concatenate(points), np.concatenate(cells))
        result.point_data[name] = np.concatenate(scalars)
        result.field_data.update(dict(grid.field_data))
        return result

    def _build_content_volume(self, key):
        self._ensure_content_source()
        cfg = self.content_render_settings(key)
        self._content_source._selected_plane_idx = cfg.get("plane_index")
        values, title, style = self._content_source.scalar_volume(self.workspace.current_t, cfg["content"])
        if values is None:
            return None
        def build():
            clean = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            grid = self._voxel_centered_grid(clean, title)
            if style.get("clim") is not None:
                grid.field_data["Content display range"] = style["clim"]
            return grid
        signature = ""
        if cfg["content"] == 27 and cfg.get("plane_index") is not None:
            idx = cfg["plane_index"]
            signature = f"_plane_{idx}_{tuple(self.workspace.planes[idx].normal)}"
        grid = self._cached(key + signature, self.workspace.current_t, build)
        return self._content_slice_dataset(key, grid)

    def select_content_render(self, uid):
        self._content_selected_uid = uid
        self._sync_content_widgets()

    def _clear_content_widgets(self):
        if self._content_widget_dragging:
            self.set_interaction_preview(False, render=False)
        self._content_widget_dragging = False
        for widget, _, _ in self._content_widgets:
            widget.Off()
            widget.RemoveAllObservers()
            widget.SetInteractor(None)
        self._content_widgets.clear()
        for actor in self._content_axis_actors:
            self._detach_actor(actor)
        self._content_axis_actors.clear()
        self._content_widget_uid = None

    def _content_geometry(self, obj):
        # All Contents use the loaded image geometry, regardless of render mode.
        shape = self._ensure_content_source()._get_volume_shape()
        if shape is None:
            return None
        spacing = np.asarray(self.workspace.resolution, dtype=float)
        offset = 0.0 if self.workspace.segmentation_labels_1mm is not None else 0.5
        lower = np.asarray(self.workspace.origin, dtype=float) + offset * spacing
        span = np.maximum(np.asarray(shape) - 1, 0) * spacing
        return lower, span

    def _sync_content_widgets(self):
        obj = self.workspace.scene_objects.get(self._content_selected_uid)
        enabled = obj is not None and self.is_content_render(obj) and obj.visible
        cfg = self.content_render_settings(obj.data_key) if enabled else None
        enabled = enabled and cfg["mode"] == "slices" and cfg["handles"] and any(cfg["planes"])
        if not enabled:
            self._clear_content_widgets()
            return
        if self._content_widget_uid == obj.uid:
            return
        self._clear_content_widgets()
        geometry = self._content_geometry(obj)
        iren = getattr(self.plotter, "iren", None)
        interactor = getattr(iren, "interactor", iren)
        if geometry is None or interactor is None:
            return
        self._content_widget_uid = obj.uid
        self.plotter._autoflow_content_widget_dispatch = self._dispatch_content_widget_event
        lower, span = geometry
        center = lower + np.asarray(cfg["positions"]) * span
        length = max(float(np.max(span)) * 0.15, 1.0)
        renderer = self.plotter.renderer
        if self._selection_on_top(obj.uid):
            self._ensure_scene_layers()
            renderer = self._selection_renderer
        for axis, color in enumerate(("#ff595e", "#63d471", "#4ea8ff")):
            if not cfg["planes"][axis] or span[axis] <= 0:
                continue
            rep = vtkPointHandleRepresentation3D()
            rep.SetHandleSize(13)
            rep.SetTranslationAxis(axis)
            rep.SetRenderer(renderer)
            rep.GetProperty().SetColor(pv.Color(color).float_rgb)
            rep.GetSelectedProperty().SetColor(1, 1, 1)
            widget = vtkHandleWidget()
            widget.SetInteractor(interactor)
            widget.SetDefaultRenderer(renderer)
            widget.SetCurrentRenderer(renderer)
            widget.SetRepresentation(rep)
            widget.AddObserver("InteractionEvent", lambda w, e, a=axis: self._on_content_axis_drag(obj.uid, a, w))
            widget.AddObserver("StartInteractionEvent", lambda w, e: self._content_interaction_state(True))
            widget.AddObserver("EndInteractionEvent", lambda w, e: self._content_interaction_state(False))
            widget.On()
            self._content_widgets.append((widget, rep, axis))
            line = pv.Line(center, center + np.eye(3)[axis] * length)
            actor = self.plotter.add_mesh(self._transform_display_dataset(line), color=color, line_width=4,
                                          lighting=False, pickable=False, render=False, reset_camera=False)
            self._content_axis_actors.append(actor)
            self._place_selected_prop(actor, obj.uid, self.plotter.renderer)
        self._position_content_widgets(obj)

    def _content_interaction_state(self, active):
        self._content_widget_dragging = active
        self.set_interaction_preview(active)

    def _dispatch_content_widget_event(self, event):
        """Forward handle gestures through the SSH QWidget without moving the camera."""
        if not self._content_widgets:
            return False
        iren = getattr(self.plotter, "iren", None)
        interactor = getattr(iren, "interactor", iren)
        active = self._content_widget_dragging
        if event == "press":
            x, y = interactor.GetEventPosition()
            if not any(rep.ComputeInteractionState(x, y) for _, rep, _ in self._content_widgets):
                return False
            interactor.InvokeEvent("LeftButtonPressEvent")
            return self._content_widget_dragging
        if active:
            interactor.InvokeEvent("MouseMoveEvent" if event == "move" else "LeftButtonReleaseEvent")
        return active

    def _position_content_widgets(self, obj):
        geometry = self._content_geometry(obj)
        if geometry is None:
            return
        lower, span = geometry
        center = lower + np.asarray(self.content_render_settings(obj.data_key)["positions"]) * span
        length = max(float(np.max(span)) * 0.15, 1.0)
        for (_, rep, axis), actor in zip(self._content_widgets, self._content_axis_actors):
            endpoint = center + np.eye(3)[axis] * length
            rep.SetWorldPosition(self.world_to_display_point(endpoint))
            actor.GetMapper().dataset.points[:] = np.stack((self.world_to_display_point(center), self.world_to_display_point(endpoint)))

    def _on_content_axis_drag(self, uid, axis, widget):
        obj = self.workspace.scene_objects.get(uid)
        if obj is None or not obj.visible:
            return
        lower, span = self._content_geometry(obj)
        endpoint = self.display_to_world_point(widget.GetRepresentation().GetWorldPosition())
        length = max(float(np.max(span)) * 0.15, 1.0)
        self.set_content_axis_position(uid, axis, (endpoint[axis] - length - lower[axis]) / max(span[axis], 1e-12))

    def set_content_axis_position(self, uid, axis, fraction):
        obj = self.workspace.scene_objects.get(uid)
        if obj is None or not self.is_content_render(obj):
            return
        cfg = self.content_render_settings(obj.data_key)
        cfg["positions"][axis] = float(np.clip(fraction, 0.0, 1.0))
        self._update_dynamic_object(obj)
        self._position_content_widgets(obj)
        if self._content_position_callback is not None:
            self._content_position_callback(uid, list(cfg["positions"]))
        getattr(self, "request_render", self.plotter.render)()
