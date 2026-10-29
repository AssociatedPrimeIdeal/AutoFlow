"""Browser selection rendering, independent of anatomical depth occlusion."""

import numpy as np
from pyvista import _vtk

from ..core.models import ObjectKind


class SceneLayersMixin:
    def _initialize_scene_layers(self):
        self._selected_layer_uids = set()
        self._selection_renderer = None
        self._hud_renderer = None
        self._front_props = {}
        self._hud_props = []

    def _ensure_scene_layers(self):
        window = getattr(self.plotter, "render_window", None)
        main = getattr(self.plotter, "renderer", None)
        if window is None or main is None:
            return
        window.SetNumberOfLayers(max(4, window.GetNumberOfLayers()))
        for attribute, layer in (("_selection_renderer", 2), ("_hud_renderer", 3)):
            renderer = getattr(self, attribute)
            if renderer is None:
                renderer = _vtk.vtkRenderer()
                renderer.SetLayer(layer)
                renderer.SetInteractive(False)
                renderer.SetPreserveColorBuffer(True)
                renderer.SetPreserveDepthBuffer(False)
                renderer.SetBackgroundAlpha(0.0)
                window.AddRenderer(renderer)
                setattr(self, attribute, renderer)
            renderer.SetActiveCamera(main.GetActiveCamera())
            if attribute == "_selection_renderer":
                # Selected anatomy uses the same material even when drawn in front.
                renderer.RemoveAllLights()
                for light in main.GetLights():
                    renderer.AddLight(light)


    def set_selected_layers(self, uids, on_top=None, render=True):
        self._selected_layer_uids = set(uids or ())
        if on_top is not None:
            self.workspace.render_settings["selected_layer_on_top"] = bool(on_top)
            self._clear_content_widgets()
        self._ensure_scene_layers()
        for obj in self.workspace.scene_objects.values():
            self._place_selected_object(obj)
        self._place_selected_highlights()
        self._sync_content_widgets()
        if render:
            self.plotter.render()

    def _selection_on_top(self, uid):
        return (uid in self._selected_layer_uids
                and self.workspace.render_settings.get("selected_layer_on_top", True))

    def _place_selected_prop(self, actor, uid, home):
        if actor is None or home is None:
            return
        key = id(actor)
        if self._selection_on_top(uid):
            self._ensure_scene_layers()
            if key not in self._front_props:
                home.RemoveViewProp(actor)
                self._selection_renderer.AddViewProp(actor)
                self._front_props[key] = (actor, home)
        elif key in self._front_props:
            self._selection_renderer.RemoveViewProp(actor)
            self._front_props.pop(key)[1].AddViewProp(actor)

    def _place_selected_object(self, obj):
        home = self._ensure_plane_overlay_renderer() if obj.kind == ObjectKind.PLANE else self.plotter.renderer
        self._place_selected_prop(obj.actor, obj.uid, home)
        self._place_selected_prop(getattr(obj, "label_actor", None), obj.uid, home)

    def _place_selected_highlights(self):
        home = self._ensure_plane_overlay_renderer()
        for kind in ("plane", "path"):
            self._place_selected_prop(getattr(self, f"_highlight_{kind}_actor"),
                                      getattr(self, f"_highlight_{kind}_uid"), home)

    def _forget_front_prop(self, actor):
        if actor is not None:
            self._front_props.pop(id(actor), None)
            if self._selection_renderer is not None:
                self._selection_renderer.RemoveViewProp(actor)

    def _clear_scene_layers(self):
        self._selected_layer_uids.clear()
        self._front_props.clear()
        self._hud_props.clear()
        for renderer in (self._selection_renderer, self._hud_renderer):
            if renderer is not None:
                renderer.RemoveAllViewProps()

    def _clear_hud_scalar_bars(self):
        if self._hud_renderer is not None:
            for actor in self._hud_props:
                self._hud_renderer.RemoveViewProp(actor)
        self._hud_props.clear()

    def _place_hud_scalar_bar(self, actor):
        if actor is None:
            return
        self._ensure_scene_layers()
        self.plotter.renderer.RemoveViewProp(actor)
        self._hud_renderer.AddViewProp(actor)
        self._hud_props.append(actor)

    def _visible_scene_bounds(self):
        bounds = []
        for actor in self._tracked_actors.values():
            if actor is None or not actor.GetVisibility():
                continue
            value = np.asarray(actor.GetBounds(), dtype=float)
            if value.shape == (6,) and np.all(np.isfinite(value)) and np.all(value[::2] <= value[1::2]):
                bounds.append(value)
        if not bounds:
            return None
        rows = np.asarray(bounds)
        result = np.empty(6)
        result[::2], result[1::2] = rows[:, ::2].min(axis=0), rows[:, 1::2].max(axis=0)
        return tuple(result)
