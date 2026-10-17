"""Temporary interactive rendering quality, restored on mouse release."""

import numpy as np


class InteractionRenderingMixin:
    def _initialize_interaction_rendering(self):
        self._interaction_preview_active = False
        self._interaction_mapper_states = {}
        self._interaction_view_state = None
        self._interaction_observers = []

    def _ensure_interaction_rendering(self):
        for target, observer in self._interaction_observers:
            target.RemoveObserver(observer)
        self._interaction_observers.clear()
        iren = getattr(self.plotter, "iren", None)
        interactor = getattr(iren, "interactor", iren)
        if interactor is not None:
            style = interactor.GetInteractorStyle()
            for target in (interactor, style):
                if target is not None:
                    for event, active in (("StartInteractionEvent", True), ("EndInteractionEvent", False)):
                        observer = target.AddObserver(event, lambda o, e, a=active: self.set_interaction_preview(a))
                        self._interaction_observers.append((target, observer))
        self.plotter._autoflow_interaction_preview = self.set_interaction_preview

    def _apply_interaction_mapper_quality(self, obj):
        if not self._interaction_preview_active or obj.actor is None or not self._is_volume_object(obj):
            return
        mapper = obj.actor.GetMapper()
        key = id(mapper)
        if key in self._interaction_mapper_states:
            return
        original = {}
        for name in ("AutoAdjustSampleDistances", "InteractiveAdjustSampleDistances", "SampleDistance", "ImageSampleDistance"):
            getter = getattr(mapper, "Get" + name, None)
            if getter is not None:
                original[name] = getter()
        self._interaction_mapper_states[key] = (mapper, original)
        spacing = max(float(np.min(np.asarray(self.workspace.resolution, dtype=float))), 0.1)
        if "InteractiveAdjustSampleDistances" in original:
            mapper.SetInteractiveAdjustSampleDistances(False)
        if "AutoAdjustSampleDistances" in original:
            mapper.SetAutoAdjustSampleDistances(False)
        if "ImageSampleDistance" in original:
            mapper.SetImageSampleDistance(max(3.0, original["ImageSampleDistance"]))
        if "SampleDistance" in original:
            mapper.SetSampleDistance(max(2.0 * spacing, original["SampleDistance"]))

    def set_interaction_preview(self, active, *, render=True):
        active = bool(active)
        if active == self._interaction_preview_active:
            return
        self._interaction_preview_active = active
        renderer = self.plotter.renderer
        window = self.plotter.render_window
        resolution = getattr(self.plotter, "_set_interaction_resolution", None)
        if callable(resolution):
            resolution(active)
        if active:
            self._interaction_view_state = (renderer.GetPass(), renderer.GetUseFXAA(), window.GetDesiredUpdateRate())
            # SSAA on EGL/software backends renders a larger image internally.
            # Keep that cost for the stationary view, not every drag movement.
            renderer.SetPass(None)
            renderer.SetUseFXAA(False)
            window.SetDesiredUpdateRate(20.0)
            for obj in self.workspace.scene_objects.values():
                if obj.visible:
                    self._apply_interaction_mapper_quality(obj)
        else:
            for mapper, original in self._interaction_mapper_states.values():
                for name, value in original.items():
                    getattr(mapper, "Set" + name)(value)
            self._interaction_mapper_states.clear()
            if self._interaction_view_state is not None:
                render_pass, fxaa, update_rate = self._interaction_view_state
                renderer.SetPass(render_pass)
                renderer.SetUseFXAA(fxaa)
                window.SetDesiredUpdateRate(update_rate)
            self._interaction_view_state = None
            if render:
                self.plotter.render()

    def _close_interaction_rendering(self):
        self.set_interaction_preview(False, render=False)
        for target, observer in self._interaction_observers:
            target.RemoveObserver(observer)
        self._interaction_observers.clear()
