"""Labelled plane handles shared by native Qt and the SSH plotter."""

import pyvista as pv
from vtkmodules.vtkInteractionWidgets import vtkHandleWidget, vtkSphereHandleRepresentation
from vtkmodules.vtkRenderingCore import vtkBillboardTextActor3D, vtkTextActor


class PlaneDragHandles:
    def __init__(self, plotter, scene, callbacks, on_start, on_end):
        self.plotter = plotter
        self.scene = scene
        self.dragging = False
        self.handles = []
        self.guides = []
        self.labels = []
        scene._ensure_scene_layers()
        self.renderer = scene._selection_renderer
        iren = getattr(plotter.iren, "interactor", plotter.iren)
        for callback in callbacks:
            color, text = "cyan", "Move"
            rep = vtkSphereHandleRepresentation()
            rep.SetHandleSize(18)
            rep.SetTolerance(12)
            rep.SetRenderer(self.renderer)
            rep.GetProperty().SetColor(pv.Color(color).float_rgb)
            rep.GetSelectedProperty().SetColor(1, 1, 1)
            widget = vtkHandleWidget()
            widget.SetInteractor(iren)
            widget.SetDefaultRenderer(self.renderer)
            widget.SetCurrentRenderer(self.renderer)
            widget.SetRepresentation(rep)
            widget.SetPriority(0.9)
            widget.KeyPressActivationOff()
            widget.AddObserver("InteractionEvent", lambda w, e, cb=callback: cb(w.GetRepresentation().GetWorldPosition()))
            widget.AddObserver("StartInteractionEvent", lambda w, e: self._start(on_start))
            widget.AddObserver("EndInteractionEvent", lambda w, e: self._end(on_end))
            self.handles.append((widget, rep))
            label = vtkBillboardTextActor3D()
            label.SetInput(text)
            label.SetDisplayOffset(14, 10)
            label.PickableOff()
            prop = label.GetTextProperty()
            prop.SetFontSize(15)
            prop.SetBold(True)
            prop.SetColor(pv.Color(color).float_rgb)
            prop.SetBackgroundColor(0.04, 0.06, 0.08)
            prop.SetBackgroundOpacity(0.8)
            self.renderer.AddActor(label)
            self.labels.append(label)
        self.help = vtkTextActor()
        self.help.SetInput("Drag a cyan center to move\nRotate in the lower 2D views\nApply Plane to save | Esc to cancel")
        self.help.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        self.help.SetPosition(0.02, 0.97)
        prop = self.help.GetTextProperty()
        prop.SetFontSize(14)
        prop.SetVerticalJustificationToTop()
        prop.SetColor(1, 1, 1)
        prop.SetBackgroundColor(0.04, 0.06, 0.08)
        prop.SetBackgroundOpacity(0.75)
        self.help.PickableOff()
        self.scene._hud_renderer.AddViewProp(self.help)
        plotter._autoflow_plane_widget_dispatch = self.dispatch

    def _start(self, callback):
        self.dragging = True
        self.scene.set_interaction_preview(True, render=False)
        callback()

    def _end(self, callback):
        self.dragging = False
        self.scene.set_interaction_preview(False, render=False)
        callback()

    def enable(self):
        for widget, _ in self.handles:
            widget.On()

    def set_mode(self, mode):
        # Rotation lives in the 2D viewer; the 3D handle always translates.
        for (widget, _), label in zip(self.handles, self.labels):
            widget.SetEnabled(True)
            label.SetVisibility(True)

    def update(self, center, axis_u, axis_v, distance):
        for (_, rep), label in zip(self.handles, self.labels):
            rep.SetWorldPosition(center)
            label.SetPosition(center)

    def dispatch(self, event):
        """Let the remote QWidget send handle gestures before camera gestures."""
        iren = getattr(self.plotter.iren, "interactor", self.plotter.iren)
        active = self.dragging
        if event == "press":
            x, y = iren.GetEventPosition()
            if not any(widget.GetEnabled() and rep.ComputeInteractionState(x, y)
                       for widget, rep in self.handles):
                return False
            iren.InvokeEvent("LeftButtonPressEvent")
            return self.dragging
        if active:
            iren.InvokeEvent("MouseMoveEvent" if event == "move" else "LeftButtonReleaseEvent")
        return active

    def close(self):
        self.scene.set_interaction_preview(False, render=False)
        self.dragging = False
        for widget, _ in self.handles:
            widget.Off()
            widget.RemoveAllObservers()
            widget.SetInteractor(None)
        for actor in self.labels + [actor for actor, _ in self.guides]:
            self.renderer.RemoveActor(actor)
        self.scene._hud_renderer.RemoveViewProp(self.help)
        self.handles.clear()
        self.labels.clear()
        self.guides.clear()
        self.plotter._autoflow_plane_widget_dispatch = None
