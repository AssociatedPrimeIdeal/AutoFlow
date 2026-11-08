"""Labelled plane handles shared by native Qt and the SSH plotter."""

import numpy as np
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
        self.axis_labels = []
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
        for text, color in (("U", "#ff6b6b"), ("V", "#69db7c"), ("N", "#74c0fc")):
            mesh = pv.Arrow(tip_length=0.23, tip_radius=0.06, shaft_radius=0.016)
            mapper = pv._vtk.vtkPolyDataMapper()
            mapper.SetInputData(mesh)
            actor = pv._vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(pv.Color(color).float_rgb)
            actor.GetProperty().LightingOff()
            actor.PickableOff()
            actor.UseBoundsOff()
            self.renderer.AddActor(actor)
            self.guides.append((actor, mesh))
            label = vtkBillboardTextActor3D()
            label.SetInput(text)
            label.SetDisplayOffset(5, 5)
            label.GetTextProperty().SetFontSize(14)
            label.GetTextProperty().SetColor(pv.Color(color).float_rgb)
            label.PickableOff()
            label.UseBoundsOff()
            self.renderer.AddActor(label)
            self.axis_labels.append(label)
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

    def update(self, center, axis_u, axis_v, distance, normal):
        for (_, rep), label in zip(self.handles, self.labels):
            rep.SetWorldPosition(center)
            label.SetPosition(center)
        for (actor, _), label, direction in zip(self.guides, self.axis_labels, (axis_u, axis_v, normal)):
            direction = np.asarray(direction, dtype=float)
            direction = direction / np.linalg.norm(direction)
            side = np.cross(direction, np.eye(3)[np.argmin(np.abs(direction))])
            side /= np.linalg.norm(side)
            matrix = pv._vtk.vtkMatrix4x4()
            basis = np.column_stack((direction, side, np.cross(direction, side))) * distance
            for row in range(3):
                for column in range(3):
                    matrix.SetElement(row, column, basis[row, column])
                matrix.SetElement(row, 3, center[row])
            actor.SetUserMatrix(matrix)
            label.SetPosition(np.asarray(center) + direction * distance)

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
        for actor in self.labels + self.axis_labels + [actor for actor, _ in self.guides]:
            self.renderer.RemoveActor(actor)
        self.scene._hud_renderer.RemoveViewProp(self.help)
        self.handles.clear()
        self.labels.clear()
        self.axis_labels.clear()
        self.guides.clear()
        self.plotter._autoflow_plane_widget_dispatch = None
