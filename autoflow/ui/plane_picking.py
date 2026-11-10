"""Click selection without stealing camera, marker or editing gestures."""

import numpy as np
import pyvista as pv
from PySide6 import QtCore, QtWidgets

from ..core.models import ObjectKind


class PlanePickingFilter(QtCore.QObject):
    def __init__(self, scene):
        super().__init__(scene.plotter)
        self.scene = scene
        self.press = None

    def _blocked(self):
        scene = self.scene
        window = scene.plotter.window()
        return (scene._plane_selection_locked or getattr(window, '_edit_mode', None) is not None
                or getattr(scene, '_content_widget_dragging', False))

    def _vtk_position(self, position):
        widget = self.scene.plotter
        width, height = widget.render_window.GetSize()
        return (position.x() * width / max(1, widget.width()),
                height - 1 - position.y() * height / max(1, widget.height()))

    def _reserved(self, position):
        scene = self.scene
        if scene._scalar_bar_object_at(position.x(), position.y()) is not None:
            return True
        marker = scene._orientation_marker
        if marker is not None and scene._axes_shown and marker.widget.GetEnabled():
            viewport = np.asarray(marker.widget.GetViewport()).reshape(2, 2)
            size = np.asarray(scene.plotter.render_window.GetSize())
            point = np.asarray(self._vtk_position(position))
            return bool(np.all((point >= viewport[0] * size) & (point <= viewport[1] * size)))
        return False

    def eventFilter(self, watched, event):  # noqa: N802
        kind = event.type()
        if kind == QtCore.QEvent.Type.MouseButtonPress:
            self.press = None
            if (event.button() not in (QtCore.Qt.LeftButton, QtCore.Qt.RightButton)
                    or self._blocked() or self._reserved(event.position())
                    or event.modifiers() & (QtCore.Qt.ShiftModifier | QtCore.Qt.AltModifier)):
                return False
            self.press = (event.position().toPoint(), event.button(), event.modifiers())
        elif kind == QtCore.QEvent.Type.MouseMove and self.press is not None:
            distance = (event.position().toPoint() - self.press[0]).manhattanLength()
            if self._blocked() or distance >= QtWidgets.QApplication.startDragDistance():
                self.press = None
        elif kind == QtCore.QEvent.Type.MouseButtonRelease and self.press is not None:
            position, button, modifiers = self.press
            self.press = None
            if (event.button() != button or self._blocked()
                    or (event.position().toPoint() - position).manhattanLength()
                    >= QtWidgets.QApplication.startDragDistance()):
                return False
            x, y = self._vtk_position(event.position())
            global_position = event.globalPosition().toPoint()
            additive = bool(modifiers & QtCore.Qt.ControlModifier)
            # Let the widget finish VTK's button-up first. A menu must never
            # leave the interactor in its camera-drag state.
            QtCore.QTimer.singleShot(0, self, lambda: self._click(x, y, button, additive, global_position))
        return False

    def _click(self, x, y, button, additive, position):
        scene = self.scene
        if self._blocked():
            return
        kind, uid, index = pick_plane_or_path(scene, x, y)
        if button == QtCore.Qt.RightButton:
            callback = getattr(scene, '_plane_context_callback', None)
            if kind == 'plane' and callback is not None:
                callback(uid, index, position)
            elif kind == 'path' and scene._path_pick_callback is not None:
                scene._path_pick_callback(uid, index)
            return
        if kind == 'plane' and scene._plane_pick_callback is not None:
            if additive:
                scene._plane_pick_callback(uid, index, additive=True)
            else:
                scene._plane_pick_callback(uid, index)
        elif kind == 'path' and scene._path_pick_callback is not None:
            scene._path_pick_callback(uid, index)
        elif not additive and scene._plane_pick_callback is not None:
            scene._plane_pick_callback(None, None)


def pick_plane_or_path(scene, x, y):
    """Keep displayed plane faces ahead of overlapping path/volume actors."""
    uid, index = scene._plane_uid_at_display_position(x, y)
    if uid is not None:
        return 'plane', uid, index
    picker = pv._vtk.vtkCellPicker()
    picker.SetTolerance(0.005)
    for renderer in (scene._selection_renderer, scene._plane_overlay_renderer, scene.plotter.renderer):
        if renderer is None or not picker.Pick(float(x), float(y), 0.0, renderer):
            continue
        actor = picker.GetActor()
        for uid, plane in scene.workspace.scene_objects.items():
            if plane.kind == ObjectKind.PLANE and plane.actor is not None and actor == plane.actor:
                from .viewer import _parse_indexed_data_key
                return 'plane', uid, _parse_indexed_data_key(plane.data_key, 'plane')
        position = picker.GetPickPosition()
        uid, index = scene.find_plane_uid_at_position(position)
        if uid is not None:
            return 'plane', uid, index
        uid, index = scene.find_path_uid_at_position(position)
        if uid is not None:
            return 'path', uid, index
    return None, None, None
