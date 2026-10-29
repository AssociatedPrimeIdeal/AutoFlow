"""Bounded QC point actors and marker-only left-click handling."""

import numpy as np
import pyvista as pv
from PySide6 import QtCore, QtWidgets

from ..qc_locations import issue_anchors


class QCOverlay(QtCore.QObject):
    def __init__(self, scene, callback, parent):
        super().__init__(parent)
        self.scene, self.callback = scene, callback
        self.actors, self.records, self.points = {}, [], np.empty((0, 3))
        self.actor_layers = {}
        self.enabled = False
        self.press_position = None
        self.hits = []
        self.widget = scene.plotter
        if isinstance(self.widget, QtWidgets.QWidget):
            self.widget.installEventFilter(self)

    def update(self, report, enabled=True, on_top=False):
        self.enabled = enabled
        self.on_top = on_top
        if not enabled:
            self.press_position, self.hits = None, []
        self.records = issue_anchors(self.scene.workspace, report) if report else []
        local = np.array([r["point_mm"] for r in self.records], dtype=float).reshape(-1, 3)
        self.points = self.scene.world_to_display_points(local + self.scene.workspace.origin) if len(local) else local
        with self.scene.render_batch():
            for status, color in (("fail", "#e03131"), ("warn", "#ffd43b"), ("stale", "#868e96")):
                indices = [i for i, record in enumerate(self.records) if record["status"] == status]
                data = pv.PolyData(self.points[indices])
                if status not in self.actors and len(indices):
                    self.actors[status] = self.scene.plotter.add_mesh(data, color=color,
                        style="points", point_size=16, render_points_as_spheres=True,
                        pickable=False, reset_camera=False, render=False, name="qc_"+status)
                elif status in self.actors:
                    self.actors[status].mapper.dataset = data
                if status in self.actors:
                    self.actors[status].SetVisibility(bool(enabled and indices))
                    if self.actor_layers.get(status, False) != on_top:
                        if on_top:
                            self.scene._move_actor_to_plane_overlay(self.actors[status])
                        else:
                            overlay = self.scene._plane_overlay_renderer
                            if overlay is not None:
                                overlay.RemoveActor(self.actors[status])
                            self.scene.plotter.renderer.AddActor(self.actors[status])
                        self.actor_layers[status] = on_top
            self.scene.request_render()

    def _screen_hits(self, position):
        renderer = getattr(self.scene.plotter, "renderer", None)
        rw = getattr(self.scene.plotter, "render_window", None)
        if renderer is None or rw is None or not self.widget.width() or not self.widget.height():
            return []
        width, height = rw.GetSize()
        hits = []
        for index, point in enumerate(self.points):
            renderer.SetWorldPoint(*point, 1.)
            renderer.WorldToDisplay()
            x, y, depth = renderer.GetDisplayPoint()
            # Render-window pixels and Qt logical pixels differ on high DPI.
            x, y = x*self.widget.width()/width, self.widget.height()-y*self.widget.height()/height
            if 0 <= depth <= 1 and np.hypot(position.x()-x, position.y()-y) <= 12:
                if not self.on_top:
                    # Reject an occluded anchor. Allow the front of the
                    # rendered point sphere itself (ten logical pixels).
                    px, py = x*width/self.widget.width(), (self.widget.height()-y)*height/self.widget.height()
                    zbuffer = renderer.GetZ(int(px), int(py))
                    renderer.SetDisplayPoint(px+10*width/self.widget.width(), py, depth)
                    renderer.DisplayToWorld()
                    edge = np.asarray(renderer.GetWorldPoint())
                    radius = np.linalg.norm(edge[:3]/edge[3]-point)
                    renderer.SetDisplayPoint(px, py, zbuffer)
                    renderer.DisplayToWorld()
                    front = np.asarray(renderer.GetWorldPoint())
                    camera = np.asarray(renderer.GetActiveCamera().GetPosition())
                    ray = (camera-point)/max(np.linalg.norm(camera-point), 1e-12)
                    if np.dot(front[:3]/front[3]-point, ray) > radius:
                        continue
                hits.append(self.records[index]["issue_id"])
        return list(dict.fromkeys(hits))

    def eventFilter(self, watched, event):
        if not self.enabled or self.scene._plane_selection_locked:
            return False
        kind = event.type()
        if kind == QtCore.QEvent.Type.MouseButtonPress and event.button() == QtCore.Qt.MouseButton.LeftButton:
            if event.modifiers() != QtCore.Qt.KeyboardModifier.NoModifier:
                return False
            self.hits = self._screen_hits(event.position())
            self.press_position = event.position() if self.hits else None
            return bool(self.hits)
        if kind == QtCore.QEvent.Type.MouseMove and self.press_position is not None:
            return True
        if kind == QtCore.QEvent.Type.MouseButtonRelease and self.press_position is not None:
            if (event.position()-self.press_position).manhattanLength() < QtWidgets.QApplication.startDragDistance():
                self.callback(self.hits)
            self.press_position, self.hits = None, []
            return True
        return False

    def close(self):
        if isinstance(self.widget, QtWidgets.QWidget):
            self.widget.removeEventFilter(self)
        for actor in self.actors.values():
            if self.scene._plane_overlay_renderer is not None:
                self.scene._plane_overlay_renderer.RemoveActor(actor)
            self.scene.plotter.remove_actor(actor, render=False)
        self.actors.clear()
