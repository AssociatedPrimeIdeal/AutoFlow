"""Visible plane intersection and endpoint rotation handles in reference slices."""

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore


class PlaneSliceControls:
    def __init__(self, view):
        self.view = view
        self.enabled = False
        self.angle = 0.0
        self.center = np.zeros(2)
        self.endpoints = np.empty((0, 2))
        self._hovered = False
        self.line = pg.PlotCurveItem(pen=pg.mkPen('#ffb347', width=2))
        self.handles = pg.ScatterPlotItem(symbol='o', size=11,
            pen=pg.mkPen('#fff1d6', width=1.5), brush=pg.mkBrush('#ffb347'))
        self.angle_label = pg.TextItem(anchor=(0, 1), color='#ffe1ac',
            fill=pg.mkBrush(8, 12, 18, 225))
        for item, layer in ((self.line, 38), (self.handles, 48), (self.angle_label, 51)):
            item.setZValue(layer)
            item.setAcceptedMouseButtons(QtCore.Qt.NoButton)
            view.view_box.addItem(item, ignoreBounds=True)
            item.hide()
        view.view_box.rotation_hit_test = self.hit_test
        view.view_box.sigRangeChanged.connect(self.refresh)
        view.view_box.sigResized.connect(self.refresh)
        view.plot.scene().sigMouseMoved.connect(self.hover)
        view.view_box.planeRotationStarted.connect(self.start_angle)
        view.view_box.planeRotationChanged.connect(self.show_angle)
        view.view_box.planeRotationFinished.connect(self.finish_angle)

    def set_geometry(self, enabled, angle):
        self.enabled = bool(enabled)
        self.angle = float(angle)
        self.refresh()
        if not self.enabled:
            self.finish_angle()

    def set_center(self, horizontal, vertical):
        self.center = np.array([horizontal, vertical], dtype=float)
        self.refresh()

    def refresh(self, *_):
        self.endpoints = np.empty((0, 2))
        if not self.enabled:
            self.line.hide(); self.handles.hide()
            self._set_hover(False)
            return
        direction = np.array([np.cos(np.radians(self.angle)), np.sin(np.radians(self.angle))])
        bounds = self.view.view_box.viewRange()
        # The actual intersection is clipped to the viewport, with enough room
        # for both handle hit targets, even after zooming or resizing a panel.
        low, high = -np.inf, np.inf
        for axis, (first, last) in enumerate(bounds):
            margin = 0.09 * (last - first)
            first, last = first + margin, last - margin
            if abs(direction[axis]) < 1e-10:
                if not first <= self.center[axis] <= last:
                    self.line.hide(); self.handles.hide()
                    return
            else:
                a, b = sorted(((first-self.center[axis])/direction[axis],
                               (last-self.center[axis])/direction[axis]))
                low, high = max(low, a), min(high, b)
        if not np.isfinite(low+high) or low >= high:
            self.line.hide(); self.handles.hide()
            return
        self.endpoints = self.center + np.array([low, high])[:, None] * direction
        self.line.setData(self.endpoints[:, 0], self.endpoints[:, 1])
        self.handles.setData(self.endpoints[:, 0], self.endpoints[:, 1])
        self.line.show(); self.handles.show()

    def hit_test(self, position):
        box = self.view.view_box
        if not self.enabled or box.contour_editable or box.editable:
            return False
        for endpoint in self.endpoints:
            point = box.mapViewToScene(QtCore.QPointF(*endpoint))
            if np.hypot(position.x()-point.x(), position.y()-point.y()) <= 12:
                return True
        return False

    def _set_hover(self, active):
        if self._hovered == bool(active):
            return
        self._hovered = bool(active)
        self.handles.setSize(14 if active else 11)
        self.line.setPen(pg.mkPen('#ffe1ac' if active else '#ffb347', width=2.5 if active else 2))
        if active:
            self.view.plot.viewport().setCursor(QtCore.Qt.PointingHandCursor)
        else:
            self.view.plot.viewport().unsetCursor()

    def hover(self, position):
        self._set_hover(self.view.view_box._rotating_plane or self.hit_test(position))

    def start_angle(self, *_):
        self._set_hover(True)
        self.show_angle(None, 0.0)

    def show_angle(self, _name, angle):
        if not self.enabled:
            return
        self.angle_label.setText(f'{np.degrees(angle):+.1f}°')
        box = self.view.view_box
        center = box.mapViewToScene(QtCore.QPointF(*self.center))
        self.angle_label.setPos(box.mapSceneToView(center + QtCore.QPointF(14, -14)))
        self.angle_label.show()

    def finish_angle(self, *_):
        self.angle_label.hide()
        self._set_hover(False)
