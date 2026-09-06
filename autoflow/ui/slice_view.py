from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
from matplotlib import colormaps


pg.setConfigOption("imageAxisOrder", "row-major")


@dataclass(frozen=True)
class PlaneSpec:
    name: str
    title: str
    horizontal_axis: int
    vertical_axis: int
    fixed_axis: int
    horizontal_name: str
    vertical_name: str
    fixed_name: str
    horizontal_ends: tuple[str, str]
    vertical_ends: tuple[str, str]


PLANE_SPECS = {
    "axial": PlaneSpec(
        "axial", "Axial", 0, 1, 2, "LR", "AP", "FH", ("L", "R"), ("A", "P")
    ),
    "coronal": PlaneSpec(
        "coronal", "Coronal", 0, 2, 1, "LR", "FH", "AP", ("L", "R"), ("F", "H")
    ),
    "sagittal": PlaneSpec(
        "sagittal", "Sagittal", 1, 2, 0, "AP", "FH", "LR", ("A", "P"), ("F", "H")
    ),
}


@lru_cache(maxsize=16)
def make_colormap(name: str) -> pg.ColorMap:
    try:
        cmap = colormaps.get_cmap(str(name or "gray"))
    except ValueError:
        cmap = colormaps.get_cmap("gray")
    positions = np.linspace(0.0, 1.0, 256)
    colors = np.asarray(cmap(positions), dtype=float)
    return pg.ColorMap(positions, np.rint(colors * 255.0).astype(np.uint8))


def make_label_overlay(
    labels: np.ndarray | None,
    colors: dict[int, str],
    opacity: float,
) -> np.ndarray | None:
    if labels is None:
        return None
    data = np.asarray(labels)
    rgba = np.zeros((*data.shape, 4), dtype=np.uint8)
    alpha = int(round(255.0 * float(np.clip(opacity, 0.0, 1.0))))
    positive = data > 0
    if not np.any(positive):
        return rgba
    max_label = int(np.max(data[positive]))
    if max_label <= 65535:
        lut = np.zeros((max_label + 1, 4), dtype=np.uint8)
        lut[1:, :] = np.array([255, 107, 107, alpha], dtype=np.uint8)
        for raw_label, raw_color in colors.items():
            label = int(raw_label)
            if not (0 < label <= max_label):
                continue
            color = QtGui.QColor(str(raw_color))
            if not color.isValid():
                color = QtGui.QColor("#ff6b6b")
            lut[label] = (color.red(), color.green(), color.blue(), alpha)
        rgba[positive] = lut[np.asarray(data[positive], dtype=np.int64)]
        return rgba
    for value in np.unique(data[positive]):
        label = int(value)
        color = QtGui.QColor(colors.get(label, "#ff6b6b"))
        if not color.isValid():
            color = QtGui.QColor("#ff6b6b")
        rgba[data == label] = (color.red(), color.green(), color.blue(), alpha)
    return rgba


class SliceViewBox(pg.ViewBox):
    cursorRequested = QtCore.Signal(str, int, int)
    strokeStarted = QtCore.Signal(str, int, int, bool)
    strokeMoved = QtCore.Signal(str, int, int, bool)
    strokeFinished = QtCore.Signal(str)
    sliceStepRequested = QtCore.Signal(str, int)
    brushSizeStepRequested = QtCore.Signal(int)
    windowLevelDragged = QtCore.Signal(float, float)
    viewDoubleClicked = QtCore.Signal(str)

    def __init__(self, plane: str):
        super().__init__(enableMenu=False)
        self.plane = str(plane)
        self.editable = False
        self.spacing = (1.0, 1.0)
        self._image_shape = (0, 0)
        self.setAspectLocked(True)
        self.setMouseEnabled(x=False, y=False)

    def set_geometry(self, shape, spacing):
        self._image_shape = tuple(int(value) for value in shape[:2])
        self.spacing = tuple(
            float(value) if np.isfinite(value) and float(value) > 0.0 else 1.0
            for value in spacing[:2]
        )

    def _voxel_at(self, scene_position):
        if self._image_shape[0] <= 0 or self._image_shape[1] <= 0:
            return None
        point = self.mapSceneToView(scene_position)
        h = int(np.clip(np.rint(point.x() / self.spacing[0]), 0, self._image_shape[0] - 1))
        v = int(np.clip(np.rint(point.y() / self.spacing[1]), 0, self._image_shape[1] - 1))
        return h, v

    def mouseClickEvent(self, event):
        point = self._voxel_at(event.scenePos())
        if point is None:
            event.ignore()
            return
        h, v = point
        modifiers = event.modifiers()
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            if self.editable and not (modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier):
                self.strokeStarted.emit(self.plane, h, v, False)
                self.strokeFinished.emit(self.plane)
            else:
                self.cursorRequested.emit(self.plane, h, v)
            event.accept()
            return
        if event.button() == QtCore.Qt.MouseButton.RightButton and self.editable:
            self.strokeStarted.emit(self.plane, h, v, True)
            self.strokeFinished.emit(self.plane)
            event.accept()
            return
        event.ignore()

    def mouseDragEvent(self, event, axis=None):
        point = self._voxel_at(event.scenePos())
        if point is None:
            event.ignore()
            return
        h, v = point
        button = event.button()
        modifiers = event.modifiers()

        if button == QtCore.Qt.MouseButton.MiddleButton:
            delta = event.pos() - event.lastPos()
            self.windowLevelDragged.emit(float(delta.x()), float(delta.y()))
            event.accept()
            return

        if button == QtCore.Qt.MouseButton.LeftButton and (
            modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier
        ):
            if not event.isStart():
                current = self.mapSceneToView(event.scenePos())
                previous = self.mapSceneToView(event.lastScenePos())
                delta = current - previous
                self.translateBy(x=-float(delta.x()), y=-float(delta.y()))
            event.accept()
            return

        temporary_erase = button == QtCore.Qt.MouseButton.RightButton
        if self.editable and button in (
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.RightButton,
        ):
            if event.isStart():
                self.strokeStarted.emit(self.plane, h, v, temporary_erase)
            elif not event.isFinish():
                self.strokeMoved.emit(self.plane, h, v, temporary_erase)
            if event.isFinish():
                self.strokeFinished.emit(self.plane)
            event.accept()
            return

        if button == QtCore.Qt.MouseButton.LeftButton:
            self.cursorRequested.emit(self.plane, h, v)
            event.accept()
            return
        event.ignore()

    def wheelEvent(self, event, axis=None):
        delta = 1 if event.delta() > 0 else -1
        modifiers = event.modifiers()
        if modifiers & QtCore.Qt.KeyboardModifier.ControlModifier:
            factor = 0.85 if delta > 0 else 1.0 / 0.85
            point = self.mapSceneToView(event.scenePos())
            self.scaleBy((factor, factor), center=point)
            event.accept()
            return
        if self.editable and modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier:
            self.brushSizeStepRequested.emit(delta)
            event.accept()
            return
        self.sliceStepRequested.emit(self.plane, delta)
        event.accept()

    def mouseDoubleClickEvent(self, event):
        self.viewDoubleClicked.emit(self.plane)
        event.accept()


class SliceView(QtWidgets.QFrame):
    cursorRequested = QtCore.Signal(str, int, int)
    hoverMoved = QtCore.Signal(str, int, int)
    strokeStarted = QtCore.Signal(str, int, int, bool)
    strokeMoved = QtCore.Signal(str, int, int, bool)
    strokeFinished = QtCore.Signal(str)
    sliceStepRequested = QtCore.Signal(str, int)
    brushSizeStepRequested = QtCore.Signal(int)
    windowLevelDragged = QtCore.Signal(float, float)
    viewDoubleClicked = QtCore.Signal(str)

    def __init__(self, plane: str, parent=None):
        super().__init__(parent)
        self.spec = PLANE_SPECS[str(plane)]
        self._spacing = (1.0, 1.0)
        self._shape = (0, 0)
        self._hover_voxel = None
        self._brush_diameter_mm = 6.0
        self._brush_color = QtGui.QColor("#39d5c5")
        self._brush_visible = False
        self._build_ui()

    def _build_ui(self):
        self.setObjectName("sliceView")
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QtWidgets.QWidget()
        header.setObjectName("sliceHeader")
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(7, 3, 7, 3)
        self.title_label = QtWidgets.QLabel()
        self.title_label.setObjectName("sliceTitle")
        self.position_label = QtWidgets.QLabel()
        self.position_label.setObjectName("slicePosition")
        header_layout.addWidget(self.title_label)
        header_layout.addStretch(1)
        header_layout.addWidget(self.position_label)
        layout.addWidget(header)

        self.view_box = SliceViewBox(self.spec.name)
        self.plot = pg.PlotWidget(viewBox=self.view_box, background="#050809")
        self.plot.setMenuEnabled(False)
        self.plot.hideAxis("left")
        self.plot.hideAxis("bottom")
        self.plot.setMouseTracking(True)
        layout.addWidget(self.plot, 1)

        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.overlay_item = pg.ImageItem(axisOrder="row-major")
        self.image_item.setZValue(0)
        self.overlay_item.setZValue(10)
        self.view_box.addItem(self.image_item)
        self.view_box.addItem(self.overlay_item)

        self.vertical_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#ff5f5f", width=1.0))
        self.horizontal_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen("#58d178", width=1.0))
        self.vertical_line.setZValue(30)
        self.horizontal_line.setZValue(30)
        self.view_box.addItem(self.vertical_line)
        self.view_box.addItem(self.horizontal_line)

        self.brush_item = QtWidgets.QGraphicsEllipseItem()
        self.brush_item.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        self.brush_item.setZValue(50)
        self.brush_item.hide()
        self.view_box.addItem(self.brush_item)

        self.orientation_items = {
            "h0": pg.TextItem(anchor=(0.0, 0.5), color="#d7e0e3"),
            "h1": pg.TextItem(anchor=(1.0, 0.5), color="#d7e0e3"),
            "v0": pg.TextItem(anchor=(0.5, 0.0), color="#d7e0e3"),
            "v1": pg.TextItem(anchor=(0.5, 1.0), color="#d7e0e3"),
        }
        for item in self.orientation_items.values():
            item.setZValue(40)
            self.view_box.addItem(item)
        self.orientation_items["h0"].setText(self.spec.horizontal_ends[0])
        self.orientation_items["h1"].setText(self.spec.horizontal_ends[1])
        self.orientation_items["v0"].setText(self.spec.vertical_ends[0])
        self.orientation_items["v1"].setText(self.spec.vertical_ends[1])

        self.view_box.cursorRequested.connect(self.cursorRequested)
        self.view_box.strokeStarted.connect(self.strokeStarted)
        self.view_box.strokeMoved.connect(self.strokeMoved)
        self.view_box.strokeFinished.connect(self.strokeFinished)
        self.view_box.sliceStepRequested.connect(self.sliceStepRequested)
        self.view_box.brushSizeStepRequested.connect(self.brushSizeStepRequested)
        self.view_box.windowLevelDragged.connect(self.windowLevelDragged)
        self.view_box.viewDoubleClicked.connect(self.viewDoubleClicked)
        self.plot.scene().sigMouseMoved.connect(self._scene_mouse_moved)

    def set_editable(self, enabled: bool):
        self.view_box.editable = bool(enabled)
        if not enabled:
            self.brush_item.hide()

    def set_brush(self, diameter_mm: float, color: str, visible: bool = True):
        self._brush_diameter_mm = max(float(diameter_mm), 0.1)
        parsed = QtGui.QColor(str(color))
        self._brush_color = parsed if parsed.isValid() else QtGui.QColor("#39d5c5")
        self._brush_visible = bool(visible)
        self._update_brush_item()

    def reset_view(self):
        if self._shape[0] <= 0 or self._shape[1] <= 0:
            return
        x_half = 0.5 * self._spacing[0]
        y_half = 0.5 * self._spacing[1]
        self.view_box.setRange(
            xRange=(-x_half, (self._shape[0] - 0.5) * self._spacing[0]),
            yRange=(-y_half, (self._shape[1] - 0.5) * self._spacing[1]),
            padding=0.02,
        )

    def _scene_mouse_moved(self, scene_position):
        if not self.plot.sceneBoundingRect().contains(scene_position):
            self._brush_visible = False
            self.brush_item.hide()
            return
        voxel = self.view_box._voxel_at(scene_position)
        if voxel is None:
            return
        self._hover_voxel = voxel
        self.hoverMoved.emit(self.spec.name, voxel[0], voxel[1])
        if self.view_box.editable:
            self._brush_visible = True
            self._update_brush_item()

    def _update_brush_item(self):
        if not self._brush_visible or self._hover_voxel is None or not self.view_box.editable:
            self.brush_item.hide()
            return
        h, v = self._hover_voxel
        center_x = h * self._spacing[0]
        center_y = v * self._spacing[1]
        radius = self._brush_diameter_mm / 2.0
        self.brush_item.setRect(center_x - radius, center_y - radius, 2.0 * radius, 2.0 * radius)
        color = self._brush_color
        self.brush_item.setPen(pg.mkPen(color, width=1.5))
        fill = QtGui.QColor(color)
        fill.setAlpha(35)
        self.brush_item.setBrush(QtGui.QBrush(fill))
        self.brush_item.show()

    def set_slice(
        self,
        image: np.ndarray,
        overlay: np.ndarray | None,
        spacing,
        cursor,
        fixed_index: int,
        colormap: pg.ColorMap,
        levels,
    ):
        data = np.asarray(image)
        self._shape = tuple(int(value) for value in data.shape[:2])
        self._spacing = tuple(
            float(value) if np.isfinite(value) and float(value) > 0.0 else 1.0
            for value in spacing[:2]
        )
        geometry_changed = (self.view_box._image_shape, self.view_box.spacing) != (
            self._shape,
            self._spacing,
        )
        self.view_box.set_geometry(self._shape, self._spacing)
        rect = QtCore.QRectF(
            -0.5 * self._spacing[0],
            -0.5 * self._spacing[1],
            self._shape[0] * self._spacing[0],
            self._shape[1] * self._spacing[1],
        )
        self.image_item.setImage(data.T, autoLevels=False, levels=levels)
        self.image_item.setLookupTable(colormap.getLookupTable(nPts=256, alpha=True))
        self.image_item.setRect(rect)
        self.set_overlay(overlay, rect=rect)

        self.update_cursor(cursor, fixed_index, levels)
        x0 = 0.0
        x1 = max(0.0, (self._shape[0] - 1) * self._spacing[0])
        y0 = 0.0
        y1 = max(0.0, (self._shape[1] - 1) * self._spacing[1])
        self.orientation_items["h0"].setPos(x0, (y0 + y1) / 2.0)
        self.orientation_items["h1"].setPos(x1, (y0 + y1) / 2.0)
        self.orientation_items["v0"].setPos((x0 + x1) / 2.0, y0)
        self.orientation_items["v1"].setPos((x0 + x1) / 2.0, y1)
        self._update_brush_item()
        if geometry_changed:
            self.reset_view()

    def set_overlay(self, overlay: np.ndarray | None, rect=None):
        if overlay is None:
            self.overlay_item.clear()
            self.overlay_item.hide()
            return
        if rect is None:
            rect = QtCore.QRectF(
                -0.5 * self._spacing[0],
                -0.5 * self._spacing[1],
                self._shape[0] * self._spacing[0],
                self._shape[1] * self._spacing[1],
            )
        rgba = np.asarray(overlay, dtype=np.uint8)
        self.overlay_item.setImage(np.transpose(rgba, (1, 0, 2)), autoLevels=False)
        self.overlay_item.setOpacity(float(np.max(rgba[..., 3])) / 255.0)
        self.overlay_item.setRect(rect)
        self.overlay_item.show()

    def update_image(self, image: np.ndarray, overlay: np.ndarray | None, cursor, fixed_index: int, levels):
        """Replace pixels during playback without rebuilding static presentation."""
        self.image_item.setImage(np.asarray(image).T, autoLevels=False, levels=levels)
        if overlay is None:
            self.overlay_item.clear()
            self.overlay_item.hide()
        else:
            rgba = np.asarray(overlay, dtype=np.uint8)
            self.overlay_item.setImage(np.transpose(rgba, (1, 0, 2)), autoLevels=False)
            self.overlay_item.setOpacity(float(np.max(rgba[..., 3])) / 255.0)
            self.overlay_item.show()
        self.update_cursor(cursor, fixed_index, levels)

    def update_cursor(self, cursor, fixed_index: int, levels=None):
        h, v = int(cursor[0]), int(cursor[1])
        self.vertical_line.setPos(h * self._spacing[0])
        self.horizontal_line.setPos(v * self._spacing[1])
        self.title_label.setText(
            f"{self.spec.title}  {self.spec.horizontal_name} x {self.spec.vertical_name}"
        )
        self.position_label.setText(f"{self.spec.fixed_name} {int(fixed_index) + 1}")
        if levels is not None:
            self.image_item.setLevels(levels)

    def clear(self):
        self.image_item.clear()
        self.overlay_item.clear()
        self.overlay_item.hide()
        self.position_label.setText("-")
        self.brush_item.hide()
