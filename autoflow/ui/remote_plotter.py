"""Interactive PyVista rendering for SSH-forwarded Qt sessions.

Qt's regular widgets work over forwarded X11 on systems where QOpenGLWidget
cannot obtain a usable GLX context.  This widget keeps VTK on an off-screen
software render window and transfers completed RGB frames into a normal QWidget.
"""

from __future__ import annotations

import contextlib
import math
import os
import time

import numpy as np

if os.environ.get("AUTOFLOW_SSH_RENDERING") == "1":
    os.environ["VTK_DEFAULT_OPENGL_WINDOW"] = "vtkOSOpenGLRenderWindow"
    os.environ["QT_X11_NO_MITSHM"] = "1"

import pyvista as pv
from PySide6 import QtCore, QtGui, QtWidgets


_KEY_SYMS = {
    QtCore.Qt.Key_Backspace: "BackSpace",
    QtCore.Qt.Key_Delete: "Delete",
    QtCore.Qt.Key_Down: "Down",
    QtCore.Qt.Key_End: "End",
    QtCore.Qt.Key_Enter: "Return",
    QtCore.Qt.Key_Escape: "Escape",
    QtCore.Qt.Key_Home: "Home",
    QtCore.Qt.Key_Insert: "Insert",
    QtCore.Qt.Key_Left: "Left",
    QtCore.Qt.Key_PageDown: "Next",
    QtCore.Qt.Key_PageUp: "Prior",
    QtCore.Qt.Key_Return: "Return",
    QtCore.Qt.Key_Right: "Right",
    QtCore.Qt.Key_Space: "space",
    QtCore.Qt.Key_Tab: "Tab",
    QtCore.Qt.Key_Up: "Up",
}


def _prepare_rgb_frame(image):
    """Return an RGB888-compatible array with tightly packed rows."""
    return np.ascontiguousarray(image[:, :, :3], dtype=np.uint8)


class RemotePlotter(QtWidgets.QWidget):
    """A QWidget facade around an off-screen :class:`pyvista.Plotter`."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent, True)
        self.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.setMouseTracking(True)
        self.setMinimumSize(160, 120)

        self._plotter = pv.Plotter(off_screen=True, window_size=(640, 480))
        self._frame = QtGui.QImage()
        self._capturing = False
        self._closed = False
        self._wheel_remainder = 0
        self._last_capture_at = 0.0
        self._minimum_frame_interval_ms = 30
        self._autoflow_window_level_active = False

        self._frame_timer = QtCore.QTimer(self)
        self._frame_timer.setSingleShot(True)
        self._frame_timer.timeout.connect(self._capture_frame)
        self._resize_timer = QtCore.QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(50)
        self._resize_timer.timeout.connect(self._resize_render_window)

        self._end_render_observer = self._plotter.render_window.AddObserver(
            "EndEvent", self._on_render_complete
        )
        application = QtWidgets.QApplication.instance()
        if application is not None:
            application.aboutToQuit.connect(self.shutdown)

    def __getattr__(self, name):
        plotter = self.__dict__.get("_plotter")
        if plotter is None:
            raise AttributeError(name)
        return getattr(plotter, name)

    @property
    def iren(self):
        """Return the underlying VTK interactor used by mouse forwarding."""
        wrapper = self._plotter.iren
        interactor = getattr(wrapper, "interactor", None)
        if interactor is not None:
            return interactor
        interactor = self._plotter.render_window.GetInteractor()
        if interactor is None:
            wrapper.initialize()
            interactor = wrapper.interactor
        return interactor

    @property
    def interactor(self):
        """Expose the VTK interactor for code shared with QtInteractor."""
        return self._plotter.iren

    def _on_render_complete(self, *_args):
        if not self._capturing and not self._closed:
            self.request_frame()

    def request_frame(self, delay_ms=0):
        """Coalesce render notifications into one Qt image update."""
        if self._closed:
            return
        elapsed_ms = (time.monotonic() - self._last_capture_at) * 1000.0
        throttle_ms = max(0, math.ceil(self._minimum_frame_interval_ms - elapsed_ms))
        delay_ms = max(0, int(delay_ms), throttle_ms)
        if self._frame_timer.isActive():
            remaining_ms = self._frame_timer.remainingTime()
            if 0 <= remaining_ms <= delay_ms:
                return
        self._frame_timer.start(delay_ms)

    def render(self):
        """Render the VTK scene and schedule its image for display."""
        if self._closed:
            return None
        result = self._plotter.render()
        self.request_frame()
        return result

    def _capture_frame(self):
        if self._closed or self._capturing or not self.isVisible():
            return
        self._capturing = True
        try:
            image = self._plotter.screenshot(return_img=True)
            if image is None or image.ndim != 3 or image.shape[2] < 3:
                return
            image = _prepare_rgb_frame(image)
            height, width = image.shape[:2]
            qimage = QtGui.QImage(
                image.data,
                width,
                height,
                int(image.strides[0]),
                QtGui.QImage.Format_RGB888,
            )
            self._frame = qimage.copy()
            self._last_capture_at = time.monotonic()
            self.update()
        finally:
            self._capturing = False

    def paintEvent(self, _event):
        painter = QtGui.QPainter(self)
        if self._frame.isNull():
            painter.fillRect(self.rect(), QtGui.QColor("#202124"))
            return
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, False)
        painter.drawImage(self.rect(), self._frame)

    def showEvent(self, event):
        super().showEvent(event)
        self._resize_timer.start(0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._resize_timer.start()

    def _render_size(self):
        scale = max(1.0, float(self.devicePixelRatioF()))
        return max(1, round(self.width() * scale)), max(1, round(self.height() * scale))

    def _resize_render_window(self):
        if self._closed or self.width() <= 0 or self.height() <= 0:
            return
        width, height = self._render_size()
        if tuple(self._plotter.window_size) != (width, height):
            self._plotter.window_size = (width, height)
        self.render()

    def _vtk_position(self, event):
        position = event.position()
        scale = max(1.0, float(self.devicePixelRatioF()))
        width, height = self._render_size()
        x = min(width - 1, max(0, round(position.x() * scale)))
        y = min(height - 1, max(0, height - 1 - round(position.y() * scale)))
        return x, y

    @staticmethod
    def _modifier_state(event):
        modifiers = event.modifiers()
        control = int(bool(modifiers & QtCore.Qt.ControlModifier))
        shift = int(bool(modifiers & QtCore.Qt.ShiftModifier))
        return control, shift

    def _set_mouse_event(self, event):
        x, y = self._vtk_position(event)
        control, shift = self._modifier_state(event)
        interactor = self.iren
        interactor.SetEventInformation(x, y, control, shift, "\0", 0, None)
        # ``SetEventInformation`` has no Alt argument.  Keep it in the VTK
        # interactor explicitly so SceneController can distinguish an
        # Alt+left rotation from a plain left-button pan.
        try:
            interactor.SetAltKey(int(bool(event.modifiers() & QtCore.Qt.AltModifier)))
        except Exception:
            pass

    def mousePressEvent(self, event):
        self.setFocus(QtCore.Qt.MouseFocusReason)
        self._set_mouse_event(event)
        interactor = self.iren
        style = interactor.GetInteractorStyle()
        shift_left = (
            event.button() == QtCore.Qt.LeftButton
            and bool(event.modifiers() & QtCore.Qt.ShiftModifier)
        )
        if event.button() == QtCore.Qt.LeftButton:
            if shift_left:
                dispatch = getattr(
                    self, "_autoflow_window_level_dispatch", None
                )
                if callable(dispatch):
                    dispatch("press")
                if not self._autoflow_window_level_active:
                    style.OnLeftButtonDown()
            elif style.GetEnabled():
                style.OnLeftButtonDown()
        elif event.button() == QtCore.Qt.MiddleButton:
            style.OnMiddleButtonDown()
        elif event.button() == QtCore.Qt.RightButton:
            style.OnRightButtonDown()
            try:
                interactor.InvokeEvent("RightButtonPressEvent")
            except Exception:
                pass
        else:
            event.ignore()
            return
        event.accept()

    def mouseMoveEvent(self, event):
        self._set_mouse_event(event)
        interactor = self.iren
        style = interactor.GetInteractorStyle()
        if not self._autoflow_window_level_active:
            style.OnMouseMove()
        # A UserEvent gives the window/level control a reliable drag
        # notification without disturbing VTK's own state.
        if self._autoflow_window_level_active:
            dispatch = getattr(self, "_autoflow_window_level_dispatch", None)
            if callable(dispatch):
                dispatch("move")
        event.accept()

    def mouseReleaseEvent(self, event):
        self._set_mouse_event(event)
        interactor = self.iren
        style = interactor.GetInteractorStyle()
        event_name = None
        if event.button() == QtCore.Qt.LeftButton:
            if self._autoflow_window_level_active:
                dispatch = getattr(
                    self, "_autoflow_window_level_dispatch", None
                )
                if callable(dispatch):
                    dispatch("release")
            else:
                style.OnLeftButtonUp()
                event_name = "LeftButtonReleaseEvent"
        elif event.button() == QtCore.Qt.MiddleButton:
            event_name = "MiddleButtonReleaseEvent"
            style.OnMiddleButtonUp()
        elif event.button() == QtCore.Qt.RightButton:
            style.OnRightButtonUp()
            event_name = "RightButtonReleaseEvent"
        else:
            event.ignore()
            return
        # vtkGenericRenderWindowInteractor's convenience release methods update
        # button state but do not always invoke observers on an off-screen
        # interactor.  Explicitly dispatch the event so camera and window/level
        # gestures can reliably restore their state.
        if event_name is not None:
            try:
                interactor.InvokeEvent(event_name)
            except Exception:
                pass
        event.accept()

    def wheelEvent(self, event):
        self._set_mouse_event(event)
        self._wheel_remainder += int(event.angleDelta().y())
        interactor = self.iren
        while self._wheel_remainder >= 120:
            interactor.MouseWheelForwardEvent()
            self._wheel_remainder -= 120
        while self._wheel_remainder <= -120:
            interactor.MouseWheelBackwardEvent()
            self._wheel_remainder += 120
        event.accept()

    @staticmethod
    def _key_symbol(event):
        if event.key() in _KEY_SYMS:
            return _KEY_SYMS[event.key()]
        text = event.text()
        if text:
            return text
        return ""

    def _set_key_event(self, event):
        text = event.text()
        key_code = text[0] if text else "\0"
        key_sym = self._key_symbol(event)
        control, shift = self._modifier_state(event)
        self.iren.SetKeyEventInformation(
            control, shift, key_code, int(event.isAutoRepeat()), key_sym
        )

    def keyPressEvent(self, event):
        self._set_key_event(event)
        self.iren.KeyPressEvent()
        self.iren.CharEvent()
        event.accept()

    def keyReleaseEvent(self, event):
        self._set_key_event(event)
        self.iren.KeyReleaseEvent()
        event.accept()

    def shutdown(self):
        if self._closed:
            return
        self._closed = True
        self._frame_timer.stop()
        self._resize_timer.stop()
        with contextlib.suppress(Exception):
            self._plotter.render_window.RemoveObserver(self._end_render_observer)
        with contextlib.suppress(Exception):
            self._plotter.close()

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)
