"""Modal task progress with live activity and cooperative cancellation."""

import math
import time

from PySide6 import QtCore, QtGui, QtWidgets

from ..task_control import CancellationToken


class _ActivityDots(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.phase = 0.0
        self.setFixedSize(56, 40)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        for index in range(5):
            wave = (math.sin(self.phase - index * 0.8) + 1.0) * 0.5
            color = QtGui.QColor("#087f68")
            color.setAlphaF(0.25 + 0.75 * wave)
            painter.setBrush(color)
            painter.drawEllipse(QtCore.QPointF(8 + index * 10, 22 - wave * 8), 3.3, 3.3)


class TaskProgressDialog(QtWidgets.QDialog):
    """The close button requests cancellation; the task owns final dismissal."""

    cancelRequested = QtCore.Signal()

    def __init__(self, title, message, parent=None):
        super().__init__(parent)
        self.cancel_token = CancellationToken()
        self._finished = False
        self._started = self._last_update = time.monotonic()
        self._message = str(message)
        self._last_payload_key = None
        self.setWindowTitle(str(title))
        self.setWindowFlags(QtCore.Qt.WindowType.Dialog | QtCore.Qt.WindowType.CustomizeWindowHint
                            | QtCore.Qt.WindowType.WindowTitleHint | QtCore.Qt.WindowType.WindowCloseButtonHint)
        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        self.setMinimumWidth(480)
        self.setMaximumWidth(680)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24, 18, 24, 20)
        layout.setSpacing(12)
        top = QtWidgets.QHBoxLayout()
        self.activity = _ActivityDots(self)
        top.addWidget(self.activity)
        heading = QtWidgets.QLabel(str(title))
        heading.setStyleSheet("font-size: 17px; font-weight: 600; color: #075e49;")
        top.addWidget(heading, 1)
        layout.addLayout(top)
        self.message_label = QtWidgets.QLabel(self._message)
        self.message_label.setWordWrap(True)
        self.message_label.setMinimumHeight(44)
        layout.addWidget(self.message_label)
        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 0)
        self.bar.setFixedHeight(24)
        self.bar.setStyleSheet("QProgressBar { border: 1px solid #cfdfdb; border-radius: 7px; "
                               "background: #e9f1ef; padding: 0px; text-align: center; } "
                               "QProgressBar::chunk { background: #169e82; border-radius: 6px; }")
        layout.addWidget(self.bar)
        self.detail_label = QtWidgets.QLabel()
        self.detail_bar = QtWidgets.QProgressBar()
        self.detail_bar.setMaximumHeight(12)
        self.detail_bar.setTextVisible(False)
        self.detail_label.hide()
        self.detail_bar.hide()
        layout.addWidget(self.detail_label)
        layout.addWidget(self.detail_bar)
        self.timing_label = QtWidgets.QLabel()
        self.timing_label.setStyleSheet("color: #576b65;")
        layout.addWidget(self.timing_label)
        self.hint_label = QtWidgets.QLabel("Close (×) to cancel. Other actions are locked while this task runs.")
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet("color: #687871; font-size: 12px;")
        layout.addWidget(self.hint_label)
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        QtWidgets.QApplication.instance().installEventFilter(self)
        self._tick()

    def eventFilter(self, watched, event):
        # Application-wide shortcuts and direct events must respect the same
        # lock as native window modality (including floating docks).
        input_events = {
            QtCore.QEvent.Type.MouseButtonPress, QtCore.QEvent.Type.MouseButtonRelease,
            QtCore.QEvent.Type.MouseButtonDblClick, QtCore.QEvent.Type.Wheel,
            QtCore.QEvent.Type.KeyPress, QtCore.QEvent.Type.KeyRelease,
            QtCore.QEvent.Type.Shortcut, QtCore.QEvent.Type.ShortcutOverride,
            QtCore.QEvent.Type.TouchBegin, QtCore.QEvent.Type.TouchUpdate,
            QtCore.QEvent.Type.TouchEnd, QtCore.QEvent.Type.NativeGesture,
            QtCore.QEvent.Type.TabletPress, QtCore.QEvent.Type.TabletMove,
            QtCore.QEvent.Type.TabletRelease, QtCore.QEvent.Type.InputMethod,
            QtCore.QEvent.Type.Close, QtCore.QEvent.Type.Drop,
            QtCore.QEvent.Type.DragEnter, QtCore.QEvent.Type.ContextMenu,
        }
        if not self._finished and event.type() in input_events:
            if isinstance(watched, QtWidgets.QWidget) and watched.window() is not self:
                return True
            if isinstance(watched, QtGui.QShortcut):
                return True
            if event.type() in {QtCore.QEvent.Type.Shortcut, QtCore.QEvent.Type.ShortcutOverride}:
                if not isinstance(watched, QtWidgets.QWidget) or watched.window() is not self:
                    return True
            if isinstance(watched, QtGui.QWindow) and watched is not self.windowHandle():
                return True
        return super().eventFilter(watched, event)

    def _tick(self):
        self.activity.phase += 0.45
        self.activity.update()
        now = time.monotonic()
        elapsed = int(now - self._started)
        age = int(now - self._last_update)
        status = "Cancelling" if self.cancel_token.cancelled else "Task active"
        self.timing_label.setText(f"{status}  ·  Elapsed {elapsed // 60:02d}:{elapsed % 60:02d}"
                                  f"  ·  Last progress update {age}s ago")

    def setRange(self, minimum, maximum):
        self.bar.setRange(int(minimum), int(maximum))

    def setValue(self, value):
        if int(value) != self.bar.value():
            self._last_update = time.monotonic()
        self.bar.setValue(int(value))

    def setLabelText(self, message):
        if str(message) != self._message:
            self._last_update = time.monotonic()
        self._message = str(message)
        if not self.cancel_token.cancelled:
            self.message_label.setText(self._message)

    def update_progress(self, payload):
        payload = dict(payload or {})
        if self.cancel_token.cancelled:
            return
        current, total = payload.get("current"), payload.get("total")
        if total is not None and int(total) > 0:
            self.setRange(0, int(total))
            self.setValue(min(max(int(current or 0), 0), int(total)))
            self.bar.setFormat(f"{int(current or 0)} / {int(total)}  (%p%)")
        else:
            self.setRange(0, 0)
        self.setLabelText(payload.get("message", self._message))
        detail_total = int(payload.get("detail_total") or 0)
        detail_current = int(payload.get("detail_current") or 0)
        if detail_total > 0:
            self.detail_label.setText(str(payload.get("detail_message") or f"{detail_current} / {detail_total}"))
            self.detail_bar.setRange(0, detail_total)
            self.detail_bar.setValue(min(detail_current, detail_total))
        single_task_detail = detail_total > 0 and int(total or 0) == 1
        if single_task_detail:
            # A single pipeline step/video should show its actual plane/frame
            # count prominently rather than sit at "0 / 1" throughout the run.
            self.setRange(0, detail_total)
            self.setValue(min(detail_current, detail_total))
            self.bar.setFormat(f"{detail_current} / {detail_total}  (%p%)")
        self.detail_label.setVisible(detail_total > 0 and not single_task_detail)
        self.detail_bar.setVisible(detail_total > 0 and not single_task_detail)
        key = (payload.get("stage"), current, total, detail_current, detail_total, payload.get("message"))
        if key != self._last_payload_key:
            self._last_update = time.monotonic()
            self._last_payload_key = key

    def reject(self):
        # Esc does not silently dismiss the progress lock.
        return

    def closeEvent(self, event):
        if self._finished:
            self._timer.stop()
            QtWidgets.QApplication.instance().removeEventFilter(self)
            event.accept()
            return
        if not self.cancel_token.cancelled:
            self.cancel_token.cancel()
            self.message_label.setText("Cancelling… Waiting for the current operation to stop safely.")
            self.hint_label.setText("The workspace will unlock after the task has stopped.")
            self.cancelRequested.emit()
        event.ignore()

    def finish(self):
        self._finished = True
        self.close()
