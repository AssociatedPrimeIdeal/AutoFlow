"""Window-scoped repeating arrow navigation without stealing editor keys."""

from PySide6 import QtCore, QtWidgets


class FrameNavigationFilter(QtCore.QObject):
    def __init__(self, window, step, finish):
        super().__init__(window)
        self.window = window
        self.step = step
        self.finish = finish
        self.held = set()
        app = QtWidgets.QApplication.instance()
        window.installEventFilter(self)
        if app is not None:
            # Global Python event filters can recursively wrap Chromium's
            # internal QObjects in PySide 6.11 (getWrapperForQObject crash).
            app.focusChanged.connect(self._focus_changed)
            self._focus_changed(None, app.focusWidget())

    def _focus_changed(self, previous, current):
        if self.held:
            self.held.clear()
            self.finish()
        if current is not None and current.window() is self.window:
            current.installEventFilter(self)

    def _eligible(self, watched):
        if not isinstance(watched, QtWidgets.QWidget):
            return False
        if watched.window() is not self.window:
            return False
        focus = QtWidgets.QApplication.focusWidget() or watched
        if focus.window() is not self.window:
            focus = watched
        # Text, numeric and object-browser navigation retain their native keys.
        blocked = (QtWidgets.QLineEdit, QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit,
                   QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox, QtWidgets.QAbstractItemView)
        widget = focus
        while widget is not None and widget is not self.window:
            if widget.property('autoflowHtmlDocument'):
                return False
            if isinstance(widget, blocked):
                return False
            if isinstance(widget, QtWidgets.QSlider) and widget is not getattr(self.window, "slider_t", None):
                return False
            widget = widget.parentWidget()
        return True

    def eventFilter(self, watched, event):
        kind = event.type()
        if kind in (QtCore.QEvent.Type.WindowDeactivate, QtCore.QEvent.Type.FocusOut):
            if watched is self.window and getattr(self.window, "scene", None) is not None:
                self.window.scene.set_interaction_preview(False)
            if isinstance(watched, QtWidgets.QWidget) and watched.window() is self.window and self.held:
                self.held.clear()
                self.finish()
            return False
        if kind not in (QtCore.QEvent.Type.ShortcutOverride, QtCore.QEvent.Type.KeyPress,
                        QtCore.QEvent.Type.KeyRelease):
            return False
        key = event.key()
        if key not in (QtCore.Qt.Key.Key_Left, QtCore.Qt.Key.Key_Right):
            return False
        if event.modifiers() != QtCore.Qt.KeyboardModifier.NoModifier or not self._eligible(watched):
            return False
        if kind == QtCore.QEvent.Type.ShortcutOverride:
            event.accept()  # Avoid ambiguity with standalone OrthoViewer shortcuts.
            return True
        if kind == QtCore.QEvent.Type.KeyPress:
            self.held.add(key)
            self.step(-1 if key == QtCore.Qt.Key.Key_Left else 1)
        elif not event.isAutoRepeat():
            self.held.discard(key)
            if not self.held:
                self.finish()
        event.accept()
        return True
