"""Qt context menu for the continuous colorbar in native and remote 3D views."""

import math

from PySide6 import QtCore, QtWidgets

from ..core.models import ObjectKind


class ScalarBarControlsMixin:
    def _ensure_scalar_bar_interaction(self):
        widget = self.plotter
        if getattr(self, "_scalar_bar_filter", None) is not None or not hasattr(widget, "installEventFilter"):
            return
        owner = self

        class BarFilter(QtCore.QObject):
            def __init__(self):
                super().__init__(widget)
                self.pressed_uid = None

            def eventFilter(self, watched, event):
                kind = event.type()
                if kind == QtCore.QEvent.Type.MouseButtonPress and event.button() == QtCore.Qt.RightButton:
                    obj = owner._scalar_bar_object_at(event.position().x(), event.position().y())
                    self.pressed_uid = obj.uid if obj is not None else None
                    return self.pressed_uid is not None
                if kind == QtCore.QEvent.Type.MouseMove and self.pressed_uid is not None:
                    return True
                if kind == QtCore.QEvent.Type.MouseButtonRelease and event.button() == QtCore.Qt.RightButton:
                    uid, self.pressed_uid = self.pressed_uid, None
                    if uid is not None:
                        owner._show_scalar_bar_menu(uid, event.globalPosition().toPoint())
                        return True
                return False

        self._scalar_bar_filter = BarFilter()
        widget.installEventFilter(self._scalar_bar_filter)

    def _scalar_bar_object_at(self, x, y):
        obj = self.workspace.scene_objects.get(self._active_scalar_bar_uid)
        if not self._object_can_drive_scalar_bar(obj) or obj.kind == ObjectKind.SEGMENTATION:
            return None
        width, height = self.plotter.width(), self.plotter.height()
        if width <= 0 or height <= 0:
            return None
        nx, ny = x / width, 1.0 - y / height
        for bar in self._hud_props:
            if not bar.IsA("vtkScalarBarActor") or not bar.GetVisibility():
                continue
            left, bottom = bar.GetPosition()
            if (left - 18 / width <= nx <= left + bar.GetWidth() + 45 / width
                    and bottom - 18 / height <= ny <= bottom + bar.GetHeight() + 25 / height):
                return obj
        return None

    def _show_scalar_bar_menu(self, uid, position):
        if uid not in self.workspace.scene_objects:
            return
        menu = QtWidgets.QMenu(self.plotter)
        menu.addAction("Color limits…", lambda: self._edit_scalar_bar_clim(uid))
        menu.addAction("Reset to automatic", lambda: self._set_scalar_bar_clim(uid, None))
        self._scalar_bar_menu = menu
        menu.popup(position)

    def _set_scalar_bar_clim(self, uid, limits):
        obj = self.workspace.scene_objects.get(uid)
        if obj is None:
            return False
        if limits is not None:
            low, high = map(float, limits)
            if not math.isfinite(low) or not math.isfinite(high) or low >= high:
                raise ValueError("Enter finite values with minimum < maximum.")
            limits = (low, high)
        obj.clim = limits
        self.apply_object_properties(obj, render=False, refresh_scalar_bar=False)
        self._refresh_shared_scalar_bar(preferred_uid=uid, render=False)
        self.request_render()
        window = self.plotter.window()
        refresh = getattr(window, "_refresh_render_range_control", None)
        if callable(refresh):
            refresh()
        return True

    def _edit_scalar_bar_clim(self, uid):
        obj = self.workspace.scene_objects.get(uid)
        if obj is None or obj.actor is None:
            return
        limits = (self._volume_scalar_range(obj, self._build_dataset(obj.data_key)) if self._is_volume_object(obj)
                  else obj.clim if obj.clim is not None else obj.actor.GetMapper().GetScalarRange())
        dialog = QtWidgets.QDialog(self.plotter.window())
        dialog.setWindowTitle("Color limits")
        form = QtWidgets.QFormLayout(dialog)
        low = QtWidgets.QLineEdit(f"{float(limits[0]):.10g}")
        high = QtWidgets.QLineEdit(f"{float(limits[1]):.10g}")
        low.setObjectName("climMinimum")
        high.setObjectName("climMaximum")
        form.addRow("Minimum", low)
        form.addRow("Maximum", high)
        error = QtWidgets.QLabel()
        form.addRow(error)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        form.addRow(buttons)

        def apply():
            try:
                self._set_scalar_bar_clim(uid, (float(low.text()), float(high.text())))
            except (ValueError, TypeError) as exc:
                error.setText(str(exc))
                return
            dialog.accept()

        buttons.accepted.connect(apply)
        buttons.rejected.connect(dialog.reject)
        try:
            dialog.exec()
        finally:
            dialog.deleteLater()
