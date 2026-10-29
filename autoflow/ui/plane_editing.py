"""Plane edit transactions inside the existing Selection panel."""

import copy
import re

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..core.models import ObjectKind, PlaneData
from ..core.plane_state import (invalidate_planes, metric_validity, mutation_snapshot,
    plane_signature, refresh_plane_consistency, selected_metric, stamp_metric, workspace_signature)
from ..algorithms.paths import _project_point_to_path
from .plane_tasks import PlaneCalculationQueue


def _basis(normal):
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    u = np.cross(np.eye(3)[int(np.argmin(np.abs(n)))], n)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def transport_plane_basis(axis_u, axis_v, new_normal):
    """Minimal rigid rotation, continuous across canonical reference-axis ties."""
    u, v = np.asarray(axis_u), np.asarray(axis_v)
    old = np.cross(u, v)
    old /= np.linalg.norm(old)
    target = np.asarray(new_normal)/np.linalg.norm(new_normal)
    cross = np.cross(old, target)
    cosine = float(np.clip(np.dot(old, target), -1, 1))
    if cosine < -1+1e-10:
        return u.copy(), -v.copy()
    skew = np.array([[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]])
    rotation = np.eye(3)+skew+skew@skew/max(1+cosine, 1e-12)
    return rotation@u, rotation@v


class _PlaneCommand(QtGui.QUndoCommand):
    def __init__(self, owner, before, after, text):
        super().__init__(text)
        self.owner, self.before, self.after = owner, before, after
        self.initial = True

    def redo(self):
        if self.initial:
            self.initial = False
        else:
            self.owner._restore_plane_snapshot(self.after, persist=True)

    def undo(self):
        self.owner._restore_plane_snapshot(self.before, persist=True)


class PlaneEditingMixin:
    def _begin_roi_transaction(self, index):
        self._roi_before = mutation_snapshot(self.workspace)
        if self._play_timer.isActive():
            self._on_pause()

    def _finish_roi_transaction(self, index, changed):
        before = getattr(self, "_roi_before", None)
        self._roi_before = None
        if changed and before is not None:
            self._on_plane_roi_changed(index)
            self._plane_undo_stack.push(_PlaneCommand(self, before, mutation_snapshot(self.workspace), "Edit ROI"))
        self._last_quality_report = None
        self._refresh_selection_info()
        self.ortho_viewer._update_plane_metric_label()

    def _initialize_plane_transactions(self):
        self._plane_edit_session = None
        self._plane_draft_undo, self._plane_draft_redo = [], []
        self._plane_gesture_before = None
        self._plane_undo_stack = QtGui.QUndoStack(self)
        self._plane_jobs = PlaneCalculationQueue(self)
        self._plane_jobs.ready.connect(self._plane_result_ready)
        self._plane_jobs.failed.connect(self.log)
        self._plane_jobs.idle.connect(self._refresh_plane_edit_controls)
        self._plane_preview_timer = QtCore.QTimer(self)
        self._plane_preview_timer.setSingleShot(True)
        self._plane_preview_timer.setInterval(30)
        self._plane_preview_timer.timeout.connect(self._preview_plane_geometry)

    def _build_plane_transaction_controls(self, layout, *, source_layout=None, actions_layout=None):
        self.combo_plane_add_source = QtWidgets.QComboBox()
        self.combo_plane_add_source.addItems(["At cursor", "On selected path"])
        (source_layout if source_layout is not None else layout).addWidget(self.combo_plane_add_source)
        self.plane_edit_tools = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(self.plane_edit_tools)
        form.setContentsMargins(0, 0, 0, 0)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.WrapAllRows)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.combo_plane_edit_mode = QtWidgets.QComboBox()
        self.combo_plane_edit_mode.addItems(["Along path", "Move freely"])
        self.combo_plane_edit_mode.currentIndexChanged.connect(self._plane_edit_mode_changed)
        form.addRow("Movement", self.combo_plane_edit_mode)
        from .parameter_help import bind_help
        bind_help(form, self.combo_plane_edit_mode, "Drag a cyan 2D center or the 3D handle to move. "
            "Drag orange axes in either lower 2D view to rotate. Along path constrains translation to the centreline; "
            "Move freely retains the plane's vessel labels and path identity.")
        self.plane_center_controls, self.plane_normal_controls = [], []
        for caption, target, minimum, maximum in (
                ("Center LR / AP / FH (mm)", self.plane_center_controls, -10000, 10000),
                ("Normal LR / AP / FH", self.plane_normal_controls, -1, 1)):
            row = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            for _ in range(3):
                spin = QtWidgets.QDoubleSpinBox()
                spin.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
                spin.setMinimumWidth(76)
                spin.setRange(minimum, maximum)
                spin.setDecimals(3)
                spin.setSingleStep(.5 if maximum > 1 else .05)
                spin.setKeyboardTracking(False)
                spin.editingFinished.connect(self._apply_plane_numeric_controls)
                target.append(spin)
                row_layout.addWidget(spin)
            form.addRow(caption, row)
            bind_help(form, row, "Local physical centre coordinates in millimetres, ordered LR/AP/FH." if target is self.plane_center_controls
                      else "Plane normal direction in LR/AP/FH. A non-zero vector is normalised on Apply.")
        self.check_plane_path_locked = QtWidgets.QCheckBox("Locked to path")
        self.check_plane_path_locked.toggled.connect(self._plane_path_lock_changed)
        self.spin_plane_distance = QtWidgets.QDoubleSpinBox()
        self.spin_plane_distance.setRange(0, 10000)
        self.spin_plane_distance.setSuffix(" mm")
        self.spin_plane_distance.setDecimals(2)
        self.spin_plane_distance.setKeyboardTracking(False)
        self.spin_plane_distance.editingFinished.connect(self._move_plane_to_distance)
        form.addRow(self.check_plane_path_locked, self.spin_plane_distance)
        self.btn_flip_plane_normal = QtWidgets.QPushButton("Flip normal")
        self.btn_flip_plane_normal.clicked.connect(self._flip_plane_normal)
        self.btn_cancel_plane_edit = QtWidgets.QPushButton("Cancel / Esc")
        self.btn_cancel_plane_edit.clicked.connect(self._cancel_plane_edit)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.btn_flip_plane_normal)
        if actions_layout is None:
            row.addWidget(self.btn_cancel_plane_edit)
        else:
            actions_layout.addWidget(self.btn_cancel_plane_edit)
            self.btn_cancel_plane_edit.hide()
        form.addRow(row)
        row = QtWidgets.QHBoxLayout()
        self.btn_snap_plane = QtWidgets.QPushButton("Snap to centerline")
        self.btn_snap_plane.clicked.connect(self._snap_plane_to_path)
        self.btn_reset_plane_draft = QtWidgets.QPushButton("Reset draft")
        self.btn_reset_plane_draft.clicked.connect(self._reset_plane_draft)
        row.addWidget(self.btn_snap_plane)
        row.addWidget(self.btn_reset_plane_draft)
        form.addRow(row)
        layout.addWidget(self.plane_edit_tools)
        self.plane_edit_tools.hide()
        row = QtWidgets.QHBoxLayout()
        self.btn_undo_plane = QtWidgets.QPushButton("Undo plane")
        self.btn_redo_plane = QtWidgets.QPushButton("Redo plane")
        self.btn_undo_plane.clicked.connect(self._undo_plane_change)
        self.btn_redo_plane.clicked.connect(self._redo_plane_change)
        row.addWidget(self.btn_undo_plane)
        row.addWidget(self.btn_redo_plane)
        layout.addLayout(row)
        self.btn_cancel_plane_calculation = QtWidgets.QPushButton("Cancel plane calculation")
        self.btn_cancel_plane_calculation.clicked.connect(self._plane_jobs.cancel)
        layout.addWidget(self.btn_cancel_plane_calculation)
        self.btn_cancel_plane_calculation.hide()

    def _sync_plane_numeric_controls(self):
        session = self._plane_edit_session
        if not session or not hasattr(self, "plane_center_controls"):
            return
        plane = self.workspace.planes[session["index"]]
        for control, value in zip(self.plane_center_controls + self.plane_normal_controls,
                                  list(plane.center) + list(plane.normal)):
            blocker = QtCore.QSignalBlocker(control)
            control.setValue(float(value))
            del blocker
        blocker = QtCore.QSignalBlocker(self.check_plane_path_locked)
        self.check_plane_path_locked.setChecked(session.get("path_locked", False))
        self.check_plane_path_locked.setEnabled(plane.path_index >= 0)
        del blocker
        blocker = QtCore.QSignalBlocker(self.spin_plane_distance)
        self.spin_plane_distance.setValue(float(plane.distance))
        self.spin_plane_distance.setEnabled(plane.path_index >= 0 and session.get("path_locked", False))
        del blocker
        blocker = QtCore.QSignalBlocker(self.combo_plane_edit_mode)
        self.combo_plane_edit_mode.setCurrentIndex(0 if session.get("path_locked", False) else 1)
        del blocker

    def _start_plane_session(self, index, kind="edit", before=None):
        if self._plane_edit_session is not None:
            return False
        before = before or mutation_snapshot(self.workspace)
        original = copy.deepcopy(self.workspace.planes[index])
        draft = copy.deepcopy(original)
        draft.placement = dict(draft.placement, _draft=True)
        self.workspace.planes[index] = draft
        self._plane_edit_session = {"kind": kind, "index": index, "before": before,
                                    "original": original, "had_metrics": bool(original.metrics),
                                    "path_locked": kind == "add" and original.path_index >= 0}
        self._plane_draft_undo, self._plane_draft_redo = [], []
        return True

    def _add_manual_plane(self):
        if self._plane_edit_session is not None:
            self.statusBar().showMessage("Apply or cancel the current plane draft first.", 6000)
            return
        if self._pipeline_running_guard("adding a plane") or self._autoseg_running_guard("adding a plane"):
            return
        ws = self.workspace
        if not ws.data_loaded:
            self.log("Load a case before adding a plane.")
            return
        before = mutation_snapshot(ws)
        center = np.asarray(ws.ortho_cursor, dtype=float) * np.asarray(ws.resolution, dtype=float)
        plane = PlaneData(center=center, normal=np.array([1., 0., 0.]), label=0, path_index=-1)
        if self.combo_plane_add_source.currentIndex() == 1:
            paths = ws.centerline_paths_smooth or ws.centerline_paths
            index = int(ws.selected_path_index)
            if not 0 <= index < len(paths):
                self.log("Select a centerline path before using On selected path.")
                return
            path = np.asarray(paths[index], dtype=float).reshape(-1, 3)
            if len(path) < 2:
                self.log("The selected path needs at least two points.")
                return
            center, segment, _, distance, _, _ = _project_point_to_path(path, center)
            normal = path[min(segment+1, len(path)-1)] - path[max(0, segment)]
            if np.linalg.norm(normal) <= 1e-12:
                self.log("The selected path has no valid local tangent.")
                return
            group = str(ws.path_info[index].get("group_name", "")) if index < len(ws.path_info) else ""
            plane = PlaneData(center=center, normal=normal/np.linalg.norm(normal), label=index+1,
                              path_index=index, distance=distance, group_name=group)
        ws.planes.append(plane)
        ws.derived.plane_metrics = [dict(p.metrics or {}) for p in ws.planes]
        self._start_plane_session(len(ws.planes)-1, "add", before)
        self._sync_plane_scene(before["planes"])
        self._selected_plane_index = len(ws.planes)-1
        self._activate_plane_drag_widgets(self._selected_plane_index)
        self._refresh_browser()
        self._select_browser_item_by_uid(self._find_uid_by_indexed_data_key("plane", self._selected_plane_index))
        self._refresh_selection_info()
        self.log("New plane preview. Apply to save, or Cancel / Esc to discard.")

    def _activate_plane_drag_widgets(self, plane_idx):
        index = int(plane_idx)
        if not 0 <= index < len(self.workspace.planes):
            return
        if self._plane_edit_session is None:
            self._start_plane_session(index)
        elif self._plane_edit_session["index"] != index:
            self.log("Apply or cancel the current plane draft first.")
            return
        self._plane_internal_setup = True
        try:
            self._legacy_activate_plane_drag_widgets(index)
        finally:
            self._plane_internal_setup = False
        self._sync_plane_numeric_controls()
        if hasattr(self, "selection_tabs"):
            self.selection_tabs.setCurrentIndex(0)
            self.plane_controls_scroll.verticalScrollBar().setValue(0)
        blocker = QtCore.QSignalBlocker(self.combo_plane_edit_mode)
        self.combo_plane_edit_mode.setCurrentIndex(0 if self._plane_edit_session.get("path_locked") else 1)
        del blocker
        if self._plane_handles is not None:
            self._plane_handles.set_mode("move")
        self.ortho_viewer.set_plane_rotation_enabled(self._plane_edit_enabled)
        self._preview_plane_geometry()
        if not self._plane_edit_enabled:
            self._cancel_plane_edit()

    def _plane_edit_mode_changed(self, mode):
        if not self._plane_edit_session:
            return
        if mode == 0:
            original = self._plane_edit_session["original"]
            if original.path_index < 0:
                self.log("This plane has no assigned path. Use On selected path when adding a path-constrained plane.")
                return
            self.check_plane_path_locked.setChecked(True)
        elif mode == 1:
            self.check_plane_path_locked.setChecked(False)
        if self._plane_handles is not None:
            self._plane_handles.set_mode("move")
        self.scene.request_render()

    def _set_plane_edit_unlocked(self, unlocked):
        if unlocked:
            if not self._plane_edit_session:
                self._toggle_selected_plane_edit()
        elif self._plane_edit_session:
            self._finalize_plane_drag(self._plane_edit_session["index"])
        self._refresh_plane_edit_controls()

    def _begin_plane_2d_translation(self, name):
        if not self._plane_edit_session:
            return
        self._begin_plane_gesture()
        plane = self.workspace.planes[self._plane_edit_session["index"]]
        horizontal, vertical = self.ortho_viewer._plane_view_bases(plane.normal)[name]
        self.ortho_viewer._plane_translation_reference = {"name": name,
            "basis": (horizontal.copy(), vertical.copy())}

    def _move_plane_from_2d(self, name, horizontal, vertical):
        reference = self.ortho_viewer._plane_translation_reference
        before = self._plane_gesture_before
        if not reference or reference["name"] != name or before is None or not self._plane_edit_session:
            return
        u, v = reference["basis"]
        center = np.asarray(before["plane"].center)+horizontal*u+vertical*v
        shape = self.ortho_viewer._get_volume_shape()
        if shape is not None:
            center = np.clip(center, 0, np.asarray(shape[:3])*np.asarray(self.workspace.resolution))
        self._update_plane_from_drag(self._plane_edit_session["index"], center=center)
        self._preview_plane_geometry()
        self.scene.request_render()

    def _finish_plane_2d_translation(self, name):
        reference = self.ortho_viewer._plane_translation_reference
        if reference and reference["name"] == name:
            self.ortho_viewer._plane_translation_reference = None
            self._end_plane_gesture()

    def _begin_plane_2d_rotation(self, name):
        if not self._plane_edit_session or name not in {"coronal", "sagittal"}:
            return
        self._begin_plane_gesture()
        plane = self.workspace.planes[self._plane_edit_session["index"]]
        viewer = self.ortho_viewer
        horizontal, vertical = viewer._plane_view_bases(plane.normal)[name]
        viewer._plane_rotation_reference = {"name": name,
            "basis": (horizontal.copy(), vertical.copy()),
            "geometry": viewer._plane_orthogonal_geometry(plane, self.workspace.current_t)}
        self._plane_rotation_axis = np.cross(horizontal, vertical)
        self._plane_rotation_axis /= np.linalg.norm(self._plane_rotation_axis)

    def _rotate_plane_from_2d(self, name, angle):
        reference = self.ortho_viewer._plane_rotation_reference
        before = self._plane_gesture_before
        if not reference or reference["name"] != name or before is None or not self._plane_edit_session:
            return
        axis = self._plane_rotation_axis
        cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        rotation = np.eye(3) + np.sin(angle)*cross + (1-np.cos(angle))*(cross@cross)
        self._plane_axis_u = rotation @ before["u"]
        self._plane_axis_v = rotation @ before["v"]
        self._update_plane_from_drag(self._plane_edit_session["index"], normal=rotation @ before["plane"].normal)
        self.scene.request_render()

    def _finish_plane_2d_rotation(self, name):
        reference = self.ortho_viewer._plane_rotation_reference
        if reference and reference["name"] == name:
            self.ortho_viewer._plane_rotation_reference = None
            self._end_plane_gesture()

    def _snap_plane_to_path(self):
        if self._plane_edit_session and self._plane_edit_session["original"].path_index >= 0:
            self.check_plane_path_locked.setChecked(True)
            self._begin_plane_gesture()
            index = self._plane_edit_session["index"]
            self._update_plane_from_drag(index, center=self.workspace.planes[index].center)
            self._end_plane_gesture()

    def _reset_plane_draft(self):
        if not self._plane_edit_session:
            return
        self._begin_plane_gesture()
        index = self._plane_edit_session["index"]
        plane = copy.deepcopy(self._plane_edit_session["original"])
        plane.placement["_draft"] = True
        self.workspace.planes[index] = plane
        self._plane_axis_u, self._plane_axis_v = _basis(plane.normal)
        self.scene.update_plane_geometry(self._find_uid_by_indexed_data_key("plane", index))
        self._position_plane_handles()
        self._sync_plane_numeric_controls()
        self._end_plane_gesture()

    def _clear_plane_drag_widgets(self):
        if self._plane_edit_session is not None and not getattr(self, "_plane_internal_setup", False):
            self._cancel_plane_edit()
        else:
            self._legacy_clear_plane_drag_widgets()

    def _toggle_selected_plane_edit(self):
        if self._plane_edit_enabled:
            self._finalize_plane_drag(self._plane_drag_index)
        elif 0 <= int(self._selected_plane_index) < len(self.workspace.planes):
            if not self._pipeline_running_guard("editing a plane"):
                self._activate_plane_drag_widgets(int(self._selected_plane_index))

    def _begin_plane_gesture(self):
        if self._play_timer.isActive():
            self._on_pause()
        if self._plane_edit_session:
            self._plane_gesture_before = {"plane": copy.deepcopy(self.workspace.planes[self._plane_edit_session["index"]]),
                "u": copy.deepcopy(self._plane_axis_u), "v": copy.deepcopy(self._plane_axis_v)}

    def _end_plane_gesture(self):
        if self._plane_gesture_before is not None and self._plane_edit_session:
            plane = self.workspace.planes[self._plane_edit_session["index"]]
            if plane_signature(plane) != plane_signature(self._plane_gesture_before["plane"]):
                self._plane_draft_undo.append(self._plane_gesture_before)
                self._plane_draft_redo.clear()
        self._plane_gesture_before = None
        self._plane_preview_timer.stop()
        self._preview_plane_geometry()
        self._refresh_plane_edit_controls()

    def _update_plane_from_drag(self, plane_idx, center=None, normal=None):
        self._legacy_update_plane_from_drag(plane_idx, center=center, normal=normal)
        self._sync_plane_numeric_controls()
        if self._plane_edit_session and not self._plane_preview_timer.isActive():
            self._plane_preview_timer.start()

    def _preview_plane_geometry(self):
        if self._plane_edit_session and self._plane_edit_enabled:
            reference = self.ortho_viewer._plane_translation_reference
            if reference is not None:
                plane = self.workspace.planes[self._plane_edit_session["index"]]
                offset = np.asarray(plane.center)-self._plane_gesture_before["plane"].center
                u, v = reference["basis"]
                self.ortho_viewer.slice_views[reference["name"]].set_plane_drag_center(
                    float(np.dot(offset, u)), float(np.dot(offset, v)))
                return  # Keep the slice image fixed under the center drag; reslice on release.
            self.ortho_viewer._draft_plane_basis = (self._plane_axis_u.copy(), self._plane_axis_v.copy())
            self.ortho_viewer.set_selected_plane(self._plane_edit_session["index"])
            self.ortho_viewer.label_plane_metric.setText("Plane preview · Apply to update measurements")

    def _apply_plane_numeric_controls(self):
        if not self._plane_edit_session:
            return
        center = np.array([spin.value() for spin in self.plane_center_controls])
        normal = np.array([spin.value() for spin in self.plane_normal_controls])
        if np.linalg.norm(normal) <= 1e-12:
            self.log("Plane normal must be non-zero.")
            return
        self._begin_plane_gesture()
        self._plane_axis_u, self._plane_axis_v = transport_plane_basis(self._plane_axis_u, self._plane_axis_v, normal)
        self._update_plane_from_drag(self._plane_edit_session["index"], center, normal)
        self._end_plane_gesture()

    def _plane_path_lock_changed(self, locked):
        if not self._plane_edit_session:
            return
        self._begin_plane_gesture()
        plane = self.workspace.planes[self._plane_edit_session["index"]]
        self._plane_edit_session["path_locked"] = bool(locked and plane.path_index >= 0)
        if self._plane_edit_session["path_locked"]:
            self._update_plane_from_drag(self._plane_edit_session["index"], center=plane.center)
        self._sync_plane_numeric_controls()
        self._end_plane_gesture()

    def _move_plane_to_distance(self):
        if not self._plane_edit_session:
            return
        plane = self.workspace.planes[self._plane_edit_session["index"]]
        paths = self.workspace.centerline_paths_smooth or self.workspace.centerline_paths
        if not 0 <= plane.path_index < len(paths):
            return
        path = np.asarray(paths[plane.path_index], dtype=float)
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        distance = np.clip(self.spin_plane_distance.value(), 0, arc[-1])
        center = np.array([np.interp(distance, arc, path[:, axis]) for axis in range(3)])
        self._begin_plane_gesture()
        self._update_plane_from_drag(self._plane_edit_session["index"], center=center)
        self._end_plane_gesture()

    def _flip_plane_normal(self):
        if self._plane_edit_session:
            self._begin_plane_gesture()
            index = self._plane_edit_session["index"]
            self._plane_axis_v = -self._plane_axis_v
            self._update_plane_from_drag(index, normal=-self.workspace.planes[index].normal)
            self._end_plane_gesture()

    def _finalize_plane_drag(self, plane_idx):
        session = self._plane_edit_session
        if session is None:
            return self._legacy_finalize_plane_drag(plane_idx)
        plane = self.workspace.planes[session["index"]]
        if not np.all(np.isfinite(plane.center)) or not np.all(np.isfinite(plane.normal)) or np.linalg.norm(plane.normal) <= 1e-12:
            self.log("Enter finite coordinates and a non-zero plane normal before Apply.")
            return
        plane.normal = np.asarray(plane.normal) / np.linalg.norm(plane.normal)
        if self.workspace.flow_raw is not None:
            bounds = np.asarray(self.workspace.flow_raw.shape[:3])*np.asarray(self.workspace.resolution)
            if np.any(plane.center < 0) or np.any(plane.center > bounds):
                self.log("Plane center is outside the loaded physical field of view. Move it inside before Apply.")
                return
        # Re-express attached manual UV contours in the final canonical basis.
        # Continuous handle axes can differ when the reference-axis selector
        # changes at a normal-component tie.
        old = session["original"]
        if self._plane_axis_u is not None and self._plane_axis_v is not None:
            final_u, final_v = _basis(plane.normal)
            matrix = np.array([[np.dot(self._plane_axis_u, final_u), np.dot(self._plane_axis_v, final_u)],
                               [np.dot(self._plane_axis_u, final_v), np.dot(self._plane_axis_v, final_v)]])
            def converted(points):
                return (np.asarray(points, dtype=float).reshape(-1, 2) @ matrix.T).tolist() if points else []
            plane.roi_polygon_uv_mm = converted(old.roi_polygon_uv_mm)
            plane.roi_edit_operations = copy.deepcopy(old.roi_edit_operations)
            for operations in plane.roi_edit_operations.values():
                for operation in operations:
                    if "polygon" in operation:
                        operation["polygon"] = converted(operation["polygon"])
        plane.placement.pop("_draft", None)
        changed = session["kind"] == "add" or plane_signature(plane) != plane_signature(old)
        self._legacy_clear_plane_drag_widgets()
        self._plane_preview_timer.stop()
        self._plane_edit_session = None
        if changed:
            invalidate_planes(self.workspace, [plane.plane_id], "Plane geometry changed", geometry=True)
            self._last_quality_report = None
            after = mutation_snapshot(self.workspace)
            self._plane_undo_stack.push(_PlaneCommand(self, session["before"], after,
                                                    "Add plane" if session["kind"] == "add" else "Edit plane"))
            self._persist_plane_outputs(include_pixelwise=False)
            if session["had_metrics"]:
                self._schedule_plane_calculation(session["index"])
        self._refresh_plane_edit_controls()
        self._refresh_selection_info()
        self.scene.request_render()

    def _cancel_plane_edit(self):
        session = self._plane_edit_session
        if session is None:
            return
        self._plane_preview_timer.stop()
        self._legacy_clear_plane_drag_widgets()
        self._plane_edit_session = None
        self._restore_plane_snapshot(session["before"], persist=False)
        self.statusBar().showMessage("Plane draft discarded.", 5000)

    def _restore_plane_snapshot(self, snapshot, persist=True):
        ws = self.workspace
        if snapshot.get("case_identity") != (id(ws), str(ws.paths.flow_path), ws.input_state.source_group):
            self.log("Plane undo belongs to another case and was discarded.")
            return
        self._plane_jobs.cancel()
        previous = list(self.workspace.planes)
        same_input = snapshot["input_signature"] == workspace_signature(ws, include_analysis=False)
        ws.planes = copy.deepcopy(snapshot["planes"])
        ws.derived.plane_metrics = copy.deepcopy(snapshot["metrics"])
        ws.derived.plane_qc = copy.deepcopy(snapshot["qc"])
        ws.derived.pwv_results = copy.deepcopy(snapshot["pwv"])
        ws.derived.plane_pixelwise_file = snapshot["pixelwise"]
        ws.pathline_cache, ws.pathline_seed_cache = dict(snapshot["pathlines"]), dict(snapshot["pathline_seeds"])
        ws.pathline_colors = dict(snapshot["pathline_colors"])
        ws.active_pathline_plane_indices = list(snapshot["active_pathlines"])
        ws.analysis_revision += 1
        if not same_input:
            invalidate_planes(ws, [p.plane_id for p in ws.planes], "Input changed since this undo snapshot")
        by_id = {plane.plane_id: plane for plane in ws.planes}
        for key, ids in snapshot["group_plane_ids"].items():
            if key in ws.multilabel_groups:
                ws.multilabel_groups[key]["planes"] = [by_id[uid] for uid in ids if uid in by_id]
        self._last_quality_report = None
        self._sync_plane_scene(previous)
        self._selected_plane_index = min(int(self._selected_plane_index), len(ws.planes)-1)
        self.ortho_viewer.set_selected_plane(self._selected_plane_index if self._selected_plane_index >= 0 else None)
        self._refresh_browser()
        self._refresh_selection_info()
        if persist:
            self._persist_plane_outputs(include_pixelwise=False)

    def _sync_plane_scene(self, previous_planes):
        ids = {i: plane.plane_id for i, plane in enumerate(previous_planes)}
        indices = {plane.plane_id: i for i, plane in enumerate(self.workspace.planes)}
        present = set()
        with self.scene.render_batch():
            for obj in list(self.workspace.scene_objects.values()):
                if obj.kind == ObjectKind.PLANE:
                    old_index = int(str(obj.data_key).rsplit("_", 1)[-1])
                    uid = ids.get(old_index)
                    if uid not in indices:
                        self.scene.remove_object(obj.uid)
                        continue
                    index = indices[uid]
                    obj.data_key = obj.data_key.rsplit("_", 1)[0] + "_" + str(index)
                    if re.fullmatch(r"plane\s+\d+", obj.name):
                        obj.name = f"plane {index}"
                    self.scene.update_plane_geometry(obj.uid, render=False)
                    present.add(uid)
            for plane in self.workspace.planes:
                if plane.plane_id not in present:
                    i = indices[plane.plane_id]
                    self.workspace.add_object(name=f"plane {i}", kind=ObjectKind.PLANE,
                        data_key=self._plane_data_key("plane", i), group_name=plane.group_name,
                        visible=True, opacity=.75, color="yellow", line_width=2)
            self.scene.sync_from_workspace(rebuild_prefixes=())

    def _undo_plane_change(self):
        self._change_plane_history(False)

    def _redo_plane_change(self):
        self._change_plane_history(True)

    def _change_plane_history(self, redo):
        if not self._plane_edit_session:
            self._plane_undo_stack.redo() if redo else self._plane_undo_stack.undo()
            return
        source = self._plane_draft_redo if redo else self._plane_draft_undo
        target = self._plane_draft_undo if redo else self._plane_draft_redo
        if source:
            index = self._plane_edit_session["index"]
            target.append({"plane": copy.deepcopy(self.workspace.planes[index]), "u": copy.deepcopy(self._plane_axis_u), "v": copy.deepcopy(self._plane_axis_v)})
            state = source.pop()
            self.workspace.planes[index] = state["plane"]
            self._plane_axis_u, self._plane_axis_v = state["u"], state["v"]
            self.scene.update_plane_geometry(self._find_uid_by_indexed_data_key("plane", index))
            self._position_plane_handles()
            self._sync_plane_numeric_controls()
            self._preview_plane_geometry()
            self._refresh_plane_edit_controls()

    def _schedule_plane_calculation(self, index):
        ws = self.workspace
        if not 0 <= int(index) < len(ws.planes) or ws.flow_raw is None or ws.segmask_binary is None:
            return
        plane = ws.planes[int(index)]
        payload = {"plane": copy.deepcopy(plane), "signature": plane_signature(plane),
            "input_signature": workspace_signature(ws, include_analysis=False),
            "data": {"flow": ws.flow_raw, "mask": ws.segmask_binary, "spacing": np.asarray(ws.resolution).copy(),
                "origin": np.asarray(ws.origin).copy(), "rr": float(ws.rr), "branches": ws.branch_labels,
                "labels3": ws.segmask_labels_3d, "labels4": ws.segmask_labels,
                "paths": ws.centerline_paths_smooth or ws.centerline_paths,
                "path_info": copy.deepcopy(ws.path_info), "forks": copy.deepcopy(ws.forks),
                "tke": ws.derived.tke_array, "pg": ws.derived.pressure_gradient_array,
                "pressure": ws.derived.relative_pressure_array, "wss": ws.derived.wss_surfaces,
                "pressure_support": ws.derived.pressure_gradient_support_mask}}
        stale = set(ws.input_state.metadata.get("stale_metric_families", []))
        for family, keys in (("wss", ("wss",)), ("tke", ("tke",)), ("pressure", ("pg", "pressure", "pressure_support"))):
            if family in stale:
                for key in keys:
                    payload["data"][key] = [] if key == "wss" else None
        self._plane_jobs.submit(payload)
        self.statusBar().showMessage(f"Updating plane {index} measurements…")

    @QtCore.Slot(object)
    def _plane_result_ready(self, result):
        request = result["request"]
        ws = self.workspace
        if request["input_signature"] != workspace_signature(ws, include_analysis=False):
            return
        index = next((i for i, p in enumerate(ws.planes) if p.plane_id == request["plane"].plane_id), None)
        if index is None or plane_signature(ws.planes[index]) != request["signature"]:
            return
        ws.planes[index].metrics = stamp_metric(ws.planes[index], result["metric"], index)
        refresh_plane_consistency(ws)
        self._last_quality_report = None
        self._persist_plane_outputs(include_pixelwise=False)
        self._refresh_selection_info()
        self.ortho_viewer.refresh()
        self.statusBar().showMessage(f"Plane {index} measurements updated.", 5000)

    def _recompute_dragged_plane_metrics(self, persist=False, frame_only=False):
        # Full selected-plane series also refreshes peaks, derived ROI summaries
        # and every carry-forward dependency. Global derived fields are reused.
        if self._plane_drag_index is not None:
            self._schedule_plane_calculation(int(self._plane_drag_index))
        self._plane_drag_metrics_dirty = False
