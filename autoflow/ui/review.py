"""Native QC inspection and report export actions."""

import html
import copy
import os
import base64
import numpy as np
from datetime import datetime, timezone
from PySide6 import QtCore, QtGui, QtWidgets

from ..core.plane_state import workspace_signature
from ..qc_locations import iter_issues
from .qc_overlay import QCOverlay


class ReviewMixin:
    def _capture_report_images(self):
        from ..qc_locations import issue_anchors
        images = []
        ws = self.workspace
        signature = workspace_signature(ws)
        camera = copy.deepcopy(self.plotter.camera_position)
        preview = self.scene._interaction_preview_active
        overlay_enabled = getattr(getattr(self, "_qc_overlay", None), "enabled", False)
        overlay_on_top = self.check_qc_marker_on_top.isChecked()
        label_actor = None
        def encoded(qimage):
            buffer = QtCore.QBuffer()
            buffer.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
            qimage.save(buffer, "PNG")
            return bytes(buffer.data().toBase64()).decode("ascii")
        try:
            self.scene.set_interaction_preview(False, render=False)
            self._update_qc_overlay()
            self._qc_overlay.update(self._last_quality_report, enabled=True, on_top=True)
            anchors = issue_anchors(ws, self._last_quality_report or {})
            if anchors:
                label_actor = self.plotter.add_point_labels(self._qc_overlay.points,
                    [f"QC {i+1}" for i in range(len(anchors))], show_points=False, font_size=12,
                    always_visible=True, reset_camera=False, render=False, name="qc_report_labels")
            self.plotter.view_isometric(render=False)
            self.scene.request_render()
            self.scene.flush_render()
            image = self.plotter.screenshot(return_img=True)
            image = np.ascontiguousarray(image[:, :, :3])
            qimage = QtGui.QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QtGui.QImage.Format.Format_RGB888).copy()
            images.append({"mime": "image/png", "data_base64": encoded(qimage), "plane_id": None,
                "phase": ws.current_t, "camera": str(self.plotter.camera_position),
                "caption": "3D QC context at displayed phase; red=execution Error, yellow=QC Warning, gray=Stale. Path-level markers represent the whole path / whole cycle. "+
                    "; ".join(f"QC {i+1}: {record['issue_id']}" for i, record in enumerate(anchors)),
                "issues": [record["issue_id"] for record in anchors], "anchors": anchors,
                "dependency_signature": signature})
            index = self.ortho_viewer._selected_plane_idx
            if index is not None and 0 <= index < len(ws.planes):
                images.append({"mime": "image/png", "data_base64": encoded(self.ortho_viewer.slice_views["axial"].grab().toImage()),
                    "plane_id": ws.planes[index].plane_id, "phase": ws.current_t,
                    "caption": f"Selected plane {index} cross section and displayed contour", "dependency_signature": signature})
        finally:
            if label_actor is not None:
                self.plotter.remove_actor(label_actor, render=False)
            if hasattr(self, "_qc_overlay"):
                self._qc_overlay.update(self._last_quality_report, enabled=overlay_enabled, on_top=overlay_on_top)
            self.plotter.camera_position = camera
            self.scene.set_interaction_preview(preview, render=False)
            self.scene.request_render()
            self.scene.flush_render()
        return images

    def _on_export_analysis_report(self):
        from ..report_snapshot import build_report_snapshot, save_report_html
        if self._pipeline_running_guard("exporting an analysis report") or not self.workspace.data_loaded:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export Analysis Report",
            os.path.join(self.pipeline._output_dir(self.workspace), "analysis_report.html"), "HTML (*.html)")
        if not path:
            return
        if not path.lower().endswith(".html"):
            path += ".html"
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Report sections")
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(QtWidgets.QLabel("Required QC findings, missing results, and provenance are always included."))
        options = {}
        for key, label in (("metrics", "Metrics"), ("curves", "Flow curves"), ("images", "3D QC and selected-plane image"),
                           ("qc", "Full QC evidence and human review"), ("methods", "Parameters and methods"),
                           ("videos", "Rotation and playback videos")):
            box = QtWidgets.QCheckBox(label)
            box.setChecked(True)
            layout.addWidget(box)
            options[key] = box
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        self._on_pause()
        self._refresh_quality_panel()
        try:
            sections = [key for key, box in options.items() if box.isChecked()]
            images = self._capture_report_images() if "images" in sections else []
            snapshot = build_report_snapshot(self.workspace, images=images)
            self._run_modal_task("Export analysis report", "Writing curves and a self-contained HTML report…",
                                 lambda: save_report_html(snapshot, path, sections=sections))
            self.log("Analysis report exported: "+path)
            from .html_view import open_html_document
            open_html_document(self, path, "QC", "qc")
        except Exception as exc:
            self.log(f"REPORT EXPORT ERROR: {type(exc).__name__}: {exc}")

    def _build_qc_details(self, layout):
        self.check_qc_markers = QtWidgets.QCheckBox("Show Fail / Warn / Stale markers in 3D")
        self.check_qc_markers.setChecked(True)
        self.check_qc_markers.toggled.connect(lambda _: self._update_qc_overlay())
        self.check_qc_all = QtWidgets.QCheckBox("Show passed checks and unavailable checks")
        self.check_qc_all.toggled.connect(self._filter_qc_rows)
        layout.addWidget(self.check_qc_markers)
        self.check_qc_marker_on_top = QtWidgets.QCheckBox("Show QC markers through surfaces")
        self.check_qc_marker_on_top.setChecked(True)
        self.check_qc_marker_on_top.setToolTip("Display overlay only; physical issue coordinates and automatic QC are unchanged")
        self.check_qc_marker_on_top.toggled.connect(lambda _: self._update_qc_overlay())
        layout.addWidget(self.check_qc_marker_on_top)
        layout.addWidget(self.check_qc_all)
        self.quality_details = QtWidgets.QTextBrowser()
        self.quality_details.setMinimumHeight(130)
        self.quality_details.setMaximumHeight(230)
        layout.addWidget(self.quality_details)
        self.quality_review_note = QtWidgets.QLineEdit()
        self.quality_review_note.setPlaceholderText("Review note (kept separately from automatic QC)")
        layout.addWidget(self.quality_review_note)
        self.btn_mark_quality_reviewed = QtWidgets.QPushButton("Mark selected issue reviewed")
        self.btn_mark_quality_reviewed.clicked.connect(self._mark_quality_reviewed)
        layout.addWidget(self.btn_mark_quality_reviewed)
        self.quality_table.itemSelectionChanged.connect(self._quality_selection_changed)

    def _filter_qc_rows(self):
        if not hasattr(self, "quality_table"):
            return
        for index in range(self.quality_table.topLevelItemCount()):
            item = self.quality_table.topLevelItem(index)
            check = item.data(0, QtCore.Qt.ItemDataRole.UserRole) or {}
            item.setHidden(not self.check_qc_all.isChecked() and (check.get("status") == "pass" or
                           (check.get("status") == "not_run" and check.get("validity") != "stale")))
            if check.get("details"):
                item.setExpanded(True)

    def _update_qc_overlay(self):
        if self.scene is None:
            return
        if not hasattr(self, "_qc_overlay"):
            self._qc_overlay = QCOverlay(self.scene, self._qc_marker_selected, self)
        enabled = self._workflow_stage_key() == "review" and self.check_qc_markers.isChecked()
        report = self._last_quality_report
        if enabled and self.workspace.data_loaded and (report is None or report.get("dependency_signature") != workspace_signature(self.workspace)):
            self._refresh_quality_panel()
            return
        self._qc_overlay.update(report, enabled=enabled, on_top=self.check_qc_marker_on_top.isChecked())

    def _qc_marker_selected(self, ids):
        if len(ids) > 1:
            menu = QtWidgets.QMenu(self)
            checks = {c["issue_id"]: c for c in iter_issues(self._last_quality_report or {})}
            actions = {menu.addAction(checks[uid]["title"]+" · "+checks[uid]["summary"]): uid for uid in ids if uid in checks}
            action = menu.exec(self.cursor().pos())
            if action not in actions:
                return
            ids = [actions[action]]
        iterator = QtWidgets.QTreeWidgetItemIterator(self.quality_table)
        while iterator.value():
            item = iterator.value()
            check = item.data(0, QtCore.Qt.ItemDataRole.UserRole) or {}
            if check.get("issue_id") in ids:
                item.setHidden(False)
                self.quality_table.setCurrentItem(item)
                self.quality_table.scrollToItem(item)
                return
            iterator += 1

    def _quality_selection_changed(self):
        item = self.quality_table.currentItem()
        check = item.data(0, QtCore.Qt.ItemDataRole.UserRole) if item is not None else None
        if not check:
            return
        esc = lambda value: html.escape(str(value if value is not None else "—"))
        self.quality_details.setHtml("<b>"+esc(check["title"])+"</b><br>"+"<br>".join(
            f"<b>{label}:</b> {esc(value)}" for label, value in (
                ("Automatic QC", check["status"]), ("Validity", check.get("validity")),
                ("Scope", str(check.get("scope_kind"))+" / "+str(check.get("phase_scope"))),
                ("Evidence", check.get("summary")), ("Observed", check.get("observed")),
                ("Threshold", check.get("threshold")), ("Next action", check.get("action")),
                ("Location", check.get("location_method")), ("Issue ID", check.get("issue_id")),
                ("Human review", check.get("review", {}).get("status", "unreviewed")))))
        self.quality_review_note.setText(check.get("review", {}).get("note", ""))
        self.btn_mark_quality_reviewed.setEnabled(check.get("validity") == "current")
        uid_set = set(check.get("plane_ids", []))
        index = next((i for i, plane in enumerate(self.workspace.planes) if plane.plane_id in uid_set), None)
        if index is not None and self._plane_edit_session is None and self.ortho_viewer._roi_transaction is None:
            self._selected_plane_index = index
            uid = self._find_uid_by_indexed_data_key("plane", index)
            self._select_browser_item_by_uid(uid)
            self.ortho_viewer.set_selected_plane(index)
            self.scene.highlight_plane(uid)
        if check.get("scope_kind") == "path" and check.get("path_ids"):
            path_index = check["path_ids"][0]
            self.scene.highlight_path(self._find_uid_by_indexed_data_key("smooth_path", path_index))

    def _mark_quality_reviewed(self):
        item = self.quality_table.currentItem()
        check = item.data(0, QtCore.Qt.ItemDataRole.UserRole) if item is not None else None
        if not check or check.get("validity") != "current":
            return
        self.workspace.input_state.metadata.setdefault("quality_reviews", {})[check["issue_id"]] = {
            "status": "reviewed", "note": self.quality_review_note.text(),
            "at": datetime.now(timezone.utc).isoformat(), "dependency_signature": check["dependency_signature"]}
        issue_id = check["issue_id"]
        self._refresh_quality_panel()
        self._qc_marker_selected([issue_id])
