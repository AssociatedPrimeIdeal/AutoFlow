"""Shared case HTML, batch HTML and video actions for the main window."""

from pathlib import Path
import time
from PySide6 import QtCore, QtWidgets

from ..task_control import TaskCancelled


class ReportActionsMixin:
    def _save_current_case_html(self, *, show=True):
        from ..report_snapshot import build_report_snapshot, save_report_html
        from .html_view import open_html_document
        if not self.workspace.data_loaded:
            self.log('Load a case before opening its QC report.')
            return None
        if self._pipeline_running_guard('opening the QC report'):
            return None
        self._refresh_quality_panel()
        snapshot = build_report_snapshot(self.workspace)
        path = Path(self.pipeline._output_dir(self.workspace))/'analysis_report.html'
        save_report_html(snapshot, path)
        if show:
            pane = open_html_document(self, path, 'QC', 'qc')
            from ..report_media import media_signature
            pane.qc_signature = media_signature(self.workspace)
            pane.set_stale(False)
        elif 'qc' in getattr(self, '_html_document_panes', {}):
            pane = self._html_document_panes['qc']
            pane.load(path)
            from ..report_media import media_signature
            pane.qc_signature = media_signature(self.workspace)
            pane.set_stale(False)
        self.log(f'QC HTML saved: {path}')
        return str(path)

    def _on_open_case_qc(self):
        if not self.workspace.data_loaded:
            self._select_workflow_page('review')
            self.qc_page.setCurrentWidget(self.qc_empty)
            return
        try:
            self._save_current_case_html()
        except Exception as exc:
            self.log(f'QC VIEW ERROR: {type(exc).__name__}: {exc}')
            QtWidgets.QMessageBox.warning(self, 'QC report', str(exc))

    def _on_open_batch_qc(self):
        from .html_view import open_html_document
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open batch QC report',
            self.workspace.paths.output_dir or '.', 'HTML (*.html)')
        if path:
            try:
                open_html_document(self, path, 'Batch QC', 'batch_qc')
            except Exception as exc:
                self.log(f'BATCH QC VIEW ERROR: {type(exc).__name__}: {exc}')

    def _on_generate_qc_videos(self):
        if self._pipeline_running_guard('generating QC videos') or not self.workspace.data_loaded:
            return
        from ..report_media import available_video_jobs, report_video_records, video_config_signature, register_report_videos
        from ..rendering.jobs import export_videos_in_process
        self._on_pause()
        self._sync_params_to_ws()
        config = self._rendering_kwargs()
        config['rotate_dynamic_video'] = True
        available = available_video_jobs(self.workspace)
        signature = video_config_signature(config)
        records = {r['family']: r for r in report_video_records(self.workspace)}
        needed = [key for key in available if records.get(key, {}).get('validity') != 'current'
                  or records[key].get('config_signature') != signature]
        if needed:
            config['qc_families'] = needed
            workspace = self.workspace.copy_for_task()
            out_dir = Path(self.pipeline._output_dir(self.workspace))/'qc_videos'
            try:
                result = self._run_modal_task('Generate QC Videos', 'Rendering available computed metric families…',
                    lambda progress, log: export_videos_in_process(workspace, out_dir, config, {'qc_all': True}, progress))
                # qc_families is a dispatch selector, not a visual setting.
                stamp_config = dict(config);stamp_config.pop('qc_families', None)
                register_report_videos(self.workspace, result['outputs'], stamp_config)
                for name, error in result.get('errors', {}).items():
                    self.log(f'QC VIDEO ERROR ({name}): {error}')
                    record = self.workspace.input_state.metadata.get('report_videos', {}).get(name)
                    if record is not None:
                        record['browser_error'] = error
                self.log('QC videos: ' + ', '.join(f'{key}: {value or "unavailable"}' for key, value in result['outputs'].items()))
            except TaskCancelled:
                self.log('QC video generation cancelled.')
                return
            except Exception as exc:
                self.log(f'QC VIDEO ERROR: {type(exc).__name__}: {exc}')
                return
        self._on_open_case_qc()

    def _on_export_current_view_video(self, checked=False, *, rotate=False):
        if self._pipeline_running_guard('exporting the current 3D view') or not self.workspace.data_loaded:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Export Current 3D View Video',
            str(Path(self.pipeline._output_dir(self.workspace))/('current_view_rotation.mp4' if rotate else 'current_view_cycle.mp4')), 'MP4 (*.mp4)')
        if not path:
            return
        if not path.lower().endswith('.mp4'):
            path += '.mp4'
        from .video_options import CurrentViewVideoDialog
        dialog = CurrentViewVideoDialog(self, self.spin_fps.value(), rotate=rotate)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        progress = QtWidgets.QProgressDialog('Capturing the current 3D view…', 'Cancel', 0, 1, self)
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        self._view_export_progress = progress
        try:
            from .view_video import export_current_view
            from ..report_media import ensure_browser_video, register_report_videos
            def update(current, total):
                progress.setMaximum(total);progress.setValue(current)
            result = export_current_view(self, path, fps=dialog.fps.value(), cycles=dialog.cycles.value(),
                                         resolution=dialog.output_resolution(), **dialog.motion_options(),
                                         cancelled=progress.wasCanceled, progress=update)
            progress.close()
            self._run_modal_task('Encode browser video', 'Preparing in-app playback…',
                lambda emit, log: ensure_browser_video(path))
            register_report_videos(self.workspace, {'current_view': path}, result,
                                   camera_behavior='rotating_current_view' if result['rotate'] else 'fixed_current_view', phase_behavior='cardiac_cycle')
            self._save_current_case_html(show=False)
            self.log(f'Current 3D view video saved: {path} ({result["frames"]} frames, {result["resolution"][0]} x {result["resolution"][1]} px)')
        except TaskCancelled:
            self.log('Current 3D view video export cancelled.')
        except Exception as exc:
            self.log(f'CURRENT VIEW VIDEO ERROR: {type(exc).__name__}: {exc}')
        finally:
            progress.close();progress.deleteLater();self._view_export_progress = None
