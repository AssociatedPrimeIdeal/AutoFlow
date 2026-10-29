"""HTML report pages and a separate documentation window."""

from pathlib import Path
import os
import sys
from PySide6 import QtCore, QtWidgets


def configure_html_runtime():
    """Configure HTML composition before creating Chromium or Qt contexts."""
    if sys.platform.startswith('linux'):
        # Browser composition only; VTK retains its independent GPU backend.
        flags = os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS', '')
        if '--disable-gpu' not in flags.split():
            os.environ['QTWEBENGINE_CHROMIUM_FLAGS'] = (flags + ' --disable-gpu').strip()
        os.environ.setdefault('QT_QUICK_BACKEND', 'software')
    if QtWidgets.QApplication.instance() is None:
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)


def _release_3d_context(window):
    # EGL/GLX must not remain current while Qt initializes its HTML compositor.
    plotter = getattr(window, 'plotter', None)
    if plotter is not None:
        plotter.ren_win.ReleaseCurrent()


class HtmlDocumentPane(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty('autoflowHtmlDocument', True)
        configure_html_runtime()
        # Keep Chromium out of numerical workers and ordinary GUI startup.
        from PySide6.QtWebEngineWidgets import QWebEngineView
        from PySide6.QtWebEngineCore import QWebEngineSettings
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.toolbar = QtWidgets.QHBoxLayout()
        self.view = QWebEngineView(self)
        self.view.settings().setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, False)
        for label, callback in [('Back', self.view.back), ('Forward', self.view.forward), ('Reload', self.view.reload)]:
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(callback)
            self.toolbar.addWidget(button)
        self.location = QtWidgets.QLabel()
        self.location.setTextFormat(QtCore.Qt.PlainText)
        self.location.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.toolbar.addWidget(self.location, 1)
        self.stale_notice = QtWidgets.QLabel('Workspace changed — refresh QC')
        self.stale_notice.setStyleSheet('font-weight: 600; color: #dfaa49;')
        self.stale_notice.hide()
        self.toolbar.addWidget(self.stale_notice)
        layout.addLayout(self.toolbar)
        layout.addWidget(self.view, 1)
        self.view.urlChanged.connect(lambda url: self.location.setText(url.fileName() or url.host()))
        self.view.page().printRequested.connect(self._print_pdf)

    def load(self, location):
        url = location if isinstance(location, QtCore.QUrl) else QtCore.QUrl.fromLocalFile(str(Path(location).resolve()))
        self.view.load(url)

    def set_stale(self, stale):
        self.stale_notice.setVisible(bool(stale))

    def _print_pdf(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save report PDF', 'analysis_report.pdf', 'PDF (*.pdf)')
        if path:
            self.view.page().printToPdf(path if path.lower().endswith('.pdf') else path + '.pdf')

    def shutdown(self):
        if self.view is not None:
            view, self.view = self.view, None
            view.stop()
            page = view.page()
            page.deleteLater()
            QtCore.QCoreApplication.sendPostedEvents(page, QtCore.QEvent.Type.DeferredDelete)
            view.deleteLater()
            QtCore.QCoreApplication.sendPostedEvents(view, QtCore.QEvent.Type.DeferredDelete)


class HtmlDocumentDialog(QtWidgets.QDialog):
    def __init__(self, window, title):
        super().__init__(window)
        self.main_window = window
        self.setObjectName('documentationDialog')
        self.setWindowTitle('AutoFlow ' + title)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowModality(QtCore.Qt.NonModal)
        self.resize(1240, 840)
        self.setMinimumSize(720, 480)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self.pane = HtmlDocumentPane(self)
        layout.addWidget(self.pane)

    def reject(self):
        # QDialog's native Escape path can bypass the Python done override.
        _release_3d_context(self.main_window)
        self.pane.shutdown()
        super().reject()

    def done(self, result):
        # Esc/reject and the native close button share the same shutdown path.
        _release_3d_context(self.main_window)
        self.pane.shutdown()
        super().done(result)

    def closeEvent(self, event):
        _release_3d_context(self.main_window)
        self.pane.shutdown()
        super().closeEvent(event)


def open_html_document(window, location, title, key):
    _release_3d_context(window)
    panes = getattr(window, '_html_document_panes', None)
    if panes is None:
        panes = window._html_document_panes = {}
    pane = panes.get(key)
    if key in {'qc', 'batch_qc'}:
        if pane is None:
            pane = HtmlDocumentPane(window.qc_page)
            panes[key] = pane
            window.qc_page.addWidget(pane)
            actions = [('Inspect in 3D', window._inspect_qc_in_3d),
                       ('Refresh QC' if key == 'qc' else 'Current case', window._on_open_case_qc),
                       ('Open batch…', window._on_open_batch_qc),
                       ('Export', window._on_open_export_page)]
            for label, callback in actions:
                button = QtWidgets.QPushButton(label)
                button.clicked.connect(callback)
                pane.toolbar.addWidget(button)
        pane.load(location)
        window._select_workflow_page('review')
        window.qc_page.setCurrentWidget(pane)
    else:
        dialogs = getattr(window, '_html_document_dialogs', None)
        if dialogs is None:
            dialogs = window._html_document_dialogs = {}
        dialog = dialogs.get(key)
        if dialog is None:
            dialog = HtmlDocumentDialog(window, title)
            dialogs[key] = dialog
            pane = panes[key] = dialog.pane

            def closed(_result):
                if dialogs.get(key) is dialog:
                    dialogs.pop(key, None)
                    panes.pop(key, None)

            dialog.finished.connect(closed)
        pane.load(location)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
    return pane
