from PySide6 import QtCore, QtGui, QtWidgets


_APP_STYLESHEET = """
QMainWindow, QDialog {
    background: #f4f6f8;
    color: #20272d;
}
QWidget {
    font-size: 13px;
}
QMenuBar {
    background: #ffffff;
    border-bottom: 1px solid #d8dee4;
    padding: 2px 6px;
}
QMenuBar::item {
    border-radius: 4px;
    padding: 5px 9px;
}
QMenuBar::item:selected, QMenu::item:selected {
    background: #e8f3ef;
    color: #075e49;
}
QMenu {
    background: #ffffff;
    border: 1px solid #cfd6dc;
    padding: 5px;
}
QMenu::item {
    border-radius: 4px;
    padding: 6px 28px 6px 10px;
}
QGroupBox {
    background: #ffffff;
    border: 1px solid #d8dee4;
    border-radius: 6px;
    font-weight: 600;
    margin-top: 11px;
    padding: 10px 8px 8px 8px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
    color: #36414a;
}
QFrame#sliceView {
    background: #050809;
    border: 1px solid #aeb9be;
    border-radius: 3px;
}
QWidget#sliceHeader {
    background: #17282c;
    border: 0;
}
QLabel#sliceTitle {
    background: transparent;
    color: #eef4f5;
    font-size: 12px;
    font-weight: 600;
}
QLabel#slicePosition {
    background: transparent;
    color: #b8c8cc;
    font-size: 11px;
}
QWidget#segmentationLabelPanel {
    background: #eef2f3;
    border: 1px solid #c5cfd3;
    border-radius: 3px;
}
QToolBar {
    background: #ffffff;
    border: 0;
    border-bottom: 1px solid #cfd6dc;
    spacing: 4px;
    padding: 5px;
}
QToolButton {
    border: 1px solid transparent;
    border-radius: 3px;
    min-height: 26px;
    padding: 3px 7px;
}
QToolButton:hover {
    background: #eef4f2;
    border-color: #a9bbb5;
}
QToolButton:checked {
    background: #dcece6;
    border-color: #147d64;
    color: #075e49;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox,
QPlainTextEdit, QTextEdit, QTreeView, QTableView, QListView {
    background: #ffffff;
    border: 1px solid #c8d0d7;
    border-radius: 4px;
    selection-background-color: #147d64;
    selection-color: #ffffff;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    min-height: 27px;
    padding: 0 7px;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus,
QComboBox:focus, QPlainTextEdit:focus, QTextEdit:focus,
QTreeView:focus, QTableView:focus, QListView:focus {
    border: 1px solid #147d64;
}
QComboBox::drop-down {
    border: 0;
    width: 24px;
}
QPushButton {
    background: #ffffff;
    border: 1px solid #bcc6ce;
    border-radius: 4px;
    color: #263139;
    min-height: 29px;
    padding: 3px 10px;
}
QPushButton:hover {
    background: #eef4f2;
    border-color: #7d9d93;
}
QPushButton:pressed, QPushButton:checked {
    background: #dcece6;
    border-color: #147d64;
    color: #075e49;
}
QPushButton:disabled {
    background: #eef1f3;
    border-color: #d8dee4;
    color: #8a969f;
}
QPushButton[role="primary"] {
    background: #087f5b;
    border-color: #087f5b;
    color: #ffffff;
    font-weight: 600;
}
QPushButton[role="primary"]:hover {
    background: #066c4d;
    border-color: #066c4d;
}
QPushButton[role="danger"] {
    color: #a12a2a;
    border-color: #d6a0a0;
}
QPushButton[role="icon"] {
    min-width: 31px;
    max-width: 31px;
    min-height: 31px;
    max-height: 31px;
    padding: 0;
}
QHeaderView::section {
    background: #edf1f4;
    border: 0;
    border-right: 1px solid #d4dbe0;
    border-bottom: 1px solid #c8d0d7;
    color: #46525b;
    font-weight: 600;
    padding: 6px;
}
QTreeView::item, QTableView::item, QListView::item {
    min-height: 24px;
}
QTreeView::item:selected, QTableView::item:selected, QListView::item:selected {
    background: #dcece6;
    color: #163b31;
}
QTabWidget::pane {
    border: 1px solid #d8dee4;
    background: #ffffff;
}
QTabBar::tab {
    background: #e9edf0;
    border: 1px solid #d2d9df;
    padding: 7px 12px;
}
QTabBar::tab:selected {
    background: #ffffff;
    border-bottom-color: #ffffff;
    color: #075e49;
    font-weight: 600;
}
QDockWidget {
    color: #273139;
    font-weight: 600;
}
QDockWidget::title {
    background: #e9edf0;
    border-bottom: 1px solid #cfd6dc;
    padding: 7px;
    text-align: left;
}
QSlider::groove:horizontal {
    background: #d8dee4;
    border-radius: 2px;
    height: 4px;
}
QSlider::sub-page:horizontal {
    background: #147d64;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #ffffff;
    border: 2px solid #147d64;
    border-radius: 7px;
    height: 14px;
    margin: -6px 0;
    width: 14px;
}
QScrollBar:vertical {
    background: transparent;
    margin: 2px;
    width: 10px;
}
QScrollBar::handle:vertical {
    background: #b8c1c8;
    border-radius: 4px;
    min-height: 24px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QSplitter::handle {
    background: #dde3e7;
}
QSplitter::handle:hover {
    background: #9ebbb2;
}
QProgressBar {
    background: #e3e8eb;
    border: 0;
    border-radius: 4px;
    min-height: 8px;
    text-align: center;
}
QProgressBar::chunk {
    background: #147d64;
    border-radius: 4px;
}
QToolTip {
    background: #20272d;
    border: 0;
    color: #ffffff;
    padding: 5px;
}
QStatusBar {
    background: #ffffff;
    border-top: 1px solid #d8dee4;
    color: #52606a;
}
"""


def configure_high_dpi():
    for attribute_name in ("AA_EnableHighDpiScaling", "AA_UseHighDpiPixmaps"):
        attribute = getattr(QtCore.Qt.ApplicationAttribute, attribute_name, None)
        if attribute is not None:
            QtWidgets.QApplication.setAttribute(attribute, True)


def apply_application_theme(app):
    app.setStyle("Fusion")
    palette = app.palette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor("#f4f6f8"))
    palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor("#20272d"))
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor("#ffffff"))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor("#f1f4f6"))
    palette.setColor(QtGui.QPalette.Text, QtGui.QColor("#20272d"))
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor("#ffffff"))
    palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor("#263139"))
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor("#147d64"))
    palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#ffffff"))
    app.setPalette(palette)
    app.setStyleSheet(_APP_STYLESHEET)


def standard_icon(widget, icon_name):
    icon_id = getattr(QtWidgets.QStyle, str(icon_name), None)
    if icon_id is None:
        return QtGui.QIcon()
    pixmap = widget.style().standardPixmap(icon_id, None, widget)
    if pixmap.isNull():
        return QtGui.QIcon()
    target = QtCore.QSize(16, 16)
    if pixmap.size() != target:
        pixmap = pixmap.scaled(
            target,
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
    pixmap.setDevicePixelRatio(1.0)
    return QtGui.QIcon(pixmap)
