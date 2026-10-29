from pathlib import Path
from PySide6 import QtCore, QtGui, QtWidgets


THEMES = {
    "dark": {
        "window": "#202226", "panel": "#2b2d32", "inset": "#35373d",
        "control": "#303238", "border": "#3c3f46", "border_hover": "#585d68",
        "text": "#eef0f4", "secondary": "#adb3be", "accent": "#1976d2",
        "accent_hover": "#2485e6", "selection": "#263d59", "hover": "#343b46",
        "disabled_bg": "#292b30", "disabled_text": "#727984", "scrollbar": "#555c67",
        "danger": "#f19aa5", "danger_border": "#64424c", "tooltip": "#363940",
        "on_accent": "#ffffff", "curve": "#62a8fa", "cursor": "#7e96b4",
        "grid": "#414650", "spine": "#4b515c", "title": "#e1e7ef",
        "detail_label": "#dbe3ef", "detail_unit": "#c9d5e5", "detail_value": "#9dccff",
    },
    "light": {
        "window": "#f2f3f5", "panel": "#fcfcfd", "inset": "#eef0f3",
        "control": "#ffffff", "border": "#e1e3e8", "border_hover": "#b9c2ce",
        "text": "#24272e", "secondary": "#6c7380", "accent": "#1678df",
        "accent_hover": "#2c88e8", "selection": "#e8f0fb", "hover": "#f0f3f8",
        "disabled_bg": "#ebedf1", "disabled_text": "#9aa1ac", "scrollbar": "#bec5cd",
        "danger": "#b7475c", "danger_border": "#e0b9c2", "tooltip": "#ffffff",
        "on_accent": "#ffffff", "curve": "#1678df", "cursor": "#8b9caf",
        "grid": "#e3e7ed", "spine": "#c7ced8", "title": "#3d4653",
        "detail_label": "#46546a", "detail_unit": "#52627a", "detail_value": "#145fa6",
    },
}


def theme_colors(widget=None):
    app = QtWidgets.QApplication.instance()
    name = app.property("autoflow_theme") if app is not None else "dark"
    return THEMES.get(str(name), THEMES["dark"])


def readable_color(value, widget=None):
    """Lift dark categorical text on dark chrome without recoloring scene data."""
    color = QtGui.QColor(value)
    if (color.isValid() and theme_colors(widget) is THEMES["dark"]
            and color.lightnessF() < .48):
        color = color.lighter(175)
    return color


def workflow_status_colors(state, widget=None):
    if theme_colors(widget) is THEMES["dark"]:
        return {
            "Ready": ("#91cfa6", "#263d31"),
            "Needs review": ("#e6bf7b", "#403624"),
            "Not ready": ("#eea2ab", "#412c32"),
            "Unavailable": ("#e6bf7b", "#403624"),
        }.get(state, ("#b6bdc9", "#30343c"))
    return {
        "Ready": ("#1b5e20", "#e8f5e9"),
        "Needs review": ("#8a4b00", "#fff4d6"),
        "Not ready": ("#8b1e1e", "#fdecec"),
        "Unavailable": ("#8a4b00", "#fff4d6"),
    }.get(state, ("#555555", "#eeeeee"))


_APP_STYLESHEET = """
QMainWindow, QDialog {
    background: @window;
    color: @text;
}
QWidget {
    font-size: 13px;
    font-family: "__AUTOFLOW_FONT__";
    color: @text;
}
QMenuBar {
    background: @panel;
    border-bottom: 1px solid @border;
    padding: 2px 6px;
}
QMenuBar::item {
    border-radius: 7px;
    padding: 5px 9px;
}
QMenuBar::item:selected, QMenu::item:selected {
    background: @hover;
    color: @text;
}
QMenu {
    background: @panel;
    border: 1px solid @border;
    padding: 5px;
}
QMenu::item {
    border-radius: 7px;
    padding: 6px 28px 6px 10px;
}
QGroupBox {
    background: @panel;
    border: 1px solid @border;
    border-radius: 10px;
    font-weight: 500;
    margin-top: 11px;
    padding: 10px 8px 8px 8px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
    color: @secondary;
}
QFrame#sliceView {
    background: #0b0c0e;
    border: 1px solid #46484e;
    border-radius: 3px;
}
QWidget#sliceHeader {
    background: #202226;
    border: 0;
}
QLabel#sliceTitle {
    background: transparent;
    color: #e4e5e8;
    font-size: 12px;
    font-weight: 600;
}
QLabel#slicePosition {
    background: transparent;
    color: #a6a8af;
    font-size: 11px;
}
QWidget#segmentationLabelPanel {
    background: @inset;
    border: 1px solid @border;
    border-radius: 3px;
}
QToolBar {
    background: @panel;
    border: 0;
    border-bottom: 1px solid @border;
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
    background: @hover;
    border-color: @border_hover;
}
QToolButton:checked {
    background: @selection;
    border-color: @accent;
    color: @text;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox,
QPlainTextEdit, QTextEdit, QTreeView, QTableView, QListView {
    background: @panel;
    border: 1px solid @border;
    border-radius: 7px;
    selection-background-color: @accent;
    selection-color: @on_accent;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    min-height: 27px;
    padding: 0 7px;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus,
QComboBox:focus, QPlainTextEdit:focus, QTextEdit:focus,
QTreeView:focus, QTableView:focus, QListView:focus {
    border: 1px solid @accent;
}
QSpinBox, QDoubleSpinBox {
    padding-left: 10px;
    padding-right: 28px;
    min-height: 30px;
}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border;
    width: 24px;
    border: 0;
    border-left: 1px solid @border;
    background: @control;
}
QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-position: top right;
    border-top-right-radius: 6px;
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-position: bottom right;
    border-bottom-right-radius: 6px;
}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background: @hover;
}
QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed,
QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {
    background: @selection;
}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow,
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    width: 9px;
    height: 6px;
}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { image: url("__SPIN_UP__"); }
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow { image: url("__SPIN_DOWN__"); }
QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled { image: url("__SPIN_UP_DISABLED__"); }
QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled { image: url("__SPIN_DOWN_DISABLED__"); }
QFrame#measurementSummary { background: transparent; border: 0; }
QLabel[measurementRole="heading"] { font-size: 18px; font-weight: 600; color: @text; }
QLabel[measurementRole="reading"] { font-size: 36px; font-weight: 600; color: @detail_value; }
QLabel[measurementRole="secondary"] { font-size: 14px; color: @detail_label; }
QLabel[measurementRole="statValue"] { font-size: 18px; font-weight: 600; color: @detail_value; }
QLabel#browserTitle { font-size: 16px; font-weight: 600; color: @text; }
QTreeView#browserObjects { font-size: 14px; }
QFrame#browserInlineInspector {
    background: @window;
    border: 1px solid @border;
    border-left: 2px solid @accent;
    border-radius: 5px;
}
QFrame#browserInlineInspector QGroupBox { border: 0; padding-top: 0; margin-top: 0; }
QToolButton#browserAppearanceToggle {
    text-align: left;
    color: @detail_label;
    font-size: 14px;
    padding: 5px 2px;
    border: 0;
}
QFrame#selectionInfo { background: transparent; border: 0; }
QLabel[infoRole="title"] { font-size: 18px; font-weight: 600; color: @text; }
QLabel[infoRole="secondary"] { color: @detail_label; font-size: 14px; }
QLabel[infoRole="unit"] { color: @detail_unit; font-size: 14px; }
QLabel[infoRole="value"] { color: @detail_value; font-size: 16px; font-weight: 600; }
QComboBox::down-arrow { image: url("__SPIN_DOWN__"); width: 10px; height: 7px; }
QComboBox::down-arrow:disabled { image: url("__SPIN_DOWN_DISABLED__"); }
QComboBox::drop-down {
    border: 0;
    width: 24px;
}
QPushButton {
    background: @control;
    border: 1px solid @border;
    border-radius: 7px;
    color: @text;
    min-height: 29px;
    padding: 3px 10px;
}
QPushButton:hover {
    background: @hover;
    border-color: @border_hover;
}
QPushButton:pressed, QPushButton:checked {
    background: @selection;
    border-color: @accent;
    color: @text;
}
QPushButton:disabled {
    background: @disabled_bg;
    border-color: @border;
    color: @disabled_text;
}
QPushButton[role="primary"] {
    background: @accent;
    border-color: @accent;
    color: @on_accent;
    font-weight: 600;
}
QPushButton[role="primary"]:hover {
    background: @accent_hover;
    border-color: @accent_hover;
}
QPushButton[role="danger"] {
    color: @danger;
    border-color: @danger_border;
}
QPushButton[role="icon"] {
    min-width: 31px;
    max-width: 31px;
    min-height: 31px;
    max-height: 31px;
    padding: 0;
}
QHeaderView::section {
    background: @inset;
    border: 0;
    border-right: 1px solid @border;
    border-bottom: 1px solid @border;
    color: @secondary;
    font-weight: 500;
    padding: 6px;
}
QTreeView::item, QTableView::item, QListView::item {
    min-height: 24px;
}
QTreeView::item:selected, QTableView::item:selected, QListView::item:selected {
    background: @selection;
    color: @text;
}
QTabWidget::pane {
    border: 1px solid @border;
    background: @panel;
}
QTabBar::tab {
    background: @window;
    border: 1px solid @border;
    padding: 7px 12px;
}
QTabBar::tab:selected {
    background: @panel;
    border-bottom-color: @panel;
    color: @text;
    font-weight: 600;
}
QDockWidget {
    color: @secondary;
    font-weight: 500;
}
QDockWidget::title {
    background: @window;
    border-bottom: 1px solid @border;
    padding: 7px;
    text-align: left;
}
QSlider::groove:horizontal {
    background: @border;
    border-radius: 2px;
    height: 4px;
}
QSlider::sub-page:horizontal {
    background: @accent;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: @panel;
    border: 2px solid @accent;
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
    background: @scrollbar;
    border-radius: 7px;
    min-height: 24px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QSplitter::handle {
    background: @window;
}
QSplitter::handle:hover {
    background: @border_hover;
}
QProgressBar {
    background: @inset;
    border: 0;
    border-radius: 7px;
    min-height: 8px;
    text-align: center;
}
QProgressBar::chunk {
    background: @accent;
    border-radius: 7px;
}
QToolTip {
    background: @tooltip;
    border: 1px solid @border;
    color: @text;
    padding: 6px;
}
QStatusBar {
    background: @panel;
    border-top: 1px solid @border;
    color: @secondary;
}

QCheckBox::indicator:unchecked, QTreeView::indicator:unchecked {
    border: 1px solid @border_hover;
    background: @control;
    border-radius: 2px;
}
QCheckBox::indicator:unchecked:hover, QTreeView::indicator:unchecked:hover {
    border-color: @accent;
}
QPushButton[role="primary"]:disabled {
    background: @disabled_bg;
    border-color: @border;
    color: @disabled_text;
}
QTabBar::tab {
    border-top-left-radius: 7px;
    border-top-right-radius: 7px;
}
"""



def configure_high_dpi():
    for attribute_name in ("AA_EnableHighDpiScaling", "AA_UseHighDpiPixmaps"):
        attribute = getattr(QtCore.Qt.ApplicationAttribute, attribute_name, None)
        if attribute is not None:
            QtWidgets.QApplication.setAttribute(attribute, True)


def application_font_family():
    """Use an installed UI sans-serif; no bundled font is required."""
    families = set(QtGui.QFontDatabase.families())
    return next((family for family in ("Segoe UI", "Noto Sans", "Arial", "DejaVu Sans")
                 if family in families), QtWidgets.QApplication.font().family())


def apply_application_theme(app, theme="dark"):
    name = str(theme or "dark").strip().lower()
    if name not in THEMES:
        raise ValueError("ui.theme must be dark or light")
    app.setProperty("autoflow_theme", name)
    app.setStyle("Fusion")
    family = application_font_family()
    app.setFont(QtGui.QFont(family, 10))
    colors = THEMES[name]
    palette = app.palette()
    for role, key in ((QtGui.QPalette.Window, "window"), (QtGui.QPalette.WindowText, "text"),
                      (QtGui.QPalette.Base, "panel"), (QtGui.QPalette.AlternateBase, "inset"),
                      (QtGui.QPalette.Text, "text"), (QtGui.QPalette.Button, "control"),
                      (QtGui.QPalette.ButtonText, "text"), (QtGui.QPalette.Highlight, "accent"),
                      (QtGui.QPalette.HighlightedText, "on_accent"),
                      (QtGui.QPalette.ToolTipBase, "tooltip"), (QtGui.QPalette.ToolTipText, "text")):
        palette.setColor(role, QtGui.QColor(colors[key]))
    for role in (QtGui.QPalette.Text, QtGui.QPalette.WindowText, QtGui.QPalette.ButtonText):
        palette.setColor(QtGui.QPalette.Disabled, role, QtGui.QColor(colors["disabled_text"]))
    app.setPalette(palette)
    stylesheet = _APP_STYLESHEET.replace("__AUTOFLOW_FONT__", family)
    for direction in ("up", "down"):
        for suffix in ("", "-disabled"):
            token = "__SPIN_" + direction.upper() + ("_DISABLED" if suffix else "") + "__"
            path = Path(__file__).with_name("icons") / f"spin-{direction}-{name}{suffix}.svg"
            stylesheet = stylesheet.replace(token, path.as_posix())
    # Longer token names first, so @accent does not partially replace @accent_hover.
    for key in sorted(colors, key=len, reverse=True):
        stylesheet = stylesheet.replace("@" + key, colors[key])
    app.setStyleSheet(stylesheet)


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
    if str(icon_name).startswith(("SP_Media", "SP_Arrow", "SP_Browser")):
        tinted = QtGui.QPixmap(pixmap.size())
        tinted.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(tinted)
        painter.drawPixmap(0, 0, pixmap)
        painter.setCompositionMode(QtGui.QPainter.CompositionMode_SourceIn)
        painter.fillRect(tinted.rect(), widget.palette().color(QtGui.QPalette.ButtonText))
        painter.end()
        pixmap = tinted
    pixmap.setDevicePixelRatio(1.0)
    return QtGui.QIcon(pixmap)
