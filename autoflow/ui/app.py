import json
from dataclasses import dataclass
import os
import sys
import time
import traceback
from functools import partial

import numpy as np
import pyvista as pv
from PyQt5 import QtCore, QtGui, QtWidgets
from pyvistaqt import QtInteractor
from pyvista import _vtk
from scipy.ndimage import label as ndi_label

from ..algorithms import (
    inspect_dicom_case,
    resolve_input_case,
    scan_dicom_cases,
    load_segmentation_file,
    generate_threshold_segmentation,
    generate_nnunet_auto_segmentation,
    save_segmentation_file,
    segmentation_timestamp,
)
from ..core.models import DicomParameterOverrides, ObjectKind, StepId, Workspace
from ..core.pipeline import PipelineEngine
from ..config import apply_config_bundle_to_workspace, load_config_bundle
from ..algorithms import compute_plane_metrics, apply_internal_consistency_to_metrics, compute_plane_metrics_multithread, augment_plane_metrics_with_derived, save_plane_pixelwise_h5
from .editors import PlaneEditor, SkeletonEditor
from .dicom_confirm import DicomImportDialog
from .ortho_viewer import OrthoViewer
from .segmentation import SegmentationConfigDialog, SegmentationDock, SOURCE_LABELS
from .viewer import SceneController


def _parse_grouped_index(data_key, prefix):
    token = f"{prefix}_"
    if not isinstance(data_key, str) or not data_key.startswith(token):
        return None
    suffix = data_key[len(token):]
    tail = suffix.rsplit("_", 1)[-1]
    if tail.isdigit():
        return int(tail)
    return None


def _parse_plane_index(data_key):
    return _parse_grouped_index(data_key, "plane")


def _parse_path_index(data_key):
    return _parse_grouped_index(data_key, "smooth_path")


def _parse_pathline_index(data_key):
    return _parse_grouped_index(data_key, "pathline")


def _default_segmentation_color(label_id):
    palette = [
        "#ff6b6b", "#4dabf7", "#51cf66", "#ffd43b", "#f783ac",
        "#74c0fc", "#63e6be", "#ffa94d", "#b197fc", "#a9e34b",
    ]
    idx = (max(1, int(label_id)) - 1) % len(palette)
    return palette[idx]


@dataclass
class _AutoSegmentationResult:
    seg: np.ndarray
    provenance: dict
    sidecar: str
    elapsed_sec: float


class _AutoSegmentationWorker(QtCore.QObject):
    progress = QtCore.pyqtSignal(dict)
    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, mag, flow, resolution, origin, *, model_folder, backend, checkpoint_name, device, auto_label_map, sidecar_path):
        super().__init__()
        self._mag = np.asarray(mag, dtype=np.float32).copy()
        self._flow = np.asarray(flow, dtype=np.float32).copy()
        self._resolution = np.asarray(resolution, dtype=np.float32).copy()
        self._origin = np.asarray(origin, dtype=np.float32).copy()
        self._model_folder = str(model_folder)
        self._backend = str(backend)
        self._checkpoint_name = str(checkpoint_name)
        self._device = str(device)
        self._auto_label_map = auto_label_map
        self._sidecar_path = str(sidecar_path)

    @QtCore.pyqtSlot()
    def run(self):
        t_start = time.perf_counter()
        try:
            seg, provenance = generate_nnunet_auto_segmentation(
                mag=self._mag,
                flow=self._flow,
                resolution=self._resolution,
                origin=self._origin,
                model_folder=self._model_folder,
                backend=self._backend,
                checkpoint_name=self._checkpoint_name,
                device=self._device,
                auto_label_map=self._auto_label_map,
                progress_callback=self.progress.emit,
            )
            save_segmentation_file(
                self._sidecar_path,
                seg,
                resolution=self._resolution,
                origin=self._origin,
                provenance=provenance,
            )
            elapsed = time.perf_counter() - t_start
            self.finished.emit(_AutoSegmentationResult(seg=seg, provenance=provenance, sidecar=self._sidecar_path, elapsed_sec=float(elapsed)))
        except Exception:
            self.failed.emit(traceback.format_exc())


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, config_dir=None):
        super().__init__()
        self.setWindowTitle("AutoFlow")
        self.resize(1800, 980)
        self._config_dir = config_dir
        self._config_bundle = load_config_bundle(config_dir)
        self.workspace = Workspace()
        apply_config_bundle_to_workspace(self.workspace, self._config_bundle)
        self.pipeline = PipelineEngine()
        self.scene = None
        self._play_timer = QtCore.QTimer(self)
        self._play_timer.timeout.connect(self._on_play_tick)
        self._edit_mode = None
        self._edit_points = None
        self._edit_edges = None
        self._edit_selected_idx = None
        self._edit_edge_mode = False
        self._edit_edge_src_idx = None
        self._edit_selected_edge_idx = None
        self._edit_sel_edge_poly = None
        self._edit_sel_edge_actor = None
        self._edit_poly = None
        self._edit_actor = None
        self._edit_edge_poly = None
        self._edit_edge_actor = None
        self._edit_sel_poly = None
        self._edit_sel_actor = None
        self._edit_widget = None
        self._edit_pick_enabled = False
        self._vtk_left_click_obs_id = None
        self._vtk_keypress_obs_id = None
        self._vtk_point_picker = None
        self._edit_overlay_dialog = None
        self._edit_info_label = None
        self._edit_status_label = None
        self._edit_btn_edge = None
        self._edit_original_points = None
        self._edit_original_edges = None
        self._plane_drag_active = False
        self._plane_drag_index = None
        self._plane_widget_initializing = False
        self._plane_drag_metrics_dirty = False
        self._selected_plane_index = -1
        self._seg_edit_active = False
        self._seg_edit_history = []
        self._seg_edit_future = []
        self._seg_surface_rebuild_timer = QtCore.QTimer(self)
        self._seg_surface_rebuild_timer.setSingleShot(True)
        self._seg_surface_rebuild_timer.timeout.connect(self._rebuild_segmentation_surface)
        self._plane_drag_timer = QtCore.QTimer(self)
        self._plane_drag_timer.setSingleShot(True)
        self._plane_drag_timer.timeout.connect(lambda: self._recompute_dragged_plane_metrics(persist=False))
        self._loader_progress_dialog = None
        self._autoseg_thread = None
        self._autoseg_worker = None
        self._autoseg_progress_dialog = None
        self._autoseg_started_at = None
        self._build_ui()
        self._bind_scene()
        self._esc_shortcut = QtWidgets.QShortcut(QtCore.Qt.Key_Escape, self)
        self._esc_shortcut.setContext(QtCore.Qt.ApplicationShortcut)
        self._esc_shortcut.activated.connect(self._force_exit_edit)
        QtCore.QTimer.singleShot(0, self._setup_focus_behavior)
        self.ortho_viewer.set_segmentation_edit_handler(self._handle_segmentation_edit)
        self._refresh_all()

    def _setup_focus_behavior(self):
        try:
            self.plotter.setFocusPolicy(QtCore.Qt.ClickFocus)
        except Exception:
            pass

    def _build_ui(self):
        self._build_menu()
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root.addWidget(splitter, 1)
        left = QtWidgets.QWidget()
        left_lay = QtWidgets.QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(6)
        self._build_browser(left_lay)
        mid = QtWidgets.QWidget()
        mid_lay = QtWidgets.QVBoxLayout(mid)
        mid_lay.setContentsMargins(0, 0, 0, 0)
        mid_lay.setSpacing(6)
        self.plotter = QtInteractor(self)
        self.plotter.setFocusPolicy(QtCore.Qt.ClickFocus)
        mid_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        mid_splitter.addWidget(self.plotter)
        step_and_params = QtWidgets.QWidget()
        sp_lay = QtWidgets.QVBoxLayout(step_and_params)
        sp_lay.setContentsMargins(0, 0, 0, 0)
        sp_lay.setSpacing(4)
        self._build_step_buttons(sp_lay)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        params_widget = QtWidgets.QWidget()
        self.params_layout = QtWidgets.QVBoxLayout(params_widget)
        self.params_layout.setContentsMargins(4, 4, 4, 4)
        self._build_preprocess_params()
        self._build_skeleton_params()
        self._build_plane_params()
        self._build_streamline_params()
        self._build_derived_params()
        self.params_layout.addStretch()
        scroll.setWidget(params_widget)
        sp_lay.addWidget(scroll, 1)
        mid_splitter.addWidget(step_and_params)
        mid_splitter.setStretchFactor(0, 4)
        mid_splitter.setStretchFactor(1, 2)
        mid_lay.addWidget(mid_splitter, 1)
        right = QtWidgets.QWidget()
        right_lay = QtWidgets.QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        self.ortho_viewer = OrthoViewer(self.workspace, self)
        right_lay.addWidget(self.ortho_viewer)
        splitter.addWidget(left)
        splitter.addWidget(mid)
        splitter.addWidget(right)
        splitter.setSizes([300, 850, 450])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)
        splitter.setStretchFactor(2, 2)
        bot = QtWidgets.QWidget()
        bot_lay = QtWidgets.QVBoxLayout(bot)
        bot_lay.setContentsMargins(0, 0, 0, 0)
        bot_lay.setSpacing(4)
        bot_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        root.addWidget(bot_splitter, 0)
        timeline_w = QtWidgets.QWidget()
        tl_lay = QtWidgets.QVBoxLayout(timeline_w)
        tl_lay.setContentsMargins(0, 0, 0, 0)
        self._build_timeline(tl_lay)
        bot_splitter.addWidget(timeline_w)
        sel_w = QtWidgets.QWidget()
        sel_lay = QtWidgets.QVBoxLayout(sel_w)
        sel_lay.setContentsMargins(0, 0, 0, 0)
        self._build_selection_info(sel_lay)
        bot_splitter.addWidget(sel_w)
        log_w = QtWidgets.QWidget()
        log_lay = QtWidgets.QVBoxLayout(log_w)
        log_lay.setContentsMargins(0, 0, 0, 0)
        self._build_log(log_lay)
        bot_splitter.addWidget(log_w)
        bot_splitter.setSizes([40, 80, 60])
        self._build_segmentation_dock()

    def _build_browser(self, parent):
        grp = QtWidgets.QGroupBox("Browser")
        lay = QtWidgets.QVBoxLayout(grp)
        self.tree_objects = QtWidgets.QTreeWidget()
        self.tree_objects.setHeaderLabels(["Name", "Kind", "Visible"])
        self.tree_objects.setColumnWidth(0, 200)
        self.tree_objects.itemSelectionChanged.connect(self._on_browser_select)
        self.tree_objects.itemChanged.connect(self._on_tree_item_changed)
        self.tree_objects.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.tree_objects.customContextMenuRequested.connect(self._on_browser_ctx_menu)
        lay.addWidget(self.tree_objects)
        row = QtWidgets.QHBoxLayout()
        self.btn_delete_obj = QtWidgets.QPushButton("Delete Selected")
        self.btn_delete_obj.clicked.connect(self._on_delete_object)
        row.addWidget(self.btn_delete_obj)
        row.addStretch()
        lay.addLayout(row)
        parent.addWidget(grp, 3)

    def _build_step_buttons(self, parent):
        grp = QtWidgets.QGroupBox("Steps")
        gl = QtWidgets.QGridLayout(grp)
        self.step_buttons = {}
        for row_idx, steps in enumerate([StepId.top_row_steps(), StepId.bottom_row_steps(), StepId.extra_row_steps()]):
            for i, s in enumerate(steps):
                b = QtWidgets.QPushButton(s.label)
                b.clicked.connect(partial(self._run_single_step, s))
                self.step_buttons[s] = b
                gl.addWidget(b, row_idx, i)
        btn_run_all = QtWidgets.QPushButton("▶▶ Run All (Generate → Metrics → WSS/TKE/PG)")
        btn_run_all.setStyleSheet("QPushButton { background-color: #2a6; color: white; font-weight: bold; padding: 4px; }")
        btn_run_all.clicked.connect(self._run_all_pipeline)
        gl.addWidget(btn_run_all, 3, 0, 1, 4)
        parent.addWidget(grp, 0)

    def _build_preprocess_params(self):
        grp = QtWidgets.QGroupBox("Input / Background Correction")
        fl = QtWidgets.QFormLayout(grp)
        self.chk_bpc_enabled = QtWidgets.QCheckBox()
        self.chk_bpc_enabled.setChecked(False)
        self.spin_bpc_fit_order = QtWidgets.QSpinBox()
        self.spin_bpc_fit_order.setRange(0, 3)
        self.spin_bpc_fit_order.setValue(3)
        self.spin_bpc_threshold = QtWidgets.QDoubleSpinBox()
        self.spin_bpc_threshold.setDecimals(3)
        self.spin_bpc_threshold.setRange(0.001, 10.0)
        self.spin_bpc_threshold.setSingleStep(0.01)
        self.spin_bpc_threshold.setValue(0.1)
        self.spin_dual_venc_ratio1 = QtWidgets.QDoubleSpinBox()
        self.spin_dual_venc_ratio1.setDecimals(4)
        self.spin_dual_venc_ratio1.setRange(-10.0, 10.0)
        self.spin_dual_venc_ratio1.setSingleStep(0.01)
        self.spin_dual_venc_ratio1.setValue(0.0)
        self.spin_dual_venc_ratio2 = QtWidgets.QDoubleSpinBox()
        self.spin_dual_venc_ratio2.setDecimals(4)
        self.spin_dual_venc_ratio2.setRange(-10.0, 10.0)
        self.spin_dual_venc_ratio2.setSingleStep(0.01)
        self.spin_dual_venc_ratio2.setValue(0.0)
        self.edit_input_resolution = QtWidgets.QLineEdit("1.0, 1.0, 1.0")
        self.edit_input_venc = QtWidgets.QLineEdit("150.0, 150.0, 150.0")
        self.edit_input_spatial_order = QtWidgets.QLineEdit("LR, AP, FH")
        self.edit_input_venc_order = QtWidgets.QLineEdit("LR, AP, FH")
        fl.addRow("Enable Correction", self.chk_bpc_enabled)
        fl.addRow("MSAC Corr Fit Order", self.spin_bpc_fit_order)
        fl.addRow("MSAC Threshold", self.spin_bpc_threshold)
        fl.addRow("Dual-VENC Ratio1", self.spin_dual_venc_ratio1)
        fl.addRow("Dual-VENC Ratio2", self.spin_dual_venc_ratio2)
        fl.addRow("Current Resolution XYZ", self.edit_input_resolution)
        fl.addRow("Current VENC XYZ", self.edit_input_venc)
        fl.addRow("Current Spatial Order", self.edit_input_spatial_order)
        fl.addRow("Current VENC Order", self.edit_input_venc_order)
        self.params_layout.addWidget(grp)

    def _build_skeleton_params(self):
        grp = QtWidgets.QGroupBox("Generate Skeleton Parameters")
        fl = QtWidgets.QFormLayout(grp)
        self.chk_remove_small_cc = QtWidgets.QCheckBox()
        self.chk_remove_small_cc.setChecked(False)
        self.edit_min_cc_volume = QtWidgets.QLineEdit("50.0")
        self.chk_closing = QtWidgets.QCheckBox()
        self.chk_closing.setChecked(True)
        self.chk_opening = QtWidgets.QCheckBox()
        self.chk_gaussian = QtWidgets.QCheckBox()
        self.chk_gaussian.setChecked(True)
        self.edit_gauss_sigma = QtWidgets.QLineEdit("0.5")
        fl.addRow("Remove Small CC", self.chk_remove_small_cc)
        fl.addRow(u"Min Volume (mm\u00b3)", self.edit_min_cc_volume)
        fl.addRow("Closing", self.chk_closing)
        fl.addRow("Opening", self.chk_opening)
        fl.addRow("Gaussian", self.chk_gaussian)
        fl.addRow(u"Gauss \u03c3", self.edit_gauss_sigma)
        self.params_layout.addWidget(grp)

    def _build_plane_params(self):
        grp = QtWidgets.QGroupBox("Generate Planes Parameters")
        fl = QtWidgets.QFormLayout(grp)
        self.radio_plane_by_distance = QtWidgets.QRadioButton("By Distance")
        self.radio_plane_center = QtWidgets.QRadioButton("Center of Path")
        self.radio_plane_center.setChecked(True)
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(self.radio_plane_by_distance)
        mode_row.addWidget(self.radio_plane_center)
        mode_w = QtWidgets.QWidget()
        mode_w.setLayout(mode_row)
        self.edit_plane_dist = QtWidgets.QLineEdit("20.0")
        self.edit_plane_start = QtWidgets.QLineEdit("5.0")
        self.edit_plane_end = QtWidgets.QLineEdit("0.0")
        self.edit_plane_smooth_win = QtWidgets.QLineEdit("15")
        self.edit_plane_smooth_poly = QtWidgets.QLineEdit("2")
        self.edit_plane_inter_time = QtWidgets.QLineEdit("10")
        fl.addRow("Plane Mode", mode_w)
        fl.addRow("Cross-section Distance (mm)", self.edit_plane_dist)
        fl.addRow("Start Distance (mm)", self.edit_plane_start)
        fl.addRow("End Distance (mm)", self.edit_plane_end)
        fl.addRow("SavGol Window", self.edit_plane_smooth_win)
        fl.addRow("SavGol Polyorder", self.edit_plane_smooth_poly)
        fl.addRow("Inter-time", self.edit_plane_inter_time)
        self.params_layout.addWidget(grp)

    def _build_streamline_params(self):
        grp = QtWidgets.QGroupBox("Streamline / Pathline Parameters")
        fl = QtWidgets.QFormLayout(grp)
        self.edit_sl_ratio = QtWidgets.QLineEdit("0.02")
        self.edit_sl_maxsteps = QtWidgets.QLineEdit("2000")
        self.edit_sl_terminal = QtWidgets.QLineEdit("0.01")
        self.edit_pathline_color = QtWidgets.QLineEdit("deepskyblue")
        fl.addRow("Seed Ratio", self.edit_sl_ratio)
        fl.addRow("Max Steps", self.edit_sl_maxsteps)
        fl.addRow("Terminal Speed", self.edit_sl_terminal)
        fl.addRow("Pathline Color", self.edit_pathline_color)
        self.params_layout.addWidget(grp)

    def _build_derived_params(self):
        grp_wss = QtWidgets.QGroupBox("WSS Parameters")
        fl_wss = QtWidgets.QFormLayout(grp_wss)
        self.edit_dm_smoothing = QtWidgets.QLineEdit("200")
        self.edit_dm_viscosity = QtWidgets.QLineEdit("4.0")
        self.edit_dm_inward = QtWidgets.QLineEdit("auto")
        self.chk_dm_parabolic = QtWidgets.QCheckBox()
        self.chk_dm_parabolic.setChecked(True)
        self.chk_dm_noslip = QtWidgets.QCheckBox()
        self.chk_dm_noslip.setChecked(False)
        fl_wss.addRow("Smoothing Iterations", self.edit_dm_smoothing)
        fl_wss.addRow(u"Viscosity (mPa\u00b7s)", self.edit_dm_viscosity)
        fl_wss.addRow("Inward Distance (mm or auto)", self.edit_dm_inward)
        fl_wss.addRow("Parabolic Fitting", self.chk_dm_parabolic)
        fl_wss.addRow("No-Slip Condition", self.chk_dm_noslip)
        self.params_layout.addWidget(grp_wss)

        grp_tke = QtWidgets.QGroupBox("Flow / TKE / Pressure Gradient Parameters")
        fl_tke = QtWidgets.QFormLayout(grp_tke)
        self.edit_dm_rho = QtWidgets.QLineEdit("1060.0")
        self.edit_dm_pg_smoothing_sigma = QtWidgets.QLineEdit("0.0")
        self.edit_dm_stepsize = QtWidgets.QLineEdit("5")
        self.edit_dm_tube = QtWidgets.QLineEdit("0.1")
        self.chk_dm_multithread = QtWidgets.QCheckBox()
        self.chk_dm_multithread.setChecked(False)
        fl_tke.addRow(u"Density \u03c1 (kg/m\u00b3)", self.edit_dm_rho)
        fl_tke.addRow("Pressure Gradient Gaussian Sigma (vox)", self.edit_dm_pg_smoothing_sigma)
        fl_tke.addRow("Step Size", self.edit_dm_stepsize)
        fl_tke.addRow("Tube Radius", self.edit_dm_tube)
        fl_tke.addRow("Multi-thread Metrics", self.chk_dm_multithread)
        self.params_layout.addWidget(grp_tke)

    def _build_timeline(self, parent):
        grp = QtWidgets.QGroupBox("Timeline")
        tl = QtWidgets.QHBoxLayout(grp)
        self.btn_prev = QtWidgets.QPushButton(u"\u25c0")
        self.btn_prev.clicked.connect(self._on_prev_frame)
        self.btn_play = QtWidgets.QPushButton(u"\u25b6 Play")
        self.btn_play.clicked.connect(self._on_play)
        self.btn_pause = QtWidgets.QPushButton(u"\u23f8 Pause")
        self.btn_pause.clicked.connect(self._on_pause)
        self.btn_next = QtWidgets.QPushButton(u"\u25b6")
        self.btn_next.clicked.connect(self._on_next_frame)
        self.slider_t = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_t.setRange(0, 0)
        self.slider_t.valueChanged.connect(self._on_t_changed)
        self.lab_t = QtWidgets.QLabel("0")
        self.spin_interval = QtWidgets.QSpinBox()
        self.spin_interval.setRange(10, 2000)
        self.spin_interval.setValue(120)
        self.spin_interval.setSuffix(" ms")
        for w in [self.btn_prev, self.btn_play, self.btn_pause, self.btn_next]:
            tl.addWidget(w)
        tl.addWidget(self.slider_t, 1)
        tl.addWidget(self.lab_t)
        tl.addWidget(self.spin_interval)
        parent.addWidget(grp)

    def _build_selection_info(self, parent):
        grp = QtWidgets.QGroupBox("Selection")
        lay = QtWidgets.QHBoxLayout(grp)
        box_plane = QtWidgets.QGroupBox("Plane")
        lay_plane = QtWidgets.QVBoxLayout(box_plane)
        self.text_plane_info = QtWidgets.QPlainTextEdit()
        self.text_plane_info.setReadOnly(True)
        self.text_plane_info.setMaximumHeight(82)
        lay_plane.addWidget(self.text_plane_info)
        box_path = QtWidgets.QGroupBox("Path")
        lay_path = QtWidgets.QVBoxLayout(box_path)
        self.text_path_info = QtWidgets.QPlainTextEdit()
        self.text_path_info.setReadOnly(True)
        self.text_path_info.setMaximumHeight(82)
        lay_path.addWidget(self.text_path_info)
        lay.addWidget(box_plane, 1)
        lay.addWidget(box_path, 1)
        parent.addWidget(grp)

    def _build_log(self, parent):
        grp = QtWidgets.QGroupBox("Log")
        ll = QtWidgets.QVBoxLayout(grp)
        self.console = QtWidgets.QTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumHeight(80)
        ll.addWidget(self.console)
        parent.addWidget(grp)

    def _build_segmentation_dock(self):
        self.segmentation_dock = QtWidgets.QDockWidget("Segmentation", self)
        self.segmentation_dock.setObjectName("SegmentationDock")
        self.segmentation_panel = SegmentationDock(self)
        self.segmentation_dock.setWidget(self.segmentation_panel)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.segmentation_dock)
        self.segmentation_panel.combo_source.currentIndexChanged.connect(self._on_segmentation_source_changed)
        self.segmentation_panel.check_visible.toggled.connect(self._on_segmentation_visibility_changed)
        self.segmentation_panel.slider_opacity.valueChanged.connect(self._on_segmentation_opacity_changed)
        self.segmentation_panel.table_labels.itemSelectionChanged.connect(self._on_segmentation_label_selected)
        self.segmentation_panel.table_labels.itemChanged.connect(self._on_segmentation_label_item_changed)
        self.segmentation_panel.spin_active_label.valueChanged.connect(self._on_active_label_changed)
        self.segmentation_panel.edit_active_name.editingFinished.connect(self._on_active_label_name_changed)
        self.segmentation_panel.btn_active_color.clicked.connect(self._on_active_label_color_clicked)
        self.segmentation_panel.chk_edit_all_timepoints.toggled.connect(self._on_segmentation_edit_scope_changed)
        self.segmentation_panel.chk_editing_enabled.toggled.connect(self._on_segmentation_edit_enabled_changed)
        for tool_name, button in [
            ("brush", self.segmentation_panel.btn_tool_brush),
            ("erase", self.segmentation_panel.btn_tool_erase),
            ("relabel", self.segmentation_panel.btn_tool_relabel),
        ]:
            button.clicked.connect(partial(self._set_segmentation_tool, tool_name))
        self.segmentation_panel.spin_brush_radius.valueChanged.connect(self._on_segmentation_brush_radius_changed)
        self.segmentation_panel.btn_undo.clicked.connect(self._undo_segmentation_edit)
        self.segmentation_panel.btn_redo.clicked.connect(self._redo_segmentation_edit)
        self.segmentation_panel.btn_apply.clicked.connect(self._apply_segmentation_edits)
        self.segmentation_panel.btn_cancel.clicked.connect(self._cancel_segmentation_edits)

    def _build_menu(self):
        mb = self.menuBar()
        mf = mb.addMenu("File")
        for label, slot in [
            ("Open H5", self._on_open_h5),
            ("Import DICOM Directory", self._on_import_dicom_directory),
            ("Clear Workspace", self._on_close_workspace),
            ("Exit", self.close),
        ]:
            a = QtWidgets.QAction(label, self)
            a.triggered.connect(slot)
            mf.addAction(a)
        mv = mb.addMenu("View")
        for label, slot in [("Reset Camera", lambda: self.scene.reset_camera()), ("Toggle Axes", lambda: self.scene.toggle_axes()),
            ("White BG", lambda: self.scene.set_background("white")), ("Dark BG", lambda: self.scene.set_background("#202124"))]:
            a = QtWidgets.QAction(label, self)
            a.triggered.connect(slot)
            mv.addAction(a)
        ms = mb.addMenu("Segmentation")
        for label, slot in [
            ("Configure Segmentation...", self._on_configure_segmentation),
            ("Use Original Segmentation", self._on_use_original_segmentation),
            ("Import Segmentation...", self._on_import_segmentation),
            ("Save Active Segmentation...", self._on_save_active_segmentation),
            ("Reset Active To Original", self._on_reset_active_to_original),
        ]:
            a = QtWidgets.QAction(label, self)
            a.triggered.connect(slot)
            ms.addAction(a)

    def _bind_scene(self):
        self.scene = SceneController(self.plotter, self.workspace, self.log)
        self.scene.initialize()
        self.scene.enable_plane_picking(self._on_3d_plane_picked)
        self.scene.enable_path_picking(self._on_3d_path_picked)

    def _segmentation_object_name(self):
        source = self.workspace.segmentation.active_source or "active"
        return f"segmentation ({SOURCE_LABELS.get(source, source.title())})"

    def _ensure_segmentation_label_metadata(self, labels=None):
        seg = self.workspace.segmentation
        if labels is None:
            labels = self.workspace.segmentation_display_3d()
        if labels is None:
            return
        label_values = [int(x) for x in np.unique(labels) if int(x) != 0]
        for label_id in label_values:
            key = str(int(label_id))
            seg.label_names.setdefault(key, f"Label {int(label_id)}")
            seg.label_colors.setdefault(key, _default_segmentation_color(label_id))
        if seg.active_label <= 0:
            seg.active_label = label_values[0] if label_values else 1

    def _has_working_segmentation(self):
        seg = self.workspace.segmentation
        return seg.working_labels_3d is not None or seg.working_labels_4d is not None

    def _working_segmentation_array(self):
        seg = self.workspace.segmentation
        if seg.working_labels_4d is not None:
            return seg.working_labels_4d
        return seg.working_labels_3d

    def _set_working_segmentation_array(self, arr):
        seg = self.workspace.segmentation
        if arr is None:
            seg.working_labels_3d = None
            seg.working_labels_4d = None
            return
        arr = np.asarray(arr, dtype=np.int16)
        if arr.ndim == 4:
            seg.working_labels_4d = arr
            seg.working_labels_3d = None
        elif arr.ndim == 3:
            seg.working_labels_3d = arr
            seg.working_labels_4d = None
        else:
            raise ValueError(f"working segmentation must be 3D or 4D, got shape={arr.shape}")

    def _current_working_labels_3d(self):
        seg = self.workspace.segmentation
        if seg.working_labels_4d is not None:
            arr = np.asarray(seg.working_labels_4d, dtype=np.int16)
            t = min(max(0, int(self.workspace.current_t)), max(0, arr.shape[3] - 1))
            return arr[..., t]
        if seg.working_labels_3d is not None:
            return seg.working_labels_3d
        return None

    def _segmentation_labels_for_ui(self):
        seg = self.workspace.segmentation
        if seg.working_labels_4d is not None:
            return self._current_working_labels_3d()
        if not seg.edit_all_timepoints:
            display = self.workspace.segmentation_display_4d()
            if display is not None and display.ndim == 4:
                t = min(max(0, int(self.workspace.current_t)), max(0, display.shape[3] - 1))
                return np.asarray(display[..., t], dtype=np.int16)
        return self.workspace.segmentation_display_3d()

    def _sync_segmentation_scene_object(self):
        ws = self.workspace
        uid = self._find_uid_by_data_key("segmask_raw_surface")
        seg = ws.get_active_segmentation()
        if seg is None:
            if uid is not None:
                self.scene.remove_object(uid)
            return
        if uid is None:
            uid = ws.add_object(
                name=self._segmentation_object_name(),
                kind=ObjectKind.SEGMENTATION,
                data_key="segmask_raw_surface",
                visible=bool(ws.segmentation.visible),
                opacity=float(ws.segmentation.opacity),
                scalars="label",
                cmap="tab10",
                dynamic=True,
                show_scalar_bar=True,
                scalar_bar_title="Label",
            )
        obj = ws.scene_objects.get(uid)
        if obj is None:
            return
        obj.name = self._segmentation_object_name()
        obj.visible = bool(ws.segmentation.visible)
        obj.opacity = float(ws.segmentation.opacity)
        obj.scalars = "label"
        obj.cmap = "tab10"
        obj.dynamic = True
        obj.show_scalar_bar = True
        obj.scalar_bar_title = "Label"

    def _refresh_segmentation_ui(self):
        panel = self.segmentation_panel
        ws = self.workspace
        seg = ws.segmentation
        self._ensure_segmentation_label_metadata()

        panel.combo_source.blockSignals(True)
        panel.combo_source.clear()
        for source in ws.segmentation_source_names():
            panel.combo_source.addItem(SOURCE_LABELS.get(source, source.title()), source)
        if seg.active_source:
            idx = panel.combo_source.findData(seg.active_source)
            if idx >= 0:
                panel.combo_source.setCurrentIndex(idx)
        panel.combo_source.blockSignals(False)

        panel.check_visible.blockSignals(True)
        panel.check_visible.setChecked(bool(seg.visible))
        panel.check_visible.blockSignals(False)
        panel.slider_opacity.blockSignals(True)
        panel.slider_opacity.setValue(int(round(float(seg.opacity) * 100.0)))
        panel.slider_opacity.blockSignals(False)

        provenance = ws.get_active_segmentation_provenance()
        panel.text_provenance.setPlainText(json.dumps(provenance, ensure_ascii=False, indent=2) if provenance else "")

        labels = self._segmentation_labels_for_ui()
        panel.table_labels.blockSignals(True)
        panel.table_labels.clearContents()
        if labels is None:
            panel.table_labels.setRowCount(0)
        else:
            label_values = [int(x) for x in np.unique(labels) if int(x) != 0]
            panel.table_labels.setRowCount(len(label_values))
            for row, label_id in enumerate(label_values):
                key = str(int(label_id))
                count = int(np.sum(labels == label_id))
                items = [
                    QtWidgets.QTableWidgetItem(str(int(label_id))),
                    QtWidgets.QTableWidgetItem(seg.label_names.get(key, f"Label {label_id}")),
                    QtWidgets.QTableWidgetItem(str(count)),
                    QtWidgets.QTableWidgetItem(seg.label_colors.get(key, _default_segmentation_color(label_id))),
                ]
                items[0].setFlags(items[0].flags() & ~QtCore.Qt.ItemIsEditable)
                items[2].setFlags(items[2].flags() & ~QtCore.Qt.ItemIsEditable)
                items[0].setData(QtCore.Qt.UserRole, int(label_id))
                for col, item in enumerate(items):
                    panel.table_labels.setItem(row, col, item)
                if int(label_id) == int(seg.active_label):
                    panel.table_labels.selectRow(row)
        panel.table_labels.blockSignals(False)

        panel.spin_active_label.blockSignals(True)
        panel.spin_active_label.setValue(max(1, int(seg.active_label)))
        panel.spin_active_label.blockSignals(False)
        active_key = str(int(seg.active_label))
        panel.edit_active_name.blockSignals(True)
        panel.edit_active_name.setText(seg.label_names.get(active_key, f"Label {seg.active_label}"))
        panel.edit_active_name.blockSignals(False)
        color = seg.label_colors.get(active_key, _default_segmentation_color(seg.active_label))
        panel.btn_active_color.setText(color)
        panel.btn_active_color.setStyleSheet(f"QPushButton {{ background-color: {color}; color: black; }}")
        panel.spin_brush_radius.blockSignals(True)
        panel.spin_brush_radius.setValue(int(seg.brush_radius))
        panel.spin_brush_radius.blockSignals(False)
        panel.chk_edit_all_timepoints.blockSignals(True)
        panel.chk_edit_all_timepoints.setChecked(bool(seg.edit_all_timepoints))
        panel.chk_edit_all_timepoints.blockSignals(False)
        panel.chk_editing_enabled.blockSignals(True)
        panel.chk_editing_enabled.setChecked(bool(seg.editing_enabled))
        panel.chk_editing_enabled.blockSignals(False)
        for tool_name, button in [
            ("brush", panel.btn_tool_brush),
            ("erase", panel.btn_tool_erase),
            ("relabel", panel.btn_tool_relabel),
        ]:
            button.blockSignals(True)
            button.setChecked(seg.tool == tool_name)
            button.blockSignals(False)
        panel.btn_undo.setEnabled(bool(self._seg_edit_history))
        panel.btn_redo.setEnabled(bool(self._seg_edit_future))
        panel.btn_apply.setEnabled(self._has_working_segmentation())
        panel.btn_cancel.setEnabled(self._has_working_segmentation())
        n_labels = 0 if labels is None else len([int(x) for x in np.unique(labels) if int(x) != 0])
        dirty = "dirty" if seg.dirty else "clean"
        active_src = seg.active_source or "none"
        working = seg.working_source or "-"
        edit_state = "enabled" if seg.editing_enabled else "locked"
        edit_mode = "3D-all-frames" if seg.edit_all_timepoints else f"4D-current-frame t={int(ws.current_t)}"
        panel.label_status.setText(
            f"Source: {active_src}   Labels: {n_labels}   State: {dirty}   Working: {working}   Edit: {edit_state}   Scope: {edit_mode}"
        )

    def _refresh_segmentation_preview(self):
        self._ensure_segmentation_label_metadata()
        self._sync_segmentation_scene_object()
        self._refresh_segmentation_ui()
        self.ortho_viewer.refresh()
        self._seg_surface_rebuild_timer.start(180)

    def _rebuild_segmentation_surface(self):
        self.scene.invalidate_cache("segmask_raw_surface")
        self.scene.rebuild_dynamic()
        self._refresh_scene()

    def _reset_segmentation_edit_history(self):
        self._seg_edit_history = []
        self._seg_edit_future = []

    def _push_segmentation_history(self):
        working = self._working_segmentation_array()
        if working is None:
            return
        self._seg_edit_history.append(np.asarray(working, dtype=np.int16).copy())
        if len(self._seg_edit_history) > 32:
            self._seg_edit_history = self._seg_edit_history[-32:]
        self._seg_edit_future = []

    def _begin_segmentation_edit_session(self):
        ws = self.workspace
        if ws.get_active_segmentation() is None:
            return False
        if not self._has_working_segmentation():
            if ws.segmentation.edit_all_timepoints:
                display = ws.segmentation_display_3d()
                if display is None:
                    return False
                self._set_working_segmentation_array(np.asarray(display, dtype=np.int16).copy())
            else:
                display = ws.segmentation_display_4d()
                if display is None:
                    return False
                self._set_working_segmentation_array(np.asarray(display, dtype=np.int16).copy())
            ws.segmentation.working_source = ws.segmentation.active_source or "imported"
            ws.segmentation.dirty = False
            self._seg_edit_active = True
            self._reset_segmentation_edit_history()
        return True

    def _commit_segmentation_source_change(self, source, log_message=None):
        ws = self.workspace
        if not ws.activate_segmentation_source(source):
            self.log(f"Segmentation source unavailable: {source}")
            return False
        ws.reset_segmentation_results()
        self._selected_plane_index = -1
        self.scene.highlight_plane(None)
        self.scene.highlight_path(None)
        self.scene.show_forks_for_path(-1)
        self.ortho_viewer.set_selected_plane(None)
        self._clear_plane_drag_widgets()
        self._seg_edit_active = False
        self._reset_segmentation_edit_history()
        self._ensure_segmentation_label_metadata(ws.segmentation_display_3d())
        self._sync_segmentation_scene_object()
        self.scene.invalidate_cache()
        self.scene.sync_from_workspace()
        self._refresh_all()
        self.ortho_viewer.refresh()
        if log_message:
            self.log(log_message)
        return True

    def _import_segmentation_path(self, path):
        if self.workspace.segmentation.dirty:
            self.log("Apply or cancel segmentation edits before importing a new source.")
            return False
        if not self.workspace.data_loaded:
            self.log("Load a case before importing segmentation.")
            return False
        try:
            seg, provenance = load_segmentation_file(
                path,
                spatial_shape=self.workspace.flow_raw.shape[:3] if self.workspace.flow_raw is not None else None,
                time_count=self.workspace.time_count(),
            )
        except Exception as e:
            self.log(f"Import segmentation failed: {type(e).__name__}: {e}")
            return False
        self.workspace.segmentation.import_path = str(path)
        self.workspace.segmentation.input_source = "imported"
        self.workspace.set_segmentation_source("imported", seg, provenance=provenance)
        self._ensure_segmentation_label_metadata(np.max(seg, axis=3))
        return self._commit_segmentation_source_change("imported", f"Imported segmentation: {path}")

    def _default_segmentation_sidecar_path(self, source):
        base = os.path.splitext(os.path.basename(self.workspace.paths.flow_path or self.workspace.paths.segmask_path or "segmentation"))[0]
        out_dir = self.workspace.paths.output_dir or os.path.dirname(self.workspace.paths.flow_path or ".") or "."
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, f"{base}_{source}_segmentation.h5")

    def _run_threshold_segmentation(self):
        ws = self.workspace
        seg_state = ws.segmentation
        if ws.segmentation.dirty:
            self.log("Apply or cancel segmentation edits before running threshold segmentation.")
            return False
        try:
            seg, provenance, _scalar, threshold_info = generate_threshold_segmentation(
                mag=ws.mag_raw,
                flow=ws.flow_raw,
                resolution=ws.resolution,
                time_count=ws.time_count(),
                scalar_name=seg_state.threshold_scalar,
                threshold=seg_state.threshold_value,
                keep_largest_cc=seg_state.threshold_keep_largest_cc,
                min_component_volume_mm3=seg_state.threshold_min_component_volume_mm3,
                closing=seg_state.threshold_closing,
                opening=seg_state.threshold_opening,
            )
        except Exception as e:
            self.log(f"Threshold segmentation failed: {type(e).__name__}: {e}")
            return False
        ws.set_segmentation_source("threshold", seg, provenance=provenance)
        self._ensure_segmentation_label_metadata(np.max(seg, axis=3))
        self._commit_segmentation_source_change(
            "threshold",
            f"Threshold segmentation ready: scalar={seg_state.threshold_scalar} {self._format_threshold_summary(threshold_info)}",
        )
        try:
            sidecar = self._default_segmentation_sidecar_path("threshold")
            save_segmentation_file(
                sidecar,
                ws.get_active_segmentation(),
                resolution=ws.resolution,
                origin=ws.origin,
                provenance=ws.get_active_segmentation_provenance(),
            )
            self.log(f"Threshold segmentation saved: {sidecar}")
        except Exception as e:
            self.log(f"Threshold sidecar save failed: {type(e).__name__}: {e}")
        return True

    def _run_auto_segmentation(self):
        ws = self.workspace
        seg_state = ws.segmentation
        if self._autoseg_thread is not None:
            self.log("Auto segmentation is already running.")
            return False
        if ws.segmentation.dirty:
            self.log("Apply or cancel segmentation edits before running auto segmentation.")
            return False
        if str(seg_state.auto_backend or "nnUNet").strip().lower() != "nnunet":
            self.log(f"Auto segmentation backend is not supported: {seg_state.auto_backend}")
            return False
        if not seg_state.auto_model:
            self.log("Auto segmentation requires a nnUNet model folder.")
            return False
        if ws.mag_raw is None or ws.flow_raw is None:
            self.log("Auto segmentation requires loaded mag and flow data.")
            return False

        resolved_device = seg_state.auto_device or "cpu"
        checkpoint_name = seg_state.auto_checkpoint or "checkpoint_final.pth"
        sidecar = self._default_segmentation_sidecar_path("auto")
        self._autoseg_started_at = time.perf_counter()
        self._autoseg_progress_dialog = self._create_progress_dialog("Auto Segmentation", "Preparing auto segmentation...")
        self._autoseg_thread = QtCore.QThread(self)
        self._autoseg_worker = _AutoSegmentationWorker(
            ws.mag_raw,
            ws.flow_raw,
            ws.resolution,
            ws.origin,
            model_folder=seg_state.auto_model,
            backend=seg_state.auto_backend,
            checkpoint_name=checkpoint_name,
            device=resolved_device,
            auto_label_map=seg_state.auto_label_map,
            sidecar_path=sidecar,
        )
        self._autoseg_worker.moveToThread(self._autoseg_thread)
        self._autoseg_thread.started.connect(self._autoseg_worker.run)
        self._autoseg_worker.progress.connect(self._on_autoseg_progress)
        self._autoseg_worker.finished.connect(self._on_autoseg_finished)
        self._autoseg_worker.failed.connect(self._on_autoseg_failed)
        self._autoseg_worker.finished.connect(self._autoseg_thread.quit)
        self._autoseg_worker.failed.connect(self._autoseg_thread.quit)
        self._autoseg_thread.finished.connect(self._cleanup_autoseg_task)
        self.log(
            f"Auto segmentation started: backend={seg_state.auto_backend} model={seg_state.auto_model} "
            f"checkpoint={checkpoint_name} device={resolved_device}"
        )
        self._autoseg_thread.start()
        return True

    def _on_autoseg_progress(self, payload):
        if payload is None:
            return
        if not isinstance(payload, dict):
            message = str(payload).strip()
            if not message:
                return
            if self._autoseg_progress_dialog is not None:
                self._autoseg_progress_dialog.setLabelText(message)
            self.log(f"[AutoSeg] {message}")
            QtWidgets.QApplication.processEvents()
            return
        stage = str(payload.get("stage", "") or "")
        message = str(payload.get("message", "") or stage or "Auto segmentation")
        elapsed_sec = payload.get("elapsed_sec")
        if elapsed_sec is not None:
            message = f"{message}\nElapsed: {float(elapsed_sec):.2f}s"
        dialog = self._autoseg_progress_dialog
        if dialog is not None:
            total = payload.get("total")
            current = payload.get("current")
            if total is not None and int(total) > 0:
                dialog.setRange(0, int(total))
                dialog.setValue(min(int(current or 0), int(total)))
            else:
                dialog.setRange(0, 0)
            dialog.setLabelText(message)
        should_log = stage in {
            "autoseg_model_ready",
            "autoseg_run_inference",
            "autoseg_read_prediction",
            "autoseg_finalize",
        }
        if stage == "autoseg_prepare_inputs":
            detail_total = int(payload.get("detail_total") or 0)
            detail_current = int(payload.get("detail_current") or 0)
            should_log = detail_current in {0, 1, detail_total} if detail_total > 0 else False
        if should_log:
            self.log(f"[AutoSeg] {message.replace(chr(10), ' | ')}")
        QtWidgets.QApplication.processEvents()

    def _on_autoseg_finished(self, result):
        ws = self.workspace
        seg_state = ws.segmentation
        ws.set_segmentation_source("auto", result.seg, provenance=result.provenance)
        self._ensure_segmentation_label_metadata(np.max(result.seg, axis=3))
        self._commit_segmentation_source_change(
            "auto",
            f"Auto segmentation ready: nnUNet model={seg_state.auto_model} | time={float(result.elapsed_sec):.2f}s",
        )
        self.log(f"Auto segmentation saved: {result.sidecar}")
        self._close_progress_dialog(self._autoseg_progress_dialog)
        self._autoseg_progress_dialog = None

    def _on_autoseg_failed(self, error_text):
        text = str(error_text or "").strip()
        if not text:
            text = "Unknown auto segmentation failure."
        first_line = text.splitlines()[0]
        self.log(f"Auto segmentation failed: {first_line}")
        self.log(text)
        self._close_progress_dialog(self._autoseg_progress_dialog)
        self._autoseg_progress_dialog = None

    def _cleanup_autoseg_task(self):
        if self._autoseg_worker is not None:
            self._autoseg_worker.deleteLater()
        if self._autoseg_thread is not None:
            self._autoseg_thread.deleteLater()
        self._autoseg_worker = None
        self._autoseg_thread = None
        self._autoseg_started_at = None

    def _format_threshold_summary(self, threshold_info):
        mode = str(threshold_info.get("mode", "manual_absolute"))
        if mode == "auto":
            return f"threshold=auto resolved={float(threshold_info['min_value']):.6g}"
        if mode == "manual_percent":
            return (
                "threshold="
                f"{float(threshold_info['min_percent']):.6g}%..{float(threshold_info['max_percent']):.6g}% "
                f"values={float(threshold_info['min_value']):.6g}..{float(threshold_info['max_value']):.6g}"
            )
        return f"threshold={float(threshold_info['min_value']):.6g}"

    def _on_configure_segmentation(self):
        if self._autoseg_running_guard("changing segmentation settings"):
            return
        if not self.workspace.data_loaded:
            self.log("Load a case before configuring segmentation.")
            return
        dlg = SegmentationConfigDialog(self.workspace, self)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        values = dlg.values()
        seg = self.workspace.segmentation
        seg.mode = values["mode"]
        seg.input_source = values["input_source"]
        seg.import_path = values["import_path"]
        seg.threshold_scalar = values["threshold_scalar"]
        seg.threshold_value = values["threshold_value"]
        seg.threshold_keep_largest_cc = bool(values["threshold_keep_largest_cc"])
        seg.threshold_closing = bool(values["threshold_closing"])
        seg.threshold_opening = bool(values["threshold_opening"])
        seg.threshold_min_component_volume_mm3 = float(values["threshold_min_component_volume_mm3"])
        seg.auto_backend = values["auto_backend"]
        seg.auto_model = values["auto_model"]
        seg.auto_checkpoint = values["auto_checkpoint"]
        seg.auto_device = values["auto_device"]
        seg.auto_label_map = values["auto_label_map"]
        if seg.mode == "input":
            if seg.input_source == "original":
                self._on_use_original_segmentation()
            elif seg.import_path:
                self._import_segmentation_path(seg.import_path)
            else:
                self.log("Input mode requires either original segmentation or an external file.")
        elif seg.mode == "threshold":
            self._run_threshold_segmentation()
        else:
            self._run_auto_segmentation()
        self._refresh_segmentation_ui()

    def _on_use_original_segmentation(self):
        if self.workspace.segmentation.dirty:
            self.log("Apply or cancel segmentation edits before switching back to original segmentation.")
            return
        if self.workspace.get_segmentation_source("original") is None:
            self.log("Original segmentation is not available for this case.")
            return
        self.workspace.segmentation.mode = "input"
        self.workspace.segmentation.input_source = "original"
        self._commit_segmentation_source_change("original", "Using original segmentation")

    def _on_import_segmentation(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import Segmentation",
            self.workspace.segmentation.import_path or "",
            "Segmentation (*.h5 *.hdf5 *.npy *.npz);;All (*)",
        )
        if not path:
            return
        self._import_segmentation_path(path)

    def _on_save_active_segmentation(self):
        seg = self.workspace.segmentation_display_4d()
        if seg is None:
            self.log("No active segmentation to save.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Active Segmentation",
            self._default_segmentation_sidecar_path(self.workspace.segmentation.active_source or "active"),
            "H5 (*.h5 *.hdf5);;NumPy (*.npy);;NPZ (*.npz)",
        )
        if not path:
            return
        provenance = self.workspace.get_active_segmentation_provenance()
        if self._has_working_segmentation():
            provenance = dict(provenance)
            provenance["saved_from_working_copy"] = True
            provenance["saved_at"] = segmentation_timestamp()
        try:
            save_segmentation_file(path, seg, resolution=self.workspace.resolution, origin=self.workspace.origin, provenance=provenance)
            self.log(f"Saved segmentation: {path}")
        except Exception as e:
            self.log(f"Save segmentation failed: {type(e).__name__}: {e}")

    def _on_reset_active_to_original(self):
        self._cancel_segmentation_edits()
        self._on_use_original_segmentation()

    def _on_segmentation_source_changed(self, _index):
        source = self.segmentation_panel.combo_source.currentData()
        if not source:
            return
        if self.workspace.segmentation.dirty:
            self.log("Apply or cancel segmentation edits before switching source.")
            self._refresh_segmentation_ui()
            return
        if source == self.workspace.segmentation.active_source:
            return
        self._commit_segmentation_source_change(source, f"Switched segmentation source: {source}")

    def _on_segmentation_visibility_changed(self, checked):
        self.workspace.segmentation.visible = bool(checked)
        self._sync_segmentation_scene_object()
        uid = self._find_uid_by_data_key("segmask_raw_surface")
        if uid is not None:
            obj = self.workspace.scene_objects.get(uid)
            if obj is not None:
                self.scene.apply_object_properties(obj)
        self.ortho_viewer.refresh()

    def _on_segmentation_opacity_changed(self, value):
        self.workspace.segmentation.opacity = float(np.clip(value / 100.0, 0.0, 1.0))
        self._sync_segmentation_scene_object()
        uid = self._find_uid_by_data_key("segmask_raw_surface")
        if uid is not None:
            obj = self.workspace.scene_objects.get(uid)
            if obj is not None:
                self.scene.apply_object_properties(obj)
        self.ortho_viewer.refresh()

    def _on_segmentation_label_selected(self):
        row = self.segmentation_panel.table_labels.currentRow()
        if row < 0:
            return
        label_item = self.segmentation_panel.table_labels.item(row, 0)
        if label_item is None:
            return
        label_id = label_item.data(QtCore.Qt.UserRole)
        if label_id is None:
            return
        self.workspace.segmentation.active_label = int(label_id)
        self._refresh_segmentation_ui()
        self.ortho_viewer.refresh()

    def _on_segmentation_label_item_changed(self, item):
        if item is None:
            return
        row = item.row()
        label_item = self.segmentation_panel.table_labels.item(row, 0)
        if label_item is None:
            return
        label_id = int(label_item.data(QtCore.Qt.UserRole))
        key = str(label_id)
        if item.column() == 1:
            self.workspace.segmentation.label_names[key] = item.text().strip() or f"Label {label_id}"
        elif item.column() == 3:
            self.workspace.segmentation.label_colors[key] = item.text().strip() or _default_segmentation_color(label_id)
            self.ortho_viewer.refresh()
        self._refresh_segmentation_ui()

    def _on_active_label_changed(self, value):
        value = max(1, int(value))
        self.workspace.segmentation.active_label = value
        key = str(value)
        self.workspace.segmentation.label_names.setdefault(key, f"Label {value}")
        self.workspace.segmentation.label_colors.setdefault(key, _default_segmentation_color(value))
        self._refresh_segmentation_ui()
        self.ortho_viewer.refresh()

    def _on_active_label_name_changed(self):
        value = max(1, int(self.workspace.segmentation.active_label))
        self.workspace.segmentation.label_names[str(value)] = (
            self.segmentation_panel.edit_active_name.text().strip() or f"Label {value}"
        )
        self._refresh_segmentation_ui()

    def _on_active_label_color_clicked(self):
        value = max(1, int(self.workspace.segmentation.active_label))
        current = self.workspace.segmentation.label_colors.get(str(value), _default_segmentation_color(value))
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(current), self, "Select Label Color")
        if not color.isValid():
            return
        self.workspace.segmentation.label_colors[str(value)] = color.name()
        self._refresh_segmentation_ui()
        self.ortho_viewer.refresh()

    def _set_segmentation_tool(self, tool_name):
        self.workspace.segmentation.tool = str(tool_name)
        self._refresh_segmentation_ui()

    def _on_segmentation_brush_radius_changed(self, value):
        self.workspace.segmentation.brush_radius = max(1, int(value))

    def _on_segmentation_edit_scope_changed(self, checked):
        if self._has_working_segmentation():
            self.log("Apply or cancel current segmentation edits before switching 3D/4D edit mode.")
            self._refresh_segmentation_ui()
            return
        self.workspace.segmentation.edit_all_timepoints = bool(checked)
        self._refresh_segmentation_ui()
        self.ortho_viewer.refresh()

    def _on_segmentation_edit_enabled_changed(self, checked):
        self.workspace.segmentation.editing_enabled = bool(checked)
        self._refresh_segmentation_ui()

    def _paint_segmentation_brush(self, view_name, x, y, z, value):
        labels = self._current_working_labels_3d()
        if labels is None:
            return False
        radius = max(1, int(self.workspace.segmentation.brush_radius))
        changed = False
        if view_name == "axial":
            yy, xx = np.ogrid[:labels.shape[1], :labels.shape[0]]
            mask = (xx - int(x)) ** 2 + (yy - int(y)) ** 2 <= radius ** 2
            plane = labels[:, :, int(z)]
            before = plane.copy()
            plane[mask.T] = int(value)
            changed = not np.array_equal(before, plane)
        elif view_name == "coronal":
            zz, xx = np.ogrid[:labels.shape[2], :labels.shape[0]]
            mask = (xx - int(x)) ** 2 + (zz - int(z)) ** 2 <= radius ** 2
            plane = labels[:, int(y), :]
            before = plane.copy()
            plane[mask.T] = int(value)
            changed = not np.array_equal(before, plane)
        elif view_name == "sagittal":
            zz, yy = np.ogrid[:labels.shape[2], :labels.shape[1]]
            mask = (yy - int(y)) ** 2 + (zz - int(z)) ** 2 <= radius ** 2
            plane = labels[int(x), :, :]
            before = plane.copy()
            plane[mask.T] = int(value)
            changed = not np.array_equal(before, plane)
        return changed

    def _relabel_segmentation_component(self, x, y, z, value):
        labels = self._current_working_labels_3d()
        if labels is None:
            return False
        target = int(labels[int(x), int(y), int(z)])
        if target <= 0 or target == int(value):
            return False
        comp_map, _ = ndi_label(labels == target)
        comp_id = int(comp_map[int(x), int(y), int(z)])
        if comp_id <= 0:
            return False
        before = labels.copy()
        labels[comp_map == comp_id] = int(value)
        return not np.array_equal(before, labels)

    def _handle_segmentation_edit(self, view_name, x, y, z, dragging):
        if self._edit_mode is not None:
            return False
        if not self.workspace.segmentation.editing_enabled:
            return False
        if not self._begin_segmentation_edit_session():
            return False
        tool = self.workspace.segmentation.tool
        label_value = int(self.workspace.segmentation.active_label)
        if not dragging:
            self._push_segmentation_history()
        changed = False
        if tool == "brush":
            changed = self._paint_segmentation_brush(view_name, x, y, z, label_value)
        elif tool == "erase":
            changed = self._paint_segmentation_brush(view_name, x, y, z, 0)
        elif tool == "relabel" and not dragging:
            changed = self._relabel_segmentation_component(x, y, z, label_value)
        if changed:
            self.workspace.segmentation.dirty = True
            self._ensure_segmentation_label_metadata(self._current_working_labels_3d())
            self._refresh_segmentation_preview()
        elif not dragging and self._seg_edit_history:
            self._seg_edit_history.pop()
        return changed

    def _undo_segmentation_edit(self):
        current = self._working_segmentation_array()
        if not self._seg_edit_history or current is None:
            return
        self._seg_edit_future.append(np.asarray(current, dtype=np.int16).copy())
        self._set_working_segmentation_array(self._seg_edit_history.pop())
        self.workspace.segmentation.dirty = True
        self._refresh_segmentation_preview()

    def _redo_segmentation_edit(self):
        current = self._working_segmentation_array()
        if not self._seg_edit_future or current is None:
            return
        self._seg_edit_history.append(np.asarray(current, dtype=np.int16).copy())
        self._set_working_segmentation_array(self._seg_edit_future.pop())
        self.workspace.segmentation.dirty = True
        self._refresh_segmentation_preview()

    def _apply_segmentation_edits(self):
        ws = self.workspace
        working = self._working_segmentation_array()
        if working is None:
            return
        target_source = ws.segmentation.active_source or "imported"
        if target_source == "original":
            target_source = "imported"
        working = np.asarray(working, dtype=np.int16)
        if working.ndim == 4:
            seg4d = working.copy()
            edit_mode = "4d"
        else:
            seg4d = np.repeat(working[..., None], max(1, ws.time_count()), axis=3)
            edit_mode = "3d"
        provenance = ws.get_active_segmentation_provenance()
        provenance = dict(provenance)
        provenance.update(
            {
                "source": "manual_edit",
                "edited_from": ws.segmentation.active_source,
                "created_at": segmentation_timestamp(),
                "edit_mode": edit_mode,
            }
        )
        ws.set_segmentation_source(target_source, seg4d, provenance=provenance)
        ws.clear_working_segmentation()
        self._seg_edit_active = False
        self._reset_segmentation_edit_history()
        self._ensure_segmentation_label_metadata(np.max(seg4d, axis=3))
        self._commit_segmentation_source_change(target_source, f"Applied segmentation edits to {target_source}")

    def _cancel_segmentation_edits(self):
        self.workspace.clear_working_segmentation()
        self._seg_edit_active = False
        self._reset_segmentation_edit_history()
        self._refresh_segmentation_preview()

    def _on_3d_plane_picked(self, uid, plane_idx):
        if self._edit_mode is not None:
            return
        if uid is None or plane_idx is None:
            self.workspace.selected_path_index = -1
            self._selected_plane_index = -1
            self._clear_plane_drag_widgets()
            self.scene.highlight_plane(None)
            self.scene.highlight_path(None)
            self.scene.show_forks_for_path(-1)
            self.ortho_viewer.set_selected_plane(None)
            self._clear_browser_selection()
            self._set_plane_info_text("")
            self._set_path_info_text("")
            return
        self.workspace.selected_path_index = -1
        self._selected_plane_index = int(plane_idx)
        self.scene.highlight_path(None)
        self.scene.show_forks_for_path(-1)
        self.scene.highlight_plane(uid)
        self.ortho_viewer.set_selected_plane(int(plane_idx))
        self._select_browser_item_by_uid(uid)
        self._activate_plane_drag_widgets(int(plane_idx))
        self._set_path_info_text("")
        self._log_selected_plane_metric(int(plane_idx))

    def _on_3d_path_picked(self, uid, path_idx):
        if self._edit_mode is not None:
            return
        if uid is None or path_idx is None:
            self.workspace.selected_path_index = -1
            self._selected_plane_index = -1
            self._clear_plane_drag_widgets()
            self.scene.highlight_plane(None)
            self.scene.highlight_path(None)
            self.scene.show_forks_for_path(-1)
            self.ortho_viewer.set_selected_plane(None)
            self._clear_browser_selection()
            self._set_plane_info_text("")
            self._set_path_info_text("")
            return
        self.workspace.selected_path_index = int(path_idx)
        self._selected_plane_index = -1
        self._clear_plane_drag_widgets()
        self.scene.highlight_plane(None)
        self.scene.highlight_path(uid)
        self.scene.show_forks_for_path(int(path_idx))
        self.ortho_viewer.set_selected_plane(None)
        self._select_browser_item_by_uid(uid)
        self._set_plane_info_text("")
        self._log_selected_path_info(int(path_idx))

    def _find_uid_by_data_key(self, data_key):
        for uid, obj in self.workspace.scene_objects.items():
            if obj.data_key == data_key:
                return uid
        return None

    def _find_uid_by_indexed_data_key(self, prefix, index):
        target = int(index)
        for uid, obj in self.workspace.scene_objects.items():
            parsed = _parse_grouped_index(obj.data_key, prefix)
            if parsed is not None and int(parsed) == target:
                return uid
        return None

    def _group_name_for_plane(self, plane_idx):
        if not (0 <= int(plane_idx) < len(self.workspace.planes)):
            return ""
        return str(getattr(self.workspace.planes[int(plane_idx)], "group_name", "") or "")

    def _plane_data_key(self, prefix, plane_idx):
        group_name = self._group_name_for_plane(plane_idx)
        if group_name:
            return f"{prefix}_{group_name}_{int(plane_idx)}"
        return f"{prefix}_{int(plane_idx)}"

    def _browser_group_name(self, obj):
        group_name = str(getattr(obj, "group_name", "") or "")
        return group_name if group_name else "Global"

    def _plane_widget_distance(self):
        spacing = self._get_spacing_xyz_from_resolution()
        return max(5.0, float(np.mean(spacing)) * 8.0)

    def _clear_plane_drag_widgets(self):
        self._plane_drag_timer.stop()
        self._plane_drag_active = False
        self._plane_drag_index = None
        self._plane_widget_initializing = False
        self._plane_drag_metrics_dirty = False
        if self._edit_mode is not None:
            return
        try:
            if hasattr(self.plotter, "clear_sphere_widgets"):
                self.plotter.clear_sphere_widgets()
        except Exception:
            pass

    def _update_plane_from_drag(self, plane_idx, center=None, normal=None):
        if self._plane_widget_initializing:
            return
        if not (0 <= int(plane_idx) < len(self.workspace.planes)):
            return
        plane = self.workspace.planes[int(plane_idx)]
        tol = max(1e-4, float(np.mean(self._get_spacing_xyz_from_resolution())) * 1e-3)
        changed = False
        if center is not None:
            c = np.asarray(center, dtype=float).reshape(3)
            if np.linalg.norm(c - np.asarray(plane.center, dtype=float).reshape(3)) > tol:
                plane.center = c
                changed = True
        if normal is not None:
            n = np.asarray(normal, dtype=float).reshape(3)
            if np.linalg.norm(n) > 1e-12:
                new_normal = n / np.linalg.norm(n)
                old_normal = np.asarray(plane.normal, dtype=float).reshape(3)
                if min(np.linalg.norm(new_normal - old_normal), np.linalg.norm(new_normal + old_normal)) > 1e-5:
                    plane.normal = new_normal
                    changed = True
        if not changed:
            return
        self.scene.invalidate_cache("plane_")
        uid = self._find_uid_by_indexed_data_key("plane", int(plane_idx))
        if uid is not None:
            obj = self.workspace.scene_objects.get(uid)
            if obj is not None:
                self.scene.readd_object(obj)
                self.scene.highlight_plane(uid)
        else:
            self.scene.sync_from_workspace()
        if int(plane_idx) in self.workspace.active_pathline_plane_indices:
            self.workspace.pathline_cache.pop(int(plane_idx), None)
            self.scene.invalidate_cache("pathline_")
            pathline_uid = self._find_uid_by_indexed_data_key("pathline", int(plane_idx))
            if pathline_uid is not None:
                pathline_obj = self.workspace.scene_objects.get(pathline_uid)
                if pathline_obj is not None:
                    self.scene.readd_object(pathline_obj)
        self._selected_plane_index = int(plane_idx)
        self.ortho_viewer._selected_plane_idx = int(plane_idx)
        self.ortho_viewer.refresh()
        self._plane_drag_index = int(plane_idx)
        self._plane_drag_metrics_dirty = True
        self._plane_drag_timer.start(250)
        try:
            self.plotter.render()
        except Exception:
            pass

    def _persist_plane_outputs(self):
        out_dir = self.pipeline._output_dir(self.workspace)
        metrics = self.workspace.derived.plane_metrics
        qc = self.workspace.derived.plane_qc
        if metrics and len(metrics) == len(self.workspace.planes):
            with open(os.path.join(out_dir, "plane_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
        if qc:
            with open(os.path.join(out_dir, "plane_qc.json"), "w", encoding="utf-8") as f:
                json.dump(qc, f, ensure_ascii=False, indent=2)
        if metrics and len(metrics) == len(self.workspace.planes):
            plane_pixelwise_path = os.path.join(out_dir, "plane_metrics_pixelwise.h5")
            try:
                _, plane_pixelwise = augment_plane_metrics_with_derived(
                    metrics,
                    self.workspace.planes,
                    self.workspace.segmask_binary,
                    self.workspace.resolution,
                    self.workspace.origin,
                    branch_labels_3d=self.workspace.branch_labels,
                    tke_array=self.workspace.derived.tke_array,
                    pressure_gradient_array=self.workspace.derived.pressure_gradient_array,
                    wss_surfaces=self.workspace.derived.wss_surfaces,
                )
                save_plane_pixelwise_h5(plane_pixelwise_path, plane_pixelwise, rr_ms=self.workspace.rr, source_format=self.workspace.input_state.source_format)
                self.workspace.derived.plane_pixelwise_file = plane_pixelwise_path
            except Exception:
                pass
        try:
            self.pipeline._save_planes_json(self.workspace)
        except Exception:
            pass

    def _finalize_plane_drag(self, plane_idx):
        if not (0 <= int(plane_idx) < len(self.workspace.planes)):
            return
        self._plane_drag_timer.stop()
        self._plane_drag_index = int(plane_idx)
        if self.workspace.flow_raw is None or self.workspace.segmask_binary is None:
            self._persist_plane_outputs()
            self._plane_drag_metrics_dirty = False
            return
        if self._plane_drag_metrics_dirty or len(self.workspace.derived.plane_metrics) != len(self.workspace.planes):
            self._recompute_dragged_plane_metrics(persist=True)
        else:
            self._persist_plane_outputs()

    def _activate_plane_drag_widgets(self, plane_idx):
        if self._edit_mode is not None or not (0 <= int(plane_idx) < len(self.workspace.planes)):
            return
        if self._plane_drag_active and int(self._plane_drag_index) == int(plane_idx):
            return
        self._clear_plane_drag_widgets()
        plane = self.workspace.planes[int(plane_idx)]
        center = np.asarray(plane.center, dtype=float).reshape(3)
        normal = np.asarray(plane.normal, dtype=float).reshape(3)
        if np.linalg.norm(normal) <= 1e-12:
            normal = np.array([1.0, 0.0, 0.0], dtype=float)
        normal = normal / np.linalg.norm(normal)
        tip = center + normal * self._plane_widget_distance()
        radius = self._edit_widget_radius()

        def _center_cb(new_center):
            self._update_plane_from_drag(plane_idx, center=new_center)

        def _normal_cb(new_tip):
            c = np.asarray(self.workspace.planes[int(plane_idx)].center, dtype=float).reshape(3)
            tip_now = np.asarray(new_tip, dtype=float).reshape(3)
            self._update_plane_from_drag(plane_idx, normal=(tip_now - c))

        def _end_cb(_widget, _event):
            self._finalize_plane_drag(plane_idx)

        try:
            self._plane_widget_initializing = True
            center_widget = self.plotter.add_sphere_widget(
                callback=_center_cb,
                center=tuple(center.tolist()),
                radius=radius,
                color="cyan",
                interaction_event="always",
            )
            normal_widget = self.plotter.add_sphere_widget(
                callback=_normal_cb,
                center=tuple(tip.tolist()),
                radius=radius,
                color="orange",
                interaction_event="always",
            )
            center_widget.AddObserver(_vtk.vtkCommand.EndInteractionEvent, _end_cb)
            normal_widget.AddObserver(_vtk.vtkCommand.EndInteractionEvent, _end_cb)
            self._plane_drag_active = True
            self._plane_drag_index = int(plane_idx)
        except Exception as e:
            self._plane_drag_active = False
            self._plane_drag_index = None
            self.log(f"Plane drag widget error: {type(e).__name__}: {e}")
        finally:
            self._plane_widget_initializing = False

    def _recompute_dragged_plane_metrics(self, persist=False):
        if self._plane_drag_index is None or not (0 <= int(self._plane_drag_index) < len(self.workspace.planes)):
            return
        if self.workspace.flow_raw is None or self.workspace.segmask_binary is None:
            self.ortho_viewer.refresh()
            if persist:
                self._persist_plane_outputs()
            self._plane_drag_metrics_dirty = False
            return
        plane_idx = int(self._plane_drag_index)
        try:
            if len(self.workspace.derived.plane_metrics) != len(self.workspace.planes):
                self.pipeline._compute_plane_metrics_internal(self.workspace, save=persist)
                self.scene.invalidate_cache("plane_")
                self.scene.sync_from_workspace()
            else:
                paths_for_tangent = (
                    self.workspace.centerline_paths_smooth
                    if len(self.workspace.centerline_paths_smooth) > 0
                    else self.workspace.centerline_paths
                )
                self.pipeline._ensure_derived_metrics(self.workspace, save_pixelwise=False, refresh_scene_objects=False)
                partial_metrics = compute_plane_metrics(
                    self.workspace.flow_raw,
                    self.workspace.segmask_binary,
                    self.workspace.resolution,
                    self.workspace.origin,
                    [self.workspace.planes[plane_idx]],
                    RR=self.workspace.rr,
                    branch_labels_3d=self.workspace.branch_labels,
                    path_info=self.workspace.path_info,
                    forks=self.workspace.forks,
                    paths=paths_for_tangent,
                    return_qc=False,
                )
                if partial_metrics:
                    partial_metrics, _ = augment_plane_metrics_with_derived(
                        partial_metrics,
                        [self.workspace.planes[plane_idx]],
                        self.workspace.segmask_binary,
                        self.workspace.resolution,
                        self.workspace.origin,
                        branch_labels_3d=self.workspace.branch_labels,
                        tke_array=self.workspace.derived.tke_array,
                        pressure_gradient_array=self.workspace.derived.pressure_gradient_array,
                        wss_surfaces=self.workspace.derived.wss_surfaces,
                    )
                    metrics = [dict(m) for m in self.workspace.derived.plane_metrics]
                    metrics[plane_idx] = dict(partial_metrics[0])
                    metrics, qc = apply_internal_consistency_to_metrics(metrics, path_info=self.workspace.path_info, forks=self.workspace.forks)
                    self.workspace.derived.plane_metrics = metrics
                    self.workspace.derived.plane_qc = qc
                    for i, metric in enumerate(metrics):
                        if i < len(self.workspace.planes):
                            self.workspace.planes[i].metrics = dict(metric)
                    if persist:
                        self._persist_plane_outputs()
            if persist:
                self._persist_plane_outputs()
            self._selected_plane_index = plane_idx
            self.ortho_viewer._selected_plane_idx = plane_idx
            self.ortho_viewer.refresh()
            self._log_selected_plane_metric(plane_idx)
            self._plane_drag_metrics_dirty = False
        except Exception as e:
            self.log(f"Plane metric update error: {type(e).__name__}: {e}")
            self.log(traceback.format_exc())

    def _log_selected_plane_metric(self, plane_idx):
        if not (0 <= int(plane_idx) < len(self.workspace.planes)):
            self._set_plane_info_text("")
            return
        plane = self.workspace.planes[int(plane_idx)]
        metric = getattr(plane, "metrics", {}) or {}
        t = int(np.clip(self.workspace.current_t, 0, max(0, self.workspace.time_count() - 1)))
        fr = metric.get("flowrate_mL_s", [])
        ar = metric.get("area_mm2", [])
        mv = metric.get("meanv_cm_s_t", [])
        flow_t = float(fr[t]) if len(fr) > t else 0.0
        area_t = float(ar[t]) if len(ar) > t else 0.0
        meanv_t = float(mv[t]) if len(mv) > t else float(metric.get("meanv_cm_s", 0.0))
        path_dir = metric.get("path_direction", "")
        header = f"Plane {int(plane_idx)} | Path {int(metric.get('path_index', plane.path_index))}"
        if path_dir:
            header += f" {path_dir}"
        text_block = (
            f"{header}\n"
            f"t={t}  Flow Rate={flow_t:.4f} mL/s  Area={area_t:.3f} mm^2  Mean Velocity={meanv_t:.3f} cm/s\n"
            f"Peak Velocity={float(metric.get('peakv_cm_s', 0.0)):.3f} cm/s  Net Flow={float(metric.get('netflow_mL_beat', 0.0)):.4f} mL/beat  IC={float(metric.get('path_ic', 1.0)):.3f}"
        )
        self._set_plane_info_text(text_block)

    def _log_selected_path_info(self, path_idx):
        if not (0 <= int(path_idx) < len(self.workspace.path_info)):
            self._set_path_info_text("")
            return
        info = self.workspace.path_info[int(path_idx)]
        incoming = [int(x) for x in info.get("incoming_path_ids", [])]
        outgoing = [int(x) for x in info.get("outgoing_path_ids", [])]
        forks = []
        for fork in self.workspace.forks:
            if int(path_idx) in fork.get("left", []) or int(path_idx) in fork.get("right", []):
                forks.append(f"node={int(fork.get('node', -1))} L={fork.get('left', [])} R={fork.get('right', [])}")
        fork_txt = " ; ".join(forks) if forks else "none"
        text_block = (
            f"Path {int(path_idx)} | dir={info.get('direction_text', '')}\n"
            f"start_node={int(info.get('start_node', -1))}  end_node={int(info.get('end_node', -1))}\n"
            f"incoming: {incoming if incoming else 'none'}  outgoing: {outgoing if outgoing else 'none'}\n"
            f"forks: {fork_txt}"
        )
        self._set_path_info_text(text_block)

    def _select_browser_item_by_uid(self, uid):
        self.tree_objects.blockSignals(True)
        for i in range(self.tree_objects.topLevelItemCount()):
            top = self.tree_objects.topLevelItem(i)
            for j in range(top.childCount()):
                child = top.child(j)
                if child.data(0, QtCore.Qt.UserRole) == uid:
                    self.tree_objects.setCurrentItem(child)
                    self.tree_objects.blockSignals(False)
                    return
        self.tree_objects.blockSignals(False)

    def _clear_browser_selection(self):
        self.tree_objects.blockSignals(True)
        self.tree_objects.clearSelection()
        self.tree_objects.blockSignals(False)

    def _set_plane_info_text(self, text):
        msg = str(text).strip() if text else "No plane selected."
        self.text_plane_info.setPlainText(msg)

    def _set_path_info_text(self, text):
        msg = str(text).strip() if text else "No path selected."
        self.text_path_info.setPlainText(msg)

    def _autoseg_running_guard(self, action_text):
        if self._autoseg_thread is None:
            return False
        self.log(f"Auto segmentation is running. Wait for it to finish before {action_text}.")
        return True

    def _refresh_selection_info(self):
        if not (0 <= int(self._selected_plane_index) < len(self.workspace.planes)):
            self._selected_plane_index = -1
            self._set_plane_info_text("")
        else:
            self._log_selected_plane_metric(int(self._selected_plane_index))
        path_idx = int(getattr(self.workspace, "selected_path_index", -1))
        if not (0 <= path_idx < len(self.workspace.path_info)):
            self.workspace.selected_path_index = -1
            self._set_path_info_text("")
        else:
            self._log_selected_path_info(path_idx)

    def log(self, text):
        self.console.append(str(text))

    def _create_progress_dialog(self, title, label_text):
        dlg = QtWidgets.QProgressDialog(label_text, "", 0, 0, self)
        dlg.setWindowTitle(str(title))
        dlg.setWindowModality(QtCore.Qt.WindowModal)
        dlg.setCancelButton(None)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setValue(0)
        dlg.show()
        QtWidgets.QApplication.processEvents()
        return dlg

    def _close_progress_dialog(self, dialog):
        if dialog is None:
            return
        try:
            dialog.close()
            dialog.deleteLater()
        except Exception:
            pass

    def _make_progress_handler(self, dialog, log_prefix):
        state = {"last_key": None}

        def _handler(payload):
            if not isinstance(payload, dict):
                message = str(payload or "").strip()
                if message:
                    dialog.setLabelText(message)
                    self.log(f"[{log_prefix}] {message}")
                QtWidgets.QApplication.processEvents()
                return
            stage = str(payload.get("stage", "") or "")
            current = payload.get("current")
            total = payload.get("total")
            message = str(payload.get("message", "") or stage or log_prefix)
            if total is not None and int(total) > 0:
                dialog.setRange(0, int(total))
                dialog.setValue(min(int(current or 0), int(total)))
            else:
                dialog.setRange(0, 0)
            dialog.setLabelText(message)
            should_log = False
            if stage in {"dicom_scan_done", "background_phase_done", "background_phase_start"}:
                should_log = True
            elif stage in {"dicom_scan_file", "dicom_load_file"} and total:
                step = max(1, int(total) // 10)
                should_log = int(current or 0) in {1, int(total)} or int(current or 0) % step == 0
            key = (stage, int(current or 0), int(total or 0), message)
            if should_log and key != state["last_key"]:
                self.log(f"[{log_prefix}] {message}")
                state["last_key"] = key
            QtWidgets.QApplication.processEvents()

        return _handler

    def _float_from_text(self, text, default=0.0):
        try:
            return float(text)
        except Exception:
            return default

    def _optional_float_from_text(self, text, default=None):
        token = str(text or "").strip()
        if token == "" or token.lower() in {"auto", "none"}:
            return default
        try:
            return float(token)
        except Exception:
            return default

    def _int_from_text(self, text, default=0):
        try:
            return int(text)
        except Exception:
            return default

    def _parse_int_list(self, text):
        r = []
        for tok in text.replace(";", ",").split(","):
            tok = tok.strip()
            if tok:
                try:
                    r.append(int(tok))
                except ValueError:
                    pass
        return r

    def _parse_float_list(self, text, default):
        values = []
        for tok in str(text or "").replace(";", ",").split(","):
            tok = tok.strip()
            if tok:
                try:
                    values.append(float(tok))
                except ValueError:
                    pass
        if len(values) == 1:
            values = values * 3
        if len(values) >= 3:
            return np.asarray(values[:3], dtype=float)
        return np.asarray(default, dtype=float).reshape(3)

    def _parse_label_order(self, text, default):
        values = [tok.strip().upper() for tok in str(text or "").replace(";", ",").split(",") if tok.strip()]
        if len(values) >= 3:
            return values[:3]
        return [str(x).upper() for x in default]

    def _sync_params_to_ws(self):
        ws = self.workspace
        ws.loader_params.background_phase_correction.enabled = self.chk_bpc_enabled.isChecked()
        ws.loader_params.background_phase_correction.corr_fit_order = int(self.spin_bpc_fit_order.value())
        ws.loader_params.background_phase_correction.threshold = float(self.spin_bpc_threshold.value())
        ws.loader_params.background_phase_correction.dual_venc_ratio1 = float(self.spin_dual_venc_ratio1.value())
        ws.loader_params.background_phase_correction.dual_venc_ratio2 = float(self.spin_dual_venc_ratio2.value())
        ws.resolution = self._parse_float_list(self.edit_input_resolution.text(), ws.resolution)
        ws.venc = self._parse_float_list(self.edit_input_venc.text(), ws.venc if np.asarray(ws.venc).size >= 3 else [150.0, 150.0, 150.0])
        ws.spatial_order = self._parse_label_order(self.edit_input_spatial_order.text(), ws.spatial_order)
        ws.venc_order = self._parse_label_order(self.edit_input_venc_order.text(), ws.venc_order)
        if isinstance(ws.input_state.metadata, dict):
            ws.input_state.metadata["spatial_order_raw"] = list(ws.spatial_order)
            ws.input_state.metadata["venc_order_raw"] = list(ws.venc_order)
        ws.skeleton_params.remove_small_cc = self.chk_remove_small_cc.isChecked()
        ws.skeleton_params.min_cc_volume_mm3 = self._float_from_text(self.edit_min_cc_volume.text(), 50.0)
        ws.skeleton_params.do_closing = self.chk_closing.isChecked()
        ws.skeleton_params.do_opening = self.chk_opening.isChecked()
        ws.skeleton_params.gaussian_enabled = self.chk_gaussian.isChecked()
        ws.skeleton_params.gaussian_sigma = self._float_from_text(self.edit_gauss_sigma.text(), 0.5)
        ws.plane_gen_params.use_center_plane = self.radio_plane_center.isChecked()
        ws.plane_gen_params.cross_section_distance = self._float_from_text(self.edit_plane_dist.text(), 20.0)
        ws.plane_gen_params.start_distance = self._float_from_text(self.edit_plane_start.text(), 5.0)
        ws.plane_gen_params.end_distance = self._float_from_text(self.edit_plane_end.text(), 0.0)
        ws.plane_gen_params.smoothing_window = self._int_from_text(self.edit_plane_smooth_win.text(), 15)
        ws.plane_gen_params.smoothing_polyorder = self._int_from_text(self.edit_plane_smooth_poly.text(), 3)
        ws.plane_gen_params.inter_time = self._int_from_text(self.edit_plane_inter_time.text(), 10)
        ws.streamline_params.seed_ratio = min(max(self._float_from_text(self.edit_sl_ratio.text(), 0.02), 0.0001), 1.0)
        ws.streamline_params.max_steps = min(max(self._int_from_text(self.edit_sl_maxsteps.text(), 2000), 1), 200000)
        ws.streamline_params.terminal_speed = min(max(self._float_from_text(self.edit_sl_terminal.text(), 0.01), 0.0), 1e6)
        ws.streamline_params.pathline_color = str(self.edit_pathline_color.text().strip() or "deepskyblue")
        ws.derived_params.smoothing_iteration = max(self._int_from_text(self.edit_dm_smoothing.text(), 200), 0)
        ws.derived_params.viscosity = max(self._float_from_text(self.edit_dm_viscosity.text(), 4.0), 0.0)
        inward_distance = self._optional_float_from_text(self.edit_dm_inward.text(), None)
        ws.derived_params.inward_distance = None if inward_distance is None else max(inward_distance, 0.01)
        ws.derived_params.parabolic_fitting = self.chk_dm_parabolic.isChecked()
        ws.derived_params.no_slip_condition = self.chk_dm_noslip.isChecked()
        ws.derived_params.rho = max(self._float_from_text(self.edit_dm_rho.text(), 1060.0), 1.0)
        ws.derived_params.pressure_gradient_smoothing_sigma = max(self._float_from_text(self.edit_dm_pg_smoothing_sigma.text(), 0.0), 0.0)
        ws.derived_params.step_size = max(self._int_from_text(self.edit_dm_stepsize.text(), 5), 1)
        ws.derived_params.tube_radius = max(self._float_from_text(self.edit_dm_tube.text(), 0.1), 0.0)
        ws.derived_params.use_multithread = self.chk_dm_multithread.isChecked()

    def _sync_params_to_ui(self):
        ws = self.workspace
        self.chk_bpc_enabled.setChecked(ws.loader_params.background_phase_correction.enabled)
        self.spin_bpc_fit_order.setValue(int(ws.loader_params.background_phase_correction.corr_fit_order))
        self.spin_bpc_threshold.setValue(float(ws.loader_params.background_phase_correction.threshold))
        self.spin_dual_venc_ratio1.setValue(float(ws.loader_params.background_phase_correction.dual_venc_ratio1))
        self.spin_dual_venc_ratio2.setValue(float(ws.loader_params.background_phase_correction.dual_venc_ratio2))
        self.edit_input_resolution.setText(", ".join(f"{float(x):.6g}" for x in np.asarray(ws.resolution, dtype=float).reshape(-1)[:3]))
        self.edit_input_venc.setText(", ".join(f"{float(x):.6g}" for x in np.asarray(ws.venc, dtype=float).reshape(-1)[:3]))
        self.edit_input_spatial_order.setText(", ".join(str(x) for x in ws.spatial_order[:3]))
        self.edit_input_venc_order.setText(", ".join(str(x) for x in ws.venc_order[:3]))
        self.chk_remove_small_cc.setChecked(ws.skeleton_params.remove_small_cc)
        self.edit_min_cc_volume.setText(str(ws.skeleton_params.min_cc_volume_mm3))
        self.chk_closing.setChecked(ws.skeleton_params.do_closing)
        self.chk_opening.setChecked(ws.skeleton_params.do_opening)
        self.chk_gaussian.setChecked(ws.skeleton_params.gaussian_enabled)
        self.edit_gauss_sigma.setText(str(ws.skeleton_params.gaussian_sigma))
        self.radio_plane_center.setChecked(ws.plane_gen_params.use_center_plane)
        self.radio_plane_by_distance.setChecked(not ws.plane_gen_params.use_center_plane)
        self.edit_plane_dist.setText(str(ws.plane_gen_params.cross_section_distance))
        self.edit_plane_start.setText(str(ws.plane_gen_params.start_distance))
        self.edit_plane_end.setText(str(ws.plane_gen_params.end_distance))
        self.edit_plane_smooth_win.setText(str(ws.plane_gen_params.smoothing_window))
        self.edit_plane_smooth_poly.setText(str(ws.plane_gen_params.smoothing_polyorder))
        self.edit_plane_inter_time.setText(str(ws.plane_gen_params.inter_time))
        self.edit_sl_ratio.setText(str(ws.streamline_params.seed_ratio))
        self.edit_sl_maxsteps.setText(str(ws.streamline_params.max_steps))
        self.edit_sl_terminal.setText(str(ws.streamline_params.terminal_speed))
        self.edit_pathline_color.setText(str(ws.streamline_params.pathline_color))
        self.edit_dm_smoothing.setText(str(ws.derived_params.smoothing_iteration))
        self.edit_dm_viscosity.setText(str(ws.derived_params.viscosity))
        self.edit_dm_inward.setText("auto" if ws.derived_params.inward_distance is None else str(ws.derived_params.inward_distance))
        self.chk_dm_parabolic.setChecked(ws.derived_params.parabolic_fitting)
        self.chk_dm_noslip.setChecked(ws.derived_params.no_slip_condition)
        self.edit_dm_rho.setText(str(ws.derived_params.rho))
        self.edit_dm_pg_smoothing_sigma.setText(str(ws.derived_params.pressure_gradient_smoothing_sigma))
        self.edit_dm_stepsize.setText(str(ws.derived_params.step_size))
        self.edit_dm_tube.setText(str(ws.derived_params.tube_radius))
        self.chk_dm_multithread.setChecked(ws.derived_params.use_multithread)

    def _rebuild_plane_objects(self):
        self._clear_plane_drag_widgets()
        ws = self.workspace
        ws.clear_pathlines()
        ws.pathline_colors = {}
        ws.remove_objects_by_prefix("plane_")
        for group_name, group_state in ws.multilabel_groups.items():
            group_state["planes"] = []
            group_state["plane_index_offset"] = 0
            ws.multilabel_groups[group_name] = group_state
        seen_groups = set()
        for i in range(len(ws.planes)):
            plane = ws.planes[i]
            group_name = str(getattr(plane, "group_name", "") or "")
            if group_name in ws.multilabel_groups:
                state = ws.multilabel_groups[group_name]
                if group_name not in seen_groups:
                    state["plane_index_offset"] = int(i)
                    seen_groups.add(group_name)
                state.setdefault("planes", []).append(plane)
                ws.multilabel_groups[group_name] = state
            ws.add_object(
                name=self._plane_data_key("plane", i),
                kind=ObjectKind.PLANE,
                data_key=self._plane_data_key("plane", i),
                group_name=group_name,
                browser_color=ws.skeleton_params.browser_color_for_group(group_name) if group_name else "",
                visible=True,
                opacity=0.6,
                color=ws.skeleton_params.scene_color_for_group(group_name, "plane") if group_name else "yellow",
                line_width=2,
            )
        self.scene.invalidate_cache("plane_")
        self.scene.invalidate_cache("pathline_")
        self.scene.sync_from_workspace()

    def _refresh_all(self):
        self._sync_segmentation_scene_object()
        self._refresh_browser()
        self._refresh_timeline()
        self._sync_params_to_ui()
        self._refresh_selection_info()
        self._refresh_segmentation_ui()
        self._refresh_scene()

    def _refresh_browser(self):
        self.tree_objects.blockSignals(True)
        self.tree_objects.clear()
        groups = {}
        for obj in self.workspace.scene_objects.values():
            if obj.data_key == "branch_surface":
                continue
            group_name = self._browser_group_name(obj)
            if group_name not in groups:
                top = QtWidgets.QTreeWidgetItem([group_name, "Group", ""])
                top.setFlags(top.flags() | QtCore.Qt.ItemIsUserCheckable)
                top.setCheckState(0, QtCore.Qt.Checked)
                color_name = str(getattr(obj, "browser_color", "") or "")
                if color_name:
                    color = QtGui.QColor(color_name)
                    if color.isValid():
                        top.setForeground(0, QtGui.QBrush(color))
                groups[group_name] = top
                self.tree_objects.addTopLevelItem(top)
            it = QtWidgets.QTreeWidgetItem([obj.name, obj.kind.value, ""])
            it.setData(0, QtCore.Qt.UserRole, obj.uid)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(0, QtCore.Qt.Checked if obj.visible else QtCore.Qt.Unchecked)
            groups[group_name].addChild(it)
        for _group_name, top in groups.items():
            vis_count = sum(1 for i in range(top.childCount()) if top.child(i).checkState(0) == QtCore.Qt.Checked)
            total = top.childCount()
            if vis_count == total:
                top.setCheckState(0, QtCore.Qt.Checked)
            elif vis_count == 0:
                top.setCheckState(0, QtCore.Qt.Unchecked)
            else:
                top.setCheckState(0, QtCore.Qt.PartiallyChecked)
        self.tree_objects.expandAll()
        self.tree_objects.blockSignals(False)

    def _selected_uid(self):
        items = self.tree_objects.selectedItems()
        if not items:
            return None
        return items[0].data(0, QtCore.Qt.UserRole)

    def _on_browser_select(self):
        uid = self._selected_uid()
        if uid:
            obj = self.workspace.scene_objects.get(uid)
            pathline_idx = _parse_pathline_index(obj.data_key) if obj else None
            if pathline_idx is not None:
                plane_uid = self._find_uid_by_indexed_data_key("plane", int(pathline_idx))
                self.workspace.selected_path_index = -1
                self.scene.highlight_path(None)
                self.scene.show_forks_for_path(-1)
                if 0 <= int(pathline_idx) < len(self.workspace.planes):
                    self._selected_plane_index = int(pathline_idx)
                    self.ortho_viewer.set_selected_plane(int(pathline_idx))
                    self._activate_plane_drag_widgets(int(pathline_idx))
                    self._set_path_info_text("")
                    self._log_selected_plane_metric(int(pathline_idx))
                else:
                    self._selected_plane_index = -1
                    self._clear_plane_drag_widgets()
                    self.ortho_viewer.set_selected_plane(None)
                    self._refresh_selection_info()
                self.scene.highlight_plane(plane_uid)
            elif obj and obj.kind == ObjectKind.PLANE:
                pidx = _parse_plane_index(obj.data_key)
                self.workspace.selected_path_index = -1
                self.scene.highlight_path(None)
                self.scene.show_forks_for_path(-1)
                if pidx is not None:
                    self._selected_plane_index = int(pidx)
                    self.ortho_viewer.set_selected_plane(int(pidx))
                    self._activate_plane_drag_widgets(int(pidx))
                    self._set_path_info_text("")
                    self._log_selected_plane_metric(int(pidx))
                self.scene.highlight_plane(uid)
            elif obj and obj.kind == ObjectKind.BRANCH:
                pidx = _parse_path_index(obj.data_key)
                self._selected_plane_index = -1
                self.scene.highlight_plane(None)
                self._clear_plane_drag_widgets()
                self.scene.highlight_path(uid)
                self._set_plane_info_text("")
                self.ortho_viewer.set_selected_plane(None)
                if pidx is not None:
                    self.workspace.selected_path_index = int(pidx)
                    self.scene.show_forks_for_path(int(pidx))
                    self._log_selected_path_info(int(pidx))
                else:
                    self.workspace.selected_path_index = -1
                    self.scene.show_forks_for_path(-1)
            else:
                self._selected_plane_index = -1
                self.workspace.selected_path_index = -1
                self._clear_plane_drag_widgets()
                self.scene.highlight_plane(None)
                self.scene.highlight_path(None)
                self.scene.show_forks_for_path(-1)
                self.ortho_viewer.set_selected_plane(None)
                self._refresh_selection_info()
        else:
            self._selected_plane_index = -1
            self.workspace.selected_path_index = -1
            self._clear_plane_drag_widgets()
            self.scene.highlight_plane(None)
            self.scene.highlight_path(None)
            self.scene.show_forks_for_path(-1)
            self.ortho_viewer.set_selected_plane(None)
            self._refresh_selection_info()

    def _on_tree_item_changed(self, item, column):
        uid = item.data(0, QtCore.Qt.UserRole)
        if uid:
            obj = self.workspace.scene_objects.get(uid)
            if obj:
                obj.visible = item.checkState(0) == QtCore.Qt.Checked
                if obj.data_key == "segmask_raw_surface":
                    self.workspace.segmentation.visible = bool(obj.visible)
                    self._refresh_segmentation_ui()
                self.scene.apply_object_properties(obj)
        else:
            checked = item.checkState(0) != QtCore.Qt.Unchecked
            self.tree_objects.blockSignals(True)
            for i in range(item.childCount()):
                child = item.child(i)
                child.setCheckState(0, QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)
                cuid = child.data(0, QtCore.Qt.UserRole)
                if cuid:
                    obj = self.workspace.scene_objects.get(cuid)
                    if obj:
                        obj.visible = checked
                        if obj.data_key == "segmask_raw_surface":
                            self.workspace.segmentation.visible = bool(checked)
                        self.scene.apply_object_properties(obj)
            self.tree_objects.blockSignals(False)
            self._refresh_segmentation_ui()
        self._refresh_scene()

    def _on_browser_ctx_menu(self, pos):
        item = self.tree_objects.itemAt(pos)
        if item is None:
            return
        uid = item.data(0, QtCore.Qt.UserRole)
        menu = QtWidgets.QMenu(self)
        if uid is None:
            act_show = menu.addAction("Show All")
            act_hide = menu.addAction("Hide All")
            act_del_all = menu.addAction("Delete All")
            action = menu.exec_(self.tree_objects.viewport().mapToGlobal(pos))
            if action == act_show:
                self._set_group_vis(item, True)
            elif action == act_hide:
                self._set_group_vis(item, False)
            elif action == act_del_all:
                self.tree_objects.setCurrentItem(item)
                self._on_delete_object()
        else:
            act_toggle = menu.addAction("Toggle Visibility")
            act_del = menu.addAction("Delete")
            obj = self.workspace.scene_objects.get(uid)
            act_plane_sl = None
            act_pathline_color = None
            pathline_idx = _parse_pathline_index(obj.data_key) if obj else None
            if obj and obj.kind == ObjectKind.PLANE:
                act_plane_sl = menu.addAction("Generate Pathlines")
            if pathline_idx is not None:
                act_pathline_color = menu.addAction("Set Pathline Color")
            action = menu.exec_(self.tree_objects.viewport().mapToGlobal(pos))
            if action == act_toggle:
                if obj:
                    obj.visible = not obj.visible
                    self.scene.apply_object_properties(obj)
                    self._refresh_browser()
            elif action == act_del:
                self._on_delete_object()
            elif act_plane_sl is not None and action == act_plane_sl:
                pidx = _parse_plane_index(obj.data_key)
                if pidx is not None:
                    self._trigger_pathlines(selected_plane_idx=pidx)
            elif act_pathline_color is not None and action == act_pathline_color and pathline_idx is not None:
                self._choose_pathline_color(pathline_idx)

    def _trigger_pathlines(self, selected_plane_idx=None):
        self._sync_params_to_ws()
        default_color = str(self.workspace.streamline_params.pathline_color or "deepskyblue")
        for plane_idx in range(len(self.workspace.planes)):
            self.workspace.pathline_colors.setdefault(int(plane_idx), default_color)
        self.pipeline.preprocess(self.workspace)
        self.scene.trigger_pathlines()
        self._refresh_browser()
        self.scene.invalidate_cache()
        self.scene.sync_from_workspace()
        self._refresh_all()
        if selected_plane_idx is not None and 0 <= int(selected_plane_idx) < len(self.workspace.planes):
            self._selected_plane_index = int(selected_plane_idx)
            self.ortho_viewer.set_selected_plane(int(selected_plane_idx))
            self._activate_plane_drag_widgets(int(selected_plane_idx))
            self.scene.highlight_plane(self._find_uid_by_indexed_data_key("plane", int(selected_plane_idx)))
            self._set_path_info_text("")
            self._log_selected_plane_metric(int(selected_plane_idx))

    def _trigger_plane_streamlines(self, plane_idx=None):
        self._trigger_pathlines(selected_plane_idx=plane_idx)

    def _set_pathline_color(self, pathline_idx, color):
        pathline_idx = int(pathline_idx)
        color_name = str(color or "").strip()
        if not color_name:
            return
        self.workspace.set_pathline_color_for_plane(pathline_idx, color_name)
        uid = self._find_uid_by_indexed_data_key("pathline", pathline_idx)
        if uid is None:
            return
        obj = self.workspace.scene_objects.get(uid)
        if obj is None:
            return
        obj.color = color_name
        self.scene.apply_object_properties(obj)
        self._refresh_browser()
        self._refresh_scene()

    def _choose_pathline_color(self, pathline_idx):
        current = self.workspace.pathline_color_for_plane(pathline_idx)
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(current), self, f"Select Pathline {int(pathline_idx)} Color")
        if not color.isValid():
            return
        self._set_pathline_color(pathline_idx, color.name())

    def _set_group_vis(self, group_item, visible):
        for i in range(group_item.childCount()):
            uid = group_item.child(i).data(0, QtCore.Qt.UserRole)
            if uid:
                obj = self.workspace.scene_objects.get(uid)
                if obj:
                    obj.visible = visible
                    if obj.data_key == "segmask_raw_surface":
                        self.workspace.segmentation.visible = bool(visible)
                    self.scene.apply_object_properties(obj)
        self._refresh_segmentation_ui()
        self._refresh_browser()
        self._refresh_scene()

    def _on_delete_object(self):
        items = self.tree_objects.selectedItems()
        if not items:
            return
        item = items[0]
        uid = item.data(0, QtCore.Qt.UserRole)
        plane_indices_removed = []
        pathline_indices_removed = []
        clear_streamlines = False
        if uid:
            obj = self.workspace.scene_objects.get(uid)
            name = obj.name if obj else uid
            if obj and obj.kind == ObjectKind.PLANE:
                pidx = _parse_plane_index(obj.data_key)
                if pidx is not None:
                    plane_indices_removed.append(pidx)
            if obj and obj.data_key == "streamlines_live":
                clear_streamlines = True
            pidx = _parse_pathline_index(obj.data_key) if obj else None
            if pidx is not None:
                pathline_indices_removed.append(pidx)
            self.scene.remove_object(uid)
            self.log(f"Deleted: {name}")
        else:
            count = item.childCount()
            if count == 0:
                return
            kind_name = item.text(0)
            uids = []
            for i in range(count):
                cuid = item.child(i).data(0, QtCore.Qt.UserRole)
                if cuid:
                    cobj = self.workspace.scene_objects.get(cuid)
                    if cobj and cobj.kind == ObjectKind.PLANE:
                        pidx = _parse_plane_index(cobj.data_key)
                        if pidx is not None:
                            plane_indices_removed.append(pidx)
                    if cobj and cobj.data_key == "streamlines_live":
                        clear_streamlines = True
                    pidx = _parse_pathline_index(cobj.data_key) if cobj else None
                    if pidx is not None:
                        pathline_indices_removed.append(pidx)
                    uids.append(cuid)
            for u in uids:
                self.scene.remove_object(u)
            self.log(f"Deleted section: {kind_name} ({len(uids)} objects)")
        if clear_streamlines:
            self.workspace.clear_streamlines()
        if pathline_indices_removed:
            removed = {int(idx) for idx in pathline_indices_removed}
            for idx in removed:
                self.workspace.pathline_cache.pop(int(idx), None)
                self.workspace.pathline_colors.pop(int(idx), None)
            self.workspace.active_pathline_plane_indices = [int(idx) for idx in self.workspace.active_pathline_plane_indices if int(idx) not in removed]
        if plane_indices_removed:
            self._clear_plane_drag_widgets()
            self.workspace.clear_pathlines()
            for pidx in sorted(plane_indices_removed, reverse=True):
                if 0 <= pidx < len(self.workspace.planes):
                    self.workspace.planes.pop(pidx)
            self._rebuild_plane_objects()
        self._selected_plane_index = -1
        self.workspace.selected_path_index = -1
        self.scene.highlight_plane(None)
        self.scene.highlight_path(None)
        self.scene.show_forks_for_path(-1)
        self._refresh_browser()
        self._refresh_selection_info()

    def _refresh_timeline(self):
        T = max(1, self.workspace.time_count())
        self.slider_t.blockSignals(True)
        self.slider_t.setMaximum(T - 1)
        self.slider_t.setValue(self.workspace.current_t)
        self.slider_t.blockSignals(False)
        self.lab_t.setText(str(self.workspace.current_t))

    def _on_t_changed(self, v):
        self.workspace.current_t = int(v)
        self.lab_t.setText(str(v))
        self.scene.update_time(int(v))
        self.ortho_viewer.refresh()
        self._refresh_segmentation_ui()
        self._refresh_selection_info()

    def _on_prev_frame(self):
        self.workspace.current_t = max(0, self.workspace.current_t - 1)
        self._refresh_timeline()
        self.scene.update_time(self.workspace.current_t)
        self.ortho_viewer.refresh()
        self._refresh_segmentation_ui()
        self._refresh_selection_info()

    def _on_next_frame(self):
        T = self.workspace.time_count()
        self.workspace.current_t = min(T - 1, self.workspace.current_t + 1)
        self._refresh_timeline()
        self.scene.update_time(self.workspace.current_t)
        self.ortho_viewer.refresh()
        self._refresh_segmentation_ui()
        self._refresh_selection_info()

    def _on_play(self):
        self.scene.set_playback_active(True)
        self._play_timer.start(self.spin_interval.value())

    def _on_pause(self):
        self._play_timer.stop()
        self.scene.set_playback_active(False)

    def _on_play_tick(self):
        T = self.workspace.time_count()
        if T <= 1:
            return
        self.workspace.current_t = (self.workspace.current_t + 1) % T
        self._refresh_timeline()
        self.scene.update_time(self.workspace.current_t)
        self.ortho_viewer.refresh()
        self._refresh_segmentation_ui()
        self._refresh_selection_info()

    def _refresh_scene(self):
        try:
            self.scene.render_all()
        except Exception as e:
            self.log(f"VIEW ERROR: {type(e).__name__}: {e}")

    def _load_selected_input_case(self, case, dicom_parameter_overrides=None):
        if self._autoseg_running_guard("loading another case"):
            return
        progress_dialog = None
        try:
            if self._edit_mode is not None:
                self._exit_interactive_edit(False)
            self._clear_plane_drag_widgets()
            self._seg_surface_rebuild_timer.stop()
            self._seg_edit_active = False
            self._reset_segmentation_edit_history()
            self._sync_params_to_ws()
            self.workspace.reset_all()
            apply_config_bundle_to_workspace(self.workspace, self._config_bundle)
            self._sync_params_to_ws()
            resolved = resolve_input_case(case)
            self.workspace.paths.segmask_path = resolved.input_path
            self.workspace.paths.flow_path = resolved.input_path
            if resolved.input_kind == "dicom":
                out_name = resolved.output_name or "dicom_case"
                self.workspace.paths.output_dir = os.path.join(resolved.input_path, f"autoflow_{out_name}")
                self.workspace.loader_params.dicom_parameter_overrides = DicomParameterOverrides.from_dict(
                    dicom_parameter_overrides or {}
                )
                progress_dialog = self._create_progress_dialog("Load DICOM", "Loading DICOM case...")
                progress_handler = self._make_progress_handler(progress_dialog, "DICOM Load")
            else:
                self.workspace.loader_params.dicom_parameter_overrides = DicomParameterOverrides()
                progress_handler = None
            self.pipeline.load_data(
                self.workspace,
                self.log,
                input_source=resolved,
                progress_callback=progress_handler,
            )
            self.scene.workspace = self.workspace
            self.scene.reset_scene()
            self._refresh_all()
            self.ortho_viewer.update_slider_ranges()
            label = resolved.display_name or resolved.input_path
            mode = "enabled" if self.workspace.loader_params.background_phase_correction.enabled else "disabled"
            self.log(f"Loaded input: {label} (BGC {mode})")
        except Exception as e:
            self.log(f"LOAD ERROR: {type(e).__name__}: {e}")
            self.log(traceback.format_exc())
        finally:
            if progress_dialog is not None:
                self._close_progress_dialog(progress_dialog)

    def _prompt_background_phase_choice(self, case):
        resolved = resolve_input_case(case)
        label = resolved.display_name or resolved.input_path
        kind_label = "DICOM case" if resolved.input_kind == "dicom" else "H5 input"
        buttons = (
            QtWidgets.QMessageBox.Yes
            | QtWidgets.QMessageBox.No
            | QtWidgets.QMessageBox.Cancel
        )
        choice = QtWidgets.QMessageBox.question(
            self,
            "Background Phase Correction",
            f"{kind_label}: {label}\n\nEnable background phase correction for this load?",
            buttons,
            QtWidgets.QMessageBox.No,
        )
        if choice == QtWidgets.QMessageBox.Cancel:
            self.log(f"Load cancelled: {label}")
            return None
        enabled = choice == QtWidgets.QMessageBox.Yes
        self.chk_bpc_enabled.setChecked(enabled)
        return resolved

    def _on_open_h5(self):
        if self._autoseg_running_guard("loading another case"):
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open H5", "", "H5 (*.h5 *.hdf5);;All (*)")
        if not path:
            return
        resolved = self._prompt_background_phase_choice(path)
        if resolved is None:
            return
        self._load_selected_input_case(resolved)

    def _on_import_dicom_directory(self):
        if self._autoseg_running_guard("loading another case"):
            return
        root = QtWidgets.QFileDialog.getExistingDirectory(self, "Import DICOM Directory", "")
        if not root:
            return
        progress_dialog = self._create_progress_dialog("Scan DICOM", "Scanning DICOM directory...")
        try:
            cases = scan_dicom_cases(root, progress_callback=self._make_progress_handler(progress_dialog, "DICOM Scan"))
        except Exception as e:
            self._close_progress_dialog(progress_dialog)
            self.log(f"DICOM SCAN ERROR: {type(e).__name__}: {e}")
            self.log(traceback.format_exc())
            return
        progress_dialog.close()
        if not cases:
            self.log(f"No supported DICOM 4D flow cases found in: {root}")
            return
        dialog = DicomImportDialog(cases, self._inspect_dicom_case_preview, self)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        selected = dialog.selected_case()
        if selected is None:
            return
        resolved = self._prompt_background_phase_choice(selected)
        if resolved is None:
            return
        self._load_selected_input_case(resolved, dicom_parameter_overrides=dialog.parameter_overrides())

    def _inspect_dicom_case_preview(self, case):
        progress_dialog = self._create_progress_dialog("Inspect DICOM", "Reading DICOM load parameters...")
        try:
            return inspect_dicom_case(
                case,
                progress_callback=self._make_progress_handler(progress_dialog, "DICOM Preview"),
            )
        finally:
            progress_dialog.close()

    def _on_close_workspace(self):
        if self._autoseg_thread is not None:
            self.log("Auto segmentation is running. Wait for it to finish before clearing the workspace.")
            return
        if self._edit_mode is not None:
            self._exit_interactive_edit(False)
        self._clear_plane_drag_widgets()
        self._seg_surface_rebuild_timer.stop()
        self._seg_edit_active = False
        self._reset_segmentation_edit_history()
        self.workspace.reset_all()
        apply_config_bundle_to_workspace(self.workspace, self._config_bundle)
        self._selected_plane_index = -1
        self.scene.reset_scene()
        self.ortho_viewer.reset_state()
        self._refresh_all()
        self.log("Workspace cleared")

    def _run_single_step(self, step):
        if self._autoseg_running_guard("running pipeline steps"):
            return
        if not self.workspace.data_loaded:
            self.log("No data loaded. Use File > Open H5 or Import DICOM Directory.")
            return
        if self._edit_mode is not None:
            if step == StepId.EDIT_SKELETON and self._edit_mode == "skeleton":
                self._exit_interactive_edit(True)
                return
            if step == StepId.EDIT_GRAPH and self._edit_mode == "graph":
                self._exit_interactive_edit(True)
                return
            self.log("Finish current interactive edit first. Press ESC to force exit.")
            return
        try:
            self._sync_params_to_ws()
            self._clear_plane_drag_widgets()
            self.setEnabled(False)
            QtWidgets.QApplication.processEvents()
            if step == StepId.EDIT_SKELETON:
                self._start_skeleton_interactive_edit()
                return
            if step == StepId.EDIT_GRAPH:
                self._start_graph_interactive_edit()
                return
            if step == StepId.GENERATE_STREAMLINES:
                self.pipeline.preprocess(self.workspace)
                self.scene.trigger_streamlines()
                self._refresh_browser()
                self.scene.invalidate_cache()
                self.scene.sync_from_workspace()
                self._refresh_all()
                return
            if step == StepId.PLANE_STREAMLINES:
                self._on_plane_streamlines_step()
                return
            t0 = time.time()
            result = self.pipeline.run_step(self.workspace, step, self.log)
            elapsed = time.time() - t0
            self.log(f"[{step.label}] {elapsed:.2f}s - {result.message}")
            self.scene.invalidate_cache()
            self.scene.sync_from_workspace()
            self._refresh_all()
            self.ortho_viewer.refresh()
        except Exception as e:
            self.log(f"STEP ERROR: {type(e).__name__}: {e}")
            self.log(traceback.format_exc())
        finally:
            self.setEnabled(True)

    def _run_all_pipeline(self):
        if self._autoseg_running_guard("running the full pipeline"):
            return
        if not self.workspace.data_loaded:
            self.log("No data loaded. Use File > Open H5 or Import DICOM Directory.")
            return
        if self._edit_mode is not None:
            self.log("Finish current interactive edit first.")
            return
        self._sync_params_to_ws()
        self._clear_plane_drag_widgets()
        self.setEnabled(False)
        QtWidgets.QApplication.processEvents()
        all_steps = [
            StepId.GENERATE_SKELETON,
            StepId.GENERATE_GRAPH,
            StepId.GENERATE_PLANES,
            StepId.COMPUTE_PLANE_METRICS,
            StepId.COMPUTE_DERIVED_METRICS,
        ]
        try:
            t_total = time.time()
            for step in all_steps:
                t0 = time.time()
                self.log(f"[Run All] Running {step.label}...")
                QtWidgets.QApplication.processEvents()
                result = self.pipeline.run_step(self.workspace, step, self.log)
                elapsed = time.time() - t0
                self.log(f"[{step.label}] {elapsed:.2f}s - {result.message}")
            self.scene.invalidate_cache()
            self.scene.sync_from_workspace()
            self._refresh_all()
            self.ortho_viewer.refresh()
            total_elapsed = time.time() - t_total
            self.log(f"[Run All] Completed in {total_elapsed:.2f}s")
        except Exception as e:
            self.log(f"RUN ALL ERROR: {type(e).__name__}: {e}")
            self.log(traceback.format_exc())
        finally:
            self.setEnabled(True)

    def _on_plane_streamlines_step(self):
        ws = self.workspace
        if len(ws.planes) == 0:
            self.log("No planes available for Pathlines.")
            return
        uid = self._selected_uid()
        pidx = 0
        if uid:
            obj = ws.scene_objects.get(uid)
            if obj and obj.kind == ObjectKind.PLANE:
                parsed = _parse_plane_index(obj.data_key)
                if parsed is not None:
                    pidx = parsed
        self._trigger_pathlines(selected_plane_idx=pidx)

    def _get_spacing_xyz_from_resolution(self):
        r = np.asarray(self.workspace.resolution, dtype=float).reshape(-1)
        if r.size >= 3:
            return np.array([float(r[0]), float(r[1]), float(r[2])], dtype=float)
        return np.array([1.0, 1.0, 1.0], dtype=float)

    def _edit_widget_radius(self):
        spacing = self._get_spacing_xyz_from_resolution()
        return max(0.1, float(np.mean(spacing)) * 0.6)

    def _graph_polydata(self, points, edges):
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        poly = pv.PolyData(points)
        edges = np.asarray(edges, dtype=int).reshape(-1, 2) if len(edges) else np.empty((0, 2), dtype=int)
        if len(edges) > 0:
            cells = np.empty((len(edges), 3), dtype=np.int64)
            cells[:, 0] = 2
            cells[:, 1] = edges[:, 0]
            cells[:, 2] = edges[:, 1]
            poly.lines = cells.ravel()
        return poly

    def _cleanup_edit_actors(self):
        for actor in [self._edit_actor, self._edit_edge_actor, self._edit_sel_actor, self._edit_sel_edge_actor]:
            if actor is not None:
                try:
                    self.plotter.remove_actor(actor)
                except Exception:
                    try:
                        self.plotter.renderer.RemoveActor(actor)
                    except Exception:
                        pass
        self._edit_actor = None
        self._edit_edge_actor = None
        self._edit_sel_actor = None
        self._edit_sel_edge_actor = None
        self._edit_poly = None
        self._edit_edge_poly = None
        self._edit_sel_poly = None
        self._edit_sel_edge_poly = None

    def _remove_edit_widget(self):
        try:
            if hasattr(self.plotter, "clear_sphere_widgets"):
                self.plotter.clear_sphere_widgets()
        except Exception:
            pass
        try:
            if hasattr(self.plotter, "remove_widget") and self._edit_widget is not None:
                try:
                    self.plotter.remove_widget(self._edit_widget)
                except Exception:
                    pass
        except Exception:
            pass
        self._edit_widget = None

    def _update_edit_labels(self):
        if self._edit_info_label is None or self._edit_status_label is None:
            return
        mode = self._edit_mode or "-"
        npts = 0 if self._edit_points is None else len(self._edit_points)
        nedges = 0 if self._edit_edges is None else len(self._edit_edges)
        sel = "-" if self._edit_selected_idx is None else str(int(self._edit_selected_idx))
        esel = "-" if self._edit_selected_edge_idx is None else str(int(self._edit_selected_edge_idx))
        self._edit_info_label.setText(f"Mode: {mode}    Points: {npts}    Edges: {nedges}    Selected Node: {sel}    Selected Edge: {esel}")
        edge_state = "ON" if self._edit_edge_mode else "OFF"
        src = "-" if self._edit_edge_src_idx is None else str(int(self._edit_edge_src_idx))
        self._edit_status_label.setText(f"Edge Mode: {edge_state}    Edge Src: {src}    Keys: Delete/Backspace delete, E toggle edge mode")
        if self._edit_btn_edge is not None:
            self._edit_btn_edge.setText("Edge Mode: ON" if self._edit_edge_mode else "Edge Mode: OFF")

    def _update_edit_points_actor(self):
        if self._edit_points is None or len(self._edit_points) == 0:
            if self._edit_actor is not None:
                try:
                    self.plotter.remove_actor(self._edit_actor)
                except Exception:
                    pass
            self._edit_actor = None
            self._edit_poly = None
            return
        if self._edit_poly is None:
            self._edit_poly = pv.PolyData(np.asarray(self._edit_points, dtype=float))
        else:
            self._edit_poly.points = np.asarray(self._edit_points, dtype=float)
        color = "red" if self._edit_mode == "skeleton" else "deepskyblue"
        if self._edit_mode == "plane":
            color = "yellow"
        if self._edit_actor is None:
            self._edit_actor = self.plotter.add_mesh(self._edit_poly, color=color, point_size=10, render_points_as_spheres=True, name="interactive_edit_points")
        else:
            try:
                self._edit_actor.GetMapper().SetInputData(self._edit_poly)
            except Exception:
                try:
                    self.plotter.remove_actor(self._edit_actor)
                except Exception:
                    pass
                self._edit_actor = self.plotter.add_mesh(self._edit_poly, color=color, point_size=10, render_points_as_spheres=True, name="interactive_edit_points")

    def _update_edit_edges_actor(self):
        if self._edit_edges is None or len(self._edit_edges) == 0 or self._edit_points is None or len(self._edit_points) == 0:
            if self._edit_edge_actor is not None:
                try:
                    self.plotter.remove_actor(self._edit_edge_actor)
                except Exception:
                    pass
            self._edit_edge_actor = None
            self._edit_edge_poly = None
            return
        poly = self._graph_polydata(self._edit_points, self._edit_edges)
        self._edit_edge_poly = poly
        color = "green" if self._edit_mode == "graph" else "orange"
        if self._edit_mode == "plane":
            color = "yellow"
        if self._edit_edge_actor is None:
            self._edit_edge_actor = self.plotter.add_mesh(poly, color=color, line_width=3, name="interactive_edit_edges")
        else:
            try:
                self._edit_edge_actor.GetMapper().SetInputData(poly)
            except Exception:
                try:
                    self.plotter.remove_actor(self._edit_edge_actor)
                except Exception:
                    pass
                self._edit_edge_actor = self.plotter.add_mesh(poly, color=color, line_width=3, name="interactive_edit_edges")

    def _set_selected_idx(self, idx):
        self._edit_selected_idx = None if idx is None else int(idx)
        if self._edit_points is None or len(self._edit_points) == 0:
            self._edit_selected_idx = None
        elif self._edit_selected_idx is not None and not (0 <= self._edit_selected_idx < len(self._edit_points)):
            self._edit_selected_idx = None
        if self._edit_selected_idx is None:
            if self._edit_sel_actor is not None:
                try:
                    self.plotter.remove_actor(self._edit_sel_actor)
                except Exception:
                    pass
            self._edit_sel_actor = None
            self._edit_sel_poly = None
            self._remove_edit_widget()
            self._update_edit_labels()
            try:
                self.plotter.render()
            except Exception:
                pass
            return
        p = np.asarray(self._edit_points[self._edit_selected_idx], dtype=float).reshape(1, 3)
        if self._edit_sel_poly is None:
            self._edit_sel_poly = pv.PolyData(p)
        else:
            self._edit_sel_poly.points = p
        if self._edit_sel_actor is None:
            self._edit_sel_actor = self.plotter.add_mesh(self._edit_sel_poly, color="yellow", point_size=16, render_points_as_spheres=True, name="interactive_edit_selected")
        else:
            try:
                self._edit_sel_actor.GetMapper().SetInputData(self._edit_sel_poly)
            except Exception:
                try:
                    self.plotter.remove_actor(self._edit_sel_actor)
                except Exception:
                    pass
                self._edit_sel_actor = self.plotter.add_mesh(self._edit_sel_poly, color="yellow", point_size=16, render_points_as_spheres=True, name="interactive_edit_selected")
        self._create_or_move_edit_widget(p[0])
        self._update_edit_labels()
        try:
            self.plotter.render()
        except Exception:
            pass

    def _create_or_move_edit_widget(self, center):
        if self._edit_selected_idx is None or self._edit_points is None or len(self._edit_points) == 0:
            return
        self._remove_edit_widget()
        radius = self._edit_widget_radius()
        def _cb(new_center):
            if self._edit_selected_idx is None or self._edit_points is None or len(self._edit_points) == 0:
                return
            c = np.asarray(new_center, dtype=float).reshape(3)
            self._edit_points[self._edit_selected_idx, :] = c
            if self._edit_poly is not None:
                try:
                    self._edit_poly.points = self._edit_points
                except Exception:
                    pass
            if self._edit_sel_poly is not None:
                try:
                    self._edit_sel_poly.points = np.asarray([c], dtype=float)
                except Exception:
                    pass
            if self._edit_edge_poly is not None:
                try:
                    self._edit_edge_poly.points = self._edit_points
                except Exception:
                    self._update_edit_edges_actor()
            try:
                self.plotter.render()
            except Exception:
                pass
        self._edit_widget = self.plotter.add_sphere_widget(callback=_cb, center=tuple(np.asarray(center, dtype=float).tolist()), radius=radius, color="orange")

    def _enable_interactive_key_events(self, enable):
        try:
            iren = self.plotter.iren.interactor
        except Exception:
            iren = None
        if not enable:
            if iren is not None and self._vtk_keypress_obs_id is not None:
                try:
                    iren.RemoveObserver(self._vtk_keypress_obs_id)
                except Exception:
                    pass
            self._vtk_keypress_obs_id = None
            return
        if iren is None:
            self.log("WARNING: No VTK interactor available; cannot enable key events.")
            return
        def _on_keypress(obj, ev):
            key = ""
            try:
                key = iren.GetKeySym()
            except Exception:
                return
            if key == "Escape":
                self._force_exit_edit()
                return
            if self._edit_mode is None:
                return
            if key in ("Delete", "BackSpace"):
                if self._edit_selected_edge_idx is not None:
                    self._delete_selected_edge()
                else:
                    self._delete_selected_interactive_point()
                return
            if key in ("e", "E"):
                self._toggle_edge_mode()
                return
        if self._vtk_keypress_obs_id is not None:
            try:
                iren.RemoveObserver(self._vtk_keypress_obs_id)
            except Exception:
                pass
            self._vtk_keypress_obs_id = None
        self._vtk_keypress_obs_id = iren.AddObserver("KeyPressEvent", _on_keypress)

    def _enable_interactive_point_picking(self, enable):
        self._edit_pick_enabled = bool(enable)
        try:
            iren = self.plotter.iren.interactor
        except Exception:
            iren = None
        if not enable:
            if iren is not None and self._vtk_left_click_obs_id is not None:
                try:
                    iren.RemoveObserver(self._vtk_left_click_obs_id)
                except Exception:
                    pass
            self._vtk_left_click_obs_id = None
            self._vtk_point_picker = None
            return
        if iren is None:
            self.log("WARNING: No VTK interactor available; cannot enable picking.")
            return
        if self._vtk_point_picker is None:
            self._vtk_point_picker = pv._vtk.vtkPointPicker()
            self._vtk_point_picker.SetTolerance(0.02)
        def _on_left_click(obj, ev):
            if self._edit_mode is None:
                try:
                    iren.GetInteractorStyle().OnLeftButtonDown()
                except Exception:
                    pass
                return
            if self._edit_actor is None or self._edit_points is None or len(self._edit_points) == 0:
                try:
                    iren.GetInteractorStyle().OnLeftButtonDown()
                except Exception:
                    pass
                return
            try:
                x, y = iren.GetEventPosition()
            except Exception:
                x, y = None, None
            if x is None:
                try:
                    iren.GetInteractorStyle().OnLeftButtonDown()
                except Exception:
                    pass
                return
            try:
                self._vtk_point_picker.InitializePickList()
                self._vtk_point_picker.AddPickList(self._edit_actor)
                if self._edit_edge_actor is not None:
                    self._vtk_point_picker.AddPickList(self._edit_edge_actor)
                self._vtk_point_picker.PickFromListOn()
            except Exception:
                pass
            try:
                ren = self.plotter.renderer
                ok = self._vtk_point_picker.Pick(float(x), float(y), 0.0, ren)
            except Exception:
                ok = 0
            if not ok:
                try:
                    iren.GetInteractorStyle().OnLeftButtonDown()
                except Exception:
                    pass
                return
            try:
                p = np.asarray(self._vtk_point_picker.GetPickPosition(), dtype=float).reshape(3)
            except Exception:
                try:
                    iren.GetInteractorStyle().OnLeftButtonDown()
                except Exception:
                    pass
                return
            pts = np.asarray(self._edit_points, dtype=float)
            d2 = np.sum((pts - p.reshape(1, 3)) ** 2, axis=1)
            node_idx = int(np.argmin(d2))
            node_dist = float(np.sqrt(d2[node_idx]))
            edge_idx, edge_dist = self._find_closest_edge(p)
            if self._edit_edge_mode and self._edit_mode == "graph":
                if self._edit_edge_src_idx is None:
                    self._edit_edge_src_idx = node_idx
                    self._set_selected_idx(node_idx)
                    self._edit_selected_edge_idx = None
                    self._clear_edge_selection_actor()
                    self._update_edit_labels()
                else:
                    src = self._edit_edge_src_idx
                    self._edit_edge_src_idx = None
                    if src != node_idx:
                        self._toggle_edge(src, node_idx)
                    self._update_edit_labels()
            elif self._edit_mode == "graph" and edge_idx is not None and edge_dist < node_dist * 0.7:
                self._set_selected_edge_idx(edge_idx)
            else:
                self._edit_selected_edge_idx = None
                self._clear_edge_selection_actor()
                self._set_selected_idx(node_idx)
            try:
                self.plotter.render()
            except Exception:
                pass
        if self._vtk_left_click_obs_id is not None:
            try:
                iren.RemoveObserver(self._vtk_left_click_obs_id)
            except Exception:
                pass
            self._vtk_left_click_obs_id = None
        try:
            self._vtk_left_click_obs_id = iren.AddObserver("LeftButtonPressEvent", _on_left_click)
        except Exception as e:
            self.log(f"WARNING: failed to add VTK observer for picking: {e}")
            self._vtk_left_click_obs_id = None

    def _toggle_edge_mode(self):
        if self._edit_mode != "graph":
            return
        self._edit_edge_mode = not self._edit_edge_mode
        self._edit_edge_src_idx = None
        self._update_edit_labels()

    def _toggle_edge(self, i, j):
        if self._edit_edges is None:
            self._edit_edges = np.empty((0, 2), dtype=int)
        edges = np.asarray(self._edit_edges, dtype=int).reshape(-1, 2)
        found = -1
        for k, (a, b) in enumerate(edges):
            if (int(a) == i and int(b) == j) or (int(a) == j and int(b) == i):
                found = k
                break
        if found >= 0:
            self._edit_edges = np.delete(edges, found, axis=0)
            self.log(f"Removed edge ({i}, {j})")
        else:
            self._edit_edges = np.vstack([edges, [i, j]]) if len(edges) > 0 else np.array([[i, j]], dtype=int)
            self.log(f"Added edge ({i}, {j})")
        self._edit_selected_edge_idx = None
        self._clear_edge_selection_actor()
        self._update_edit_edges_actor()
        self._update_edit_labels()
        try:
            self.plotter.render()
        except Exception:
            pass

    def _clear_edge_selection_actor(self):
        if self._edit_sel_edge_actor is not None:
            try:
                self.plotter.remove_actor(self._edit_sel_edge_actor)
            except Exception:
                pass
        self._edit_sel_edge_actor = None
        self._edit_sel_edge_poly = None

    def _set_selected_edge_idx(self, idx):
        self._edit_selected_edge_idx = None if idx is None else int(idx)
        if self._edit_edges is None or len(self._edit_edges) == 0:
            self._edit_selected_edge_idx = None
        elif self._edit_selected_edge_idx is not None and not (0 <= self._edit_selected_edge_idx < len(self._edit_edges)):
            self._edit_selected_edge_idx = None
        if self._edit_selected_edge_idx is None:
            self._clear_edge_selection_actor()
            self._update_edit_labels()
            return
        edge = self._edit_edges[self._edit_selected_edge_idx]
        pts = self._edit_points[edge]
        poly = pv.PolyData(pts)
        poly.lines = np.array([2, 0, 1], dtype=np.int64)
        self._edit_sel_edge_poly = poly
        if self._edit_sel_edge_actor is None:
            self._edit_sel_edge_actor = self.plotter.add_mesh(poly, color="yellow", line_width=6, name="interactive_edit_selected_edge")
        else:
            try:
                self._edit_sel_edge_actor.GetMapper().SetInputData(poly)
            except Exception:
                try:
                    self.plotter.remove_actor(self._edit_sel_edge_actor)
                except Exception:
                    pass
                self._edit_sel_edge_actor = self.plotter.add_mesh(poly, color="yellow", line_width=6, name="interactive_edit_selected_edge")
        self._set_selected_idx(None)
        self._update_edit_labels()
        try:
            self.plotter.render()
        except Exception:
            pass

    def _delete_selected_edge(self):
        if self._edit_selected_edge_idx is None or self._edit_edges is None or len(self._edit_edges) == 0:
            return
        idx = int(self._edit_selected_edge_idx)
        self.log(f"Deleted edge: {idx} ({self._edit_edges[idx].tolist()})")
        self._edit_edges = np.delete(self._edit_edges, idx, axis=0)
        self._edit_selected_edge_idx = None
        self._clear_edge_selection_actor()
        self._update_edit_edges_actor()
        self._update_edit_labels()
        try:
            self.plotter.render()
        except Exception:
            pass

    def _find_closest_edge(self, pick_pos):
        if self._edit_edges is None or len(self._edit_edges) == 0 or self._edit_points is None:
            return None, float("inf")
        pts = np.asarray(self._edit_points, dtype=float)
        p = np.asarray(pick_pos, dtype=float).reshape(3)
        best_idx = None
        best_dist = float("inf")
        for k, (a, b) in enumerate(self._edit_edges):
            a_pt = pts[int(a)]
            b_pt = pts[int(b)]
            ab = b_pt - a_pt
            ab_len2 = np.dot(ab, ab)
            if ab_len2 < 1e-24:
                d = np.linalg.norm(p - a_pt)
            else:
                t = np.clip(np.dot(p - a_pt, ab) / ab_len2, 0.0, 1.0)
                proj = a_pt + t * ab
                d = np.linalg.norm(p - proj)
            if d < best_dist:
                best_dist = d
                best_idx = k
        return best_idx, best_dist

    def _delete_selected_interactive_point(self):
        if self._edit_selected_idx is None or self._edit_points is None or len(self._edit_points) == 0:
            return
        idx = int(self._edit_selected_idx)
        self._edit_points = np.delete(np.asarray(self._edit_points, dtype=float), idx, axis=0)
        if self._edit_edges is not None and len(self._edit_edges) > 0:
            keep_idx = [i for i in range(len(self._edit_points) + 1) if i != idx]
            remap = {old: new for new, old in enumerate(keep_idx)}
            new_edges = []
            for a, b in np.asarray(self._edit_edges, dtype=int):
                a = int(a)
                b = int(b)
                if a in remap and b in remap:
                    new_edges.append([remap[a], remap[b]])
            self._edit_edges = np.asarray(new_edges, dtype=int).reshape(-1, 2) if new_edges else np.empty((0, 2), dtype=int)
        self._update_edit_points_actor()
        self._update_edit_edges_actor()
        if len(self._edit_points) == 0:
            self._set_selected_idx(None)
        else:
            self._set_selected_idx(min(idx, len(self._edit_points) - 1))
        self._update_edit_labels()
        self.log(f"Deleted point: {idx}")

    def _add_interactive_point(self):
        return

    def _show_interactive_overlay(self):
        self._edit_overlay_dialog = None
        self._edit_info_label = None
        self._edit_status_label = None
        self._edit_btn_edge = None
        mode = self._edit_mode or "-"
        if mode == "skeleton":
            hint = "Edit Skeleton: drag sphere to move | Delete/Backspace to remove | ESC to cancel | click 'Edit Skeleton' again to apply"
        elif mode == "graph":
            hint = "Edit Graph: drag sphere to move node | Delete/Backspace to remove node/edge | E toggle edge mode | Click edge to select | ESC to cancel | click 'Edit Graph' again to apply"
        else:
            hint = f"Edit {mode}: ESC to cancel"
        try:
            self.statusBar().showMessage(hint)
        except Exception:
            pass
        self.log(hint)

    def _close_interactive_overlay(self):
        self._edit_overlay_dialog = None
        self._edit_info_label = None
        self._edit_status_label = None
        self._edit_btn_edge = None
        try:
            self.statusBar().clearMessage()
        except Exception:
            pass

    def _enter_interactive_edit(self, mode, points, edges=None):
        self.scene.invalidate_cache()
        self.scene.sync_from_workspace()
        self._cleanup_edit_actors()
        self._remove_edit_widget()
        self._close_interactive_overlay()
        self._edit_mode = mode
        self._edit_points = np.asarray(points, dtype=float).reshape(-1, 3).copy()
        if edges is None:
            self._edit_edges = np.empty((0, 2), dtype=int)
        else:
            arr = np.asarray(edges, dtype=int)
            self._edit_edges = arr.reshape(-1, 2).copy() if len(arr) else np.empty((0, 2), dtype=int)
        self._edit_original_points = self._edit_points.copy()
        self._edit_original_edges = self._edit_edges.copy()
        self._edit_selected_idx = None
        self._edit_edge_mode = False
        self._edit_edge_src_idx = None
        self._edit_selected_edge_idx = None
        self._update_edit_points_actor()
        self._update_edit_edges_actor()
        self._enable_interactive_key_events(True)
        self._enable_interactive_point_picking(True)
        self._show_interactive_overlay()
        if len(self._edit_points) > 0:
            self._set_selected_idx(0)
        else:
            self._set_selected_idx(None)
        self.log(f"Interactive edit started: {mode}")
        try:
            self.plotter.render()
        except Exception:
            pass

    def _exit_interactive_edit(self, apply_changes):
        mode = self._edit_mode
        if mode is None:
            return
        try:
            self._enable_interactive_key_events(False)
            self._enable_interactive_point_picking(False)
            self._remove_edit_widget()
            if apply_changes:
                active_group = str(self.workspace.group_order[0]) if len(self.workspace.group_order) == 1 else ""
                if mode == "skeleton":
                    ed = SkeletonEditor(self.workspace)
                    ed.replace_points(self._edit_points)
                    if active_group and active_group in self.workspace.multilabel_groups:
                        group_state = dict(self.workspace.multilabel_groups.get(active_group, {}))
                        group_state["skeleton_points"] = np.asarray(self.workspace.skeleton_points, dtype=float).reshape(-1, 3)
                        self.workspace.multilabel_groups[active_group] = group_state
                        self.workspace.remove_objects_by_prefix("skeleton_")
                        self.workspace.add_object(
                            name=f"skeleton_{active_group}",
                            kind=ObjectKind.SKELETON,
                            data_key=f"skeleton_{active_group}",
                            group_name=active_group,
                            browser_color=self.workspace.skeleton_params.browser_color_for_group(active_group),
                            visible=True,
                            opacity=1.0,
                            color=self.workspace.skeleton_params.scene_color_for_group(active_group, "skeleton"),
                            point_size=8,
                        )
                    self.workspace.pipeline.mark_done(StepId.EDIT_SKELETON, skipped=False)
                    self.log(f"Skeleton edited: {len(self.workspace.skeleton_points)} points")
                elif mode == "graph":
                    self.workspace.graph.points = np.asarray(self._edit_points, dtype=float).reshape(-1, 3)
                    self.workspace.graph.edges = np.asarray(self._edit_edges, dtype=int).reshape(-1, 2) if len(self._edit_edges) else np.empty((0, 2), dtype=int)
                    if active_group and active_group in self.workspace.multilabel_groups:
                        group_state = dict(self.workspace.multilabel_groups.get(active_group, {}))
                        if group_state.get("graph") is not None:
                            group_state["graph"].points = np.asarray(self.workspace.graph.points, dtype=float).reshape(-1, 3)
                            group_state["graph"].edges = np.asarray(self.workspace.graph.edges, dtype=int).reshape(-1, 2) if len(self.workspace.graph.edges) else np.empty((0, 2), dtype=int)
                        self.workspace.multilabel_groups[active_group] = group_state
                        self.workspace.remove_objects_by_prefix("graph_")
                        self.workspace.add_object(
                            name=f"graph_{active_group}",
                            kind=ObjectKind.GRAPH,
                            data_key=f"graph_{active_group}",
                            group_name=active_group,
                            browser_color=self.workspace.skeleton_params.browser_color_for_group(active_group),
                            visible=True,
                            opacity=1.0,
                            color=self.workspace.skeleton_params.scene_color_for_group(active_group, "graph"),
                            line_width=2,
                        )
                    self.workspace.pipeline.mark_done(StepId.EDIT_GRAPH, skipped=False)
                    self.log(f"Graph edited: {len(self.workspace.graph.points)} nodes, {len(self.workspace.graph.edges)} edges")
            else:
                self.log(f"Interactive edit cancelled: {mode}")
        finally:
            self._cleanup_edit_actors()
            self._close_interactive_overlay()
            self._edit_mode = None
            self._edit_points = None
            self._edit_edges = None
            self._edit_selected_idx = None
            self._edit_edge_mode = False
            self._edit_edge_src_idx = None
            self._edit_selected_edge_idx = None
            self._edit_original_points = None
            self._edit_original_edges = None
            self.scene.invalidate_cache()
            self.scene.sync_from_workspace()
            self._refresh_all()
            try:
                self.plotter.render()
            except Exception:
                pass

    def _force_exit_edit(self):
        if self._edit_mode is None:
            return
        self.log("ESC: force exit interactive edit")
        try:
            self._exit_interactive_edit(False)
        except Exception as e:
            self.log(f"Force exit cleanup error: {type(e).__name__}: {e}")
        finally:
            self._edit_mode = None
            self._edit_points = None
            self._edit_edges = None
            self._edit_selected_idx = None
            self._edit_edge_mode = False
            self._edit_edge_src_idx = None
            self._edit_selected_edge_idx = None
            self._edit_original_points = None
            self._edit_original_edges = None
            self._edit_overlay_dialog = None
            self._edit_info_label = None
            self._edit_status_label = None
            self._edit_btn_edge = None
            try:
                self.statusBar().clearMessage()
            except Exception:
                pass
            try:
                self._cleanup_edit_actors()
            except Exception:
                pass
            try:
                self._remove_edit_widget()
            except Exception:
                pass
            try:
                self._enable_interactive_key_events(False)
            except Exception:
                pass
            try:
                self._enable_interactive_point_picking(False)
            except Exception:
                pass
            try:
                self.setEnabled(True)
            except Exception:
                pass
            try:
                self.plotter.render()
            except Exception:
                pass

    def _start_skeleton_interactive_edit(self):
        if len(self.workspace.group_order) > 1:
            self.log("Edit Skeleton is only available when exactly one segmentation group is active.")
            return
        if self.workspace.skeleton_points is None or len(self.workspace.skeleton_points) == 0:
            self.log("Edit Skeleton: no skeleton points.")
            return
        self._enter_interactive_edit("skeleton", self.workspace.skeleton_points, edges=None)

    def _start_graph_interactive_edit(self):
        if len(self.workspace.group_order) > 1:
            self.log("Edit Graph is only available when exactly one segmentation group is active.")
            return
        if self.workspace.graph is None or len(self.workspace.graph.points) == 0:
            self.log("Edit Graph: no graph data.")
            return
        self._enter_interactive_edit("graph", self.workspace.graph.points, self.workspace.graph.edges)

    def closeEvent(self, event):
        if self._autoseg_thread is not None:
            self.log("Auto segmentation is running. Wait for it to finish before closing the GUI.")
            event.ignore()
            return
        try:
            if self._edit_mode is not None:
                self._exit_interactive_edit(False)
            self._clear_plane_drag_widgets()
        except Exception:
            pass
        event.accept()


def main(config_dir=None):
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(config_dir=config_dir)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
