import hashlib
import json
from collections import deque
from dataclasses import dataclass
import importlib.util
import os
import re
import signal
import sys
import time
import traceback
from functools import partial
from pathlib import Path

os.environ["QT_API"] = "pyside6"

import numpy as np
import pyvista as pv
from PySide6 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from pyvista import _vtk
from scipy.ndimage import label as ndi_label

if os.environ.get("AUTOFLOW_SSH_RENDERING") == "1":
    from .remote_plotter import RemotePlotter
else:
    from pyvistaqt import QtInteractor

from ..algorithms import (
    inspect_dicom_case,
    resolve_input_case,
    scan_dicom_cases,
    load_segmentation_file,
    generate_threshold_segmentation,
    generate_nnunet_auto_segmentation,
    save_segmentation_file,
    save_nifti_volume,
    save_segmentation_to_source_h5,
    segmentation_timestamp,
    _plot_plane_flowrate_axes,
    _plot_pwv_axes,
    _project_point_to_path,
    filter_paths_by_segmentation,
    generate_pathlines_from_plane_at_t,
)
from ..algorithms.data import discover_h5_input_cases, inspect_h5_input_case
from ..algorithms.segmentation import (
    compute_reference_scalar,
    resolve_nnunet_model_folder,
    resolve_nnunet_4d_model_folder,
)
from ..core.models import DicomParameterOverrides, ObjectKind, PlaneData, PwvParams, StepId, Workspace
from ..core.pipeline import PipelineEngine
from ..config import apply_config_bundle_to_workspace, bundle_to_autoflow_kwargs, load_config_bundle
from ..plane_io import load_plane_position_payload, project_planes_to_workspace, save_plane_positions
from ..quality import build_quality_report, save_quality_report
from ..algorithms import compute_plane_metrics, apply_internal_consistency_to_metrics, compute_plane_metrics_multithread, augment_plane_metrics_with_derived, save_plane_pixelwise_h5
from ..algorithms.streamlines import _plane_seeds, create_pathline_temporal_source
from .editors import PlaneEditor
from .dicom_confirm import DicomImportDialog, H5CaseSelectDialog
from .ortho_viewer import OrthoViewer
from .segmentation import SegmentationConfigDialog, SegmentationDock, SOURCE_LABELS
from .theme import apply_application_theme, configure_high_dpi, standard_icon
from ..rendering import (
    render_plane_rotation_video,
    render_pressure_gradient_video,
    render_relative_pressure_video,
    render_streamlines_video,
    render_tke_video,
    render_wss_video,
)
from .viewer import SceneController


# Additional item data roles used by the Browser for non-object path nodes.
# SceneObject UIDs continue to live in Qt.UserRole so existing selection and
# deletion paths remain compatible with the rest of the UI.
_BROWSER_NODE_ROLE = int(QtCore.Qt.UserRole) + 1
_BROWSER_PATH_INDEX_ROLE = int(QtCore.Qt.UserRole) + 2
_BROWSER_GROUP_ROLE = int(QtCore.Qt.UserRole) + 3


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


def _plane_in_plane_axes(normal):
    normal = np.asarray(normal, dtype=float).reshape(3)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        normal = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        normal = normal / norm
    reference = np.eye(3, dtype=float)[int(np.argmin(np.abs(normal)))]
    axis_u = np.cross(reference, normal)
    axis_u = axis_u / (np.linalg.norm(axis_u) + 1e-12)
    axis_v = np.cross(normal, axis_u)
    axis_v = axis_v / (np.linalg.norm(axis_v) + 1e-12)
    return axis_u, axis_v


def _default_segmentation_color(label_id):
    palette = [
        "#ff6b6b", "#4dabf7", "#51cf66", "#ffd43b", "#f783ac",
        "#74c0fc", "#63e6be", "#ffa94d", "#b197fc", "#a9e34b",
    ]
    idx = (max(1, int(label_id)) - 1) % len(palette)
    return palette[idx]


_ANALYSIS_MODE_ITEMS = [
    ("PWV", "pwv"),
    ("Plane Curve", "plane_curve"),
    ("Centerline Pressure", "centerline_pressure"),
    ("Internal Consistency", "internal_consistency"),
]

_BACKGROUND_PHASE_METHOD_ITEMS = [
    ("MSAC", "msac"),
    ("WRLS + ARTO", "wrls_arto"),
]


_PLANE_CURVE_SERIES_OPTIONS = [
    ("flowrate_mL_s", "Flowrate (mL/s)"),
    ("flowrate_forward_mL_s", "Forward Flowrate (mL/s)"),
    ("flowrate_reverse_mL_s", "Reverse Flowrate (mL/s)"),
    ("flowrate_signed_mL_s", "Signed Flowrate (mL/s)"),
    ("area_mm2", "Area (mm^2)"),
    ("meanv_cm_s_t", "Mean Velocity (cm/s)"),
    ("meanv_forward_cm_s_t", "Forward Mean Velocity (cm/s)"),
    ("meanv_reverse_cm_s_t", "Reverse Mean Velocity (cm/s)"),
    ("meanv_signed_cm_s_t", "Signed Mean Velocity (cm/s)"),
    ("tke_mean_J_m3_t", "TKE Mean (J/m^3)"),
    ("tke_peak_J_m3_t", "TKE Peak (J/m^3)"),
    ("tke_p95_J_m3_t", "TKE P95 (J/m^3)"),
    ("pressure_gradient_mag_mean_Pa_m_t", "Pressure Gradient Mean (Pa/m)"),
    ("pressure_gradient_mag_peak_Pa_m_t", "Pressure Gradient Peak (Pa/m)"),
    ("pressure_gradient_mag_p95_Pa_m_t", "Pressure Gradient P95 (Pa/m)"),
    ("pressure_gradient_normal_mean_Pa_m_t", "Pressure Gradient Normal Mean (Pa/m)"),
    ("pressure_gradient_normal_peak_Pa_m_t", "Pressure Gradient Normal Peak (Pa/m)"),
    ("pressure_gradient_normal_p95_Pa_m_t", "Pressure Gradient Normal P95 (Pa/m)"),
    ("relative_pressure_mean_Pa_t", "Relative Pressure Mean (Pa)"),
    ("relative_pressure_peak_Pa_t", "Relative Pressure Peak (Pa)"),
    ("relative_pressure_p95_Pa_t", "Relative Pressure P95 (Pa)"),
    ("wss_wall_mean_Pa_t", "WSS Mean (Pa)"),
    ("wss_wall_peak_Pa_t", "WSS Peak (Pa)"),
    ("wss_wall_p95_Pa_t", "WSS P95 (Pa)"),
]


_RUNTIME_RENDER_METRICS = [
    ("wss", "WSS"),
    ("tke", "TKE"),
    ("pressure_gradient", "Pressure Gradient"),
    ("relative_pressure", "Relative Pressure"),
    ("streamline", "Streamlines"),
]

_WORKFLOW_STAGES = [
    ("Input & QC", "input"),
    ("Segmentation", "segmentation"),
    ("Phase Unwrapping", "phase_unwrap"),
    ("Centerline & Planes", "centerline"),
    ("Hemodynamics", "hemodynamics"),
    ("Review & Export", "review"),
]

_WORKFLOW_STEPS = {
    "input": set(),
    "segmentation": set(),
    "phase_unwrap": {StepId.UNWRAP_PHASE},
    "centerline": {
        StepId.GENERATE_SKELETON,
        StepId.EDIT_SKELETON,
        StepId.GENERATE_GRAPH,
        StepId.EDIT_GRAPH,
        StepId.GENERATE_PLANES,
    },
    "hemodynamics": {
        StepId.COMPUTE_PLANE_METRICS,
        StepId.COMPUTE_PWV,
        StepId.COMPUTE_DERIVED_METRICS,
        StepId.GENERATE_STREAMLINES,
        StepId.PLANE_STREAMLINES,
    },
    "review": set(),
}

_WORKFLOW_RUN_ALL_STEPS = {
    "centerline": (
        StepId.GENERATE_SKELETON,
        StepId.GENERATE_GRAPH,
        StepId.GENERATE_PLANES,
    ),
    "hemodynamics": (
        StepId.COMPUTE_PLANE_METRICS,
        StepId.COMPUTE_DERIVED_METRICS,
    ),
    "phase_unwrap": (StepId.UNWRAP_PHASE,),
}


def _workflow_run_all_steps(stage_key):
    return list(_WORKFLOW_RUN_ALL_STEPS.get(str(stage_key), ()))


@dataclass
class _AutoSegmentationResult:
    seg: np.ndarray
    provenance: dict
    cache_path: str
    elapsed_sec: float


@dataclass
class _VideoExportOptions:
    out_dir: str
    export_plane: bool
    export_wss: bool
    export_tke: bool
    export_pg: bool
    export_streamlines: bool


class _AutoSegmentationWorker(QtCore.QObject):
    progress = QtCore.Signal(dict)
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, mag, flow, resolution, origin, *, model_folder, backend, checkpoint_name, folds, device, auto_label_map, cache_path, artifact_prefix, source_spatial_order=None, source_group=None):
        super().__init__()
        self._mag = mag
        self._flow = flow
        self._resolution = resolution
        self._origin = origin
        self._model_folder = str(model_folder)
        self._backend = str(backend)
        self._checkpoint_name = str(checkpoint_name)
        self._folds = str(folds or "single")
        self._device = str(device)
        self._auto_label_map = auto_label_map
        self._cache_path = str(cache_path or "")
        self._artifact_prefix = str(artifact_prefix)
        self._source_spatial_order = tuple(str(x).upper() for x in (source_spatial_order or []))
        self._source_group = str(source_group).strip("/") if source_group else None

    def _emit_progress(self, payload):
        data = dict(payload or {})
        if int(data.get("total") or 0) == 5:
            data["total"] = 7
        self.progress.emit(data)

    @QtCore.Slot()
    def run(self):
        t_start = time.perf_counter()
        try:
            mag = np.asarray(self._mag, dtype=np.float32)
            flow = np.asarray(self._flow, dtype=np.float32)
            resolution = np.asarray(self._resolution, dtype=np.float32).copy()
            origin = np.asarray(self._origin, dtype=np.float32).copy()
            seg, provenance = generate_nnunet_auto_segmentation(
                mag=mag,
                flow=flow,
                resolution=resolution,
                origin=origin,
                model_folder=self._model_folder,
                backend=self._backend,
                checkpoint_name=self._checkpoint_name,
                folds=self._folds,
                device=self._device,
                auto_label_map=self._auto_label_map,
                artifact_prefix=self._artifact_prefix,
                progress_callback=self._emit_progress,
            )
            seg = np.asarray(seg, dtype=np.int16)
            foreground_labels = [int(value) for value in np.unique(seg) if int(value) > 0]
            if not foreground_labels:
                raise RuntimeError(
                    "nnUNet prediction contains only background (label 0); no segmentation can be displayed"
                )
            self.progress.emit({
                "stage": "autoseg_validate_prediction",
                "message": f"Validated prediction labels: {foreground_labels}",
                "current": 5,
                "total": 7,
                "elapsed_sec": float(time.perf_counter() - t_start),
            })
            cache_path = ""
            if self._cache_path.lower().endswith((".h5", ".hdf5")):
                self.progress.emit({
                    "stage": "autoseg_save_cache",
                    "message": "Saving segmentation to the selected source H5 group...",
                    "current": 6,
                    "total": 7,
                    "elapsed_sec": float(time.perf_counter() - t_start),
                })
                save_segmentation_to_source_h5(
                    self._cache_path,
                    seg,
                    resolution=resolution,
                    origin=origin,
                    provenance=provenance,
                    source_spatial_order=self._source_spatial_order,
                    source_group=self._source_group,
                )
                cache_path = self._cache_path
            else:
                self.progress.emit({
                    "stage": "autoseg_finalize_result",
                    "message": "Finalizing segmentation result...",
                    "current": 6,
                    "total": 7,
                    "elapsed_sec": float(time.perf_counter() - t_start),
                })
            elapsed = time.perf_counter() - t_start
            self.progress.emit({
                "stage": "autoseg_complete",
                "message": "Automatic segmentation completed",
                "current": 7,
                "total": 7,
                "elapsed_sec": float(elapsed),
            })
            self.finished.emit(_AutoSegmentationResult(seg=seg, provenance=provenance, cache_path=cache_path, elapsed_sec=float(elapsed)))
        except Exception:
            self.failed.emit(traceback.format_exc())


class _PipelineTaskWorker(QtCore.QObject):
    progress = QtCore.Signal(dict)
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, engine, workspace, steps):
        super().__init__()
        self._engine = engine
        self._workspace = workspace
        self._steps = list(steps)

    @QtCore.Slot()
    def run(self):
        started = time.perf_counter()
        results = []
        try:
            for index, step in enumerate(self._steps, start=1):
                self.progress.emit({
                    "stage": "start",
                    "current": index - 1,
                    "total": len(self._steps),
                    "step": step,
                    "message": f"Running {step.label}...",
                })
                step_started = time.perf_counter()
                result = self._engine.run_step(self._workspace, step, lambda _message: None)
                elapsed = time.perf_counter() - step_started
                results.append((step, result, float(elapsed)))
                self.progress.emit({
                    "stage": "done",
                    "current": index,
                    "total": len(self._steps),
                    "step": step,
                    "message": f"{step.label} completed",
                    "elapsed_sec": float(elapsed),
                    "result_message": str(result.message),
                })
            self.finished.emit({
                "results": results,
                "elapsed_sec": float(time.perf_counter() - started),
            })
        except Exception:
            self.failed.emit(traceback.format_exc())


class _PathlineTaskWorker(QtCore.QObject):
    progress = QtCore.Signal(dict)
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, flow, mask_4d, branch_labels, planes, plane_indices, *,
                 time_index, spacing, origin, rr, params):
        super().__init__()
        self._flow = flow
        self._mask_4d = mask_4d
        self._branch_labels = branch_labels
        self._planes = list(planes)
        self._plane_indices = [int(index) for index in plane_indices]
        self._time_index = int(time_index)
        self._spacing = np.asarray(spacing, dtype=float).copy()
        self._origin = np.asarray(origin, dtype=float).copy()
        self._rr = float(rr)
        self._params = dict(params or {})

    @QtCore.Slot()
    def run(self):
        started = time.perf_counter()
        try:
            results = {}
            seed_results = {}
            total = len(self._plane_indices)
            if self._mask_4d.ndim == 4:
                mask_t = self._mask_4d[..., min(max(0, self._time_index), self._mask_4d.shape[3] - 1)]
            else:
                mask_t = self._mask_4d
            temporal_cache_mb = (
                self._params["pathline_temporal_cache_mb"] if total > 1 else 0.0
            )
            temporal_source = create_pathline_temporal_source(
                self._flow,
                self._mask_4d,
                self._spacing,
                self._origin,
                self._time_index,
                self._rr,
                temporal_cache_mb=temporal_cache_mb,
            )
            frame_steps = min(
                len(temporal_source._times),
                max(1, int(self._params["pathline_max_steps"])),
            )
            for current, plane_index in enumerate(self._plane_indices, start=1):
                seeds = _plane_seeds(
                    mask_t,
                    self._planes[plane_index],
                    self._spacing,
                    self._origin,
                    seed_ratio=self._params["pathline_seed_ratio"],
                    min_seeds=self._params["pathline_min_seeds"],
                    max_seeds=self._params["pathline_max_seeds"],
                    rng_seed=self._params["pathline_rng_seed"],
                    branch_labels_3d=self._branch_labels,
                    t=self._time_index,
                    seed_mode=self._params["pathline_seed_mode"],
                )
                seed_results[int(plane_index)] = seeds
                plane_offset = (current - 1) * frame_steps
                def _step_progress(step, step_total, *, _plane=int(plane_index), _offset=plane_offset, _current=current):
                    self.progress.emit({
                        "current": min(_offset + int(step), _current * frame_steps),
                        "total": total * frame_steps,
                        "message": f"Integrating pathlines for plane {_plane} ({_current}/{total})",
                        "elapsed_sec": float(time.perf_counter() - started),
                    })
                mesh = generate_pathlines_from_plane_at_t(
                    self._flow,
                    self._time_index,
                    self._planes[plane_index],
                    self._spacing,
                    self._origin,
                    mask_4d=self._mask_4d,
                    mask_3d=mask_t,
                    max_steps=self._params["pathline_max_steps"],
                    terminal_speed=self._params["pathline_terminal_speed"],
                    seed_ratio=self._params["pathline_seed_ratio"],
                    min_seeds=self._params["pathline_min_seeds"],
                    rng_seed=self._params["pathline_rng_seed"],
                    rr=self._rr,
                    branch_labels_3d=self._branch_labels,
                    max_seeds=self._params["pathline_max_seeds"],
                    seeds=seeds,
                    progress_callback=_step_progress,
                    seed_mode=self._params["pathline_seed_mode"],
                    temporal_source=temporal_source,
                )
                results[int(plane_index)] = {int(self._time_index): mesh}
                self.progress.emit({
                    "current": current * frame_steps,
                    "total": total * frame_steps,
                    "message": f"Generated pathlines for plane {int(plane_index)} ({current}/{total})",
                    "elapsed_sec": float(time.perf_counter() - started),
                })
            self.finished.emit({
                "plane_indices": self._plane_indices,
                "meshes": results,
                "seeds": seed_results,
                "temporal_cache_all_phases": bool(temporal_source.cache_all_phases),
                "elapsed_sec": float(time.perf_counter() - started),
            })
        except Exception:
            self.failed.emit(traceback.format_exc())


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, config_dir=None):
        super().__init__()
        self.setWindowTitle("AutoFlow")
        self.resize(1800, 980)
        self.setMinimumSize(1440, 800)
        self._config_dir = config_dir
        self._config_bundle = load_config_bundle(config_dir)
        self.workspace = Workspace()
        apply_config_bundle_to_workspace(self.workspace, self._config_bundle)
        self.pipeline = PipelineEngine()
        self.scene = None
        self._play_timer = QtCore.QTimer(self)
        self._play_timer.setSingleShot(True)
        self._play_timer.timeout.connect(self._on_play_tick)
        self._playback_frame_times_ms = deque(maxlen=24)
        self._playback_render_times_ms = deque(maxlen=24)
        self._playback_last_tick_started_at = None
        self._timeline_scrub_timer = QtCore.QTimer(self)
        self._timeline_scrub_timer.setSingleShot(True)
        self._timeline_scrub_timer.setInterval(30)
        self._timeline_scrub_timer.timeout.connect(self._apply_pending_timeline_value)
        self._pending_timeline_value = None
        self._edit_mode = None
        self._edit_group_name = ""
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
        self._edit_panel = None
        self._edit_btn_save = None
        self._edit_btn_cancel = None
        self._edit_visibility_snapshot = None
        self._edit_segmentation_visible_snapshot = None
        self._edit_original_points = None
        self._edit_original_edges = None
        self._plane_drag_active = False
        self._plane_drag_index = None
        self._plane_widget_initializing = False
        self._plane_drag_metrics_dirty = False
        self._plane_edit_enabled = False
        self._plane_center_widget = None
        self._plane_axis_u_widget = None
        self._plane_axis_v_widget = None
        self._plane_axis_u = None
        self._plane_axis_v = None
        self._selected_plane_index = -1
        self._seg_edit_active = False
        self._seg_edit_history = []
        self._seg_edit_future = []
        self._seg_surface_rebuild_timer = QtCore.QTimer(self)
        self._seg_surface_rebuild_timer.setSingleShot(True)
        self._seg_surface_rebuild_timer.timeout.connect(self._rebuild_segmentation_surface)
        self._loader_progress_dialog = None
        self._autoseg_thread = None
        self._autoseg_worker = None
        self._autoseg_progress_dialog = None
        self._autoseg_started_at = None
        self._pipeline_thread = None
        self._pipeline_worker = None
        self._pipeline_progress_dialog = None
        self._pipeline_task_label = ""
        self._run_all_pathlines_after_pipeline = False
        self._pathline_thread = None
        self._pathline_worker = None
        self._pathline_progress_dialog = None
        self._pathline_selected_plane_idx = None
        self._labeler_process = None
        self._labeler_segmentation_path = ""
        self._labeler_segmentation_mtime_ns = None
        self._labeler_process_output = {"stdout": "", "stderr": ""}
        self._last_quality_report = None
        self._last_plane_import_report = None
        self._active_input_case = None
        self._active_dicom_parameter_overrides = {}
        self._input_signature = None
        self._build_ui()
        self._bind_scene()
        self._esc_shortcut = QtGui.QShortcut(QtCore.Qt.Key_Escape, self)
        self._esc_shortcut.setContext(QtCore.Qt.ApplicationShortcut)
        self._esc_shortcut.activated.connect(self._force_exit_edit)
        QtCore.QTimer.singleShot(0, self._setup_focus_behavior)
        self.ortho_viewer.timeStepRequested.connect(self._on_ortho_time_step)
        self._refresh_all()
        self.statusBar().showMessage("Ready")

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
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        workflow_row = QtWidgets.QHBoxLayout()
        workflow_row.setSpacing(8)
        self.workflow_tabs = QtWidgets.QTabBar()
        self.workflow_tabs.setDocumentMode(True)
        self.workflow_tabs.setExpanding(True)
        self.workflow_tabs.setDrawBase(False)
        for label, _stage_key in _WORKFLOW_STAGES:
            self.workflow_tabs.addTab(label.replace("&", "&&"))
        workflow_row.addWidget(self.workflow_tabs, 1)
        self.workflow_status = QtWidgets.QLabel("Not ready")
        self.workflow_status.setAlignment(QtCore.Qt.AlignCenter)
        self.workflow_status.setMinimumWidth(105)
        workflow_row.addWidget(self.workflow_status, 0)
        self.btn_show_advanced = QtWidgets.QToolButton()
        self.btn_show_advanced.setText("Advanced")
        self.btn_show_advanced.setCheckable(True)
        self.btn_show_advanced.setChecked(False)
        self.btn_show_advanced.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.btn_show_advanced.setArrowType(QtCore.Qt.RightArrow)
        workflow_row.addWidget(self.btn_show_advanced, 0)
        root.addLayout(workflow_row)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)
        root.addWidget(splitter, 1)
        left = QtWidgets.QWidget()
        left.setMinimumWidth(280)
        left_lay = QtWidgets.QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(6)
        self._build_browser(left_lay)
        mid = QtWidgets.QWidget()
        mid.setMinimumWidth(520)
        mid_lay = QtWidgets.QVBoxLayout(mid)
        mid_lay.setContentsMargins(0, 0, 0, 0)
        mid_lay.setSpacing(6)
        if os.environ.get("AUTOFLOW_SSH_RENDERING") == "1":
            self.plotter = RemotePlotter(self)
        else:
            self.plotter = QtInteractor(self)
        self.plotter.setFocusPolicy(QtCore.Qt.ClickFocus)
        mid_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.mid_splitter = mid_splitter
        mid_splitter.setChildrenCollapsible(False)
        mid_splitter.setHandleWidth(6)
        plotter_panel = QtWidgets.QWidget()
        plotter_layout = QtWidgets.QVBoxLayout(plotter_panel)
        plotter_layout.setContentsMargins(0, 0, 0, 0)
        plotter_layout.setSpacing(3)
        plotter_layout.addWidget(self.plotter, 1)
        range_bar = QtWidgets.QHBoxLayout()
        range_bar.addWidget(QtWidgets.QLabel("Window/Level:"))
        range_bar.addWidget(QtWidgets.QLabel("Window"))
        self.slider_render_window = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_render_window.setRange(1, 1000)
        self.slider_render_window.setEnabled(False)
        self.slider_render_window.setToolTip("Adjust window width")
        range_bar.addWidget(self.slider_render_window, 1)
        self.label_render_window = QtWidgets.QLabel("—")
        self.label_render_window.setMinimumWidth(72)
        range_bar.addWidget(self.label_render_window)
        range_bar.addWidget(QtWidgets.QLabel("Level"))
        self.slider_render_level = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_render_level.setRange(0, 1000)
        self.slider_render_level.setEnabled(False)
        self.slider_render_level.setToolTip("Adjust window level")
        range_bar.addWidget(self.slider_render_level, 1)
        self.label_render_level = QtWidgets.QLabel("—")
        self.label_render_level.setMinimumWidth(72)
        range_bar.addWidget(self.label_render_level)
        self.btn_reset_render_range = QtWidgets.QPushButton("Auto")
        self.btn_reset_render_range.setToolTip("Restore the automatic range for the selected scalar object")
        self.btn_reset_render_range.setEnabled(False)
        self.btn_reset_render_range.clicked.connect(self._reset_selected_render_range)
        range_bar.addWidget(self.btn_reset_render_range)
        range_bar.addStretch(1)
        self.slider_render_window.valueChanged.connect(self._on_render_range_changed)
        self.slider_render_level.valueChanged.connect(self._on_render_range_changed)
        plotter_layout.addLayout(range_bar)
        mid_splitter.addWidget(plotter_panel)
        step_and_params = QtWidgets.QWidget()
        sp_lay = QtWidgets.QVBoxLayout(step_and_params)
        sp_lay.setContentsMargins(0, 0, 0, 0)
        sp_lay.setSpacing(4)
        self._build_step_buttons(sp_lay)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        params_widget = QtWidgets.QWidget()
        self.params_layout = QtWidgets.QVBoxLayout(params_widget)
        self.params_layout.setContentsMargins(4, 4, 4, 4)
        self._workflow_param_widgets = {stage_key: [] for _label, stage_key in _WORKFLOW_STAGES}

        def _capture_params(stage_key, builder, advanced=False):
            start = self.params_layout.count()
            builder()
            for index in range(start, self.params_layout.count()):
                widget = self.params_layout.itemAt(index).widget()
                if widget is not None:
                    self._workflow_param_widgets[stage_key].append((widget, bool(advanced)))

        _capture_params("input", self._build_preprocess_params)
        _capture_params("phase_unwrap", self._build_phase_unwrap_params)
        _capture_params("centerline", self._build_skeleton_params, advanced=True)
        _capture_params("centerline", self._build_plane_params)
        _capture_params("hemodynamics", self._build_pwv_params)
        _capture_params("hemodynamics", self._build_streamline_params)
        _capture_params("hemodynamics", self._build_derived_params, advanced=True)
        _capture_params("hemodynamics", self._build_vortex_params)
        _capture_params("review", self._build_review_params)
        self.params_layout.addStretch()
        scroll.setWidget(params_widget)
        sp_lay.addWidget(scroll, 1)
        mid_splitter.addWidget(step_and_params)
        mid_splitter.setStretchFactor(0, 4)
        mid_splitter.setStretchFactor(1, 2)
        mid_splitter.setSizes([480, 220])
        mid_lay.addWidget(mid_splitter, 1)
        right = QtWidgets.QWidget()
        right.setMinimumWidth(360)
        right_lay = QtWidgets.QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        self.ortho_viewer = OrthoViewer(self.workspace, self)
        right_lay.addWidget(self.ortho_viewer)
        splitter.addWidget(left)
        splitter.addWidget(mid)
        splitter.addWidget(right)
        splitter.setSizes([360, 790, 450])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)
        splitter.setStretchFactor(2, 2)
        timeline_w = QtWidgets.QWidget()
        tl_lay = QtWidgets.QVBoxLayout(timeline_w)
        tl_lay.setContentsMargins(0, 0, 0, 0)
        self._build_timeline(tl_lay)
        root.addWidget(timeline_w, 0)

        self.bottom_tabs = QtWidgets.QTabWidget()
        self.bottom_tabs.setDocumentMode(True)
        self.bottom_tabs.setMinimumHeight(125)
        self.bottom_tabs.setMaximumHeight(175)
        sel_w = QtWidgets.QWidget()
        sel_lay = QtWidgets.QVBoxLayout(sel_w)
        sel_lay.setContentsMargins(6, 6, 6, 6)
        self._build_selection_info(sel_lay)
        log_w = QtWidgets.QWidget()
        log_lay = QtWidgets.QVBoxLayout(log_w)
        log_lay.setContentsMargins(6, 6, 6, 6)
        self._build_log(log_lay)
        self.bottom_tabs.addTab(sel_w, "Selection")
        self.bottom_tabs.addTab(log_w, "Log")
        root.addWidget(self.bottom_tabs, 0)
        self._build_segmentation_dock()
        self._build_pwv_dock()
        self.workflow_tabs.currentChanged.connect(self._on_workflow_stage_changed)
        self.btn_show_advanced.toggled.connect(self._on_advanced_toggled)
        self._on_workflow_stage_changed(self.workflow_tabs.currentIndex())

    def _build_browser(self, parent):
        grp = QtWidgets.QGroupBox("Browser")
        lay = QtWidgets.QVBoxLayout(grp)
        self.tree_objects = QtWidgets.QTreeWidget()
        self.tree_objects.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.tree_objects.setHeaderLabels(["Name", "Kind"])
        header = self.tree_objects.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Fixed)
        self.tree_objects.setColumnWidth(1, 58)
        self.tree_objects.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.tree_objects.itemSelectionChanged.connect(self._on_browser_select)
        self.tree_objects.itemChanged.connect(self._on_tree_item_changed)
        self.tree_objects.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.tree_objects.customContextMenuRequested.connect(self._on_browser_ctx_menu)
        lay.addWidget(self.tree_objects)
        opacity_row = QtWidgets.QHBoxLayout()
        self.label_browser_opacity = QtWidgets.QLabel("Opacity: —")
        self.slider_browser_opacity = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_browser_opacity.setRange(0, 100)
        self.slider_browser_opacity.setValue(100)
        self.slider_browser_opacity.setEnabled(False)
        self.slider_browser_opacity.setToolTip("Adjust opacity for the selected Browser object(s)")
        self.slider_browser_opacity.valueChanged.connect(self._on_browser_opacity_changed)
        opacity_row.addWidget(self.label_browser_opacity)
        opacity_row.addWidget(self.slider_browser_opacity, 1)
        lay.addLayout(opacity_row)
        row = QtWidgets.QHBoxLayout()
        self.btn_delete_obj = QtWidgets.QPushButton("Delete Selected")
        self.btn_delete_obj.setIcon(standard_icon(self, "SP_TrashIcon"))
        self.btn_delete_obj.setProperty("role", "danger")
        self.btn_delete_obj.clicked.connect(self._on_delete_object)
        row.addWidget(self.btn_delete_obj)
        row.addStretch()
        lay.addLayout(row)
        parent.addWidget(grp, 3)

    def _build_step_buttons(self, parent):
        grp = QtWidgets.QGroupBox("Steps")
        self.steps_group = grp
        gl = QtWidgets.QGridLayout(grp)
        self.step_buttons = {}
        all_steps = [
            StepId.UNWRAP_PHASE,
            StepId.GENERATE_SKELETON, StepId.EDIT_SKELETON,
            StepId.GENERATE_GRAPH, StepId.EDIT_GRAPH,
            StepId.GENERATE_PLANES, StepId.COMPUTE_PLANE_METRICS,
            StepId.COMPUTE_PWV, StepId.COMPUTE_DERIVED_METRICS,
            StepId.GENERATE_STREAMLINES, StepId.PLANE_STREAMLINES,
        ]
        step_tooltips = {
            StepId.UNWRAP_PHASE: "Run the selected optional phase-unwrapping method. Dual-VENC inputs are skipped.",
            StepId.EDIT_SKELETON: "Correct skeleton points interactively. Saving clears graph, paths, planes, and downstream metrics.",
            StepId.EDIT_GRAPH: "Move graph nodes, toggle edges, or delete graph elements. Saving clears paths, planes, and downstream metrics.",
            StepId.COMPUTE_PWV: "Compute PWV for the configured PWV groups.",
            StepId.PLANE_STREAMLINES: "Generate time-resolved pathlines for the selected plane or planes in a background task.",
        }
        for index, step in enumerate(all_steps):
            label = step.label
            if label == "WSS / TKE / Relative Pressure":
                label = "WSS / TKE / Pressure / Vortex"
            button = QtWidgets.QPushButton(label)
            button.setToolTip(step_tooltips.get(step, step.label))
            button.clicked.connect(partial(self._run_single_step, step))
            self.step_buttons[step] = button
            row, column = divmod(index, 3)
            gl.addWidget(button, row, column)
        self.btn_import_planes_centerline = QtWidgets.QPushButton("Import Plane Coordinates...")
        self.btn_import_planes_centerline.setIcon(standard_icon(self, "SP_DialogOpenButton"))
        self.btn_import_planes_centerline.clicked.connect(self._on_import_plane_coordinates)
        import_row = (len(all_steps) + 2) // 3
        gl.addWidget(self.btn_import_planes_centerline, import_row, 0, 1, 3)
        self.btn_run_all = QtWidgets.QPushButton("Run All")
        self.btn_run_all.setIcon(standard_icon(self, "SP_MediaPlay"))
        self.btn_run_all.setProperty("role", "primary")
        self.btn_run_all.setMinimumHeight(34)
        self.btn_run_all.clicked.connect(self._run_all_pipeline)
        run_all_row = import_row + 1
        gl.addWidget(self.btn_run_all, run_all_row, 0, 1, 3)
        parent.addWidget(grp, 0)

        edit_grp = QtWidgets.QGroupBox("Interactive Edit")
        self._edit_panel = edit_grp
        edit_lay = QtWidgets.QVBoxLayout(edit_grp)
        edit_lay.setContentsMargins(8, 8, 8, 8)
        self._edit_info_label = QtWidgets.QLabel()
        self._edit_info_label.setWordWrap(True)
        self._edit_info_label.setStyleSheet("font-weight: 600;")
        edit_lay.addWidget(self._edit_info_label)
        self._edit_status_label = QtWidgets.QLabel()
        self._edit_status_label.setWordWrap(True)
        edit_lay.addWidget(self._edit_status_label)
        self._edit_hint_label = QtWidgets.QLabel()
        self._edit_hint_label.setWordWrap(True)
        self._edit_hint_label.setStyleSheet("color: #555; padding: 3px 0;")
        edit_lay.addWidget(self._edit_hint_label)
        edit_buttons = QtWidgets.QHBoxLayout()
        self._edit_btn_edge = QtWidgets.QPushButton("Edge Mode: OFF")
        self._edit_btn_edge.setToolTip("Graph mode: click two nodes to add or remove an edge")
        self._edit_btn_edge.clicked.connect(self._toggle_edge_mode)
        edit_buttons.addWidget(self._edit_btn_edge)
        edit_buttons.addStretch()
        self._edit_btn_save = QtWidgets.QPushButton("Save Changes")
        self._edit_btn_save.setIcon(standard_icon(self, "SP_DialogSaveButton"))
        self._edit_btn_save.setProperty("role", "primary")
        self._edit_btn_save.clicked.connect(lambda: self._exit_interactive_edit(True))
        edit_buttons.addWidget(self._edit_btn_save)
        self._edit_btn_cancel = QtWidgets.QPushButton("Cancel")
        self._edit_btn_cancel.setIcon(standard_icon(self, "SP_DialogCancelButton"))
        self._edit_btn_cancel.clicked.connect(lambda: self._exit_interactive_edit(False))
        edit_buttons.addWidget(self._edit_btn_cancel)
        edit_lay.addLayout(edit_buttons)
        edit_grp.setVisible(False)
        parent.addWidget(edit_grp, 0)

    def _build_preprocess_params(self):
        grp = QtWidgets.QGroupBox("Input / Background Correction")
        fl = QtWidgets.QFormLayout(grp)
        self.chk_bpc_enabled = QtWidgets.QCheckBox()
        self.chk_bpc_enabled.setChecked(False)
        self.combo_bpc_method = QtWidgets.QComboBox()
        for label, method in _BACKGROUND_PHASE_METHOD_ITEMS:
            self.combo_bpc_method.addItem(label, method)
        self.combo_bpc_method.currentIndexChanged.connect(self._on_bpc_method_changed)
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
        fl.addRow("Correction Method", self.combo_bpc_method)
        fl.addRow("Corr Fit Order", self.spin_bpc_fit_order)
        fl.addRow("MSAC Threshold", self.spin_bpc_threshold)
        fl.addRow("Dual-VENC Ratio1", self.spin_dual_venc_ratio1)
        fl.addRow("Dual-VENC Ratio2", self.spin_dual_venc_ratio2)
        fl.addRow("Current Resolution XYZ", self.edit_input_resolution)
        fl.addRow("Current VENC XYZ", self.edit_input_venc)
        fl.addRow("Current Spatial Order", self.edit_input_spatial_order)
        fl.addRow("Current VENC Order", self.edit_input_venc_order)
        self.btn_reload_input = QtWidgets.QPushButton("Reload Input with Current Parameters")
        self.btn_reload_input.setIcon(standard_icon(self, "SP_BrowserReload"))
        self.btn_reload_input.clicked.connect(self._reload_input_case)
        fl.addRow("", self.btn_reload_input)
        self._on_bpc_method_changed()
        self.params_layout.addWidget(grp)

    def _build_skeleton_params(self):
        grp = QtWidgets.QGroupBox("Generate Skeleton Parameters")
        fl = QtWidgets.QFormLayout(grp)
        self.chk_remove_small_cc = QtWidgets.QCheckBox()
        self.chk_remove_small_cc.setChecked(False)
        self.chk_separate_special_label_contacts = QtWidgets.QCheckBox()
        self.chk_separate_special_label_contacts.setChecked(True)
        self.chk_separate_special_label_contacts.setToolTip(
            "Separate contacts between RBCT, CCA, and LBCT before skeletonization."
        )
        self.combo_cc_filter_mode = QtWidgets.QComboBox()
        self.combo_cc_filter_mode.addItems(["hybrid", "absolute", "relative", "largest"])
        self.edit_min_cc_volume = QtWidgets.QLineEdit("50.0")
        self.edit_cc_rel_min_ratio = QtWidgets.QLineEdit("0.01")
        self.chk_closing = QtWidgets.QCheckBox()
        self.chk_closing.setChecked(True)
        self.chk_opening = QtWidgets.QCheckBox()
        self.chk_gaussian = QtWidgets.QCheckBox()
        self.chk_gaussian.setChecked(True)
        self.edit_gauss_sigma = QtWidgets.QLineEdit("0.5")
        fl.addRow("Remove Small CC", self.chk_remove_small_cc)
        fl.addRow("Separate Special Label Contacts", self.chk_separate_special_label_contacts)
        fl.addRow("CC Filter Mode", self.combo_cc_filter_mode)
        fl.addRow(u"Min Volume (mm\u00b3)", self.edit_min_cc_volume)
        fl.addRow("Relative Min Ratio", self.edit_cc_rel_min_ratio)
        fl.addRow("Closing", self.chk_closing)
        fl.addRow("Opening", self.chk_opening)
        fl.addRow("Gaussian", self.chk_gaussian)
        fl.addRow(u"Gauss \u03c3", self.edit_gauss_sigma)
        self.params_layout.addWidget(grp)

    def _build_phase_unwrap_params(self):
        grp = QtWidgets.QGroupBox("Phase Unwrapping (Optional)")
        fl = QtWidgets.QFormLayout(grp)
        self.combo_phase_unwrap_method = QtWidgets.QComboBox()
        for label, method in [("Graph-cut 3D", "gc3D"), ("Laplacian 4D", "lap4D"), ("NPRS", "nprs")]:
            self.combo_phase_unwrap_method.addItem(label, method)
        self.combo_phase_unwrap_method.currentIndexChanged.connect(self._on_phase_unwrap_method_changed)
        self.combo_phase_unwrap_mask = QtWidgets.QComboBox()
        self.combo_phase_unwrap_mask.addItem("Active segmentation", "segmentation")
        self.combo_phase_unwrap_mask.addItem("All voxels", "all")
        self.combo_phase_unwrap_device = QtWidgets.QComboBox()
        for label, value in [("Auto (GPU if available)", "auto"), ("CPU", "cpu"), ("CUDA", "cuda")]:
            self.combo_phase_unwrap_device.addItem(label, value)

        # Keep method-specific controls in separate pages.  This avoids
        # presenting parameters that have no effect on the selected backend.
        self.phase_unwrap_method_stack = QtWidgets.QStackedWidget()
        gc_page = QtWidgets.QLabel("Graph-cut 3D has no additional parameters.")
        gc_page.setWordWrap(True)
        gc_page.setStyleSheet("color:#555;")
        self.phase_unwrap_method_stack.addWidget(gc_page)

        lap_page = QtWidgets.QWidget()
        lap_form = QtWidgets.QFormLayout(lap_page)
        self.edit_phase_unwrap_ts = QtWidgets.QLineEdit("2.0")
        lap_form.setContentsMargins(0, 0, 0, 0)
        lap_form.addRow("Temporal weight", self.edit_phase_unwrap_ts)
        self.phase_unwrap_method_stack.addWidget(lap_page)

        nprs_page = QtWidgets.QWidget()
        nprs_form = QtWidgets.QFormLayout(nprs_page)
        self.edit_phase_unwrap_nprs_up = QtWidgets.QLineEdit("2")
        self.chk_phase_unwrap_nprs_pi = QtWidgets.QCheckBox()
        self.chk_phase_unwrap_nprs_pi.setChecked(True)
        self.chk_phase_unwrap_nprs_crop = QtWidgets.QCheckBox()
        self.chk_phase_unwrap_nprs_crop.setChecked(True)
        nprs_form.setContentsMargins(0, 0, 0, 0)
        nprs_form.addRow("Upsampling factor", self.edit_phase_unwrap_nprs_up)
        nprs_form.addRow("π unwrap", self.chk_phase_unwrap_nprs_pi)
        nprs_form.addRow("Auto crop", self.chk_phase_unwrap_nprs_crop)
        self.phase_unwrap_method_stack.addWidget(nprs_page)

        fl.addRow("Method", self.combo_phase_unwrap_method)
        fl.addRow("Mask", self.combo_phase_unwrap_mask)
        fl.addRow("Device", self.combo_phase_unwrap_device)
        fl.addRow("Method parameters", self.phase_unwrap_method_stack)
        hint = QtWidgets.QLabel("Selecting a method opts in; nothing runs until you click Unwrap Phase. Dual‑VENC inputs are skipped automatically. Wrap Count and Estimated Wrap Locations appear in the Browser after a run.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#555;")
        fl.addRow("", hint)
        self.btn_revert_phase_unwrap = QtWidgets.QPushButton("Revert to Loaded Flow")
        self.btn_revert_phase_unwrap.clicked.connect(self._revert_phase_unwrap)
        fl.addRow("", self.btn_revert_phase_unwrap)
        self.params_layout.addWidget(grp)
        self._on_phase_unwrap_method_changed()

    def _build_plane_params(self):
        grp = QtWidgets.QGroupBox("Generate Planes Parameters")
        fl = QtWidgets.QFormLayout(grp)
        self.combo_plane_mode = QtWidgets.QComboBox()
        self.combo_plane_mode.addItem("Uniform", "uniform")
        self.combo_plane_mode.addItem("Fixed Step", "fixed_step")
        # Legacy values are retained so older workspaces can still be viewed.
        self.combo_plane_mode.addItem("Evenly Spaced (legacy)", "count")
        self.combo_plane_mode.addItem("Fixed Spacing (legacy)", "distance")
        self.combo_plane_mode.addItem("From Junction", "anchored_offset")
        self.combo_plane_mode.currentIndexChanged.connect(self._sync_plane_mode_ui)
        self.edit_plane_count = QtWidgets.QLineEdit("3")
        self.edit_plane_dist = QtWidgets.QLineEdit("5.0")
        self.edit_plane_start = QtWidgets.QLineEdit("0.0")
        self.edit_plane_end = QtWidgets.QLineEdit("0.0")
        self.combo_plane_anchor = QtWidgets.QComboBox()
        self.combo_plane_anchor.addItem("Start", "start")
        self.combo_plane_anchor.addItem("Center", "center")
        self.combo_plane_anchor.addItem("End", "end")
        self.combo_plane_anchor.addItem("Junction", "junction")
        self.combo_plane_direction = QtWidgets.QComboBox()
        self.combo_plane_direction.addItem("Toward start", "toward_start")
        self.combo_plane_direction.addItem("Toward end", "toward_end")
        self.combo_plane_direction.addItem("Both sides", "both")
        self.combo_plane_spacing_mode = QtWidgets.QComboBox()
        self.combo_plane_spacing_mode.addItem("Fraction of path", "fraction")
        self.combo_plane_spacing_mode.addItem("Distance (mm)", "distance")
        self.edit_plane_ratio = QtWidgets.QLineEdit("0.25")
        self.chk_plane_seg_filter = QtWidgets.QCheckBox()
        self.chk_plane_seg_filter.setChecked(True)
        self.edit_plane_offset = QtWidgets.QLineEdit("5.0")
        self.edit_plane_smooth_win = QtWidgets.QLineEdit("15")
        self.edit_plane_smooth_poly = QtWidgets.QLineEdit("2")
        self.edit_plane_inter_time = QtWidgets.QLineEdit("10")
        fl.addRow("Plane Mode", self.combo_plane_mode)
        fl.addRow("Plane Count (-1 = all)", self.edit_plane_count)
        fl.addRow("Spacing (mm)", self.edit_plane_dist)
        fl.addRow("Spacing Mode", self.combo_plane_spacing_mode)
        fl.addRow("Spacing Ratio", self.edit_plane_ratio)
        fl.addRow("Start Distance (mm)", self.edit_plane_start)
        fl.addRow("End Distance (mm)", self.edit_plane_end)
        fl.addRow("Anchor", self.combo_plane_anchor)
        fl.addRow("Direction", self.combo_plane_direction)
        fl.addRow("Segmentation Filter", self.chk_plane_seg_filter)
        fl.addRow("Junction Offset (mm)", self.edit_plane_offset)
        fl.addRow("SavGol Window", self.edit_plane_smooth_win)
        fl.addRow("SavGol Polyorder", self.edit_plane_smooth_poly)
        fl.addRow("Inter-time", self.edit_plane_inter_time)
        self.params_layout.addWidget(grp)
        self._sync_plane_mode_ui()

    def _build_streamline_params(self):
        grp = QtWidgets.QGroupBox("Streamline / Pathline Parameters")
        fl = QtWidgets.QFormLayout(grp)
        self.edit_sl_ratio = QtWidgets.QLineEdit("0.02")
        self.edit_sl_maxsteps = QtWidgets.QLineEdit("2000")
        self.edit_pathline_ratio = QtWidgets.QLineEdit("0.2")
        self.edit_pathline_maxsteps = QtWidgets.QLineEdit("200")
        self.edit_pathline_minseeds = QtWidgets.QLineEdit("50")
        self.combo_pathline_seed_mode = QtWidgets.QComboBox()
        self.combo_pathline_seed_mode.addItem("Fixed Count", "fixed")
        self.combo_pathline_seed_mode.addItem("Ratio", "ratio")
        self.edit_pathline_max_seeds = QtWidgets.QLineEdit("250")
        self.edit_sl_terminal = QtWidgets.QLineEdit("0.01")
        self.edit_pathline_terminal = QtWidgets.QLineEdit("0.01")
        self.edit_pathline_rng_seed = QtWidgets.QLineEdit("0")
        self.edit_pathline_tube_radius = QtWidgets.QLineEdit("0.25")
        self.edit_pathline_temporal_cache_mb = QtWidgets.QLineEdit("512")
        self.combo_pathline_color_mode = QtWidgets.QComboBox()
        self.combo_pathline_color_mode.addItem("Uniform", "uniform")
        self.combo_pathline_color_mode.addItem("Per Plane", "per_plane")
        self.combo_pathline_color_mode.addItem("Per Group", "per_group")
        self.edit_pathline_color = QtWidgets.QLineEdit("deepskyblue")
        fl.addRow("Streamline Seed Ratio", self.edit_sl_ratio)
        fl.addRow("Streamline Max Steps", self.edit_sl_maxsteps)
        fl.addRow("Streamline Terminal Speed", self.edit_sl_terminal)
        fl.addRow("Pathline Seed Mode", self.combo_pathline_seed_mode)
        fl.addRow("Pathline Seed Count / Limit", self.edit_pathline_max_seeds)
        fl.addRow("Pathline Seed Ratio", self.edit_pathline_ratio)
        fl.addRow("Pathline Minimum Seeds", self.edit_pathline_minseeds)
        fl.addRow("Pathline Max Steps", self.edit_pathline_maxsteps)
        fl.addRow("Pathline Terminal Speed", self.edit_pathline_terminal)
        fl.addRow("Pathline RNG Seed", self.edit_pathline_rng_seed)
        fl.addRow("Pathline Tube Radius (mm)", self.edit_pathline_tube_radius)
        fl.addRow("Temporal Cache (MiB)", self.edit_pathline_temporal_cache_mb)
        fl.addRow("Pathline Color Mode", self.combo_pathline_color_mode)
        fl.addRow("Uniform Pathline Color", self.edit_pathline_color)
        self.params_layout.addWidget(grp)

    def _build_pwv_params(self):
        grp = QtWidgets.QGroupBox("PWV Parameters")
        layout = QtWidgets.QVBoxLayout(grp)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        form = QtWidgets.QFormLayout()
        self.edit_pwv_interval = QtWidgets.QLineEdit("10.0")
        self.edit_pwv_start = QtWidgets.QLineEdit("0.0")
        self.edit_pwv_end = QtWidgets.QLineEdit("0.0")
        self.edit_pwv_smooth_win = QtWidgets.QLineEdit("15")
        self.edit_pwv_smooth_poly = QtWidgets.QLineEdit("2")
        self.edit_pwv_inter_time = QtWidgets.QLineEdit("10")
        self.edit_pwv_waveform = QtWidgets.QLineEdit("flowrate_mL_s")
        self.combo_pwv_tt_method = QtWidgets.QComboBox()
        self.combo_pwv_tt_method.addItems(["foot_to_foot", "cross_correlation"])
        self.combo_pwv_foot_method = QtWidgets.QComboBox()
        self.combo_pwv_foot_method.addItems(["tangent", "threshold"])
        self.edit_pwv_foot_win = QtWidgets.QLineEdit("5")
        self.edit_pwv_foot_poly = QtWidgets.QLineEdit("2")
        self.edit_pwv_foot_threshold = QtWidgets.QLineEdit("10.0")
        self.combo_pwv_xcorr_window = QtWidgets.QComboBox()
        self.combo_pwv_xcorr_window.addItems(["full", "upstroke"])
        self.edit_pwv_xcorr_interp = QtWidgets.QLineEdit("10")
        self.chk_pwv_allow_wrap = QtWidgets.QCheckBox()
        self.chk_pwv_allow_wrap.setChecked(True)
        self.edit_pwv_min_planes = QtWidgets.QLineEdit("2")
        self.chk_pwv_scene_visible = QtWidgets.QCheckBox()
        self.chk_pwv_scene_visible.setChecked(True)
        self.edit_pwv_scene_color = QtWidgets.QLineEdit("#ffd43b")
        self.edit_pwv_plot_color = QtWidgets.QLineEdit("#2b8a3e")
        self.edit_pwv_fit_color = QtWidgets.QLineEdit("#f08c00")
        self.edit_pwv_plot_dpi = QtWidgets.QLineEdit("160")

        form.addRow("Plane Interval (mm)", self.edit_pwv_interval)
        form.addRow("Start Distance (mm)", self.edit_pwv_start)
        form.addRow("End Distance (mm)", self.edit_pwv_end)
        form.addRow("Path SavGol Window", self.edit_pwv_smooth_win)
        form.addRow("Path SavGol Polyorder", self.edit_pwv_smooth_poly)
        form.addRow("Inter-time", self.edit_pwv_inter_time)
        form.addRow("Waveform Key", self.edit_pwv_waveform)
        form.addRow("Transit-Time Method", self.combo_pwv_tt_method)
        form.addRow("Foot Method", self.combo_pwv_foot_method)
        form.addRow("Foot SavGol Window", self.edit_pwv_foot_win)
        form.addRow("Foot SavGol Polyorder", self.edit_pwv_foot_poly)
        form.addRow("Foot Threshold (%)", self.edit_pwv_foot_threshold)
        form.addRow("XCorr Window", self.combo_pwv_xcorr_window)
        form.addRow("XCorr Interp", self.edit_pwv_xcorr_interp)
        form.addRow("Allow Cycle Wrap", self.chk_pwv_allow_wrap)
        form.addRow("Minimum Valid Planes", self.edit_pwv_min_planes)
        form.addRow("Show PWV Planes", self.chk_pwv_scene_visible)
        form.addRow("PWV Plane Color", self.edit_pwv_scene_color)
        form.addRow("Plot Color", self.edit_pwv_plot_color)
        form.addRow("Fit Color", self.edit_pwv_fit_color)
        form.addRow("Plot DPI", self.edit_pwv_plot_dpi)
        layout.addLayout(form)

        group_box = QtWidgets.QGroupBox("PWV Groups")
        group_layout = QtWidgets.QVBoxLayout(group_box)
        group_layout.setContentsMargins(6, 6, 6, 6)
        group_layout.setSpacing(6)
        self.table_pwv_groups = QtWidgets.QTableWidget(0, 2)
        self.table_pwv_groups.setHorizontalHeaderLabels(["Name", "Labels"])
        self.table_pwv_groups.verticalHeader().setVisible(False)
        self.table_pwv_groups.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table_pwv_groups.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.table_pwv_groups.setAlternatingRowColors(True)
        self.table_pwv_groups.setMinimumHeight(130)
        header = self.table_pwv_groups.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        group_layout.addWidget(self.table_pwv_groups)

        group_buttons = QtWidgets.QHBoxLayout()
        self.btn_pwv_group_add = QtWidgets.QPushButton("Add Group")
        self.btn_pwv_group_remove = QtWidgets.QPushButton("Remove Selected")
        self.btn_pwv_group_add.clicked.connect(self._add_pwv_group_row)
        self.btn_pwv_group_remove.clicked.connect(self._remove_selected_pwv_group_rows)
        group_buttons.addWidget(self.btn_pwv_group_add)
        group_buttons.addWidget(self.btn_pwv_group_remove)
        group_buttons.addStretch()
        group_layout.addLayout(group_buttons)

        hint = QtWidgets.QLabel(
            "Labels accept ids or symbols from labels.json, separated by comma, semicolon, or +."
        )
        hint.setWordWrap(True)
        group_layout.addWidget(hint)
        layout.addWidget(group_box)
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

        render_group = QtWidgets.QGroupBox("Render / Colorbar")
        render_layout = QtWidgets.QVBoxLayout(render_group)
        render_layout.setContentsMargins(6, 6, 6, 6)
        render_layout.setSpacing(6)

        self.runtime_render_controls = {}
        for key, label in _RUNTIME_RENDER_METRICS:
            row = QtWidgets.QHBoxLayout()
            row.addWidget(QtWidgets.QLabel(f"{label} clim"))
            edit_min = QtWidgets.QLineEdit()
            edit_min.setPlaceholderText("auto")
            edit_max = QtWidgets.QLineEdit()
            edit_max.setPlaceholderText("auto")
            row.addWidget(edit_min)
            row.addWidget(QtWidgets.QLabel("to"))
            row.addWidget(edit_max)
            row_widget = QtWidgets.QWidget()
            row_widget.setLayout(row)
            render_layout.addWidget(row_widget)
            self.runtime_render_controls[key] = {"min": edit_min, "max": edit_max}

        render_form = QtWidgets.QFormLayout()
        self.edit_runtime_scalar_bar_width = QtWidgets.QLineEdit()
        self.edit_runtime_scalar_bar_height = QtWidgets.QLineEdit()
        self.edit_runtime_scalar_bar_gap = QtWidgets.QLineEdit()
        self.edit_runtime_scalar_bar_pos_x = QtWidgets.QLineEdit()
        self.edit_runtime_scalar_bar_pos_y = QtWidgets.QLineEdit()
        self.edit_runtime_scalar_bar_title_font = QtWidgets.QLineEdit()
        self.edit_runtime_scalar_bar_label_font = QtWidgets.QLineEdit()
        render_form.addRow("Colorbar Width", self.edit_runtime_scalar_bar_width)
        render_form.addRow("Colorbar Height", self.edit_runtime_scalar_bar_height)
        render_form.addRow("Colorbar Gap", self.edit_runtime_scalar_bar_gap)
        render_form.addRow("Colorbar X", self.edit_runtime_scalar_bar_pos_x)
        render_form.addRow("Colorbar Y", self.edit_runtime_scalar_bar_pos_y)
        render_form.addRow("Title Font", self.edit_runtime_scalar_bar_title_font)
        render_form.addRow("Label Font", self.edit_runtime_scalar_bar_label_font)
        render_layout.addLayout(render_form)

        render_hint = QtWidgets.QLabel("These values default to config, update the live 3D scene immediately, and are reused by Export Videos in the current session.")
        render_hint.setWordWrap(True)
        render_layout.addWidget(render_hint)

        for widgets in self.runtime_render_controls.values():
            widgets["min"].editingFinished.connect(self._on_runtime_render_settings_changed)
            widgets["max"].editingFinished.connect(self._on_runtime_render_settings_changed)
        for widget in [
            self.edit_runtime_scalar_bar_width,
            self.edit_runtime_scalar_bar_height,
            self.edit_runtime_scalar_bar_gap,
            self.edit_runtime_scalar_bar_pos_x,
            self.edit_runtime_scalar_bar_pos_y,
            self.edit_runtime_scalar_bar_title_font,
            self.edit_runtime_scalar_bar_label_font,
        ]:
            widget.editingFinished.connect(self._on_runtime_render_settings_changed)

        self.params_layout.addWidget(grp_wss)
        self.params_layout.addWidget(render_group)

        grp_tke = QtWidgets.QGroupBox("Flow / TKE / Pressure Parameters")
        fl_tke = QtWidgets.QFormLayout(grp_tke)
        self.edit_dm_rho = QtWidgets.QLineEdit("1060.0")
        self.edit_dm_pg_smoothing_sigma = QtWidgets.QLineEdit("0.0")
        self.edit_dm_pg_support_erosion = QtWidgets.QLineEdit("1")
        self.edit_dm_pg_opacity = QtWidgets.QLineEdit("0.6")
        self.edit_dm_rp_opacity = QtWidgets.QLineEdit("0.6")
        self.combo_dm_pressure_method = QtWidgets.QComboBox()
        self.combo_dm_pressure_method.addItems(["least_squares", "ppe"])
        self.chk_dm_multithread = QtWidgets.QCheckBox()
        self.chk_dm_multithread.setChecked(False)
        fl_tke.addRow(u"Density \u03c1 (kg/m\u00b3)", self.edit_dm_rho)
        fl_tke.addRow("Relative Pressure Method", self.combo_dm_pressure_method)
        fl_tke.addRow("Pressure-Gradient Gaussian Sigma (vox)", self.edit_dm_pg_smoothing_sigma)
        fl_tke.addRow("Pressure Support Erosion (vox)", self.edit_dm_pg_support_erosion)
        fl_tke.addRow("Pressure Gradient Opacity", self.edit_dm_pg_opacity)
        fl_tke.addRow("Relative Pressure Opacity", self.edit_dm_rp_opacity)
        fl_tke.addRow("Multi-thread Metrics", self.chk_dm_multithread)
        self.params_layout.addWidget(grp_tke)

    def _build_vortex_params(self):
        """Keep vortex controls visible without expanding every derived setting."""
        grp = QtWidgets.QGroupBox("Vortex Kinematics Parameters")
        form = QtWidgets.QFormLayout(grp)
        self.edit_dm_vortex_smoothing_sigma = QtWidgets.QLineEdit("0.0")
        self.edit_dm_vortex_support_erosion = QtWidgets.QLineEdit("1")
        form.addRow("Gaussian Sigma (vox)", self.edit_dm_vortex_smoothing_sigma)
        form.addRow("Support Erosion (vox)", self.edit_dm_vortex_support_erosion)
        self.params_layout.addWidget(grp)

    def _runtime_render_clim_keys(self):
        return {
            "wss": "wss_clim",
            "tke": "tke_clim",
            "pressure_gradient": "pressure_gradient_clim",
            "relative_pressure": "relative_pressure_clim",
            "streamline": "streamline_clim",
        }

    def _runtime_bar_cfg_keys(self):
        return [
            "wss_bar_cfg",
            "tke_bar_cfg",
            "pressure_gradient_bar_cfg",
            "relative_pressure_bar_cfg",
            "streamline_bar_cfg",
        ]

    def _parse_optional_clim_pair(self, min_text, max_text, fallback=None):
        min_token = str(min_text or "").strip().lower()
        max_token = str(max_text or "").strip().lower()
        if min_token in {"", "auto"} and max_token in {"", "auto"}:
            return None
        lo = self._optional_float_from_text(min_text, None)
        hi = self._optional_float_from_text(max_text, None)
        if lo is None or hi is None:
            return fallback
        if hi < lo:
            lo, hi = hi, lo
        return (float(lo), float(hi))

    def _apply_render_settings_to_scene_objects(self):
        ws = self.workspace
        render_cfg = dict(getattr(ws, "render_settings", {}) or {})
        mapping = {
            "segmask_raw_surface": (None, "shared_colorbar_show", "shared_colorbar_bar_cfg"),
            "wss_surface_live": ("wss_clim", "wss_show_scalar_bar", "wss_bar_cfg"),
            "tke_volume": ("tke_clim", "tke_show_scalar_bar", "tke_bar_cfg"),
            "pressure_gradient_volume": ("pressure_gradient_clim", "pressure_gradient_show_scalar_bar", "pressure_gradient_bar_cfg"),
            "relative_pressure_volume": ("relative_pressure_clim", "relative_pressure_show_scalar_bar", "relative_pressure_bar_cfg"),
            "streamlines_live": ("streamline_clim", "streamline_show_scalar_bar", "streamline_bar_cfg"),
            "vorticity_magnitude_volume": (None, "shared_colorbar_show", "shared_colorbar_bar_cfg"),
            "q_criterion_volume": (None, "shared_colorbar_show", "shared_colorbar_bar_cfg"),
            "swirling_strength_volume": (None, "shared_colorbar_show", "shared_colorbar_bar_cfg"),
        }
        touched = False
        for obj in ws.scene_objects.values():
            keys = mapping.get(str(getattr(obj, "data_key", "") or ""))
            if keys is None:
                continue
            clim_key, show_key, bar_key = keys
            if clim_key is not None:
                obj.clim = render_cfg.get(clim_key)
            obj.show_scalar_bar = bool(render_cfg.get(show_key, True))
            obj.scalar_bar_cfg = dict(render_cfg.get(bar_key, render_cfg.get("shared_colorbar_bar_cfg", {})) or {})
            self.scene.readd_object(obj)
            touched = True
        if touched:
            self._refresh_browser()
            self._refresh_scene()
            self.ortho_viewer.refresh()

    def _on_runtime_render_settings_changed(self):
        if not hasattr(self, "runtime_render_controls"):
            return
        ws = self.workspace
        render_cfg = dict(getattr(ws, "render_settings", {}) or {})
        default_cfg = bundle_to_autoflow_kwargs(self._config_bundle)
        for metric_key, cfg_key in self._runtime_render_clim_keys().items():
            widgets = self.runtime_render_controls.get(metric_key, {})
            clim_value = self._parse_optional_clim_pair(
                widgets.get("min").text() if widgets.get("min") is not None else "",
                widgets.get("max").text() if widgets.get("max") is not None else "",
                fallback=default_cfg.get(cfg_key),
            )
            render_cfg[cfg_key] = clim_value

        width = max(self._float_from_text(self.edit_runtime_scalar_bar_width.text(), 0.08), 0.01)
        height = max(self._float_from_text(self.edit_runtime_scalar_bar_height.text(), 0.65), 0.05)
        gap = max(self._float_from_text(self.edit_runtime_scalar_bar_gap.text(), 0.03), 0.0)
        pos_x = min(max(self._float_from_text(self.edit_runtime_scalar_bar_pos_x.text(), 0.87), 0.0), 0.98)
        pos_y = min(max(self._float_from_text(self.edit_runtime_scalar_bar_pos_y.text(), 0.15), 0.0), 0.95)
        title_font = max(self._int_from_text(self.edit_runtime_scalar_bar_title_font.text(), 14), 1)
        label_font = max(self._int_from_text(self.edit_runtime_scalar_bar_label_font.text(), 11), 1)
        shared_bar_cfg = dict(render_cfg.get("shared_colorbar_bar_cfg", {}) or {})
        shared_bar_cfg["width"] = float(width)
        shared_bar_cfg["height"] = float(height)
        shared_bar_cfg["position_x"] = float(pos_x)
        shared_bar_cfg["position_y"] = float(pos_y)
        shared_bar_cfg["stack_gap"] = float(gap)
        shared_bar_cfg["title_font_size"] = int(title_font)
        shared_bar_cfg["label_font_size"] = int(label_font)
        render_cfg["shared_colorbar_bar_cfg"] = shared_bar_cfg
        for bar_key in self._runtime_bar_cfg_keys():
            bar_cfg = dict(render_cfg.get(bar_key, {}) or {})
            bar_cfg["width"] = float(width)
            bar_cfg["height"] = float(height)
            bar_cfg["position_x"] = float(pos_x)
            bar_cfg["position_y"] = float(pos_y)
            bar_cfg["stack_gap"] = float(gap)
            bar_cfg["title_font_size"] = int(title_font)
            bar_cfg["label_font_size"] = int(label_font)
            render_cfg[bar_key] = bar_cfg
        ws.render_settings = render_cfg
        self._apply_render_settings_to_scene_objects()

    def _add_pwv_group_row(self, name="", labels=""):
        row = self.table_pwv_groups.rowCount()
        self.table_pwv_groups.insertRow(row)
        self.table_pwv_groups.setItem(row, 0, QtWidgets.QTableWidgetItem(str(name or "")))
        self.table_pwv_groups.setItem(row, 1, QtWidgets.QTableWidgetItem(str(labels or "")))
        self.table_pwv_groups.setCurrentCell(row, 0)

    def _remove_selected_pwv_group_rows(self):
        rows = sorted({idx.row() for idx in self.table_pwv_groups.selectionModel().selectedRows()}, reverse=True)
        if not rows and self.table_pwv_groups.rowCount() > 0:
            rows = [self.table_pwv_groups.currentRow()]
        for row in rows:
            if row >= 0:
                self.table_pwv_groups.removeRow(row)

    def _split_pwv_group_labels_text(self, text):
        tokens = re.split(r"[,;+\n\t]+", str(text or ""))
        return [tok.strip() for tok in tokens if tok.strip()]

    def _pwv_group_payload_from_ui(self):
        groups = []
        for row in range(self.table_pwv_groups.rowCount()):
            name_item = self.table_pwv_groups.item(row, 0)
            labels_item = self.table_pwv_groups.item(row, 1)
            name = str(name_item.text()).strip() if name_item is not None else ""
            labels_text = str(labels_item.text()).strip() if labels_item is not None else ""
            groups.append({
                "name": name,
                "labels": self._split_pwv_group_labels_text(labels_text),
            })
        return groups

    def _pwv_group_labels_for_ui(self, labels):
        reverse_map = {}
        for key, value in dict(self.workspace.label_params.label_map or {}).items():
            reverse_map.setdefault(int(value), str(key))
        display = []
        for label_value in list(labels or []):
            try:
                label_id = int(label_value)
            except Exception:
                continue
            display.append(reverse_map.get(label_id, str(label_id)))
        return ", ".join(display)

    def _build_review_params(self):
        quality_group = QtWidgets.QGroupBox("Quality Control")
        quality_layout = QtWidgets.QVBoxLayout(quality_group)
        self.quality_summary_label = QtWidgets.QLabel("QC report has not been generated for this workspace.")
        self.quality_summary_label.setWordWrap(True)
        quality_layout.addWidget(self.quality_summary_label)

        self.quality_table = QtWidgets.QTreeWidget()
        self.quality_table.setHeaderLabels(["Status", "Stage", "Check", "Result"])
        self.quality_table.setRootIsDecorated(False)
        self.quality_table.setAlternatingRowColors(True)
        quality_header = self.quality_table.header()
        quality_header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        quality_header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        quality_header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        quality_header.setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)
        self.quality_table.setMinimumHeight(150)
        quality_layout.addWidget(self.quality_table)

        quality_buttons = QtWidgets.QHBoxLayout()
        self.btn_refresh_quality = QtWidgets.QPushButton("Refresh QC")
        self.btn_refresh_quality.clicked.connect(self._refresh_quality_panel)
        self.btn_export_quality = QtWidgets.QPushButton("Export QC Report")
        self.btn_export_quality.clicked.connect(self._on_export_quality_report)
        quality_buttons.addWidget(self.btn_refresh_quality)
        quality_buttons.addWidget(self.btn_export_quality)
        quality_buttons.addStretch()
        quality_layout.addLayout(quality_buttons)
        self.params_layout.addWidget(quality_group)

        transfer_group = QtWidgets.QGroupBox("Plane Coordinates")
        transfer_layout = QtWidgets.QHBoxLayout(transfer_group)
        self.btn_export_planes = QtWidgets.QPushButton("Export Selected...")
        self.btn_export_planes.clicked.connect(self._on_export_plane_coordinates)
        transfer_layout.addWidget(self.btn_export_planes)
        transfer_layout.addStretch()
        self.params_layout.addWidget(transfer_group)

    def _workflow_stage_key(self, index=None):
        if index is None:
            index = self.workflow_tabs.currentIndex()
        if not (0 <= int(index) < len(_WORKFLOW_STAGES)):
            return "input"
        return _WORKFLOW_STAGES[int(index)][1]

    def _workflow_stage_state(self, stage_key):
        ws = self.workspace
        if stage_key == "input":
            return "Ready" if ws.data_loaded else "Not ready"
        if stage_key == "segmentation":
            if not ws.data_loaded:
                return "Not ready"
            return "Ready" if ws.segmask_raw is not None else "Needs review"
        if stage_key == "phase_unwrap":
            if not ws.data_loaded:
                return "Not ready"
            if str(ws.input_state.source_format).lower() == "legacy_h5_dual_venc" or bool((ws.input_state.metadata or {}).get("dual_venc", {}).get("enabled", False)):
                return "Skipped (dual-VENC)"
            if not ws.input_state.capabilities.has_wrapped_phase:
                return "Unavailable"
            return "Ready" if ws.pipeline.is_done(StepId.UNWRAP_PHASE) else "Optional"
        if stage_key == "centerline":
            if len(ws.planes) > 0:
                return "Ready"
            if len(ws.centerline_paths) > 0 or len(ws.graph.points) > 0:
                return "Needs review"
            return "Not ready"
        if stage_key == "hemodynamics":
            if ws.derived.plane_metrics:
                return "Ready"
            return "Needs review" if len(ws.planes) > 0 else "Not ready"
        report_status = str((self._last_quality_report or {}).get("overall_status", ""))
        return {
            "ready": "Ready",
            "needs_review": "Needs review",
            "not_ready": "Not ready",
            "incomplete": "Incomplete",
        }.get(report_status, "Incomplete" if ws.data_loaded else "Not ready")

    def _refresh_workflow_status(self):
        if not hasattr(self, "workflow_tabs"):
            return
        for index, (_label, stage_key) in enumerate(_WORKFLOW_STAGES):
            self.workflow_tabs.setTabToolTip(index, self._workflow_stage_state(stage_key))
        state = self._workflow_stage_state(self._workflow_stage_key())
        palette = {
            "Ready": ("#1b5e20", "#e8f5e9"),
            "Needs review": ("#8a4b00", "#fff4d6"),
            "Incomplete": ("#555555", "#eeeeee"),
            "Not ready": ("#8b1e1e", "#fdecec"),
            "Skipped (dual-VENC)": ("#555555", "#eeeeee"),
            "Unavailable": ("#8a4b00", "#fff4d6"),
            "Optional": ("#555555", "#eeeeee"),
        }
        foreground, background = palette.get(state, ("#555555", "#eeeeee"))
        self.workflow_status.setText(state)
        self.workflow_status.setStyleSheet(
            f"QLabel {{ color: {foreground}; background: {background}; border: 1px solid {foreground}; padding: 3px 8px; border-radius: 4px; }}"
        )

    def _on_advanced_toggled(self, checked):
        self.btn_show_advanced.setArrowType(QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow)
        self._on_workflow_stage_changed(self.workflow_tabs.currentIndex())

    def _on_workflow_stage_changed(self, index):
        stage_key = self._workflow_stage_key(index)
        show_advanced = bool(self.btn_show_advanced.isChecked())
        for key, entries in self._workflow_param_widgets.items():
            for widget, advanced in entries:
                widget.setVisible(key == stage_key and (show_advanced or not advanced))

        visible_steps = _WORKFLOW_STEPS.get(stage_key, set())
        # Reflow the action grid for the active stage so hidden stages do not
        # leave an empty first row or a confusing button order.
        step_layout = self.steps_group.layout()
        while step_layout.count():
            item = step_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setVisible(False)
        ordered_steps = [
            StepId.UNWRAP_PHASE,
            StepId.GENERATE_SKELETON, StepId.EDIT_SKELETON,
            StepId.GENERATE_GRAPH, StepId.EDIT_GRAPH, StepId.GENERATE_PLANES,
            StepId.COMPUTE_PLANE_METRICS, StepId.COMPUTE_PWV,
            StepId.COMPUTE_DERIVED_METRICS, StepId.GENERATE_STREAMLINES,
            StepId.PLANE_STREAMLINES,
        ]
        active_steps = [step for step in ordered_steps if step in visible_steps]
        if stage_key == "hemodynamics":
            flow_steps = [
                StepId.COMPUTE_PLANE_METRICS,
                StepId.COMPUTE_DERIVED_METRICS,
                StepId.GENERATE_STREAMLINES,
                StepId.PLANE_STREAMLINES,
            ]
            for idx, step in enumerate(flow_steps):
                step_layout.addWidget(self.step_buttons[step], idx // 2, idx % 2)
            step_layout.addWidget(self.btn_run_all, 2, 0, 1, 2)
            pwv_button = self.step_buttons[StepId.COMPUTE_PWV]
            pwv_button.setMinimumHeight(34)
            step_layout.addWidget(pwv_button, 3, 0, 1, 2)
        else:
            for idx, step in enumerate(active_steps):
                step_layout.addWidget(self.step_buttons[step], idx // 2, idx % 2)
            action_row = (len(active_steps) + 1) // 2
            if stage_key in {"centerline", "hemodynamics"}:
                step_layout.addWidget(self.btn_run_all, action_row, 0, 1, 2)
        if stage_key == "centerline":
            plane_row = active_steps.index(StepId.GENERATE_PLANES) // 2
            step_layout.addWidget(self.btn_import_planes_centerline, plane_row, 1)
        for step, button in self.step_buttons.items():
            button.setVisible(step in visible_steps)
        if hasattr(self, "btn_import_planes_centerline"):
            self.btn_import_planes_centerline.setVisible(stage_key == "centerline")
        self.btn_run_all.setVisible(stage_key in {"centerline", "hemodynamics"})
        if stage_key in {"centerline", "hemodynamics"}:
            self.btn_run_all.setText("Run All")
        self.steps_group.setVisible(bool(visible_steps) or self.btn_run_all.isVisible())
        stage_label = _WORKFLOW_STAGES[int(index)][0].replace("&", "&&")
        self.steps_group.setTitle(f"{stage_label} Actions")
        if hasattr(self, "mid_splitter"):
            self.mid_splitter.setSizes([330, 360] if stage_key == "review" else [480, 220])

        if hasattr(self, "segmentation_dock"):
            self.segmentation_dock.setVisible(stage_key == "segmentation")
            if stage_key == "segmentation":
                self.segmentation_dock.raise_()
        if hasattr(self, "pwv_dock"):
            self.pwv_dock.setVisible(stage_key in {"hemodynamics", "review"})
            if stage_key in {"hemodynamics", "review"}:
                self.pwv_dock.raise_()
        if self._edit_mode is not None and self._edit_panel is not None:
            self.steps_group.setVisible(False)
            self._edit_panel.setVisible(True)
        if stage_key == "review" and self.workspace.data_loaded:
            QtCore.QTimer.singleShot(0, self._refresh_quality_panel)
        self._refresh_workflow_status()

    def _build_timeline(self, parent):
        grp = QtWidgets.QGroupBox("Timeline")
        tl = QtWidgets.QHBoxLayout(grp)
        self.btn_prev = QtWidgets.QPushButton()
        self.btn_prev.setIcon(standard_icon(self, "SP_MediaSkipBackward"))
        self.btn_prev.setToolTip("Previous frame")
        self.btn_prev.clicked.connect(self._on_prev_frame)
        self.btn_play = QtWidgets.QPushButton()
        self.btn_play.setIcon(standard_icon(self, "SP_MediaPlay"))
        self.btn_play.setToolTip("Play")
        self.btn_play.clicked.connect(self._on_play)
        self.btn_pause = QtWidgets.QPushButton()
        self.btn_pause.setIcon(standard_icon(self, "SP_MediaPause"))
        self.btn_pause.setToolTip("Pause")
        self.btn_pause.clicked.connect(self._on_pause)
        self.btn_next = QtWidgets.QPushButton()
        self.btn_next.setIcon(standard_icon(self, "SP_MediaSkipForward"))
        self.btn_next.setToolTip("Next frame")
        self.btn_next.clicked.connect(self._on_next_frame)
        self.slider_t = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_t.setRange(0, 0)
        self.slider_t.setTracking(True)
        self.slider_t.valueChanged.connect(self._on_t_changed)
        self.slider_t.sliderReleased.connect(self._finish_timeline_scrub)
        self.lab_t = QtWidgets.QLabel("0")
        self.lab_t.setAlignment(QtCore.Qt.AlignCenter)
        self.lab_t.setMinimumWidth(32)
        self.spin_interval = QtWidgets.QSpinBox()
        self.spin_interval.setRange(10, 2000)
        self.spin_interval.setValue(120)
        self.spin_interval.setSuffix(" ms")
        self.lab_playback_performance = QtWidgets.QLabel("Render: --")
        self.lab_playback_performance.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.lab_playback_performance.setMinimumWidth(180)
        self.lab_playback_performance.setToolTip("Rolling time spent updating the timeline scene and slice views, plus effective playback FPS.")
        for w in [self.btn_prev, self.btn_play, self.btn_pause, self.btn_next]:
            w.setProperty("role", "icon")
            w.setAccessibleName(w.toolTip())
            tl.addWidget(w)
        tl.addWidget(self.slider_t, 1)
        tl.addWidget(self.lab_t)
        tl.addWidget(self.lab_playback_performance)
        tl.addWidget(self.spin_interval)
        parent.addWidget(grp)

    def _build_selection_info(self, parent):
        lay = QtWidgets.QHBoxLayout()
        lay.setContentsMargins(0, 0, 0, 0)
        box_plane = QtWidgets.QGroupBox("Plane")
        lay_plane = QtWidgets.QVBoxLayout(box_plane)
        self.text_plane_info = QtWidgets.QPlainTextEdit()
        self.text_plane_info.setReadOnly(True)
        self.text_plane_info.setMaximumHeight(82)
        lay_plane.addWidget(self.text_plane_info)
        plane_actions = QtWidgets.QHBoxLayout()
        self.btn_add_plane = QtWidgets.QPushButton("Add Plane")
        self.btn_add_plane.setToolTip("Add a free plane at the current ortho cursor")
        self.btn_add_plane.clicked.connect(self._add_manual_plane)
        self.btn_edit_plane = QtWidgets.QPushButton("Edit Plane")
        self.btn_edit_plane.setToolTip("Move and orient the selected plane with the 3D handles")
        self.btn_edit_plane.clicked.connect(self._toggle_selected_plane_edit)
        plane_actions.addWidget(self.btn_add_plane)
        plane_actions.addWidget(self.btn_edit_plane)
        lay_plane.addLayout(plane_actions)
        box_path = QtWidgets.QGroupBox("Path")
        lay_path = QtWidgets.QVBoxLayout(box_path)
        self.text_path_info = QtWidgets.QPlainTextEdit()
        self.text_path_info.setReadOnly(True)
        self.text_path_info.setMaximumHeight(82)
        lay_path.addWidget(self.text_path_info)
        lay.addWidget(box_plane, 1)
        lay.addWidget(box_path, 1)
        parent.addLayout(lay)

    def _build_log(self, parent):
        self.console = QtWidgets.QTextEdit()
        self.console.setReadOnly(True)
        parent.addWidget(self.console)

    def _build_segmentation_dock(self):
        self.segmentation_dock = QtWidgets.QDockWidget("Segmentation", self)
        self.segmentation_dock.setObjectName("SegmentationDock")
        self.segmentation_panel = SegmentationDock(self)
        self.segmentation_dock.setWidget(self.segmentation_panel)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.segmentation_dock)
        self.segmentation_panel.combo_source.currentIndexChanged.connect(self._on_segmentation_source_changed)
        self.segmentation_panel.check_visible.toggled.connect(self._on_segmentation_visibility_changed)
        self.segmentation_panel.slider_opacity.valueChanged.connect(self._on_segmentation_opacity_changed)
        self.segmentation_panel.btn_configure.clicked.connect(self._on_configure_segmentation)
        self.segmentation_panel.btn_run_auto.clicked.connect(self._on_run_auto_segmentation)
        self.segmentation_panel.btn_import.clicked.connect(self._on_import_segmentation)
        self.segmentation_panel.btn_save.clicked.connect(self._on_save_active_segmentation)
        self.segmentation_panel.table_labels.itemSelectionChanged.connect(self._on_segmentation_label_selected)
        self.segmentation_panel.table_labels.itemChanged.connect(self._on_segmentation_label_item_changed)
        self.segmentation_panel.spin_active_label.valueChanged.connect(self._on_active_label_changed)
        self.segmentation_panel.edit_active_name.editingFinished.connect(self._on_active_label_name_changed)
        self.segmentation_panel.btn_active_color.clicked.connect(self._on_active_label_color_clicked)
        self.segmentation_panel.btn_external_editor.clicked.connect(self._open_external_segmentation_editor)
        self.segmentation_panel.btn_apply_cleanup_4d.clicked.connect(self._on_apply_4d_cleanup)

    def _on_apply_4d_cleanup(self):
        if not self.workspace.data_loaded or self.workspace.segmask_raw is None:
            self.log("Load a 4D segmentation before applying connected-component cleanup.")
            return
        self._sync_params_to_ws()
        self.workspace.reset_segmentation_results()
        try:
            self.pipeline.preprocess(self.workspace)
            self.scene.sync_from_workspace()
            self._refresh_all()
            self.log("Applied 4D connected-component cleanup and rebuilt segmentation surfaces.")
        except Exception as exc:
            self.log(f"4D cleanup failed: {type(exc).__name__}: {exc}")

    def _build_pwv_dock(self):
        self.pwv_dock = QtWidgets.QDockWidget("Analysis", self)
        self.pwv_dock.setObjectName("PWVDock")
        panel = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel("Display"))
        self.combo_analysis_mode = QtWidgets.QComboBox()
        for label, value in _ANALYSIS_MODE_ITEMS:
            self.combo_analysis_mode.addItem(label, value)
        self.combo_analysis_mode.setCurrentIndex(self.combo_analysis_mode.findData("plane_curve"))
        self.combo_analysis_mode.currentIndexChanged.connect(self._on_analysis_mode_changed)
        mode_row.addWidget(self.combo_analysis_mode, 1)
        layout.addLayout(mode_row)

        self.analysis_controls_stack = QtWidgets.QStackedWidget()

        pwv_controls = QtWidgets.QWidget(self)
        pwv_row = QtWidgets.QHBoxLayout(pwv_controls)
        pwv_row.setContentsMargins(0, 0, 0, 0)
        pwv_row.addWidget(QtWidgets.QLabel("PWV Group"))
        self.combo_pwv_group = QtWidgets.QComboBox()
        self.combo_pwv_group.currentIndexChanged.connect(self._on_pwv_group_changed)
        pwv_row.addWidget(self.combo_pwv_group, 1)
        self.analysis_controls_stack.addWidget(pwv_controls)

        curve_controls = QtWidgets.QWidget(self)
        curve_row = QtWidgets.QHBoxLayout(curve_controls)
        curve_row.setContentsMargins(0, 0, 0, 0)
        curve_row.addWidget(QtWidgets.QLabel("Plane Metric"))
        self.combo_plane_curve_metric = QtWidgets.QComboBox()
        self.combo_plane_curve_metric.currentIndexChanged.connect(self._on_plane_curve_metric_changed)
        curve_row.addWidget(self.combo_plane_curve_metric, 1)
        self.analysis_controls_stack.addWidget(curve_controls)

        pressure_controls = QtWidgets.QWidget(self)
        pressure_row = QtWidgets.QHBoxLayout(pressure_controls)
        pressure_row.setContentsMargins(0, 0, 0, 0)
        pressure_row.addWidget(QtWidgets.QLabel("Path"))
        self.combo_centerline_pressure_path = QtWidgets.QComboBox()
        self.combo_centerline_pressure_path.currentIndexChanged.connect(self._on_centerline_pressure_path_changed)
        pressure_row.addWidget(self.combo_centerline_pressure_path, 1)
        self.analysis_controls_stack.addWidget(pressure_controls)

        ic_controls = QtWidgets.QWidget(self)
        ic_row = QtWidgets.QHBoxLayout(ic_controls)
        ic_row.setContentsMargins(0, 0, 0, 0)
        self.label_ic_target = QtWidgets.QLabel("Target: none")
        self.label_ic_target.setWordWrap(True)
        ic_row.addWidget(self.label_ic_target, 1)
        self.analysis_controls_stack.addWidget(ic_controls)

        layout.addWidget(self.analysis_controls_stack)

        self.label_pwv_status = QtWidgets.QLabel("No analysis available.")
        self.label_pwv_status.setWordWrap(True)
        layout.addWidget(self.label_pwv_status)

        self.fig_pwv = Figure(figsize=(4.2, 5.4), dpi=90, facecolor="white")
        self.canvas_pwv = FigureCanvas(self.fig_pwv)
        layout.addWidget(self.canvas_pwv, 1)

        self.pwv_dock.setWidget(panel)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.pwv_dock)
        self.tabifyDockWidget(self.segmentation_dock, self.pwv_dock)
        self._sync_analysis_controls()
        self.segmentation_dock.raise_()

    def _analysis_mode(self):
        value = self.combo_analysis_mode.currentData() if hasattr(self, "combo_analysis_mode") else "pwv"
        return str(value or "pwv")

    def _sync_plane_mode_ui(self):
        mode = str(self.combo_plane_mode.currentData() or "fixed_step") if hasattr(self, "combo_plane_mode") else "fixed_step"
        is_anchor = mode in {"anchored_offset", "fixed_step"}
        for widget in [getattr(self, "edit_plane_count", None)]:
            if widget is not None:
                widget.setEnabled(True)
        for widget in [getattr(self, "edit_plane_dist", None)]:
            if widget is not None:
                widget.setEnabled(True)
        for widget in [getattr(self, "combo_plane_anchor", None), getattr(self, "edit_plane_offset", None)]:
            if widget is not None:
                widget.setEnabled(is_anchor)

    def _sync_analysis_controls(self):
        if not hasattr(self, "analysis_controls_stack"):
            return
        index_map = {value: idx for idx, (_label, value) in enumerate(_ANALYSIS_MODE_ITEMS)}
        self.analysis_controls_stack.setCurrentIndex(index_map.get(self._analysis_mode(), 0))

    def _on_analysis_mode_changed(self, _index=None):
        self._sync_analysis_controls()
        self._refresh_analysis_panel()

    def _on_pwv_group_changed(self, _index=None):
        self._refresh_analysis_panel()

    def _on_plane_curve_metric_changed(self, _index=None):
        self._refresh_analysis_panel()

    def _on_centerline_pressure_path_changed(self, _index=None):
        self._refresh_analysis_panel()

    def _refresh_centerline_pressure_path_selector(self):
        if not hasattr(self, "combo_centerline_pressure_path"):
            return
        current = self.combo_centerline_pressure_path.currentData()
        profiles = list(self.workspace.derived.centerline_pressure_profiles or [])
        self.combo_centerline_pressure_path.blockSignals(True)
        self.combo_centerline_pressure_path.clear()
        for idx, item in enumerate(profiles):
            path_index = int(item.get("path_index", idx))
            peak = float(item.get("pressure_drop_peak_Pa", 0.0) or 0.0)
            self.combo_centerline_pressure_path.addItem(f"Path {path_index} ({peak:.3g} Pa)", path_index)
        self.combo_centerline_pressure_path.setEnabled(bool(profiles))
        if profiles:
            target = 0
            for idx, item in enumerate(profiles):
                if int(item.get("path_index", idx)) == int(current if current is not None else -1):
                    target = idx
                    break
            self.combo_centerline_pressure_path.setCurrentIndex(target)
        self.combo_centerline_pressure_path.blockSignals(False)

    def _selected_centerline_pressure_profile(self):
        path_index = self.combo_centerline_pressure_path.currentData() if hasattr(self, "combo_centerline_pressure_path") else None
        profiles = list(self.workspace.derived.centerline_pressure_profiles or [])
        if path_index is None and profiles:
            return profiles[0]
        for item in profiles:
            if int(item.get("path_index", -1)) == int(path_index):
                return item
        return None

    def _plot_centerline_pressure(self):
        profile = self._selected_centerline_pressure_profile()
        if profile is None:
            self.label_ic_target.setText("Target: path")
            self.label_pwv_status.setText("No centerline pressure profile available.")
            self._clear_pwv_axes("Run relative pressure to populate centerline pressure drop.", title="Centerline Pressure")
            return
        distances = np.asarray(profile.get("distances_mm", []), dtype=float).reshape(-1)
        pressure_t = list(profile.get("relative_pressure_Pa_t", []) or [])
        drop_t = np.asarray(profile.get("pressure_drop_Pa_t", []), dtype=float).reshape(-1)
        current_t = int(np.clip(self.workspace.current_t, 0, max(0, len(pressure_t) - 1)))
        values = np.asarray(pressure_t[current_t] if current_t < len(pressure_t) else [], dtype=float).reshape(-1)
        self.fig_pwv.clear()
        ax = self.fig_pwv.add_subplot(211)
        ax_drop = self.fig_pwv.add_subplot(212)
        if distances.size and values.size == distances.size:
            ax.plot(distances, values, color="#c92a2a", linewidth=2.0)
            ax.scatter([distances[0], distances[-1]], [values[0], values[-1]], color="#f08c00", zorder=4)
            ax.set_xlabel("Distance Along Centerline (mm)")
            ax.set_ylabel("Relative Pressure (Pa)")
            ax.set_title(f"Path {int(profile.get('path_index', -1))}: Relative Pressure at t={current_t}")
            ax.grid(True, alpha=0.25)
        else:
            ax.text(0.5, 0.5, "No centerline samples", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        phases = np.arange(drop_t.size, dtype=float)
        if drop_t.size:
            ax_drop.plot(phases, drop_t, color="#1f77b4", marker="o", linewidth=2.0, markersize=4)
            ax_drop.axvline(current_t, color="#f08c00", linestyle="--", linewidth=1.2, alpha=0.8)
            ax_drop.scatter([current_t], [drop_t[min(current_t, drop_t.size - 1)]], color="#f08c00", zorder=4)
            ax_drop.set_xlabel("Cardiac Phase")
            ax_drop.set_ylabel("Pressure Drop (Pa)")
            ax_drop.set_title("Centerline Pressure Drop")
            ax_drop.grid(True, alpha=0.25)
        else:
            ax_drop.text(0.5, 0.5, "No pressure-drop series", ha="center", va="center", transform=ax_drop.transAxes)
            ax_drop.set_xticks([])
            ax_drop.set_yticks([])
        self.fig_pwv.tight_layout()
        self.canvas_pwv.draw_idle()
        lines = [
            f"Path {int(profile.get('path_index', -1))}   Current phase: {current_t}",
            f"Current drop: {float(drop_t[min(current_t, drop_t.size - 1)]) if drop_t.size else 0.0:.4g} Pa   Mean drop: {float(profile.get('pressure_drop_mean_Pa', 0.0)):.4g} Pa   Peak |drop|: {float(profile.get('pressure_drop_peak_Pa', 0.0)):.4g} Pa",
        ]
        self.label_pwv_status.setText("\n".join(lines))

    def _selected_pwv_result(self):
        idx = self.combo_pwv_group.currentData()
        results = list(self.workspace.derived.pwv_results or [])
        if idx is None:
            return None
        try:
            idx = int(idx)
        except Exception:
            return None
        if not (0 <= idx < len(results)):
            return None
        return results[idx]

    def _selected_plane_metric(self):
        plane_idx = int(getattr(self, "_selected_plane_index", -1))
        if not (0 <= plane_idx < len(self.workspace.planes)):
            return -1, None
        plane = self.workspace.planes[plane_idx]
        metric = getattr(plane, "metrics", {}) or {}
        if not metric and plane_idx < len(self.workspace.derived.plane_metrics):
            metric = self.workspace.derived.plane_metrics[plane_idx]
        if not metric:
            return plane_idx, None
        return plane_idx, dict(metric)

    def _selected_analysis_path_index(self):
        path_idx = int(getattr(self.workspace, "selected_path_index", -1))
        if 0 <= path_idx < len(self.workspace.path_info):
            return path_idx
        plane_idx, metric = self._selected_plane_metric()
        if metric is None:
            return -1
        try:
            path_idx = int(metric.get("path_index", getattr(self.workspace.planes[plane_idx], "path_index", -1)))
        except Exception:
            path_idx = -1
        return path_idx if 0 <= path_idx < len(self.workspace.path_info) else -1

    def _available_plane_curve_series(self, metric):
        options = []
        payload = dict(metric or {})
        for key, label in _PLANE_CURVE_SERIES_OPTIONS:
            values = payload.get(key)
            if values is None:
                continue
            try:
                arr = np.asarray(values, dtype=float).reshape(-1)
            except Exception:
                continue
            if arr.size == 0:
                continue
            options.append((key, label))
        return options

    def _refresh_plane_curve_metric_options(self):
        current_key = self.combo_plane_curve_metric.currentData() if hasattr(self, "combo_plane_curve_metric") else None
        _plane_idx, metric = self._selected_plane_metric()
        options = self._available_plane_curve_series(metric)
        if not hasattr(self, "combo_plane_curve_metric"):
            return options
        self.combo_plane_curve_metric.blockSignals(True)
        self.combo_plane_curve_metric.clear()
        for key, label in options:
            self.combo_plane_curve_metric.addItem(label, key)
        self.combo_plane_curve_metric.setEnabled(bool(options))
        if options:
            target_idx = 0
            for idx, (key, _label) in enumerate(options):
                if key == current_key:
                    target_idx = idx
                    break
            self.combo_plane_curve_metric.setCurrentIndex(target_idx)
        self.combo_plane_curve_metric.blockSignals(False)
        return options

    def _clear_pwv_axes(self, message="No analysis available.", title="Analysis"):
        self.fig_pwv.clear()
        ax = self.fig_pwv.add_subplot(111)
        ax.set_title(str(title))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.5, str(message), ha="center", va="center", transform=ax.transAxes)
        self.fig_pwv.tight_layout()
        self.canvas_pwv.draw_idle()

    def _plot_pwv_result(self, result):
        if result is None:
            self.label_pwv_status.setText("No PWV results.")
            self._clear_pwv_axes("Compute PWV\nto populate this view.", title="PWV")
            return
        self.fig_pwv.clear()
        self.ax_pwv = self.fig_pwv.add_subplot(211)
        self.ax_pwv_flow = self.fig_pwv.add_subplot(212)
        plot_color = str(self.workspace.pwv_params.plot_color or "#2b8a3e")
        fit_color = str(self.workspace.pwv_params.fit_color or "#f08c00")
        _plot_pwv_axes(self.ax_pwv, result, color=plot_color, fit_color=fit_color)
        _plot_plane_flowrate_axes(self.ax_pwv_flow, result, color=plot_color)

        status = str(result.get("status", "") or "unknown")
        valid_count = int(result.get("valid_plane_count", 0) or 0)
        plane_count = int(result.get("plane_count", 0) or 0)
        longest_path = result.get("longest_path_length_mm")
        pwv = result.get("pwv_m_s")
        fit_r2 = result.get("fit_r2")
        tt_method = str(result.get("transit_time_method", "") or "")
        foot_method = str(result.get("foot_method", "") or "")
        lines = [f"Status: {status}   Valid planes: {valid_count}/{plane_count}"]
        extras = []
        if pwv is not None:
            extras.append(f"PWV: {float(pwv):.4g} m/s")
        if tt_method:
            extras.append(f"TT: {tt_method}")
        if foot_method and tt_method == "foot_to_foot":
            extras.append(f"Foot: {foot_method}")
        if fit_r2 is not None:
            extras.append(f"R2: {float(fit_r2):.3f}")
        if longest_path is not None:
            extras.append(f"Longest path: {float(longest_path):.1f} mm")
        if extras:
            lines.append("   ".join(extras))
        message = str(result.get("message", "") or "")
        if message:
            lines.append(message)
        plot_file = str(result.get("plot_file", "") or "")
        if plot_file:
            lines.append(f"Plot file: {plot_file}")
        plot_error = str(result.get("plot_error", "") or "")
        if plot_error:
            lines.append(f"Plot error: {plot_error}")
        self.label_pwv_status.setText("\n".join(lines))
        self.fig_pwv.tight_layout()
        self.canvas_pwv.draw_idle()

    def _plot_plane_curve(self):
        plane_idx, metric = self._selected_plane_metric()
        if metric is None:
            self.label_pwv_status.setText("No plane selected.")
            self._clear_pwv_axes("Select a plane to inspect a cardiac-phase curve.", title="Plane Curve")
            return
        options = self._refresh_plane_curve_metric_options()
        if not options:
            self.label_pwv_status.setText(f"Plane {int(plane_idx)} has no time-resolved metric series.")
            self._clear_pwv_axes("Run plane metrics to populate plane curves.", title="Plane Curve")
            return
        series_key = self.combo_plane_curve_metric.currentData()
        label_map = {key: label for key, label in _PLANE_CURVE_SERIES_OPTIONS}
        series_label = label_map.get(str(series_key), str(series_key or "Metric"))
        values = np.asarray(metric.get(series_key, []), dtype=float).reshape(-1)
        if values.size == 0:
            self.label_pwv_status.setText(f"Plane {int(plane_idx)} has no samples for {series_label}.")
            self._clear_pwv_axes("Selected metric is not available for this plane.", title="Plane Curve")
            return
        self.fig_pwv.clear()
        ax = self.fig_pwv.add_subplot(111)
        phases = np.arange(values.size, dtype=float)
        ax.plot(phases, values, color="#1f77b4", marker="o", linewidth=2.0, markersize=4)
        current_t = int(np.clip(self.workspace.current_t, 0, max(0, values.size - 1)))
        ax.axvline(current_t, color="#f08c00", linestyle="--", linewidth=1.2, alpha=0.8, label="Current Phase")
        ax.scatter([current_t], [values[current_t]], color="#f08c00", zorder=4)
        ax.set_title(f"Plane {int(plane_idx)}: {series_label}")
        ax.set_xlabel("Cardiac Phase")
        ax.set_ylabel(series_label)
        ax.grid(True, alpha=0.25)
        if values.size > 1:
            ax.set_xlim(0.0, float(values.size - 1))
        ax.legend(loc="best")
        self.fig_pwv.tight_layout()
        self.canvas_pwv.draw_idle()

        plane = self.workspace.planes[int(plane_idx)]
        path_idx = int(metric.get("path_index", getattr(plane, "path_index", -1)))
        path_dir = str(metric.get("path_direction", "") or "")
        header = f"Plane {int(plane_idx)}   Path {path_idx}"
        if path_dir:
            header += f"   {path_dir}"
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            finite = np.array([0.0], dtype=float)
        path_ic_value = metric.get("path_ic")
        path_ic_text = "undefined" if path_ic_value is None else f"{float(path_ic_value):.3f}"
        lines = [
            header,
            f"Metric: {series_label}   Frames: {int(values.size)}   Current phase: {current_t}",
            f"Current: {float(values[current_t]):.4g}   Min: {float(np.min(finite)):.4g}   Mean: {float(np.mean(finite)):.4g}   Max: {float(np.max(finite)):.4g}",
            f"Path IC: {path_ic_text}   Net Flow: {float(metric.get('netflow_mL_beat', 0.0)):.4g} mL/beat   Peak Velocity: {float(metric.get('peakv_cm_s', 0.0)):.4g} cm/s",
        ]
        related = []
        for item in list(metric.get("fork_ic", []) or []):
            try:
                ic_value = item.get("ic", 1.0)
                ic_text = "undefined" if ic_value is None else f"{float(ic_value):.3f}"
                related.append(f"fork {int(item.get('fork_id', -1))} ({str(item.get('role', 'path'))}): {ic_text}")
            except Exception:
                continue
        if related:
            lines.append("Related branch IC: " + ", ".join(related))
        self.label_pwv_status.setText("\n".join(lines))

    def _plot_internal_consistency(self):
        qc = dict(self.workspace.derived.plane_qc or {})
        path_idx = self._selected_analysis_path_index()
        if path_idx < 0:
            self.label_ic_target.setText("Target: none")
            self.label_pwv_status.setText("No path or plane selected.")
            self._clear_pwv_axes("Select a path or plane to inspect path and branch internal consistency.", title="Internal Consistency")
            return
        if not qc:
            self.label_ic_target.setText(f"Target: path {int(path_idx)}")
            self.label_pwv_status.setText("No internal consistency results available.")
            self._clear_pwv_axes("Run plane metrics to populate internal consistency.", title="Internal Consistency")
            return
        info = self.workspace.path_info[int(path_idx)]
        self.label_ic_target.setText(f"Target: path {int(path_idx)}")
        path_ic_raw = (qc.get("path_ic", {}) or {}).get(str(int(path_idx)))
        path_ic = None if path_ic_raw is None else float(path_ic_raw)
        qc_forks = {int(item.get("fork_id", -1)): item for item in list(qc.get("forks", []) or [])}
        related_fork_ids = []
        for fork_id in list(info.get("fork_ids", []) or []):
            try:
                fork_id = int(fork_id)
            except Exception:
                continue
            if fork_id in qc_forks:
                related_fork_ids.append(fork_id)
        if not related_fork_ids:
            for fork_id, item in qc_forks.items():
                members = [int(x) for x in item.get("left", [])] + [int(x) for x in item.get("right", [])]
                if int(path_idx) in members:
                    related_fork_ids.append(int(fork_id))
        related_fork_ids = sorted(dict.fromkeys(related_fork_ids))
        labels = []
        values = []
        colors = []
        if path_ic is not None:
            labels.append(f"Path {int(path_idx)}")
            values.append(path_ic)
            colors.append("#1f77b4")
        for fork_id in related_fork_ids:
            item = qc_forks.get(int(fork_id), {})
            fork_ic = item.get("ic", 1.0)
            # Matplotlib bars cannot represent None.  Undefined forks remain
            # visible in the status text below, but are omitted from the bar
            # chart rather than plotted as a misleading zero.
            if fork_ic is not None:
                labels.append(f"Fork {int(fork_id)}")
                values.append(float(fork_ic))
                colors.append("#d9480f")
        self.fig_pwv.clear()
        ax = self.fig_pwv.add_subplot(111)
        xpos = np.arange(len(values), dtype=float)
        bars = ax.bar(xpos, values, color=colors, alpha=0.88)
        ax.set_title(f"Path / Branch Internal Consistency: Path {int(path_idx)}")
        ax.set_ylabel("Internal Consistency")
        ax.set_ylim(0.0, 1.05)
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() * 0.5, min(1.03, value + 0.03), f"{float(value):.3f}", ha="center", va="bottom", fontsize=9)
        self.fig_pwv.tight_layout()
        self.canvas_pwv.draw_idle()

        incoming = [int(x) for x in info.get("incoming_path_ids", [])]
        outgoing = [int(x) for x in info.get("outgoing_path_ids", [])]
        lines = [
            f"Path {int(path_idx)}   dir={str(info.get('direction_text', '') or '')}",
            f"Path IC: {'undefined' if path_ic is None else f'{path_ic:.3f}'}   Incoming: {incoming if incoming else 'none'}   Outgoing: {outgoing if outgoing else 'none'}",
        ]
        if related_fork_ids:
            for fork_id in related_fork_ids:
                item = qc_forks.get(int(fork_id), {})
                fork_ic = item.get("ic", 1.0)
                ic_text = "undefined" if fork_ic is None else f"{float(fork_ic):.3f}"
                lines.append(f"Branch/Fork {int(fork_id)}: IC={ic_text}   left={item.get('left', [])}   right={item.get('right', [])}")
        else:
            lines.append("No related branch junctions were found for the selected path.")
        self.label_pwv_status.setText("\n".join(lines))

    def _refresh_pwv_group_selector(self):
        results = list(self.workspace.derived.pwv_results or [])
        current_name = ""
        current = self._selected_pwv_result()
        if current is not None:
            current_name = str(current.get("name", "") or "")
        self.combo_pwv_group.blockSignals(True)
        self.combo_pwv_group.clear()
        for idx, result in enumerate(results):
            name = str(result.get("name", f"group_{idx}") or f"group_{idx}")
            status = str(result.get("status", "") or "")
            pwv = result.get("pwv_m_s")
            label = name
            if pwv is not None:
                label = f"{name} ({float(pwv):.3g} m/s)"
            elif status:
                label = f"{name} [{status}]"
            self.combo_pwv_group.addItem(label, int(idx))
        self.combo_pwv_group.setEnabled(bool(results))
        if results:
            target_idx = 0
            if current_name:
                for idx, result in enumerate(results):
                    if str(result.get("name", "") or "") == current_name:
                        target_idx = idx
                        break
            self.combo_pwv_group.setCurrentIndex(target_idx)
        self.combo_pwv_group.blockSignals(False)

    def _refresh_analysis_panel(self):
        self._sync_analysis_controls()
        self._refresh_pwv_group_selector()
        self._refresh_centerline_pressure_path_selector()
        mode = self._analysis_mode()
        if mode == "pwv":
            self.label_ic_target.setText("Target: PWV group")
            self._plot_pwv_result(self._selected_pwv_result())
        elif mode == "plane_curve":
            self.label_ic_target.setText("Target: selected plane")
            self._plot_plane_curve()
        elif mode == "centerline_pressure":
            self.label_ic_target.setText("Target: selected path")
            self._plot_centerline_pressure()
        else:
            self._plot_internal_consistency()

    def _refresh_pwv_plot(self):
        self._refresh_plane_curve_metric_options()
        self._refresh_analysis_panel()

    def _build_menu(self):
        mb = self.menuBar()
        mf = mb.addMenu("File")
        for label, slot in [
            ("Open H5", self._on_open_h5),
            ("Import DICOM Directory", self._on_import_dicom_directory),
            ("Clear Workspace", self._on_close_workspace),
            ("Exit", self.close),
        ]:
            a = QtGui.QAction(label, self)
            a.triggered.connect(slot)
            mf.addAction(a)
        me = mb.addMenu("Export")
        a = QtGui.QAction("Export Videos...", self)
        a.triggered.connect(self._on_export_videos)
        me.addAction(a)
        a = QtGui.QAction("Export Plane Coordinates...", self)
        a.triggered.connect(self._on_export_plane_coordinates)
        me.addAction(a)
        a = QtGui.QAction("Export QC Report...", self)
        a.triggered.connect(self._on_export_quality_report)
        me.addAction(a)
        ms = mb.addMenu("Settings")
        a = QtGui.QAction("3D Axis Orientation...", self)
        a.triggered.connect(self._open_display_orientation_settings)
        ms.addAction(a)
        a = QtGui.QAction("3D Background Color...", self)
        a.triggered.connect(self._open_background_color_settings)
        ms.addAction(a)
        mv = mb.addMenu("View")
        for label, slot in [("Reset Camera", lambda: self.scene.reset_camera()), ("Toggle Axes", lambda: self.scene.toggle_axes()),
            ("White BG", lambda: self.scene.set_background("white")), ("Dark BG", lambda: self.scene.set_background("#202124"))]:
            a = QtGui.QAction(label, self)
            a.triggered.connect(slot)
            mv.addAction(a)

    def _open_display_orientation_settings(self):
        if self.scene is None:
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("3D Axis Orientation")
        dialog.setModal(True)
        layout = QtWidgets.QVBoxLayout(dialog)
        form = QtWidgets.QFormLayout()
        combos = []
        for axis, choices in (("X", ("LR", "RL")), ("Y", ("AP", "PA")), ("Z", ("FH", "HF"))):
            combo = QtWidgets.QComboBox()
            combo.addItems(list(choices))
            combos.append(combo)
            form.addRow(f"{axis} positive direction", combo)
        current = self.scene.display_axis_directions()
        for combo, value in zip(combos, current):
            combo.setCurrentText(str(value))
        layout.addLayout(form)
        hint = QtWidgets.QLabel(
            "The displayed 3D scene and orientation marker will be mirrored around the scene center. "
            "Internal data and analysis coordinates are unchanged."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        reset_button = buttons.addButton("Reset to default", QtWidgets.QDialogButtonBox.ResetRole)
        reset_button.clicked.connect(lambda: [combo.setCurrentIndex(0) for combo in combos])
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        directions = [combo.currentText() for combo in combos]
        if self.scene.set_display_axis_directions(directions):
            self.log(f"3D display orientation: {', '.join(directions)}")

    def _open_background_color_settings(self):
        if self.scene is None:
            return
        try:
            current = QtGui.QColor(str(self.scene.plotter.background_color.hex_rgb))
        except Exception:
            current = QtGui.QColor("#000000")
        color = QtWidgets.QColorDialog.getColor(current, self, "3D Background Color")
        if color.isValid():
            value = color.name()
            self.scene.set_background(value)
            self._config_bundle.setdefault("ui", {})["background_color"] = value

    def _refresh_quality_panel(self):
        if not hasattr(self, "quality_table"):
            return
        self.quality_table.clear()
        if not self.workspace.data_loaded:
            self._last_quality_report = None
            self.quality_summary_label.setText("Load a case to generate QC results.")
            self._refresh_workflow_status()
            return
        try:
            report = build_quality_report(
                self.workspace,
                source_path=str(self.workspace.paths.flow_path or self.workspace.paths.segmask_path or ""),
            )
        except Exception as exc:
            self.quality_summary_label.setText(f"QC generation failed: {type(exc).__name__}: {exc}")
            self.log(f"QC ERROR: {traceback.format_exc()}")
            return
        self._last_quality_report = report
        counts = dict(report.get("status_counts", {}) or {})
        self.quality_summary_label.setText(
            f"Overall: {str(report.get('overall_status', 'unknown')).replace('_', ' ').title()} | "
            f"pass={int(counts.get('pass', 0))}, warn={int(counts.get('warn', 0))}, "
            f"fail={int(counts.get('fail', 0))}, not run={int(counts.get('not_run', 0))}"
        )
        status_colors = {
            "pass": "#1b5e20",
            "warn": "#8a4b00",
            "fail": "#b71c1c",
            "not_run": "#616161",
        }
        for check in list(report.get("checks", []) or []):
            status = str(check.get("status", "not_run"))
            item = QtWidgets.QTreeWidgetItem([
                status.replace("_", " ").upper(),
                str(check.get("stage", "")),
                str(check.get("title", check.get("id", ""))),
                str(check.get("summary", "")),
            ])
            color = QtGui.QColor(status_colors.get(status, "#616161"))
            item.setForeground(0, QtGui.QBrush(color))
            action = str(check.get("action", "") or "")
            threshold = str(check.get("threshold", "") or "")
            tooltip = "\n".join(part for part in [threshold, action] if part)
            if tooltip:
                item.setToolTip(2, tooltip)
                item.setToolTip(3, tooltip)
            self.quality_table.addTopLevelItem(item)
        self._refresh_workflow_status()

    def _on_export_quality_report(self):
        if not self.workspace.data_loaded:
            self.log("Load a case before exporting a QC report.")
            return
        if self._last_quality_report is None:
            self._refresh_quality_panel()
        if self._last_quality_report is None:
            return
        default_path = os.path.join(self.pipeline._output_dir(self.workspace), "quality_report.json")
        out_path, _selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export QC Report",
            default_path,
            "JSON (*.json)",
        )
        if not out_path:
            return
        if not out_path.lower().endswith(".json"):
            out_path += ".json"
        try:
            save_quality_report(
                self.workspace,
                out_path,
                source_path=str(self.workspace.paths.flow_path or self.workspace.paths.segmask_path or ""),
                report=self._last_quality_report,
            )
            self.log(f"QC report exported: {out_path}")
            self.statusBar().showMessage(f"QC report exported: {out_path}", 8000)
        except Exception as exc:
            self.log(f"QC EXPORT ERROR: {type(exc).__name__}: {exc}")

    def _selected_plane_indices(self):
        selected = []
        for item in self.tree_objects.selectedItems():
            for leaf in self._iter_browser_leaf_items(item):
                uid = leaf.data(0, QtCore.Qt.UserRole)
                obj = self.workspace.scene_objects.get(uid) if uid else None
                if obj is None or obj.kind != ObjectKind.PLANE:
                    continue
                plane_index = _parse_plane_index(obj.data_key)
                if plane_index is not None:
                    selected.append(int(plane_index))
        return sorted(set(selected))

    def _plane_import_options(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Import Plane Coordinates")
        layout = QtWidgets.QVBoxLayout(dialog)
        form = QtWidgets.QFormLayout()
        mode_combo = QtWidgets.QComboBox()
        mode_combo.addItem("World coordinates", "world")
        mode_combo.addItem("Local physical coordinates", "local")
        mode_combo.addItem("Relative centerline position", "path_relative")
        mode_combo.setItemData(0, "Use only when cases share a registered AutoFlow world-mm coordinate frame.", QtCore.Qt.ToolTipRole)
        mode_combo.setItemData(1, "Use when cases share the same cropped local geometry but have different origins.", QtCore.Qt.ToolTipRole)
        mode_combo.setItemData(2, "Map each generated plane by group, path rank, and fractional centerline distance.", QtCore.Qt.ToolTipRole)
        operation_combo = QtWidgets.QComboBox()
        operation_combo.addItem("Replace existing planes", "replace")
        operation_combo.addItem("Append to existing planes", "append")
        form.addRow("Mapping", mode_combo)
        form.addRow("Operation", operation_combo)
        layout.addLayout(form)
        warning = QtWidgets.QLabel(
            "World coordinates require registered cases. Relative centerline mapping requires target centerlines and is usually safer for unregistered cases."
        )
        warning.setWordWrap(True)
        layout.addWidget(warning)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return None
        return str(mode_combo.currentData()), str(operation_combo.currentData())

    def _on_import_plane_coordinates(self):
        if self._pipeline_running_guard("importing plane coordinates") or self._autoseg_running_guard("importing plane coordinates"):
            return
        if not self.workspace.data_loaded:
            self.log("Load the target case before importing plane coordinates.")
            return
        in_path, _selected_filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import Plane Coordinates",
            "",
            "Plane coordinates (*.json);;JSON (*.json)",
        )
        if not in_path:
            return
        options = self._plane_import_options()
        if options is None:
            return
        mapping_mode, operation = options
        try:
            payload = load_plane_position_payload(in_path)
            imported, report = project_planes_to_workspace(
                payload,
                self.workspace,
                mapping_mode=mapping_mode,
                return_report=True,
            )
            if not imported:
                raise ValueError("the file did not contain any importable planes")
            old_count = len(self.workspace.planes)
            if operation == "append":
                self.workspace.planes.extend(imported)
                first_imported = old_count
            else:
                self.workspace.planes = list(imported)
                first_imported = 0
            self._last_plane_import_report = report
            self._last_quality_report = None
            self.workspace.derived.plane_metrics = []
            self.workspace.derived.plane_qc = {}
            self.pipeline._clear_pwv_state(self.workspace)
            for step in (StepId.COMPUTE_PLANE_METRICS, StepId.COMPUTE_PWV):
                self.workspace.pipeline.completed.pop(step.value, None)
                self.workspace.pipeline.skipped.pop(step.value, None)
            self._rebuild_plane_objects()
            self._refresh_all()
            source_path = str(self.workspace.paths.flow_path or self.workspace.paths.segmask_path or "")
            save_plane_positions(
                self.workspace,
                os.path.join(self.pipeline._output_dir(self.workspace), "plane_positions.json"),
                source_path=source_path,
            )
            self.pipeline._save_planes_json(self.workspace)
            if 0 <= first_imported < len(self.workspace.planes):
                uid = self._find_uid_by_indexed_data_key("plane", first_imported)
                self._selected_plane_index = int(first_imported)
                self.ortho_viewer.set_selected_plane(int(first_imported))
                self._select_browser_item_by_uid(uid)
                self.scene.highlight_plane(uid)
            warning_count = len(list(report.get("warnings", []) or []))
            self.log(
                f"Imported {len(imported)} plane coordinates from {in_path} "
                f"mode={mapping_mode} operation={operation} warnings={warning_count}"
            )
            for message in list(report.get("warnings", []) or []):
                self.log(f"[Plane Import] warning: {message}")
            self._refresh_workflow_status()
        except Exception as exc:
            self.log(f"PLANE IMPORT ERROR: {type(exc).__name__}: {exc}")
            QtWidgets.QMessageBox.critical(self, "Plane Import Failed", f"{type(exc).__name__}: {exc}")

    def _on_export_plane_coordinates(self):
        if not self.workspace.data_loaded or not self.workspace.planes:
            self.log("No planes are available to export.")
            return
        plane_indices = self._selected_plane_indices()
        if not plane_indices:
            plane_indices = list(range(len(self.workspace.planes)))
        default_path = os.path.join(self.pipeline._output_dir(self.workspace), "plane_positions.json")
        out_path, _selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Plane Coordinates",
            default_path,
            "Plane coordinates (*.json)",
        )
        if not out_path:
            return
        if not out_path.lower().endswith(".json"):
            out_path += ".json"
        try:
            save_plane_positions(
                self.workspace,
                out_path,
                source_path=str(self.workspace.paths.flow_path or self.workspace.paths.segmask_path or ""),
                plane_indices=plane_indices,
            )
            self.log(f"Exported {len(plane_indices)} plane coordinates: {out_path}")
            self.statusBar().showMessage(f"Exported {len(plane_indices)} plane coordinates", 8000)
        except Exception as exc:
            self.log(f"PLANE EXPORT ERROR: {type(exc).__name__}: {exc}")

    def _bind_scene(self):
        self.scene = SceneController(self.plotter, self.workspace, self.log)
        background = str(self._config_bundle.get("ui", {}).get("background_color", "#000000") or "#000000")
        self.scene.set_background(background)
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
        self._ensure_segmentation_label_metadata(seg)
        render_cfg = dict(getattr(ws, "render_settings", {}) or {})
        shared_bar_cfg = dict(render_cfg.get("shared_colorbar_bar_cfg", {}) or {})
        show_shared_colorbar = bool(render_cfg.get("shared_colorbar_show", True))
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
                show_scalar_bar=show_shared_colorbar,
                scalar_bar_title="Label",
                scalar_bar_cfg=shared_bar_cfg,
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
        obj.show_scalar_bar = show_shared_colorbar
        obj.scalar_bar_title = "Label"
        obj.scalar_bar_cfg = shared_bar_cfg

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
        labeler_running = (
            self._labeler_process is not None
            and self._labeler_process.state() != QtCore.QProcess.ProcessState.NotRunning
        )
        panel.btn_configure.setEnabled(bool(ws.data_loaded) and not labeler_running)
        panel.btn_run_auto.setEnabled(bool(ws.data_loaded) and not labeler_running and self._autoseg_thread is None)
        panel.btn_import.setEnabled(bool(ws.data_loaded) and not labeler_running)
        panel.btn_save.setEnabled(ws.get_active_segmentation() is not None and not labeler_running)
        panel.btn_external_editor.setEnabled(
            ws.get_active_segmentation() is not None
            and ws.mag_raw is not None
            and ws.flow_raw is not None
            and not labeler_running
        )
        panel.check_cleanup_4d.setEnabled(ws.get_active_segmentation() is not None and not labeler_running)
        panel.combo_cleanup_4d_mode.setEnabled(ws.get_active_segmentation() is not None and not labeler_running)
        panel.spin_cleanup_4d_volume.setEnabled(ws.get_active_segmentation() is not None and not labeler_running)
        panel.btn_apply_cleanup_4d.setEnabled(ws.get_active_segmentation() is not None and not labeler_running)
        panel.btn_external_editor.setText(
            "SpatioTemporal Labeler Open" if labeler_running else "Open in SpatioTemporal Labeler"
        )
        n_labels = 0 if labels is None else len([int(x) for x in np.unique(labels) if int(x) != 0])
        active_src = seg.active_source or "none"
        panel.label_status.setText(f"Source: {active_src}   Labels: {n_labels}")

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
        self._last_quality_report = None
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
        backend_token = str(seg_state.auto_backend or "nnUNet").strip().lower()
        if backend_token not in {"nnunet", "nnunet4d", "nnunet_4d", "4d"}:
            self.log(f"Auto segmentation backend is not supported: {seg_state.auto_backend}")
            return False
        if ws.mag_raw is None or ws.flow_raw is None:
            self.log("Auto segmentation requires loaded mag and flow data.")
            return False
        try:
            resolver = resolve_nnunet_4d_model_folder if backend_token in {"nnunet4d", "nnunet_4d", "4d"} else resolve_nnunet_model_folder
            resolved_model = resolver(seg_state.auto_model)
        except Exception:
            self._on_autoseg_failed(traceback.format_exc())
            return False

        resolved_device = seg_state.auto_device or "cpu"
        checkpoint_name = seg_state.auto_checkpoint or "checkpoint_final.pth"
        model_argument = (
            str(seg_state.auto_model)
            if backend_token in {"nnunet4d", "nnunet_4d", "4d"}
            and str(seg_state.auto_model or "").lower().endswith((".sh", ".bash"))
            else resolved_model
        )
        cache_path = ws.paths.flow_path if str(ws.input_state.source_format or "").lower().endswith("h5") else ""
        artifact_prefix = os.path.splitext(self._default_segmentation_sidecar_path("auto"))[0]
        self._autoseg_started_at = time.perf_counter()
        self._autoseg_progress_dialog = self._create_progress_dialog("Auto Segmentation", "Preparing auto segmentation...")
        self._autoseg_thread = QtCore.QThread(self)
        self._autoseg_worker = _AutoSegmentationWorker(
            ws.mag_raw,
            ws.flow_raw,
            ws.resolution,
            ws.origin,
            model_folder=model_argument,
            backend=seg_state.auto_backend,
            checkpoint_name=checkpoint_name,
            folds=seg_state.auto_folds,
            device=resolved_device,
            auto_label_map=seg_state.auto_label_map,
            cache_path=cache_path,
            artifact_prefix=artifact_prefix,
            source_spatial_order=tuple(str(x).upper() for x in (ws.input_state.metadata.get("spatial_order_raw") or [])),
            source_group=ws.input_state.source_group,
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
            f"Auto segmentation started: backend={seg_state.auto_backend} model={resolved_model} "
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
            if self._autoseg_progress_dialog is not None and self._autoseg_progress_dialog.isVisible():
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
        if dialog is not None and dialog.isVisible():
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
            "autoseg_gpu_preprocessing_fallback",
            "autoseg_read_prediction",
            "autoseg_finalize",
            "autoseg_validate_prediction",
            "autoseg_save_cache",
            "autoseg_complete",
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
        resolved_model = str(result.provenance.get("model_folder") or seg_state.auto_model or "bundled default")
        self._commit_segmentation_source_change(
            "auto",
            f"Auto segmentation ready: nnUNet model={resolved_model} | time={float(result.elapsed_sec):.2f}s",
        )
        if result.cache_path:
            self.log(f"Auto segmentation cached in source h5: {result.cache_path}")
        else:
            self.log("Auto segmentation cache skipped for this input format.")
        feature_files = [str(path) for path in result.provenance.get("feature_files") or [] if str(path).strip()]
        seg_nifti = str(result.provenance.get("segmentation_nifti") or result.provenance.get("prediction_file") or "").strip()
        if feature_files:
            self.log(
                f"Auto segmentation feature NIfTI saved: {len(feature_files)} file(s) in {os.path.dirname(feature_files[0])}"
            )
        if seg_nifti:
            self.log(f"Auto segmentation NIfTI saved: {seg_nifti}")
        self._close_progress_dialog(self._autoseg_progress_dialog)
        self._autoseg_progress_dialog = None

    def _on_autoseg_failed(self, error_text):
        text = str(error_text or "").strip()
        if not text:
            text = "Unknown auto segmentation failure."
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        summary = lines[-1] if lines else "Unknown auto segmentation failure."
        self.log(f"Auto segmentation failed: {summary}")
        self.log(text)
        self._close_progress_dialog(self._autoseg_progress_dialog)
        self._autoseg_progress_dialog = None
        message_box = QtWidgets.QMessageBox(self)
        message_box.setIcon(QtWidgets.QMessageBox.Critical)
        message_box.setWindowTitle("Automatic Segmentation Failed")
        message_box.setText(summary)
        message_box.setInformativeText("No segmentation result was applied. Expand Details for the complete error.")
        message_box.setDetailedText(text)
        message_box.setWindowModality(QtCore.Qt.WindowModal)
        message_box.exec()

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
        if dlg.exec() != QtWidgets.QDialog.Accepted:
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
        seg.auto_folds = values["auto_folds"]
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
            self.log("Automatic segmentation is configured. Click Run Automatic Segmentation to start it.")
        self._refresh_segmentation_ui()

    def _on_run_auto_segmentation(self):
        if self._autoseg_running_guard("starting automatic segmentation"):
            return
        if not self.workspace.data_loaded:
            self.log("Load a case before running automatic segmentation.")
            return
        self.workspace.segmentation.mode = "auto"
        self._run_auto_segmentation()

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
            "Segmentation (*.h5 *.hdf5 *.npy *.npz *.nii *.nii.gz);;All (*)",
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
            "H5 (*.h5 *.hdf5);;NIfTI (*.nii *.nii.gz);;NumPy (*.npy);;NPZ (*.npz)",
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

    def _labeler_exchange_metadata(self, segmentation):
        """Describe a reusable Labeler working directory for the current input."""
        ws = self.workspace
        source_path = str(
            getattr(self._active_input_case, "input_path", "")
            or ws.paths.flow_path
            or ws.paths.segmask_path
            or ""
        )
        source_id = os.path.abspath(source_path) if source_path else "unsaved_workspace"
        return {
            "schema_version": 2,
            "source_id": source_id,
            "input_signature": str(self._input_signature or ""),
            "mag_shape": [int(v) for v in np.asarray(ws.mag_raw).shape],
            "flow_shape": [int(v) for v in np.asarray(ws.flow_raw).shape],
            "segmentation_shape": [int(v) for v in np.asarray(segmentation).shape],
            "resolution": [float(v) for v in np.asarray(ws.resolution, dtype=float).reshape(3)],
            "origin": [float(v) for v in np.asarray(ws.origin, dtype=float).reshape(3)],
            "features": ["mag", "flow_x", "flow_y", "flow_z", "pcmra"],
        }

    def _labeler_exchange_directory(self, metadata):
        output_dir = self.workspace.paths.output_dir or os.path.join(os.getcwd(), "autoflow_out")
        source_id = str(metadata["source_id"])
        source_stem = os.path.splitext(os.path.basename(source_id))[0] or "workspace"
        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", source_stem).strip("._") or "workspace"
        source_hash = hashlib.sha1(source_id.encode("utf-8")).hexdigest()[:12]
        return os.path.join(output_dir, "spatiotemporal_labeler", f"{safe_stem}_{source_hash}")

    @staticmethod
    def _can_reuse_labeler_exchange(exchange_dir, metadata, required_paths):
        manifest_path = os.path.join(exchange_dir, "exchange.json")
        try:
            with open(manifest_path, "r", encoding="utf-8") as handle:
                saved_metadata = json.load(handle)
        except (OSError, ValueError, TypeError):
            return False
        return saved_metadata == metadata and all(os.path.isfile(path) for path in required_paths)

    def _open_external_segmentation_editor(self):
        """Prepare a reusable exchange directory, launch Labeler, and re-import on exit."""
        ws = self.workspace
        segmentation = ws.segmentation_display_4d()
        if segmentation is None or ws.mag_raw is None or ws.flow_raw is None:
            self.log("SpatioTemporal Labeler requires magnitude, flow, and an active segmentation.")
            return
        if (
            self._labeler_process is not None
            and self._labeler_process.state() != QtCore.QProcess.ProcessState.NotRunning
        ):
            self.log("SpatioTemporal Labeler is already open for this AutoFlow session.")
            return
        labeler_spec = importlib.util.find_spec("spatiotemporal_labeler")
        if labeler_spec is None:
            self.log(
                "SpatioTemporal Labeler is not installed in the AutoFlow Python environment. "
                "Install with pip install .[gui,labeler]."
            )
            return
        progress = QtWidgets.QProgressDialog(
            "Preparing SpatioTemporal Labeler...", "", 0, 6, self
        )
        progress.setWindowTitle("Export Labeler Features")
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setValue(0)
        progress.show()
        QtWidgets.QApplication.processEvents()
        try:
            segmentation = np.asarray(segmentation, dtype=np.int16)
            mag = np.asarray(ws.mag_raw, dtype=np.float32)
            flow = np.asarray(ws.flow_raw, dtype=np.float32)
            if segmentation.ndim != 4:
                raise ValueError(f"active segmentation must be XYZT, got shape={segmentation.shape}")
            if mag.ndim == 3:
                mag = np.repeat(mag[..., None], segmentation.shape[3], axis=3)
            if mag.shape != segmentation.shape:
                raise ValueError(
                    "magnitude and segmentation must have matching XYZT shapes, "
                    f"got {mag.shape} and {segmentation.shape}"
                )
            if flow.ndim != 5 or flow.shape[:4] != segmentation.shape or flow.shape[-1] != 3:
                raise ValueError(
                    "flow and segmentation must have matching XYZT dimensions, "
                    f"got flow={flow.shape}, segmentation={segmentation.shape}"
                )
            metadata = self._labeler_exchange_metadata(segmentation)
            exchange_dir = self._labeler_exchange_directory(metadata)
            os.makedirs(exchange_dir, exist_ok=True)
            feature_paths = [
                os.path.join(exchange_dir, "mag.nii"),
                os.path.join(exchange_dir, "flow_x.nii"),
                os.path.join(exchange_dir, "flow_y.nii"),
                os.path.join(exchange_dir, "flow_z.nii"),
                os.path.join(exchange_dir, "pcmra.nii"),
            ]
            label_path = os.path.join(exchange_dir, "segmentation.nii")
            required_paths = [*feature_paths, label_path]
            reuse_exchange = self._can_reuse_labeler_exchange(exchange_dir, metadata, required_paths)
            pcmra = None
            if not reuse_exchange:
                pcmra_3d = compute_reference_scalar(mag, flow, "pcmra")
                pcmra = np.repeat(pcmra_3d[..., None], segmentation.shape[3], axis=3)
            export_steps = [
                ("Magnitude", lambda: save_nifti_volume(
                    feature_paths[0], mag, resolution=ws.resolution, origin=ws.origin
                )),
                ("Flow X (LR)", lambda: save_nifti_volume(
                    feature_paths[1], flow[..., 0], resolution=ws.resolution, origin=ws.origin
                )),
                ("Flow Y (AP)", lambda: save_nifti_volume(
                    feature_paths[2], flow[..., 1], resolution=ws.resolution, origin=ws.origin
                )),
                ("Flow Z (FH)", lambda: save_nifti_volume(
                    feature_paths[3], flow[..., 2], resolution=ws.resolution, origin=ws.origin
                )),
                ("PC-MRA", lambda: save_nifti_volume(
                    feature_paths[4], pcmra,
                    resolution=ws.resolution,
                    origin=ws.origin,
                )),
                ("Segmentation", lambda: save_segmentation_file(
                    label_path,
                    segmentation,
                    resolution=ws.resolution,
                    origin=ws.origin,
                    provenance={
                        "source": "autoflow_spatiotemporal_labeler_exchange",
                        "created_at": segmentation_timestamp(),
                    },
                )),
            ]
            for step_index, (feature_name, writer) in enumerate(export_steps, start=1):
                progress.setLabelText(
                    f"Exporting {feature_name} ({step_index}/{len(export_steps)})..."
                )
                progress.setValue(step_index - 1)
                QtWidgets.QApplication.processEvents()
                if not reuse_exchange:
                    writer()
                progress.setValue(step_index)
                progress.setLabelText(
                    f"{'Reusing' if reuse_exchange else 'Exported'} {feature_name} "
                    f"({step_index}/{len(export_steps)})"
                )
                QtWidgets.QApplication.processEvents()
            if not reuse_exchange:
                with open(os.path.join(exchange_dir, "exchange.json"), "w", encoding="utf-8") as handle:
                    json.dump(metadata, handle, indent=2, sort_keys=True)
        except Exception as exc:
            self._close_progress_dialog(progress)
            self.log(f"Could not launch SpatioTemporal Labeler: {type(exc).__name__}: {exc}")
            return
        self._close_progress_dialog(progress)

        process = QtCore.QProcess(self)
        process.setWorkingDirectory(exchange_dir)
        process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.SeparateChannels)
        environment = QtCore.QProcessEnvironment.systemEnvironment()
        # AutoFlow switches to an EGL-backed off-screen renderer for forwarded
        # X11. Labeler uses VTK's embedded Qt render widget, for which that
        # renderer cannot create the required OpenGL context. Give the separate
        # Labeler process its native X11/GLX environment instead.
        isolated_render_variables = (
            "AUTOFLOW_SSH_RENDERING",
            "PYVISTA_OFF_SCREEN",
            "VTK_DEFAULT_OPENGL_WINDOW",
        )
        removed_render_variables = [
            name for name in isolated_render_variables if environment.contains(name)
        ]
        for name in isolated_render_variables:
            environment.remove(name)
        labeler_source = (
            Path(__file__).resolve().parents[2]
            / "third_party"
            / "SpatioTemporalLabeler"
            / "src"
        )
        if labeler_source.is_dir():
            existing_pythonpath = environment.value("PYTHONPATH", "")
            environment.insert(
                "PYTHONPATH",
                os.pathsep.join(
                    part
                    for part in (str(labeler_source), existing_pythonpath)
                    if part
                ),
            )
        process.setProcessEnvironment(environment)
        # Run the package installed beside AutoFlow, not an arbitrary executable
        # found on PATH. The checked-out v0.4.0 source is selected above while
        # the optional installed dependency supplies Labeler's runtime packages.
        program = sys.executable
        arguments = ["-m", "spatiotemporal_labeler"]
        # Labeler activates the most recently loaded image.  Keep magnitude last
        # so it is the initial editing feature while every image remains 4D.
        launch_image_paths = [
            feature_paths[1], feature_paths[2], feature_paths[3], feature_paths[4], feature_paths[0]
        ]
        for image_path in launch_image_paths:
            arguments.extend(["--image", image_path])
        arguments.extend(["--mask", label_path])
        self._labeler_process_output = {"stdout": "", "stderr": ""}
        process.finished.connect(self._on_external_segmentation_editor_finished)
        process.errorOccurred.connect(self._on_external_segmentation_editor_error)
        process.readyReadStandardOutput.connect(self._on_external_segmentation_editor_stdout)
        process.readyReadStandardError.connect(self._on_external_segmentation_editor_stderr)
        self._labeler_process = process
        self._labeler_segmentation_path = label_path
        self._labeler_segmentation_mtime_ns = os.stat(label_path).st_mtime_ns
        if removed_render_variables:
            self.log(
                "Launching SpatioTemporal Labeler with native embedded VTK rendering "
                f"(cleared: {', '.join(removed_render_variables)})."
            )
        process.start(program, arguments)
        if not process.waitForStarted(5000):
            error_text = process.errorString()
            process.deleteLater()
            self._labeler_process = None
            self._labeler_segmentation_path = ""
            self._labeler_segmentation_mtime_ns = None
            self._refresh_segmentation_ui()
            self.log(f"Could not launch SpatioTemporal Labeler: {error_text}")
            return
        self._refresh_segmentation_ui()
        self.log(
            f"Opened SpatioTemporal Labeler v0.4.0 ({'reused' if reuse_exchange else 'exported'} workspace): {exchange_dir}. "
            "Use Save/Ctrl+S before closing to return the edited segmentation to AutoFlow."
        )

    def _append_labeler_process_output(self, channel, data):
        text = bytes(data).decode("utf-8", errors="replace").strip()
        if not text:
            return
        output = self._labeler_process_output
        existing = str(output.get(channel, ""))
        # Preserve the last diagnostic output without retaining an unbounded log.
        output[channel] = f"{existing}\n{text}"[-12000:]
        for line in text.splitlines():
            self.log(f"[Labeler {channel}] {line}")

    def _on_external_segmentation_editor_stdout(self):
        process = self._labeler_process
        if process is not None:
            self._append_labeler_process_output("stdout", process.readAllStandardOutput())

    def _on_external_segmentation_editor_stderr(self):
        process = self._labeler_process
        if process is not None:
            self._append_labeler_process_output("stderr", process.readAllStandardError())

    def _on_external_segmentation_editor_error(self, _process_error):
        process = self._labeler_process
        if process is not None:
            self._on_external_segmentation_editor_stdout()
            self._on_external_segmentation_editor_stderr()
            self.log(f"SpatioTemporal Labeler process error: {process.errorString()}")

    def _on_external_segmentation_editor_finished(self, exit_code, exit_status):
        process = self._labeler_process
        label_path = self._labeler_segmentation_path
        initial_mtime_ns = self._labeler_segmentation_mtime_ns
        self._on_external_segmentation_editor_stdout()
        self._on_external_segmentation_editor_stderr()
        self._labeler_process = None
        self._labeler_segmentation_path = ""
        self._labeler_segmentation_mtime_ns = None
        if process is not None:
            process.deleteLater()
        self._refresh_segmentation_ui()

        status_name = "normal" if exit_status == QtCore.QProcess.ExitStatus.NormalExit else "crashed"
        self.log(f"SpatioTemporal Labeler closed: exit={int(exit_code)} status={status_name}")
        if status_name == "crashed":
            stderr = str(self._labeler_process_output.get("stderr", "")).strip()
            if stderr:
                self.log(f"Labeler crash diagnostics:\n{stderr}")
            else:
                self.log("Labeler crashed without stderr. Check the AutoFlow log after retrying; child-process diagnostics are now captured there.")
        self._labeler_process_output = {"stdout": "", "stderr": ""}
        try:
            changed = bool(
                label_path
                and os.path.isfile(label_path)
                and os.stat(label_path).st_mtime_ns != initial_mtime_ns
            )
        except OSError as exc:
            self.log(f"Could not inspect Labeler result: {type(exc).__name__}: {exc}")
            return
        if not changed:
            self.log("Labeler segmentation was not saved; the active AutoFlow segmentation is unchanged.")
            return
        choice = QtWidgets.QMessageBox.question(
            self,
            "Apply Labeler Segmentation",
            "SpatioTemporal Labeler saved a modified segmentation. Apply it as the imported source?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.Yes,
        )
        if choice == QtWidgets.QMessageBox.StandardButton.Yes:
            self._import_segmentation_path(label_path)
        else:
            self.log(f"Labeler result kept without applying: {label_path}")

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
            self._refresh_analysis_panel()
            return
        self.workspace.selected_path_index = -1
        self._selected_plane_index = int(plane_idx)
        self.scene.highlight_path(None)
        self.scene.show_forks_for_path(-1)
        self.scene.highlight_plane(uid)
        self.ortho_viewer.set_selected_plane(int(plane_idx))
        self._select_browser_item_by_uid(uid)
        self._clear_plane_drag_widgets()
        self._set_path_info_text("")
        self._log_selected_plane_metric(int(plane_idx))
        self._refresh_analysis_panel()

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
            self._refresh_analysis_panel()
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
        self._refresh_analysis_panel()

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

    def _browser_type_name(self, obj):
        data_key = str(getattr(obj, "data_key", "") or "")
        if data_key == "segmask_raw_surface" or data_key.startswith("segmask_group_"):
            return "Segmentation"
        if data_key == "pcmra_volume":
            return "PC-MRA"
        if data_key == "pwv_planes":
            return "PWV"
        if data_key.startswith("skeleton_") or obj.kind == ObjectKind.SKELETON:
            return "Skeleton"
        if data_key.startswith("graph_") or obj.kind == ObjectKind.GRAPH:
            return "Graph"
        if data_key.startswith("forks_"):
            return "Forks"
        if data_key.startswith("smooth_path_") or obj.kind == ObjectKind.BRANCH:
            return "Paths"
        if obj.kind == ObjectKind.PLANE:
            return "Planes"
        if data_key.startswith("pathline_"):
            return "Pathlines"
        if data_key == "streamlines_live":
            return "Streamlines"
        if data_key == "wss_surface_live":
            return "WSS"
        if data_key == "tke_volume":
            return "TKE"
        if data_key == "vorticity_magnitude_volume":
            return "Vorticity Magnitude"
        if data_key == "q_criterion_volume":
            return "Q-Criterion"
        if data_key == "swirling_strength_volume":
            return "Swirling Strength"
        if data_key == "pressure_gradient_volume":
            return "Pressure Gradient"
        if data_key == "relative_pressure_volume":
            return "Relative Pressure"
        if obj.kind == ObjectKind.METRIC:
            return "Metrics"
        if obj.kind == ObjectKind.FLOW:
            return "Flow"
        if obj.kind == ObjectKind.AUX:
            return "Aux"
        return str(obj.kind.value)

    def _browser_type_sort_key(self, type_name):
        order = [
            "PC-MRA",
            "Segmentation",
            "Skeleton",
            "Graph",
            "Forks",
            "Paths",
            "Unbound Planes",
            "Planes",
            "Pathlines",
            "Streamlines",
            "PWV",
            "WSS",
            "TKE",
            "Vorticity Magnitude",
            "Q-Criterion",
            "Swirling Strength",
            "Pressure Gradient",
            "Relative Pressure",
            "Metrics",
            "Flow",
            "Aux",
        ]
        try:
            return (order.index(str(type_name)), str(type_name).lower())
        except ValueError:
            return (len(order), str(type_name).lower())

    def _browser_path_index_for_object(self, obj):
        """Return the path associated with a scene object, if any.

        Planes and pathlines are keyed by their plane index, while paths carry
        their own path index.  The helper deliberately resolves this from the
        workspace rather than changing SceneObject or PlaneData semantics.
        """
        if obj is None:
            return None
        data_key = str(getattr(obj, "data_key", "") or "")
        if data_key.startswith("smooth_path_") or obj.kind == ObjectKind.BRANCH:
            return _parse_path_index(data_key)
        if obj.kind == ObjectKind.PLANE:
            plane_idx = _parse_plane_index(data_key)
        elif data_key.startswith("pathline_"):
            plane_idx = _parse_pathline_index(data_key)
        else:
            return None
        if plane_idx is None or not (0 <= int(plane_idx) < len(self.workspace.planes)):
            return None
        path_idx = int(getattr(self.workspace.planes[int(plane_idx)], "path_index", -1))
        return path_idx if path_idx >= 0 else None

    def _browser_plane_group_name(self, plane_idx):
        if not (0 <= int(plane_idx) < len(self.workspace.planes)):
            return "Global"
        plane_group = str(getattr(self.workspace.planes[int(plane_idx)], "group_name", "") or "")
        return plane_group if plane_group else "Global"

    def _browser_plane_path_key(self, plane_idx):
        if not (0 <= int(plane_idx) < len(self.workspace.planes)):
            return None
        plane = self.workspace.planes[int(plane_idx)]
        path_idx = int(getattr(plane, "path_index", -1))
        if path_idx < 0:
            return None
        return self._browser_plane_group_name(plane_idx), path_idx

    def _browser_path_label(self, group_name, path_idx):
        path_idx = int(path_idx)
        count = sum(
            1
            for plane in self.workspace.planes
            if int(getattr(plane, "path_index", -1)) == path_idx
            and str(getattr(plane, "group_name", "") or "") == ("" if group_name == "Global" else str(group_name))
        )
        suffix = "plane" if count == 1 else "planes"
        return f"Path {path_idx} · {count} {suffix}"

    def _browser_make_checkable(self, item, checked=True):
        item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
        item.setCheckState(0, QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)
        return item

    def _browser_add_leaf(self, parent, obj, *, name=None):
        item = QtWidgets.QTreeWidgetItem([str(name or obj.name), obj.kind.value])
        tooltip = str(name or obj.name)
        if str(getattr(obj, "data_key", "")) == "pcmra_volume":
            tooltip += "\nShift+left-drag in 3D: window width / window level"
        item.setToolTip(0, tooltip)
        item.setToolTip(1, obj.kind.value)
        item.setData(0, QtCore.Qt.UserRole, obj.uid)
        self._browser_make_checkable(item, bool(obj.visible))
        parent.addChild(item)
        return item

    def _browser_add_path_node(self, group_entry, path_idx):
        path_idx = int(path_idx)
        key = (str(group_entry["name"]), path_idx)
        existing = group_entry["paths"].get(key)
        if existing is not None:
            return existing
        type_item = group_entry["types"].setdefault("Paths", None)
        if type_item is None:
            type_item = QtWidgets.QTreeWidgetItem(["Paths", "Type"])
            type_item.setToolTip(0, "Paths")
            type_item.setToolTip(1, "Type")
            self._browser_make_checkable(type_item)
            group_entry["types"]["Paths"] = type_item
            group_entry["item"].addChild(type_item)
        path_item = QtWidgets.QTreeWidgetItem([self._browser_path_label(group_entry["name"], path_idx), "Path"])
        path_item.setToolTip(0, self._browser_path_label(group_entry["name"], path_idx))
        path_item.setToolTip(1, "Path and associated planes")
        path_item.setData(0, _BROWSER_NODE_ROLE, "path")
        path_item.setData(0, _BROWSER_PATH_INDEX_ROLE, path_idx)
        path_item.setData(0, _BROWSER_GROUP_ROLE, str(group_entry["name"]))
        self._browser_make_checkable(path_item)
        type_item.addChild(path_item)
        group_entry["paths"][key] = path_item
        return path_item

    def _browser_path_node_for_object(self, group_entry, obj):
        path_idx = self._browser_path_index_for_object(obj)
        if path_idx is None:
            return None
        return self._browser_add_path_node(group_entry, path_idx)

    def _browser_sort_path_children(self, path_item):
        children = [path_item.child(i) for i in range(path_item.childCount())]
        def sort_key(item):
            uid = item.data(0, QtCore.Qt.UserRole)
            obj = self.workspace.scene_objects.get(uid) if uid is not None else None
            if obj is not None and (obj.kind == ObjectKind.BRANCH or str(obj.data_key).startswith("smooth_path_")):
                return (0, -1.0, str(item.text(0)).lower())
            if obj is not None and obj.kind == ObjectKind.PLANE:
                plane_idx = _parse_plane_index(obj.data_key)
                distance = float(getattr(self.workspace.planes[int(plane_idx)], "distance", 0.0)) if plane_idx is not None and 0 <= int(plane_idx) < len(self.workspace.planes) else 0.0
                return (1, distance, int(plane_idx) if plane_idx is not None else 10**9)
            return (2, 0.0, str(item.text(0)).lower())
        children.sort(key=sort_key)
        for idx, child in enumerate(children):
            path_item.removeChild(child)
            path_item.insertChild(idx, child)
        path_item.setCheckState(0, self._browser_check_state_for_item(path_item))

    def _browser_sort_path_nodes(self, type_item):
        nodes = [type_item.child(i) for i in range(type_item.childCount())]
        def sort_key(item):
            raw = item.data(0, _BROWSER_PATH_INDEX_ROLE)
            try:
                return (0, int(raw))
            except (TypeError, ValueError):
                return (1, str(item.text(0)).lower())
        nodes.sort(key=sort_key)
        for idx, node in enumerate(nodes):
            type_item.removeChild(node)
            type_item.insertChild(idx, node)
            self._browser_sort_path_children(node)

    def _iter_browser_leaf_items(self, item):
        if item is None:
            return
        uid = item.data(0, QtCore.Qt.UserRole)
        if uid is not None:
            yield item
        for i in range(item.childCount()):
            yield from self._iter_browser_leaf_items(item.child(i))

    def _browser_check_state_for_item(self, item):
        leaves = list(self._iter_browser_leaf_items(item))
        if not leaves:
            return QtCore.Qt.Unchecked
        checked = sum(1 for leaf in leaves if leaf.checkState(0) == QtCore.Qt.Checked)
        if checked == len(leaves):
            return QtCore.Qt.Checked
        if checked == 0:
            return QtCore.Qt.Unchecked
        return QtCore.Qt.PartiallyChecked

    def _sync_browser_parent_states(self, item):
        parent = item.parent()
        while parent is not None:
            parent.setCheckState(0, self._browser_check_state_for_item(parent))
            parent = parent.parent()

    def _set_browser_item_visibility(self, item, visible):
        target_state = QtCore.Qt.Checked if visible else QtCore.Qt.Unchecked
        refresh_segmentation = False
        uid = item.data(0, QtCore.Qt.UserRole)
        item.setCheckState(0, target_state)
        if uid is not None:
            obj = self.workspace.scene_objects.get(uid)
            if obj is not None:
                obj.visible = bool(visible)
                if obj.data_key == "segmask_raw_surface":
                    self.workspace.segmentation.visible = bool(visible)
                    refresh_segmentation = True
                self.scene.apply_object_properties(obj, render=False, refresh_scalar_bar=False)
        for i in range(item.childCount()):
            refresh_segmentation = self._set_browser_item_visibility(item.child(i), visible) or refresh_segmentation
        return refresh_segmentation

    def _find_browser_item_by_uid(self, uid):
        if uid is None:
            return None

        def _search(node):
            if node.data(0, QtCore.Qt.UserRole) == uid:
                return node
            for idx in range(node.childCount()):
                found = _search(node.child(idx))
                if found is not None:
                    return found
            return None

        for i in range(self.tree_objects.topLevelItemCount()):
            found = _search(self.tree_objects.topLevelItem(i))
            if found is not None:
                return found
        return None

    def _collect_browser_uids(self, item):
        return [leaf.data(0, QtCore.Qt.UserRole) for leaf in self._iter_browser_leaf_items(item) if leaf.data(0, QtCore.Qt.UserRole) is not None]

    def _plane_widget_distance(self):
        spacing = self._get_spacing_xyz_from_resolution()
        return max(5.0, float(np.mean(spacing)) * 8.0)

    def _refresh_plane_edit_controls(self):
        selected = 0 <= int(self._selected_plane_index) < len(self.workspace.planes)
        editing = bool(self._plane_edit_enabled and selected)
        if hasattr(self, "btn_edit_plane"):
            self.btn_add_plane.setEnabled(bool(self.workspace.data_loaded))
            self.btn_edit_plane.setEnabled(selected)
            self.btn_edit_plane.setText("Finish Plane Edit" if editing else "Edit Plane")

    def _add_manual_plane(self):
        if self._pipeline_running_guard("adding a plane") or self._autoseg_running_guard("adding a plane"):
            return
        if not self.workspace.data_loaded:
            self.log("Load a case before adding a plane.")
            return
        if self._plane_edit_enabled and 0 <= int(self._selected_plane_index) < len(self.workspace.planes):
            self._finalize_plane_drag(int(self._selected_plane_index))
            self._clear_plane_drag_widgets()

        spacing = self._get_spacing_xyz_from_resolution()
        cursor = np.asarray(self.workspace.ortho_cursor, dtype=float).reshape(3)
        plane = PlaneData(
            center=cursor * spacing,
            normal=np.array([1.0, 0.0, 0.0], dtype=float),
            label=0,
            path_index=-1,
            distance=0.0,
            group_name="",
        )
        old_count = len(self.workspace.planes)
        self.workspace.planes.append(plane)
        if len(self.workspace.derived.plane_metrics) == old_count:
            self.workspace.derived.plane_metrics.append({})
        else:
            self.workspace.derived.plane_metrics = []
        self.workspace.derived.plane_qc = {}
        self._rebuild_plane_objects()
        self._refresh_browser()

        plane_idx = len(self.workspace.planes) - 1
        uid = self._find_uid_by_indexed_data_key("plane", plane_idx)
        self._selected_plane_index = plane_idx
        self.workspace.selected_path_index = -1
        self.ortho_viewer.set_selected_plane(plane_idx)
        self.scene.highlight_plane(uid)
        self._select_browser_item_by_uid(uid)
        self._refresh_selection_info()
        try:
            self.pipeline._save_planes_json(self.workspace)
        except Exception as exc:
            self.log(f"Manual plane save failed: {type(exc).__name__}: {exc}")
        self._activate_plane_drag_widgets(plane_idx)
        self.log(f"Added manual plane {plane_idx} at the current ortho cursor")

    def _toggle_selected_plane_edit(self):
        plane_idx = int(self._selected_plane_index)
        if not (0 <= plane_idx < len(self.workspace.planes)):
            self.log("Select a plane before editing it.")
            return
        if self._pipeline_running_guard("editing a plane") or self._autoseg_running_guard("editing a plane"):
            return
        if self._plane_edit_enabled:
            self._finalize_plane_drag(plane_idx)
            self._clear_plane_drag_widgets()
            self.log(f"Plane {plane_idx} edit finished")
            return
        self._activate_plane_drag_widgets(plane_idx)
        if self._plane_edit_enabled:
            self.log(f"Editing plane {plane_idx}")

    def _clear_plane_drag_widgets(self):
        self._plane_drag_active = False
        self._plane_drag_index = None
        self._plane_widget_initializing = False
        self._plane_drag_metrics_dirty = False
        self._plane_edit_enabled = False
        self._plane_center_widget = None
        self._plane_axis_u_widget = None
        self._plane_axis_v_widget = None
        self._plane_axis_u = None
        self._plane_axis_v = None
        self._refresh_plane_edit_controls()
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
            path_idx = int(getattr(plane, "path_index", -1))
            paths = (
                self.workspace.centerline_paths_smooth
                if len(self.workspace.centerline_paths_smooth) > 0
                else self.workspace.centerline_paths
            )
            if 0 <= path_idx < len(paths) and len(np.asarray(paths[path_idx]).reshape(-1, 3)) > 0:
                c, _segment_idx, _alpha, distance, _error, _cum = _project_point_to_path(
                    paths[path_idx], c
                )
                plane.distance = float(distance)
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
        self._last_quality_report = None
        uid = self._find_uid_by_indexed_data_key("plane", int(plane_idx))
        if uid is not None:
            self.scene.update_plane_geometry(uid, render=False)
        self._selected_plane_index = int(plane_idx)
        self.ortho_viewer._selected_plane_idx = int(plane_idx)
        self._plane_drag_index = int(plane_idx)
        self._plane_drag_metrics_dirty = True
        if self._plane_edit_enabled:
            self._plane_widget_initializing = True
            try:
                current_center = (
                    np.asarray(plane.center, dtype=float).reshape(3)
                    + np.asarray(self.workspace.origin, dtype=float).reshape(3)
                )
                current_center = self.scene.world_to_display_point(current_center)
                current_normal = np.asarray(plane.normal, dtype=float).reshape(3)
                current_normal = current_normal / (np.linalg.norm(current_normal) + 1e-12)
                if self._plane_axis_u is None or self._plane_axis_v is None:
                    self._plane_axis_u, self._plane_axis_v = _plane_in_plane_axes(current_normal)
                axis_u = np.asarray(self._plane_axis_u, dtype=float).reshape(3)
                axis_v = np.asarray(self._plane_axis_v, dtype=float).reshape(3)
                distance = self._plane_widget_distance()
                if self._plane_center_widget is not None:
                    self._plane_center_widget.SetCenter(*current_center.tolist())
                if self._plane_axis_u_widget is not None:
                    self._plane_axis_u_widget.SetCenter(*(current_center + axis_u * distance).tolist())
                if self._plane_axis_v_widget is not None:
                    self._plane_axis_v_widget.SetCenter(*(current_center + axis_v * distance).tolist())
            finally:
                self._plane_widget_initializing = False
        try:
            self.plotter.render()
        except Exception:
            pass

    def _persist_plane_outputs(self, include_pixelwise=True):
        out_dir = self.pipeline._output_dir(self.workspace)
        metrics = self.workspace.derived.plane_metrics
        qc = self.workspace.derived.plane_qc
        if metrics and len(metrics) == len(self.workspace.planes):
            with open(os.path.join(out_dir, "plane_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
        if qc:
            with open(os.path.join(out_dir, "plane_qc.json"), "w", encoding="utf-8") as f:
                json.dump(qc, f, ensure_ascii=False, indent=2)
        if include_pixelwise and metrics and len(metrics) == len(self.workspace.planes):
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
                    relative_pressure_array=self.workspace.derived.relative_pressure_array,
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
        self._plane_drag_index = int(plane_idx)
        uid = self._find_uid_by_indexed_data_key("plane", int(plane_idx))
        if uid is not None:
            self.scene.update_plane_geometry(uid, render=False)
        if int(plane_idx) in self.workspace.active_pathline_plane_indices:
            self.workspace.pathline_cache.pop(int(plane_idx), None)
            self.scene.invalidate_cache("pathline_")
            pathline_uid = self._find_uid_by_indexed_data_key("pathline", int(plane_idx))
            if pathline_uid is not None:
                pathline_obj = self.workspace.scene_objects.get(pathline_uid)
                if pathline_obj is not None:
                    self.scene.readd_object(pathline_obj, refresh_scalar_bar=False)
        self.ortho_viewer.set_selected_plane(int(plane_idx))
        if self.workspace.flow_raw is None or self.workspace.segmask_binary is None:
            self._persist_plane_outputs(include_pixelwise=False)
            self._plane_drag_metrics_dirty = False
            return
        if self._plane_drag_metrics_dirty or len(self.workspace.derived.plane_metrics) != len(self.workspace.planes):
            self._recompute_dragged_plane_metrics(persist=True)
        else:
            self._persist_plane_outputs(include_pixelwise=False)

    def _activate_plane_drag_widgets(self, plane_idx):
        if self._edit_mode is not None or not (0 <= int(plane_idx) < len(self.workspace.planes)):
            return
        if self._plane_drag_active and int(self._plane_drag_index) == int(plane_idx):
            return
        self._clear_plane_drag_widgets()
        plane = self.workspace.planes[int(plane_idx)]
        origin = np.asarray(self.workspace.origin, dtype=float).reshape(3)
        center = np.asarray(plane.center, dtype=float).reshape(3) + origin
        normal = np.asarray(plane.normal, dtype=float).reshape(3)
        if np.linalg.norm(normal) <= 1e-12:
            normal = np.array([1.0, 0.0, 0.0], dtype=float)
        normal = normal / np.linalg.norm(normal)
        axis_u, axis_v = _plane_in_plane_axes(normal)
        display_center = self.scene.world_to_display_point(center)
        display_axis_u = self.scene.world_to_display_vector(axis_u)
        display_axis_v = self.scene.world_to_display_vector(axis_v)
        self._plane_axis_u = axis_u
        self._plane_axis_v = axis_v
        distance = self._plane_widget_distance()
        tip_u = display_center + display_axis_u * distance
        tip_v = display_center + display_axis_v * distance
        radius = self._edit_widget_radius()

        def _center_cb(new_center):
            self._update_plane_from_drag(
                plane_idx,
                center=self.scene.display_to_world_point(new_center) - origin,
            )

        def _axis_u_cb(new_tip):
            if self._plane_axis_v is None:
                return
            c = self.scene.world_to_display_point(np.asarray(self.workspace.planes[int(plane_idx)].center, dtype=float).reshape(3) + origin)
            proposed = np.asarray(new_tip, dtype=float).reshape(3) - c
            current_v = self.scene.world_to_display_vector(self._plane_axis_v)
            proposed = proposed - np.dot(proposed, current_v) * current_v
            if np.linalg.norm(proposed) <= 1e-12:
                return
            self._plane_axis_u = proposed / np.linalg.norm(proposed)
            new_normal = np.cross(self._plane_axis_u, current_v)
            self._update_plane_from_drag(plane_idx, normal=self.scene.display_to_world_vector(new_normal))

        def _axis_v_cb(new_tip):
            if self._plane_axis_u is None:
                return
            c = self.scene.world_to_display_point(np.asarray(self.workspace.planes[int(plane_idx)].center, dtype=float).reshape(3) + origin)
            proposed = np.asarray(new_tip, dtype=float).reshape(3) - c
            current_u = self.scene.world_to_display_vector(self._plane_axis_u)
            proposed = proposed - np.dot(proposed, current_u) * current_u
            if np.linalg.norm(proposed) <= 1e-12:
                return
            self._plane_axis_v = proposed / np.linalg.norm(proposed)
            new_normal = np.cross(current_u, self._plane_axis_v)
            self._update_plane_from_drag(plane_idx, normal=self.scene.display_to_world_vector(new_normal))

        def _end_cb(_widget, _event):
            self._finalize_plane_drag(plane_idx)

        try:
            self._plane_widget_initializing = True
            center_widget = self.plotter.add_sphere_widget(
                callback=_center_cb,
                center=tuple(display_center.tolist()),
                radius=radius,
                color="cyan",
                interaction_event="always",
            )
            axis_u_widget = self.plotter.add_sphere_widget(
                callback=_axis_u_cb,
                center=tuple(tip_u.tolist()),
                radius=radius,
                color="orange",
                interaction_event="always",
            )
            axis_v_widget = self.plotter.add_sphere_widget(
                callback=_axis_v_cb,
                center=tuple(tip_v.tolist()),
                radius=radius,
                color="yellow",
                interaction_event="always",
            )
            center_widget.AddObserver(_vtk.vtkCommand.EndInteractionEvent, _end_cb)
            axis_u_widget.AddObserver(_vtk.vtkCommand.EndInteractionEvent, _end_cb)
            axis_v_widget.AddObserver(_vtk.vtkCommand.EndInteractionEvent, _end_cb)
            self._plane_center_widget = center_widget
            self._plane_axis_u_widget = axis_u_widget
            self._plane_axis_v_widget = axis_v_widget
            self._plane_axis_u = axis_u
            self._plane_axis_v = axis_v
            self._plane_drag_active = True
            self._plane_drag_index = int(plane_idx)
            self._plane_edit_enabled = True
        except Exception as e:
            self._plane_drag_active = False
            self._plane_drag_index = None
            self.log(f"Plane drag widget error: {type(e).__name__}: {e}")
        finally:
            self._plane_widget_initializing = False
            self._refresh_plane_edit_controls()

    def _recompute_dragged_plane_metrics(self, persist=False):
        if self._plane_drag_index is None or not (0 <= int(self._plane_drag_index) < len(self.workspace.planes)):
            return
        if self.workspace.flow_raw is None or self.workspace.segmask_binary is None:
            self.ortho_viewer.refresh()
            if persist:
                self._persist_plane_outputs(include_pixelwise=False)
            self._plane_drag_metrics_dirty = False
            return
        plane_idx = int(self._plane_drag_index)
        try:
            include_derived = bool(
                (self.workspace.derived.wss_surfaces and self.workspace.derived.wss_volume is not None)
                or self.workspace.derived.tke_array is not None
                or self.workspace.derived.pressure_gradient_array is not None
                or self.workspace.derived.relative_pressure_array is not None
            )
            if len(self.workspace.derived.plane_metrics) != len(self.workspace.planes):
                self.pipeline._compute_plane_metrics_internal(
                    self.workspace,
                    save=False,
                    include_derived=include_derived,
                    ensure_derived=False,
                )
            else:
                paths_for_tangent = (
                    self.workspace.centerline_paths_smooth
                    if len(self.workspace.centerline_paths_smooth) > 0
                    else self.workspace.centerline_paths
                )
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
                if partial_metrics and include_derived:
                    partial_metrics, _ = augment_plane_metrics_with_derived(
                        partial_metrics,
                        [self.workspace.planes[plane_idx]],
                        self.workspace.segmask_binary,
                        self.workspace.resolution,
                        self.workspace.origin,
                        branch_labels_3d=self.workspace.branch_labels,
                        tke_array=self.workspace.derived.tke_array,
                        pressure_gradient_array=self.workspace.derived.pressure_gradient_array,
                        relative_pressure_array=self.workspace.derived.relative_pressure_array,
                        wss_surfaces=self.workspace.derived.wss_surfaces,
                    )
                if partial_metrics:
                    partial_metrics[0]["plane_index"] = plane_idx
                    metrics = [dict(m) for m in self.workspace.derived.plane_metrics]
                    metrics[plane_idx] = dict(partial_metrics[0])
                    metrics, qc = apply_internal_consistency_to_metrics(metrics, path_info=self.workspace.path_info, forks=self.workspace.forks)
                    self.workspace.derived.plane_metrics = metrics
                    self.workspace.derived.plane_qc = qc
                    for i, metric in enumerate(metrics):
                        if i < len(self.workspace.planes):
                            self.workspace.planes[i].metrics = dict(metric)
            if persist:
                self._persist_plane_outputs(include_pixelwise=False)
            self._selected_plane_index = plane_idx
            self.ortho_viewer._selected_plane_idx = plane_idx
            self.ortho_viewer.refresh()
            self._log_selected_plane_metric(plane_idx)
            self._refresh_analysis_panel()
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
        path_ic_value = metric.get("path_ic")
        path_ic_text = "undefined" if path_ic_value is None else f"{float(path_ic_value):.3f}"
        text_block = (
            f"{header}\n"
            f"t={t}  Flow Rate={flow_t:.4f} mL/s  Area={area_t:.3f} mm^2  Mean Velocity={meanv_t:.3f} cm/s\n"
            f"Peak Velocity={float(metric.get('peakv_cm_s', 0.0)):.3f} cm/s  Net Flow={float(metric.get('netflow_mL_beat', 0.0)):.4f} mL/beat  IC={path_ic_text}"
        )
        self._set_plane_info_text(text_block)

    def _log_selected_path_info(self, path_idx):
        if not (0 <= int(path_idx) < len(self.workspace.path_info)):
            self._set_path_info_text("")
            return
        info = self.workspace.path_info[int(path_idx)]
        # Plane generation records the segmentation owner selected for each
        # path.  Prefer that value so the GUI reports the same label used by
        # the segmentation filter (including paths that cross a fork).  The
        # nested ``plane_layout`` form is written after plane metrics; the
        # top-level form is present immediately after plane generation.
        owner_label = 0
        plane_qc = getattr(getattr(self.workspace, "derived", None), "plane_qc", {})
        if isinstance(plane_qc, dict):
            layout_paths = plane_qc.get("paths")
            if not isinstance(layout_paths, list):
                layout = plane_qc.get("plane_layout", {})
                layout_paths = layout.get("paths") if isinstance(layout, dict) else None
            for item in layout_paths or []:
                if not isinstance(item, dict) or int(item.get("path_index", -1)) != int(path_idx):
                    continue
                try:
                    owner_label = int(item.get("owner_label", 0) or 0)
                except (TypeError, ValueError):
                    owner_label = 0
                break
        if owner_label <= 0:
            # A plane can retain the owner label even when the aggregate QC is
            # not available (for example, immediately after a workspace load).
            values = []
            for plane in getattr(self.workspace, "planes", []):
                if int(getattr(plane, "path_index", -1)) != int(path_idx):
                    continue
                try:
                    value = int(getattr(plane, "segmentation_label", 0) or 0)
                except (TypeError, ValueError):
                    value = 0
                if value > 0:
                    values.append(value)
            if values:
                owner_label = int(max(set(values), key=values.count))
        if owner_label <= 0 and getattr(self.workspace, "segmask_labels_3d", None) is not None:
            # Before planes are generated, derive the same owner label on
            # demand so path inspection remains informative after only the
            # graph step.  This does not modify the path or workspace masks.
            paths = getattr(self.workspace, "centerline_paths", [])
            if 0 <= int(path_idx) < len(paths):
                try:
                    _unused_paths, fallback_qc = filter_paths_by_segmentation(
                        [paths[int(path_idx)]],
                        self.workspace.segmask_labels_3d,
                        spacing=self.workspace.resolution,
                        origin=self.workspace.origin,
                        path_info=[info],
                        inter_time=int(getattr(self.workspace.plane_gen_params, "inter_time", 10) or 10),
                    )
                    owner_label = int((fallback_qc[0] if fallback_qc else {}).get("owner_label", 0) or 0)
                except Exception:
                    owner_label = 0
        # Always prefer a fresh evaluation of the currently displayed graph
        # path.  A workspace can contain planes/QC imported from an earlier
        # skeleton run whose path numbering or topology no longer matches the
        # current graph; those cached labels must not override the current
        # segmentation assignment shown to the user.
        paths = getattr(self.workspace, "centerline_paths", [])
        if getattr(self.workspace, "segmask_labels_3d", None) is not None and 0 <= int(path_idx) < len(paths):
            try:
                _current_paths, current_qc = filter_paths_by_segmentation(
                    [paths[int(path_idx)]],
                    self.workspace.segmask_labels_3d,
                    spacing=self.workspace.resolution,
                    origin=self.workspace.origin,
                    path_info=[info],
                    inter_time=int(getattr(self.workspace.plane_gen_params, "inter_time", 10) or 10),
                )
                current_owner = int((current_qc[0] if current_qc else {}).get("owner_label", 0) or 0)
                if current_owner > 0:
                    owner_label = current_owner
            except Exception:
                pass
        label_map = dict(getattr(getattr(self.workspace, "skeleton_params", None), "label_map", {}) or {})
        id_to_name = {}
        for name, value in label_map.items():
            try:
                id_to_name[int(value)] = str(name)
            except (TypeError, ValueError):
                continue
        owner_name = id_to_name.get(owner_label, "unknown") if owner_label > 0 else "unknown"
        incoming = [int(x) for x in info.get("incoming_path_ids", [])]
        outgoing = [int(x) for x in info.get("outgoing_path_ids", [])]
        forks = []
        for fork in self.workspace.forks:
            if int(path_idx) in fork.get("left", []) or int(path_idx) in fork.get("right", []):
                forks.append(f"node={int(fork.get('node', -1))} L={fork.get('left', [])} R={fork.get('right', [])}")
        fork_txt = " ; ".join(forks) if forks else "none"
        text_block = (
            f"Path {int(path_idx)} | label={owner_name} (id={owner_label}) | dir={info.get('direction_text', '')}\n"
            f"start_node={int(info.get('start_node', -1))}  end_node={int(info.get('end_node', -1))}\n"
            f"incoming: {incoming if incoming else 'none'}  outgoing: {outgoing if outgoing else 'none'}\n"
            f"forks: {fork_txt}"
        )
        self._set_path_info_text(text_block)

    def _select_browser_item_by_uid(self, uid):
        self.tree_objects.blockSignals(True)
        item = self._find_browser_item_by_uid(uid)
        if item is not None:
            parent = item.parent()
            while parent is not None:
                parent.setExpanded(True)
                parent = parent.parent()
            self.tree_objects.setCurrentItem(item)
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
        self._refresh_plane_edit_controls()
        self._refresh_analysis_panel()

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
                if message and dialog.isVisible():
                    dialog.setLabelText(message)
                if message:
                    self.log(f"[{log_prefix}] {message}")
                QtWidgets.QApplication.processEvents()
                return
            stage = str(payload.get("stage", "") or "")
            current = payload.get("current")
            total = payload.get("total")
            message = str(payload.get("message", "") or stage or log_prefix)
            if dialog.isVisible():
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

    def _set_background_phase_method(self, method):
        token = str(method or "wrls_arto").strip().lower().replace("-", "_").replace("+", "_")
        token = {"wrls": "wrls_arto", "arto": "wrls_arto", "wrlsarto": "wrls_arto"}.get(token, token)
        index = self.combo_bpc_method.findData(token)
        self.combo_bpc_method.setCurrentIndex(index if index >= 0 else 0)

    def _on_bpc_method_changed(self, *_args):
        is_msac = str(self.combo_bpc_method.currentData() or "wrls_arto") == "msac"
        self.spin_bpc_threshold.setEnabled(is_msac)

    def _on_phase_unwrap_method_changed(self, *_args):
        """Show only the controls belonging to the selected unwrap method."""
        if not hasattr(self, "phase_unwrap_method_stack"):
            return
        method = str(self.combo_phase_unwrap_method.currentData() or "gc3D")
        page = {"gc3D": 0, "lap4D": 1, "nprs": 2}.get(method, 0)
        self.phase_unwrap_method_stack.setCurrentIndex(page)

    def _sync_params_to_ws(self):
        ws = self.workspace
        if hasattr(self, "combo_phase_unwrap_method"):
            # Selecting a method is the opt-in action.  The step itself still
            # runs only when the user presses "Unwrap Phase" (or explicitly
            # requests it from the CLI/configuration).
            ws.phase_unwrap_params.enabled = True
            ws.phase_unwrap_params.method = str(self.combo_phase_unwrap_method.currentData() or "gc3D")
            ws.phase_unwrap_params.mask_source = str(self.combo_phase_unwrap_mask.currentData() or "segmentation")
            ws.phase_unwrap_params.device = str(self.combo_phase_unwrap_device.currentData() or "auto")
            ws.phase_unwrap_params.lap4d_ts = self._float_from_text(self.edit_phase_unwrap_ts.text(), 2.0)
            ws.phase_unwrap_params.nprs_upsampling_factor = max(1, self._int_from_text(self.edit_phase_unwrap_nprs_up.text(), 2))
            ws.phase_unwrap_params.nprs_pi_unwrap = bool(self.chk_phase_unwrap_nprs_pi.isChecked())
            ws.phase_unwrap_params.nprs_auto_crop = bool(self.chk_phase_unwrap_nprs_crop.isChecked())
        ws.loader_params.background_phase_correction.enabled = self.chk_bpc_enabled.isChecked()
        ws.loader_params.background_phase_correction.method = str(
            self.combo_bpc_method.currentData() or "wrls_arto"
        )
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
        ws.skeleton_params.separate_special_label_contacts = self.chk_separate_special_label_contacts.isChecked()
        ws.skeleton_params.cc_filter_mode = str(self.combo_cc_filter_mode.currentText().strip() or "hybrid")
        ws.skeleton_params.min_cc_volume_mm3 = self._float_from_text(self.edit_min_cc_volume.text(), 50.0)
        ws.skeleton_params.cc_rel_min_ratio = self._float_from_text(self.edit_cc_rel_min_ratio.text(), 0.01)
        ws.skeleton_params.do_closing = self.chk_closing.isChecked()
        ws.skeleton_params.do_opening = self.chk_opening.isChecked()
        ws.skeleton_params.gaussian_enabled = self.chk_gaussian.isChecked()
        ws.skeleton_params.gaussian_sigma = self._float_from_text(self.edit_gauss_sigma.text(), 0.5)
        if hasattr(self, "segmentation_panel"):
            seg_ui = self.segmentation_panel
            ws.segmentation.cleanup_4d_components = bool(seg_ui.check_cleanup_4d.isChecked())
            ws.segmentation.cleanup_4d_mode = str(seg_ui.combo_cleanup_4d_mode.currentData() or "absolute")
            ws.segmentation.cleanup_4d_min_volume_mm3 = float(seg_ui.spin_cleanup_4d_volume.value())
        ws.plane_gen_params.plane_mode = str(self.combo_plane_mode.currentData() or "fixed_step")
        plane_count = self._int_from_text(self.edit_plane_count.text(), 3)
        ws.plane_gen_params.plane_count = -1 if plane_count == -1 else max(plane_count, 1)
        ws.plane_gen_params.cross_section_distance = self._float_from_text(self.edit_plane_dist.text(), 5.0)
        ws.plane_gen_params.start_distance = self._float_from_text(self.edit_plane_start.text(), 0.0)
        ws.plane_gen_params.end_distance = self._float_from_text(self.edit_plane_end.text(), 0.0)
        ws.plane_gen_params.anchor = str(self.combo_plane_anchor.currentData() or "center")
        ws.plane_gen_params.anchor_offset_mm = self._float_from_text(self.edit_plane_offset.text(), 5.0)
        ws.plane_gen_params.direction = str(self.combo_plane_direction.currentData() or "both")
        ws.plane_gen_params.spacing_mode = str(self.combo_plane_spacing_mode.currentData() or "fraction")
        ws.plane_gen_params.spacing_ratio = self._float_from_text(self.edit_plane_ratio.text(), 0.25)
        ws.plane_gen_params.segmentation_filter = bool(self.chk_plane_seg_filter.isChecked())
        ws.plane_gen_params.smoothing_window = self._int_from_text(self.edit_plane_smooth_win.text(), 15)
        ws.plane_gen_params.smoothing_polyorder = self._int_from_text(self.edit_plane_smooth_poly.text(), 3)
        ws.plane_gen_params.inter_time = self._int_from_text(self.edit_plane_inter_time.text(), 10)
        pwv_payload = {
            "enabled": True,
            "groups": self._pwv_group_payload_from_ui(),
            "plane_interval_mm": max(self._float_from_text(self.edit_pwv_interval.text(), 10.0), 0.1),
            "start_distance": self._float_from_text(self.edit_pwv_start.text(), 0.0),
            "end_distance": self._float_from_text(self.edit_pwv_end.text(), 0.0),
            "smoothing_window": max(self._int_from_text(self.edit_pwv_smooth_win.text(), 15), 1),
            "smoothing_polyorder": max(self._int_from_text(self.edit_pwv_smooth_poly.text(), 2), 0),
            "inter_time": max(self._int_from_text(self.edit_pwv_inter_time.text(), 10), 1),
            "waveform_key": str(self.edit_pwv_waveform.text().strip() or "flowrate_mL_s"),
            "transit_time_method": str(self.combo_pwv_tt_method.currentText().strip() or "foot_to_foot"),
            "foot_method": str(self.combo_pwv_foot_method.currentText().strip() or "tangent"),
            "foot_savgol_window": max(self._int_from_text(self.edit_pwv_foot_win.text(), 5), 1),
            "foot_savgol_polyorder": max(self._int_from_text(self.edit_pwv_foot_poly.text(), 2), 0),
            "foot_threshold_percent": min(max(self._float_from_text(self.edit_pwv_foot_threshold.text(), 10.0), 0.0), 100.0),
            "xcorr_window": str(self.combo_pwv_xcorr_window.currentText().strip() or "full"),
            "xcorr_interp_factor": max(self._int_from_text(self.edit_pwv_xcorr_interp.text(), 10), 1),
            "allow_cycle_wrap": self.chk_pwv_allow_wrap.isChecked(),
            "minimum_valid_planes": max(self._int_from_text(self.edit_pwv_min_planes.text(), 2), 2),
            "scene_visible": self.chk_pwv_scene_visible.isChecked(),
            "scene_color": str(self.edit_pwv_scene_color.text().strip() or "#ffd43b"),
            "plot_color": str(self.edit_pwv_plot_color.text().strip() or "#2b8a3e"),
            "fit_color": str(self.edit_pwv_fit_color.text().strip() or "#f08c00"),
            "plot_dpi": max(self._int_from_text(self.edit_pwv_plot_dpi.text(), 160), 72),
        }
        ws.pwv_params = PwvParams.from_dict(pwv_payload, label_map=ws.label_params.label_map)
        ws.streamline_params.seed_ratio = min(max(self._float_from_text(self.edit_sl_ratio.text(), 0.02), 0.0001), 1.0)
        ws.streamline_params.max_steps = min(max(self._int_from_text(self.edit_sl_maxsteps.text(), 2000), 1), 200000)
        ws.streamline_params.terminal_speed = min(max(self._float_from_text(self.edit_sl_terminal.text(), 0.01), 0.0), 1e6)
        ws.streamline_params.pathline_seed_ratio = min(max(self._float_from_text(self.edit_pathline_ratio.text(), 0.2), 0.0001), 1.0)
        ws.streamline_params.pathline_max_steps = min(max(self._int_from_text(self.edit_pathline_maxsteps.text(), 200), 1), 200000)
        ws.streamline_params.pathline_min_seeds = min(max(self._int_from_text(self.edit_pathline_minseeds.text(), 50), 1), 100000)
        ws.streamline_params.pathline_seed_mode = str(self.combo_pathline_seed_mode.currentData() or "fixed")
        ws.streamline_params.pathline_max_seeds = min(max(self._int_from_text(self.edit_pathline_max_seeds.text(), 250), 1), 100000)
        ws.streamline_params.pathline_terminal_speed = min(max(self._float_from_text(self.edit_pathline_terminal.text(), 0.01), 0.0), 1e6)
        ws.streamline_params.pathline_rng_seed = self._int_from_text(self.edit_pathline_rng_seed.text(), 0)
        ws.streamline_params.pathline_tube_radius = min(max(self._float_from_text(self.edit_pathline_tube_radius.text(), 0.25), 0.001), 1e4)
        ws.streamline_params.pathline_temporal_cache_mb = min(max(self._float_from_text(self.edit_pathline_temporal_cache_mb.text(), 512.0), 0.0), 8192.0)
        ws.streamline_params.pathline_color = str(self.edit_pathline_color.text().strip() or "deepskyblue")
        ws.streamline_params.pathline_color_mode = str(self.combo_pathline_color_mode.currentData() or "per_plane")
        ws.derived_params.smoothing_iteration = max(self._int_from_text(self.edit_dm_smoothing.text(), 200), 0)
        ws.derived_params.viscosity = max(self._float_from_text(self.edit_dm_viscosity.text(), 4.0), 0.0)
        inward_distance = self._optional_float_from_text(self.edit_dm_inward.text(), None)
        ws.derived_params.inward_distance = None if inward_distance is None else max(inward_distance, 0.01)
        ws.derived_params.parabolic_fitting = self.chk_dm_parabolic.isChecked()
        ws.derived_params.no_slip_condition = self.chk_dm_noslip.isChecked()
        ws.derived_params.rho = max(self._float_from_text(self.edit_dm_rho.text(), 1060.0), 1.0)
        pressure_method = str(self.combo_dm_pressure_method.currentText().strip() or "least_squares").lower()
        if pressure_method not in {"least_squares", "ppe"}:
            pressure_method = "least_squares"
        ws.derived_params.pressure_method = pressure_method
        ws.derived_params.pressure_gradient_smoothing_sigma = max(self._float_from_text(self.edit_dm_pg_smoothing_sigma.text(), 0.0), 0.0)
        ws.derived_params.pressure_gradient_support_erosion_iters = max(self._int_from_text(self.edit_dm_pg_support_erosion.text(), 1), 0)
        ws.derived_params.pressure_gradient_layer_opacity = min(max(self._float_from_text(self.edit_dm_pg_opacity.text(), 0.6), 0.0), 1.0)
        ws.derived_params.relative_pressure_layer_opacity = min(max(self._float_from_text(self.edit_dm_rp_opacity.text(), 0.6), 0.0), 1.0)
        ws.derived_params.vortex_smoothing_sigma = max(self._float_from_text(self.edit_dm_vortex_smoothing_sigma.text(), 0.0), 0.0)
        ws.derived_params.vortex_support_erosion_iters = max(self._int_from_text(self.edit_dm_vortex_support_erosion.text(), 1), 0)
        ws.derived_params.use_multithread = self.chk_dm_multithread.isChecked()
        if hasattr(self, "runtime_render_controls"):
            default_cfg = bundle_to_autoflow_kwargs(self._config_bundle)
            render_cfg = dict(getattr(ws, "render_settings", {}) or default_cfg)
            for metric_key, cfg_key in self._runtime_render_clim_keys().items():
                widgets = self.runtime_render_controls.get(metric_key, {})
                render_cfg[cfg_key] = self._parse_optional_clim_pair(
                    widgets.get("min").text() if widgets.get("min") is not None else "",
                    widgets.get("max").text() if widgets.get("max") is not None else "",
                    fallback=default_cfg.get(cfg_key),
                )
            width = max(self._float_from_text(self.edit_runtime_scalar_bar_width.text(), 0.08), 0.01)
            height = max(self._float_from_text(self.edit_runtime_scalar_bar_height.text(), 0.65), 0.05)
            gap = max(self._float_from_text(self.edit_runtime_scalar_bar_gap.text(), 0.03), 0.0)
            pos_x = min(max(self._float_from_text(self.edit_runtime_scalar_bar_pos_x.text(), 0.87), 0.0), 0.98)
            pos_y = min(max(self._float_from_text(self.edit_runtime_scalar_bar_pos_y.text(), 0.15), 0.0), 0.95)
            title_font = max(self._int_from_text(self.edit_runtime_scalar_bar_title_font.text(), 14), 1)
            label_font = max(self._int_from_text(self.edit_runtime_scalar_bar_label_font.text(), 11), 1)
            shared_bar_cfg = dict(render_cfg.get("shared_colorbar_bar_cfg", {}) or {})
            shared_bar_cfg.update({
                "width": float(width),
                "height": float(height),
                "position_x": float(pos_x),
                "position_y": float(pos_y),
                "stack_gap": float(gap),
                "title_font_size": int(title_font),
                "label_font_size": int(label_font),
            })
            render_cfg["shared_colorbar_bar_cfg"] = shared_bar_cfg
            for bar_key in self._runtime_bar_cfg_keys():
                bar_cfg = dict(render_cfg.get(bar_key, {}) or {})
                bar_cfg["width"] = float(width)
                bar_cfg["height"] = float(height)
                bar_cfg["position_x"] = float(pos_x)
                bar_cfg["position_y"] = float(pos_y)
                bar_cfg["stack_gap"] = float(gap)
                bar_cfg["title_font_size"] = int(title_font)
                bar_cfg["label_font_size"] = int(label_font)
                render_cfg[bar_key] = bar_cfg
            ws.render_settings = render_cfg

    def _sync_params_to_ui(self):
        ws = self.workspace
        if hasattr(self, "combo_phase_unwrap_method"):
            idx = self.combo_phase_unwrap_method.findData(str(ws.phase_unwrap_params.method))
            # Older workspaces used method="none" plus an Enable checkbox.
            # Select the first real backend when opening those workspaces;
            # unwrapping remains an explicit step in the GUI.
            self.combo_phase_unwrap_method.setCurrentIndex(max(0, idx))
            idx = self.combo_phase_unwrap_mask.findData(str(ws.phase_unwrap_params.mask_source))
            self.combo_phase_unwrap_mask.setCurrentIndex(max(0, idx))
            idx = self.combo_phase_unwrap_device.findData(str(ws.phase_unwrap_params.device))
            self.combo_phase_unwrap_device.setCurrentIndex(max(0, idx))
            self.edit_phase_unwrap_ts.setText(str(ws.phase_unwrap_params.lap4d_ts))
            self.edit_phase_unwrap_nprs_up.setText(str(ws.phase_unwrap_params.nprs_upsampling_factor))
            self.chk_phase_unwrap_nprs_pi.setChecked(bool(ws.phase_unwrap_params.nprs_pi_unwrap))
            self.chk_phase_unwrap_nprs_crop.setChecked(bool(ws.phase_unwrap_params.nprs_auto_crop))
            self._on_phase_unwrap_method_changed()
        self.chk_bpc_enabled.setChecked(ws.loader_params.background_phase_correction.enabled)
        self._set_background_phase_method(ws.loader_params.background_phase_correction.method)
        self.spin_bpc_fit_order.setValue(int(ws.loader_params.background_phase_correction.corr_fit_order))
        self.spin_bpc_threshold.setValue(float(ws.loader_params.background_phase_correction.threshold))
        self.spin_dual_venc_ratio1.setValue(float(ws.loader_params.background_phase_correction.dual_venc_ratio1))
        self.spin_dual_venc_ratio2.setValue(float(ws.loader_params.background_phase_correction.dual_venc_ratio2))
        self.edit_input_resolution.setText(", ".join(f"{float(x):.6g}" for x in np.asarray(ws.resolution, dtype=float).reshape(-1)[:3]))
        self.edit_input_venc.setText(", ".join(f"{float(x):.6g}" for x in np.asarray(ws.venc, dtype=float).reshape(-1)[:3]))
        self.edit_input_spatial_order.setText(", ".join(str(x) for x in ws.spatial_order[:3]))
        self.edit_input_venc_order.setText(", ".join(str(x) for x in ws.venc_order[:3]))
        self.chk_remove_small_cc.setChecked(ws.skeleton_params.remove_small_cc)
        self.chk_separate_special_label_contacts.setChecked(bool(getattr(ws.skeleton_params, "separate_special_label_contacts", True)))
        self.combo_cc_filter_mode.setCurrentText(str(getattr(ws.skeleton_params, "cc_filter_mode", "hybrid") or "hybrid"))
        self.edit_min_cc_volume.setText(str(ws.skeleton_params.min_cc_volume_mm3))
        self.edit_cc_rel_min_ratio.setText(str(getattr(ws.skeleton_params, "cc_rel_min_ratio", 0.01)))
        self.chk_closing.setChecked(ws.skeleton_params.do_closing)
        self.chk_opening.setChecked(ws.skeleton_params.do_opening)
        self.chk_gaussian.setChecked(ws.skeleton_params.gaussian_enabled)
        self.edit_gauss_sigma.setText(str(ws.skeleton_params.gaussian_sigma))
        if hasattr(self, "segmentation_panel"):
            seg_ui = self.segmentation_panel
            seg_ui.check_cleanup_4d.setChecked(bool(ws.segmentation.cleanup_4d_components))
            seg_ui.combo_cleanup_4d_mode.setCurrentIndex(max(seg_ui.combo_cleanup_4d_mode.findData(str(ws.segmentation.cleanup_4d_mode)), 0))
            seg_ui.spin_cleanup_4d_volume.setValue(float(ws.segmentation.cleanup_4d_min_volume_mm3))
        mode = str(getattr(ws.plane_gen_params, "plane_mode", "fixed_step") or "fixed_step")
        idx = max(self.combo_plane_mode.findData(mode), 0)
        self.combo_plane_mode.setCurrentIndex(idx)
        self.edit_plane_count.setText(str(getattr(ws.plane_gen_params, "plane_count", 3)))
        self.edit_plane_dist.setText(str(ws.plane_gen_params.cross_section_distance))
        self.edit_plane_start.setText(str(ws.plane_gen_params.start_distance))
        self.edit_plane_end.setText(str(ws.plane_gen_params.end_distance))
        self.combo_plane_anchor.setCurrentIndex(max(self.combo_plane_anchor.findData(str(getattr(ws.plane_gen_params, "anchor", "center") or "center")), 0))
        self.edit_plane_offset.setText(str(getattr(ws.plane_gen_params, "anchor_offset_mm", 5.0)))
        self.combo_plane_direction.setCurrentIndex(max(self.combo_plane_direction.findData(str(getattr(ws.plane_gen_params, "direction", "both") or "both")), 0))
        self.combo_plane_spacing_mode.setCurrentIndex(max(self.combo_plane_spacing_mode.findData(str(getattr(ws.plane_gen_params, "spacing_mode", "fraction") or "fraction")), 0))
        self.edit_plane_ratio.setText(str(getattr(ws.plane_gen_params, "spacing_ratio", 0.25)))
        self.chk_plane_seg_filter.setChecked(bool(getattr(ws.plane_gen_params, "segmentation_filter", True)))
        self.edit_plane_smooth_win.setText(str(ws.plane_gen_params.smoothing_window))
        self.edit_plane_smooth_poly.setText(str(ws.plane_gen_params.smoothing_polyorder))
        self.edit_plane_inter_time.setText(str(ws.plane_gen_params.inter_time))
        self._sync_plane_mode_ui()
        self.edit_pwv_interval.setText(str(ws.pwv_params.plane_interval_mm))
        self.edit_pwv_start.setText(str(ws.pwv_params.start_distance))
        self.edit_pwv_end.setText(str(ws.pwv_params.end_distance))
        self.edit_pwv_smooth_win.setText(str(ws.pwv_params.smoothing_window))
        self.edit_pwv_smooth_poly.setText(str(ws.pwv_params.smoothing_polyorder))
        self.edit_pwv_inter_time.setText(str(ws.pwv_params.inter_time))
        self.edit_pwv_waveform.setText(str(ws.pwv_params.waveform_key))
        self.combo_pwv_tt_method.setCurrentText(str(ws.pwv_params.transit_time_method))
        self.combo_pwv_foot_method.setCurrentText(str(ws.pwv_params.foot_method))
        self.edit_pwv_foot_win.setText(str(ws.pwv_params.foot_savgol_window))
        self.edit_pwv_foot_poly.setText(str(ws.pwv_params.foot_savgol_polyorder))
        self.edit_pwv_foot_threshold.setText(str(ws.pwv_params.foot_threshold_percent))
        self.combo_pwv_xcorr_window.setCurrentText(str(ws.pwv_params.xcorr_window))
        self.edit_pwv_xcorr_interp.setText(str(ws.pwv_params.xcorr_interp_factor))
        self.chk_pwv_allow_wrap.setChecked(bool(ws.pwv_params.allow_cycle_wrap))
        self.edit_pwv_min_planes.setText(str(ws.pwv_params.minimum_valid_planes))
        self.chk_pwv_scene_visible.setChecked(bool(ws.pwv_params.scene_visible))
        self.edit_pwv_scene_color.setText(str(ws.pwv_params.scene_color))
        self.edit_pwv_plot_color.setText(str(ws.pwv_params.plot_color))
        self.edit_pwv_fit_color.setText(str(ws.pwv_params.fit_color))
        self.edit_pwv_plot_dpi.setText(str(ws.pwv_params.plot_dpi))
        self.table_pwv_groups.blockSignals(True)
        self.table_pwv_groups.setRowCount(0)
        for group in list(ws.pwv_params.groups or []):
            self._add_pwv_group_row(name=str(group.name), labels=self._pwv_group_labels_for_ui(group.labels))
        self.table_pwv_groups.clearSelection()
        self.table_pwv_groups.blockSignals(False)
        self.edit_sl_ratio.setText(str(ws.streamline_params.seed_ratio))
        self.edit_sl_maxsteps.setText(str(ws.streamline_params.max_steps))
        self.edit_sl_terminal.setText(str(ws.streamline_params.terminal_speed))
        self.combo_pathline_seed_mode.setCurrentIndex(max(self.combo_pathline_seed_mode.findData(str(getattr(ws.streamline_params, "pathline_seed_mode", "fixed") or "fixed")), 0))
        self.edit_pathline_max_seeds.setText(str(getattr(ws.streamline_params, "pathline_max_seeds", 250)))
        self.edit_pathline_ratio.setText(str(getattr(ws.streamline_params, "pathline_seed_ratio", 0.2)))
        self.edit_pathline_minseeds.setText(str(getattr(ws.streamline_params, "pathline_min_seeds", 50)))
        self.edit_pathline_maxsteps.setText(str(getattr(ws.streamline_params, "pathline_max_steps", 200)))
        self.edit_pathline_terminal.setText(str(getattr(ws.streamline_params, "pathline_terminal_speed", 0.01)))
        self.edit_pathline_rng_seed.setText(str(getattr(ws.streamline_params, "pathline_rng_seed", 0)))
        self.edit_pathline_tube_radius.setText(str(getattr(ws.streamline_params, "pathline_tube_radius", 0.25)))
        self.edit_pathline_temporal_cache_mb.setText(str(getattr(ws.streamline_params, "pathline_temporal_cache_mb", 512.0)))
        self.edit_pathline_color.setText(str(ws.streamline_params.pathline_color))
        self.combo_pathline_color_mode.setCurrentIndex(max(self.combo_pathline_color_mode.findData(str(getattr(ws.streamline_params, "pathline_color_mode", "per_plane") or "per_plane")), 0))
        self.edit_dm_smoothing.setText(str(ws.derived_params.smoothing_iteration))
        self.edit_dm_viscosity.setText(str(ws.derived_params.viscosity))
        self.edit_dm_inward.setText("auto" if ws.derived_params.inward_distance is None else str(ws.derived_params.inward_distance))
        self.chk_dm_parabolic.setChecked(ws.derived_params.parabolic_fitting)
        self.chk_dm_noslip.setChecked(ws.derived_params.no_slip_condition)
        self.edit_dm_rho.setText(str(ws.derived_params.rho))
        self.combo_dm_pressure_method.setCurrentText(str(ws.derived_params.pressure_method))
        self.edit_dm_pg_smoothing_sigma.setText(str(ws.derived_params.pressure_gradient_smoothing_sigma))
        self.edit_dm_pg_support_erosion.setText(str(ws.derived_params.pressure_gradient_support_erosion_iters))
        self.edit_dm_pg_opacity.setText(str(ws.derived_params.pressure_gradient_layer_opacity))
        self.edit_dm_rp_opacity.setText(str(ws.derived_params.relative_pressure_layer_opacity))
        self.edit_dm_vortex_smoothing_sigma.setText(str(ws.derived_params.vortex_smoothing_sigma))
        self.edit_dm_vortex_support_erosion.setText(str(ws.derived_params.vortex_support_erosion_iters))
        self.chk_dm_multithread.setChecked(ws.derived_params.use_multithread)
        render_cfg = dict((getattr(ws, "render_settings", {}) or {}) or self._rendering_kwargs())
        for metric_key, cfg_key in self._runtime_render_clim_keys().items():
            widgets = getattr(self, "runtime_render_controls", {}).get(metric_key, {})
            value = render_cfg.get(cfg_key)
            min_widget = widgets.get("min")
            max_widget = widgets.get("max")
            if min_widget is None or max_widget is None:
                continue
            min_widget.blockSignals(True)
            max_widget.blockSignals(True)
            if value is None:
                min_widget.setText("auto")
                max_widget.setText("auto")
            else:
                min_widget.setText(f"{float(value[0]):.6g}")
                max_widget.setText(f"{float(value[1]):.6g}")
            min_widget.blockSignals(False)
            max_widget.blockSignals(False)
        shared_bar_cfg = dict(render_cfg.get("shared_colorbar_bar_cfg", {}) or {})
        for widget, key, default in [
            (self.edit_runtime_scalar_bar_width, "width", 0.08),
            (self.edit_runtime_scalar_bar_height, "height", 0.65),
            (self.edit_runtime_scalar_bar_gap, "stack_gap", 0.03),
            (self.edit_runtime_scalar_bar_pos_x, "position_x", 0.87),
            (self.edit_runtime_scalar_bar_pos_y, "position_y", 0.15),
            (self.edit_runtime_scalar_bar_title_font, "title_font_size", 14),
            (self.edit_runtime_scalar_bar_label_font, "label_font_size", 11),
        ]:
            widget.blockSignals(True)
            widget.setText(f"{float(shared_bar_cfg.get(key, default)):.6g}")
            widget.blockSignals(False)

    def _rebuild_plane_objects(self):
        self._clear_plane_drag_widgets()
        ws = self.workspace
        plane_render_cfg = dict((getattr(ws, "render_settings", {}) or {}).get("plane_render_cfg", {}) or {})
        default_cfg = plane_render_cfg.get("default", {}) if isinstance(plane_render_cfg.get("default"), dict) else {}
        groups_cfg = plane_render_cfg.get("groups", {}) if isinstance(plane_render_cfg.get("groups"), dict) else {}
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
            group_render_cfg = groups_cfg.get(group_name, {}) if group_name and isinstance(groups_cfg.get(group_name), dict) else {}
            merged_render_cfg = dict(default_cfg)
            merged_render_cfg.update(group_render_cfg)
            plane_color = str(merged_render_cfg.get("plane_color", "") or "")
            if not plane_color:
                plane_color = ws.skeleton_params.scene_color_for_group(group_name, "plane") if group_name else "yellow"
            try:
                plane_opacity = float(merged_render_cfg.get("plane_opacity", 0.75))
            except Exception:
                plane_opacity = 0.75
            plane_opacity = max(0.0, min(1.0, plane_opacity))
            if group_name in ws.multilabel_groups:
                state = ws.multilabel_groups[group_name]
                if group_name not in seen_groups:
                    state["plane_index_offset"] = int(i)
                    seen_groups.add(group_name)
                state.setdefault("planes", []).append(plane)
                ws.multilabel_groups[group_name] = state
            ws.add_object(
                name=f"plane {int(i)}",
                kind=ObjectKind.PLANE,
                data_key=self._plane_data_key("plane", i),
                group_name=group_name,
                browser_color=ws.skeleton_params.browser_color_for_group(group_name) if group_name else "",
                visible=True,
                opacity=plane_opacity,
                color=plane_color,
                line_width=2,
            )
        self.scene.invalidate_cache("plane_")
        self.scene.invalidate_cache("pathline_")
        self.scene.sync_from_workspace()

    def _refresh_all(self):
        self._sync_segmentation_scene_object()
        self._refresh_browser()
        self._refresh_render_range_control()
        self._refresh_timeline()
        self._sync_params_to_ui()
        self._refresh_selection_info()
        self._refresh_segmentation_ui()
        self._refresh_pwv_plot()
        self._refresh_workflow_status()
        self._refresh_scene()

    def _refresh_browser(self):
        self.tree_objects.blockSignals(True)
        self.tree_objects.clear()
        groups = {}
        path_objects = []
        plane_objects = []
        pathline_objects = []

        def ensure_group(obj, group_name=None):
            group_name = str(group_name or self._browser_group_name(obj))
            if group_name in groups:
                return groups[group_name]
            top = QtWidgets.QTreeWidgetItem([group_name, "Group"])
            top.setToolTip(0, group_name)
            top.setToolTip(1, "Group")
            self._browser_make_checkable(top)
            color_name = str(getattr(obj, "browser_color", "") or "")
            if color_name:
                color = QtGui.QColor(color_name)
                if color.isValid():
                    top.setForeground(0, QtGui.QBrush(color))
            entry = {"name": group_name, "item": top, "types": {}, "paths": {}}
            groups[group_name] = entry
            self.tree_objects.addTopLevelItem(top)
            return entry

        def ensure_type(group_entry, type_name):
            type_map = group_entry["types"]
            if type_name in type_map:
                return type_map[type_name]
            type_item = QtWidgets.QTreeWidgetItem([type_name, "Type"])
            type_item.setToolTip(0, type_name)
            type_item.setToolTip(1, "Type")
            self._browser_make_checkable(type_item)
            type_map[type_name] = type_item
            group_entry["item"].addChild(type_item)
            return type_item

        # Create the ordinary group/type rows first, then build the path tree
        # so planes and pathlines can be attached to their owning path even if
        # SceneObjects were inserted in a different order.
        for obj in self.workspace.scene_objects.values():
            if obj.data_key == "branch_surface":
                continue
            if obj.kind == ObjectKind.BRANCH or str(obj.data_key).startswith("smooth_path_"):
                path_objects.append(obj)
                ensure_group(obj)
            elif obj.kind == ObjectKind.PLANE:
                plane_objects.append(obj)
                ensure_group(obj)
            elif str(obj.data_key).startswith("pathline_"):
                pathline_objects.append(obj)
                ensure_group(obj)
            else:
                group_entry = ensure_group(obj)
                self._browser_add_leaf(ensure_type(group_entry, self._browser_type_name(obj)), obj)

        # Path geometry is the anchor row for each path.  A path node can still
        # be created when only planes/pathlines exist (for example after an
        # imported plane file), which keeps the relationship visible.
        for obj in path_objects:
            group_entry = ensure_group(obj)
            path_idx = self._browser_path_index_for_object(obj)
            if path_idx is None:
                self._browser_add_leaf(ensure_type(group_entry, "Paths"), obj)
                continue
            self._browser_add_leaf(self._browser_add_path_node(group_entry, path_idx), obj)

        plane_items = {}
        for obj in plane_objects:
            plane_idx = _parse_plane_index(obj.data_key)
            group_entry = ensure_group(obj)
            path_key = self._browser_plane_path_key(plane_idx) if plane_idx is not None else None
            if path_key is None:
                plane_item = self._browser_add_leaf(ensure_type(group_entry, "Unbound Planes"), obj)
            else:
                path_group, path_idx = path_key
                # Use the plane's group as the authoritative grouping key even
                # if a stale SceneObject group_name is present.
                group_entry = ensure_group(obj, path_group)
                plane_item = self._browser_add_leaf(self._browser_add_path_node(group_entry, path_idx), obj)
            if plane_idx is not None:
                plane_items[int(plane_idx)] = plane_item

        for obj in pathline_objects:
            plane_idx = _parse_pathline_index(obj.data_key)
            path_group = self._browser_plane_group_name(int(plane_idx)) if plane_idx is not None else None
            group_entry = ensure_group(obj, path_group) if path_group else ensure_group(obj)
            plane_item = plane_items.get(int(plane_idx)) if plane_idx is not None else None
            if plane_item is not None:
                self._browser_add_leaf(plane_item, obj)
                continue
            path_idx = self._browser_path_index_for_object(obj)
            if path_idx is not None:
                self._browser_add_leaf(self._browser_add_path_node(group_entry, path_idx), obj)
            else:
                self._browser_add_leaf(ensure_type(group_entry, "Pathlines"), obj)

        for group_entry in groups.values():
            top = group_entry["item"]
            type_items = list(group_entry["types"].items())
            type_items.sort(key=lambda pair: self._browser_type_sort_key(pair[0]))
            for idx, (_type_name, type_item) in enumerate(type_items):
                top.removeChild(type_item)
                top.insertChild(idx, type_item)
                if _type_name == "Paths":
                    self._browser_sort_path_nodes(type_item)
                type_item.setCheckState(0, self._browser_check_state_for_item(type_item))
            top.setCheckState(0, self._browser_check_state_for_item(top))
        # Keep the useful navigation levels visible while collapsing each path
        # body.  A case with dozens of planes therefore starts compact, and
        # selecting an object still expands its ancestors on demand.
        for i in range(self.tree_objects.topLevelItemCount()):
            top = self.tree_objects.topLevelItem(i)
            top.setExpanded(True)
            for j in range(top.childCount()):
                type_item = top.child(j)
                type_item.setExpanded(True)
                if type_item.text(0) == "Paths":
                    for k in range(type_item.childCount()):
                        type_item.child(k).setExpanded(False)
        self.tree_objects.blockSignals(False)
        self._refresh_browser_opacity_control()

    def _selected_uid(self):
        items = self.tree_objects.selectedItems()
        if not items:
            return None
        return items[0].data(0, QtCore.Qt.UserRole)

    def _browser_selected_objects(self):
        objects = []
        seen = set()
        for item in self.tree_objects.selectedItems():
            for leaf in self._iter_browser_leaf_items(item):
                uid = leaf.data(0, QtCore.Qt.UserRole)
                obj = self.workspace.scene_objects.get(uid) if uid else None
                if obj is not None and obj.uid not in seen:
                    objects.append(obj)
                    seen.add(obj.uid)
        return objects

    def _refresh_browser_opacity_control(self):
        objects = self._browser_selected_objects()
        if not objects:
            self.slider_browser_opacity.blockSignals(True)
            self.slider_browser_opacity.setEnabled(False)
            self.slider_browser_opacity.setValue(100)
            self.slider_browser_opacity.blockSignals(False)
            self.label_browser_opacity.setText("Opacity: —")
            return
        values = [float(np.clip(getattr(obj, "opacity", 1.0), 0.0, 1.0)) for obj in objects]
        value = int(round(values[0] * 100.0))
        mixed = any(abs(current - values[0]) > 1e-6 for current in values[1:])
        self.slider_browser_opacity.blockSignals(True)
        self.slider_browser_opacity.setEnabled(True)
        self.slider_browser_opacity.setValue(value)
        self.slider_browser_opacity.blockSignals(False)
        self.label_browser_opacity.setText(
            f"Opacity: {value}%" + (" (mixed)" if mixed else "")
        )

    def _set_browser_opacity_for_items(self, items, value):
        try:
            opacity = float(np.clip(float(value), 0.0, 1.0))
        except (TypeError, ValueError):
            return
        objects = []
        seen = set()
        for item in items or []:
            for leaf in self._iter_browser_leaf_items(item):
                uid = leaf.data(0, QtCore.Qt.UserRole)
                obj = self.workspace.scene_objects.get(uid) if uid else None
                if obj is not None and obj.uid not in seen:
                    objects.append(obj)
                    seen.add(obj.uid)
        for obj in objects:
            obj.opacity = opacity
            if obj.data_key == "segmask_raw_surface":
                self.workspace.segmentation.opacity = opacity
            self.scene.apply_object_properties(obj, render=False, refresh_scalar_bar=False)
        if objects:
            self.ortho_viewer.refresh()
            self._refresh_segmentation_ui()
            self._refresh_browser_opacity_control()
            self._refresh_scene()

    def _on_browser_opacity_changed(self, value):
        self._set_browser_opacity_for_items(
            self.tree_objects.selectedItems(), float(value) / 100.0
        )

    def _selected_render_object(self):
        for obj in self._browser_selected_objects():
            if obj.scalars or obj.data_key == "pcmra_volume":
                return obj
        return None

    def _render_object_range(self, obj):
        if obj.data_key == "pcmra_volume":
            data = self.scene._build_dataset(obj.data_key)
            return self.scene._volume_scalar_range(obj, data, respect_clim=False)
        data = self.scene._build_dataset(obj.data_key)
        if data is None or not obj.scalars:
            return None
        values = None
        if obj.scalars in getattr(data, "point_data", {}):
            values = np.asarray(data.point_data[obj.scalars], dtype=float)
        elif obj.scalars in getattr(data, "cell_data", {}):
            values = np.asarray(data.cell_data[obj.scalars], dtype=float)
        if values is None:
            return None
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return None
        low = float(np.min(finite))
        high = float(np.percentile(finite, 99.5))
        if not np.isfinite(high) or high <= low:
            high = float(np.max(finite))
        return (low, high if high > low else low + 1.0)

    def _refresh_render_range_control(self):
        obj = self._selected_render_object()
        if obj is None:
            self.slider_render_window.setEnabled(False)
            self.slider_render_level.setEnabled(False)
            self.btn_reset_render_range.setEnabled(False)
            self.label_render_window.setText("—")
            self.label_render_level.setText("—")
            return
        clim = tuple(obj.clim) if obj.clim is not None else self._render_object_range(obj)
        if clim is None:
            self.slider_render_window.setEnabled(False)
            self.slider_render_level.setEnabled(False)
            self.btn_reset_render_range.setEnabled(False)
            self.label_render_window.setText("—")
            self.label_render_level.setText("—")
            return
        auto_range = self._render_object_range(obj)
        if auto_range is None:
            auto_range = clim
        auto_low, auto_high = (float(value) for value in auto_range)
        auto_width = max(auto_high - auto_low, 1e-12)
        width = max(float(clim[1]) - float(clim[0]), 1e-12)
        level = (float(clim[0]) + float(clim[1])) * 0.5
        width_slider_value = int(round(np.clip((width / auto_width - 0.01) / 1.99, 0.0, 1.0) * 999.0)) + 1
        level_slider_value = int(round(np.clip((level - (auto_low - auto_width)) / (3.0 * auto_width), 0.0, 1.0) * 1000.0))
        self.slider_render_window.blockSignals(True)
        self.slider_render_level.blockSignals(True)
        self.slider_render_window.setValue(width_slider_value)
        self.slider_render_level.setValue(level_slider_value)
        self.slider_render_window.blockSignals(False)
        self.slider_render_level.blockSignals(False)
        self.slider_render_window.setEnabled(True)
        self.slider_render_level.setEnabled(True)
        self.btn_reset_render_range.setEnabled(True)
        self.label_render_window.setText(f"{width:.4g}")
        self.label_render_level.setText(f"{level:.4g}")

    def _on_render_range_changed(self, _value):
        obj = self._selected_render_object()
        if obj is None:
            return
        auto_range = self._render_object_range(obj)
        if auto_range is None:
            return
        auto_low, auto_high = (float(value) for value in auto_range)
        auto_width = max(auto_high - auto_low, 1e-12)
        width = auto_width * (0.01 + 1.99 * self.slider_render_window.value() / 999.0)
        level = auto_low - auto_width + 3.0 * auto_width * self.slider_render_level.value() / 1000.0
        low = level - width * 0.5
        high = level + width * 0.5
        if high <= low:
            return
        obj.clim = (low, high)
        self.label_render_window.setText(f"{width:.4g}")
        self.label_render_level.setText(f"{level:.4g}")
        self.scene.readd_object(obj, refresh_scalar_bar=False)
        self.scene.render_all()

    def _reset_selected_render_range(self):
        obj = self._selected_render_object()
        if obj is None:
            return
        obj.clim = None
        self.scene.readd_object(obj, refresh_scalar_bar=False)
        self.scene.render_all()
        self._refresh_render_range_control()

    def _on_browser_select(self):
        self._refresh_browser_opacity_control()
        self._refresh_render_range_control()
        items = self.tree_objects.selectedItems()
        item = items[0] if items else None
        uid = item.data(0, QtCore.Qt.UserRole) if item is not None else None
        node_type = item.data(0, _BROWSER_NODE_ROLE) if item is not None else None
        if uid is None and node_type == "path":
            path_idx = item.data(0, _BROWSER_PATH_INDEX_ROLE)
            try:
                path_idx = int(path_idx)
            except (TypeError, ValueError):
                path_idx = -1
            path_uid = self._find_uid_by_indexed_data_key("smooth_path", path_idx)
            self.workspace.selected_path_index = path_idx if 0 <= path_idx < len(self.workspace.path_info) else -1
            self._selected_plane_index = -1
            self._clear_plane_drag_widgets()
            self.scene.highlight_plane(None)
            self.scene.highlight_path(path_uid)
            self.scene.show_forks_for_path(self.workspace.selected_path_index)
            self.ortho_viewer.set_selected_plane(None)
            self._set_plane_info_text("")
            if self.workspace.selected_path_index >= 0:
                self._log_selected_path_info(self.workspace.selected_path_index)
            else:
                self._set_path_info_text("")
            self._refresh_analysis_panel()
            return
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
                    self._clear_plane_drag_widgets()
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
                    self._clear_plane_drag_widgets()
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
        _ = column
        refresh_segmentation = False
        uid = item.data(0, QtCore.Qt.UserRole)
        self.tree_objects.blockSignals(True)
        if uid is not None:
            refresh_segmentation = self._set_browser_item_visibility(
                item, item.checkState(0) == QtCore.Qt.Checked
            )
            self._sync_browser_parent_states(item)
        else:
            checked = item.checkState(0) == QtCore.Qt.Checked
            refresh_segmentation = self._set_browser_item_visibility(item, checked)
            self._sync_browser_parent_states(item)
        self.tree_objects.blockSignals(False)
        if refresh_segmentation:
            self._refresh_segmentation_ui()
        self._refresh_scene()

    def _on_browser_ctx_menu(self, pos):
        item = self.tree_objects.itemAt(pos)
        if item is None:
            return
        if not item.isSelected():
            self.tree_objects.setCurrentItem(item)
        uid = item.data(0, QtCore.Qt.UserRole)
        menu = QtWidgets.QMenu(self)
        if uid is None:
            node_type = item.data(0, _BROWSER_NODE_ROLE)
            path_idx = item.data(0, _BROWSER_PATH_INDEX_ROLE)
            if node_type == "path":
                act_show = menu.addAction("Show Path Contents")
                act_hide = menu.addAction("Hide Path Contents")
                act_generate = menu.addAction("Generate Pathlines for This Path")
                act_del_all = menu.addAction("Delete Path Contents")
            else:
                act_show = menu.addAction("Show All")
                act_hide = menu.addAction("Hide All")
                act_generate = None
                act_del_all = menu.addAction("Delete All")
            action = menu.exec(self.tree_objects.viewport().mapToGlobal(pos))
            if action == act_show:
                self._set_group_vis(item, True)
            elif action == act_hide:
                self._set_group_vis(item, False)
            elif act_generate is not None and action == act_generate:
                try:
                    path_idx = int(path_idx)
                except (TypeError, ValueError):
                    path_idx = -1
                plane_indices = [
                    idx for idx, plane in enumerate(self.workspace.planes)
                    if int(getattr(plane, "path_index", -1)) == path_idx
                    and self._browser_plane_group_name(idx) == str(item.data(0, _BROWSER_GROUP_ROLE) or "Global")
                ]
                self._trigger_pathlines(plane_indices=plane_indices, selected_plane_idx=plane_indices[0] if plane_indices else None)
            elif action == act_del_all:
                self.tree_objects.setCurrentItem(item)
                self._on_delete_object()
        else:
            act_toggle = menu.addAction("Toggle Visibility")
            act_opacity = menu.addAction("Set Opacity…")
            act_reset_opacity = menu.addAction("Reset Opacity (100%)")
            act_reset_window_level = None
            act_del = menu.addAction("Delete")
            obj = self.workspace.scene_objects.get(uid)
            act_plane_sl = None
            act_selected_sl = None
            act_pathline_color = None
            pathline_idx = _parse_pathline_index(obj.data_key) if obj else None
            if obj and obj.kind == ObjectKind.PLANE:
                act_plane_sl = menu.addAction("Generate Pathlines for This Plane")
                selected_plane_indices = self._selected_plane_indices()
                if len(selected_plane_indices) > 1:
                    act_selected_sl = menu.addAction(f"Generate Pathlines for Selected Planes ({len(selected_plane_indices)})")
            if obj and str(getattr(obj, "data_key", "")) == "pcmra_volume":
                act_reset_window_level = menu.addAction("Reset PC-MRA Window/Level")
            if pathline_idx is not None:
                act_pathline_color = menu.addAction("Set Pathline Color")
            action = menu.exec(self.tree_objects.viewport().mapToGlobal(pos))
            if action == act_toggle:
                if obj:
                    self._set_browser_item_visibility(item, not obj.visible)
                    self._sync_browser_parent_states(item)
                    self._refresh_scene()
            elif action == act_opacity:
                obj = self.workspace.scene_objects.get(uid)
                if obj is not None:
                    value, accepted = QtWidgets.QInputDialog.getDouble(
                        self,
                        "Set Opacity",
                        f"Opacity for {obj.name} (0–100%):",
                        float(np.clip(obj.opacity, 0.0, 1.0)) * 100.0,
                        0.0,
                        100.0,
                        1,
                    )
                    if accepted:
                        self._set_browser_opacity_for_items([item], value / 100.0)
            elif action == act_reset_opacity:
                self._set_browser_opacity_for_items([item], 1.0)
            elif act_reset_window_level is not None and action == act_reset_window_level:
                self.scene.reset_volume_window_level()
            elif action == act_del:
                self._on_delete_object()
            elif act_plane_sl is not None and action == act_plane_sl:
                pidx = _parse_plane_index(obj.data_key)
                if pidx is not None:
                    self._trigger_pathlines(plane_indices=[pidx], selected_plane_idx=pidx)
            elif act_selected_sl is not None and action == act_selected_sl:
                selected_plane_indices = self._selected_plane_indices()
                self._trigger_pathlines(
                    plane_indices=selected_plane_indices,
                    selected_plane_idx=selected_plane_indices[0] if selected_plane_indices else None,
                )
            elif act_pathline_color is not None and action == act_pathline_color and pathline_idx is not None:
                self._choose_pathline_color(pathline_idx)

    def _trigger_pathlines(self, plane_indices=None, selected_plane_idx=None, *, prepare_segmentation=True):
        if self._pathline_running_guard("starting another pathline task"):
            return
        if self._pipeline_running_guard("starting pathlines"):
            return
        self._sync_params_to_ws()
        self._refresh_automatic_pathline_colors()
        if plane_indices is None:
            plane_indices = list(range(len(self.workspace.planes)))
        valid_indices = sorted({int(idx) for idx in plane_indices if 0 <= int(idx) < len(self.workspace.planes)})
        if not valid_indices:
            self.log("No valid planes selected for Pathlines.")
            return
        if self.workspace.flow_raw is None or self.workspace.segmask_raw is None:
            self.log("Pathlines require loaded flow and segmentation.")
            return
        active_indices = {
            int(idx)
            for idx in self.workspace.active_pathline_plane_indices
            if 0 <= int(idx) < len(self.workspace.planes)
            and self._find_uid_by_indexed_data_key("pathline", int(idx)) is not None
        }
        missing_indices = [idx for idx in valid_indices if idx not in active_indices]
        if not missing_indices:
            selected_idx = (
                int(selected_plane_idx)
                if selected_plane_idx is not None and int(selected_plane_idx) in active_indices
                else valid_indices[0]
            )
            self._select_pathline_for_plane(selected_idx)
            self.log(
                f"Pathline for plane {selected_idx} is already generated; selected the existing object."
            )
            self.statusBar().showMessage(
                f"Pathline for plane {selected_idx} is already generated", 5000
            )
            return
        self._pathline_progress_dialog = self._create_progress_dialog(
            "Generate Pathlines",
            "Preparing segmentation for pathline integration...",
        )
        self._pathline_progress_dialog.setRange(0, 1)
        self._pathline_progress_dialog.setValue(0)
        QtWidgets.QApplication.processEvents()
        if prepare_segmentation:
            try:
                self.pipeline.preprocess(self.workspace)
            except Exception as exc:
                self.log(f"Pathline preparation failed: {type(exc).__name__}: {exc}")
                self._close_progress_dialog(self._pathline_progress_dialog)
                self._pathline_progress_dialog = None
                return
        if self.workspace.segmask_binary is None:
            self.log("Pathlines require a valid preprocessed segmentation.")
            self._close_progress_dialog(self._pathline_progress_dialog)
            self._pathline_progress_dialog = None
            return
        self._start_pathline_task(missing_indices, selected_plane_idx=selected_plane_idx)

    def _refresh_automatic_pathline_colors(self):
        for obj in self.workspace.scene_objects.values():
            pathline_idx = _parse_pathline_index(obj.data_key)
            if pathline_idx is None or int(pathline_idx) in self.workspace.pathline_colors:
                continue
            color = self.workspace.pathline_color_for_plane(pathline_idx)
            if obj.color == color:
                continue
            obj.color = color
            self.scene.apply_object_properties(obj, render=False, refresh_scalar_bar=False)

    def _select_pathline_for_plane(self, plane_idx):
        uid = self._find_uid_by_indexed_data_key("pathline", int(plane_idx))
        if uid is None:
            return False
        obj = self.workspace.scene_objects.get(uid)
        made_visible = False
        if obj is not None and not obj.visible:
            obj.visible = True
            made_visible = True
            self.scene.apply_object_properties(obj, render=False, refresh_scalar_bar=False)
        if made_visible:
            self._refresh_browser()
        self._select_browser_item_by_uid(uid)
        self._on_browser_select()
        self._refresh_scene()
        return True

    def _pathline_running_guard(self, action_text):
        if self._pathline_thread is None:
            return False
        self.log(f"Pathlines are running. Wait for them to finish before {action_text}.")
        return True

    def _start_pathline_task(self, plane_indices, *, selected_plane_idx=None):
        ws = self.workspace
        if ws.flow_raw is None or ws.segmask_binary is None:
            return False
        params = ws.streamline_params
        self._on_pause()
        if self._pathline_progress_dialog is None:
            self._pathline_progress_dialog = self._create_progress_dialog(
                "Generate Pathlines",
                "Preparing pathline integration...",
            )
        else:
            self._pathline_progress_dialog.setLabelText("Starting pathline integration...")
        pathline_frame_steps = min(
            int(ws.flow_raw.shape[3]) + 1,
            max(1, int(params.pathline_max_steps)),
        )
        progress_total = max(1, len(plane_indices) * pathline_frame_steps)
        self._pathline_progress_dialog.setRange(0, progress_total)
        self._pathline_progress_dialog.setValue(0)
        self._pathline_thread = QtCore.QThread(self)
        self._pathline_worker = _PathlineTaskWorker(
            ws.flow_raw,
            ws.segmask_binary,
            ws.branch_labels,
            ws.planes,
            plane_indices,
            # Pathlines are launched from phase zero and retain that complete
            # cardiac-cycle trajectory while the timeline is played.
            time_index=0,
            spacing=ws.resolution,
            origin=ws.origin,
            rr=ws.rr,
            params={
                "pathline_seed_ratio": params.pathline_seed_ratio,
                "pathline_max_steps": params.pathline_max_steps,
                "pathline_min_seeds": params.pathline_min_seeds,
                "pathline_seed_mode": getattr(params, "pathline_seed_mode", "fixed"),
                "pathline_max_seeds": getattr(params, "pathline_max_seeds", 250),
                "pathline_terminal_speed": params.pathline_terminal_speed,
                "pathline_rng_seed": params.pathline_rng_seed,
                "pathline_temporal_cache_mb": getattr(params, "pathline_temporal_cache_mb", 512.0),
            },
        )
        self._pathline_worker.moveToThread(self._pathline_thread)
        self._pathline_thread.started.connect(self._pathline_worker.run)
        self._pathline_worker.progress.connect(self._on_pathline_progress)
        self._pathline_selected_plane_idx = selected_plane_idx
        self._pathline_worker.finished.connect(
            self._on_pathline_finished,
            QtCore.Qt.QueuedConnection,
        )
        self._pathline_worker.failed.connect(self._on_pathline_failed)
        self._pathline_worker.finished.connect(self._pathline_thread.quit)
        self._pathline_worker.failed.connect(self._pathline_thread.quit)
        self._pathline_thread.finished.connect(self._cleanup_pathline_task)
        self.statusBar().showMessage("Generating pathlines...")
        self._pathline_thread.start()
        return True

    def _on_pathline_progress(self, payload):
        data = dict(payload or {})
        dialog = self._pathline_progress_dialog
        if dialog is not None and dialog.isVisible():
            dialog.setRange(0, max(1, int(data.get("total", 1))))
            dialog.setValue(max(0, int(data.get("current", 0))))
            dialog.setLabelText(str(data.get("message", "Generating pathlines...")))

    def _on_pathline_finished(self, payload):
        selected_plane_idx = self._pathline_selected_plane_idx
        self._pathline_selected_plane_idx = None
        data = dict(payload or {})
        indices = [int(idx) for idx in data.get("plane_indices", [])]
        meshes = dict(data.get("meshes", {}) or {})
        seed_payload = {
            int(plane_idx): np.asarray(seeds, dtype=float)
            for plane_idx, seeds in dict(data.get("seeds", {}) or {}).items()
            if seeds is not None
        }
        self.scene.trigger_pathlines(indices, precomputed=meshes)
        self.workspace.pathline_seed_cache.update(seed_payload)
        self._refresh_browser()
        self.scene.invalidate_cache("pathline_")
        self.scene.sync_from_workspace()
        self._refresh_all()
        if selected_plane_idx is not None and 0 <= int(selected_plane_idx) < len(self.workspace.planes):
            self._selected_plane_index = int(selected_plane_idx)
            self.ortho_viewer.set_selected_plane(int(selected_plane_idx))
            self._clear_plane_drag_widgets()
            self.scene.highlight_plane(self._find_uid_by_indexed_data_key("plane", int(selected_plane_idx)))
            self._set_path_info_text("")
            self._log_selected_plane_metric(int(selected_plane_idx))
            self._select_pathline_for_plane(int(selected_plane_idx))
        elapsed = float(data.get("elapsed_sec", 0.0))
        temporal_cache = "reused all temporal VTK frames" if bool(data.get("temporal_cache_all_phases", False)) else "used bounded temporal VTK caching"
        self.log(f"Generated pathlines for {len(indices)} plane(s) in {elapsed:.2f}s; {temporal_cache}.")
        self.statusBar().showMessage(f"Pathlines completed in {elapsed:.2f}s", 8000)
        self._close_progress_dialog(self._pathline_progress_dialog)
        self._pathline_progress_dialog = None

    def _on_pathline_failed(self, error_text):
        self._pathline_selected_plane_idx = None
        self.log(f"PATHLINE ERROR:\n{str(error_text).strip()}")
        self.statusBar().showMessage("Pathlines failed", 10000)
        self._close_progress_dialog(self._pathline_progress_dialog)
        self._pathline_progress_dialog = None

    def _cleanup_pathline_task(self):
        if self._pathline_worker is not None:
            self._pathline_worker.deleteLater()
        if self._pathline_thread is not None:
            self._pathline_thread.deleteLater()
        self._pathline_worker = None
        self._pathline_thread = None
        self._pathline_selected_plane_idx = None

    def _trigger_plane_streamlines(self, plane_idx=None):
        indices = None if plane_idx is None else [int(plane_idx)]
        self._trigger_pathlines(plane_indices=indices, selected_plane_idx=plane_idx)

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
        self.scene.apply_object_properties(obj, render=False, refresh_scalar_bar=False)
        self._refresh_browser()
        self._refresh_scene()

    def _choose_pathline_color(self, pathline_idx):
        current = self.workspace.pathline_color_for_plane(pathline_idx)
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(current), self, f"Select Pathline {int(pathline_idx)} Color")
        if not color.isValid():
            return
        self._set_pathline_color(pathline_idx, color.name())

    def _set_group_vis(self, group_item, visible):
        self.tree_objects.blockSignals(True)
        refresh_segmentation = self._set_browser_item_visibility(group_item, visible)
        self._sync_browser_parent_states(group_item)
        self.tree_objects.blockSignals(False)
        if refresh_segmentation:
            self._refresh_segmentation_ui()
        self._refresh_browser()
        self._refresh_scene()

    def _on_delete_object(self):
        items = self.tree_objects.selectedItems()
        if not items:
            return
        plane_indices_removed = []
        pathline_indices_removed = []
        clear_streamlines = False
        uids = []
        for item in items:
            uids.extend(self._collect_browser_uids(item))
        uids = list(dict.fromkeys(uids))
        if not uids:
            return
        log_message = f"Deleted {len(uids)} selected object{'s' if len(uids) != 1 else ''}"
        for current_uid in uids:
            obj = self.workspace.scene_objects.get(current_uid)
            if obj and obj.kind == ObjectKind.PLANE:
                pidx = _parse_plane_index(obj.data_key)
                if pidx is not None:
                    plane_indices_removed.append(pidx)
            if obj and obj.data_key == "streamlines_live":
                clear_streamlines = True
            pidx = _parse_pathline_index(obj.data_key) if obj else None
            if pidx is not None:
                pathline_indices_removed.append(pidx)
            self.scene.remove_object(current_uid)
        self.log(log_message)
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
        self.lab_t.setText(str(v))
        self._pending_timeline_value = int(v)
        if not self._timeline_scrub_timer.isActive():
            self._timeline_scrub_timer.start()

    def _apply_pending_timeline_value(self, final=False):
        if self._pending_timeline_value is None:
            if final:
                self._refresh_segmentation_ui()
                self._refresh_selection_info()
            return
        value = int(self._pending_timeline_value)
        self._pending_timeline_value = None
        self.workspace.current_t = value
        self.scene.update_time(value)
        self.ortho_viewer.refresh()
        self._refresh_render_range_control()
        if final or not self.slider_t.isSliderDown():
            self._refresh_segmentation_ui()
            self._refresh_selection_info()

    def _finish_timeline_scrub(self):
        self._timeline_scrub_timer.stop()
        self._apply_pending_timeline_value(final=True)

    def _on_ortho_time_step(self, delta):
        if int(delta) < 0:
            self._on_prev_frame()
        elif int(delta) > 0:
            self._on_next_frame()

    def _on_prev_frame(self):
        self.workspace.current_t = max(0, self.workspace.current_t - 1)
        self._refresh_timeline()
        self.scene.update_time(self.workspace.current_t)
        self.ortho_viewer.refresh()
        self._refresh_render_range_control()
        self._refresh_segmentation_ui()
        self._refresh_selection_info()

    def _on_next_frame(self):
        T = self.workspace.time_count()
        self.workspace.current_t = min(T - 1, self.workspace.current_t + 1)
        self._refresh_timeline()
        self.scene.update_time(self.workspace.current_t)
        self.ortho_viewer.refresh()
        self._refresh_render_range_control()
        self._refresh_segmentation_ui()
        self._refresh_selection_info()

    def _on_play(self):
        if self.workspace.time_count() <= 1:
            return
        self._playback_frame_times_ms.clear()
        self._playback_render_times_ms.clear()
        self._playback_last_tick_started_at = None
        self.lab_playback_performance.setText("Render: --")
        self.scene.set_playback_active(True)
        self.ortho_viewer.set_playback_active(True)
        self._play_timer.start(0)

    def _on_pause(self):
        self._play_timer.stop()
        self._playback_last_tick_started_at = None
        self.scene.set_playback_active(False)
        self.ortho_viewer.set_playback_active(False)
        self.ortho_viewer.refresh()
        self._refresh_segmentation_ui()
        self._refresh_selection_info()

    def _on_play_tick(self):
        T = self.workspace.time_count()
        if T <= 1:
            return
        tick_started_at = time.perf_counter()
        if self._playback_last_tick_started_at is not None:
            self._playback_frame_times_ms.append(
                (tick_started_at - self._playback_last_tick_started_at) * 1000.0
            )
        self._playback_last_tick_started_at = tick_started_at
        self.workspace.current_t = (self.workspace.current_t + 1) % T
        self._refresh_timeline()
        self.scene.update_time(self.workspace.current_t)
        self.ortho_viewer.refresh(update_plane=False)
        self._refresh_render_range_control()
        render_elapsed_ms = (time.perf_counter() - tick_started_at) * 1000.0
        self._playback_render_times_ms.append(render_elapsed_ms)
        average_render_ms = sum(self._playback_render_times_ms) / len(self._playback_render_times_ms)
        if self._playback_frame_times_ms:
            average_frame_ms = sum(self._playback_frame_times_ms) / len(self._playback_frame_times_ms)
            fps = 1000.0 / max(average_frame_ms, 1e-6)
            self.lab_playback_performance.setText(f"Render: {average_render_ms:.0f} ms | {fps:.1f} fps")
        else:
            self.lab_playback_performance.setText(f"Render: {average_render_ms:.0f} ms")

        # A single-shot timer prevents overdue playback events from accumulating
        # while VTK is busy. A requested interval below render time runs at the
        # fastest rate the scene can sustain.
        delay_ms = max(0, int(self.spin_interval.value()) - int(round(render_elapsed_ms)))
        if self.scene is not None and self._play_timer.isSingleShot():
            self._play_timer.start(delay_ms)

    def _refresh_scene(self):
        try:
            self.scene.render_all()
        except Exception as e:
            self.log(f"VIEW ERROR: {type(e).__name__}: {e}")

    def _load_selected_input_case(self, case, dicom_parameter_overrides=None):
        if self._autoseg_running_guard("loading another case"):
            return
        progress_dialog = None
        load_succeeded = False
        try:
            resolved = resolve_input_case(case)
            self._active_input_case = resolved
            self._active_dicom_parameter_overrides = dict(dicom_parameter_overrides or {})
            if self._edit_mode is not None:
                self._exit_interactive_edit(False)
            self._clear_plane_drag_widgets()
            self._seg_surface_rebuild_timer.stop()
            self._seg_edit_active = False
            self._reset_segmentation_edit_history()
            self._sync_params_to_ws()
            self.workspace.reset_all()
            self._last_quality_report = None
            self._last_plane_import_report = None
            apply_config_bundle_to_workspace(self.workspace, self._config_bundle)
            self._sync_params_to_ws()
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
                if self.workspace.loader_params.background_phase_correction.enabled:
                    progress_dialog = self._create_progress_dialog(
                        "Background Phase Correction",
                        "Loading H5 and preparing background phase correction...",
                    )
                    progress_handler = self._make_progress_handler(progress_dialog, "H5 Load")
                else:
                    progress_handler = None
            self.pipeline.load_data(
                self.workspace,
                self.log,
                input_source=resolved,
                progress_callback=progress_handler,
            )
            self.scene.workspace = self.workspace
            self.scene.reset_display_reference()
            self.scene.reset_scene()
            self._refresh_all()
            self.ortho_viewer.update_slider_ranges()
            label = resolved.display_name or resolved.input_path
            mode = "enabled" if self.workspace.loader_params.background_phase_correction.enabled else "disabled"
            self.log(f"Loaded input: {label} (BGC {mode})")
            load_succeeded = True
        except Exception as e:
            self.log(f"LOAD ERROR: {type(e).__name__}: {e}")
            self.log(traceback.format_exc())
        finally:
            if progress_dialog is not None:
                self._close_progress_dialog(progress_dialog)
        if load_succeeded:
            self._input_signature = self._current_input_signature()

    def _current_input_signature(self):
        """Stable signature for deciding whether an input reload is necessary."""
        self._sync_params_to_ws()
        ws = self.workspace
        return json.dumps({
            "path": str(getattr(self._active_input_case, "input_path", "") or ws.paths.flow_path),
            "kind": str(getattr(self._active_input_case, "input_kind", "")),
            "group": getattr(self._active_input_case, "source_group", None),
            "correction": ws.loader_params.background_phase_correction.to_dict(),
            "dicom": dict(self._active_dicom_parameter_overrides or {}),
            "resolution": self.edit_input_resolution.text().strip(),
            "venc": self.edit_input_venc.text().strip(),
            "spatial_order": self.edit_input_spatial_order.text().strip(),
            "venc_order": self.edit_input_venc_order.text().strip(),
        }, sort_keys=True, default=str)

    def _reload_input_case(self):
        if self._active_input_case is None:
            self.log("No input case is loaded.")
            return
        if self._pipeline_running_guard("reloading input") or self._autoseg_running_guard("reloading input"):
            return
        signature = self._current_input_signature()
        if signature == self._input_signature:
            self.log("Input parameters unchanged; reload skipped.")
            self.statusBar().showMessage("Input unchanged", 4000)
            return
        self.log("Input parameters changed; reloading input and invalidating downstream results.")
        self._load_selected_input_case(
            self._active_input_case,
            dicom_parameter_overrides=self._active_dicom_parameter_overrides,
        )

    def _prompt_background_phase_choice(self, case):
        resolved = resolve_input_case(case)
        label = resolved.display_name or resolved.input_path
        if resolved.input_kind == "h5":
            features = dict(resolved.metadata or {})
            if "has_background_correction_cache" not in features:
                features.update(inspect_h5_input_case(resolved))
                resolved.metadata.update(features)
            if bool(features.get("has_background_correction_cache", False)):
                self.chk_bpc_enabled.setChecked(True)
                method = str(features.get("background_correction_method", "wrls_arto") or "wrls_arto")
                self._set_background_phase_method(method)
                method_label = self.combo_bpc_method.currentText()
                self.log(f"Using embedded {method_label} background correction cache: {label}")
                return resolved
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
        if enabled:
            method_labels = [item[0] for item in _BACKGROUND_PHASE_METHOD_ITEMS]
            selected, accepted = QtWidgets.QInputDialog.getItem(
                self,
                "Background Phase Correction Method",
                "Correction method:",
                method_labels,
                0,
                False,
            )
            if not accepted:
                self.chk_bpc_enabled.setChecked(False)
                self.log(f"Load cancelled: {label}")
                return None
            method_by_label = dict(_BACKGROUND_PHASE_METHOD_ITEMS)
            self._set_background_phase_method(method_by_label.get(str(selected), "wrls_arto"))
            self.log(f"Background correction method: {self.combo_bpc_method.currentText()}")
        return resolved

    def _on_open_h5(self):
        if self._autoseg_running_guard("loading another case"):
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open H5", "", "H5 (*.h5 *.hdf5);;All (*)")
        if not path:
            return
        cases = discover_h5_input_cases(path)
        if not cases:
            self.log(f"LOAD ERROR: ValueError: no supported H5 cases found: {path}")
            return
        resolved = cases[0]
        if len(cases) > 1:
            dialog = H5CaseSelectDialog(cases, self)
            if dialog.exec() != QtWidgets.QDialog.Accepted:
                return
            selected = dialog.selected_case()
            if selected is None:
                return
            resolved = selected
        resolved = self._prompt_background_phase_choice(resolved)
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
        if dialog.exec() != QtWidgets.QDialog.Accepted:
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

    def _rendering_kwargs(self):
        rendering_cfg = bundle_to_autoflow_kwargs(self._config_bundle)
        runtime_cfg = dict(getattr(self.workspace, "render_settings", {}) or {})
        rendering_cfg.update(copy.deepcopy(runtime_cfg))
        return {
            "fps": int(rendering_cfg.get("fps", 12)),
            "plane_rotation_frames": int(rendering_cfg.get("plane_rotation_frames", 180)),
            "camera_view": str(rendering_cfg.get("camera_view", "right")),
            "camera_distance_scale": float(rendering_cfg.get("camera_distance_scale", 1.5)),
            "rotate_dynamic_video": bool(rendering_cfg.get("rotate_dynamic_video", True)),
            "dynamic_rotation_frames": int(rendering_cfg.get("dynamic_rotation_frames", 180)),
            "dynamic_rotation_elevation_deg": rendering_cfg.get("dynamic_rotation_elevation_deg", 10.0),
            "dynamic_time_repeat": int(rendering_cfg.get("dynamic_time_repeat", 3)),
            "add_plane_idx": bool(rendering_cfg.get("add_plane_idx", True)),
            "add_path_idx": bool(rendering_cfg.get("add_path_idx", False)),
            "plane_video_cfg": dict(rendering_cfg.get("plane_video_cfg", {})),
            "window_size": tuple(rendering_cfg.get("window_size", (1600, 1200))),
            "shared_colorbar_show": bool(rendering_cfg.get("shared_colorbar_show", True)),
            "shared_colorbar_bar_cfg": dict(rendering_cfg.get("shared_colorbar_bar_cfg", {})),
            "wss_clim": tuple(rendering_cfg.get("wss_clim", (0.0, 10.0))),
            "wss_show_scalar_bar": bool(rendering_cfg.get("wss_show_scalar_bar", True)),
            "wss_bar_cfg": dict(rendering_cfg.get("wss_bar_cfg", {})),
            "tke_clim": tuple(rendering_cfg.get("tke_clim", (0.0, 100.0))),
            "tke_show_scalar_bar": bool(rendering_cfg.get("tke_show_scalar_bar", True)),
            "tke_bar_cfg": dict(rendering_cfg.get("tke_bar_cfg", {})),
            "pressure_gradient_clim": None if rendering_cfg.get("pressure_gradient_clim", None) is None else tuple(rendering_cfg.get("pressure_gradient_clim", (-1.0, 1.0))),
            "pressure_gradient_show_scalar_bar": bool(rendering_cfg.get("pressure_gradient_show_scalar_bar", True)),
            "pressure_gradient_bar_cfg": dict(rendering_cfg.get("pressure_gradient_bar_cfg", {})),
            "relative_pressure_clim": None if rendering_cfg.get("relative_pressure_clim", None) is None else tuple(rendering_cfg.get("relative_pressure_clim", (-1.0, 1.0))),
            "relative_pressure_show_scalar_bar": bool(rendering_cfg.get("relative_pressure_show_scalar_bar", True)),
            "relative_pressure_bar_cfg": dict(rendering_cfg.get("relative_pressure_bar_cfg", {})),
            "streamline_clim": None if rendering_cfg.get("streamline_clim", None) is None else tuple(rendering_cfg.get("streamline_clim")),
            "streamline_show_scalar_bar": bool(rendering_cfg.get("streamline_show_scalar_bar", True)),
            "streamline_bar_cfg": dict(rendering_cfg.get("streamline_bar_cfg", {})),
        }

    def _update_video_export_summary(self, out_dir, requested_flags, video_outputs, video_times, total_time_sec):
        summary_path = os.path.join(out_dir, "summary.json")
        summary = {}
        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, dict):
                    summary = payload
            except Exception as e:
                self.log(f"[Video Export] warning: failed to read existing summary.json: {type(e).__name__}: {e}")

        existing_videos = summary.get("videos")
        if not isinstance(existing_videos, dict):
            existing_videos = {}
        for name, path_value in video_outputs.items():
            existing_videos[str(name)] = str(path_value or "")
        summary["videos"] = existing_videos

        existing_requested = summary.get("requested_videos")
        if not isinstance(existing_requested, dict):
            existing_requested = {}
        merged_requested = {}
        for name in sorted(set(existing_requested) | set(requested_flags)):
            merged_requested[str(name)] = bool(existing_requested.get(name, False) or requested_flags.get(name, False))
        summary["requested_videos"] = merged_requested

        existing_video_times = summary.get("video_times_sec")
        if not isinstance(existing_video_times, dict):
            existing_video_times = {}
        for name, elapsed in video_times.items():
            existing_video_times[str(name)] = float(elapsed)
        summary["video_times_sec"] = existing_video_times

        summary.setdefault("output_dir", str(out_dir))
        summary["gui_video_export"] = {
            "output_dir": str(out_dir),
            "requested_videos": {str(name): bool(enabled) for name, enabled in requested_flags.items()},
            "videos": {str(name): str(path_value or "") for name, path_value in video_outputs.items()},
            "video_times_sec": {str(name): float(elapsed) for name, elapsed in video_times.items()},
            "total_time_sec": float(total_time_sec),
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return summary_path

    def _prompt_video_export_options(self):
        default_dir = self.workspace.paths.output_dir or os.path.dirname(self.workspace.paths.flow_path or ".") or "."
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Export Videos")
        layout = QtWidgets.QVBoxLayout(dialog)
        form = QtWidgets.QFormLayout()

        path_edit = QtWidgets.QLineEdit(str(default_dir))
        browse_btn = QtWidgets.QPushButton("Browse...")
        path_row = QtWidgets.QHBoxLayout()
        path_row.addWidget(path_edit, 1)
        path_row.addWidget(browse_btn)
        path_widget = QtWidgets.QWidget()
        path_widget.setLayout(path_row)
        form.addRow("Output Directory", path_widget)

        check_plane = QtWidgets.QCheckBox("Plane")
        check_wss = QtWidgets.QCheckBox("WSS")
        check_tke = QtWidgets.QCheckBox("TKE")
        check_pg = QtWidgets.QCheckBox("Relative Pressure")
        check_streamlines = QtWidgets.QCheckBox("Streamlines")
        check_plane.setChecked(True)

        video_box = QtWidgets.QGroupBox("Export Items")
        video_layout = QtWidgets.QVBoxLayout(video_box)
        for widget in [check_plane, check_wss, check_tke, check_pg, check_streamlines]:
            video_layout.addWidget(widget)
        form.addRow(video_box)
        layout.addLayout(form)

        hint = QtWidgets.QLabel("WSS, TKE, and relative-pressure videos require derived data. Streamlines require segmentation and flow.")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        layout.addWidget(buttons)

        def _browse_dir():
            chosen = QtWidgets.QFileDialog.getExistingDirectory(dialog, "Choose Video Output Directory", path_edit.text().strip() or default_dir)
            if chosen:
                path_edit.setText(chosen)

        browse_btn.clicked.connect(_browse_dir)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return None
        out_dir = str(path_edit.text().strip() or default_dir)
        if not any([check_plane.isChecked(), check_wss.isChecked(), check_tke.isChecked(), check_pg.isChecked(), check_streamlines.isChecked()]):
            self.log("Video export cancelled: no video items selected.")
            return None
        return _VideoExportOptions(
            out_dir=out_dir,
            export_plane=check_plane.isChecked(),
            export_wss=check_wss.isChecked(),
            export_tke=check_tke.isChecked(),
            export_pg=check_pg.isChecked(),
            export_streamlines=check_streamlines.isChecked(),
        )

    def _on_export_videos(self):
        if self._autoseg_running_guard("exporting videos"):
            return
        if not self.workspace.data_loaded:
            self.log("No data loaded. Use File > Open H5 or Import DICOM Directory.")
            return
        options = self._prompt_video_export_options()
        if options is None:
            return
        self._sync_params_to_ws()
        os.makedirs(options.out_dir, exist_ok=True)
        self.workspace.paths.output_dir = options.out_dir
        progress = self._create_progress_dialog("Export Videos", "Preparing video export...")
        requested_flags = {
            "plane": bool(options.export_plane),
            "wss": bool(options.export_wss),
            "tke": bool(options.export_tke),
            "pg": bool(options.export_pg),
            "streamlines": bool(options.export_streamlines),
        }
        rendered = {}
        video_outputs = {}
        video_times = {}
        export_started_at = time.perf_counter()
        try:
            if self.workspace.segmask_binary is None and self.workspace.segmask_raw is not None:
                self.pipeline.preprocess(self.workspace)

            need_wss = bool(options.export_wss)
            need_tke = bool(options.export_tke)
            need_pg = bool(options.export_pg)
            if need_wss or need_tke or need_pg:
                progress.setLabelText("Computing derived data for video export...")
                QtWidgets.QApplication.processEvents()
                if self.workspace.segmask_raw is None:
                    raise ValueError("derived videos require segmentation")
                self.pipeline._ensure_derived_metrics(
                    self.workspace,
                    save_pixelwise=False,
                    refresh_scene_objects=False,
                    compute_wss=need_wss,
                    compute_tke=need_tke,
                    compute_pressure_gradient=need_pg,
                )

            render_cfg = self._rendering_kwargs()

            def _run(name, fn, enabled, available=True):
                if not enabled:
                    return
                if not available:
                    video_outputs[name] = ""
                    self.log(f"[Video Export] skipped {name}: upstream data unavailable")
                    return
                progress.setLabelText(f"Rendering {name} video...")
                QtWidgets.QApplication.processEvents()
                t0 = time.perf_counter()
                out = fn()
                elapsed = time.perf_counter() - t0
                video_times[name] = float(elapsed)
                video_outputs[name] = str(out or "")
                if out:
                    rendered[name] = out
                    self.log(f"[Video Export] {name} saved: {out} | time={elapsed:.2f}s")
                else:
                    self.log(f"[Video Export] {name} produced no output | time={elapsed:.2f}s")

            _run(
                "plane",
                lambda: render_plane_rotation_video(
                    self.workspace,
                    options.out_dir,
                    fps=render_cfg["fps"],
                    n_frames=render_cfg["plane_rotation_frames"],
                    smoothing_iteration=self.workspace.derived_params.smoothing_iteration,
                    distance_scale=render_cfg["camera_distance_scale"],
                    add_plane_idx=render_cfg["add_plane_idx"],
                    add_path_idx=render_cfg["add_path_idx"],
                    plane_video_cfg=render_cfg["plane_video_cfg"],
                    window_size=render_cfg["window_size"],
                ),
                options.export_plane,
                available=len(self.workspace.planes) > 0,
            )
            _run(
                "wss",
                lambda: render_wss_video(
                    self.workspace,
                    options.out_dir,
                    fps=render_cfg["fps"],
                    smoothing_iteration=self.workspace.derived_params.smoothing_iteration,
                    view=render_cfg["camera_view"],
                    distance_scale=render_cfg["camera_distance_scale"],
                    wss_clim=render_cfg["wss_clim"],
                    show_scalar_bar=render_cfg["wss_show_scalar_bar"],
                    wss_bar_cfg=render_cfg["wss_bar_cfg"],
                    rotate=render_cfg["rotate_dynamic_video"],
                    rotation_frames=render_cfg["dynamic_rotation_frames"],
                    elevation_deg=render_cfg["dynamic_rotation_elevation_deg"],
                    time_repeat=render_cfg["dynamic_time_repeat"],
                    window_size=render_cfg["window_size"],
                ),
                options.export_wss,
                available=self.workspace.derived.wss_surfaces is not None and len(self.workspace.derived.wss_surfaces) > 0,
            )
            _run(
                "tke",
                lambda: render_tke_video(
                    self.workspace,
                    options.out_dir,
                    fps=render_cfg["fps"],
                    smoothing_iteration=self.workspace.derived_params.smoothing_iteration,
                    view=render_cfg["camera_view"],
                    distance_scale=render_cfg["camera_distance_scale"],
                    tke_clim=render_cfg["tke_clim"],
                    show_scalar_bar=render_cfg["tke_show_scalar_bar"],
                    tke_bar_cfg=render_cfg["tke_bar_cfg"],
                    rotate=render_cfg["rotate_dynamic_video"],
                    rotation_frames=render_cfg["dynamic_rotation_frames"],
                    elevation_deg=render_cfg["dynamic_rotation_elevation_deg"],
                    time_repeat=render_cfg["dynamic_time_repeat"],
                    window_size=render_cfg["window_size"],
                ),
                options.export_tke,
                available=self.workspace.derived.tke_array is not None or self.workspace.derived.tke_volume is not None,
            )
            _run(
                "pressure_gradient",
                lambda: render_pressure_gradient_video(
                    self.workspace,
                    options.out_dir,
                    fps=render_cfg["fps"],
                    smoothing_iteration=self.workspace.derived_params.smoothing_iteration,
                    view=render_cfg["camera_view"],
                    distance_scale=render_cfg["camera_distance_scale"],
                    pressure_gradient_clim=render_cfg["pressure_gradient_clim"],
                    show_scalar_bar=render_cfg["pressure_gradient_show_scalar_bar"],
                    pressure_gradient_bar_cfg=render_cfg["pressure_gradient_bar_cfg"],
                    rotate=render_cfg["rotate_dynamic_video"],
                    rotation_frames=render_cfg["dynamic_rotation_frames"],
                    elevation_deg=render_cfg["dynamic_rotation_elevation_deg"],
                    time_repeat=render_cfg["dynamic_time_repeat"],
                    window_size=render_cfg["window_size"],
                ),
                options.export_pg,
                available=self.workspace.derived.pressure_gradient_magnitude is not None,
            )
            _run(
                "relative_pressure",
                lambda: render_relative_pressure_video(
                    self.workspace,
                    options.out_dir,
                    fps=render_cfg["fps"],
                    smoothing_iteration=self.workspace.derived_params.smoothing_iteration,
                    view=render_cfg["camera_view"],
                    distance_scale=render_cfg["camera_distance_scale"],
                    relative_pressure_clim=render_cfg["relative_pressure_clim"],
                    show_scalar_bar=render_cfg["relative_pressure_show_scalar_bar"],
                    relative_pressure_bar_cfg=render_cfg["relative_pressure_bar_cfg"],
                    rotate=render_cfg["rotate_dynamic_video"],
                    rotation_frames=render_cfg["dynamic_rotation_frames"],
                    elevation_deg=render_cfg["dynamic_rotation_elevation_deg"],
                    time_repeat=render_cfg["dynamic_time_repeat"],
                    window_size=render_cfg["window_size"],
                ),
                options.export_pg,
                available=self.workspace.derived.relative_pressure_array is not None,
            )
            _run(
                "streamlines",
                lambda: render_streamlines_video(
                    self.workspace,
                    options.out_dir,
                    fps=render_cfg["fps"],
                    smoothing_iteration=self.workspace.derived_params.smoothing_iteration,
                    view=render_cfg["camera_view"],
                    distance_scale=render_cfg["camera_distance_scale"],
                    streamline_clim=render_cfg["streamline_clim"],
                    show_scalar_bar=render_cfg["streamline_show_scalar_bar"],
                    streamline_bar_cfg=render_cfg["streamline_bar_cfg"],
                    rotate=render_cfg["rotate_dynamic_video"],
                    rotation_frames=render_cfg["dynamic_rotation_frames"],
                    elevation_deg=render_cfg["dynamic_rotation_elevation_deg"],
                    time_repeat=render_cfg["dynamic_time_repeat"],
                    window_size=render_cfg["window_size"],
                ),
                options.export_streamlines,
                available=self.workspace.flow_raw is not None and self.workspace.segmask_binary is not None and self.workspace.segmask_3d is not None,
            )
            if rendered:
                self.log(f"[Video Export] completed: {', '.join(sorted(rendered))}")
            else:
                self.log("[Video Export] completed with no saved videos")
            summary_path = self._update_video_export_summary(
                options.out_dir,
                requested_flags,
                video_outputs,
                video_times,
                time.perf_counter() - export_started_at,
            )
            self.log(f"[Video Export] summary updated: {summary_path}")
        except Exception as e:
            self.log(f"VIDEO EXPORT ERROR: {type(e).__name__}: {e}")
            self.log(traceback.format_exc())
        finally:
            self._close_progress_dialog(progress)

    def _on_close_workspace(self):
        if self._autoseg_thread is not None:
            self.log("Auto segmentation is running. Wait for it to finish before clearing the workspace.")
            return
        if self._pathline_thread is not None:
            self.log("Pathlines are running. Wait for them to finish before clearing the workspace.")
            return
        if self._edit_mode is not None:
            self._exit_interactive_edit(False)
        self._clear_plane_drag_widgets()
        self._seg_surface_rebuild_timer.stop()
        self._seg_edit_active = False
        self._reset_segmentation_edit_history()
        self.workspace.reset_all()
        self._last_quality_report = None
        self._last_plane_import_report = None
        apply_config_bundle_to_workspace(self.workspace, self._config_bundle)
        self._selected_plane_index = -1
        self.scene.reset_display_reference()
        self.scene.reset_scene()
        self.ortho_viewer.reset_state()
        self._refresh_all()
        self.log("Workspace cleared")

    def _pipeline_running_guard(self, action_text):
        if self._pipeline_thread is None:
            return False
        self.log(f"A pipeline task is running. Wait for it to finish before {action_text}.")
        return True

    def _start_pipeline_task(self, steps, task_label):
        if self._pipeline_thread is not None:
            return False
        steps = list(steps)
        if not steps:
            return False
        self._on_pause()
        self._pipeline_task_label = str(task_label)
        self._pipeline_progress_dialog = self._create_progress_dialog(
            self._pipeline_task_label,
            f"Preparing {self._pipeline_task_label}...",
        )
        self._pipeline_progress_dialog.setRange(0, len(steps))
        self._pipeline_progress_dialog.setValue(0)
        self._pipeline_thread = QtCore.QThread(self)
        self._pipeline_worker = _PipelineTaskWorker(self.pipeline, self.workspace, steps)
        self._pipeline_worker.moveToThread(self._pipeline_thread)
        self._pipeline_thread.started.connect(self._pipeline_worker.run)
        self._pipeline_worker.progress.connect(self._on_pipeline_progress)
        self._pipeline_worker.finished.connect(self._on_pipeline_finished)
        self._pipeline_worker.failed.connect(self._on_pipeline_failed)
        self._pipeline_worker.finished.connect(self._pipeline_thread.quit)
        self._pipeline_worker.failed.connect(self._pipeline_thread.quit)
        self._pipeline_thread.finished.connect(self._cleanup_pipeline_task)
        self.statusBar().showMessage(f"Running {self._pipeline_task_label}...")
        self._pipeline_thread.start()
        return True

    def _on_pipeline_progress(self, payload):
        data = dict(payload or {})
        dialog = self._pipeline_progress_dialog
        if dialog is not None and dialog.isVisible():
            dialog.setRange(0, max(int(data.get("total", 1)), 1))
            dialog.setValue(max(int(data.get("current", 0)), 0))
            dialog.setLabelText(str(data.get("message", self._pipeline_task_label)))
        step = data.get("step")
        if data.get("stage") == "start" and step is not None:
            self.log(f"[{self._pipeline_task_label}] Running {step.label}...")
        elif data.get("stage") == "done" and step is not None:
            self.log(
                f"[{step.label}] {float(data.get('elapsed_sec', 0.0)):.2f}s - "
                f"{data.get('result_message', '')}"
            )

    def _finish_pipeline_scene_refresh(self, steps):
        if self._pipeline_progress_dialog is not None and self._pipeline_progress_dialog.isVisible():
            self._pipeline_progress_dialog.setLabelText("Refreshing views...")
        steps = set(steps or [])
        prefix_map = {
            StepId.UNWRAP_PHASE: ("phase_wrap_mask", "phase_wrap_count"),
            StepId.GENERATE_SKELETON: ("segmask_group_", "skeleton_"),
            StepId.GENERATE_GRAPH: ("segmask_group_", "skeleton_", "graph_", "forks_", "path_"),
            StepId.GENERATE_PLANES: ("smooth_path_", "plane_", "pathline_", "pwv_planes"),
            StepId.GENERATE_STREAMLINES: ("streamlines_live",),
            StepId.COMPUTE_PWV: ("pwv_planes",),
            StepId.COMPUTE_DERIVED_METRICS: (
                "wss_surface_live",
                "tke_volume",
                "pressure_gradient_volume",
                "relative_pressure_volume",
                "vorticity_magnitude_volume",
                "q_criterion_volume",
                "swirling_strength_volume",
            ),
        }
        rebuild_prefixes = []
        for step in steps:
            rebuild_prefixes.extend(prefix_map.get(step, ()))
        if any(
            obj.actor is None and obj.data_key.startswith("segmask_group_")
            for obj in self.workspace.scene_objects.values()
        ):
            rebuild_prefixes.append("segmask_group_")
        rebuild_prefixes = tuple(dict.fromkeys(rebuild_prefixes))
        for prefix in rebuild_prefixes:
            self.scene.invalidate_cache(prefix)
        self._sync_segmentation_scene_object()
        self.scene.sync_from_workspace(rebuild_prefixes=rebuild_prefixes)
        self._refresh_browser()
        self._refresh_timeline()
        if steps & {
            StepId.GENERATE_PLANES,
            StepId.COMPUTE_PLANE_METRICS,
            StepId.COMPUTE_PWV,
            StepId.COMPUTE_DERIVED_METRICS,
        }:
            self._refresh_plane_curve_metric_options()
            self._refresh_analysis_panel()
        self._refresh_selection_info()
        self._refresh_segmentation_ui()
        self.ortho_viewer.refresh()

    def _on_pipeline_finished(self, payload):
        self._last_quality_report = None
        data = dict(payload or {})
        try:
            completed_steps = [item[0] for item in data.get("results", []) if item]
            self._finish_pipeline_scene_refresh(completed_steps)
            elapsed = float(data.get("elapsed_sec", 0.0))
            self.log(f"[{self._pipeline_task_label}] Completed in {elapsed:.2f}s")
            self.statusBar().showMessage(f"{self._pipeline_task_label} completed in {elapsed:.2f}s", 8000)
        except Exception:
            self.log(f"VIEW REFRESH ERROR: {traceback.format_exc()}")
        finally:
            self._close_progress_dialog(self._pipeline_progress_dialog)
            self._pipeline_progress_dialog = None

    def _on_pipeline_failed(self, error_text):
        self._run_all_pathlines_after_pipeline = False
        self.log(f"PIPELINE ERROR:\n{str(error_text).strip()}")
        self.statusBar().showMessage(f"{self._pipeline_task_label} failed", 10000)
        self._close_progress_dialog(self._pipeline_progress_dialog)
        self._pipeline_progress_dialog = None

    def _cleanup_pipeline_task(self):
        if self._pipeline_worker is not None:
            self._pipeline_worker.deleteLater()
        if self._pipeline_thread is not None:
            self._pipeline_thread.deleteLater()
        self._pipeline_worker = None
        self._pipeline_thread = None
        self._pipeline_task_label = ""
        if self._run_all_pathlines_after_pipeline:
            self._run_all_pathlines_after_pipeline = False
            QtCore.QTimer.singleShot(0, self._finish_run_all_hemodynamics)

    def _finish_run_all_hemodynamics(self):
        """Add live flow views after the background Hemodynamics stages finish."""
        if self._pathline_running_guard("finishing Hemodynamics Run All"):
            return
        try:
            self.pipeline.preprocess(self.workspace)
            self.scene.trigger_streamlines()
            self._refresh_browser()
            self.scene.invalidate_cache("streamlines")
            self.scene.sync_from_workspace()
            self._refresh_all()
            self.log("[Run All: Hemodynamics] Live streamlines enabled; generating pathlines for all planes.")
            self._trigger_pathlines(
                plane_indices=list(range(len(self.workspace.planes))),
                prepare_segmentation=False,
            )
        except Exception as exc:
            self.log(f"RUN ALL FLOW VISUALIZATION ERROR: {type(exc).__name__}: {exc}")
            self.log(traceback.format_exc())

    def _run_single_step(self, step):
        if self._autoseg_running_guard("running pipeline steps"):
            return
        if self._pathline_running_guard("running pipeline steps"):
            return
        if self._pipeline_running_guard("running another pipeline step"):
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
            self._start_pipeline_task([step], step.label)
        except Exception as e:
            self.log(f"STEP ERROR: {type(e).__name__}: {e}")
            self.log(traceback.format_exc())

    def _revert_phase_unwrap(self):
        if self._pipeline_running_guard("restoring loaded flow"):
            return
        if self.workspace.flow_input is None or "flow_unwrapped" not in (self.workspace.phase_unwrap_result or {}):
            self.log("No active phase-unwrapping result is available to revert.")
            return
        result = self.pipeline.revert_phase_unwrap(self.workspace)
        self.log(result.message)
        self.scene.invalidate_cache()
        self.scene.sync_from_workspace()
        self._refresh_all()

    def _run_all_pipeline(self):
        if self._autoseg_running_guard("running the full pipeline"):
            return
        if self._pathline_running_guard("running the full pipeline"):
            return
        if self._pipeline_running_guard("running the full pipeline"):
            return
        if not self.workspace.data_loaded:
            self.log("No data loaded. Use File > Open H5 or Import DICOM Directory.")
            return
        if self._edit_mode is not None:
            self.log("Finish current interactive edit first.")
            return
        self._sync_params_to_ws()
        self._clear_plane_drag_widgets()
        stage_key = self._workflow_stage_key()
        all_steps = _workflow_run_all_steps(stage_key)
        if not all_steps:
            self.log("Run All is only available in Centerline & Planes or Hemodynamics.")
            return
        stage_label = next(
            label for label, key in _WORKFLOW_STAGES if key == stage_key
        )
        self._run_all_pathlines_after_pipeline = stage_key == "hemodynamics"
        if not self._start_pipeline_task(all_steps, f"Run All: {stage_label}"):
            self._run_all_pathlines_after_pipeline = False

    def _on_plane_streamlines_step(self):
        ws = self.workspace
        if len(ws.planes) == 0:
            self.log("No planes available for Pathlines.")
            return
        # The workflow action is the explicit all-plane operation. Plane context
        # menus still call _trigger_pathlines directly for one or more planes.
        self._trigger_pathlines(plane_indices=list(range(len(ws.planes))))

    def _get_spacing_xyz_from_resolution(self):
        r = np.asarray(self.workspace.resolution, dtype=float).reshape(-1)
        if r.size >= 3:
            return np.array([float(r[0]), float(r[1]), float(r[2])], dtype=float)
        return np.array([1.0, 1.0, 1.0], dtype=float)

    def _edit_widget_radius(self):
        spacing = self._get_spacing_xyz_from_resolution()
        return max(0.1, float(np.mean(spacing)) * 0.6)

    def _display_edit_points(self, points):
        values = np.asarray(points, dtype=float).reshape(-1, 3)
        if self.scene is None:
            return values.copy()
        origin = np.asarray(self.workspace.origin, dtype=float).reshape(1, 3)
        return self.scene.world_to_display_points(values + origin) - origin

    def _display_to_edit_point(self, point):
        value = np.asarray(point, dtype=float).reshape(3)
        if self.scene is None:
            return value.copy()
        origin = np.asarray(self.workspace.origin, dtype=float).reshape(3)
        return self.scene.display_to_world_point(value + origin) - origin

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
        group_text = self._edit_group_name or "Global"
        sel = "-" if self._edit_selected_idx is None else str(int(self._edit_selected_idx))
        esel = "-" if self._edit_selected_edge_idx is None else str(int(self._edit_selected_edge_idx))
        selected_label = "Selected Point" if mode == "skeleton" else "Selected Node"
        self._edit_info_label.setText(
            f"Group: {group_text}    Mode: {mode}    Points: {npts}    Edges: {nedges}    "
            f"{selected_label}: {sel}    Selected Edge: {esel}"
        )
        edge_state = "ON" if self._edit_edge_mode else "OFF"
        src = "-" if self._edit_edge_src_idx is None else str(int(self._edit_edge_src_idx))
        if mode == "skeleton":
            self._edit_status_label.setText("Point editing active    Keys: Delete/Backspace delete, Esc cancel")
        else:
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
        display_points = self._display_edit_points(self._edit_points)
        if self._edit_poly is None:
            self._edit_poly = pv.PolyData(display_points)
        else:
            self._edit_poly.points = display_points
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
        poly = self._graph_polydata(self._display_edit_points(self._edit_points), self._edit_edges)
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
        p = self._display_edit_points(
            np.asarray(self._edit_points[self._edit_selected_idx], dtype=float).reshape(1, 3)
        )
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
            c = self._display_to_edit_point(new_center)
            self._edit_points[self._edit_selected_idx, :] = c
            if self._edit_poly is not None:
                try:
                    self._edit_poly.points = self._display_edit_points(self._edit_points)
                except Exception:
                    pass
            if self._edit_sel_poly is not None:
                try:
                    self._edit_sel_poly.points = self._display_edit_points(np.asarray([c], dtype=float))
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
                p = self._display_to_edit_point(self._vtk_point_picker.GetPickPosition())
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
        pts = self._display_edit_points(self._edit_points[edge])
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

    def _snapshot_edit_visibility(self):
        self._edit_segmentation_visible_snapshot = bool(
            getattr(getattr(self.workspace, "segmentation", None), "visible", True)
        )
        self._edit_visibility_snapshot = {
            str(uid): (
                str(getattr(obj, "data_key", "") or ""),
                str(getattr(obj, "group_name", "") or ""),
                bool(getattr(obj, "visible", True)),
            )
            for uid, obj in self.workspace.scene_objects.items()
        }

    def _set_edit_scene_visibility(self, mode):
        """Show only the object being edited while retaining the browser state."""
        prefix = "skeleton_" if str(mode) == "skeleton" else "graph_"
        edit_group = str(self._edit_group_name or "")
        if hasattr(self.workspace, "segmentation"):
            self.workspace.segmentation.visible = False
        for obj in self.workspace.scene_objects.values():
            is_target = (
                str(getattr(obj, "data_key", "") or "").startswith(prefix)
                and str(getattr(obj, "group_name", "") or "") == edit_group
            )
            obj.visible = bool(is_target)
            self.scene.apply_object_properties(obj, render=False, refresh_scalar_bar=False)
        self._refresh_browser()
        self.scene.sync_from_workspace()

    def _restore_edit_visibility(self):
        snapshot = dict(self._edit_visibility_snapshot or {})
        by_key = {
            (data_key, group_name): visible
            for data_key, group_name, visible in snapshot.values()
        }
        for uid, obj in self.workspace.scene_objects.items():
            entry = snapshot.get(str(uid))
            if entry is None:
                entry_visible = by_key.get(
                    (str(getattr(obj, "data_key", "") or ""), str(getattr(obj, "group_name", "") or "")),
                    None,
                )
            else:
                entry_visible = entry[2]
            if entry_visible is not None:
                obj.visible = bool(entry_visible)
                self.scene.apply_object_properties(obj, render=False, refresh_scalar_bar=False)
        if hasattr(self.workspace, "segmentation") and self._edit_segmentation_visible_snapshot is not None:
            self.workspace.segmentation.visible = bool(self._edit_segmentation_visible_snapshot)
        self._edit_visibility_snapshot = None
        self._edit_segmentation_visible_snapshot = None
        self._refresh_browser()

    def _show_interactive_overlay(self):
        mode = self._edit_mode or "-"
        if mode == "skeleton":
            hint = "Click a point to select it. Drag the orange sphere to move it. Delete/Backspace removes the selected point."
        elif mode == "graph":
            hint = "Click a node to select it, or click an edge to select it. Drag the orange sphere to move a node. Delete/Backspace removes the selection."
        else:
            hint = f"Edit {mode}: ESC to cancel"
        group_text = self._edit_group_name or "Global"
        if self._edit_panel is not None:
            self._edit_panel.setTitle(f"Editing {mode.title()} - {group_text}")
            self._edit_panel.setVisible(True)
        if self.steps_group is not None:
            self.steps_group.setVisible(False)
        if hasattr(self, "tree_objects"):
            self.tree_objects.setEnabled(False)
        if hasattr(self, "btn_delete_obj"):
            self.btn_delete_obj.setEnabled(False)
        if hasattr(self, "workflow_tabs"):
            self.workflow_tabs.setEnabled(False)
        if hasattr(self, "btn_show_advanced"):
            self.btn_show_advanced.setEnabled(False)
        if self._edit_btn_edge is not None:
            self._edit_btn_edge.setVisible(mode == "graph")
        if getattr(self, "_edit_hint_label", None) is not None:
            self._edit_hint_label.setText(
                f"{hint}\n\nSave Changes applies the edit and rebuilds dependent objects. Cancel or Esc discards it."
            )
        self._update_edit_labels()
        try:
            self.statusBar().showMessage(hint)
        except Exception:
            pass
        self.log(hint)

    def _close_interactive_overlay(self):
        if self._edit_panel is not None:
            self._edit_panel.setVisible(False)
        if self.steps_group is not None:
            self.steps_group.setVisible(True)
        if hasattr(self, "tree_objects"):
            self.tree_objects.setEnabled(True)
        if hasattr(self, "btn_delete_obj"):
            self.btn_delete_obj.setEnabled(True)
        if hasattr(self, "workflow_tabs"):
            self.workflow_tabs.setEnabled(True)
        if hasattr(self, "btn_show_advanced"):
            self.btn_show_advanced.setEnabled(True)
        try:
            self.statusBar().clearMessage()
        except Exception:
            pass

    def _enter_interactive_edit(self, mode, points, edges=None, *, group_name=""):
        self._snapshot_edit_visibility()
        self.scene.invalidate_cache()
        self.scene.sync_from_workspace()
        self._cleanup_edit_actors()
        self._remove_edit_widget()
        self._close_interactive_overlay()
        self._edit_mode = mode
        self._edit_group_name = str(group_name or "")
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
        self._set_edit_scene_visibility(mode)
        self._update_edit_points_actor()
        self._update_edit_edges_actor()
        self._enable_interactive_key_events(True)
        self._enable_interactive_point_picking(True)
        self._show_interactive_overlay()
        if len(self._edit_points) > 0:
            self._set_selected_idx(0)
        else:
            self._set_selected_idx(None)
        self.log(f"Interactive edit started: {mode} group={self._edit_group_name or 'Global'}")
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
                ws = self.workspace
                active_group = str(self._edit_group_name or "")
                if mode == "skeleton":
                    edited_points = np.asarray(self._edit_points, dtype=float).reshape(-1, 3)
                    if active_group in ws.multilabel_groups:
                        group_state = dict(ws.multilabel_groups[active_group])
                        group_state["skeleton_points"] = edited_points
                        group_state["graph"] = ws.graph.__class__()
                        ws.multilabel_groups[active_group] = group_state
                        ws.remove_object_by_data_key(f"skeleton_{active_group}")
                        ws.remove_object_by_data_key(f"graph_{active_group}")
                        ws.add_object(
                            name=f"skeleton_{active_group}",
                            kind=ObjectKind.SKELETON,
                            data_key=f"skeleton_{active_group}",
                            group_name=active_group,
                            browser_color=ws.skeleton_params.browser_color_for_group(active_group),
                            visible=True,
                            opacity=1.0,
                            color=ws.skeleton_params.scene_color_for_group(active_group, "skeleton"),
                            point_size=8,
                        )
                        self._rebuild_workspace_skeleton_points()
                        self._rebuild_workspace_graph()
                    else:
                        ws.skeleton_points = edited_points
                        ws.graph = ws.graph.__class__()
                    self._clear_centerline_edit_dependents()
                    ws.pipeline.completed.pop(StepId.GENERATE_GRAPH.value, None)
                    ws.pipeline.skipped.pop(StepId.GENERATE_GRAPH.value, None)
                    ws.pipeline.mark_done(StepId.EDIT_SKELETON, skipped=False)
                    self.log(
                        f"Skeleton edited: group={active_group or 'Global'} "
                        f"points={len(edited_points)}"
                    )
                elif mode == "graph":
                    edited_points = np.asarray(self._edit_points, dtype=float).reshape(-1, 3)
                    raw_edges = np.asarray(self._edit_edges, dtype=int).reshape(-1, 2) if len(self._edit_edges) else np.empty((0, 2), dtype=int)
                    valid = (
                        (raw_edges[:, 0] >= 0) & (raw_edges[:, 1] >= 0)
                        & (raw_edges[:, 0] < len(edited_points)) & (raw_edges[:, 1] < len(edited_points))
                        & (raw_edges[:, 0] != raw_edges[:, 1])
                    ) if len(raw_edges) else np.zeros(0, dtype=bool)
                    canonical = {tuple(sorted((int(a), int(b)))) for a, b in raw_edges[valid]}
                    edited_edges = (
                        np.asarray(sorted(canonical), dtype=int).reshape(-1, 2)
                        if canonical else np.empty((0, 2), dtype=int)
                    )
                    if active_group in ws.multilabel_groups:
                        group_state = dict(ws.multilabel_groups[active_group])
                        group_graph = ws.graph.__class__(points=edited_points, edges=edited_edges)
                        group_state["graph"] = group_graph
                        ws.multilabel_groups[active_group] = group_state
                        ws.remove_object_by_data_key(f"graph_{active_group}")
                        ws.add_object(
                            name=f"graph_{active_group}",
                            kind=ObjectKind.GRAPH,
                            data_key=f"graph_{active_group}",
                            group_name=active_group,
                            browser_color=ws.skeleton_params.browser_color_for_group(active_group),
                            visible=True,
                            opacity=1.0,
                            color=ws.skeleton_params.scene_color_for_group(active_group, "graph"),
                            line_width=2,
                        )
                        self._rebuild_workspace_graph()
                    else:
                        ws.graph = ws.graph.__class__(points=edited_points, edges=edited_edges)
                    self._clear_centerline_edit_dependents()
                    ws.pipeline.mark_done(StepId.EDIT_GRAPH, skipped=False)
                    self.log(
                        f"Graph edited: group={active_group or 'Global'} "
                        f"nodes={len(edited_points)} edges={len(edited_edges)}"
                    )
            else:
                self.log(f"Interactive edit cancelled: {mode}")
        finally:
            self._restore_edit_visibility()
            self._cleanup_edit_actors()
            self._close_interactive_overlay()
            self._edit_mode = None
            self._edit_group_name = ""
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

    def _rebuild_workspace_skeleton_points(self):
        ws = self.workspace
        point_sets = []
        masks = []
        for group_name in ws.group_order:
            group_state = ws.multilabel_groups.get(group_name, {})
            points = group_state.get("skeleton_points")
            if points is not None and len(points):
                point_sets.append(np.asarray(points, dtype=float).reshape(-1, 3))
            mask = group_state.get("skeleton_mask")
            if mask is not None:
                masks.append(np.asarray(mask, dtype=bool))
        ws.skeleton_points = (
            np.vstack(point_sets) if point_sets else np.empty((0, 3), dtype=float)
        )
        if masks:
            skeleton_mask = np.zeros_like(masks[0], dtype=bool)
            for mask in masks:
                skeleton_mask |= mask
            ws.skeleton_mask = skeleton_mask

    def _rebuild_workspace_graph(self):
        ws = self.workspace
        point_sets = []
        edge_sets = []
        node_offset = 0
        for group_name in ws.group_order:
            group_state = ws.multilabel_groups.get(group_name, {})
            graph = group_state.get("graph")
            points = np.asarray(getattr(graph, "points", np.empty((0, 3))), dtype=float).reshape(-1, 3)
            edges = np.asarray(getattr(graph, "edges", np.empty((0, 2))), dtype=int)
            edges = edges.reshape(-1, 2) if edges.size else np.empty((0, 2), dtype=int)
            if len(edges):
                valid = (
                    (edges[:, 0] >= 0) & (edges[:, 1] >= 0)
                    & (edges[:, 0] < len(points)) & (edges[:, 1] < len(points))
                    & (edges[:, 0] != edges[:, 1])
                )
                edges = edges[valid]
            if len(points):
                point_sets.append(points)
            if len(edges):
                edge_sets.append(edges + node_offset)
            node_offset += len(points)
        ws.graph = ws.graph.__class__(
            points=np.vstack(point_sets) if point_sets else np.empty((0, 3), dtype=float),
            edges=np.vstack(edge_sets) if edge_sets else np.empty((0, 2), dtype=int),
        )

    def _clear_centerline_edit_dependents(self):
        """Remove paths, planes, and metrics that no longer match edited topology."""
        ws = self.workspace
        ws.centerline_paths = []
        ws.centerline_node_paths = []
        ws.centerline_paths_smooth = []
        ws.path_info = []
        ws.forks = []
        ws.planes = []
        ws.selected_path_index = -1
        if ws.segmask_3d is not None:
            ws.branch_labels = np.zeros_like(ws.segmask_3d, dtype=np.int16)
        for group_name in ws.group_order:
            group_state = dict(ws.multilabel_groups.get(group_name, {}))
            group_state["branch_labels"] = None
            group_state["centerline_paths"] = []
            group_state["centerline_node_paths"] = []
            group_state["centerline_paths_smooth"] = []
            group_state["path_info"] = []
            group_state["forks"] = []
            group_state["planes"] = []
            group_state["path_index_offset"] = 0
            group_state["plane_index_offset"] = 0
            ws.multilabel_groups[group_name] = group_state
        ws.clear_pathlines()
        ws.pathline_colors = {}
        ws.remove_object_by_data_key("branch_surface")
        ws.remove_objects_by_prefix("forks_")
        ws.remove_objects_by_prefix("path_")
        ws.remove_objects_by_prefix("smooth_path_")
        ws.remove_objects_by_prefix("path_arrow_")
        ws.remove_objects_by_prefix("plane_")
        ws.derived.plane_metrics = []
        ws.derived.plane_qc = {}
        self.pipeline._clear_pwv_state(ws)
        for step in (
            StepId.GENERATE_PLANES,
            StepId.COMPUTE_PLANE_METRICS,
            StepId.COMPUTE_PWV,
            StepId.COMPUTE_DERIVED_METRICS,
            StepId.PLANE_STREAMLINES,
        ):
            ws.pipeline.completed.pop(step.value, None)
            ws.pipeline.skipped.pop(step.value, None)

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
            self._edit_group_name = ""
            self._edit_points = None
            self._edit_edges = None
            self._edit_selected_idx = None
            self._edit_edge_mode = False
            self._edit_edge_src_idx = None
            self._edit_selected_edge_idx = None
            self._edit_original_points = None
            self._edit_original_edges = None
            self._edit_overlay_dialog = None
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

    def _edit_group_candidates(self, mode):
        field = "skeleton_points" if mode == "skeleton" else "graph"
        candidates = []
        for group_name in self.workspace.group_order:
            group_state = self.workspace.multilabel_groups.get(group_name, {})
            value = group_state.get(field)
            point_count = len(value) if mode == "skeleton" and value is not None else len(getattr(value, "points", []))
            if point_count:
                candidates.append(str(group_name))
        return candidates

    def _choose_edit_group(self, mode):
        candidates = self._edit_group_candidates(mode)
        if not candidates:
            return None
        selected_uid = self._selected_uid()
        selected_object = self.workspace.scene_objects.get(selected_uid) if selected_uid else None
        selected_group = str(getattr(selected_object, "group_name", "") or "")
        if selected_group in candidates:
            return selected_group
        selected_plane_group = self._group_name_for_plane(self._selected_plane_index)
        if selected_plane_group in candidates:
            return selected_plane_group
        if len(candidates) == 1:
            return candidates[0]
        group_name, accepted = QtWidgets.QInputDialog.getItem(
            self,
            f"Edit {mode.title()}",
            "Vessel group:",
            candidates,
            0,
            False,
        )
        return str(group_name) if accepted else ""

    def _start_skeleton_interactive_edit(self):
        group_name = self._choose_edit_group("skeleton")
        if group_name == "":
            return
        if group_name is not None:
            points = self.workspace.multilabel_groups[group_name].get("skeleton_points")
            self._enter_interactive_edit("skeleton", points, edges=None, group_name=group_name)
            return
        if not self.workspace.group_order and self.workspace.skeleton_points is not None and len(self.workspace.skeleton_points):
            self._enter_interactive_edit("skeleton", self.workspace.skeleton_points, edges=None)
            return
        self.log("Edit Skeleton: no skeleton points are available for a vessel group.")

    def _start_graph_interactive_edit(self):
        group_name = self._choose_edit_group("graph")
        if group_name == "":
            return
        if group_name is not None:
            graph = self.workspace.multilabel_groups[group_name].get("graph")
            self._enter_interactive_edit("graph", graph.points, graph.edges, group_name=group_name)
            return
        if not self.workspace.group_order and self.workspace.graph is not None and len(self.workspace.graph.points):
            self._enter_interactive_edit("graph", self.workspace.graph.points, self.workspace.graph.edges)
            return
        self.log("Edit Graph: no graph nodes are available for a vessel group.")

    def closeEvent(self, event):
        if (
            self._labeler_process is not None
            and self._labeler_process.state() != QtCore.QProcess.ProcessState.NotRunning
        ):
            self.log("Close SpatioTemporal Labeler before closing AutoFlow so its result can be imported.")
            event.ignore()
            return
        if self._autoseg_thread is not None:
            self.log("Auto segmentation is running. Wait for it to finish before closing the GUI.")
            event.ignore()
            return
        if self._pipeline_thread is not None:
            self.log("A pipeline task is running. Wait for it to finish before closing the GUI.")
            event.ignore()
            return
        if self._pathline_thread is not None:
            self.log("Pathlines are running. Wait for them to finish before closing the GUI.")
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
    configure_high_dpi()
    app = QtWidgets.QApplication(sys.argv)
    apply_application_theme(app)
    signal.signal(signal.SIGINT, lambda *_args: app.quit())
    signal_timer = QtCore.QTimer()
    signal_timer.timeout.connect(lambda: None)
    signal_timer.start(100)
    w = MainWindow(config_dir=config_dir)
    w.show()
    w.raise_()
    w.activateWindow()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
