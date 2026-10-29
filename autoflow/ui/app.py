from .selection_info import SelectionInfoPanel, consistency_rows
from .browser import BrowserTree, ContextStack, INLINE_ROLE
from .measurement_summary import MeasurementSummary
from .pinned_metrics import pinned_metric_keys, pinned_metric_rows
from .content_controls import ContentRenderControls
from .workspace_layout import WorkspaceLayoutMixin
from ..config import BACKGROUND_COLOR
import hashlib
import json
import copy
from collections import deque
from dataclasses import dataclass
import importlib.util
import os
import re
import signal
import sys
import time
import traceback
import uuid
from contextlib import nullcontext
from functools import partial, wraps
from pathlib import Path

os.environ["QT_API"] = "pyside6"

import numpy as np
import pyvista as pv
from PySide6 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from pyvista import _vtk
from scipy.interpolate import make_interp_spline
from scipy.ndimage import label as ndi_label

if os.environ.get("AUTOFLOW_SSH_RENDERING") == "1":
    from .remote_plotter import RemotePlotter
else:
    from pyvistaqt import QtInteractor

from ..algorithms import (
    resolve_input_case,
    load_segmentation_file,
    generate_threshold_segmentation,
    generate_nnunet_auto_segmentation,
    save_segmentation_file,
    save_segmentation_to_source_h5,
    segmentation_timestamp,
    _plot_plane_flowrate_axes,
    _plot_pwv_axes,
    _project_point_to_path,
    filter_paths_by_segmentation,
    generate_pathlines_from_plane_at_t,
)
from ..algorithms.data import discover_h5_input_cases, inspect_h5_input_case
from ..algorithms.dicom_conversion import convert_dicom_input
from ..algorithms.segmentation import (
    resolve_nnunet_model_folder,
    resolve_nnunet_4d_model_folder,
)
from ..core.models import ObjectKind, PlaneData, StepId, Workspace
from ..core.pipeline import PipelineEngine
from ..task_control import CancellationToken, TaskCancelled, check_cancelled, task_scope
from .progress import TaskProgressDialog
from .plane_handles import PlaneDragHandles
from .time_navigation import FrameNavigationFilter
from .plane_editing import PlaneEditingMixin, _PlaneCommand
from .review import ReviewMixin
from .report_actions import ReportActionsMixin
from .report_actions import ReportActionsMixin
from .analysis_plotting import (update_plane_curve, move_phase_cursor,
                                style_analysis_figure, analysis_palette)
from ..core.plane_state import (metric_validity, mutation_snapshot, remove_planes, selected_metric,
                                invalidate_planes, refresh_plane_consistency)
from ..config import apply_config_bundle_to_workspace, bundle_to_autoflow_kwargs, load_config_bundle
from ..plane_io import load_plane_position_payload, project_planes_to_workspace, save_plane_positions
from ..quality import build_quality_report, save_quality_report
from ..algorithms import compute_plane_metrics, apply_internal_consistency_to_metrics, compute_plane_metrics_multithread, augment_plane_metrics_with_derived, save_plane_pixelwise_h5
from ..algorithms.streamlines import _plane_seeds, create_pathline_temporal_source, pathline_sample_times
from ..algorithms.flow_seeding import plane_seed_points, phase_mask
from ..core.plane_state import workspace_signature, plane_signature
from .editors import PlaneEditor
from .input_dialogs import DualVencSelectDialog, H5CaseSelectDialog
from .action_settings import ActionSettingsMixin, action_sections
from .ortho_viewer import OrthoViewer
from .segmentation import SegmentationDock, SOURCE_LABELS
from .theme import (apply_application_theme, configure_high_dpi, standard_icon,
                    theme_colors, readable_color, workflow_status_colors)
from ..rendering import (
    render_plane_rotation_video,
    render_pressure_gradient_video,
    render_relative_pressure_video,
    render_streamlines_video,
    render_tke_video,
    render_wss_video,
)
from ..rendering.style import plane_display_size
from .viewer import SceneController
from .labeler_exchange import export_labeler_exchange


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


def _flowrate_display_curve(values, order=3):
    """Interpolate finite runs for display without changing measured samples."""
    values = np.asarray(values, dtype=float).reshape(-1)
    phases = np.arange(values.size, dtype=float)
    if values.size < 2 or int(order) <= 1:
        return phases, values
    # Include every measured phase exactly, and keep missing phases as gaps.
    dense_phases = np.linspace(0.0, values.size - 1, (values.size - 1) * 16 + 1)
    dense_values = np.full(dense_phases.shape, np.nan)
    finite_indices = np.flatnonzero(np.isfinite(values))
    for run in np.split(finite_indices, np.flatnonzero(np.diff(finite_indices) > 1) + 1):
        if run.size == 0:
            continue
        start, stop = int(run[0]) * 16, int(run[-1]) * 16 + 1
        if run.size == 1:
            dense_values[start] = values[run[0]]
            continue
        degree = min(max(1, int(order)), 3, run.size - 1)
        spline = make_interp_spline(phases[run], values[run], k=degree)
        dense_values[start:stop] = spline(dense_phases[start:stop])
    return dense_phases, dense_values


_WORKFLOW_STAGES = [
    ("Input & QC", "input"),
    ("Correction", "correction"),
    ("Segmentation", "segmentation"),
    ("Centerline & Planes", "centerline"),
    ("Hemodynamics", "hemodynamics"),
    ("Review QC", "review"),
    ("Export", "export"),
]

_WORKFLOW_STEPS = {
    "input": set(),
    "segmentation": set(),
    "correction": {StepId.BACKGROUND_CORRECTION, StepId.REMOVE_NOISE, StepId.UNWRAP_PHASE, StepId.GENERATE_PCMRA},
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
    "export": set(),
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
    "correction": (StepId.BACKGROUND_CORRECTION, StepId.REMOVE_NOISE, StepId.UNWRAP_PHASE, StepId.GENERATE_PCMRA),
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


def _cancellable_worker(method):
    @wraps(method)
    def run(self):
        try:
            with task_scope(self.cancel_token):
                return method(self)
        except TaskCancelled:
            self.failed.emit("Cancelled")
    return run


class _TaskWorker(QtCore.QObject):
    def __init__(self):
        super().__init__()
        self.cancel_token = CancellationToken()


class _FunctionTaskWorker(_TaskWorker):
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    progress = QtCore.Signal(object)
    log_message = QtCore.Signal(str)

    def __init__(self, function):
        super().__init__()
        self.function = function

    @_cancellable_worker
    def run(self):
        try:
            with task_scope(self.cancel_token, self.progress.emit):
                result = self.function(self.progress.emit, self.log_message.emit)
                check_cancelled()
            self.finished.emit(result)
        except TaskCancelled:
            raise
        except Exception:
            self.failed.emit(traceback.format_exc())


class _AutoSegmentationWorker(_TaskWorker):
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
        check_cancelled()
        data = dict(payload or {})
        if int(data.get("total") or 0) == 5:
            data["total"] = 7
        self.progress.emit(data)

    @QtCore.Slot()
    @_cancellable_worker
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
                # Keep the GUI process free of the in-process torch predictor.
                # Native CUDA/Qt failures must be contained by the nnUNet
                # subprocess instead of terminating the application.
                grouped_preprocessing=None,
            )
            check_cancelled()
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
            check_cancelled()
            self.finished.emit(_AutoSegmentationResult(seg=seg, provenance=provenance, cache_path=cache_path, elapsed_sec=float(elapsed)))
        except TaskCancelled:
            raise
        except Exception:
            self.failed.emit(traceback.format_exc())


class _DicomConversionWorker(_TaskWorker):
    progress = QtCore.Signal(dict)
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, root, output_h5):
        super().__init__()
        self.root = root
        self.output_h5 = output_h5

    @QtCore.Slot()
    @_cancellable_worker
    def run(self):
        try:
            result = convert_dicom_input(self.root, self.output_h5, progress_callback=self.progress.emit)
            check_cancelled()
            self.finished.emit(result)
        except TaskCancelled:
            raise
        except Exception:
            self.failed.emit(traceback.format_exc())


class _PipelineTaskWorker(_TaskWorker):
    progress = QtCore.Signal(dict)
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, engine, workspace, steps):
        super().__init__()
        self._engine = engine
        self._workspace = workspace
        self._steps = list(steps)
        self.completed_steps = []

    @QtCore.Slot()
    @_cancellable_worker
    def run(self):
        started = time.perf_counter()
        results = []
        try:
            for index, step in enumerate(self._steps, start=1):
                check_cancelled()
                self.progress.emit({
                    "stage": "start",
                    "current": index - 1,
                    "total": len(self._steps),
                    "step": step,
                    "message": f"Running {step.label}...",
                })
                step_started = time.perf_counter()
                def _step_progress(payload, *, _step=step):
                    check_cancelled()
                    data = dict(payload or {})
                    data.setdefault("step", _step)
                    data.setdefault("detail_current", data.get("current", 0))
                    data.setdefault("detail_total", data.get("total", 0))
                    data["current"] = index - 1
                    data["total"] = len(self._steps)
                    self.progress.emit(data)
                candidate = self._workspace.copy_for_task()
                step_kwargs = {"force_recompute_corr": True} if step == StepId.BACKGROUND_CORRECTION else {}
                with task_scope(self.cancel_token, _step_progress):
                    result = self._engine.run_step(
                        candidate, step, lambda _message: None,
                        progress_callback=_step_progress,
                        **step_kwargs,
                    )
                check_cancelled()
                elapsed = time.perf_counter() - step_started
                if not result.success:
                    raise RuntimeError(f"{step.label}: {result.message}")
                self._workspace.__dict__.update(vars(candidate))
                results.append((step, result, float(elapsed)))
                self.completed_steps.append(step)
                self.progress.emit({
                    "stage": "done",
                    "current": index,
                    "total": len(self._steps),
                    "step": step,
                    "message": f"{step.label} completed",
                    "elapsed_sec": float(elapsed),
                    "result_message": str(result.message),
                })
            report_result = {}
            report_steps = {
                StepId.GENERATE_SKELETON, StepId.GENERATE_GRAPH, StepId.GENERATE_PLANES,
                StepId.COMPUTE_PLANE_METRICS, StepId.COMPUTE_DERIVED_METRICS, StepId.COMPUTE_PWV,
            }
            if report_steps.intersection(self.completed_steps):
                self.progress.emit({"stage": "report", "current": len(self._steps),
                                    "total": len(self._steps), "message": "Writing single-case analysis HTML..."})
                report_started = time.perf_counter()
                try:
                    from ..report_snapshot import build_report_snapshot, save_report_html
                    snapshot = build_report_snapshot(self._workspace)
                    report_path = save_report_html(snapshot, os.path.join(
                        self._engine._output_dir(self._workspace), "analysis_report.html"))
                    report_result = {"status": "complete", "file": report_path,
                                     "snapshot_id": snapshot["snapshot_id"]}
                except TaskCancelled:
                    raise
                except Exception as exc:
                    report_result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                report_result["elapsed_sec"] = time.perf_counter() - report_started
            self.finished.emit({
                "results": results,
                "elapsed_sec": float(time.perf_counter() - started),
                "analysis_report": report_result,
            })
        except TaskCancelled:
            raise
        except Exception:
            self.failed.emit(traceback.format_exc())


class _PathlineTaskWorker(_TaskWorker):
    progress = QtCore.Signal(dict)
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, flow, mask_4d, branch_labels, planes, plane_indices, *,
                 time_index, spacing, origin, rr, params, seed_segmentation=None, segmentation_labels=None):
        super().__init__()
        self._flow, self._mask_4d, self._branch_labels = flow, mask_4d, branch_labels
        self._planes = [copy.deepcopy(plane) for plane in planes]
        self._plane_indices = [int(index) for index in plane_indices]
        self._time_index = int(time_index)
        self._spacing, self._origin = np.asarray(spacing, float).copy(), np.asarray(origin, float).copy()
        self._rr, self._params = float(rr), dict(params or {})
        self._seed_segmentation = seed_segmentation if seed_segmentation is not None else (
            mask_4d if self._params.get("pathline_seed_region", "segmentation") == "segmentation" else None)
        self._segmentation_labels = segmentation_labels

    @QtCore.Slot()
    @_cancellable_worker
    def run(self):
        started = time.perf_counter()
        try:
            results, seed_results, roi_sources = {}, {}, {}
            total = len(self._plane_indices)
            mask_t = phase_mask(self._mask_4d, self._flow.shape[:3], self._time_index)
            temporal_cache_mb = self._params["pathline_temporal_cache_mb"] if total > 1 else 0.0
            temporal_source = create_pathline_temporal_source(self._flow, self._mask_4d,
                self._spacing, self._origin, self._time_index, self._rr, temporal_cache_mb=temporal_cache_mb)
            frame_steps = len(pathline_sample_times(temporal_source, self._flow,
                self._params.get("pathline_integration_step_mm"), self._params["pathline_max_steps"]))
            for current, index in enumerate(self._plane_indices, start=1):
                check_cancelled()
                seeds, roi_source = plane_seed_points(self._flow, self._planes[index], self._spacing, self._origin,
                    phase=self._time_index, region=self._params.get("pathline_seed_region", "segmentation"),
                    segmentation=self._seed_segmentation, segmentation_labels=self._segmentation_labels,
                    branch_labels=self._branch_labels,
                    distribution=self._params.get("pathline_seed_distribution", "random"),
                    grid_spacing_mm=self._params.get("pathline_grid_spacing_mm", 1.),
                    mode=self._params["pathline_seed_mode"], count=self._params["pathline_max_seeds"],
                    ratio=self._params["pathline_seed_ratio"], min_seeds=self._params["pathline_min_seeds"],
                    rng_seed=self._params["pathline_rng_seed"])
                seed_results[index], roi_sources[index] = seeds, roi_source
                def progress(step, step_total, *, _index=index, _current=current):
                    check_cancelled()
                    self.progress.emit({"current": (_current-1)*frame_steps+step,
                        "total": total*frame_steps, "message": f"Integrating plane {_index}: {len(seeds):,} seeds",
                        "elapsed_sec": time.perf_counter()-started})
                mesh = generate_pathlines_from_plane_at_t(self._flow, self._time_index, self._planes[index],
                    self._spacing, self._origin, mask_4d=self._mask_4d, mask_3d=mask_t,
                    max_steps=self._params["pathline_max_steps"], terminal_speed=self._params["pathline_terminal_speed"],
                    rng_seed=self._params["pathline_rng_seed"], rr=self._rr, seeds=seeds,
                    max_seeds=None if self._params.get("pathline_seed_distribution") == "uniform" else self._params["pathline_max_seeds"],
                    progress_callback=progress, temporal_source=temporal_source,
                    integration_step_mm=self._params.get("pathline_integration_step_mm"))
                if mesh is not None:
                    mesh.field_data["ROISourcePhase"] = np.asarray([-1 if roi_source is None else roi_source], np.int64)
                results[index] = {self._time_index: mesh}
            check_cancelled()
            self.finished.emit({"plane_indices": self._plane_indices, "meshes": results, "seeds": seed_results,
                "roi_source_frames": roi_sources, "launch_phase": self._time_index,
                "input_signature": self._params.get("input_signature"),
                "plane_signatures": self._params.get("plane_signatures"),
                "temporal_cache_all_phases": bool(temporal_source.cache_all_phases),
                "elapsed_sec": time.perf_counter()-started})
        except TaskCancelled:
            raise
        except Exception:
            self.failed.emit(traceback.format_exc())

class MainWindow(WorkspaceLayoutMixin, ReportActionsMixin, ReviewMixin, PlaneEditingMixin, ActionSettingsMixin, QtWidgets.QMainWindow):
    def __init__(self, config_dir=None):
        super().__init__()
        self.setWindowTitle("AutoFlow")
        self.setAcceptDrops(True)
        self.resize(1800, 980)
        self.setMinimumSize(1440, 800)
        self._config_dir = config_dir
        self._config_bundle = load_config_bundle(config_dir)
        apply_application_theme(QtWidgets.QApplication.instance(),
                                self._config_bundle.get("ui", {}).get("theme", "dark"))
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
        self._keyboard_navigation_active = False
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
        self._initialize_plane_transactions()
        self._plane_handles = None
        self._plane_edit_had_metrics = False
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
        self._dicom_conversion_thread = None
        self._dicom_conversion_worker = None
        self._dicom_conversion_progress = None
        self._pipeline_worker = None
        self._pipeline_progress_dialog = None
        self._pipeline_task_label = ""
        self._run_all_pathlines_after_pipeline = False
        self._pathline_thread = None
        self._pathline_worker = None
        self._pathline_progress_dialog = None
        self._pathline_selected_plane_idx = None
        self._labeler_process = None
        self._labeler_export_running = False
        self._labeler_segmentation_path = ""
        self._labeler_segmentation_mtime_ns = None
        self._labeler_process_output = {"stdout": "", "stderr": ""}
        self._last_quality_report = None
        self._last_plane_import_report = None
        self._active_input_case = None
        self._input_signature = None
        self._build_ui()
        self._bind_scene()
        self._frame_navigation_filter = FrameNavigationFilter(
            self, self._queue_keyboard_time_step, self._finish_keyboard_time_step)
        self._esc_shortcut = QtGui.QShortcut(QtCore.Qt.Key_Escape, self)
        self._esc_shortcut.setContext(QtCore.Qt.ApplicationShortcut)
        self._esc_shortcut.activated.connect(self._force_exit_edit)
        QtCore.QTimer.singleShot(0, self._setup_focus_behavior)
        self.ortho_viewer.timeStepRequested.connect(self._on_ortho_time_step)
        self.ortho_viewer.planeRoiChanged.connect(self._on_plane_roi_changed)
        self.ortho_viewer.planeRoiEditStarted.connect(self._begin_roi_transaction)
        self.ortho_viewer.planeRoiEditFinished.connect(self._finish_roi_transaction)
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
        root.addLayout(workflow_row)

        self.workspace_pages = QtWidgets.QStackedWidget()
        self.workspace_pages.setObjectName('workspacePages')
        root.addWidget(self.workspace_pages, 1)
        self.workspace_page = QtWidgets.QWidget()
        self.workspace_pages.addWidget(self.workspace_page)
        workspace_layout = QtWidgets.QVBoxLayout(self.workspace_page)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.setSpacing(8)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.workspace_splitter = splitter
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)
        workspace_layout.addWidget(splitter, 1)
        left = QtWidgets.QWidget()
        left.setMinimumWidth(300)
        left_lay = QtWidgets.QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(6)
        self.left_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.left_splitter.setChildrenCollapsible(False)
        self.left_splitter.setHandleWidth(6)
        browser_panel = QtWidgets.QWidget()
        browser_layout = QtWidgets.QVBoxLayout(browser_panel)
        browser_layout.setContentsMargins(0, 0, 0, 0)
        self._build_browser(browser_layout)
        self.left_splitter.addWidget(browser_panel)
        left_lay.addWidget(self.left_splitter)
        mid = QtWidgets.QWidget()
        mid.setMinimumWidth(420)
        mid_lay = QtWidgets.QVBoxLayout(mid)
        mid_lay.setContentsMargins(0, 0, 0, 0)
        mid_lay.setSpacing(6)
        if os.environ.get("AUTOFLOW_SSH_RENDERING") == "1":
            self.plotter = RemotePlotter(self)
        else:
            self.plotter = QtInteractor(self, auto_update=False)
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
        range_bar = QtWidgets.QGridLayout()
        range_bar.setVerticalSpacing(2)
        range_bar.addWidget(QtWidgets.QLabel("Window"), 0, 0)
        self.slider_render_window = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_render_window.setRange(1, 1000)
        self.slider_render_window.setEnabled(False)
        self.slider_render_window.setToolTip("Adjust window width")
        range_bar.addWidget(self.slider_render_window, 0, 1)
        self.label_render_window = QtWidgets.QLabel("—")
        self.label_render_window.setMinimumWidth(72)
        range_bar.addWidget(self.label_render_window, 0, 2)
        range_bar.addWidget(QtWidgets.QLabel("Level"), 1, 0)
        self.slider_render_level = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_render_level.setRange(0, 1000)
        self.slider_render_level.setEnabled(False)
        self.slider_render_level.setToolTip("Adjust window level")
        range_bar.addWidget(self.slider_render_level, 1, 1)
        self.label_render_level = QtWidgets.QLabel("—")
        self.label_render_level.setMinimumWidth(72)
        range_bar.addWidget(self.label_render_level, 1, 2)
        self.btn_reset_render_range = QtWidgets.QPushButton("Auto")
        self.btn_reset_render_range.setToolTip("Restore the automatic range for the selected scalar object")
        self.btn_reset_render_range.setEnabled(False)
        self.btn_reset_render_range.clicked.connect(self._reset_selected_render_range)
        range_bar.addWidget(self.btn_reset_render_range, 0, 3, 2, 1)
        range_bar.setColumnStretch(1, 1)
        self.slider_render_window.valueChanged.connect(self._on_render_range_changed)
        self.slider_render_level.valueChanged.connect(self._on_render_range_changed)
        self.render_range_controls = QtWidgets.QWidget()
        self.render_range_controls.setLayout(range_bar)
        self.appearance_layout.addWidget(self.render_range_controls)
        mid_splitter.addWidget(plotter_panel)
        step_and_params = QtWidgets.QWidget()
        sp_lay = QtWidgets.QVBoxLayout(step_and_params)
        sp_lay.setContentsMargins(0, 0, 0, 0)
        sp_lay.setSpacing(4)
        self._build_step_buttons(sp_lay)
        self.review_panel = QtWidgets.QScrollArea()
        self.review_panel.setWidgetResizable(True)
        self.review_panel.setFrameShape(QtWidgets.QFrame.NoFrame)
        review_widget = QtWidgets.QWidget()
        self.params_layout = QtWidgets.QVBoxLayout(review_widget)
        self._build_review_params()
        self.params_layout.addStretch()
        self.review_panel.setWidget(review_widget)
        sp_lay.addWidget(self.review_panel, 1)
        mid_splitter.addWidget(step_and_params)
        mid_splitter.setStretchFactor(0, 1)
        mid_splitter.setStretchFactor(1, 0)
        mid_splitter.setSizes([640, 100])
        mid_lay.addWidget(mid_splitter, 1)
        right = QtWidgets.QWidget()
        right.setMinimumWidth(440)
        right_lay = QtWidgets.QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        self.ortho_viewer = OrthoViewer(self.workspace, self)
        self.ortho_viewer.contentRenderRequested.connect(self._on_content_render_requested)
        self.ortho_viewer.planeTranslationStarted.connect(self._begin_plane_2d_translation)
        self.ortho_viewer.planeTranslationChanged.connect(self._move_plane_from_2d)
        self.ortho_viewer.planeTranslationFinished.connect(self._finish_plane_2d_translation)
        self.ortho_viewer.planeRotationStarted.connect(self._begin_plane_2d_rotation)
        self.ortho_viewer.planeRotationChanged.connect(self._rotate_plane_from_2d)
        self.ortho_viewer.planeRotationFinished.connect(self._finish_plane_2d_rotation)
        right_lay.addWidget(self.ortho_viewer, 1)
        splitter.addWidget(left)
        splitter.addWidget(mid)
        splitter.addWidget(right)
        splitter.setSizes([330, 600, 830])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 3)
        timeline_w = QtWidgets.QWidget()
        tl_lay = QtWidgets.QVBoxLayout(timeline_w)
        tl_lay.setContentsMargins(0, 0, 0, 0)
        self._build_timeline(tl_lay)
        mid_lay.addWidget(timeline_w, 0)

        self.selection_tabs = ContextStack(self._browser_parking)
        self.selection_tabs.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sel_w = QtWidgets.QWidget()
        sel_lay = QtWidgets.QVBoxLayout(sel_w)
        sel_lay.setContentsMargins(0, 0, 0, 0)
        sel_lay.setSpacing(8)
        self._build_selection_info(sel_lay)
        path_w = QtWidgets.QWidget()
        path_lay = QtWidgets.QVBoxLayout(path_w)
        path_lay.setContentsMargins(0, 0, 0, 0)
        self.text_path_info = SelectionInfoPanel()
        path_lay.addWidget(self.text_path_info)
        layer_w = QtWidgets.QWidget()
        layer_lay = QtWidgets.QVBoxLayout(layer_w)
        layer_lay.setContentsMargins(0, 0, 0, 0)
        self.text_layer_info = SelectionInfoPanel()
        layer_lay.addWidget(self.text_layer_info)
        self.selection_tabs.addWidget(sel_w)
        self.selection_tabs.addWidget(path_w)
        self.selection_tabs.addWidget(layer_w)
        log_w = QtWidgets.QWidget()
        log_lay = QtWidgets.QVBoxLayout(log_w)
        log_lay.setContentsMargins(6, 6, 6, 6)
        self._build_log(log_lay)
        self._build_segmentation_dock()
        self._build_pwv_dock()
        self._finish_compact_workspace(log_w)
        self.workflow_tabs.currentChanged.connect(self._on_workflow_stage_changed)
        self.workflow_tabs.tabBarClicked.connect(self._reopen_workflow_page)
        self._on_workflow_stage_changed(self.workflow_tabs.currentIndex())

    def _build_browser(self, parent):
        self.browser_panel = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(self.browser_panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self._browser_parking = QtWidgets.QWidget(self.browser_panel)
        self._browser_parking.hide()
        parking_layout = QtWidgets.QVBoxLayout(self._browser_parking)
        self._build_browser_tools(lay)
        self.tree_objects = BrowserTree()
        self.tree_objects.setMinimumHeight(120)
        self.tree_objects.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.tree_objects.itemSelectionChanged.connect(self._on_browser_select)
        self.tree_objects.itemChanged.connect(self._on_tree_item_changed)
        self.tree_objects.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.tree_objects.customContextMenuRequested.connect(self._on_browser_ctx_menu)
        lay.addWidget(self.tree_objects, 1)

        self.browser_options_menu = QtWidgets.QMenu(self)
        for name, title, callback in (
                ('chk_all_planes_visible', 'Show all planes', lambda checked: self._set_render_category_visibility('planes', checked)),
                ('chk_all_pathlines_visible', 'Show all pathlines', lambda checked: self._set_render_category_visibility('pathlines', checked))):
            box = QtWidgets.QCheckBox(title)
            box.setTristate(True)
            box.clicked.connect(callback)
            setattr(self, name, box)
            action = QtWidgets.QWidgetAction(self.browser_options_menu)
            action.setDefaultWidget(box)
            self.browser_options_menu.addAction(action)
        self.chk_selection_on_top = QtWidgets.QCheckBox('Selection on top')
        self.chk_selection_on_top.setChecked(True)
        self.chk_selection_on_top.toggled.connect(self._on_selection_on_top_changed)
        action = QtWidgets.QWidgetAction(self.browser_options_menu)
        action.setDefaultWidget(self.chk_selection_on_top)
        self.browser_options_menu.addAction(action)
        self.browser_options_menu.addSeparator()
        self.btn_delete_obj = QtGui.QAction('Delete selected', self)
        self.btn_delete_obj.setIcon(standard_icon(self, 'SP_TrashIcon'))
        self.btn_delete_obj.triggered.connect(self._on_delete_object)
        self.browser_options_menu.addAction(self.btn_delete_obj)
        self.browser_options_button.setMenu(self.browser_options_menu)

        opacity_row = QtWidgets.QHBoxLayout()
        self.label_browser_opacity = QtWidgets.QLabel('Opacity: —')
        self.slider_browser_opacity = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_browser_opacity.setRange(0, 100)
        self.slider_browser_opacity.setValue(100)
        self.slider_browser_opacity.setEnabled(False)
        self.slider_browser_opacity.setToolTip('Adjust opacity for selected objects')
        self.slider_browser_opacity.valueChanged.connect(self._on_browser_opacity_changed)
        opacity_row.addWidget(self.label_browser_opacity)
        opacity_row.addWidget(self.slider_browser_opacity, 1)
        self._build_layer_appearance(parking_layout)
        self.appearance_layout.addLayout(opacity_row)
        self.content_render_controls = ContentRenderControls(self)
        self.content_render_controls.settingsChanged.connect(self._on_content_render_settings)
        self.content_render_controls.positionChanged.connect(self._on_content_render_position)
        self.content_render_controls.interactionChanged.connect(lambda active: self.scene.set_interaction_preview(active))
        self.appearance_layout.addWidget(self.content_render_controls)
        self.content_render_controls.layout().labelForField(self.content_render_controls.cmap).hide()
        self.content_render_controls.cmap.hide()
        self.plane_display_controls = QtWidgets.QWidget()
        self._build_plane_display_controls(QtWidgets.QVBoxLayout(self.plane_display_controls))
        self.appearance_layout.addWidget(self.plane_display_controls)
        parent.addWidget(self.browser_panel, 1)

    def _build_step_buttons(self, parent):
        grp = QtWidgets.QGroupBox("Steps")
        self.steps_group = grp
        gl = QtWidgets.QGridLayout(grp)
        self.step_buttons = {}
        all_steps = [
            StepId.BACKGROUND_CORRECTION, StepId.REMOVE_NOISE, StepId.UNWRAP_PHASE, StepId.GENERATE_PCMRA,
            StepId.GENERATE_SKELETON, StepId.EDIT_SKELETON,
            StepId.GENERATE_GRAPH, StepId.EDIT_GRAPH,
            StepId.GENERATE_PLANES, StepId.COMPUTE_PLANE_METRICS,
            StepId.COMPUTE_PWV, StepId.COMPUTE_DERIVED_METRICS,
            StepId.GENERATE_STREAMLINES, StepId.PLANE_STREAMLINES,
        ]
        step_tooltips = {
            StepId.GENERATE_PCMRA: "Compute PC-MRA from current working velocity and create its 3D rendering layer.",
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
            if action_sections(step, self.workspace):
                self._attach_action_settings(button, step, partial(self._run_single_step, step))
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
        self._attach_action_settings(self.btn_run_all, "run_all", self._run_all_pipeline)
        run_all_row = import_row + 1
        gl.addWidget(self.btn_run_all, run_all_row, 0, 1, 3)
        parent.addWidget(grp, 0)

        self.btn_reload_input = QtWidgets.QPushButton("Reload Input")
        self.btn_reload_input.setIcon(standard_icon(self, "SP_BrowserReload"))
        self.btn_reload_input.clicked.connect(self._reload_input_case)
        self._attach_action_settings(self.btn_reload_input, "input", self._reload_input_case)
        parent.addWidget(self.btn_reload_input)
        self.btn_run_segmentation_auto = QtWidgets.QPushButton("Run Automatic Segmentation")
        self.btn_run_segmentation_auto.setProperty("role", "primary")
        self.btn_run_segmentation_auto.clicked.connect(self._on_run_auto_segmentation)
        self._attach_action_settings(self.btn_run_segmentation_auto, "segmentation", self._on_run_auto_segmentation)
        parent.addWidget(self.btn_run_segmentation_auto)

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
        self._edit_hint_label.setStyleSheet(f"color: {theme_colors(self)['secondary']}; padding: 3px 0;")
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

    def _reset_noise_removal(self):
        if self._pipeline_running_guard("resetting noise removal"):
            return
        self.workspace.pcmra_render_mask = None
        self.workspace.noise_removal_result = {}
        self.workspace.remove_object_by_data_key("noise_region")
        self.workspace.pipeline.completed.pop(StepId.REMOVE_NOISE.value, None)
        self.scene.invalidate_cache("pcmra_volume")
        self.scene.invalidate_cache("noise_region")
        self.scene.sync_from_workspace(rebuild_prefixes=("pcmra_volume", "noise_region"))
        self.ortho_viewer.refresh()
        self._refresh_browser()
        self._refresh_workflow_status()

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
            if obj.data_key == "pressure_gradient_volume":
                obj.opacity = float(ws.derived_params.pressure_gradient_layer_opacity)
            elif obj.data_key == "relative_pressure_volume":
                obj.opacity = float(ws.derived_params.relative_pressure_layer_opacity)
            self.scene.readd_object(obj)
            touched = True
        if touched:
            self._refresh_browser()
            self._refresh_scene()
            self.ortho_viewer.refresh()

    def _build_review_params(self):
        quality_group = QtWidgets.QGroupBox("Quality Control")
        quality_layout = QtWidgets.QVBoxLayout(quality_group)
        self.quality_summary_label = QtWidgets.QLabel("QC report has not been generated for this workspace.")
        self.quality_summary_label.setWordWrap(True)
        quality_layout.addWidget(self.quality_summary_label)

        self.quality_table = QtWidgets.QTreeWidget()
        self.quality_table.setHeaderLabels(["Status", "Stage", "Check", "Result"])
        self.quality_table.setRootIsDecorated(True)
        self.quality_table.setWordWrap(True)
        self.quality_table.setAlternatingRowColors(True)
        quality_header = self.quality_table.header()
        quality_header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        quality_header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        quality_header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        quality_header.setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)
        self.quality_table.setMinimumHeight(150)
        quality_layout.addWidget(self.quality_table)
        self._build_qc_details(quality_layout)

        self.flow_report_summary_label = QtWidgets.QLabel(
            "Vessel flow hierarchy becomes available after plane metrics are calculated."
        )
        self.flow_report_summary_label.setWordWrap(True)
        quality_layout.addWidget(self.flow_report_summary_label)

        self.flow_report_tree = QtWidgets.QTreeWidget()
        self.flow_report_tree.setHeaderLabels(["Vessel / Path", "Level", "Flow & Velocity (mean ± SD)", "Consistency"])
        self.flow_report_tree.setRootIsDecorated(True)
        self.flow_report_tree.setAlternatingRowColors(True)
        flow_header = self.flow_report_tree.header()
        flow_header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        flow_header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        flow_header.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        flow_header.setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)
        self.flow_report_tree.setMinimumHeight(230)
        quality_layout.addWidget(self.flow_report_tree)

        quality_buttons = QtWidgets.QHBoxLayout()
        self.btn_refresh_quality = QtWidgets.QPushButton("Refresh QC")
        self.btn_refresh_quality.clicked.connect(self._refresh_quality_panel)
        self.btn_export_quality = QtWidgets.QPushButton("Export QC Report")
        self.btn_export_quality.clicked.connect(self._on_export_quality_report)
        quality_buttons.addWidget(self.btn_refresh_quality)
        self.btn_export_analysis_report = QtWidgets.QPushButton("Export Analysis Report…")
        self.btn_export_analysis_report.clicked.connect(self._on_export_analysis_report)
        quality_buttons.addStretch()
        quality_layout.addLayout(quality_buttons)
        html_buttons = QtWidgets.QHBoxLayout()
        for title, callback, name in [("Open QC Report", self._on_open_case_qc, "openCaseQC"),
                                      ("Open Batch QC…", self._on_open_batch_qc, "openBatchQC")]:
            button = QtWidgets.QPushButton(title)
            button.setObjectName(name)
            button.clicked.connect(callback)
            html_buttons.addWidget(button)
        quality_layout.addLayout(html_buttons)
        self.params_layout.addWidget(quality_group)

        self.btn_export_planes = QtWidgets.QPushButton("Export selected planes…")
        self.btn_export_planes.clicked.connect(self._on_export_plane_coordinates)
        self._build_export_page()

    def _workflow_stage_key(self, index=None):
        if index is None:
            index = self.workflow_tabs.currentIndex()
        if not (0 <= int(index) < len(_WORKFLOW_STAGES)):
            return "input"
        return _WORKFLOW_STAGES[int(index)][1]

    def _workflow_stage_state(self, stage_key):
        ws = self.workspace
        pending = set(ws.input_state.metadata.get("gui_pending_settings", []))
        if ws.data_loaded and any(step.value in pending and not ws.pipeline.is_done(step)
                                  for step in _WORKFLOW_STEPS.get(stage_key, ())):
            return "Needs review"
        if stage_key == "input":
            return "Ready" if ws.data_loaded else "Not ready"
        if stage_key == "segmentation":
            if not ws.data_loaded:
                return "Not ready"
            return "Ready" if ws.segmask_raw is not None else "Needs review"
        if stage_key == "correction":
            if not ws.data_loaded:
                return "Not ready"
            steps = _WORKFLOW_RUN_ALL_STEPS["correction"]
            return "Ready" if all(ws.pipeline.is_done(step) for step in steps) else "Optional"
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
        if stage_key == "export":
            return "Ready" if ws.data_loaded else "Not ready"
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
        foreground, background = workflow_status_colors(state, self)
        self.workflow_status.setText(state)
        self.workflow_status.setStyleSheet(
            f"QLabel {{ color: {foreground}; background: {background}; border: 1px solid {foreground}; padding: 3px 8px; border-radius: 4px; }}"
        )

    def _on_workflow_stage_changed(self, index, *, open_report=True):
        QtCore.QTimer.singleShot(0, self._update_qc_overlay)
        stage_key = self._workflow_stage_key(index)
        self.review_panel.setVisible(stage_key == "review")
        self.btn_reload_input.setVisible(stage_key == "input")
        self.btn_run_segmentation_auto.setVisible(stage_key == "segmentation")
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
            StepId.BACKGROUND_CORRECTION, StepId.REMOVE_NOISE, StepId.UNWRAP_PHASE, StepId.GENERATE_PCMRA,
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
            if stage_key in {"correction", "centerline", "hemodynamics"}:
                step_layout.addWidget(self.btn_run_all, action_row, 0, 1, 2)
        if stage_key == "centerline":
            plane_row = active_steps.index(StepId.GENERATE_PLANES) // 2
            step_layout.addWidget(self.btn_import_planes_centerline, plane_row, 1)
        for step, button in self.step_buttons.items():
            button.setVisible(step in visible_steps)
        if hasattr(self, "btn_import_planes_centerline"):
            self.btn_import_planes_centerline.setVisible(stage_key == "centerline")
        self.btn_run_all.setVisible(stage_key in {"correction", "centerline", "hemodynamics"})
        if stage_key in {"correction", "centerline", "hemodynamics"}:
            self.btn_run_all.setText("Run All")
        self.steps_group.setVisible(bool(visible_steps) or self.btn_run_all.isVisible())
        stage_label = _WORKFLOW_STAGES[int(index)][0].replace("&", "&&")
        self.steps_group.setTitle(f"{stage_label} Actions")
        if hasattr(self, "mid_splitter"):
            self.mid_splitter.setSizes([330, 360] if stage_key == "review" else [640, 100])

        if hasattr(self, "segmentation_dock"):
            self.segmentation_dock.setVisible(stage_key == "segmentation")
            if stage_key == "segmentation":
                self.segmentation_dock.raise_()
        if hasattr(self, "pwv_dock"):
            if stage_key not in {"hemodynamics", "review"}:
                self.pwv_dock.hide()
        if self._edit_mode is not None and self._edit_panel is not None:
            self.steps_group.setVisible(False)
            self._edit_panel.setVisible(True)
        if stage_key == "review":
            self.workspace_pages.setCurrentWidget(self.qc_page)
            if not self.workspace.data_loaded:
                self.qc_page.setCurrentWidget(self.qc_empty)
            elif open_report:
                QtCore.QTimer.singleShot(0, lambda: self._on_open_case_qc()
                    if self._workflow_stage_key() == 'review' else None)
        elif stage_key == "export":
            self._refresh_export_page()
            self.workspace_pages.setCurrentWidget(self._export_page)
        else:
            self.workspace_pages.setCurrentWidget(self.workspace_page)
        self._sync_workspace_docks(self.workspace_pages.currentIndex())
        self._refresh_workflow_status()

    def _build_timeline(self, parent):
        grp = QtWidgets.QGroupBox("Timeline")
        tl = QtWidgets.QGridLayout(grp)
        tl.setVerticalSpacing(2)
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
        self.spin_fps = QtWidgets.QDoubleSpinBox()
        self.spin_fps.setRange(0.5, 100.0)
        self.spin_fps.setDecimals(1)
        self.spin_fps.setSingleStep(1.0)
        self.spin_fps.setValue(10.0)
        self.spin_fps.setSuffix(" fps")
        self.spin_fps.setAlignment(QtCore.Qt.AlignCenter)
        self.spin_fps.setMinimumWidth(114)
        self.spin_fps.setAccessibleName("Playback FPS")
        self.spin_fps.setToolTip("Target playback FPS. Actual FPS depends on the visible scene; one acquired phase is shown per frame.")
        self.lab_playback_performance = QtWidgets.QLabel("Render: --")
        self.lab_playback_performance.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.lab_playback_performance.setMinimumWidth(180)
        self.lab_playback_performance.setToolTip("Rolling time spent updating the timeline scene and slice views, plus effective playback FPS.")
        for column, w in enumerate([self.btn_prev, self.btn_play, self.btn_pause, self.btn_next]):
            w.setProperty("role", "icon")
            w.setAccessibleName(w.toolTip())
            tl.addWidget(w, 0, column)
        tl.addWidget(self.slider_t, 0, 4)
        tl.addWidget(self.lab_t, 0, 5)
        tl.addWidget(self.lab_playback_performance, 1, 0, 1, 5)
        tl.addWidget(self.spin_fps, 1, 5)
        tl.setColumnStretch(4, 1)
        parent.addWidget(grp)

    def _build_selection_info(self, parent):
        self.text_plane_info = SelectionInfoPanel()
        parent.addWidget(self.text_plane_info)
        self.check_plane_unlocked = QtWidgets.QPushButton('Edit plane')
        self.check_plane_unlocked.setCheckable(True)
        self.check_plane_unlocked.setToolTip('Edit the selected plane; Apply saves, Cancel / Esc discards the draft.')
        self.check_plane_unlocked.toggled.connect(self._set_plane_edit_unlocked)
        parent.addWidget(self.check_plane_unlocked)
        plane_actions = QtWidgets.QHBoxLayout()
        self.btn_add_plane = QtWidgets.QPushButton('Add Plane', self._browser_parking)
        self.btn_add_plane.clicked.connect(self._add_manual_plane)
        self.btn_add_plane.hide()
        self.btn_edit_plane = QtWidgets.QPushButton('Apply plane')
        self.btn_edit_plane.clicked.connect(self._toggle_selected_plane_edit)
        plane_actions.addWidget(self.btn_edit_plane)
        parent.addLayout(plane_actions)
        self.plane_controls_scroll = QtWidgets.QScrollArea()
        self.plane_controls_scroll.setWidgetResizable(True)
        self.plane_controls_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.plane_controls_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        details = QtWidgets.QWidget()
        details_layout = QtWidgets.QVBoxLayout(details)
        details_layout.setContentsMargins(0, 0, 4, 0)
        details_layout.setSizeConstraint(QtWidgets.QLayout.SetMinAndMaxSize)
        self._build_plane_transaction_controls(details_layout, source_layout=parent, actions_layout=plane_actions)
        self.plane_controls_scroll.setWidget(details)
        parent.addWidget(self.plane_controls_scroll)

    def _build_plane_display_controls(self, parent):
        plane_display_row = QtWidgets.QHBoxLayout()
        plane_display_row.addWidget(QtWidgets.QLabel("Display size"))
        self.slider_plane_size = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_plane_size.setRange(10, 2000)
        self.slider_plane_size.setToolTip("Resize every plane outline immediately; metric regions stay unchanged")
        self.spin_plane_size = QtWidgets.QDoubleSpinBox()
        self.spin_plane_size.setRange(1.0, 200.0)
        self.spin_plane_size.setDecimals(1)
        self.spin_plane_size.setSingleStep(1.0)
        self.spin_plane_size.setSuffix(" mm")
        plane_display_row.addWidget(self.slider_plane_size, 1)
        plane_display_row.addWidget(self.spin_plane_size)
        parent.addLayout(plane_display_row)
        self.chk_plane_selected_only = QtWidgets.QCheckBox("Only selected plane")
        self.chk_plane_selected_only.setToolTip("Select a plane in the Browser to show its outline; other outlines stay hidden")
        parent.addWidget(self.chk_plane_selected_only)
        self.slider_plane_size.valueChanged.connect(lambda value: self._on_plane_display_size_changed(value / 10.0))
        self.spin_plane_size.valueChanged.connect(self._on_plane_display_size_changed)
        self.chk_plane_selected_only.toggled.connect(self._on_plane_selected_only_changed)
        self._sync_plane_display_controls()

    def _sync_plane_display_controls(self):
        if not hasattr(self, "spin_plane_size"):
            return
        cfg = self.workspace.render_settings.get("plane_render_cfg", {}) or {}
        size = plane_display_size(self.workspace)
        values = ((self.spin_plane_size, size), (self.slider_plane_size, round(size * 10)),
                  (self.chk_plane_selected_only, bool(cfg.get("selected_only", False))))
        for widget, value in values:
            previous = widget.blockSignals(True)
            try:
                if widget is self.chk_plane_selected_only:
                    widget.setChecked(value)
                else:
                    widget.setValue(value)
            finally:
                widget.blockSignals(previous)

    def _on_plane_display_size_changed(self, size_mm):
        if self.scene is None:
            self.workspace.render_settings.setdefault("plane_render_cfg", {})["plane_size_mm"] = float(size_mm)
        else:
            self.scene.set_plane_display_settings(size_mm=float(size_mm), render=False)
        self._sync_plane_display_controls()
        if self._plane_edit_enabled and 0 <= int(self._selected_plane_index) < len(self.workspace.planes):
            self._position_plane_handles()
        if self.scene is not None:
            self.plotter.render()

    def _on_plane_selected_only_changed(self, enabled):
        if self.scene is None:
            self.workspace.render_settings.setdefault("plane_render_cfg", {})["selected_only"] = bool(enabled)
        else:
            self.scene.set_plane_display_settings(selected_only=enabled)

    def _build_log(self, parent):
        self.console = QtWidgets.QTextEdit()
        self.console.setReadOnly(True)
        parent.addWidget(self.console)

    def _build_segmentation_dock(self):
        self.segmentation_dock = QtWidgets.QDockWidget("Segmentation", self)
        self.segmentation_dock.setObjectName("SegmentationDock")
        self.segmentation_panel = SegmentationDock(self)
        self.segmentation_dock.setWidget(self.segmentation_panel)
        self.segmentation_dock.visibilityChanged.connect(
            lambda visible: self._refresh_segmentation_ui(force=True) if visible else None)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.segmentation_dock)
        self.segmentation_panel.combo_source.currentIndexChanged.connect(self._on_segmentation_source_changed)
        self.segmentation_panel.check_visible.toggled.connect(self._on_segmentation_visibility_changed)
        self.segmentation_panel.slider_opacity.valueChanged.connect(self._on_segmentation_opacity_changed)
        self.segmentation_panel.btn_run_auto.clicked.connect(self._on_run_auto_segmentation)
        self._attach_action_settings(self.segmentation_panel.btn_run_auto, "segmentation", self._on_run_auto_segmentation)
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
        self.pwv_dock = QtWidgets.QDockWidget("Measurements", self)
        self.pwv_dock.setObjectName("PWVDock")
        panel = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 10)
        layout.setSpacing(10)
        self.measurements_panel = panel

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
        curve_layout = QtWidgets.QVBoxLayout(curve_controls)
        curve_layout.setContentsMargins(0, 0, 0, 0)
        curve_row = QtWidgets.QHBoxLayout()
        curve_row.addWidget(QtWidgets.QLabel("Metric"))
        self.combo_plane_curve_metric = QtWidgets.QComboBox()
        self.combo_plane_curve_metric.currentIndexChanged.connect(self._on_plane_curve_metric_changed)
        curve_row.addWidget(self.combo_plane_curve_metric, 1)
        curve_layout.addLayout(curve_row)
        self.analysis_controls_stack.addWidget(curve_controls)

        pressure_controls = QtWidgets.QWidget(self)
        pressure_row = QtWidgets.QHBoxLayout(pressure_controls)
        pressure_row.setContentsMargins(0, 0, 0, 0)
        pressure_row.addWidget(QtWidgets.QLabel("Path"))
        self.combo_centerline_pressure_path = QtWidgets.QComboBox()
        self.combo_centerline_pressure_path.currentIndexChanged.connect(self._on_centerline_pressure_path_changed)
        pressure_row.addWidget(self.combo_centerline_pressure_path, 1)
        self.analysis_controls_stack.addWidget(pressure_controls)

        layout.addWidget(self.analysis_controls_stack)

        self.label_pwv_status = MeasurementSummary()
        self.label_pwv_status.parametersRequested.connect(lambda: self._open_action_settings("measurements"))
        layout.addWidget(self.label_pwv_status)

        self.fig_pwv = Figure(figsize=(4.2, 5.4), dpi=90, facecolor=analysis_palette(self)["panel"])
        self.canvas_pwv = FigureCanvas(self.fig_pwv)
        for target in (panel, self.canvas_pwv, self.btn_measurements):
            target.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
            target.customContextMenuRequested.connect(lambda _pos: self._open_action_settings("measurements"))
        panel._settings_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Shift+F10"), panel)
        panel._settings_shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
        panel._settings_shortcut.activated.connect(lambda: self._open_action_settings("measurements"))
        self.btn_measurements.setToolTip("Open measurements; right-click for curve parameters")
        self.canvas_pwv.mpl_connect(
            "resize_event", lambda event: style_analysis_figure(self)
            if event.width >= 160 and event.height >= 160 else None)
        layout.addWidget(self.canvas_pwv, 1)

        self.pwv_dock.visibilityChanged.connect(lambda visible: self._refresh_analysis_panel() if visible else None)
        self.pwv_dock.setWidget(panel)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.pwv_dock)
        self.tabifyDockWidget(self.segmentation_dock, self.pwv_dock)
        self._sync_analysis_controls()
        self.segmentation_dock.raise_()

    def _analysis_mode(self):
        value = self.combo_analysis_mode.currentData() if hasattr(self, "combo_analysis_mode") else "pwv"
        return str(value or "pwv")

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

    def _flowrate_spline_order(self):
        cfg = (getattr(self.workspace, "render_settings", {}) or {}).get("measurements", {}) or {}
        if not isinstance(cfg, dict):
            return 3
        try:
            return min(3, max(1, int(cfg.get("flowrate_spline_order", 3))))
        except (TypeError, ValueError):
            return 3

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
            peak = item.get("pressure_drop_peak_Pa")
            peak_text = f'{float(peak):.3g} Pa' if peak is not None and np.isfinite(peak) else 'drop unavailable'
            self.combo_centerline_pressure_path.addItem(f"Path {path_index} ({peak_text})", path_index)
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
            ax.plot(distances, values, color=analysis_palette(self)["curve"], linewidth=1.65)
            ax.scatter([distances[0], distances[-1]], [values[0], values[-1]], color=analysis_palette(self)["curve"], s=20, zorder=4)
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
            ax_drop.plot(phases, drop_t, color=analysis_palette(self)["curve"], marker="o", linewidth=1.65, markersize=2.5, markerfacecolor=analysis_palette(self)["panel"])
            ax_drop.axvline(current_t, color=analysis_palette(self)["cursor"], linestyle=(0, (2, 4)), linewidth=.8)
            ax_drop.scatter([current_t], [drop_t[min(current_t, drop_t.size - 1)]], color=analysis_palette(self)["curve"], s=24, zorder=4)
            ax_drop.set_xlabel("Cardiac Phase")
            ax_drop.set_ylabel("Pressure Drop (Pa)")
            ax_drop.set_title("Centerline Pressure Drop")
            ax_drop.grid(True, alpha=0.25)
        else:
            ax_drop.text(0.5, 0.5, "No pressure-drop series", ha="center", va="center", transform=ax_drop.transAxes)
            ax_drop.set_xticks([])
            ax_drop.set_yticks([])
        style_analysis_figure(self)
        self.canvas_pwv.draw_idle()
        def pressure_text(value):
            return f'{float(value):.4g} Pa' if value is not None and np.isfinite(value) else 'Unavailable'
        current_drop = drop_t[min(current_t, drop_t.size - 1)] if drop_t.size else None
        lines = [
            f"Path {int(profile.get('path_index', -1))}   Current phase: {current_t}",
            f"Current drop: {pressure_text(current_drop)}   Mean drop: {pressure_text(profile.get('pressure_drop_mean_Pa'))}   Peak |drop|: {pressure_text(profile.get('pressure_drop_peak_Pa'))}",
        ]
        self.label_pwv_status.set_measurement(
            f"Path {int(profile.get('path_index', -1))}", f"Phase {current_t} · Pressure drop",
            pressure_text(current_drop).removesuffix(" Pa"), "Pa",
            [("Mean", pressure_text(profile.get("pressure_drop_mean_Pa"))),
             ("Peak |drop|", pressure_text(profile.get("pressure_drop_peak_Pa")))],
            plain_text="\n".join(lines))

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
        metric = selected_metric(self.workspace, plane_idx)
        if not metric:
            return plane_idx, None
        return plane_idx, dict(metric)

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
        token = (str(message), str(title))
        if getattr(self, "_empty_analysis_token", None) == token and len(self.fig_pwv.axes) == 1 and not self.fig_pwv.axes[0].lines:
            return
        self._empty_analysis_token = token
        self.fig_pwv.clear()
        ax = self.fig_pwv.add_subplot(111)
        ax.set_title(str(title))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.5, str(message), ha="center", va="center", transform=ax.transAxes)
        style_analysis_figure(self)
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
        self.label_pwv_status.set_measurement(
            "", f"{status} · {valid_count}/{plane_count} valid planes",
            f"{float(pwv):.4g}" if pwv is not None and np.isfinite(pwv) else "—", "m/s",
            [("Fit R²", f"{float(fit_r2):.3f}" if fit_r2 is not None and np.isfinite(fit_r2) else "—"),
             ("Path length", f"{float(longest_path):.1f} mm" if longest_path is not None else "—")],
            notes="\n".join(value for value in (message, plot_error) if value), plain_text="\n".join(lines))
        style_analysis_figure(self)
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
        if values.size == 0 or not np.any(np.isfinite(values)):
            self.label_pwv_status.setText(f"Plane {int(plane_idx)} has no valid samples for {series_label}.")
            self._clear_pwv_axes("Selected metric is not available for this plane.", title="Plane Curve")
            return
        values = update_plane_curve(self, int(plane_idx), metric, series_key, series_label, _flowrate_display_curve)
        current_t = int(np.clip(self.workspace.current_t, 0, max(0, values.size-1)))
        if not np.any(np.isfinite(values)):
            self.label_pwv_status.setText("Not assessable: no valid ROI samples.")
            self._clear_pwv_axes("No valid ROI samples for this measurement.", title="Plane Curve")
            return

        plane = self.workspace.planes[int(plane_idx)]
        path_idx = int(metric.get("path_index", getattr(plane, "path_index", -1)))
        path_dir = str(metric.get("path_direction", "") or "")
        finite = values[np.isfinite(values)]
        def display_number(value):
            return "—" if value is None or not np.isfinite(float(value)) else f"{float(value):.4g}"
        complete = not metric.get("sample_valid_t") or all(metric["sample_valid_t"])
        parts = series_label.rsplit(" (", 1)
        unit = parts[1].rstrip(")") if len(parts) == 2 else ""
        context = f"Path {path_idx}" + (f" · {path_dir}" if path_dir else "")
        context += f" · Phase {current_t} · {values.size} phases"
        self.label_pwv_status.set_measurement(
            f"Plane {int(plane_idx)}", context, display_number(values[current_t]), unit,
            [("Min", display_number(np.min(finite))),
             ("Mean", display_number(np.mean(finite))),
             ("Max", display_number(np.max(finite)))],
            notes="" if complete else "Incomplete cycle · unavailable phases remain gaps.",
            plain_text="\n".join([
                f"Plane {int(plane_idx)} | Path {path_idx}", context, f"Metric: {series_label}",
                f"Current: {display_number(values[current_t])} {unit}",
                f"Min: {display_number(np.min(finite))} | Mean: {display_number(np.mean(finite))} | Max: {display_number(np.max(finite))}",
            ]),
        )

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
        if hasattr(self, "pwv_dock") and not self.pwv_dock.isVisible():
            self._analysis_dirty = True
            return
        self._analysis_dirty = False
        self._sync_analysis_controls()
        self._refresh_pwv_group_selector()
        self._refresh_centerline_pressure_path_selector()
        mode = self._analysis_mode()
        if mode == "pwv":
            self._plot_pwv_result(self._selected_pwv_result())
        elif mode == "plane_curve":
            self._plot_plane_curve()
        elif mode == "centerline_pressure":
            self._plot_centerline_pressure()
        else:
            self._plot_plane_curve()
        plane_index, metric = self._selected_plane_metric()
        keys = pinned_metric_keys(self.workspace)
        rows = pinned_metric_rows(metric, keys) if mode == "plane_curve" or plane_index >= 0 else []
        heading = f"Plane {plane_index} · Cycle summary" if mode != "plane_curve" and plane_index >= 0 else "Cycle summary"
        self.label_pwv_status.set_pinned_metrics(heading, rows)

    def _refresh_pwv_plot(self):
        self._refresh_plane_curve_metric_options()
        self._refresh_analysis_panel()

    def _build_menu(self):
        mb = self.menuBar()
        mf = mb.addMenu("File")
        for label, slot in [
            ("Load H5", self._on_load_h5),
            ("Load DICOM", self._on_load_dicom),
            ("Clear Workspace", self._on_close_workspace),
            ("Exit", self.close),
        ]:
            a = QtGui.QAction(label, self)
            a.triggered.connect(slot)
            mf.addAction(a)
        me = mb.addMenu("Export")
        a = QtGui.QAction("Export page", self)
        a.triggered.connect(self._on_open_export_page)
        me.addAction(a)
        me.addSeparator()
        a = QtGui.QAction("Export Videos...", self)
        a.triggered.connect(self._on_export_videos)
        me.addAction(a)
        a = QtGui.QAction("Export Current 3D View Video...", self)
        a.setObjectName("actionExportCurrentViewVideo")
        a.triggered.connect(self._on_export_current_view_video)
        me.addAction(a)
        a = QtGui.QAction("Export Rotating 3D View Video...", self)
        a.setObjectName("actionExportRotatingViewVideo")
        a.triggered.connect(lambda: self._on_export_current_view_video(rotate=True))
        me.addAction(a)
        a = QtGui.QAction("Export Plane Coordinates...", self)
        a.triggered.connect(self._on_export_plane_coordinates)
        me.addAction(a)
        a = QtGui.QAction("Export QC Report...", self)
        a.triggered.connect(self._on_export_quality_report)
        me.addAction(a)
        a = QtGui.QAction("Export Analysis Report...", self)
        a.triggered.connect(self._on_export_analysis_report)
        me.addAction(a)
        ms = mb.addMenu("Settings")
        a = QtGui.QAction("3D Axis Orientation...", self)
        a.triggered.connect(self._open_display_orientation_settings)
        ms.addAction(a)
        a = QtGui.QAction("3D Background Color...", self)
        a.triggered.connect(self._open_background_color_settings)
        ms.addAction(a)
        a = QtGui.QAction("Viewer Display Settings...", self)
        a.triggered.connect(lambda: self.ortho_viewer.open_plane_display_settings())
        ms.addAction(a)
        a = QtGui.QAction("3D Colorbar Layout...", self)
        a.triggered.connect(lambda: self._open_action_settings("render"))
        ms.addAction(a)
        mv = mb.addMenu("View")
        self.view_menu = mv
        for label, slot in [("Reset Camera", lambda: self.scene.reset_camera()), ("Toggle Axes", lambda: self.scene.toggle_axes()),
            ("White BG", lambda: self.scene.set_background("white")), ("Dark BG", lambda: self.scene.set_background(BACKGROUND_COLOR))]:
            a = QtGui.QAction(label, self)
            a.triggered.connect(slot)
            mv.addAction(a)

    def _on_content_render_requested(self, content):
        try:
            had_scene = bool(self.scene._tracked_actors)
            obj = self.scene.add_content_render(content, plane_index=self.ortho_viewer._selected_plane_idx)
            if obj is None:
                self.statusBar().showMessage("Selected Content has no 3D volume available; see Log.", 8000)
                return
            self._refresh_browser()
            item = self._find_browser_item_by_uid(obj.uid)
            if item is not None:
                self.tree_objects.setCurrentItem(item, 0, QtCore.QItemSelectionModel.ClearAndSelect)
                self.tree_objects.scrollToItem(item)
            self._refresh_content_render_controls()
            if not had_scene:
                self.scene.reset_camera()
            self._refresh_render_range_control()
            message = (f"3D volume shown: {obj.name}. Controls are below Browser."
                       if obj.actor is not None else f"3D rendering failed for {obj.name}; see Log.")
            self.log(message)
            self.statusBar().showMessage(message, 8000)
        except Exception as exc:
            self.log(f"3D RENDER ERROR: {type(exc).__name__}: {exc}")
            self.log(traceback.format_exc())
            self.statusBar().showMessage(f"3D rendering failed: {exc}; see Log.", 8000)

    def _refresh_content_render_controls(self):
        objects = self._browser_selected_objects()
        obj = objects[0] if len(objects) == 1 and self.scene.is_content_render(objects[0]) else None
        self.content_render_controls.set_object(obj, self.scene.content_render_settings(obj.data_key) if obj else None)
        self.scene._content_position_callback = self._on_content_drag_position
        self.scene.select_content_render(obj.uid if obj else None)
        self._size_layer_appearance()

    def _on_content_render_settings(self, settings):
        uid = self._selected_uid()
        if uid is not None:
            self.scene.update_content_render(uid, **settings)
            self._refresh_render_range_control()

    def _on_content_render_position(self, axis, fraction):
        uid = self._selected_uid()
        if uid is not None:
            self.scene.set_content_axis_position(uid, axis, fraction)

    def _on_content_drag_position(self, uid, positions):
        if self._selected_uid() == uid:
            self.content_render_controls.set_positions(positions)

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
            current = QtGui.QColor(BACKGROUND_COLOR)
        color = QtWidgets.QColorDialog.getColor(current, self, "3D Background Color")
        if color.isValid():
            value = color.name()
            self.scene.set_background(value)
            self._config_bundle.setdefault("ui", {})["background_color"] = value

    @staticmethod
    def _quality_mean_sd(summary, digits=2):
        payload = dict(summary or {})
        mean = payload.get("mean")
        std = payload.get("std")
        if mean is None or std is None:
            return "n/a"
        return f"{float(mean):.{int(digits)}f} ± {float(std):.{int(digits)}f}"

    def _quality_flow_statistics_text(self, statistics):
        stats = dict(statistics or {})
        return (
            f"Qnet {self._quality_mean_sd(stats.get('net_flow_mL_beat'))} mL/beat  ·  "
            f"Qpeak {self._quality_mean_sd(stats.get('peak_flow_mL_s'))} mL/s  ·  "
            f"Vmean {self._quality_mean_sd(stats.get('mean_velocity_cm_s'))} cm/s  ·  "
            f"Vpeak {self._quality_mean_sd(stats.get('peak_velocity_cm_s'))} cm/s  ·  "
            f"n={int(stats.get('plane_count', 0) or 0)}"
        )

    @staticmethod
    def _quality_consistency_text(node):
        value = node.get("internal_consistency")
        internal = "undefined" if value is None else f"{float(value):.3f}"
        relations = []
        for junction in list(node.get("junctions", []) or []):
            junction_value = junction.get("internal_consistency")
            junction_text = "undefined" if junction_value is None else f"{float(junction_value):.3f}"
            relations.append(f"{junction.get('equation', '? = ?')}: {junction_text}")
        text = f"Path IC {internal}"
        if relations:
            text += "  ·  " + "  ·  ".join(relations)
        return text

    def _add_quality_flow_path_item(self, parent, node):
        item = QtWidgets.QTreeWidgetItem([
            str(node.get("name", "Path")),
            str(node.get("branch_level")) if node.get("branch_level") is not None else "?",
            self._quality_flow_statistics_text(node.get("statistics", {})),
            self._quality_consistency_text(node),
        ])
        path_index = int(node.get("path_index", -1))
        item.setToolTip(
            0,
            f"Label: {node.get('label_name', 'Unlabeled')}\nPath index: {path_index}\n"
            "Child branches are ordered by descending mean absolute net flow.",
        )
        item.setToolTip(2, self._quality_flow_statistics_text(node.get("statistics", {})))
        item.setToolTip(3, self._quality_consistency_text(node))
        item.setData(0, _BROWSER_PATH_INDEX_ROLE, path_index)
        if node.get("parent_path_index") is None:
            for column in range(4):
                font = item.font(column)
                font.setBold(True)
                item.setFont(column, font)
        consistency_values = [node.get("internal_consistency")]
        consistency_values.extend(
            junction.get("internal_consistency")
            for junction in list(node.get("junctions", []) or [])
        )
        consistency_values = [float(value) for value in consistency_values if value is not None]
        if consistency_values:
            ic_value = min(consistency_values)
            color = "#1b5e20" if ic_value >= 0.8 else ("#8a4b00" if ic_value >= 0.6 else "#b71c1c")
            item.setForeground(3, QtGui.QBrush(readable_color(color, self)))
        if parent is None:
            self.flow_report_tree.addTopLevelItem(item)
        else:
            parent.addChild(item)

        for child in list(node.get("children", []) or []):
            self._add_quality_flow_path_item(item, child)

        planes = list(node.get("planes", []) or [])
        if planes:
            plane_group = QtWidgets.QTreeWidgetItem([f"Planes ({len(planes)})", "Details", "", ""])
            item.addChild(plane_group)
            for plane in planes:
                def number(key, digits=2):
                    value = plane.get(key)
                    return "n/a" if value is None else f"{float(value):.{int(digits)}f}"

                sequence = int(plane.get("sequence", 0) or 0)
                plane_index = int(plane.get("plane_index", -1))
                distance = number("distance_mm", 1)
                detail = (
                    f"Qnet {number('net_flow_mL_beat')} mL/beat  ·  "
                    f"Qpeak {number('peak_flow_mL_s')} mL/s  ·  "
                    f"Vmean {number('mean_velocity_cm_s')} cm/s  ·  "
                    f"Vpeak {number('peak_velocity_cm_s')} cm/s"
                )
                plane_item = QtWidgets.QTreeWidgetItem([
                    f"Plane {sequence}",
                    f"{distance} mm",
                    detail,
                    "",
                ])
                plane_item.setToolTip(0, f"Global plane index: {plane_index}\nPath index: {path_index}")
                plane_group.addChild(plane_item)
        item.setExpanded(False)
        return item

    def _populate_quality_flow_tree(self, hierarchy):
        if not hasattr(self, "flow_report_tree"):
            return
        self.flow_report_tree.clear()
        payload = dict(hierarchy or {})
        roots = list(payload.get("roots", []) or [])
        status = str(payload.get("status", "not_run"))
        if not roots:
            self.flow_report_tree.setEnabled(False)
            self.flow_report_summary_label.setText(
                "Vessel flow hierarchy is not available. Generate paths, planes, and plane metrics, then refresh QC."
            )
            return
        self.flow_report_summary_label.setText(
            f"Vessel flow report: {payload.get('summary', '')}. "
            "Expand root rows for downstream branches, conservation equations, and per-plane details. Graph grades and loop connections are available after Generate Graph."
        )
        groups, subgroups = {}, {}
        metadata = {item["subgroup_id"]: item for item in payload.get("subgroups", [])}
        for root in roots:
            parent = None
            subgroup_id = root.get("subgroup_id")
            if subgroup_id in metadata:
                subgroup = metadata[subgroup_id]
                group_name = subgroup["group_name"]
                if group_name not in groups:
                    groups[group_name] = QtWidgets.QTreeWidgetItem([group_name, "Group", "", ""])
                    self.flow_report_tree.addTopLevelItem(groups[group_name])
                if subgroup_id not in subgroups:
                    subgroups[subgroup_id] = QtWidgets.QTreeWidgetItem([
                        f"Subgroup {subgroup['subgroup_index'] + 1}", "Component",
                        f"{len(subgroup['path_indices'])} paths", ""])
                    subgroups[subgroup_id].setToolTip(0, subgroup_id)
                    groups[group_name].addChild(subgroups[subgroup_id])
                parent = subgroups[subgroup_id]
            self._add_quality_flow_path_item(parent, root)
        self.flow_report_tree.collapseAll()
        for item in [*groups.values(), *subgroups.values()]:
            item.setExpanded(True)
        self.flow_report_tree.setEnabled(status in {"available", "topology_only"})

    def _refresh_quality_panel(self):
        if not hasattr(self, "quality_table"):
            return
        self.quality_table.clear()
        if hasattr(self, "flow_report_tree"):
            self.flow_report_tree.clear()
        if not self.workspace.data_loaded:
            self._last_quality_report = None
            self.quality_summary_label.setText("Load a case to generate QC results.")
            if hasattr(self, "flow_report_summary_label"):
                self.flow_report_summary_label.setText("Load a case to generate the vessel flow report.")
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
            f"error={int(counts.get('fail', 0))}, not run={int(counts.get('not_run', 0))}"
        )
        status_colors = {
            "pass": "#1b5e20",
            "warn": "#8a4b00",
            "fail": "#b71c1c",
            "not_run": "#616161",
        }
        def add_check(check, parent=None):
            status = str(check.get("status", "not_run"))
            item = QtWidgets.QTreeWidgetItem([
                ("ERROR" if status == "fail" else status.replace("_", " ").upper()),
                str(check.get("stage", "")),
                str(check.get("title", check.get("id", ""))),
                str(check.get("summary", "")),
            ])
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, check)
            color = QtGui.QColor(status_colors.get(status, "#616161"))
            item.setForeground(0, QtGui.QBrush(readable_color(color, self)))
            action = str(check.get("action", "") or "")
            threshold = str(check.get("threshold", "") or "")
            tooltip = "\n".join(part for part in [str(check.get("summary", "")), threshold, action] if part)
            if tooltip:
                item.setToolTip(2, tooltip)
                item.setToolTip(3, tooltip)
            if parent is None:
                self.quality_table.addTopLevelItem(item)
            else:
                parent.addChild(item)
            for detail in check.get("details", []):
                add_check(detail, item)
        for check in list(report.get("checks", []) or []):
            add_check(check)
        self._populate_quality_flow_tree(report.get("flow_hierarchy", {}))
        self._filter_qc_rows()
        self._update_qc_overlay()
        self._refresh_workflow_status()

    def _on_export_quality_report(self):
        if self._pipeline_running_guard("exporting QC"):
            return
        if not self.workspace.data_loaded:
            self.log("Load a case before exporting a QC report.")
            return
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
            before = mutation_snapshot(self.workspace)
            previous_planes = list(self.workspace.planes)
            for plane in imported:
                plane.placement["source_plane_id"] = plane.plane_id
                plane.plane_id = uuid.uuid4().hex
                plane.geometry_revision = plane.roi_revision = 0
                plane.metrics = {}
            if operation == "append":
                self.workspace.planes.extend(imported)
                first_imported = old_count
            else:
                self.workspace.planes = list(imported)
                first_imported = 0
            self._last_plane_import_report = report
            self._last_quality_report = None
            self.workspace.analysis_revision += 1
            refresh_plane_consistency(self.workspace)
            self.pipeline._clear_pwv_state(self.workspace)
            for step in (StepId.COMPUTE_PLANE_METRICS, StepId.COMPUTE_PWV):
                self.workspace.pipeline.completed.pop(step.value, None)
                self.workspace.pipeline.skipped.pop(step.value, None)
            self._sync_plane_scene(previous_planes)
            self._plane_undo_stack.push(_PlaneCommand(self, before, mutation_snapshot(self.workspace), "Import planes"))
            self._refresh_all()
            source_path = str(self.workspace.paths.flow_path or self.workspace.paths.segmask_path or "")
            save_plane_positions(
                self.workspace,
                os.path.join(self.pipeline._output_dir(self.workspace), "plane_positions.json"),
                source_path=source_path,
            )
            self._persist_plane_outputs(include_pixelwise=False)
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
        if self._pipeline_running_guard("exporting plane coordinates"):
            return
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
        background = str(self._config_bundle.get("ui", {}).get("background_color") or BACKGROUND_COLOR)
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
        self.workspace.invalidate_segmentation_display()
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
        has_grouped_surfaces = any(
            str(getattr(item, "data_key", "")).startswith("segmask_group_")
            for item in ws.scene_objects.values()
        )
        obj.name = self._segmentation_object_name()
        obj.visible = bool(ws.segmentation.visible) and not has_grouped_surfaces
        obj.opacity = float(ws.segmentation.opacity)
        obj.scalars = "label"
        obj.cmap = "tab10"
        obj.dynamic = True
        obj.show_scalar_bar = show_shared_colorbar
        obj.scalar_bar_title = "Label"
        obj.scalar_bar_cfg = shared_bar_cfg
        # Group surfaces keep their Browser visibility during ordinary refreshes.
        # Only an explicit segmentation visibility action changes every group.

    def _refresh_segmentation_ui(self, force=False):
        panel = self.segmentation_panel
        ws = self.workspace
        seg = ws.segmentation
        self._sync_segmentation_action_state()
        if not force and not panel.isVisible():
            self._segmentation_ui_dirty = True
            return
        self._segmentation_ui_dirty = False
        labels = self._segmentation_labels_for_ui()
        key = (id(labels), int(seg.revision), repr(seg.label_names), repr(seg.label_colors),
               seg.active_source, int(seg.active_label), float(seg.opacity), bool(seg.visible),
               repr(ws.get_active_segmentation_provenance()))
        if not force and getattr(self, "_segmentation_panel_signature", None) == key:
            return
        self._segmentation_panel_signature = key
        self._ensure_segmentation_label_metadata(labels)

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

        panel.table_labels.blockSignals(True)
        panel.table_labels.clearContents()
        if labels is None:
            panel.table_labels.setRowCount(0)
        else:
            values, counts = np.unique(labels, return_counts=True)
            label_counts = {int(value): int(count) for value, count in zip(values, counts) if int(value) != 0}
            label_values = list(label_counts)
            panel.table_labels.setRowCount(len(label_values))
            for row, label_id in enumerate(label_values):
                key = str(int(label_id))
                count = label_counts[label_id]
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
        n_labels = 0 if labels is None else len(label_values)
        active_src = seg.active_source or "none"
        panel.label_status.setText(f"Source: {active_src}   Labels: {n_labels}")

    def _sync_segmentation_action_state(self):
        panel = self.segmentation_panel
        ws = self.workspace
        labeler_running = (
            self._labeler_process is not None
            and self._labeler_process.state() != QtCore.QProcess.ProcessState.NotRunning
        )
        panel.btn_run_auto.setEnabled(bool(ws.data_loaded) and not labeler_running and self._autoseg_thread is None)
        self.btn_run_segmentation_auto.setEnabled(panel.btn_run_auto.isEnabled())
        self.btn_reload_input.setEnabled(bool(ws.data_loaded))
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

    def _refresh_segmentation_preview(self):
        self.workspace.invalidate_segmentation_display()
        self._last_quality_report = None
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
        if ws.get_segmentation_source(source) is None:
            self.log(f"Segmentation source unavailable: {source}")
            return False
        candidate = ws.copy_for_task()
        def prepare(progress, log):
            candidate.activate_segmentation_source(source)
            progress({"stage": "segmentation_update", "message": "Checking segmentation topology…"})
            return self.pipeline.refresh_segmentation_dependents(candidate)
        try:
            reuse = self._run_modal_task("Update Segmentation", "Preparing segmentation…", prepare)
        except TaskCancelled:
            self.log("Segmentation update cancelled; the active workspace is retained.")
            return False
        ws.__dict__.update(vars(candidate))
        if reuse == "geometry":
            self.log("Processed 3D topology unchanged; centerlines and planes retained. Rerun metrics and trajectories.")
        elif reuse == "unchanged":
            self.log("Processed segmentation unchanged; existing geometry and metrics retained.")
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
        ws = self.workspace
        if ws.segmentation.dirty:
            self.log("Apply or cancel segmentation edits before importing a new source.")
            return False
        if not ws.data_loaded:
            self.log("Load a case before importing segmentation.")
            return False
        try:
            from ..algorithms.geometry_grid import grid_shape, bind_segmentation_geometry
            seg, provenance = load_segmentation_file(path, spatial_shape=None, time_count=ws.time_count())
            field = ws.flow_raw if ws.flow_raw is not None else ws.mag_raw
            native_shape = field.shape[:3]
            fine_shape = grid_shape(native_shape, ws.resolution)
            if seg.shape[:3] == fine_shape and (seg.shape[:3] != native_shape or provenance.get("grid") == "shared_1mm"):
                if not np.allclose(provenance.get("resolution", [1., 1., 1.]), 1.):
                    raise ValueError("geometry sidecar must use 1-mm spacing")
                if not np.allclose(provenance.get("origin", ws.origin), ws.origin):
                    raise ValueError("segmentation origin does not match the current input")
                seg = bind_segmentation_geometry(ws, "imported", seg, provenance, self.pipeline._output_dir(ws))
            elif seg.shape[:3] == native_shape:
                ws.set_segmentation_source("imported", seg, provenance=provenance)
            else:
                raise ValueError(f"incompatible segmentation shape {seg.shape[:3]}")
        except Exception as error:
            self.log(f"Import segmentation failed: {type(error).__name__}: {error}")
            return False
        ws.segmentation.import_path = str(path)
        ws.segmentation.input_source = "imported"
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
        checkpoint_name = seg_state.auto_checkpoint or "auto"
        model_argument = (
            str(seg_state.auto_model)
            if backend_token in {"nnunet4d", "nnunet_4d", "4d"}
            and str(seg_state.auto_model or "").lower().endswith((".sh", ".bash"))
            else resolved_model
        )
        cache_path = ws.paths.flow_path if (str(ws.input_state.source_format or "").lower().endswith("h5") and bool(seg_state.write_auto_cache)) else ""
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
        self._bind_task_progress(self._autoseg_progress_dialog, self._autoseg_worker)
        self._autoseg_worker.moveToThread(self._autoseg_thread)
        self._autoseg_thread.started.connect(self._autoseg_worker.run)
        self._autoseg_worker.progress.connect(self._on_autoseg_progress)
        self._autoseg_worker.finished.connect(self._on_autoseg_finished)
        self._autoseg_worker.failed.connect(self._on_autoseg_failed)
        self._autoseg_worker.finished.connect(self._autoseg_thread.quit)
        self._autoseg_worker.failed.connect(self._autoseg_thread.quit)
        self._autoseg_thread.finished.connect(self._autoseg_worker.deleteLater)
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
        dialog = self._autoseg_progress_dialog
        if dialog is not None and dialog.isVisible():
            dialog.update_progress(payload)
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
        if self._autoseg_progress_dialog is not None and self._autoseg_progress_dialog.cancel_token.cancelled:
            self._on_autoseg_failed("Cancelled")
            return
        self._autoseg_result = result

    def _apply_autoseg_result(self, result):
        if self._autoseg_progress_dialog is not None and self._autoseg_progress_dialog.cancel_token.cancelled:
            self._on_autoseg_failed("Cancelled")
            return
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

    def _on_autoseg_failed(self, error_text):
        if str(error_text) == "Cancelled":
            self.log("Automatic segmentation cancelled; no result was applied.")
            return
        text = str(error_text or "").strip()
        if not text:
            text = "Unknown auto segmentation failure."
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        summary = lines[-1] if lines else "Unknown auto segmentation failure."
        self.log(f"Auto segmentation failed: {summary}")
        self.log(text)
        self._autoseg_error = (summary, text)

    def _cleanup_autoseg_task(self):
        cancelled = self._autoseg_progress_dialog is not None and self._autoseg_progress_dialog.cancel_token.cancelled
        result = getattr(self, "_autoseg_result", None)
        self._autoseg_result = None
        self._close_progress_dialog(self._autoseg_progress_dialog)
        self._autoseg_progress_dialog = None
        if self._autoseg_thread is not None:
            self._autoseg_thread.deleteLater()
        self._autoseg_worker = None
        self._autoseg_thread = None
        self._autoseg_started_at = None
        error = getattr(self, "_autoseg_error", None)
        self._autoseg_error = None
        if error is not None:
            self._show_autoseg_error(*error)
        elif result is not None and not cancelled:
            self._apply_autoseg_result(result)

    def _show_autoseg_error(self, summary, text):
        message_box = QtWidgets.QMessageBox(self)
        message_box.setIcon(QtWidgets.QMessageBox.Critical)
        message_box.setWindowTitle("Automatic Segmentation Failed")
        message_box.setText(summary)
        message_box.setInformativeText("No segmentation result was applied. Expand Details for the complete error.")
        message_box.setDetailedText(text)
        message_box.setWindowModality(QtCore.Qt.WindowModal)
        message_box.exec()

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

    def _on_run_auto_segmentation(self):
        if self._autoseg_running_guard("starting automatic segmentation"):
            return
        if not self.workspace.data_loaded:
            self.log("Load a case before running automatic segmentation.")
            return
        self._sync_params_to_ws()
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
            save_segmentation_file(path, seg, resolution=self.workspace.segmentation_spacing, origin=self.workspace.origin, provenance=dict(provenance, grid="shared_1mm"))
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
            "schema_version": 3,
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

    def _open_external_segmentation_editor(self):
        """Prepare a reusable exchange directory, launch Labeler, and re-import on exit."""
        if self._labeler_export_running:
            self.log("SpatioTemporal Labeler export is already running.")
            return
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
        self._labeler_export_running = True
        try:
            segmentation = np.asarray(segmentation, dtype=np.int16)
            mag, flow = np.asarray(ws.mag_raw, dtype=np.float32), np.asarray(ws.flow_raw, dtype=np.float32)
            if segmentation.ndim != 4:
                raise ValueError(f"active segmentation must be XYZT, got shape={segmentation.shape}")
            if mag.ndim == 3:
                mag = np.repeat(mag[..., None], segmentation.shape[3], axis=3)
            if mag.shape[3] != segmentation.shape[3] or flow.shape[3] != segmentation.shape[3] or flow.shape[-1] != 3:
                raise ValueError("magnitude, flow and segmentation must have matching XYZT dimensions")
            metadata = self._labeler_exchange_metadata(segmentation)
            exchange_dir = self._labeler_exchange_directory(metadata)
            def export(progress, log):
                from ..algorithms.geometry_grid import field_on_geometry_grid
                mag_fine = field_on_geometry_grid(mag, ws, segmentation.shape[:3])
                flow_fine = field_on_geometry_grid(flow, ws, segmentation.shape[:3])
                count = 0
                progress({"stage": "labeler_digest", "message": "Checking image and segmentation files…"})
                def written(name):
                    nonlocal count
                    count += 1
                    progress({"stage": "labeler_export", "current": count, "total": 6,
                              "message": f"Exported {name} ({count}/6)"})
                result = export_labeler_exchange(exchange_dir, dict(metadata, grid="shared_1mm", image_sampling="temporary_linear_for_editor"), mag_fine, flow_fine, segmentation,
                                                 ws.segmentation_spacing, ws.origin, written)
                progress({"stage": "labeler_export", "current": 6, "total": 6,
                          "message": "Labeler features ready"})
                return result
            exported = self._run_modal_task("Export Labeler Features", "Preparing SpatioTemporal Labeler…", export)
            feature_paths, label_path = exported["feature_paths"], exported["label_path"]
            reuse_exchange = exported["reused_features"] and exported["reused_labels"]
            if exported["previous_label_path"]:
                self.log(f"Preserved previous Labeler mask: {exported['previous_label_path']}")
        except TaskCancelled:
            self.log("Labeler export cancelled; the editor was not launched.")
            return
        except Exception as exc:
            self.log(f"Could not launch SpatioTemporal Labeler: {type(exc).__name__}: {exc}")
            return
        finally:
            self._labeler_export_running = False

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
        # found on PATH. The checked-out v0.4.7 source is selected above while
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
        self._labeler_origin_workspace = self.workspace
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
            f"Opened SpatioTemporal Labeler v0.4.7 ({'reused' if reuse_exchange else 'exported'} workspace): {exchange_dir}. "
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
        origin_workspace = getattr(self, "_labeler_origin_workspace", self.workspace)
        self._labeler_origin_workspace = None
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
        self._offer_labeler_import(label_path, origin_workspace)

    def _offer_labeler_import(self, label_path, origin_workspace):
        if self.workspace is not origin_workspace:
            self.log(f"Labeler result saved for the previous case: {label_path}")
            return
        app = QtWidgets.QApplication.instance()
        if app.activeModalWidget() is not None:
            QtCore.QTimer.singleShot(250, lambda: self._offer_labeler_import(label_path, origin_workspace))
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
        for obj in self.workspace.scene_objects.values():
            if obj.data_key == "segmask_raw_surface" or obj.data_key.startswith("segmask_group_"):
                if obj.data_key.startswith("segmask_group_"):
                    obj.visible = bool(checked)
                self.scene.apply_object_properties(obj, render=False)
        self.scene.render_all()
        self.ortho_viewer.refresh()

    def _on_segmentation_opacity_changed(self, value):
        self.workspace.segmentation.opacity = float(np.clip(value / 100.0, 0.0, 1.0))
        self._sync_segmentation_scene_object()
        for obj in self.workspace.scene_objects.values():
            if obj.data_key == "segmask_raw_surface" or obj.data_key.startswith("segmask_group_"):
                obj.opacity = float(self.workspace.segmentation.opacity)
                self.scene.apply_object_properties(obj, render=False)
        self.scene.render_all()
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
        fine_shape = np.asarray(self._current_working_labels_3d().shape)
        xyz = np.clip(np.rint(np.array([x, y, z])*self.workspace.resolution).astype(int), 0, fine_shape-1)
        x, y, z = xyz.tolist()
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
        from ..algorithms.geometry_grid import bind_segmentation_geometry
        bind_segmentation_geometry(ws, target_source, seg4d, provenance, self.pipeline._output_dir(ws))
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
        if self._edit_mode is not None or self._plane_edit_enabled:
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
        self.selection_tabs.setCurrentIndex(0)
        self._refresh_analysis_panel()

    def _on_3d_path_picked(self, uid, path_idx):
        if self._edit_mode is not None or self._plane_edit_enabled:
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
        self.selection_tabs.setCurrentIndex(1)
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
        if data_key == "magnitude_volume" or data_key.startswith("content_volume_"):
            return "3D Volumes"
        if data_key == "noise_region":
            return "Noise"
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
            "3D Volumes",
            "Noise",
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
        hierarchy = self.workspace.path_info[path_idx].get("hierarchy", {}) if 0 <= path_idx < len(self.workspace.path_info) else {}
        if not hierarchy:
            return f"Path {path_idx} · {count} {suffix}"
        names = "/".join(hierarchy.get("label_names", [])) or "Unlabeled"
        level = hierarchy.get("group_level")
        grade = f"G{level}" if level is not None else "G?"
        loop = " · Loop" if hierarchy.get("loop_id") else " · Cycle" if hierarchy.get("topological_cycle") else ""
        return f"Path {path_idx} · {names} · {grade}{loop} · {count} {suffix}"

    def _browser_subgroup_nodes(self, container, records, checked=True):
        """Create connected-component containers before arranging flow paths."""
        subgroups = {}
        indices = sorted({record["subgroup_index"] for record in records.values() if "subgroup_index" in record})
        for index in indices:
            members = [record for record in records.values() if record.get("subgroup_index") == index]
            names = "/".join(sorted({name for record in members for name in record.get("label_names", [])})) or "Unlabeled"
            item = QtWidgets.QTreeWidgetItem([f"Subgroup {index + 1} · {names} · {len(members)} paths", "Subgroup"])
            item.setData(0, _BROWSER_NODE_ROLE, "subgroup")
            item.setToolTip(0, f"{members[0]['subgroup_id']}\nOne connected graph component. Inlets and grades are independent of other subgroups.")
            self._browser_make_checkable(item, checked)
            container.addChild(item)
            item.setExpanded(True)
            subgroups[index] = item
        return subgroups

    def _browser_apply_path_hierarchy(self, group_entry):
        """One canonical row per path; joins/loops retain navigable references."""
        nodes = {index: item for (_group, index), item in group_entry["paths"].items()}
        container = group_entry["types"].get("Paths")
        if container is None:
            return
        records = {index: self.workspace.path_info[index].get("hierarchy", {}) for index in nodes
                   if 0 <= index < len(self.workspace.path_info)}
        subgroups = self._browser_subgroup_nodes(container, records)
        loops = {}

        def move(item, parent):
            if item.parent() is not None:
                item.parent().removeChild(item)
            parent.addChild(item)

        for index, record in records.items():
            loop_id = record.get("loop_id")
            if loop_id and loop_id not in loops:
                level = record.get("group_level")
                item = QtWidgets.QTreeWidgetItem([f"Loop · G{level if level is not None else '?'}", "Loop"])
                item.setData(0, _BROWSER_NODE_ROLE, "loop")
                item.setToolTip(0, loop_id + "\nOriginal paths and all connections are retained.")
                self._browser_make_checkable(item)
                subgroups.get(record.get("subgroup_index"), container).addChild(item)
                loops[loop_id] = item
        for index, record in records.items():
            parent_index = record.get("display_parent_path_index")
            loop_id = record.get("loop_id")
            if loop_id:
                move(nodes[index], loops[loop_id])
            elif parent_index in nodes:
                parent_loop = records.get(parent_index, {}).get("loop_id")
                move(nodes[index], loops[parent_loop] if parent_loop else nodes[parent_index])
            else:
                move(nodes[index], subgroups.get(record.get("subgroup_index"), container))
            tooltip = dict(record)
            nodes[index].setToolTip(0, nodes[index].text(0) + "\n" + json.dumps(tooltip, ensure_ascii=False, indent=2))
            for other in record.get("reference_parent_path_indices", []):
                if other == index:
                    continue
                reference = QtWidgets.QTreeWidgetItem([f"↗ Upstream Path {other}", "Connection"])
                reference.setFlags(reference.flags() & ~QtCore.Qt.ItemIsUserCheckable)
                reference.setData(0, _BROWSER_NODE_ROLE, "path_reference")
                reference.setData(0, _BROWSER_PATH_INDEX_ROLE, other)
                reference.setToolTip(0, "Additional graph connection; this reference does not duplicate visibility controls.")
                nodes[index].addChild(reference)
        for loop_id, item in loops.items():
            members = [index for index, record in records.items() if record.get("loop_id") == loop_id]
            parents = {parent for index in members for parent in records[index].get("parent_path_indices", [])
                       if parent in nodes and parent not in members}
            if parents:
                parent = min(parents, key=lambda index: (-(records[index].get("group_level") or 0), index))
                parent_loop = records[parent].get("loop_id")
                move(item, loops[parent_loop] if parent_loop else nodes[parent])

    def _browser_add_joint_hierarchies(self):
        for report in self.workspace.graph_hierarchy.get("joint_systems", []):
            top = QtWidgets.QTreeWidgetItem(["Joint cerebral arterial topology", "Analysis"])
            self._browser_make_checkable(top, False)
            top.setToolTip(0, "Combined connected mask topology; source groups, paths and measurements remain separate.")
            self.tree_objects.addTopLevelItem(top)
            records = {record["path_index"]: record for record in report.get("paths", [])}
            subgroups = self._browser_subgroup_nodes(top, records, False)
            nodes, loops = {}, {}
            for record in records.values():
                loop_id = record.get("loop_id")
                if loop_id and loop_id not in loops:
                    loops[loop_id] = QtWidgets.QTreeWidgetItem(["Loop", "Analysis loop"])
                    self._browser_make_checkable(loops[loop_id], False)
                    subgroups.get(record.get("subgroup_index"), top).addChild(loops[loop_id])
            for index, record in records.items():
                grade = record.get("group_level")
                title = f"Branch {index} · {'/'.join(record['label_names']) or 'Unlabeled'} · G{grade if grade is not None else '?'}"
                item = QtWidgets.QTreeWidgetItem([title, "Analysis branch"])
                data_key = f"hierarchy_path_{report['group_name']}_{index}"
                obj = next((obj for obj in self.workspace.scene_objects.values() if obj.data_key == data_key), None)
                if obj is not None:
                    item.setData(0, QtCore.Qt.UserRole, obj.uid)
                    self._browser_make_checkable(item, obj.visible)
                item.setData(0, _BROWSER_NODE_ROLE, "joint_path")
                item.setData(0, _BROWSER_GROUP_ROLE, report["group_name"])
                item.setData(0, _BROWSER_PATH_INDEX_ROLE, index)
                item.setToolTip(0, json.dumps(record, ensure_ascii=False, indent=2))
                top.addChild(item)
                nodes[index] = item
            for index, record in records.items():
                loop_id = record.get("loop_id")
                if loop_id:
                    parent = loops[loop_id]
                else:
                    parent_index = record.get("display_parent_path_index")
                    parent_loop = records.get(parent_index, {}).get("loop_id")
                    base = subgroups.get(record.get("subgroup_index"), top)
                    parent = loops.get(parent_loop, base) if parent_loop else nodes.get(parent_index, base)
                nodes[index].parent().removeChild(nodes[index])
                parent.addChild(nodes[index])
                for other in record.get("reference_parent_path_indices", []):
                    if other != index:
                        reference = QtWidgets.QTreeWidgetItem([f"↗ Upstream Branch {other}", "Connection"])
                        reference.setFlags(reference.flags() & ~QtCore.Qt.ItemIsUserCheckable)
                        nodes[index].addChild(reference)
            for loop_id, item in loops.items():
                members = {i for i, record in records.items() if record.get("loop_id") == loop_id}
                parents = {parent for i in members for parent in records[i]["parent_path_indices"] if parent not in members}
                if parents:
                    parent = min(parents, key=lambda i: (-(records[i].get("group_level") or 0), i))
                    parent_loop = records[parent].get("loop_id")
                    item.parent().removeChild(item)
                    (loops[parent_loop] if parent_loop else nodes[parent]).addChild(item)
                item.setCheckState(0, self._browser_check_state_for_item(item))
            for item in [*subgroups.values(), top]:
                item.setCheckState(0, self._browser_check_state_for_item(item))
            for item in [top, *subgroups.values(), *loops.values()]:
                item.setExpanded(True)

    def _browser_make_checkable(self, item, checked=True):
        item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
        item.setCheckState(0, QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)
        return item

    def _browser_add_leaf(self, parent, obj, *, name=None):
        display_name = name or ('Centerline' if str(obj.data_key).startswith('smooth_path_') else obj.name)
        item = QtWidgets.QTreeWidgetItem([str(display_name), obj.kind.value])
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
            if node.data(0, _BROWSER_NODE_ROLE) in {"path", "loop", "subgroup"}:
                self._browser_sort_path_nodes(node)
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
        if item.data(0, INLINE_ROLE):
            return False
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
        return max(2.0, plane_display_size(self.workspace) * 0.45)

    def _refresh_plane_edit_controls(self):
        selected = 0 <= int(self._selected_plane_index) < len(self.workspace.planes)
        editing = bool(self._plane_edit_enabled and selected)
        if hasattr(self, "btn_edit_plane"):
            self.btn_add_plane.setEnabled(bool(self.workspace.data_loaded) and not editing)
            self.btn_edit_plane.setEnabled(selected)
            self.btn_edit_plane.setText("Apply Plane")
            self.btn_edit_plane.setVisible(editing)
        if hasattr(self, "check_plane_unlocked"):
            blocker = QtCore.QSignalBlocker(self.check_plane_unlocked)
            self.check_plane_unlocked.setChecked(editing)
            self.check_plane_unlocked.setText('Editing plane' if editing else 'Edit plane')
            self.check_plane_unlocked.setEnabled(selected)
            del blocker
        if hasattr(self, "plane_edit_tools"):
            self.btn_add_plane.setVisible(False)
            self.combo_plane_add_source.hide()
            self.btn_undo_plane.setVisible(editing or self._plane_undo_stack.canUndo())
            self.btn_redo_plane.setVisible(editing or self._plane_undo_stack.canRedo())
            self.btn_cancel_plane_edit.setVisible(editing)
            self.plane_edit_tools.setVisible(editing)
            self.combo_plane_add_source.setEnabled(not editing)
            self.btn_undo_plane.setEnabled(bool(self._plane_draft_undo) if editing else self._plane_undo_stack.canUndo())
            self.btn_redo_plane.setEnabled(bool(self._plane_draft_redo) if editing else self._plane_undo_stack.canRedo())
            self.btn_cancel_plane_calculation.setVisible(self._plane_jobs.is_running)

        self._queue_inline_inspector_size()

    def _position_plane_handles(self):
        if self._plane_handles is None or self._plane_drag_index is None:
            return
        plane = self.workspace.planes[int(self._plane_drag_index)]
        self._plane_handles.update(
            self.scene.world_to_display_point(np.asarray(plane.center) + self.workspace.origin),
            self.scene.world_to_display_vector(self._plane_axis_u),
            self.scene.world_to_display_vector(self._plane_axis_v), self._plane_widget_distance())

    def _legacy_add_manual_plane(self):
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
        # Append only the new actor. Rebuilding all planes also discarded
        # unrelated pathlines and recreated every existing scene actor.
        cfg = self.workspace.render_settings.get("plane_render_cfg", {}).get("default", {}) or {}
        self.workspace.add_object(
            name=f"plane {old_count}", kind=ObjectKind.PLANE, data_key=f"plane_{old_count}",
            visible=True, opacity=float(cfg.get("plane_opacity", 0.75)),
            color=str(cfg.get("plane_color") or "yellow"), line_width=2)
        self.scene.sync_from_workspace(rebuild_prefixes=())
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

    def _legacy_toggle_selected_plane_edit(self):
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

    def _legacy_clear_plane_drag_widgets(self):
        self.ortho_viewer.set_plane_rotation_enabled(False)
        self.ortho_viewer._draft_plane_basis = None
        self.ortho_viewer._plane_rotation_reference = None
        self.ortho_viewer._plane_translation_reference = None
        if self._plane_handles is not None:
            self._plane_handles.close()
            self._plane_handles = None
        if self.scene is not None:
            self.scene._plane_selection_locked = False
        self._plane_drag_active = False
        self._plane_drag_index = None
        self._plane_widget_initializing = False
        self._plane_drag_metrics_dirty = False
        self._plane_edit_enabled = False
        self._plane_edit_had_metrics = False
        self._plane_center_widget = None
        self._plane_axis_u_widget = None
        self._plane_axis_v_widget = None
        self._plane_axis_u = None
        self._plane_axis_v = None
        self._refresh_plane_edit_controls()

    def _legacy_update_plane_from_drag(self, plane_idx, center=None, normal=None):
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
                projected, _segment_idx, _alpha, distance, _error, _cum = _project_point_to_path(paths[path_idx], c)
                session = self._plane_edit_session
                if session is None or session.get("path_locked", False):
                    c = projected
                plane.distance = float(distance)
            if np.linalg.norm(c - np.asarray(plane.center, dtype=float).reshape(3)) > tol:
                plane.center = c
                changed = True
        if normal is not None:
            n = np.asarray(normal, dtype=float).reshape(3)
            if np.linalg.norm(n) > 1e-12:
                new_normal = n / np.linalg.norm(n)
                old_normal = np.asarray(plane.normal, dtype=float).reshape(3)
                if np.linalg.norm(new_normal - old_normal) > 1e-5:
                    plane.normal = new_normal
                    changed = True
        if not changed:
            return
        self._last_quality_report = None
        if self._plane_edit_session is None and not self._plane_drag_metrics_dirty:
            # Geometry no longer matches the previously calculated values.
            plane.metrics = {}
            if len(self.workspace.derived.plane_metrics) == len(self.workspace.planes):
                self.workspace.derived.plane_metrics[int(plane_idx)] = {}
            self.workspace.derived.plane_qc = {}
            self.workspace.derived.plane_pixelwise_file = ""
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
                if self._plane_axis_u is None or self._plane_axis_v is None:
                    self._plane_axis_u, self._plane_axis_v = _plane_in_plane_axes(plane.normal)
                self._position_plane_handles()
            finally:
                self._plane_widget_initializing = False
        # vtkHandleWidget renders after its InteractionEvent. Rendering here
        # would draw the same volume twice for every mouse movement.

    def _persist_plane_outputs(self, include_pixelwise=True):
        from ..core.output import atomic_json_write
        if self._plane_edit_session is not None or self.ortho_viewer._roi_transaction is not None:
            return
        out_dir = self.pipeline._output_dir(self.workspace)
        metrics = [dict(selected_metric(self.workspace, i, current_only=False) or {}) for i in range(len(self.workspace.planes))]
        for i, (plane, metric) in enumerate(zip(self.workspace.planes, metrics)):
            if metric:
                metric.update(plane_id=plane.plane_id, plane_index=i, validity=metric_validity(plane, metric))
        self.workspace.derived.plane_metrics = metrics
        qc = copy.deepcopy(self.workspace.derived.plane_qc)
        atomic_json_write(os.path.join(out_dir, "plane_metrics.json"), metrics)
        atomic_json_write(os.path.join(out_dir, "plane_qc.json"), qc)
        if not include_pixelwise:
            from ..algorithms.metrics.export import refresh_pixelwise_validity
            try:
                refresh_pixelwise_validity(os.path.join(out_dir, "plane_metrics_pixelwise.h5"), self.workspace)
            except OSError as exc:
                self.log(f"Could not update previous pixelwise-export validity: {exc}")
        if include_pixelwise and metrics and len(metrics) == len(self.workspace.planes):
            plane_pixelwise_path = os.path.join(out_dir, "plane_metrics_pixelwise.h5")
            try:
                _, plane_pixelwise = augment_plane_metrics_with_derived(
                    metrics,
                    self.workspace.planes,
                    self.workspace.segmentation_analysis_binary,
                    self.workspace.segmentation_spacing,
                    self.workspace.origin,
                    branch_labels_3d=self.workspace.geometry_branch_labels,
                    segmentation_labels_3d=self.workspace.segmentation_analysis_labels_3d,
                    segmentation_labels_4d=self.workspace.segmentation_analysis_labels,
                    tke_array=self.workspace.derived.tke_array,
                    pressure_gradient_array=self.workspace.derived.pressure_gradient_array,
                    relative_pressure_array=self.workspace.derived.relative_pressure_array,
                    wss_surfaces=self.workspace.derived.wss_surfaces,
                    field_spacing=self.workspace.resolution,
                )
                save_plane_pixelwise_h5(plane_pixelwise_path, plane_pixelwise, rr_ms=self.workspace.rr, source_format=self.workspace.input_state.source_format)
                self.workspace.derived.plane_pixelwise_file = plane_pixelwise_path
            except Exception:
                pass
        try:
            self.pipeline._save_planes_json(self.workspace)
        except Exception:
            pass

    def _legacy_finalize_plane_drag(self, plane_idx):
        if not (0 <= int(plane_idx) < len(self.workspace.planes)):
            return
        if self._plane_edit_enabled and not self._plane_drag_metrics_dirty:
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
        if self._plane_edit_enabled and not self._plane_edit_had_metrics:
            self._persist_plane_outputs(include_pixelwise=False)
            self._plane_drag_metrics_dirty = False
            self._refresh_selection_info()
            return
        if self.workspace.flow_raw is None or self.workspace.segmask_binary is None:
            self._persist_plane_outputs(include_pixelwise=False)
            self._plane_drag_metrics_dirty = False
            return
        if self._plane_drag_metrics_dirty or len(self.workspace.derived.plane_metrics) != len(self.workspace.planes):
            self._recompute_dragged_plane_metrics(persist=True)
        else:
            self._persist_plane_outputs(include_pixelwise=False)

    def _on_plane_roi_changed(self, plane_idx):
        if not (0 <= int(plane_idx) < len(self.workspace.planes)):
            return
        invalidate_planes(self.workspace, [self.workspace.planes[int(plane_idx)].plane_id], "Measurement ROI changed", roi=True)
        self._last_quality_report = None
        self._selected_plane_index = int(plane_idx)
        self._plane_drag_index = int(plane_idx)
        self._plane_drag_metrics_dirty = True
        # A newly generated plane may not have metrics yet. Editing its
        # contour only changes the saved ROI definition; defer all metric
        # work until the user explicitly runs the plane-metrics step.
        metrics = self.workspace.derived.plane_metrics
        has_selected_metric = (
            len(metrics) == len(self.workspace.planes)
            and bool(metrics[int(plane_idx)])
        )
        if not has_selected_metric:
            self._persist_plane_outputs(include_pixelwise=False)
            self.ortho_viewer._selected_plane_idx = int(plane_idx)
            self.ortho_viewer.refresh_plane_contours()
            self._plane_drag_metrics_dirty = False
            self.log(f"Plane {int(plane_idx)} contour saved; metric calculation deferred")
            return
        # Recalculate this plane's time series when later frames may depend
        # on the edited contour. Fully observed contours keep the fast path.
        metric = metrics[int(plane_idx)]
        frame = int(self.workspace.current_t)
        # A new/deleted observation can alter both neighboring gaps, including
        # the cycle boundary, even if the previous result was fully observed.
        self._recompute_dragged_plane_metrics(persist=True, frame_only=False)
        self.log(f"Plane {int(plane_idx)} metric ROI updated")

    def _legacy_activate_plane_drag_widgets(self, plane_idx):
        if self._edit_mode is not None or not (0 <= int(plane_idx) < len(self.workspace.planes)):
            return
        if self._plane_drag_active and int(self._plane_drag_index) == int(plane_idx):
            return
        if self._play_timer.isActive():
            self._on_pause()
        self._clear_plane_drag_widgets()
        plane = self.workspace.planes[int(plane_idx)]
        origin = np.asarray(self.workspace.origin, dtype=float).reshape(3)
        center = np.asarray(plane.center, dtype=float).reshape(3) + origin
        normal = np.asarray(plane.normal, dtype=float).reshape(3)
        if np.linalg.norm(normal) <= 1e-12:
            normal = np.array([1.0, 0.0, 0.0], dtype=float)
        normal = normal / np.linalg.norm(normal)
        axis_u, axis_v = _plane_in_plane_axes(normal)
        self._plane_axis_u = axis_u
        self._plane_axis_v = axis_v
        metrics = self.workspace.derived.plane_metrics
        self._plane_edit_had_metrics = bool(plane.metrics) or (
            len(metrics) == len(self.workspace.planes) and bool(metrics[int(plane_idx)]))

        def _center_cb(new_center):
            self._update_plane_from_drag(
                plane_idx,
                center=self.scene.display_to_world_point(new_center) - origin,
            )

        def _end_cb():
            self._end_plane_gesture()
            # Preview only: no time-series metrics, pathlines or disk writes.
            if self._plane_drag_metrics_dirty:
                self.ortho_viewer.set_selected_plane(int(plane_idx))
                self.ortho_viewer.label_plane_metric.setText("Plane metrics: pending — Apply Plane")
                self._set_plane_info_text(f"Plane {plane_idx} · geometry preview\nApply to update metrics, or Cancel to restore.")
            self.plotter.render()

        try:
            self._plane_widget_initializing = True
            self._plane_handles = PlaneDragHandles(
                self.plotter, self.scene, (_center_cb,),
                on_start=self._begin_plane_gesture,
                on_end=_end_cb)
            self._plane_center_widget = self._plane_handles.handles[0][1]
            self._plane_drag_active = True
            self._plane_drag_index = int(plane_idx)
            self._plane_edit_enabled = True
            self.scene._plane_selection_locked = True
            self._position_plane_handles()
            self._plane_handles.enable()
            self.plotter.render()
        except Exception as e:
            self._clear_plane_drag_widgets()
            self.log(f"Plane drag widget error: {type(e).__name__}: {e}")
        finally:
            self._plane_widget_initializing = False
            self._refresh_plane_edit_controls()

    @staticmethod
    def _merge_plane_metric_frame(previous, updated, frame_index, frame_count, rr_ms):
        """Patch one frame into an existing metric without losing other frames."""
        result = dict(previous or {})
        index = int(np.clip(frame_index, 0, max(0, int(frame_count) - 1)))
        series_keys = {
            key for key, value in (previous or {}).items()
            if isinstance(value, list) and len(value) == int(frame_count)
        }
        for key, value in (updated or {}).items():
            if not isinstance(value, list) or len(value) != 1:
                continue
            # A contour can be edited before plane metrics have ever been
            # computed. In that case there is no existing full-length series
            # to patch; initialize only the affected metric series and leave
            # the other cardiac frames at zero until a full metric run.
            if key not in series_keys and key not in result:
                series_keys.add(key)
            if key not in series_keys:
                continue
            patched = list(result.get(key, [0.0] * int(frame_count)))
            if len(patched) != int(frame_count):
                patched = [0.0] * int(frame_count)
            patched[index] = value[0]
            result[key] = patched
            if key.endswith("_t"):
                scalar_key = key[:-2]
                if scalar_key in result:
                    finite = np.asarray(patched, dtype=float)
                    result[scalar_key] = float(np.mean(finite)) if finite.size else 0.0

        flow = np.asarray(result.get("flowrate_mL_s", []), dtype=float)
        flow_forward = np.asarray(result.get("flowrate_forward_mL_s", []), dtype=float)
        flow_reverse = np.asarray(result.get("flowrate_reverse_mL_s", []), dtype=float)
        flow_signed = np.asarray(result.get("flowrate_signed_mL_s", []), dtype=float)
        if flow.size:
            result["netflow_mL_beat"] = float(abs(np.mean(flow) * float(rr_ms) / 1000.0))
        if flow_forward.size:
            result["netflow_forward_mL_beat"] = float(np.mean(flow_forward) * float(rr_ms) / 1000.0)
        if flow_reverse.size:
            result["netflow_reverse_mL_beat"] = float(np.mean(flow_reverse) * float(rr_ms) / 1000.0)
        if flow_forward.size and flow_reverse.size:
            result["net_netflow_signed_mL_beat"] = float(
                result.get("netflow_forward_mL_beat", 0.0)
                - result.get("netflow_reverse_mL_beat", 0.0)
            )
            forward = float(result.get("netflow_forward_mL_beat", 0.0))
            result["reflux_fraction"] = float(
                result.get("netflow_reverse_mL_beat", 0.0) / forward
            ) if forward > 1e-12 else 0.0
        for series_key, scalar_key in (
            ("meanv_cm_s_t", "meanv_cm_s"),
            ("meanv_forward_cm_s_t", "meanv_forward_cm_s"),
            ("meanv_reverse_cm_s_t", "meanv_reverse_cm_s"),
            ("meanv_signed_cm_s_t", "meanv_signed_cm_s"),
        ):
            values = np.asarray(result.get(series_key, []), dtype=float)
            if values.size:
                result[scalar_key] = float(np.mean(values))
        return result

    def _recompute_plane_metric_frame(self, plane_idx, frame_index, include_derived):
        """Recompute only the frame touched by a contour edit."""
        ws = self.workspace
        plane = ws.planes[int(plane_idx)]
        frame = int(frame_index)
        frame_plane = copy.copy(plane)
        frame_ops = list(
            (getattr(plane, "roi_edit_operations", {}) or {}).get(str(frame), []) or []
        )
        # The metric code uses the local index of the supplied 4-D arrays.
        # Remap the frame-local operation to index zero for a one-frame call.
        frame_plane.roi_edit_operations = {"0": copy.deepcopy(frame_ops)} if frame_ops else {}
        flow = np.asarray(ws.flow_raw)
        flow_frame = flow[..., frame:frame + 1, :]
        mask = np.asarray(ws.segmentation_analysis_binary, dtype=bool)
        mask_frame = mask[..., None] if mask.ndim == 3 else mask[..., frame:frame + 1]
        labels = getattr(ws, "segmentation_analysis_labels", None)
        labels_frame = None
        if labels is not None:
            labels = np.asarray(labels)
            labels_frame = labels[..., None] if labels.ndim == 3 else labels[..., frame:frame + 1]
        paths_for_tangent = (
            ws.centerline_paths_smooth
            if len(ws.centerline_paths_smooth) > 0
            else ws.centerline_paths
        )
        partial = compute_plane_metrics(
            flow_frame,
            mask_frame,
            ws.segmentation_spacing,
            ws.origin,
            [frame_plane],
            RR=ws.rr,
            branch_labels_3d=ws.geometry_branch_labels,
            segmentation_labels_3d=getattr(ws, "segmentation_analysis_labels_3d", None),
            segmentation_labels_4d=labels_frame,
            path_info=ws.path_info,
            forks=ws.forks,
            paths=paths_for_tangent,
            return_qc=False,
            skip_support_mesh=True, flow_spacing=ws.resolution,
        )
        if not partial:
            return None
        if "roi_source_frame_t" in partial[0]:
            partial[0]["roi_source_frame_t"] = [
                None if source is None else frame + int(source)
                for source in partial[0]["roi_source_frame_t"]]
        previous = dict(
            ws.derived.plane_metrics[int(plane_idx)]
            or getattr(plane, "metrics", {})
            or {}
        )
        metric = self._merge_plane_metric_frame(
            previous,
            partial[0],
            frame,
            flow.shape[3],
            ws.rr,
        )
        if include_derived:
            def frame_slice(value):
                if value is None:
                    return None
                array = np.asarray(value)
                if array.ndim == 3:
                    return array[..., None]
                if array.ndim == 5 and array.shape[-1] == 3:
                    return array[..., frame:frame + 1, :]
                return array[..., frame:frame + 1]

            wss = getattr(ws.derived, "wss_surfaces", None) or []
            wss_frame = [wss[frame]] if frame < len(wss) else []
            derived_metrics, _ = augment_plane_metrics_with_derived(
                [metric],
                [frame_plane],
                mask_frame,
                ws.segmentation_spacing,
                ws.origin,
                branch_labels_3d=ws.geometry_branch_labels,
                segmentation_labels_3d=ws.segmentation_analysis_labels_3d,
                segmentation_labels_4d=frame_slice(ws.segmentation_analysis_labels),
                tke_array=frame_slice(ws.derived.tke_array),
                pressure_gradient_array=frame_slice(ws.derived.pressure_gradient_array),
                relative_pressure_array=frame_slice(ws.derived.relative_pressure_array),
                wss_surfaces=wss_frame, field_spacing=ws.resolution,
            )
            if derived_metrics:
                metric = self._merge_plane_metric_frame(
                    metric,
                    derived_metrics[0],
                    frame,
                    flow.shape[3],
                    ws.rr,
                )
        return metric

    def _legacy_recompute_dragged_plane_metrics(self, persist=False, frame_only=False):
        if self._plane_drag_index is None or not (0 <= int(self._plane_drag_index) < len(self.workspace.planes)):
            return
        if self.workspace.flow_raw is None or self.workspace.segmask_binary is None:
            if frame_only:
                self.ortho_viewer.refresh_plane_contours()
            else:
                self.ortho_viewer.refresh()
            if persist:
                self._persist_plane_outputs(include_pixelwise=False)
            self._plane_drag_metrics_dirty = False
            return
        plane_idx = int(self._plane_drag_index)
        try:
            if frame_only and (
                len(self.workspace.derived.plane_metrics) != len(self.workspace.planes)
                or not bool(self.workspace.derived.plane_metrics[plane_idx])
            ):
                # There is no prior metric series to patch. The contour-only
                # caller persists the ROI and defers calculation explicitly.
                self._plane_drag_metrics_dirty = False
                return
            include_derived = False if frame_only else bool(
                (self.workspace.derived.wss_surfaces and self.workspace.derived.wss_volume is not None)
                or self.workspace.derived.tke_array is not None
                or self.workspace.derived.pressure_gradient_array is not None
                or self.workspace.derived.relative_pressure_array is not None
            )
            if frame_only:
                # Contour edits are local to one cardiac frame. This must
                # remain true even when the user edits a newly generated
                # plane before running the plane-metrics step at all.
                frame_metric = self._recompute_plane_metric_frame(
                    plane_idx,
                    self.workspace.current_t,
                    include_derived,
                )
                if frame_metric is not None:
                    metrics = (
                        [dict(m) for m in self.workspace.derived.plane_metrics]
                        if len(self.workspace.derived.plane_metrics) == len(self.workspace.planes)
                        else [{} for _ in self.workspace.planes]
                    )
                    metrics[plane_idx] = frame_metric
                    metrics, qc = apply_internal_consistency_to_metrics(
                        metrics,
                        path_info=self.workspace.path_info,
                        forks=self.workspace.forks,
                    )
                    self.workspace.derived.plane_metrics = metrics
                    self.workspace.derived.plane_qc = qc
                    for i, metric in enumerate(metrics):
                        if i < len(self.workspace.planes):
                            self.workspace.planes[i].metrics = dict(metric)
            elif len(self.workspace.derived.plane_metrics) != len(self.workspace.planes):
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
                    self.workspace.segmentation_analysis_binary,
                    self.workspace.segmentation_spacing,
                    self.workspace.origin,
                    [self.workspace.planes[plane_idx]],
                    RR=self.workspace.rr,
                    branch_labels_3d=self.workspace.geometry_branch_labels,
                    segmentation_labels_4d=self.workspace.segmentation_analysis_labels,
                    path_info=self.workspace.path_info,
                    forks=self.workspace.forks,
                    paths=paths_for_tangent,
                    return_qc=False,
                    flow_spacing=self.workspace.resolution,
                )
                if partial_metrics and include_derived:
                    partial_metrics, _ = augment_plane_metrics_with_derived(
                        partial_metrics,
                        [self.workspace.planes[plane_idx]],
                        self.workspace.segmentation_analysis_binary,
                        self.workspace.segmentation_spacing,
                        self.workspace.origin,
                        branch_labels_3d=self.workspace.geometry_branch_labels,
                        segmentation_labels_3d=self.workspace.segmentation_analysis_labels_3d,
                        segmentation_labels_4d=self.workspace.segmentation_analysis_labels,
                        tke_array=self.workspace.derived.tke_array,
                        pressure_gradient_array=self.workspace.derived.pressure_gradient_array,
                        relative_pressure_array=self.workspace.derived.relative_pressure_array,
                        wss_surfaces=self.workspace.derived.wss_surfaces,
                        field_spacing=self.workspace.resolution,
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
            if frame_only:
                self.ortho_viewer.refresh_plane_contours()
            else:
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
        metric = selected_metric(self.workspace, int(plane_idx))
        if not metric:
            state = metric_validity(plane)
            self._set_plane_info_text(f"Plane {plane_idx} | Path {plane.path_index}\n" +
                ("Preview / Needs recalculation" if state == "stale" else "Not calculated") + " · —")
            return
        t = int(np.clip(self.workspace.current_t, 0, max(0, self.workspace.time_count() - 1)))
        valid = metric.get("sample_valid_t", [])
        if t < len(valid) and not valid[t]:
            self._set_plane_info_text(f"Plane {plane_idx} · t={t}\nNot assessable · Empty or non-finite ROI · —")
            return
        fr = metric.get("flowrate_mL_s", [])
        ar = metric.get("area_mm2", [])
        mv = metric.get("meanv_cm_s_t", [])
        def display(value, digits):
            return "—" if value is None or not np.isfinite(float(value)) else f"{float(value):.{digits}f}"
        flow_t = display(fr[t] if len(fr) > t else None, 4)
        area_t = display(ar[t] if len(ar) > t else None, 3)
        meanv_t = display(mv[t] if len(mv) > t else None, 3)
        complete = not valid or all(valid)
        peak = display(metric.get("peakv_cm_s") if complete else None, 3)
        net = display(metric.get("netflow_mL_beat") if complete else None, 4)
        path_dir = metric.get("path_direction", "")
        header = f"Plane {int(plane_idx)} | Path {int(metric.get('path_index', plane.path_index))}"
        if path_dir:
            header += f" {path_dir}"
        path_ic_value = metric.get("path_ic")
        path_ic_text = "undefined" if path_ic_value is None else f"{float(path_ic_value):.3f}"
        text_block = (
            f"{header}\n"
            f"t={t}  Flow Rate={flow_t} mL/s  Area={area_t} mm^2  Mean Velocity={meanv_t} cm/s\n"
            f"Peak Velocity={peak} cm/s  Net Flow={net} mL/beat  IC={path_ic_text}"
        )
        if not complete:
            text_block += "\nCycle summary not assessable: some phases have missing ROI samples."
        carried = metric.get("roi_carried_forward_t", [])
        sources = metric.get("roi_source_frame_t", [])
        interpolated = metric.get("roi_interpolated_t", [])
        endpoints = metric.get("roi_source_frames_t", [])
        if t < len(interpolated) and interpolated[t] and t < len(endpoints):
            text_block += f"\nROI interpolated between phases {endpoints[t]}; velocity sampled at phase {t}."
        elif t < len(carried) and carried[t] and t < len(sources):
            text_block += f"\nROI held from the only available phase {sources[t]}; velocity sampled at phase {t}."
        notes = "\n".join(text_block.splitlines()[3:])
        ic_rows = consistency_rows(self.workspace, int(metric.get("path_index", plane.path_index)), metric)
        for name, value, _unit in ic_rows[1:]:
            text_block += f"\n{name}: {value}"
        self._set_plane_info_text(text_block, summary=(
            f"Plane {int(plane_idx)}", f"Path {int(metric.get('path_index', plane.path_index))} · {path_dir} · Phase {t}".replace(" ·  · ", " · "),
            [("Flow rate", display(fr[t] if len(fr) > t else None, 2), "mL/s"),
             ("Area", display(ar[t] if len(ar) > t else None, 2), "mm²"),
             ("Mean velocity", display(mv[t] if len(mv) > t else None, 2), "cm/s"),
             ("Peak velocity", display(metric.get("peakv_cm_s") if complete else None, 2), "cm/s"),
             ("Net flow", display(metric.get("netflow_mL_beat") if complete else None, 2), "mL/beat"),
             *ic_rows], notes))

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
        hierarchy = info.get("hierarchy", {})
        if hierarchy:
            text_block += (f"\nGroup level: {hierarchy.get('group_level')} ({hierarchy.get('level_status')})"
                           f" | label-local levels: {hierarchy.get('label_local_levels', {})}"
                           f"\nParents: {hierarchy.get('parent_path_indices', [])}"
                           f" | children: {hierarchy.get('child_path_indices', [])}"
                           f" | loop: {hierarchy.get('loop_id') or 'none'}")
        rows = [
            ("Label", f"{owner_name} ({owner_label})", ""),
            ("Nodes", f"{int(info.get('start_node', -1))} → {int(info.get('end_node', -1))}", ""),
            ("Incoming", ", ".join(map(str, incoming)) or "—", ""),
            ("Outgoing", ", ".join(map(str, outgoing)) or "—", ""),
        ]
        ic_rows = consistency_rows(self.workspace, int(path_idx))
        rows.extend(ic_rows)
        text_block += "\n" + " · ".join(f"{name}: {value}" for name, value, _unit in ic_rows)
        if hierarchy:
            rows.extend([
                ("Group level", str(hierarchy.get("group_level", "—")), ""),
                ("Parents", ", ".join(map(str, (hierarchy.get("parent_path_indices") or []))) or "—", ""),
                ("Children", ", ".join(map(str, (hierarchy.get("child_path_indices") or []))) or "—", ""),
                ("Loop", str(hierarchy.get("loop_id") or "—"), ""),
            ])
        fork_notes = []
        for fork in self.workspace.forks:
            if int(path_idx) in fork.get("left", []) or int(path_idx) in fork.get("right", []):
                left = ", ".join(map(str, fork.get("left", []))) or "—"
                right = ", ".join(map(str, fork.get("right", []))) or "—"
                fork_notes.append(f"Fork {int(fork.get('node', -1))} · Left {left} · Right {right}")
        notes = "\n".join(fork_notes)
        if hierarchy:
            levels = hierarchy.get("label_local_levels") or {}
            if not isinstance(levels, dict):
                levels = {}
            formatted = []
            for label, level in levels.items():
                name = id_to_name.get(int(label), label) if str(label).isdigit() else label
                formatted.append(f"{name}: {level}")
            extra = [str(hierarchy.get("level_status") or "")]
            if formatted:
                extra.append("Label levels · " + "; ".join(formatted))
            notes = "\n".join(value for value in [notes, *extra] if value)
        self._set_path_info_text(text_block, summary=(
            f"Path {int(path_idx)}", str(info.get("direction_text", "")), rows, notes))

    def _select_browser_item_by_uid(self, uid):
        self.tree_objects.blockSignals(True)
        item = self._find_browser_item_by_uid(uid)
        if item is not None:
            parent = item.parent()
            while parent is not None:
                parent.setExpanded(True)
                parent = parent.parent()
            self.tree_objects.setCurrentItem(item)
            item.setExpanded(True)
            self.tree_objects.blockSignals(False)
            self._sync_context_panels()
            return
        self.tree_objects.blockSignals(False)

    def _clear_browser_selection(self):
        self.tree_objects.blockSignals(True)
        self.tree_objects.clearSelection()
        self.tree_objects.blockSignals(False)
        self._sync_context_panels()

    def _set_plane_info_text(self, text, summary=None):
        msg = str(text).strip() if text else "No plane selected."
        if summary is None:
            self.text_plane_info.setPlainText(msg)
        else:
            self.text_plane_info.set_summary(*summary, plain_text=msg)
        self._queue_inline_inspector_size()

    def _set_path_info_text(self, text, summary=None):
        msg = str(text).strip() if text else "No path selected."
        if summary is None:
            self.text_path_info.setPlainText(msg)
        else:
            self.text_path_info.set_summary(*summary, plain_text=msg)
        self._queue_inline_inspector_size()

    def _autoseg_running_guard(self, action_text):
        if getattr(self, "_dicom_conversion_thread", None) is not None:
            self.log(f"DICOM conversion is running. Wait before {action_text}.")
            return True
        if self._autoseg_thread is None:
            return False
        self.log(f"Auto segmentation is running. Wait for it to finish before {action_text}.")
        return True

    def _refresh_selection_info(self):
        self._update_qc_overlay()
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
        self._sync_context_panels()
        self._refresh_analysis_panel()

    def log(self, text):
        self.console.append(str(text))

    def _create_progress_dialog(self, title, label_text):
        self._on_pause()
        self._timeline_scrub_timer.stop()
        self._pending_timeline_value = None
        self._seg_surface_rebuild_timer.stop()
        dlg = TaskProgressDialog(title, label_text, self)
        dlg.show()
        QtWidgets.QApplication.processEvents()
        return dlg

    def _bind_task_progress(self, dialog, worker):
        worker.cancel_token = dialog.cancel_token

    def _run_modal_task(self, title, message, function):
        """Wait in a Qt event loop while numerical/file work runs off the UI thread."""
        dialog = self._create_progress_dialog(title, message)
        worker = _FunctionTaskWorker(function)
        self._bind_task_progress(dialog, worker)
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        loop = QtCore.QEventLoop(self)
        outcome = {}
        worker.progress.connect(dialog.update_progress)
        worker.log_message.connect(self.log)
        worker.finished.connect(lambda result: outcome.update(result=result))
        worker.failed.connect(lambda error: outcome.update(error=error))
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(loop.quit)
        thread.started.connect(worker.run)
        thread.start()
        loop.exec()
        thread.wait()
        thread.deleteLater()
        self._close_progress_dialog(dialog)
        dialog.cancel_token.check()
        if "error" in outcome:
            raise RuntimeError(outcome["error"])
        return outcome.get("result")

    def _close_progress_dialog(self, dialog):
        if dialog is None:
            return
        try:
            if hasattr(dialog, "finish"):
                dialog.finish()
            else:
                dialog.close()
            dialog.deleteLater()
        except Exception:
            pass


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
        pane = getattr(self, '_html_document_panes', {}).get('qc')
        if pane is not None:
            from ..report_media import media_signature
            if getattr(pane, 'qc_signature', None) != media_signature(self.workspace):
                pane.set_stale(True)
        if self._workflow_stage_key() == 'review' and not self.workspace.data_loaded:
            self.qc_page.setCurrentWidget(self.qc_empty)

    def _refresh_browser(self):
        view_state = self._capture_browser_view()
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
                    top.setForeground(0, QtGui.QBrush(readable_color(color, self)))
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
            if str(obj.data_key).startswith("hierarchy_path_"):
                continue
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
            self._browser_apply_path_hierarchy(group_entry)
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
                        child = type_item.child(k)
                        child.setExpanded(any(child.child(n).data(0, _BROWSER_NODE_ROLE) in {"path", "loop"}
                                              for n in range(child.childCount())))
        self._browser_add_joint_hierarchies()
        self._compact_browser_types()
        self._restore_browser_view(view_state)
        self.tree_objects.blockSignals(False)
        self._sync_render_category_visibility_controls()
        self._refresh_browser_opacity_control()
        self._refresh_content_render_controls()

        self.chk_selection_on_top.blockSignals(True)
        self.chk_selection_on_top.setChecked(self.workspace.render_settings.get("selected_layer_on_top", True))
        self.chk_selection_on_top.blockSignals(False)
        self.scene.set_selected_layers([obj.uid for obj in self._browser_selected_objects()], render=False)
        self._sync_context_panels()

    def _render_category_objects(self, category):
        if category == "planes":
            return [obj for obj in self.workspace.scene_objects.values() if obj.kind == ObjectKind.PLANE]
        if category == "pathlines":
            return [
                obj for obj in self.workspace.scene_objects.values()
                if str(getattr(obj, "data_key", "") or "").startswith("pathline_")
            ]
        return []

    def _sync_render_category_visibility_controls(self):
        for category, checkbox in (
            ("planes", getattr(self, "chk_all_planes_visible", None)),
            ("pathlines", getattr(self, "chk_all_pathlines_visible", None)),
        ):
            if checkbox is None:
                continue
            objects = self._render_category_objects(category)
            checkbox.blockSignals(True)
            checkbox.setEnabled(bool(objects))
            if not objects or all(bool(getattr(obj, "visible", False)) for obj in objects):
                checkbox.setCheckState(QtCore.Qt.Checked if objects else QtCore.Qt.Unchecked)
            elif not any(bool(getattr(obj, "visible", False)) for obj in objects):
                checkbox.setCheckState(QtCore.Qt.Unchecked)
            else:
                checkbox.setCheckState(QtCore.Qt.PartiallyChecked)
            checkbox.blockSignals(False)

    def _set_render_category_visibility(self, category, visible):
        objects = self._render_category_objects(category)
        if not objects:
            self._sync_render_category_visibility_controls()
            return
        previous = self.tree_objects.blockSignals(True)
        try:
            # Update checks in place so visibility changes preserve the current
            # selection, expansion, and selected-plane display filter.
            for obj in objects:
                obj.visible = bool(visible)
                self.scene.apply_object_properties(obj, render=False, refresh_scalar_bar=False)
                item = self._find_browser_item_by_uid(obj.uid)
                if item is not None:
                    item.setCheckState(0, QtCore.Qt.Checked if visible else QtCore.Qt.Unchecked)
            for obj in objects:
                item = self._find_browser_item_by_uid(obj.uid)
                if item is not None:
                    self._sync_browser_parent_states(item)
        finally:
            self.tree_objects.blockSignals(previous)
        self._sync_render_category_visibility_controls()
        self._refresh_browser_opacity_control()
        self._refresh_scene()

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
        self._refresh_layer_appearance(objects)
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
            if obj.data_key == "noise_region" or obj.data_key.startswith("segmask_"):
                continue
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
        self.render_range_controls.setVisible(obj is not None)
        self._size_layer_appearance()
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

    def _on_selection_on_top_changed(self, checked):
        self.scene.set_selected_layers([obj.uid for obj in self._browser_selected_objects()], on_top=checked)

    def _apply_browser_selection(self):
        if self._plane_edit_enabled:
            current_uid = self._find_uid_by_indexed_data_key("plane", self._plane_drag_index)
            self._select_browser_item_by_uid(current_uid)
            self.statusBar().showMessage("Apply or cancel the current plane draft before changing selection.", 5000)
            return
        self.scene.set_selected_layers([obj.uid for obj in self._browser_selected_objects()], render=False)
        self._refresh_content_render_controls()
        self._refresh_browser_opacity_control()
        self._refresh_render_range_control()
        items = self.tree_objects.selectedItems()
        item = items[0] if items else None
        uid = item.data(0, QtCore.Qt.UserRole) if item is not None else None
        node_type = item.data(0, _BROWSER_NODE_ROLE) if item is not None else None
        if self.selection_tabs.currentIndex() != 2:
            selected_object = self.workspace.scene_objects.get(uid)
            if node_type in {"joint_path", "path", "path_reference"} or (selected_object and selected_object.kind == ObjectKind.BRANCH):
                self.selection_tabs.setCurrentIndex(1)
            elif selected_object and (selected_object.kind == ObjectKind.PLANE or _parse_pathline_index(selected_object.data_key) is not None):
                self.selection_tabs.setCurrentIndex(0)
        if uid is not None and node_type == "joint_path":
            self.workspace.selected_path_index = -1
            self._selected_plane_index = -1
            self._clear_plane_drag_widgets()
            self.scene.highlight_plane(None)
            self.scene.highlight_path(uid)
            self.scene.show_forks_for_path(-1)
            self.ortho_viewer.set_selected_plane(None)
            self._set_plane_info_text("")
            self._set_path_info_text(item.toolTip(0))
            self._refresh_analysis_panel()
            return
        if uid is None and node_type in {"path", "path_reference"}:
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
        if item.data(0, INLINE_ROLE):
            return
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
        self._sync_render_category_visibility_controls()
        self._refresh_scene()

    def _on_browser_ctx_menu(self, pos):
        item = self.tree_objects.itemAt(pos)
        if item is None or item.data(0, INLINE_ROLE):
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
        if self.workspace.flow_raw is None or (self.workspace.streamline_params.pathline_seed_region == "segmentation" and self.workspace.segmask_raw is None):
            self.log("Pathlines require velocity and the selected seed region.")
            return
        active_indices = {
            int(idx)
            for idx in self.workspace.active_pathline_plane_indices
            if 0 <= int(idx) < len(self.workspace.planes)
            and self._find_uid_by_indexed_data_key("pathline", int(idx)) is not None
        }
        pending_indices = set(self.workspace.input_state.metadata.get("gui_pending_pathline_planes", []))
        missing_indices = [idx for idx in valid_indices if idx not in active_indices or idx in pending_indices]
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
        if prepare_segmentation and self.workspace.segmask_raw is not None:
            try:
                self.pipeline.preprocess(self.workspace)
            except Exception as exc:
                self.log(f"Pathline preparation failed: {type(exc).__name__}: {exc}")
                self._close_progress_dialog(self._pathline_progress_dialog)
                self._pathline_progress_dialog = None
                return
        if self.workspace.streamline_params.pathline_seed_region == "segmentation" and self.workspace.segmask_binary is None:
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
        if ws.flow_raw is None or (ws.streamline_params.pathline_seed_region == "segmentation" and ws.segmask_binary is None):
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
            ws.segmask_binary if params.pathline_seed_region == "segmentation" else np.ones(ws.flow_raw.shape[:3], bool),
            ws.branch_labels,
            ws.planes,
            plane_indices,
            # Pathlines are launched from phase zero and retain that complete
            # cardiac-cycle trajectory while the timeline is played.
            time_index=int(ws.current_t) if params.pathline_seed_region == "contour" else 0,
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
                "pathline_seed_region": params.pathline_seed_region,
                "pathline_seed_distribution": params.pathline_seed_distribution,
                "pathline_grid_spacing_mm": params.pathline_grid_spacing_mm,
                "pathline_integration_step_mm": params.pathline_integration_step_mm,
                "input_signature": workspace_signature(ws, include_analysis=False),
                "plane_signatures": {int(i): plane_signature(ws.planes[int(i)]) for i in plane_indices},
            },
            seed_segmentation=ws.segmask_binary, segmentation_labels=ws.segmask_labels,
        )
        self._bind_task_progress(self._pathline_progress_dialog, self._pathline_worker)
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
        self._pathline_thread.finished.connect(self._pathline_worker.deleteLater)
        self._pathline_thread.finished.connect(self._cleanup_pathline_task)
        self.statusBar().showMessage("Generating pathlines...")
        self._pathline_thread.start()
        return True

    def _on_pathline_progress(self, payload):
        data = dict(payload or {})
        dialog = self._pathline_progress_dialog
        if dialog is not None and dialog.isVisible():
            dialog.update_progress(data)

    def _on_pathline_finished(self, payload):
        if self._pathline_progress_dialog is not None and self._pathline_progress_dialog.cancel_token.cancelled:
            self._on_pathline_failed("Cancelled")
            return
        selected_plane_idx = self._pathline_selected_plane_idx
        self._pathline_selected_plane_idx = None
        data = dict(payload or {})
        if data.get("input_signature") is not None and data["input_signature"] != workspace_signature(self.workspace, include_analysis=False):
            self._on_pathline_failed("Input changed; discarded obsolete pathline result")
            return
        if any(not 0 <= int(index) < len(self.workspace.planes) or plane_signature(self.workspace.planes[int(index)]) != signature
               for index, signature in (data.get("plane_signatures") or {}).items()):
            self._on_pathline_failed("Plane or contour changed; discarded obsolete pathline result")
            return
        indices = [int(idx) for idx in data.get("plane_indices", [])]
        meshes = dict(data.get("meshes", {}) or {})
        seed_payload = {
            int(plane_idx): np.asarray(seeds, dtype=float)
            for plane_idx, seeds in dict(data.get("seeds", {}) or {}).items()
            if seeds is not None
        }
        # Publish replacement caches only after integration succeeds. A failed
        # rerun keeps the previous visible trajectories and their caches.
        for idx in indices:
            self.workspace.pathline_cache.pop(idx, None)
        self.scene.trigger_pathlines(indices, precomputed=meshes)
        pending_indices = set(self.workspace.input_state.metadata.get("gui_pending_pathline_planes", [])) - set(indices)
        self.workspace.input_state.metadata["gui_pending_pathline_planes"] = sorted(pending_indices)
        if not pending_indices and set(self.workspace.active_pathline_plane_indices) >= set(range(len(self.workspace.planes))):
            self.workspace.pipeline.mark_done(StepId.PLANE_STREAMLINES)
        self.workspace.pathline_seed_cache.update(seed_payload)
        counts = self.workspace.input_state.metadata.setdefault("flow_seed_counts", {}).setdefault("pathlines", {})
        for index, seeds in seed_payload.items():
            counts[self.workspace.planes[index].plane_id] = len(seeds)
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
        incomplete = [index for index, phases in meshes.items() for mesh in phases.values()
                      if mesh is not None and not bool(np.asarray(mesh.field_data.get("CompletedCycle", [1]))[0])]
        if incomplete:
            self.log(f"Pathlines stopped at Maximum steps before the full cycle for planes {incomplete}; increase Maximum steps or Integration step (mm).")
        self.statusBar().showMessage(f"Pathlines completed in {elapsed:.2f}s", 8000)
        self._close_progress_dialog(self._pathline_progress_dialog)
        self._pathline_progress_dialog = None

    def _on_pathline_failed(self, error_text):
        self._pathline_selected_plane_idx = None
        if str(error_text) == "Cancelled":
            self.log("Pathlines cancelled; no new trajectories were applied.")
            self.statusBar().showMessage("Pathlines cancelled", 8000)
        else:
            self.log(f"PATHLINE ERROR:\n{str(error_text).strip()}")
            self.statusBar().showMessage("Pathlines failed", 10000)

    def _cleanup_pathline_task(self):
        self._close_progress_dialog(self._pathline_progress_dialog)
        self._pathline_progress_dialog = None
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
        obj.scalars, obj.show_scalar_bar = None, False
        self.scene.readd_object(obj, refresh_scalar_bar=False)
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
        if self._pipeline_running_guard("deleting scene objects"):
            return
        before = mutation_snapshot(self.workspace)
        previous_planes = list(self.workspace.planes)
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
            remove_planes(self.workspace, [previous_planes[i].plane_id for i in set(plane_indices_removed)])
            refresh_plane_consistency(self.workspace)
            self._last_quality_report = None
            self._sync_plane_scene(previous_planes)
            self._plane_undo_stack.push(_PlaneCommand(self, before, mutation_snapshot(self.workspace), "Delete planes"))
            self._persist_plane_outputs(include_pixelwise=False)
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
        self._refresh_timeline()
        self._refresh_time_views()
        if final or (not self.slider_t.isSliderDown() and not self._keyboard_navigation_active):
            self._refresh_segmentation_ui()
            self._refresh_selection_info()

    def _queue_keyboard_time_step(self, delta):
        count = int(self.workspace.time_count())
        if count <= 1 or self._pipeline_thread is not None:
            return
        if self._play_timer.isActive():
            self._on_pause()
        self._keyboard_navigation_active = True
        base = self.workspace.current_t if self._pending_timeline_value is None else self._pending_timeline_value
        self._pending_timeline_value = (int(base) + int(delta)) % count
        self.lab_t.setText(str(self._pending_timeline_value))
        # A throttle, not a trailing debounce: repeats never restart the timer.
        # It paints throughout a held key, rather than waiting for key release.
        if not self._timeline_scrub_timer.isActive():
            self._timeline_scrub_timer.start()

    def _finish_keyboard_time_step(self):
        if not self._keyboard_navigation_active:
            return
        self._keyboard_navigation_active = False
        self._finish_timeline_scrub()

    def _finish_timeline_scrub(self):
        self._timeline_scrub_timer.stop()
        self._apply_pending_timeline_value(final=True)

    def _on_ortho_time_step(self, delta):
        if int(delta) < 0:
            self._on_prev_frame()
        elif int(delta) > 0:
            self._on_next_frame()

    def _on_prev_frame(self):
        self.workspace.current_t = (self.workspace.current_t - 1) % max(1, self.workspace.time_count())
        self._refresh_timeline()
        self._refresh_time_views()
        self._refresh_segmentation_ui()
        self._refresh_selection_info()

    def _on_next_frame(self):
        T = max(1, self.workspace.time_count())
        self.workspace.current_t = (self.workspace.current_t + 1) % T
        self._refresh_timeline()
        self._refresh_time_views()
        self._refresh_segmentation_ui()
        self._refresh_selection_info()

    def _refresh_time_views(self):
        viewer = self.ortho_viewer
        if viewer._roi_transaction is not None and getattr(viewer, "_roi_edit_frame", self.workspace.current_t) != self.workspace.current_t:
            viewer._cancel_plane_roi()
            self.statusBar().showMessage("ROI draft discarded when changing phase. Apply ROI before navigating to keep it.", 6000)
        with getattr(self.scene, "render_batch", nullcontext)():
            self.scene.update_time(self.workspace.current_t)
            self.ortho_viewer.refresh()
            self._refresh_render_range_control()
        move_phase_cursor(self)

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
        with getattr(self.scene, "render_batch", nullcontext)():
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
        frame_interval_ms = 1000.0 / self.spin_fps.value()
        delay_ms = max(0, int(round(frame_interval_ms - render_elapsed_ms)))
        if self.scene is not None and self._play_timer.isSingleShot():
            self._play_timer.start(delay_ms)

    def _refresh_scene(self):
        try:
            self.scene.render_all()
        except Exception as e:
            self.log(f"VIEW ERROR: {type(e).__name__}: {e}")

    def _load_selected_input_case(self, case, parameter_overrides=None):
        if self._pipeline_running_guard("loading another case") or self._autoseg_running_guard("loading another case"):
            return
        try:
            resolved = resolve_input_case(case)
            previous = self.workspace
            pending = Workspace()
            apply_config_bundle_to_workspace(pending, self._config_bundle)
            # Carry processing settings, but retain source metadata and review
            # arrays in the previous case until the load succeeds.
            self._sync_params_to_ws()
            for name in ("loader_params", "phase_unwrap_params", "noise_removal_params", "skeleton_params",
                         "plane_gen_params", "pwv_params", "streamline_params", "derived_params", "render_settings"):
                setattr(pending, name, copy.deepcopy(getattr(previous, name)))
            for name in ("auto_backend", "auto_model", "auto_checkpoint", "auto_folds", "auto_device", "auto_label_map",
                         "cleanup_4d_components", "cleanup_4d_mode", "cleanup_4d_min_volume_mm3"):
                setattr(pending.segmentation, name, copy.deepcopy(getattr(previous.segmentation, name)))
            pending.paths.segmask_path = resolved.input_path
            pending.paths.flow_path = resolved.input_path
            configured_out = str(self._config_bundle.get("batch", {}).get("output_dir", "") or "").strip()
            if configured_out:
                pending.paths.output_dir = os.path.join(configured_out, resolved.output_name or Path(resolved.input_path).stem)

            def load(progress, log):
                self.pipeline.load_data(pending, log, input_source=resolved, progress_callback=progress,
                                        reuse_existing_corr=True, parameter_overrides=parameter_overrides)
                return pending

            self._run_modal_task("Load Input", f"Loading {resolved.display_name or resolved.input_path}…", load)
            if self._edit_mode is not None:
                self._exit_interactive_edit(False)
            self._clear_plane_drag_widgets()
            self._seg_surface_rebuild_timer.stop()
            self._seg_edit_active = False
            self._reset_segmentation_edit_history()
            self.workspace = pending
            self._plane_undo_stack.clear()
            self._active_input_case = resolved
            self._last_quality_report = None
            self._last_plane_import_report = None
            self.scene.workspace = pending
            self.ortho_viewer.workspace = pending
            self.ortho_viewer.reset_state()
            self.ortho_viewer.update_slider_ranges()
            self.scene.reset_display_reference()
            self.scene.reset_scene()
            self._refresh_all()
            self.log(f"Loaded input: {resolved.display_name or resolved.input_path}. Saved corr applied when available; Background Correction recomputes from source.")
            self._input_signature = self._current_input_signature()
            return True
        except TaskCancelled:
            self.log("Input loading cancelled; the previous workspace is retained.")
        except Exception as e:
            self.log(f"LOAD ERROR: {type(e).__name__}: {e}")
            self.log(traceback.format_exc())
        return False

    def _current_input_signature(self):
        """Stable signature for deciding whether an input reload is necessary."""
        ws = self.workspace
        return json.dumps({
            "path": str(getattr(self._active_input_case, "input_path", "") or ws.paths.flow_path),
            "kind": str(getattr(self._active_input_case, "input_kind", "")),
            "group": getattr(self._active_input_case, "source_group", None),
            "dual_venc_ratios": [ws.loader_params.background_phase_correction.dual_venc_ratio1, ws.loader_params.background_phase_correction.dual_venc_ratio2],
            "parameters": self._input_parameters_from_ui(),
            "dual_venc_mode": dict(getattr(self._active_input_case, "metadata", {}) or {}).get("dual_venc_mode", "dv"),
        }, sort_keys=True, default=str)

    def _prompt_dual_venc_choice(self, case):
        """Ask which LV/HV/DV source to expose for a dual-VENC H5 case."""
        resolved = resolve_input_case(case)
        if resolved.input_kind != "h5":
            return resolved
        features = dict(resolved.metadata or {})
        if "is_dual_venc" not in features:
            try:
                inspected = inspect_h5_input_case(resolved)
            except Exception:
                inspected = {}
            features.update(inspected)
            resolved.metadata.update(inspected)
        dual_info = features.get("dual_venc", {})
        if not bool(features.get("is_dual_venc", False) or (isinstance(dual_info, dict) and dual_info.get("enabled", False))):
            return resolved
        dialog = DualVencSelectDialog(
            lv_venc=dual_info.get("lv_venc") if isinstance(dual_info, dict) else None,
            hv_venc=dual_info.get("hv_venc") if isinstance(dual_info, dict) else None,
            parent=self,
        )
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            self.log(f"Load cancelled: dual-VENC source selection for {resolved.display_name or resolved.input_path}")
            return None
        mode = dialog.selected_mode()
        resolved.metadata = dict(resolved.metadata or {})
        resolved.metadata["dual_venc_mode"] = mode
        selected_dual_info = dict(dual_info) if isinstance(dual_info, dict) else {}
        selected_dual_info.update(selected_mode=mode, mode=mode, enabled=True)
        resolved.metadata["dual_venc"] = selected_dual_info
        self.log(f"Dual-VENC source: {mode.upper()}")
        return resolved

    def _reload_input_case(self):
        if self._active_input_case is None:
            self.log("No input case is loaded.")
            return
        if self._pipeline_running_guard("reloading input") or self._autoseg_running_guard("reloading input"):
            return
        try:
            parameters = self._input_parameters_from_ui()
            signature = self._current_input_signature()
        except ValueError as exc:
            self.log(f"INPUT PARAMETER ERROR: {exc}")
            self.statusBar().showMessage(str(exc), 10000)
            return
        if signature == self._input_signature:
            self.log("Input parameters unchanged; reload skipped.")
            self.statusBar().showMessage("Input unchanged", 4000)
            return
        self.log("Input parameters changed; reloading input and invalidating downstream results.")
        self._load_selected_input_case(self._active_input_case, parameter_overrides=parameters)

    @staticmethod
    def _input_drop_path(mime):
        if not mime.hasUrls():
            return None
        urls = mime.urls()
        if len(urls) != 1 or not urls[0].isLocalFile():
            return None
        path = urls[0].toLocalFile()
        if os.path.isdir(path) or (
            path.lower().endswith((".h5", ".hdf5")) and os.path.isfile(path)
        ):
            return path
        return None

    def dragEnterEvent(self, event):
        if self._input_drop_path(event.mimeData()) is None:
            event.ignore()
            if event.mimeData().hasUrls():
                self.statusBar().showMessage(
                    "Drop one H5/HDF5 file or one DICOM directory accessible on this computer.", 8000
                )
            return
        # Loading never moves or removes the file-manager source.
        event.setDropAction(QtCore.Qt.DropAction.CopyAction)
        event.accept()

    def dragMoveEvent(self, event):
        if self._input_drop_path(event.mimeData()) is None:
            event.ignore()
            return
        event.setDropAction(QtCore.Qt.DropAction.CopyAction)
        event.accept()

    def dropEvent(self, event):
        path = self._input_drop_path(event.mimeData())
        if path is None:
            event.ignore()
            return
        if self._pipeline_running_guard("loading another case") or self._autoseg_running_guard("loading another case"):
            event.ignore()
            self.statusBar().showMessage("Cannot switch cases yet. Finish or cancel the active task or edit; see Log.", 8000)
            return
        event.setDropAction(QtCore.Qt.DropAction.CopyAction)
        event.accept()
        loader = self._load_dicom_directory if os.path.isdir(path) else self._load_h5_path
        # Finish the desktop drag before opening selection/progress dialogs.
        QtCore.QTimer.singleShot(0, lambda: loader(path))

    def _on_load_h5(self):
        if self._pipeline_running_guard("loading another case") or self._autoseg_running_guard("loading another case"):
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load H5", "", "H5 (*.h5 *.hdf5);;All (*)")
        if not path:
            return
        self._load_h5_path(path)

    def _load_h5_path(self, path):
        if self._pipeline_running_guard("loading another case") or self._autoseg_running_guard("loading another case"):
            return
        try:
            cases = discover_h5_input_cases(path)
        except Exception as exc:
            self.log(f"LOAD ERROR: {type(exc).__name__}: {exc}")
            self.statusBar().showMessage("H5 could not be opened; see Log.", 10000)
            return
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
        resolved = self._prompt_dual_venc_choice(resolved)
        if resolved is None:
            return
        self._load_selected_input_case(resolved)

    def _on_load_dicom(self):
        if getattr(self, "_dicom_conversion_thread", None) is not None:
            self.statusBar().showMessage("DICOM conversion is already running.", 8000)
            return
        if self._pipeline_running_guard("loading another case") or self._autoseg_running_guard("loading another case"):
            return
        options = QtWidgets.QFileDialog.Option.DontUseNativeDialog
        root = QtWidgets.QFileDialog.getExistingDirectory(self, "Load DICOM", "", options=options)
        if not root:
            self.statusBar().showMessage("DICOM import cancelled: no directory selected.", 8000)
            return
        self._load_dicom_directory(root)

    def _load_dicom_directory(self, root):
        if self._pipeline_running_guard("loading another case") or self._autoseg_running_guard("loading another case"):
            return
        options = QtWidgets.QFileDialog.Option.DontUseNativeDialog
        self.statusBar().showMessage("DICOM directory selected. Choose a new H5 destination in Save Converted H5.")
        self.log(f"DICOM directory selected: {root}. Waiting for H5 destination.")
        target, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Converted H5", os.path.join(root, "../converted-flow.h5"),
            "H5 (*.h5 *.hdf5)",
            options=options | QtWidgets.QFileDialog.Option.DontConfirmOverwrite,
        )
        if not target:
            self.statusBar().showMessage("DICOM import cancelled: no H5 destination selected.", 8000)
            self.log("DICOM import cancelled before conversion: no H5 destination selected.")
            return
        if not target.lower().endswith((".h5", ".hdf5")):
            target += ".h5"
        if os.path.exists(target):
            message = f"Destination exists: {target}. Choose a new filename or use File > Load H5."
            self.log(f"DICOM CONVERSION ERROR: {message}")
            self.statusBar().showMessage("DICOM import stopped: destination already exists.", 10000)
            QtWidgets.QMessageBox.warning(self, "DICOM import", message)
            return
        self._dicom_conversion_progress = self._create_progress_dialog(
            "Load DICOM", "Converting DICOM and validating H5..."
        )
        self._dicom_conversion_progress.setRange(0, 0)
        self._dicom_conversion_thread = QtCore.QThread(self)
        self._dicom_conversion_worker = _DicomConversionWorker(root, target)
        self._bind_task_progress(self._dicom_conversion_progress, self._dicom_conversion_worker)
        self._dicom_conversion_worker.progress.connect(self._dicom_conversion_progress.update_progress)
        self._dicom_conversion_worker.moveToThread(self._dicom_conversion_thread)
        self._dicom_conversion_thread.started.connect(self._dicom_conversion_worker.run)
        self._dicom_conversion_worker.finished.connect(self._on_dicom_conversion_finished)
        self._dicom_conversion_worker.failed.connect(self._on_dicom_conversion_failed)
        self._dicom_conversion_worker.finished.connect(self._dicom_conversion_thread.quit)
        self._dicom_conversion_worker.failed.connect(self._dicom_conversion_thread.quit)
        self._dicom_conversion_thread.finished.connect(self._dicom_conversion_worker.deleteLater)
        self._dicom_conversion_thread.finished.connect(self._cleanup_dicom_conversion)
        self.log(f"Dicom2H5: {root} -> {target}")
        self.statusBar().showMessage("Converting DICOM to H5...")
        self._dicom_conversion_thread.start()

    def _on_dicom_conversion_finished(self, cases):
        if not self._dicom_conversion_progress.cancel_token.cancelled:
            self._dicom_completed_cases = cases
            self.statusBar().showMessage("DICOM converted. Loading H5...")

    def _apply_dicom_conversion_cases(self, cases):
        self.log(f"Dicom2H5 saved {len(cases)} case(s): {cases[0].input_path}")
        selected = cases[0]
        if len(cases) > 1:
            dialog = H5CaseSelectDialog(cases, self)
            if dialog.exec() != QtWidgets.QDialog.Accepted:
                self.statusBar().showMessage("Converted H5 saved; case selection cancelled.", 10000)
                return
            selected = dialog.selected_case()
            if selected is None:
                return
        if selected is not None:
            if self._load_selected_input_case(selected):
                frames = self.workspace.mag_raw.shape[3] if self.workspace.mag_raw.ndim == 4 else 1
                self.statusBar().showMessage(f"DICOM loaded: {frames} phases.", 10000)
            else:
                self.statusBar().showMessage("H5 conversion completed, but input was not loaded; see Log.", 10000)

    def _on_dicom_conversion_failed(self, details):
        self.log("DICOM conversion cancelled." if str(details) == "Cancelled" else f"DICOM CONVERSION ERROR: {details}")
        self.statusBar().showMessage("DICOM conversion cancelled." if str(details) == "Cancelled" else "DICOM conversion failed; see Log.", 10000)
        self._dicom_conversion_error = None if str(details) == "Cancelled" else str(details)

    @QtCore.Slot()
    def _cleanup_dicom_conversion(self):
        self._close_progress_dialog(self._dicom_conversion_progress)
        # finished() precedes final thread-local/Python wrapper destruction.
        # Join with QThread.wait (which releases the GIL) before deleting the
        # QThread, avoiding a destructor wait while the GUI holds the GIL.
        self._dicom_conversion_thread.wait()
        self._dicom_conversion_thread.deleteLater()
        self._dicom_conversion_worker = None
        self._dicom_conversion_thread = None
        cases = getattr(self, "_dicom_completed_cases", None)
        self._dicom_completed_cases = None
        if cases is not None:
            QtCore.QTimer.singleShot(0, lambda: self._apply_dicom_conversion_cases(cases))
        self._dicom_conversion_progress = None
        details = getattr(self, "_dicom_conversion_error", None)
        self._dicom_conversion_error = None
        if details:
            message = details.strip().splitlines()[-1] + "\nSee Log for details."
            QtCore.QTimer.singleShot(0, lambda: QtWidgets.QMessageBox.warning(self, "DICOM conversion failed", message))

    def _rendering_kwargs(self):
        rendering_cfg = bundle_to_autoflow_kwargs(self._config_bundle)
        runtime_cfg = dict(getattr(self.workspace, "render_settings", {}) or {})
        rendering_cfg.update(copy.deepcopy(runtime_cfg))
        # Export the current layer ranges and visibility of its scalar bar.
        for obj in self.workspace.scene_objects.values():
            prefix = {"wss_surface_live": "wss", "tke_volume": "tke",
                      "pressure_gradient_volume": "pressure_gradient",
                      "relative_pressure_volume": "relative_pressure",
                      "streamlines_live": "streamline"}.get(obj.data_key)
            if prefix:
                rendering_cfg[f"{prefix}_clim"] = obj.clim
                rendering_cfg[f"{prefix}_show_scalar_bar"] = obj.show_scalar_bar
        return {
            "render_background_color": str(rendering_cfg.get("render_background_color") or BACKGROUND_COLOR),
            "fps": int(rendering_cfg.get("fps", 12)),
            "browser_companion": True,
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
            "plane_render_cfg": copy.deepcopy(rendering_cfg.get("plane_render_cfg", {})),
            "window_size": tuple(rendering_cfg.get("window_size", (1600, 1200))),
            "render_style_cfg": copy.deepcopy(rendering_cfg.get("render_style_cfg", {})),
            "shared_colorbar_show": bool(rendering_cfg.get("shared_colorbar_show", True)),
            "shared_colorbar_bar_cfg": dict(rendering_cfg.get("shared_colorbar_bar_cfg", {})),
            "wss_clim": None if rendering_cfg.get("wss_clim") is None else tuple(rendering_cfg["wss_clim"]),
            "wss_show_scalar_bar": bool(rendering_cfg.get("wss_show_scalar_bar", True)),
            "wss_bar_cfg": dict(rendering_cfg.get("wss_bar_cfg", {})),
            "tke_clim": None if rendering_cfg.get("tke_clim") is None else tuple(rendering_cfg["tke_clim"]),
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

        hint = QtWidgets.QLabel("WSS, TKE, and relative-pressure videos require derived data. Streamlines require velocity and the selected seed region; Full FOV needs no segmentation.")
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
        if self._pipeline_running_guard("exporting videos") or self._autoseg_running_guard("exporting videos"):
            return
        if not self.workspace.data_loaded:
            self.log("No data loaded. Use File > Load H5 or File > Load DICOM.")
            return
        options = self._prompt_video_export_options()
        if options is None:
            return
        self._sync_params_to_ws()
        requested = {"plane": bool(options.export_plane), "wss": bool(options.export_wss),
                     "tke": bool(options.export_tke), "pg": bool(options.export_pg),
                     "streamlines": bool(options.export_streamlines)}
        render_config = self._rendering_kwargs()
        workspace = self.workspace.copy_for_task()
        started = time.perf_counter()
        def export(progress, log):
            from ..rendering.jobs import export_videos_in_process
            if workspace.segmask_binary is None and workspace.segmask_raw is not None:
                self.pipeline.preprocess(workspace)
            check_cancelled()
            if requested["wss"] or requested["tke"] or requested["pg"]:
                progress({"stage": "video_prepare", "message": "Computing derived data for video export…"})
                if workspace.segmask_raw is None:
                    raise ValueError("derived videos require segmentation")
                self.pipeline._ensure_derived_metrics(workspace, save_pixelwise=False,
                    refresh_scene_objects=False, compute_wss=requested["wss"],
                    compute_tke=requested["tke"], compute_pressure_gradient=requested["pg"])
            check_cancelled()
            return export_videos_in_process(workspace, options.out_dir, render_config, requested, progress)
        try:
            result = self._run_modal_task("Export Videos", "Preparing video export…", export)
            workspace.paths.output_dir = options.out_dir
            self.workspace.__dict__.update(vars(workspace))
            for name, output in result["outputs"].items():
                self.log(f"[Video Export] {name}: {output or 'skipped: upstream data unavailable'}"
                         f" | time={result['times'].get(name, 0):.2f}s")
            summary_path = self._update_video_export_summary(options.out_dir, requested,
                result["outputs"], result["times"], time.perf_counter() - started)
            self.log(f"[Video Export] summary updated: {summary_path}")
            from ..report_media import register_report_videos
            register_report_videos(self.workspace, result['outputs'], render_config,
                camera_behavior='rotation' if render_config['rotate_dynamic_video'] else 'fixed')
            for name, error in result.get('errors', {}).items():
                self.log(f'VIDEO PLAYBACK WARNING ({name}): {error}')
                record = self.workspace.input_state.metadata.get('report_videos', {}).get(name)
                if record is not None:
                    record['browser_error'] = error
            self._save_current_case_html(show=False)
        except TaskCancelled:
            self.log("Video export cancelled.")
        except Exception as e:
            self.log(f"VIDEO EXPORT ERROR: {type(e).__name__}: {e}")
            self.log(traceback.format_exc())

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
        self._plane_undo_stack.clear()
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
        if getattr(self, "_scene_export_running", False):
            self.log("Current-view video export is running; cancel or wait before " + action_text + ".")
            return True
        if getattr(self.ortho_viewer, "_roi_transaction", None) is not None:
            self.log("Apply or cancel the ROI draft before " + action_text + ".")
            return True
        if self._plane_edit_session is not None and action_text not in {"editing a plane", "adding a plane"}:
            self.log("Apply or cancel the plane draft before " + action_text + ".")
            return True
        if self._plane_jobs.is_running and action_text not in {"editing a plane", "adding a plane"}:
            self.log("Plane measurements are updating; wait or cancel before " + action_text + ".")
            return True
        if getattr(self, "_dicom_conversion_thread", None) is not None:
            self.log(f"DICOM conversion is running. Wait before {action_text}.")
            return True
        if self._pipeline_thread is None:
            return False
        self.log(f"A pipeline task is running. Wait for it to finish before {action_text}.")
        return True

    def _start_pipeline_task(self, steps, task_label):
        if self._pipeline_running_guard("starting another pipeline task"):
            return False
        steps = list(steps)
        if not steps:
            return False
        self._on_pause()
        self._pipeline_task_label = str(task_label)
        self._pipeline_task_steps = list(steps)
        self._pipeline_progress_dialog = self._create_progress_dialog(
            self._pipeline_task_label,
            f"Preparing {self._pipeline_task_label}...",
        )
        self._pipeline_progress_dialog.setRange(0, len(steps))
        self._pipeline_progress_dialog.setValue(0)
        self._pipeline_thread = QtCore.QThread(self)
        self._pipeline_worker = _PipelineTaskWorker(self.pipeline, self.workspace, steps)
        self._bind_task_progress(self._pipeline_progress_dialog, self._pipeline_worker)
        self._pipeline_worker.moveToThread(self._pipeline_thread)
        self._pipeline_thread.started.connect(self._pipeline_worker.run)
        self._pipeline_worker.progress.connect(self._on_pipeline_progress)
        self._pipeline_worker.finished.connect(self._on_pipeline_finished)
        self._pipeline_worker.failed.connect(self._on_pipeline_failed)
        self._pipeline_worker.finished.connect(self._pipeline_thread.quit)
        self._pipeline_worker.failed.connect(self._pipeline_thread.quit)
        self._pipeline_thread.finished.connect(self._pipeline_worker.deleteLater)
        self._pipeline_thread.finished.connect(self._cleanup_pipeline_task)
        self.statusBar().showMessage(f"Running {self._pipeline_task_label}...")
        self._pipeline_thread.start()
        return True

    def _on_pipeline_progress(self, payload):
        data = dict(payload or {})
        dialog = self._pipeline_progress_dialog
        if dialog is not None and dialog.isVisible():
            dialog.update_progress(data)
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
            StepId.BACKGROUND_CORRECTION: ("phase_wrap_mask", "phase_wrap_count"),
            StepId.REMOVE_NOISE: ("noise_region", "pcmra_volume"),
            StepId.GENERATE_PCMRA: ("pcmra_volume",),
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
        rebuild_prefixes = ["content_volume_"]
        for step in steps:
            rebuild_prefixes.extend(prefix_map.get(step, ()))
        if any(
            obj.actor is None and obj.data_key.startswith("segmask_group_")
            for obj in self.workspace.scene_objects.values()
        ):
            rebuild_prefixes.append("segmask_group_")
        rebuild_prefixes = tuple(dict.fromkeys(rebuild_prefixes))
        if steps & {StepId.GENERATE_SKELETON, StepId.GENERATE_GRAPH, StepId.GENERATE_PLANES}:
            self._clear_plane_drag_widgets()
            self._selected_plane_index = -1
            self._pathline_selected_plane_idx = None
            self.ortho_viewer.set_selected_plane(None)
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
        self._refresh_workflow_status()
        if StepId.REMOVE_NOISE in steps and self.workspace.pcmra_render_mask is not None:
            self.ortho_viewer.btn_noise_overlay.setChecked(True)
        self._refresh_render_range_control()
        self.ortho_viewer.refresh()
        if StepId.GENERATE_PCMRA in steps and self.workspace.pcmra_array is not None:
            self.ortho_viewer.combo_content.setCurrentIndex(self.ortho_viewer.combo_content.findData(4))
            self._on_content_render_requested(4)

    def _on_pipeline_finished(self, payload):
        if self._pipeline_progress_dialog is not None and self._pipeline_progress_dialog.cancel_token.cancelled:
            self._on_pipeline_failed("Cancelled")
            return
        self._last_quality_report = None
        data = dict(payload or {})
        try:
            completed_steps = [item[0] for item in data.get("results", []) if item]
            self._finish_pipeline_scene_refresh(completed_steps)
            elapsed = float(data.get("elapsed_sec", 0.0))
            self.log(f"[{self._pipeline_task_label}] Completed in {elapsed:.2f}s")
            report = data.get("analysis_report") or {}
            if report.get("status") == "complete":
                self.log(f"Analysis HTML saved: {report['file']} ({float(report.get('elapsed_sec', 0.0)):.2f}s)")
            elif report.get("status") == "failed":
                self.log(f"REPORT EXPORT ERROR: {report.get('error', 'Unknown error')}; completed measurements were retained.")
            self.statusBar().showMessage(f"{self._pipeline_task_label} completed in {elapsed:.2f}s", 8000)
        except Exception:
            self.log(f"VIEW REFRESH ERROR: {traceback.format_exc()}")

    def _on_pipeline_failed(self, error_text):
        self._run_all_pathlines_after_pipeline = False
        if str(error_text) == "Cancelled":
            steps = list(getattr(self._pipeline_worker, "completed_steps", []))
            if steps:
                self._finish_pipeline_scene_refresh(steps)
            self.log(f"{self._pipeline_task_label} cancelled; completed steps were retained.")
            self.statusBar().showMessage("Task cancelled", 8000)
            return
        correction_steps = {StepId.BACKGROUND_CORRECTION, StepId.REMOVE_NOISE, StepId.UNWRAP_PHASE, StepId.GENERATE_PCMRA}
        partial_steps = correction_steps.intersection(getattr(self, "_pipeline_task_steps", []))
        if partial_steps:
            self._finish_pipeline_scene_refresh(partial_steps)
        self.log(f"PIPELINE ERROR:\n{str(error_text).strip()}")
        self.statusBar().showMessage(f"{self._pipeline_task_label} failed", 10000)

    def _cleanup_pipeline_task(self):
        self._close_progress_dialog(self._pipeline_progress_dialog)
        self._pipeline_progress_dialog = None
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
            if self.workspace.streamline_params.seed_region == "segmentation":
                self.pipeline.preprocess(self.workspace)
            if self.scene.trigger_streamlines() is False:
                return
            self.workspace.pipeline.mark_done(StepId.GENERATE_STREAMLINES)
            self._refresh_browser()
            self._refresh_workflow_status()
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
            self.log("No data loaded. Use File > Load H5 or File > Load DICOM.")
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
            if step == StepId.UNWRAP_PHASE:
                self._prepare_unwrap_action()
            if step == StepId.COMPUTE_PWV:
                self.workspace.pwv_params.enabled = True
                self._show_measurements("pwv")
            self._clear_plane_drag_widgets()
            if step == StepId.EDIT_SKELETON:
                self._start_skeleton_interactive_edit()
                return
            if step == StepId.EDIT_GRAPH:
                self._start_graph_interactive_edit()
                return
            if step == StepId.GENERATE_STREAMLINES:
                if self.workspace.streamline_params.seed_region == "segmentation":
                    self.pipeline.preprocess(self.workspace)
                if self.scene.trigger_streamlines() is False:
                    return
                self.workspace.pipeline.mark_done(StepId.GENERATE_STREAMLINES)
                self._sync_params_to_ui()
                self._refresh_browser()
                self._refresh_workflow_status()
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
        self._finish_pipeline_scene_refresh([StepId.UNWRAP_PHASE])

    def _run_all_pipeline(self):
        if self._autoseg_running_guard("running the full pipeline"):
            return
        if self._pathline_running_guard("running the full pipeline"):
            return
        if self._pipeline_running_guard("running the full pipeline"):
            return
        if not self.workspace.data_loaded:
            self.log("No data loaded. Use File > Load H5 or File > Load DICOM.")
            return
        if self._edit_mode is not None:
            self.log("Finish current interactive edit first.")
            return
        self._sync_params_to_ws()
        self._clear_plane_drag_widgets()
        stage_key = self._workflow_stage_key()
        all_steps = _workflow_run_all_steps(stage_key)
        if StepId.UNWRAP_PHASE in all_steps:
            self._prepare_unwrap_action()
        if not all_steps:
            self.log("Run All is available in Correction, Centerline & Planes or Hemodynamics.")
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
        ws.graph_hierarchy = {}
        ws.remove_objects_by_prefix("hierarchy_path_")
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
        if self.ortho_viewer._roi_transaction is not None:
            self.ortho_viewer._cancel_plane_roi()
            return
        if self._plane_edit_session is not None:
            self._cancel_plane_edit()
            return
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
        if getattr(self, "_scene_export_running", False):
            progress = getattr(self, '_view_export_progress', None)
            if progress is not None:
                progress.cancel()
            event.ignore()
            return
        self.ortho_viewer._cancel_plane_roi()
        if not self._plane_jobs.close():
            self.log("Stopping plane calculation; close again after it stops.")
            event.ignore()
            return
        if self._plane_edit_session is not None:
            self._cancel_plane_edit()
        self._timeline_scrub_timer.stop()
        self._frame_navigation_filter.held.clear()
        if self._dicom_conversion_thread is not None:
            self.log("Wait for DICOM conversion to finish before closing the GUI.")
            event.ignore()
            return
        if self._labeler_export_running:
            self.log("Wait for the Labeler export to finish before closing AutoFlow.")
            event.ignore()
            return
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
        self.scene._clear_content_widgets()
        self.scene._close_interaction_rendering()
        if hasattr(self, "_qc_overlay"):
            self._qc_overlay.close()
        self.scene.cancel_pending_render()
        from ..algorithms.segmentation.shared_worker import close_shared_inference_worker
        close_shared_inference_worker()
        for dialog in list(getattr(self, '_html_document_dialogs', {}).values()):
            dialog.close()
        for pane in list(getattr(self, '_html_document_panes', {}).values()):
            pane.shutdown()
        # Finalize VTK while its Qt/Win32 window still exists. Letting Python
        # destroy it after the native window closes leaves an invalid WGL handle.
        self.plotter.close()
        event.accept()


def main(config_dir=None):
    from .html_view import configure_html_runtime
    configure_html_runtime()
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
