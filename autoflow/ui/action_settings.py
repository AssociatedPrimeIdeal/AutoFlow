"""Transactional action settings, with method-dependent form rows."""

import copy
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from ..algorithms.data.input_parameters import normalize_input_parameters
from ..algorithms.phase_unwrapping import backend_available, mask_sources_for_method, resolve_mask_source
from ..algorithms.segmentation import resolve_nnunet_model_folder, resolve_nnunet_4d_model_folder
from ..config import apply_config_bundle_to_workspace
from ..core.models import PwvParams, StepId, Workspace
from ..plane_rules import load_plane_rules_file, normalize_label_plane_rules, save_plane_rules_file


class _RatioSpinBox(QtWidgets.QDoubleSpinBox):
    def textFromValue(self, value):
        return f"{value:.6g}"

    def valueFromText(self, text):
        # Compact display must not round stored thirds when focus changes.
        if text.strip() == self.textFromValue(self.value()):
            return self.value()
        return super().valueFromText(text)


@dataclass
class Field:
    key: str
    label: str
    kind: str = "text"
    choices: tuple = ()
    minimum: float = 0
    maximum: float = 1e6
    visible: object = None
    default: object = None


def option(key, label, *choices, visible=None):
    return Field(key, label, "choice", tuple((x, x) if isinstance(x, str) else x for x in choices), visible=visible)


def number(key, label, minimum=0, maximum=1e6, integer=False, visible=None):
    return Field(key, label, "int" if integer else "float", minimum=minimum, maximum=maximum, visible=visible)


def flag(key, label, visible=None):
    return Field(key, label, "bool", visible=visible)


def _get(root, path, default=None):
    value = root
    for key in path.split("."):
        value = value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)
        if value is None:
            break
    return copy.deepcopy(value)


def _set(root, path, value):
    keys = path.split(".")
    for key in keys[:-1]:
        root = root.setdefault(key, {}) if isinstance(root, dict) else getattr(root, key)
    if isinstance(root, dict):
        root[keys[-1]] = copy.deepcopy(value)
    else:
        setattr(root, keys[-1], copy.deepcopy(value))


def action_sections(action, ws):
    """Return only GUI-exposed settings; existing config-only fields stay intact."""
    b = "loader_params.background_phase_correction."
    s = "skeleton_params."
    p = "plane_gen_params."
    f = "streamline_params."
    d = "derived_params."
    u = "phase_unwrap_params."
    w = "pwv_params."
    if action == "input":
        return [("Input", [Field("input.resolution", "Resolution (mm, LR/AP/FH)"),
                           Field("input.venc", "VENC (cm/s, LR/AP/FH)"),
                           Field("input.spatial_order", "Source spatial order"),
                           Field("input.venc_order", "Source velocity order")])]
    if action == "segmentation":
        return [("Segmentation", [
            option("segmentation.auto_backend", "Mode", ("3D", "nnUNet"), ("4D", "nnUNet4D")),
            Field("segmentation.auto_model", "Model", "model"),
            Field("segmentation.auto_checkpoint", "Checkpoint", "checkpoint"),
            Field("segmentation.auto_folds", "Folds"),
            option("segmentation.auto_device", "Device", ("Auto", "auto"), ("CPU", "cpu"), ("CUDA", "cuda")),
            Field("segmentation.auto_label_map", "Label map (JSON)", "json"),
        ])]
    if action == StepId.BACKGROUND_CORRECTION:
        dual_info = ws.input_state.metadata.get("dual_venc", {})
        dual = isinstance(dual_info, dict) and bool(dual_info.get("enabled", False))
        return [("Background correction", [
            option(b + "method", "Method", ("WRLS + ARTO", "wrls_arto"), ("MSAC", "msac")),
            number(b + "corr_fit_order", "Fit order", 0, 3, integer=True),
            number(b + "threshold", "MSAC threshold", .001, 10,
                   visible=lambda v: v[b + "method"] == "msac"),
            number(b + "dual_venc_ratio1", "Dual-VENC ratio 1", -10, 10, visible=lambda v: dual),
            number(b + "dual_venc_ratio2", "Dual-VENC ratio 2", -10, 10, visible=lambda v: dual),
        ])]
    if action == StepId.REMOVE_NOISE:
        n = "noise_removal_params."
        return [("Noise removal", [
            option(n + "method", "Method", ("Magnitude + temporal speed SD", "magnitude_temporal"),
                   ("Magnitude only", "magnitude")),
            number(n + "magnitude_fraction", "Magnitude fraction of maximum", 0, 1),
            number(n + "velocity_std_max", "Temporal SD fraction of maximum", 0, 1,
                   visible=lambda v: v[n + "method"] == "magnitude_temporal"),
        ])]
    if action == StepId.UNWRAP_PHASE:
        methods = [("Laplacian 4D", "lap4D"), ("Graph-cut 3D", "gc3D"), ("NPRS", "nprs")]
        methods += [(label, key) for label, key in [("PUDIP-Flow", "pudip"), ("GUST-Flow", "gust")]
                    if backend_available(key)]
        return [("Phase unwrapping", [
            option(u + "method", "Method", *methods),
            option(u + "mask_source", "Mask / initialization", ("None (whole volume)", "none")),
            option(u + "device", "Device", ("Auto", "auto"), ("CPU", "cpu"), ("CUDA", "cuda")),
            number(u + "lap4d_ts", "Temporal weight", .0001, 1e4, visible=lambda v: v[u + "method"] == "lap4D"),
            number(u + "nprs_upsampling_factor", "Upsampling factor", 1, 32, integer=True,
                   visible=lambda v: v[u + "method"] == "nprs"),
            flag(u + "nprs_pi_unwrap", "π unwrap", visible=lambda v: v[u + "method"] == "nprs"),
            flag(u + "nprs_auto_crop", "Auto crop", visible=lambda v: v[u + "method"] == "nprs"),
        ])]
    if action == StepId.GENERATE_SKELETON:
        filtering = lambda v: v[s + "remove_small_cc"]
        return [("Skeleton", [
            option(s + "special_handling", "Special handling", ("Three-pass merge", "three_pass_merge"),
                   ("Contact surface", "contact_surface")),
            flag(s + "remove_small_cc", "Remove small components"),
            option(s + "cc_filter_mode", "Component filter", "hybrid", "absolute", "relative", "largest", visible=filtering),
            number(s + "min_cc_volume_mm3", "Minimum volume (mm³)", visible=lambda v: filtering(v) and v[s + "cc_filter_mode"] in {"hybrid", "absolute"}),
            number(s + "cc_rel_min_ratio", "Relative minimum ratio", 0, 1,
                   visible=lambda v: filtering(v) and v[s + "cc_filter_mode"] in {"hybrid", "relative"}),
            flag(s + "do_closing", "Closing"), flag(s + "do_opening", "Opening"),
            flag(s + "gaussian_enabled", "Gaussian smoothing"),
            number(s + "gaussian_sigma", "Gaussian σ (vox)", visible=lambda v: v[s + "gaussian_enabled"]),
        ])]
    if action == StepId.GENERATE_GRAPH:
        return [("Graph", [option(s + "graph_connectivity", "Skeleton graph connectivity",
                                  ("6 — face neighbors", 6), ("26 — face, edge and corner neighbors", 26)),
                           flag(s + "junction_cluster_enabled", "Simplify compact junction clusters"),
                           number(s + "junction_cluster_max_extent_voxels", "Maximum cluster extent per axis (voxels)", 1, 10000),
                           number(s + "junction_cluster_min_cycle_rank", "Minimum cluster cycle rank", 1, 1000000, integer=True),
                           number(s + "cycle_flow_threshold", "Flow C threshold (delete paths below)", 0, 1),
                           flag(s + "cycle_preserve_connectivity", "Protect label connectivity and endpoints"),
                           number(s + "min_edge_points", "Minimum terminal edge count", 0, 100000, integer=True)])]
    if action == StepId.GENERATE_PLANES:
        fixed = lambda v: v[p + "plane_mode"] == "fixed_step"
        global_layout = lambda v: v[p + "plane_mode"] != "by_label"
        spacing = lambda v: v[p + "plane_mode"] in {"distance", "anchored_offset"} or (fixed(v) and v[p + "spacing_mode"] == "distance") or (v[p + "plane_mode"] in {"uniform", "count"} and v.get(p + "plane_count", ws.plane_gen_params.plane_count) == -1)
        return [("Planes", [
            option(p + "plane_mode", "Mode", ("By label (default)", "by_label"), ("Uniform", "uniform"), ("Fixed step", "fixed_step"),
                   ("From junction", "anchored_offset"), ("Evenly spaced (legacy)", "count"),
                   ("Fixed spacing (legacy)", "distance")),
            Field(p + "label_rules_file", "Plane config", "plane_file", visible=lambda v: not global_layout(v)),
            Field(p + "label_rules", "Per-label measurements", "plane_rules", visible=lambda v: not global_layout(v)),
            number(p + "plane_count", "Plane count (-1 = all)", -1, 100000, integer=True, visible=global_layout),
            option(p + "anchor", "Anchor", ("Start", "start"), ("Center", "center"), ("End", "end"), ("Junction", "junction"), visible=fixed),
            option(p + "direction", "Direction", ("Toward start", "toward_start"), ("Toward end", "toward_end"), ("Both sides", "both"), visible=fixed),
            option(p + "spacing_mode", "Spacing", ("Fraction of path", "fraction"), ("Distance (mm)", "distance"), visible=fixed),
            number(p + "spacing_ratio", "Spacing ratio", .0001, 1, visible=lambda v: fixed(v) and v[p + "spacing_mode"] == "fraction"),
            number(p + "cross_section_distance", "Spacing (mm)", .0001, visible=spacing),
            number(p + "anchor_offset_mm", "Junction offset (mm)", visible=lambda v: v[p + "plane_mode"] == "anchored_offset"),
            number(p + "start_distance", "Start distance (mm)", visible=global_layout),
            number(p + "end_distance", "End distance (mm)", visible=global_layout),
            flag(p + "segmentation_filter", "Segmentation filter", visible=global_layout),
            flag(p + "fov_boundary_filter", "Exclude planes at FOV boundary"),
            number(p + "fov_boundary_max_fraction", "Maximum FOV contact fraction", 0, 1,
                   visible=lambda v: v[p + "fov_boundary_filter"]),
            number(p + "smoothing_window", "Path smoothing window", 1, 100001, integer=True),
            number(p + "smoothing_polyorder", "Path smoothing order", 0, 100, integer=True),
            number(p + "inter_time", "Path interpolation factor", 1, 1000, integer=True),
        ])]
    if action == StepId.COMPUTE_PLANE_METRICS:
        return [("Plane metrics", [flag(d + "use_multithread", "Parallel plane metrics")])]
    if action == StepId.COMPUTE_DERIVED_METRICS:
        return [
            ("Fluid", [number(d + "viscosity", "Viscosity (mPa·s)"), number(d + "rho", "Density (kg/m³)", 1)]),
            ("WSS", [number(d + "smoothing_iteration", "Surface smoothing iterations", 0, 100000, integer=True),
                     Field(d + "inward_distance", "Inward distance (mm or auto)", "optional_positive"),
                     flag(d + "parabolic_fitting", "Parabolic fitting"), flag(d + "no_slip_condition", "No-slip condition")]),
            ("Pressure", [option(d + "pressure_method", "Relative pressure method", ("PPE", "ppe"), ("STE", "ste")),
                          number(d + "pressure_gradient_smoothing_sigma", "Gaussian σ (vox)"),
                          number(d + "pressure_gradient_support_erosion_iters", "Support erosion (vox)", 0, 100, integer=True)]),
            ("Vortex", [number(d + "vortex_smoothing_sigma", "Gaussian σ (vox)"),
                        number(d + "vortex_support_erosion_iters", "Support erosion (vox)", 0, 100, integer=True)]),
        ]
    if action == StepId.GENERATE_STREAMLINES:
        return [("Streamlines", [number(f + "seed_ratio", "Seed ratio", .0001, 1),
                                number(f + "max_steps", "Maximum steps", 1, 200000, integer=True),
                                number(f + "terminal_speed", "Terminal speed (m/s)")])]
    if action == StepId.PLANE_STREAMLINES:
        ratio = lambda v: v[f + "pathline_seed_mode"] == "ratio"
        return [("Integration", [
            option(f + "pathline_seed_mode", "Seed mode", ("Fixed count", "fixed"), ("Ratio", "ratio")),
            number(f + "pathline_max_seeds", "Seed count / limit", 1, 100000, integer=True),
            number(f + "pathline_seed_ratio", "Seed ratio", .0001, 1, visible=ratio),
            number(f + "pathline_min_seeds", "Minimum seeds", 1, 100000, integer=True, visible=ratio),
            number(f + "pathline_max_steps", "Maximum steps", 1, 200000, integer=True),
            number(f + "pathline_terminal_speed", "Terminal speed (m/s)"),
            number(f + "pathline_rng_seed", "Random seed", -2147483647, 2147483647, integer=True),
            number(f + "pathline_temporal_cache_mb", "Temporal cache (MiB)", 0, 8192),
        ]), ("Display", [number(f + "pathline_tube_radius", "Tube radius (mm)", .001, 1e4),
                         option(f + "pathline_color_mode", "Color mode", ("Per plane", "per_plane"), ("Per group", "per_group"), ("Uniform", "uniform")),
                         Field(f + "pathline_color", "Uniform color", "color", visible=lambda v: v[f + "pathline_color_mode"] == "uniform")])]
    if action == StepId.COMPUTE_PWV:
        foot = lambda v: v[w + "transit_time_method"] == "foot_to_foot"
        tangent = lambda v: foot(v) and v[w + "foot_method"] == "tangent"
        xcorr = lambda v: v[w + "transit_time_method"] == "cross_correlation"
        return [
            ("Groups", [Field(w + "groups", "PWV groups", "groups")]),
            ("Path / planes", [number(w + "plane_interval_mm", "Plane interval (mm)", .1),
                               number(w + "start_distance", "Start distance (mm)"), number(w + "end_distance", "End distance (mm)"),
                               number(w + "smoothing_window", "Path smoothing window", 1, 100001, integer=True),
                               number(w + "smoothing_polyorder", "Path smoothing order", 0, 100, integer=True),
                               number(w + "inter_time", "Path interpolation factor", 1, 1000, integer=True),
                               number(w + "minimum_valid_planes", "Minimum valid planes", 2, 100000, integer=True)]),
            ("Transit time", [Field(w + "waveform_key", "Waveform"),
                              option(w + "transit_time_method", "Method", ("Foot-to-foot", "foot_to_foot"), ("Cross-correlation", "cross_correlation")),
                              option(w + "foot_method", "Foot method", ("Tangent", "tangent"), ("Threshold", "threshold"), visible=foot),
                              number(w + "foot_savgol_window", "Foot smoothing window", 1, 100001, integer=True, visible=foot),
                              number(w + "foot_savgol_polyorder", "Foot smoothing order", 0, 100, integer=True, visible=foot),
                              number(w + "foot_threshold_percent", "Foot threshold (%)", 0, 100, visible=lambda v: foot(v) and not tangent(v)),
                              option(w + "xcorr_window", "Correlation window", ("Full cycle", "full"), ("Upstroke", "upstroke"), visible=xcorr),
                              number(w + "xcorr_interp_factor", "Correlation interpolation", 1, 1000, integer=True, visible=xcorr),
                              flag(w + "allow_cycle_wrap", "Allow cycle wrap")]),
            ("Display / output", [flag(w + "scene_visible", "Show PWV planes"),
                                  Field(w + "scene_color", "Plane color", "color", visible=lambda v: v[w + "scene_visible"]),
                                  Field(w + "plot_color", "Waveform color", "color"), Field(w + "fit_color", "Fit color", "color"),
                                  number(w + "plot_dpi", "Plot DPI", 72, 1200, integer=True)]),
        ]
    if action == "render":
        fields = [Field("render_settings." + key + "_clim", label + " range (auto or min, max)", "range")
                  for key, label in [("wss", "WSS"), ("tke", "TKE"), ("pressure_gradient", "Pressure gradient"),
                                     ("relative_pressure", "Relative pressure"), ("streamline", "Streamlines")]]
        fields += [number(d + "pressure_gradient_layer_opacity", "Pressure gradient opacity", 0, 1),
                   number(d + "relative_pressure_layer_opacity", "Relative pressure opacity", 0, 1)]
        bars = []
        for key, label, value, lo, hi in [("width", "Width", .08, .01, 1), ("height", "Height", .65, .05, 1),
                                         ("stack_gap", "Gap", .03, 0, 1), ("position_x", "X", .87, 0, .98),
                                         ("position_y", "Y", .15, 0, .95), ("title_font_size", "Title font", 14, 1, 100),
                                         ("label_font_size", "Label font", 11, 1, 100)]:
            field = number("render_settings.shared_colorbar_bar_cfg." + key, label, lo, hi, integer="font" in key)
            field.default = value
            bars.append(field)
        return [("3D display", fields), ("Colorbar", bars)]
    return []


def input_values(ws):
    saved = ws.input_state.metadata.get("gui_input_parameters")
    return normalize_input_parameters(saved if saved is not None else {
        "resolution": ws.resolution, "venc": ws.venc,
        "spatial_order": ws.spatial_order, "venc_order": ws.venc_order,
    })


class ActionSettingsDialog(QtWidgets.QDialog):
    """Own draft values only. Callers publish values after Accepted."""

    def __init__(self, title, sections, workspace, defaults, parent=None, *, run_label="Save and run", auxiliary=()):
        super().__init__(parent)
        self.setWindowTitle(title + " settings")
        self.setMinimumWidth(480)
        self.resize(620, 520)
        self.workspace = workspace
        self.fields = [field for _, fields in sections for field in fields]
        if any(field.kind == "plane_rules" for field in self.fields):
            self.resize(1000, 620)
        self.controls = {}
        self.rows = {}
        self.forms = {}
        self._pages = []
        self._auto_height = None
        self.run_requested = False
        self.operation = None
        self._initial = self._read_workspace(workspace)
        self._defaults = self._read_workspace(defaults)
        self._mask_memory = copy.deepcopy(workspace.input_state.metadata.get("gui_phase_unwrap_masks", {}))
        self._mask_method = None
        self._model_backend = None
        self._model_presets = set()
        layout = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)
        nested_tabs = {}
        for title, fields in sections:
            self._pages.append(fields)
            target_tabs = self.tabs
            if "::" in title:
                group, title = title.split("::", 1)
                if group not in nested_tabs:
                    nested_tabs[group] = QtWidgets.QTabWidget()
                    self.tabs.addTab(nested_tabs[group], group)
                target_tabs = nested_tabs[group]
            page = QtWidgets.QWidget()
            form = QtWidgets.QFormLayout(page)
            form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
            form.setLabelAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            for field in fields:
                control, widget = self._make_control(field)
                control.setObjectName(field.key)
                form.addRow(field.label, widget)
                self.controls[field.key] = control
                self.rows[field.key] = widget
                self.forms[field.key] = form
            if any(field.key == "derived_params.rho" for field in fields):
                available = workspace.source_sigma is not None or workspace.source_tke_array is not None
                notice = QtWidgets.QLabel("TKE: source data available" if available else "TKE: unavailable; skipped when source data are missing")
                notice.setWordWrap(True)
                form.addRow(notice)
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
            scroll.setWidget(page)
            target_tabs.addTab(scroll, title)
        self.tabs.tabBar().setVisible(self.tabs.count() > 1)
        self.error = QtWidgets.QLabel()
        self.error.setWordWrap(True)
        self.error.setStyleSheet("color: #a12622")
        layout.addWidget(self.error)
        if auxiliary:
            row = QtWidgets.QHBoxLayout()
            for label, operation in auxiliary:
                button = QtWidgets.QPushButton(label)
                button.clicked.connect(lambda checked=False, op=operation: self._choose_operation(op))
                row.addWidget(button)
            layout.addLayout(row)
        buttons = QtWidgets.QDialogButtonBox()
        self.defaults_button = buttons.addButton("Restore defaults", QtWidgets.QDialogButtonBox.ResetRole)
        self.defaults_button.clicked.connect(lambda: self.set_values(self._defaults))
        buttons.addButton(QtWidgets.QDialogButtonBox.Cancel).clicked.connect(self.reject)
        save = buttons.addButton("Save", QtWidgets.QDialogButtonBox.AcceptRole)
        save.setDefault(True)
        save.clicked.connect(self.accept)
        if run_label:
            self.run_button = buttons.addButton(run_label, QtWidgets.QDialogButtonBox.ActionRole)
            self.run_button.setEnabled(bool(workspace.data_loaded))
            self.run_button.clicked.connect(self._save_and_run)
        layout.addWidget(buttons)
        self.set_values(self._initial)
        for control in self.controls.values():
            if isinstance(control, QtWidgets.QComboBox):
                control.currentIndexChanged.connect(self._update_visibility)
            elif isinstance(control, QtWidgets.QCheckBox):
                control.toggled.connect(self._update_visibility)
            elif isinstance(control, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
                control.valueChanged.connect(self._update_visibility)

    def _read_workspace(self, ws):
        values = {}
        for field in self.fields:
            if field.key.startswith("input."):
                value = input_values(ws)[field.key.split(".")[1]]
            elif field.kind == "groups":
                value = [{"name": group.name, "labels": list(group.labels)} for group in ws.pwv_params.groups]
            else:
                value = _get(ws, field.key, field.default)
                layer = {"derived_params.pressure_gradient_layer_opacity": "pressure_gradient_volume",
                         "derived_params.relative_pressure_layer_opacity": "relative_pressure_volume"}.get(field.key)
                if layer:
                    value = next((obj.opacity for obj in ws.scene_objects.values() if obj.data_key == layer), value)
            values[field.key] = value
        return values

    def _make_control(self, field):
        if field.kind in {"choice", "model"}:
            control = QtWidgets.QComboBox()
            control.setEditable(field.kind == "model")
            for label, value in field.choices:
                control.addItem(label, value)
        elif field.kind == "bool":
            control = QtWidgets.QCheckBox()
        elif field.kind in {"int", "float"}:
            control = QtWidgets.QSpinBox() if field.kind == "int" else QtWidgets.QDoubleSpinBox()
            if field.kind == "float":
                control.setDecimals(4)
                control.setSingleStep(.01)
            control.setRange(field.minimum, field.maximum)
        elif field.kind == "json":
            control = QtWidgets.QPlainTextEdit()
            control.setMaximumHeight(90)
            control.setPlaceholderText('{"aorta": 1}')
        elif field.kind == "plane_file":
            control = QtWidgets.QLineEdit()
            control.setReadOnly(True)
            widget = QtWidgets.QWidget()
            layout = QtWidgets.QHBoxLayout(widget)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(control, 1)
            from ..config import default_config_dir
            presets = QtWidgets.QComboBox()
            presets.setObjectName("plane_plan_preset")
            presets.addItem("Choose plane plan…", None)
            for title, filename in (("Combined default", "default"), ("Cardiopulmonary", "cardiopulmonary"),
                ("Portal", "portal"), ("Renal", "renal"), ("Brain", "brain"), ("Neck (ICA / VA / IJV)", "neck")):
                presets.addItem(title, str(default_config_dir() / "plane_plans" / (filename + ".json")))
            presets.activated.connect(lambda index: self._load_plane_preset(presets.itemData(index)))
            self._plane_preset_combo = presets
            layout.addWidget(presets)
            load = QtWidgets.QPushButton("Load…")
            save = QtWidgets.QPushButton("Save…")
            load.clicked.connect(self._load_plane_config)
            save.clicked.connect(self._save_plane_config)
            layout.addWidget(load)
            layout.addWidget(save)
            return control, widget
        elif field.kind == "plane_rules":
            control = QtWidgets.QTableWidget(0, 7)
            control.setHorizontalHeaderLabels(["Label", "Planes", "Legacy spacing ratio", "Legacy main path", "Layout", "Edit plan", "Connections, paths and positions"])
            control.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
            control.horizontalHeader().setSectionResizeMode(6, QtWidgets.QHeaderView.Stretch)
            control.itemDoubleClicked.connect(lambda item: self._edit_label_rule(control, item.row()) if item.column() == 6 else None)
            control.setMinimumHeight(300)
            control.setToolTip("Landmark plans use explicit anatomical junctions or path levels. Missing or ambiguous anchors are omitted, with no flow-root fallback. Zero disables a label.")
        elif field.kind == "groups":
            control = QtWidgets.QTableWidget(0, 2)
            control.setHorizontalHeaderLabels(["Name", "Labels (ids or names)"])
            control.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
            control.setMinimumHeight(180)
            widget = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(widget)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(control)
            row = QtWidgets.QHBoxLayout()
            add = QtWidgets.QPushButton("Add group")
            remove = QtWidgets.QPushButton("Remove selected")
            add.clicked.connect(lambda: control.insertRow(control.rowCount()))
            remove.clicked.connect(lambda: self._remove_groups(control))
            row.addWidget(add)
            row.addWidget(remove)
            layout.addLayout(row)
            return control, widget
        else:
            control = QtWidgets.QLineEdit()
        if field.kind in {"model", "checkpoint"}:
            widget = QtWidgets.QWidget()
            row = QtWidgets.QHBoxLayout(widget)
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(control, 1)
            browse = QtWidgets.QPushButton("Browse…")
            browse.clicked.connect(lambda: self._browse(field))
            row.addWidget(browse)
            return control, widget
        return control, control

    def _load_plane_config(self, *_args):
        field = self.controls["plane_gen_params.label_rules_file"]
        source, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load plane config", field.text(), "JSON (*.json)")
        if not source:
            return
        self._load_plane_preset(source)

    def _load_plane_preset(self, source):
        if not source:
            return
        field = self.controls["plane_gen_params.label_rules_file"]
        try:
            rules = load_plane_rules_file(source)
            self._populate_plane_rules(self.controls["plane_gen_params.label_rules"], rules)
            field.setText(str(Path(source).resolve()))
            preset = getattr(self, "_plane_preset_combo", None)
            if preset is not None:
                index = next((i for i in range(1, preset.count())
                              if Path(preset.itemData(i)).resolve() == Path(source).resolve()), 0)
                preset.setCurrentIndex(index)
            self.error.clear()
            self._update_visibility()
        except (OSError, ValueError, TypeError) as exc:
            self.error.setText(str(exc))

    def _save_plane_config(self, *_args):
        field = self.controls["plane_gen_params.label_rules_file"]
        destination, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save shared plane config", field.text(), "JSON (*.json)")
        if not destination:
            return
        try:
            rule_field = next(f for f in self.fields if f.kind == "plane_rules")
            saved = save_plane_rules_file(destination, self._raw_value(rule_field))
            field.setText(saved)
            self.error.clear()
        except (OSError, ValueError, TypeError) as exc:
            self.error.setText(str(exc))

    def _populate_plane_rules(self, table, rules):
        from ..plane_rules import default_label_placements
        rules = dict(rules)
        for name, value in self.workspace.skeleton_params.label_map.items():
            if int(value) > 0 and name not in rules and str(value) not in rules:
                placements = default_label_placements(name)
                rules[name] = ({"plane_count": 0, "plane_mode": "landmark", "placements": placements}
                               if placements else {"plane_count": 0})
        table.setRowCount(len(rules))
        for row, (name, raw_rule) in enumerate(rules.items()):
            rule = normalize_label_plane_rules({name: raw_rule})[name]
            item = QtWidgets.QTableWidgetItem(name)
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
            item.setData(QtCore.Qt.UserRole, rule)
            table.setItem(row, 0, item)
            count = QtWidgets.QSpinBox()
            count.setRange(-1, 100000)
            count.setToolTip("0 disables this label; -1 places all planes that fit the selected mode.")
            count.setValue(rule["plane_count"])
            table.setCellWidget(row, 1, count)
            ratio = _RatioSpinBox()
            ratio.setDecimals(12)
            ratio.setSingleStep(.01)
            ratio.setRange(.000000000001, 1)
            ratio.setValue(rule["spacing_ratio"])
            count.valueChanged.connect(lambda n, widget=ratio: widget.setValue(1 / 3 if n == 2 else .3))
            table.setCellWidget(row, 2, ratio)
            combo = QtWidgets.QComboBox()
            combo.addItem("Not used (landmark plan)", None)
            combo.addItem("Inflow roots (all entries)", "inflow")
            combo.addItem("Outflow roots (all exits)", "outflow")
            combo.setCurrentIndex(combo.findData(rule.get("main_path")))
            table.setCellWidget(row, 3, combo)
            mode = QtWidgets.QComboBox()
            for title, value in (("Connections / path plan", "landmark"), ("Fixed step", "fixed_step"), ("Uniform", "uniform"),
                                 ("From junction", "anchored_offset"), ("Evenly spaced (legacy)", "count"),
                                 ("Fixed spacing (legacy)", "distance")):
                mode.addItem(title, value)
            mode.setCurrentIndex(mode.findData(rule["plane_mode"]))
            def refresh_mode(_index=0, mode=mode, combo=combo, ratio=ratio, count=count):
                landmark = mode.currentData() == "landmark"
                combo.setEnabled(not landmark)
                ratio.setEnabled(not landmark)
                if landmark:
                    combo.setCurrentIndex(0)
                    count.setToolTip("0 disables this plan. A positive count describes positions per resolved connection/path; an explicit fallback may use a different count. Use Edit to change positions.")
                elif combo.currentData() is None:
                    combo.setCurrentIndex(1)
            mode.currentIndexChanged.connect(refresh_mode)
            refresh_mode()
            table.setCellWidget(row, 4, mode)
            mode.setMinimumContentsLength(18)
            parameters = QtWidgets.QPushButton("Edit…")
            parameters.clicked.connect(lambda checked=False, target=row: self._edit_label_rule(table, target))
            table.setCellWidget(row, 5, parameters)
            summary = QtWidgets.QTableWidgetItem()
            summary.setFlags(summary.flags() & ~QtCore.Qt.ItemIsEditable)
            table.setItem(row, 6, summary)
            mode.currentIndexChanged.connect(lambda *_args, target=table: self._refresh_plane_plan_table(target))
            count.valueChanged.connect(lambda *_args, target=table: self._refresh_plane_plan_table(target))
        self._refresh_plane_plan_table(table)

    def _refresh_plane_plan_table(self, table):
        from .landmark_editor import placement_summary
        legacy = False
        for row in range(table.rowCount()):
            if table.cellWidget(row, 4) is None or table.item(row, 6) is None:
                continue
            mode = table.cellWidget(row, 4).currentData()
            legacy |= mode != "landmark"
            rule = dict(table.item(row, 0).data(QtCore.Qt.UserRole) or {})
            rule.update(plane_mode=mode, plane_count=table.cellWidget(row, 1).value())
            summary = placement_summary(rule)
            table.item(row, 6).setText(summary)
            table.item(row, 6).setToolTip(summary + "\nDouble-click or use Edit plan to change anchors and positions.")
        table.setColumnHidden(2, not legacy)
        table.setColumnHidden(3, not legacy)

    def _edit_label_rule(self, table, row):
        item = table.item(row, 0)
        rule = dict(item.data(QtCore.Qt.UserRole))
        rule.update(plane_count=table.cellWidget(row, 1).value(), spacing_ratio=table.cellWidget(row, 2).value(),
                    main_path=table.cellWidget(row, 3).currentData(), plane_mode=table.cellWidget(row, 4).currentData())
        if rule["plane_mode"] == "landmark":
            self._edit_landmark_rule(table, row, rule)
            return
        draft = copy.copy(self.workspace)
        draft.plane_gen_params = copy.copy(self.workspace.plane_gen_params)
        global_values = {field.key.split(".", 1)[1]: self._raw_value(field) for field in self.fields
                         if field.key.startswith("plane_gen_params.") and field.kind not in {"plane_rules", "plane_file"}}
        for key, value in rule.items():
            if hasattr(draft.plane_gen_params, key):
                setattr(draft.plane_gen_params, key, global_values.get(key, getattr(draft.plane_gen_params, key))
                        if value is None else value)
        fields = [copy.copy(field) for _, group in action_sections(StepId.GENERATE_PLANES, draft) for field in group
                  if field.kind not in {"plane_rules", "plane_file"} and not field.key.endswith(".plane_count")]
        fields[0].choices = tuple(pair for pair in fields[0].choices if pair[1] != "by_label")
        defaults = copy.copy(draft)
        defaults.plane_gen_params = copy.copy(draft.plane_gen_params)
        default_rule = normalize_label_plane_rules({item.text(): {"plane_count": rule["plane_count"]}})[item.text()]
        for key, value in default_rule.items():
            if hasattr(defaults.plane_gen_params, key):
                setattr(defaults.plane_gen_params, key, global_values.get(key, getattr(self.workspace.plane_gen_params, key))
                        if value is None else value)
        dialog = ActionSettingsDialog(item.text() + " planes", [("Parameters", fields)], draft, defaults,
                                      self, run_label="")
        try:
            if dialog.exec() != QtWidgets.QDialog.Accepted:
                return
            for key, value in dialog.values().items():
                key = key.split(".", 1)[1]
                rule[key] = None if default_rule.get(key, "explicit") is None and value == global_values.get(key) else value
            rule = normalize_label_plane_rules({item.text(): rule})[item.text()]
            item.setData(QtCore.Qt.UserRole, rule)
            table.cellWidget(row, 2).setValue(rule["spacing_ratio"])
            table.cellWidget(row, 4).setCurrentIndex(table.cellWidget(row, 4).findData(rule["plane_mode"]))
        finally:
            dialog.deleteLater()

    def _edit_landmark_rule(self, table, row, rule):
        from .landmark_editor import LandmarkPlanDialog
        item = table.item(row, 0)
        rule.pop("main_path", None)
        if not rule.get("placements"):
            from ..plane_rules import default_label_placements
            rule["placements"] = default_label_placements(item.text())
        dialog = LandmarkPlanDialog(item.text(), rule, self)
        try:
            if dialog.exec() == QtWidgets.QDialog.Accepted:
                normalized = dialog.rule
                item.setData(QtCore.Qt.UserRole, normalized)
                table.cellWidget(row, 1).setValue(normalized["plane_count"])
                self._refresh_plane_plan_table(table)
        finally:
            dialog.deleteLater()

    def _remove_groups(self, table):
        rows = {index.row() for index in table.selectedIndexes()}
        for row in sorted(rows or {table.currentRow()}, reverse=True):
            if row >= 0:
                table.removeRow(row)

    def _browse(self, field):
        if field.kind == "model":
            value = QtWidgets.QFileDialog.getExistingDirectory(self, "Select nnUNet model")
            if value:
                self.controls[field.key].setEditText(value)
        else:
            value, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select checkpoint", "", "Checkpoint (*.pth)")
            if value:
                path = Path(value)
                self.controls[field.key].setText(path.name)
                if path.parent.name.startswith("fold_"):
                    self.controls["segmentation.auto_model"].setEditText(str(path.parent.parent))

    def set_values(self, values):
        for field in self.fields:
            control = self.controls[field.key]
            value = values[field.key]
            blocker = QtCore.QSignalBlocker(control)
            if field.kind == "choice":
                if field.key == "phase_unwrap_params.method" and value not in {control.itemData(i) for i in range(control.count())}:
                    value = "lap4D"
                index = control.findData(value)
                control.setCurrentIndex(max(index, 0))
            elif field.kind == "model":
                control.setEditText(str(value or "auto"))
            elif field.kind == "bool":
                control.setChecked(bool(value))
            elif field.kind in {"int", "float"}:
                control.setValue(value)
            elif field.kind == "json":
                control.setPlainText(str(value or ""))
            elif field.kind == "plane_rules":
                self._populate_plane_rules(control, value)
            elif field.kind == "groups":
                control.setRowCount(len(value))
                reverse = {int(v): k for k, v in self.workspace.label_params.label_map.items()}
                for row, group in enumerate(value):
                    control.setItem(row, 0, QtWidgets.QTableWidgetItem(group["name"]))
                    control.setItem(row, 1, QtWidgets.QTableWidgetItem(", ".join(str(reverse.get(x, x)) for x in group["labels"])))
            else:
                if isinstance(value, (list, tuple)):
                    value = ", ".join(f"{x:g}" if isinstance(x, (int, float)) else str(x) for x in value)
                control.setText("auto" if value is None else str(value))
            del blocker
        self._mask_method = None
        if "phase_unwrap_params.mask_source" in values:
            method = self.controls["phase_unwrap_params.method"].currentData()
            self._mask_memory[method] = values["phase_unwrap_params.mask_source"]
        self._model_backend = None
        self._update_visibility()

    def _raw_value(self, field):
        control = self.controls[field.key]
        if field.kind == "choice":
            return control.currentData()
        if field.kind == "model":
            # An edited path must not inherit the previously selected item's data.
            index = control.currentIndex()
            return str(control.itemData(index) if index >= 0 and control.currentText() == control.itemText(index) else control.currentText()).strip() or "auto"
        if field.kind == "bool":
            return control.isChecked()
        if field.kind in {"int", "float"}:
            return control.value()
        if field.kind == "json":
            return control.toPlainText().strip()
        if field.kind == "plane_rules":
            rules = {}
            for row in range(control.rowCount()):
                item = control.item(row, 0)
                rule = dict(item.data(QtCore.Qt.UserRole) or {})
                rule["plane_count"] = control.cellWidget(row, 1).value()
                rule["spacing_ratio"] = control.cellWidget(row, 2).value()
                rule["main_path"] = control.cellWidget(row, 3).currentData()
                rule["plane_mode"] = control.cellWidget(row, 4).currentData()
                rules[item.text()] = rule
            return rules
        if field.kind == "groups":
            groups = []
            for row in range(control.rowCount()):
                name = control.item(row, 0)
                labels = control.item(row, 1)
                groups.append({"name": name.text().strip() if name else "",
                               "labels": [x.strip() for x in re.split(r"[,;+\s]+", labels.text() if labels else "") if x.strip()]})
            return groups
        return control.text().strip()

    def _update_visibility(self, *_args):
        self._update_masks()
        self._update_models()
        values = {field.key: self._raw_value(field) for field in self.fields}
        for field in self.fields:
            visible = field.visible is None or field.visible(values)
            form = self.forms[field.key]
            widget = self.rows[field.key]
            widget.setVisible(visible)
            form.labelForField(widget).setVisible(visible)
        if self._auto_height is None or self.height() == self._auto_height:
            height = max(sum(320 if field.kind == "plane_rules" else 190 if field.kind == "groups" else 90 if field.kind == "json" else 32
                             for field in fields if not self.rows[field.key].isHidden()) for fields in self._pages)
            height = min(650, max(220, height + 125 + (32 if len(self._pages) > 1 else 0)))
            self.resize(self.width(), height)
            self._auto_height = self.height()

    def _update_masks(self):
        method_control = self.controls.get("phase_unwrap_params.method")
        if method_control is None:
            return
        method = method_control.currentData()
        combo = self.controls["phase_unwrap_params.mask_source"]
        if method == self._mask_method:
            return
        if self._mask_method is not None:
            self._mask_memory[self._mask_method] = combo.currentData()
        sources = mask_sources_for_method(method)
        selected = self._mask_memory.get(method, "auto")
        try:
            selected = resolve_mask_source(method, selected)
        except ValueError:
            selected = sources[0]
        if selected == "segmask" and self.workspace.segmask_raw is None:
            selected = sources[0]
        blocker = QtCore.QSignalBlocker(combo)
        combo.clear()
        for source in sources:
            combo.addItem({"none": "None (whole volume)", "segmask": "Segmentation", "pcmra_std": "PC-MRA SD", "pcmra_mean": "PC-MRA mean"}[source], source)
            if source == "segmask":
                item = combo.model().item(combo.count() - 1)
                item.setEnabled(self.workspace.segmask_raw is not None)
                item.setToolTip("Generate or import a segmentation first" if self.workspace.segmask_raw is None else "Active segmentation")
        combo.setCurrentIndex(max(0, combo.findData(selected)))
        del blocker
        self._mask_method = method

    def _update_models(self):
        mode = self.controls.get("segmentation.auto_backend")
        if mode is None or mode.currentData() == self._model_backend:
            return
        combo = self.controls["segmentation.auto_model"]
        value = self._raw_value(next(f for f in self.fields if f.key == "segmentation.auto_model"))
        if self._model_backend is not None and value in self._model_presets:
            value = "auto"
        resolver = resolve_nnunet_4d_model_folder if mode.currentData() == "nnUNet4D" else resolve_nnunet_model_folder
        try:
            resolved = str(resolver("auto"))
        except Exception:
            resolved = ""
        blocker = QtCore.QSignalBlocker(combo)
        combo.clear()
        combo.addItem("Automatic model", "auto")
        self._model_presets = {"auto"}
        if resolved and Path(resolved).is_dir():
            combo.addItem("Detected local model", resolved)
            self._model_presets.add(resolved)
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)
        else:
            combo.setEditText(value)
        del blocker
        self._model_backend = mode.currentData()

    def values(self):
        values = dict(self._initial)
        for field in self.fields:
            if field.visible is not None and self.rows[field.key].isHidden():
                continue
            value = self._raw_value(field)
            if field.kind == "optional_positive":
                value = None if value.lower() in {"", "auto", "none"} else float(value)
                if value is not None and (not math.isfinite(value) or value <= 0):
                    raise ValueError("Inward distance must be positive or auto.")
            elif field.kind == "range":
                if value.lower() in {"", "auto", "none"}:
                    value = None
                else:
                    value = [float(x.strip()) for x in value.split(",")]
                    if len(value) != 2 or not all(math.isfinite(x) for x in value) or value[0] >= value[1]:
                        raise ValueError(field.label + ": enter min, max with min < max.")
            elif field.kind == "color" and not (QtGui.QColor(value).isValid() or value in {"deepskyblue"}):
                # Matplotlib also accepts names Qt does not know.
                from matplotlib.colors import is_color_like
                if not is_color_like(value):
                    raise ValueError(field.label + ": invalid color.")
            elif field.kind == "plane_rules":
                value = normalize_label_plane_rules(value)
            elif field.kind == "json" and value:
                parsed = json.loads(value)
                if not isinstance(parsed, dict) or any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in parsed.values()):
                    raise ValueError("Label map must be a JSON object of names to nonnegative integer ids.")
            values[field.key] = value
        if "input.resolution" in values:
            normalize_input_parameters({key.split(".")[1]: value for key, value in values.items()})
        if values.get("plane_gen_params.plane_count") == 0:
            raise ValueError("Plane count must be positive, or -1 for all.")
        if "pwv_params.groups" in values:
            groups = values["pwv_params.groups"]
            names = [group["name"] for group in groups]
            if any(not group["name"] or not group["labels"] for group in groups) or len(set(names)) != len(names):
                raise ValueError("Each PWV group needs a unique name and at least one label.")
            parsed = PwvParams.from_dict({key.split(".")[1]: value for key, value in values.items() if key.startswith("pwv_params.")}, label_map=self.workspace.label_params.label_map)
            values["pwv_params.groups"] = [{"name": group.name, "labels": list(group.labels)} for group in parsed.groups]
        return values

    def accept(self):
        try:
            self.values()
        except (ValueError, TypeError) as exc:
            self.error.setText(str(exc))
            return
        super().accept()

    def _save_and_run(self):
        self.run_requested = True
        self.accept()
        if self.result() != QtWidgets.QDialog.Accepted:
            self.run_requested = False

    def _choose_operation(self, operation):
        self.operation = operation
        # Reset/revert act on existing results and discard parameter drafts.
        super().accept()


class ActionSettingsMixin:
    def _prepare_unwrap_action(self):
        params = self.workspace.phase_unwrap_params
        if str(params.method or "").lower() in {"", "none", "disabled"}:
            params.method = "lap4D"
        params.enabled = True

    def _attach_action_settings(self, button, action, run=None):
        button.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        button.customContextMenuRequested.connect(lambda position: self._open_action_settings(action, run))
        button._settings_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Shift+F10"), button)
        button._settings_shortcut.setContext(QtCore.Qt.WidgetShortcut)
        button._settings_shortcut.activated.connect(lambda: self._open_action_settings(action, run))
        button.setToolTip(button.toolTip() + "\nLeft-click to run; right-click or Shift+F10 for settings.")

    def _settings_sections(self, action):
        if isinstance(action, tuple):
            sections = []
            for key in action:
                name = {StepId.COMPUTE_PLANE_METRICS: "Plane metrics", StepId.COMPUTE_DERIVED_METRICS: "Derived",
                        StepId.GENERATE_STREAMLINES: "Streamlines", StepId.PLANE_STREAMLINES: "Pathlines",
                        StepId.GENERATE_SKELETON: "Skeleton", StepId.GENERATE_GRAPH: "Graph",
                        StepId.GENERATE_PLANES: "Planes", StepId.BACKGROUND_CORRECTION: "Background",
                        StepId.REMOVE_NOISE: "Noise", StepId.UNWRAP_PHASE: "Unwrap"}.get(key, key.label)
                pages = action_sections(key, self.workspace)
                sections.extend((name + ("::" + title if len(pages) > 1 else ""), fields)
                                for title, fields in pages)
            return sections
        return action_sections(action, self.workspace)

    def _open_action_settings(self, action, run=None):
        if (self._pipeline_running_guard("changing action settings") or self._autoseg_running_guard("changing action settings")
                or self._pathline_running_guard("changing action settings") or self._edit_mode is not None):
            return
        if action == "run_all":
            stage = self._workflow_stage_key()
            action = {
                "correction": (StepId.BACKGROUND_CORRECTION, StepId.REMOVE_NOISE, StepId.UNWRAP_PHASE),
                "centerline": (StepId.GENERATE_SKELETON, StepId.GENERATE_GRAPH, StepId.GENERATE_PLANES),
                "hemodynamics": (StepId.COMPUTE_PLANE_METRICS, StepId.COMPUTE_DERIVED_METRICS, StepId.GENERATE_STREAMLINES, StepId.PLANE_STREAMLINES),
            }.get(stage, ())
        sections = self._settings_sections(action)
        if not sections:
            return
        defaults = Workspace()
        apply_config_bundle_to_workspace(defaults, self._config_bundle)
        if action == "input":
            # Input defaults belong to the current source, never a generic 1 mm / 150 VENC case.
            defaults.resolution = self.workspace.resolution
            defaults.venc = self.workspace.venc
            defaults.spatial_order = self.workspace.spatial_order
            defaults.venc_order = self.workspace.venc_order
        title = action.label.replace("&&", "&") if isinstance(action, StepId) else {
            "input": "Reload input", "segmentation": "Automatic segmentation", "render": "3D display",
        }.get(action if isinstance(action, str) else "", "Run all")
        auxiliary = (("Reset noise mask", "reset_noise"),) if action == StepId.REMOVE_NOISE else (
            (("Revert unwrapping", "revert_unwrap"),) if action == StepId.UNWRAP_PHASE else ())
        dialog = ActionSettingsDialog(title, sections, self.workspace, defaults, self,
                                      run_label="Apply and reload" if action == "input" else ("Save and run" if run is not None else None),
                                      auxiliary=auxiliary)
        try:
            if dialog.exec() != QtWidgets.QDialog.Accepted:
                return
            if dialog.operation == "reset_noise":
                self._reset_noise_removal()
                return
            if dialog.operation == "revert_unwrap":
                self._revert_phase_unwrap()
                return
            values = dialog.values()
            changed = any(value != dialog._initial[key] for key, value in values.items())
            self._publish_action_settings(values)
            if "phase_unwrap_params.method" in values:
                method = values["phase_unwrap_params.method"]
                dialog._mask_memory[method] = values["phase_unwrap_params.mask_source"]
                self.workspace.input_state.metadata["gui_phase_unwrap_masks"] = dialog._mask_memory
            if changed and (not isinstance(action, str) or action not in {"render", "input", "segmentation"}):
                keys = action if isinstance(action, tuple) else (action,)
                self._mark_action_settings_pending(keys)
            if changed:
                self.log(f"{title} settings saved." + (" Rerun to update results." if action != "render" else ""))
            self._sync_params_to_ui()
            if dialog.run_requested and run is not None:
                run()
        finally:
            dialog.deleteLater()

    def _publish_action_settings(self, values):
        ws = self.workspace
        input_parameters = {key.split(".")[1]: value for key, value in values.items() if key.startswith("input.")}
        if input_parameters:
            ws.input_state.metadata["gui_input_parameters"] = normalize_input_parameters(input_parameters)
            return
        # Validate everything on a small parameter-only candidate before publishing.
        candidate = copy.copy(ws)
        roots = {key.split(".")[0] for key in values}
        for root in roots:
            current = getattr(ws, root)
            # SegmentationState contains image arrays; settings edits must not
            # duplicate a complete 4D case on the GUI thread.
            setattr(candidate, root, copy.copy(current) if root == "segmentation" else copy.deepcopy(current))
        for key, value in values.items():
            if key == "pwv_params.groups":
                continue
            _set(candidate, key, value)
        if "pwv_params.groups" in values:
            payload = candidate.pwv_params.to_dict()
            payload["groups"] = values["pwv_params.groups"]
            candidate.pwv_params = PwvParams.from_dict(payload, label_map=ws.label_params.label_map)
        if "phase_unwrap_params.method" in values:
            candidate.phase_unwrap_params.enabled = True
        for root in roots:
            setattr(ws, root, getattr(candidate, root))
        if any(key.startswith("render_settings.") for key in values):
            shared = ws.render_settings.get("shared_colorbar_bar_cfg", {})
            for key in ("wss", "tke", "pressure_gradient", "relative_pressure", "streamline"):
                ws.render_settings.setdefault(key + "_bar_cfg", {}).update(shared)
            self._apply_render_settings_to_scene_objects()
        if "streamline_params.pathline_color" in values:
            self._refresh_automatic_pathline_colors()

    def _mark_action_settings_pending(self, keys):
        dependencies = {
            StepId.BACKGROUND_CORRECTION: (StepId.GENERATE_PCMRA, StepId.GENERATE_SKELETON, StepId.GENERATE_GRAPH, StepId.GENERATE_PLANES, StepId.COMPUTE_PLANE_METRICS, StepId.COMPUTE_DERIVED_METRICS, StepId.COMPUTE_PWV, StepId.GENERATE_STREAMLINES, StepId.PLANE_STREAMLINES),
            StepId.UNWRAP_PHASE: (StepId.GENERATE_PCMRA, StepId.COMPUTE_PLANE_METRICS, StepId.COMPUTE_DERIVED_METRICS, StepId.COMPUTE_PWV, StepId.GENERATE_STREAMLINES, StepId.PLANE_STREAMLINES),
            StepId.GENERATE_SKELETON: (StepId.GENERATE_GRAPH, StepId.GENERATE_PLANES, StepId.COMPUTE_PLANE_METRICS, StepId.COMPUTE_PWV),
            StepId.GENERATE_GRAPH: (StepId.GENERATE_PLANES, StepId.COMPUTE_PLANE_METRICS, StepId.COMPUTE_PWV),
            StepId.GENERATE_PLANES: (StepId.COMPUTE_PLANE_METRICS, StepId.COMPUTE_PWV),
            StepId.COMPUTE_DERIVED_METRICS: (StepId.COMPUTE_PLANE_METRICS,),
        }
        pending = set(self.workspace.input_state.metadata.get("gui_pending_settings", []))
        pathlines_changed = False
        for key in keys:
            for step in (key,) + dependencies.get(key, ()):
                if isinstance(step, StepId):
                    self.workspace.pipeline.completed.pop(step.value, None)
                    self.workspace.pipeline.skipped.pop(step.value, None)
                    pending.add(step.value)
                    pathlines_changed = pathlines_changed or step == StepId.PLANE_STREAMLINES
        self.workspace.input_state.metadata["gui_pending_settings"] = sorted(pending)
        if pathlines_changed:
            self.workspace.input_state.metadata["gui_pending_pathline_planes"] = list(range(len(self.workspace.planes)))

    def _sync_params_to_ws(self):
        # Computation settings already live in Workspace. Only the review dock
        # has live controls outside the transactional dialogs.
        if hasattr(self, "segmentation_panel"):
            panel = self.segmentation_panel
            self.workspace.segmentation.cleanup_4d_components = panel.check_cleanup_4d.isChecked()
            self.workspace.segmentation.cleanup_4d_mode = str(panel.combo_cleanup_4d_mode.currentData() or "absolute")
            self.workspace.segmentation.cleanup_4d_min_volume_mm3 = panel.spin_cleanup_4d_volume.value()

    def _sync_params_to_ui(self):
        pending = set(self.workspace.input_state.metadata.get("gui_pending_settings", []))
        for step, button in self.step_buttons.items():
            if self.workspace.pipeline.is_done(step):
                pending.discard(step.value)
            if not hasattr(button, "_settings_base_tooltip"):
                button._settings_base_tooltip = button.toolTip()
            fields = [field for _, fields in action_sections(step, self.workspace) for field in fields]
            summary = []
            for field in fields:
                if field.visible is None and field.kind != "groups":
                    value = _get(self.workspace, field.key, field.default)
                    summary.append(f"{field.label}: {value if value is not None else 'auto'}")
                if len(summary) == 3:
                    break
            button.setToolTip(button._settings_base_tooltip + ("\n" + "\n".join(summary) if summary else "") +
                              ("\nSettings changed; rerun to update results." if step.value in pending else ""))
        self.workspace.input_state.metadata["gui_pending_settings"] = sorted(pending)
        self._refresh_workflow_status()

    def _input_parameters_from_ui(self):
        # Retained method name for the reload/signature call sites.
        return input_values(self.workspace)
