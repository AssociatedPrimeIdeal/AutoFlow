import colorsys
import copy, json, uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ..case_types import BackgroundPhaseCorrectionConfig, PhaseUnwrappingConfig, InputState, LoadedCase, LoaderCapabilities


PATHLINE_CATEGORICAL_COLORS = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
    "#3182bd", "#31a354", "#756bb1", "#e6550d", "#636363",
)


def pathline_categorical_color(index):
    """Return a stable categorical color without repeating after 20 planes."""
    color_index = max(0, int(index))
    if color_index < len(PATHLINE_CATEGORICAL_COLORS):
        return PATHLINE_CATEGORICAL_COLORS[color_index]
    extended_index = color_index - len(PATHLINE_CATEGORICAL_COLORS)
    hue = (0.13 + extended_index * 0.618033988749895) % 1.0
    saturation = 0.62 + 0.10 * (extended_index % 2)
    value = 0.78 + 0.12 * ((extended_index // 2) % 2)
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"


class ObjectKind(Enum):
    SEGMENTATION = "Segmentation"
    SKELETON = "Skeleton"
    GRAPH = "Graph"
    BRANCH = "Branch"
    PLANE = "Plane"
    FLOW = "Flow"
    METRIC = "Metric"
    AUX = "Aux"


class StepId(Enum):
    UNWRAP_PHASE = "step_unwrap_phase"
    GENERATE_SKELETON = "step_skeleton"
    EDIT_SKELETON = "step_edit_skeleton"
    GENERATE_GRAPH = "step_graph"
    EDIT_GRAPH = "step_edit_graph"
    GENERATE_PLANES = "step_planes"
    EDIT_PLANES = "step_edit_planes"
    COMPUTE_PWV = "step_pwv"
    GENERATE_STREAMLINES = "step_streamlines"
    PLANE_STREAMLINES = "step_plane_streamlines"
    COMPUTE_PLANE_METRICS = "step_plane_metrics"
    COMPUTE_DERIVED_METRICS = "step_derived_metrics"

    @property
    def label(self):
        return {
            StepId.UNWRAP_PHASE: "Unwrap Phase",
            StepId.GENERATE_SKELETON: "Generate Skeleton",
            StepId.EDIT_SKELETON: "Edit Skeleton",
            StepId.GENERATE_GRAPH: "Generate Graph",
            StepId.EDIT_GRAPH: "Edit Graph",
            StepId.GENERATE_PLANES: "Generate Planes",
            StepId.EDIT_PLANES: "Edit Planes",
            StepId.COMPUTE_PWV: "Compute PWV",
            StepId.GENERATE_STREAMLINES: "Generate Streamlines",
            StepId.PLANE_STREAMLINES: "Pathlines",
            StepId.COMPUTE_PLANE_METRICS: "Calculate && Save Metrics",
            StepId.COMPUTE_DERIVED_METRICS: "WSS / TKE / Pressure / Vortex",
        }[self]

    @staticmethod
    def top_row_steps():
        return [
            StepId.UNWRAP_PHASE,
            StepId.GENERATE_SKELETON,
            StepId.GENERATE_GRAPH,
            StepId.GENERATE_PLANES,
            StepId.COMPUTE_PLANE_METRICS,
            StepId.COMPUTE_PWV,
        ]

    @staticmethod
    def bottom_row_steps():
        return [
            StepId.COMPUTE_DERIVED_METRICS,
            StepId.GENERATE_STREAMLINES,
            StepId.PLANE_STREAMLINES,
        ]

    @staticmethod
    def extra_row_steps():
        return [
            StepId.EDIT_SKELETON,
            StepId.EDIT_GRAPH,
        ]


@dataclass
class PreprocessParams:
    def to_dict(self):
        return {}

    @staticmethod
    def from_dict(d):
        return PreprocessParams()


@dataclass
class SkeletonParams:
    remove_small_cc: bool = False
    separate_special_label_contacts: bool = True
    special_contact_labels: List[str] = field(default_factory=lambda: ["RBCT", "CCA", "LBCT"])
    min_cc_volume_mm3: float = 50.0
    cc_filter_mode: str = "hybrid"
    cc_rel_min_ratio: float = 0.01
    do_closing: bool = True
    do_opening: bool = False
    gaussian_sigma: float = 0.5
    gaussian_enabled: bool = True
    dilation_iters: int = 0
    erosion_iters: int = 0
    opening_iters: int = 0
    closing_iters: int = 0
    label_map: Dict[str, int] = field(default_factory=dict)
    label_groups: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    single_label_group_name: str = "single_label"
    single_label_browser_color: str = "#d9480f"
    default_group_browser_color: str = "#1c7ed6"

    def to_dict(self):
        return {
            "remove_small_cc": self.remove_small_cc,
            "separate_special_label_contacts": self.separate_special_label_contacts,
            "special_contact_labels": [str(value) for value in self.special_contact_labels],
            "min_cc_volume_mm3": self.min_cc_volume_mm3,
            "cc_filter_mode": str(self.cc_filter_mode),
            "cc_rel_min_ratio": float(self.cc_rel_min_ratio),
            "do_closing": self.do_closing,
            "do_opening": self.do_opening,
            "gaussian_sigma": self.gaussian_sigma,
            "gaussian_enabled": self.gaussian_enabled,
            "dilation_iters": int(self.dilation_iters),
            "erosion_iters": int(self.erosion_iters),
            "opening_iters": int(self.opening_iters),
            "closing_iters": int(self.closing_iters),
            "label_map": {str(k): int(v) for k, v in self.label_map.items()},
            "label_groups": copy.deepcopy(self.label_groups),
            "single_label_group_name": str(self.single_label_group_name),
            "single_label_browser_color": str(self.single_label_browser_color),
            "default_group_browser_color": str(self.default_group_browser_color),
        }

    @staticmethod
    def from_dict(d):
        payload = dict(d or {})
        if "keep_largest_cc" in payload and "remove_small_cc" not in payload:
            payload["remove_small_cc"] = payload["keep_largest_cc"]
        raw_label_map = dict(payload.get("label_map", {}))
        label_map = {}
        for key, value in raw_label_map.items():
            try:
                label_map[str(key)] = int(value)
            except Exception:
                continue
        label_groups = {}
        for group_name, raw_cfg in dict(payload.get("label_groups", {})).items():
            cfg = copy.deepcopy(raw_cfg if isinstance(raw_cfg, dict) else {"labels": raw_cfg})
            labels = []
            for item in list(cfg.get("labels", [])):
                if isinstance(item, str):
                    token = item.strip()
                    if token in label_map:
                        labels.append(int(label_map[token]))
                        continue
                    try:
                        labels.append(int(token))
                    except Exception:
                        continue
                else:
                    try:
                        labels.append(int(item))
                    except Exception:
                        continue
            cfg["labels"] = labels
            preprocess = cfg.get("preprocess", {})
            cfg["preprocess"] = copy.deepcopy(preprocess) if isinstance(preprocess, dict) else {}
            label_groups[str(group_name)] = cfg
        raw_special_labels = payload.get("special_contact_labels", ["RBCT", "CCA", "LBCT"])
        if not isinstance(raw_special_labels, (list, tuple)):
            raw_special_labels = [raw_special_labels]
        special_labels = [str(value).strip() for value in raw_special_labels if str(value).strip()]
        # Accept the pre-rename key when opening an older workspace/config.
        separate_special = payload.get("separate_special_label_contacts", payload.get("separate_label_contacts", True))
        return SkeletonParams(
            remove_small_cc=bool(payload.get("remove_small_cc", False)),
            separate_special_label_contacts=bool(separate_special),
            special_contact_labels=special_labels,
            min_cc_volume_mm3=float(payload.get("min_cc_volume_mm3", 50.0)),
            cc_filter_mode=str(payload.get("cc_filter_mode", "hybrid") or "hybrid"),
            cc_rel_min_ratio=float(payload.get("cc_rel_min_ratio", 0.01)),
            do_closing=bool(payload.get("do_closing", True)),
            do_opening=bool(payload.get("do_opening", False)),
            gaussian_sigma=float(payload.get("gaussian_sigma", 0.5)),
            gaussian_enabled=bool(payload.get("gaussian_enabled", True)),
            dilation_iters=int(payload.get("dilation_iters", 0) or 0),
            erosion_iters=int(payload.get("erosion_iters", 0) or 0),
            opening_iters=int(payload.get("opening_iters", 0) or 0),
            closing_iters=int(payload.get("closing_iters", 0) or 0),
            label_map=label_map,
            label_groups=label_groups,
            single_label_group_name=str(payload.get("single_label_group_name", "single_label") or "single_label"),
            single_label_browser_color=str(payload.get("single_label_browser_color", "#d9480f") or "#d9480f"),
            default_group_browser_color=str(payload.get("default_group_browser_color", "#1c7ed6") or "#1c7ed6"),
        )

    def browser_color_for_group(self, group_name):
        cfg = self.label_groups.get(str(group_name), {})
        color = str(cfg.get("browser_color", "") or "")
        if color:
            return color
        if str(group_name) == str(self.single_label_group_name):
            return str(self.single_label_browser_color)
        return str(self.default_group_browser_color)

    def scene_color_for_group(self, group_name, kind="scene"):
        cfg = self.label_groups.get(str(group_name), {})
        if kind == "skeleton" and cfg.get("skeleton_color"):
            return str(cfg.get("skeleton_color"))
        if kind == "graph" and cfg.get("graph_color"):
            return str(cfg.get("graph_color"))
        if kind == "path" and cfg.get("path_color"):
            return str(cfg.get("path_color"))
        if kind == "plane" and cfg.get("plane_color"):
            return str(cfg.get("plane_color"))
        if cfg.get("scene_color"):
            return str(cfg.get("scene_color"))
        return self.browser_color_for_group(group_name)

    def params_for_group(self, group_name):
        cfg = self.label_groups.get(str(group_name), {})
        overrides = cfg.get("preprocess", {}) if isinstance(cfg.get("preprocess", {}), dict) else {}
        params = SkeletonParams(
            remove_small_cc=self.remove_small_cc,
            separate_special_label_contacts=self.separate_special_label_contacts,
            special_contact_labels=list(self.special_contact_labels),
            min_cc_volume_mm3=self.min_cc_volume_mm3,
            cc_filter_mode=self.cc_filter_mode,
            cc_rel_min_ratio=self.cc_rel_min_ratio,
            do_closing=self.do_closing,
            do_opening=self.do_opening,
            gaussian_sigma=self.gaussian_sigma,
            gaussian_enabled=self.gaussian_enabled,
            dilation_iters=self.dilation_iters,
            erosion_iters=self.erosion_iters,
            opening_iters=self.opening_iters,
            closing_iters=self.closing_iters,
        )
        for key in [
            "remove_small_cc",
            "separate_special_label_contacts",
            "special_contact_labels",
            "min_cc_volume_mm3",
            "cc_filter_mode",
            "cc_rel_min_ratio",
            "do_closing",
            "do_opening",
            "gaussian_sigma",
            "gaussian_enabled",
            "dilation_iters",
            "erosion_iters",
            "opening_iters",
            "closing_iters",
        ]:
            if key in overrides and overrides.get(key) is not None:
                setattr(params, key, overrides.get(key))
        # Group-level configs written before the rename remain readable.
        if "separate_special_label_contacts" not in overrides and "separate_label_contacts" in overrides:
            params.separate_special_label_contacts = bool(overrides.get("separate_label_contacts"))
        return params


@dataclass
class PlaneGenerationParams:
    plane_mode: str = "fixed_step"
    plane_count: int = 3
    cross_section_distance: float = 5.0
    start_distance: float = 0.0
    end_distance: float = 0.0
    anchor: str = "center"
    anchor_offset_mm: float = 5.0
    direction: str = "both"
    spacing_mode: str = "fraction"
    spacing_ratio: float = 0.25
    segmentation_filter: bool = True
    smoothing_window: int = 15
    smoothing_polyorder: int = 2
    inter_time: int = 10

    def to_dict(self):
        return {
            "plane_mode": str(self.plane_mode or "fixed_step"),
            "plane_count": int(self.plane_count),
            "cross_section_distance": float(self.cross_section_distance),
            "start_distance": float(self.start_distance),
            "end_distance": float(self.end_distance),
            "anchor": str(self.anchor or "end"),
            "anchor_offset_mm": float(self.anchor_offset_mm),
            "direction": str(self.direction or "both"),
            "spacing_mode": str(self.spacing_mode or "fraction"),
            "spacing_ratio": float(self.spacing_ratio),
            "segmentation_filter": bool(self.segmentation_filter),
            "smoothing_window": int(self.smoothing_window),
            "smoothing_polyorder": int(self.smoothing_polyorder),
            "inter_time": int(self.inter_time),
        }

    @staticmethod
    def from_dict(d):
        payload = dict(d or {})
        plane_mode = str(payload.get("plane_mode", "") or "").strip().lower()
        if plane_mode not in {"count", "distance", "anchored_offset", "uniform", "fixed_step"}:
            use_center_plane = payload.get("use_center_plane", None)
            plane_mode = "count" if bool(True if use_center_plane is None else use_center_plane) else "distance"
        plane_count = int(payload.get("plane_count", 3) or 3)
        if plane_count != -1:
            plane_count = max(1, plane_count)
        anchor = str(payload.get("anchor", "center") or "center").strip().lower()
        if anchor not in {"start", "center", "end", "junction"}:
            anchor = "center"
        direction = str(payload.get("direction", "both") or "both").strip().lower()
        if direction not in {"toward_start", "toward_end", "both", "start", "end"}:
            direction = "both"
        spacing_mode = str(payload.get("spacing_mode", "fraction") or "fraction").strip().lower()
        if spacing_mode not in {"distance", "fraction"}:
            spacing_mode = "fraction"
        return PlaneGenerationParams(
            plane_mode=plane_mode,
            plane_count=plane_count,
            cross_section_distance=float(payload.get("cross_section_distance", 5.0)),
            start_distance=float(payload.get("start_distance", 5.0)),
            end_distance=float(payload.get("end_distance", 0.0)),
            anchor=anchor,
            anchor_offset_mm=float(payload.get("anchor_offset_mm", 5.0)),
            direction=direction,
            spacing_mode=spacing_mode,
            spacing_ratio=float(payload.get("spacing_ratio", 0.25)),
            segmentation_filter=bool(payload.get("segmentation_filter", True)),
            smoothing_window=int(payload.get("smoothing_window", 15)),
            smoothing_polyorder=int(payload.get("smoothing_polyorder", 2)),
            inter_time=int(payload.get("inter_time", 10)),
        )


@dataclass
class StreamlineParams:
    seed_ratio: float = 0.02
    max_steps: int = 2000
    min_seeds: int = 50
    terminal_speed: float = 0.01
    rng_seed: int = 0
    tube_radius: float = 0.25

    pathline_seed_ratio: float = 0.2
    pathline_max_steps: int = 200
    pathline_min_seeds: int = 50
    pathline_seed_mode: str = "fixed"
    pathline_max_seeds: int = 250
    pathline_terminal_speed: float = 0.01
    pathline_rng_seed: int = 0
    pathline_tube_radius: float = 0.25
    pathline_color: str = "deepskyblue"
    pathline_color_mode: str = "per_plane"
    pathline_temporal_cache_mb: float = 512.0

    def to_dict(self):
        return {
            "seed_ratio": self.seed_ratio,
            "max_steps": self.max_steps,
            "min_seeds": self.min_seeds,
            "terminal_speed": self.terminal_speed,
            "rng_seed": self.rng_seed,
            "tube_radius": self.tube_radius,
            "pathline_seed_ratio": self.pathline_seed_ratio,
            "pathline_max_steps": self.pathline_max_steps,
            "pathline_min_seeds": self.pathline_min_seeds,
            "pathline_seed_mode": str(self.pathline_seed_mode),
            "pathline_max_seeds": self.pathline_max_seeds,
            "pathline_terminal_speed": self.pathline_terminal_speed,
            "pathline_rng_seed": self.pathline_rng_seed,
            "pathline_tube_radius": self.pathline_tube_radius,
            "pathline_color": str(self.pathline_color),
            "pathline_color_mode": str(self.pathline_color_mode),
            "pathline_temporal_cache_mb": float(self.pathline_temporal_cache_mb),
        }

    @staticmethod
    def from_dict(d):
        return StreamlineParams(
            seed_ratio=float(d.get("seed_ratio", 0.02)),
            max_steps=int(d.get("max_steps", 2000)),
            min_seeds=int(d.get("min_seeds", 50)),
            terminal_speed=float(d.get("terminal_speed", 0.01)),
            rng_seed=int(d.get("rng_seed", 0)),
            tube_radius=float(d.get("tube_radius", 0.25)),
            pathline_seed_ratio=float(d.get("pathline_seed_ratio", d.get("seed_ratio", 0.2))),
            pathline_max_steps=int(d.get("pathline_max_steps", d.get("max_steps", 200))),
            pathline_min_seeds=int(d.get("pathline_min_seeds", d.get("min_seeds", 50))),
            pathline_seed_mode=("ratio" if str(d.get("pathline_seed_mode", "fixed") or "fixed").strip().lower() == "ratio" else "fixed"),
            pathline_max_seeds=max(1, int(d.get("pathline_max_seeds", 250))),
            pathline_terminal_speed=float(d.get("pathline_terminal_speed", d.get("terminal_speed", 0.01))),
            pathline_rng_seed=int(d.get("pathline_rng_seed", d.get("rng_seed", 0))),
            pathline_tube_radius=float(d.get("pathline_tube_radius", d.get("tube_radius", 0.25))),
            pathline_color=str(d.get("pathline_color", d.get("plane_pathline_color", "deepskyblue")) or "deepskyblue"),
            pathline_color_mode=(str(d.get("pathline_color_mode", "per_plane") or "per_plane").strip().lower() if str(d.get("pathline_color_mode", "per_plane") or "per_plane").strip().lower() in {"uniform", "per_plane", "per_group"} else "per_plane"),
            pathline_temporal_cache_mb=max(0.0, float(d.get("pathline_temporal_cache_mb", 512.0))),
        )


@dataclass
class DerivedMetricsParams:
    wss_smoothing_iteration: int = 200
    wss_viscosity: float = 4.0
    wss_inward_distance: Optional[float] = None
    wss_parabolic_fitting: bool = True
    wss_no_slip_condition: bool = False
    tke_rho: float = 1060.0
    pressure_gradient_rho: float = 1060.0
    pressure_gradient_viscosity: float = 4.0
    pressure_gradient_smoothing_sigma: float = 0.0
    pressure_gradient_support_erosion_iters: int = 1
    pressure_gradient_layer_opacity: float = 0.6
    relative_pressure_layer_opacity: float = 0.6
    pressure_gradient_use_convective_acceleration: bool = True
    pressure_method: str = "least_squares"
    vortex_smoothing_sigma: float = 0.0
    vortex_support_erosion_iters: int = 1
    step_size: int = 5
    tube_radius: float = 0.1
    use_multithread: bool = False

    def to_dict(self):
        return {
            "wss_smoothing_iteration": self.wss_smoothing_iteration,
            "wss_viscosity": self.wss_viscosity,
            "wss_inward_distance": self.wss_inward_distance,
            "wss_parabolic_fitting": self.wss_parabolic_fitting,
            "wss_no_slip_condition": self.wss_no_slip_condition,
            "tke_rho": self.tke_rho,
            "pressure_gradient_rho": self.pressure_gradient_rho,
            "pressure_gradient_viscosity": self.pressure_gradient_viscosity,
            "pressure_gradient_smoothing_sigma": self.pressure_gradient_smoothing_sigma,
            "pressure_gradient_support_erosion_iters": self.pressure_gradient_support_erosion_iters,
            "pressure_gradient_layer_opacity": self.pressure_gradient_layer_opacity,
            "relative_pressure_layer_opacity": self.relative_pressure_layer_opacity,
            "pressure_gradient_use_convective_acceleration": self.pressure_gradient_use_convective_acceleration,
            "pressure_method": str(self.pressure_method),
            "vortex_smoothing_sigma": self.vortex_smoothing_sigma,
            "vortex_support_erosion_iters": self.vortex_support_erosion_iters,
            "step_size": self.step_size,
            "tube_radius": self.tube_radius,
            "use_multithread": self.use_multithread,
        }

    @staticmethod
    def from_dict(d):
        payload = dict(d or {})
        inward_distance = payload.get("wss_inward_distance", payload.get("inward_distance", None))
        if isinstance(inward_distance, str):
            token = inward_distance.strip().lower()
            inward_distance = None if token in {"", "auto", "none"} else float(inward_distance)
        elif inward_distance is not None:
            inward_distance = float(inward_distance)
        legacy_viscosity = float(payload.get("viscosity", 4.0))
        legacy_rho = float(payload.get("rho", 1060.0))
        pressure_method = str(payload.get("pressure_method", payload.get("pressure_gradient_method", "least_squares")) or "least_squares").strip().lower()
        if pressure_method not in {"least_squares", "ppe"}:
            pressure_method = "least_squares"
        return DerivedMetricsParams(
            wss_smoothing_iteration=int(payload.get("wss_smoothing_iteration", payload.get("smoothing_iteration", 200))),
            wss_viscosity=float(payload.get("wss_viscosity", legacy_viscosity)),
            wss_inward_distance=inward_distance,
            wss_parabolic_fitting=bool(payload.get("wss_parabolic_fitting", payload.get("parabolic_fitting", True))),
            wss_no_slip_condition=bool(payload.get("wss_no_slip_condition", payload.get("no_slip_condition", False))),
            tke_rho=float(payload.get("tke_rho", legacy_rho)),
            pressure_gradient_rho=float(payload.get("pressure_gradient_rho", legacy_rho)),
            pressure_gradient_viscosity=float(payload.get("pressure_gradient_viscosity", legacy_viscosity)),
            pressure_gradient_smoothing_sigma=float(payload.get("pressure_gradient_smoothing_sigma", 0.0)),
            pressure_gradient_support_erosion_iters=int(payload.get("pressure_gradient_support_erosion_iters", 1)),
            pressure_gradient_layer_opacity=float(payload.get("pressure_gradient_layer_opacity", 0.6)),
            relative_pressure_layer_opacity=float(payload.get("relative_pressure_layer_opacity", 0.6)),
            pressure_gradient_use_convective_acceleration=bool(payload.get("pressure_gradient_use_convective_acceleration", True)),
            pressure_method=pressure_method,
            vortex_smoothing_sigma=max(float(payload.get("vortex_smoothing_sigma", 0.0)), 0.0),
            vortex_support_erosion_iters=max(int(payload.get("vortex_support_erosion_iters", 1)), 0),
            step_size=int(payload.get("step_size", 5)),
            tube_radius=float(payload.get("tube_radius", 0.1)),
            use_multithread=bool(payload.get("use_multithread", False)),
        )

    @property
    def smoothing_iteration(self):
        return self.wss_smoothing_iteration

    @smoothing_iteration.setter
    def smoothing_iteration(self, value):
        self.wss_smoothing_iteration = int(value)

    @property
    def viscosity(self):
        return self.wss_viscosity

    @viscosity.setter
    def viscosity(self, value):
        val = float(value)
        self.wss_viscosity = val
        self.pressure_gradient_viscosity = val

    @property
    def inward_distance(self):
        return self.wss_inward_distance

    @inward_distance.setter
    def inward_distance(self, value):
        self.wss_inward_distance = value

    @property
    def parabolic_fitting(self):
        return self.wss_parabolic_fitting

    @parabolic_fitting.setter
    def parabolic_fitting(self, value):
        self.wss_parabolic_fitting = bool(value)

    @property
    def no_slip_condition(self):
        return self.wss_no_slip_condition

    @no_slip_condition.setter
    def no_slip_condition(self, value):
        self.wss_no_slip_condition = bool(value)

    @property
    def rho(self):
        return self.pressure_gradient_rho

    @rho.setter
    def rho(self, value):
        val = float(value)
        self.tke_rho = val
        self.pressure_gradient_rho = val


@dataclass
class LabelParams:
    label_map: Dict[str, int] = field(default_factory=dict)
    label_groups: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    single_label_group_name: str = "single_label"
    single_label_browser_color: str = "#d9480f"
    default_group_browser_color: str = "#1c7ed6"

    def to_dict(self):
        return {
            "label_map": {str(k): int(v) for k, v in self.label_map.items()},
            "label_groups": copy.deepcopy(self.label_groups),
            "single_label_group_name": str(self.single_label_group_name),
            "single_label_browser_color": str(self.single_label_browser_color),
            "default_group_browser_color": str(self.default_group_browser_color),
        }

    @staticmethod
    def from_dict(d):
        payload = dict(d or {})
        raw_label_map = dict(payload.get("label_map", {}))
        label_map = {}
        for key, value in raw_label_map.items():
            try:
                label_map[str(key)] = int(value)
            except Exception:
                continue
        label_groups = {}
        for group_name, raw_cfg in dict(payload.get("label_groups", {})).items():
            cfg = copy.deepcopy(raw_cfg if isinstance(raw_cfg, dict) else {"labels": raw_cfg})
            labels = []
            for item in list(cfg.get("labels", [])):
                if isinstance(item, str):
                    token = item.strip()
                    if token in label_map:
                        labels.append(int(label_map[token]))
                        continue
                    try:
                        labels.append(int(token))
                    except Exception:
                        continue
                else:
                    try:
                        labels.append(int(item))
                    except Exception:
                        continue
            cfg["labels"] = labels
            preprocess = cfg.get("preprocess", {})
            cfg["preprocess"] = copy.deepcopy(preprocess) if isinstance(preprocess, dict) else {}
            label_groups[str(group_name)] = cfg
        return LabelParams(
            label_map=label_map,
            label_groups=label_groups,
            single_label_group_name=str(payload.get("single_label_group_name", "single_label") or "single_label"),
            single_label_browser_color=str(payload.get("single_label_browser_color", "#d9480f") or "#d9480f"),
            default_group_browser_color=str(payload.get("default_group_browser_color", "#1c7ed6") or "#1c7ed6"),
        )

    def browser_color_for_group(self, group_name):
        cfg = self.label_groups.get(str(group_name), {})
        color = str(cfg.get("browser_color", "") or "")
        if color:
            return color
        if str(group_name) == str(self.single_label_group_name):
            return str(self.single_label_browser_color)
        return str(self.default_group_browser_color)

    def scene_color_for_group(self, group_name, kind="scene"):
        cfg = self.label_groups.get(str(group_name), {})
        if kind == "skeleton" and cfg.get("skeleton_color"):
            return str(cfg.get("skeleton_color"))
        if kind == "graph" and cfg.get("graph_color"):
            return str(cfg.get("graph_color"))
        if kind == "path" and cfg.get("path_color"):
            return str(cfg.get("path_color"))
        if kind == "plane" and cfg.get("plane_color"):
            return str(cfg.get("plane_color"))
        if kind == "pwv_plane" and cfg.get("pwv_plane_color"):
            return str(cfg.get("pwv_plane_color"))
        if cfg.get("scene_color"):
            return str(cfg.get("scene_color"))
        return self.browser_color_for_group(group_name)


@dataclass
class PwvGroupParams:
    name: str = ""
    labels: List[int] = field(default_factory=list)

    def to_dict(self):
        return {
            "name": str(self.name),
            "labels": [int(x) for x in self.labels],
        }


@dataclass
class PwvParams:
    enabled: bool = False
    groups: List[PwvGroupParams] = field(default_factory=list)
    plane_interval_mm: float = 10.0
    start_distance: float = 0.0
    end_distance: float = 0.0
    smoothing_window: int = 15
    smoothing_polyorder: int = 2
    inter_time: int = 10
    waveform_key: str = "flowrate_mL_s"
    transit_time_method: str = "foot_to_foot"
    foot_method: str = "tangent"
    foot_savgol_window: int = 5
    foot_savgol_polyorder: int = 2
    foot_threshold_percent: float = 10.0
    xcorr_window: str = "full"
    xcorr_interp_factor: int = 10
    allow_cycle_wrap: bool = True
    minimum_valid_planes: int = 2
    scene_visible: bool = True
    scene_color: str = "#ffd43b"
    plot_color: str = "#2b8a3e"
    fit_color: str = "#f08c00"
    plot_dpi: int = 160

    def to_dict(self):
        return {
            "enabled": bool(self.enabled),
            "groups": [item.to_dict() for item in self.groups],
            "plane_interval_mm": float(self.plane_interval_mm),
            "start_distance": float(self.start_distance),
            "end_distance": float(self.end_distance),
            "smoothing_window": int(self.smoothing_window),
            "smoothing_polyorder": int(self.smoothing_polyorder),
            "inter_time": int(self.inter_time),
            "waveform_key": str(self.waveform_key),
            "transit_time_method": str(self.transit_time_method),
            "foot_method": str(self.foot_method),
            "foot_savgol_window": int(self.foot_savgol_window),
            "foot_savgol_polyorder": int(self.foot_savgol_polyorder),
            "foot_threshold_percent": float(self.foot_threshold_percent),
            "xcorr_window": str(self.xcorr_window),
            "xcorr_interp_factor": int(self.xcorr_interp_factor),
            "allow_cycle_wrap": bool(self.allow_cycle_wrap),
            "minimum_valid_planes": int(self.minimum_valid_planes),
            "scene_visible": bool(self.scene_visible),
            "scene_color": str(self.scene_color),
            "plot_color": str(self.plot_color),
            "fit_color": str(self.fit_color),
            "plot_dpi": int(self.plot_dpi),
        }

    @staticmethod
    def _coerce_labels(raw_labels, label_map):
        labels = []
        for item in list(raw_labels or []):
            if isinstance(item, str):
                token = item.strip()
                if token in label_map:
                    labels.append(int(label_map[token]))
                    continue
                try:
                    labels.append(int(token))
                except Exception:
                    continue
            else:
                try:
                    labels.append(int(item))
                except Exception:
                    continue
        return sorted(set(int(x) for x in labels if int(x) != 0))

    @staticmethod
    def from_dict(d, label_map=None):
        payload = dict(d or {})
        resolved_label_map = dict(label_map or {})
        groups = []
        for idx, raw_group in enumerate(list(payload.get("groups", []))):
            item = raw_group if isinstance(raw_group, dict) else {"labels": raw_group}
            labels = PwvParams._coerce_labels(item.get("labels", []), resolved_label_map)
            if not labels:
                continue
            name = str(item.get("name", "") or f"pwv_{idx}")
            groups.append(PwvGroupParams(name=name, labels=labels))
        return PwvParams(
            enabled=bool(payload.get("enabled", False)),
            groups=groups,
            plane_interval_mm=float(payload.get("plane_interval_mm", 10.0)),
            start_distance=float(payload.get("start_distance", 0.0)),
            end_distance=float(payload.get("end_distance", 0.0)),
            smoothing_window=int(payload.get("smoothing_window", 15)),
            smoothing_polyorder=int(payload.get("smoothing_polyorder", 2)),
            inter_time=int(payload.get("inter_time", 10)),
            waveform_key=str(payload.get("waveform_key", "flowrate_mL_s") or "flowrate_mL_s"),
            transit_time_method=str(payload.get("transit_time_method", "foot_to_foot") or "foot_to_foot"),
            foot_method=str(payload.get("foot_method", "tangent") or "tangent"),
            foot_savgol_window=int(payload.get("foot_savgol_window", 5)),
            foot_savgol_polyorder=int(payload.get("foot_savgol_polyorder", 2)),
            foot_threshold_percent=float(payload.get("foot_threshold_percent", 10.0) or 10.0),
            xcorr_window=str(payload.get("xcorr_window", "full") or "full"),
            xcorr_interp_factor=max(1, int(payload.get("xcorr_interp_factor", 10) or 10)),
            allow_cycle_wrap=bool(payload.get("allow_cycle_wrap", True)),
            minimum_valid_planes=max(2, int(payload.get("minimum_valid_planes", 2) or 2)),
            scene_visible=bool(payload.get("scene_visible", True)),
            scene_color=str(payload.get("scene_color", "#ffd43b") or "#ffd43b"),
            plot_color=str(payload.get("plot_color", "#2b8a3e") or "#2b8a3e"),
            fit_color=str(payload.get("fit_color", "#f08c00") or "#f08c00"),
            plot_dpi=max(72, int(payload.get("plot_dpi", 160) or 160)),
        )


@dataclass
class DicomParameterOverrides:
    resolution: Optional[List[float]] = None
    venc: Optional[List[float]] = None
    spatial_order: Optional[List[str]] = None
    venc_order: Optional[List[str]] = None
    rr: Optional[float] = None

    @staticmethod
    def _coerce_float_triplet(value):
        if value is None:
            return None
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size == 1:
            arr = np.repeat(arr, 3)
        if arr.size < 3:
            return None
        return [float(x) for x in arr[:3]]

    @staticmethod
    def _coerce_label_triplet(value):
        if value is None:
            return None
        labels = [str(x).strip().upper() for x in list(value) if str(x).strip()]
        if len(labels) < 3:
            return None
        return labels[:3]

    def to_dict(self):
        return {
            "resolution": None if self.resolution is None else [float(x) for x in self.resolution[:3]],
            "venc": None if self.venc is None else [float(x) for x in self.venc[:3]],
            "spatial_order": None if self.spatial_order is None else [str(x).upper() for x in self.spatial_order[:3]],
            "venc_order": None if self.venc_order is None else [str(x).upper() for x in self.venc_order[:3]],
            "rr": None if self.rr is None else float(self.rr),
        }

    def to_loader_kwargs(self):
        payload = {}
        if self.resolution is not None:
            payload["resolution"] = [float(x) for x in self.resolution[:3]]
        if self.venc is not None:
            payload["venc"] = [float(x) for x in self.venc[:3]]
        if self.spatial_order is not None:
            payload["spatial_order"] = [str(x).upper() for x in self.spatial_order[:3]]
        if self.venc_order is not None:
            payload["venc_order"] = [str(x).upper() for x in self.venc_order[:3]]
        if self.rr is not None:
            payload["rr"] = float(self.rr)
        return payload

    def has_values(self):
        return bool(self.to_loader_kwargs())

    @staticmethod
    def from_dict(d):
        payload = d or {}
        rr = payload.get("rr")
        return DicomParameterOverrides(
            resolution=DicomParameterOverrides._coerce_float_triplet(payload.get("resolution")),
            venc=DicomParameterOverrides._coerce_float_triplet(payload.get("venc")),
            spatial_order=DicomParameterOverrides._coerce_label_triplet(payload.get("spatial_order")),
            venc_order=DicomParameterOverrides._coerce_label_triplet(payload.get("venc_order")),
            rr=None if rr in (None, "") else float(rr),
        )


@dataclass
class LoaderParams:
    background_phase_correction: BackgroundPhaseCorrectionConfig = field(
        default_factory=lambda: BackgroundPhaseCorrectionConfig(enabled=False)
    )
    dicom_parameter_overrides: "DicomParameterOverrides" = field(
        default_factory=lambda: DicomParameterOverrides()
    )
    dicom_read_workers: int = 1
    # Explicitly bypass segmentation embedded in an input file.  This is
    # useful for cold-start benchmarks and for forcing the model/threshold
    # segmentation branch without changing the source file.
    ignore_embedded_segmentation: bool = False

    def to_dict(self):
        return {
            "background_phase_correction": self.background_phase_correction.to_dict(),
            "dicom_parameter_overrides": self.dicom_parameter_overrides.to_dict(),
            "dicom_read_workers": int(self.dicom_read_workers),
            "ignore_embedded_segmentation": bool(self.ignore_embedded_segmentation),
        }

    @staticmethod
    def from_dict(d):
        payload = d or {}
        return LoaderParams(
            background_phase_correction=BackgroundPhaseCorrectionConfig.from_dict(
                payload.get("background_phase_correction", {"enabled": False})
            ),
            dicom_parameter_overrides=DicomParameterOverrides.from_dict(
                payload.get("dicom_parameter_overrides", {})
            ),
            dicom_read_workers=int(payload.get("dicom_read_workers", 1) or 1),
            ignore_embedded_segmentation=bool(payload.get("ignore_embedded_segmentation", False)),
        )


@dataclass
class PathsState:
    segmask_path: str = ""
    flow_path: str = ""
    workspace_path: str = ""
    output_dir: str = ""


@dataclass
class PipelineFlags:
    completed: Dict[str, bool] = field(default_factory=dict)
    skipped: Dict[str, bool] = field(default_factory=dict)

    def mark_done(self, step: StepId, skipped: bool = False):
        self.completed[step.value] = True
        self.skipped[step.value] = bool(skipped)

    def is_done(self, step: StepId) -> bool:
        return bool(self.completed.get(step.value, False))

    def reset(self):
        self.completed.clear()
        self.skipped.clear()


@dataclass
class SceneObject:
    uid: str
    name: str
    kind: ObjectKind
    data_key: str
    group_name: str = ""
    browser_color: str = ""
    visible: bool = True
    opacity: float = 1.0
    color: str = "white"
    scalars: Optional[str] = None
    cmap: str = "turbo"
    clim: Optional[Tuple[float, float]] = None
    point_size: int = 8
    line_width: int = 2
    tube_radius: float = 0.0
    show_scalar_bar: bool = False
    scalar_bar_title: Optional[str] = None
    scalar_bar_cfg: Dict[str, Any] = field(default_factory=dict)
    dynamic: bool = False
    actor: Any = None
    label_actor: Any = None


@dataclass
class PlaneData:
    center: np.ndarray
    normal: np.ndarray
    label: int = 1
    path_index: int = 0
    distance: float = 0.0
    segmentation_label: int = 0
    group_name: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphData:
    points: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=float))
    edges: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=int))


@dataclass
class DerivedResults:
    plane_metrics: List[Dict[str, Any]] = field(default_factory=list)
    plane_qc: Dict[str, Any] = field(default_factory=dict)
    wss_surfaces: List[Any] = field(default_factory=list)
    wss_volume: Optional[np.ndarray] = None
    tke_volume: Any = None
    tke_array: Optional[np.ndarray] = None
    pressure_gradient_array: Optional[np.ndarray] = None
    pressure_gradient_magnitude: Optional[np.ndarray] = None
    pressure_gradient_peak: Optional[np.ndarray] = None
    pressure_gradient_dt_s: Optional[float] = None
    pressure_gradient_temporal_scheme: str = ""
    pressure_gradient_support_mask: Optional[np.ndarray] = None
    pressure_gradient_display_clim: Optional[Tuple[float, float]] = None
    relative_pressure_array: Optional[np.ndarray] = None
    relative_pressure_peak: Optional[np.ndarray] = None
    relative_pressure_display_clim: Optional[Tuple[float, float]] = None
    centerline_pressure_profiles: List[Dict[str, Any]] = field(default_factory=list)
    vorticity_array: Optional[np.ndarray] = None
    vorticity_magnitude: Optional[np.ndarray] = None
    vorticity_magnitude_peak: Optional[np.ndarray] = None
    q_criterion_array: Optional[np.ndarray] = None
    q_criterion_peak: Optional[np.ndarray] = None
    swirling_strength_array: Optional[np.ndarray] = None
    swirling_strength_peak: Optional[np.ndarray] = None
    vortex_support_mask: Optional[np.ndarray] = None
    streamlines: List[Any] = field(default_factory=list)
    pixelwise_export: Dict[str, Any] = field(default_factory=dict)
    plane_pixelwise_file: str = ""
    pwv_results: List[Dict[str, Any]] = field(default_factory=list)
    pwv_planes: List[Dict[str, Any]] = field(default_factory=list)
    pwv_file: str = ""
    pwv_json_file: str = ""
    pwv_h5_file: str = ""
    # Hashes of the inputs and parameters used to create each derived family.
    # Missing signatures deliberately invalidate legacy workspace caches.
    artifact_signatures: Dict[str, str] = field(default_factory=dict)


@dataclass
class SegmentationVersion:
    data: Optional[np.ndarray] = None
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {
            "data": None if self.data is None else np.asarray(self.data, dtype=np.int16).tolist(),
            "provenance": copy.deepcopy(self.provenance),
        }

    @staticmethod
    def from_dict(d):
        payload = d or {}
        data = payload.get("data")
        return SegmentationVersion(
            data=None if data is None else np.asarray(data, dtype=np.int16),
            provenance=copy.deepcopy(payload.get("provenance", {})),
        )


@dataclass
class SegmentationState:
    original: SegmentationVersion = field(default_factory=SegmentationVersion)
    imported: SegmentationVersion = field(default_factory=SegmentationVersion)
    threshold: SegmentationVersion = field(default_factory=SegmentationVersion)
    auto: SegmentationVersion = field(default_factory=SegmentationVersion)
    active_source: str = ""
    visible: bool = True
    opacity: float = 0.35
    active_label: int = 1
    label_names: Dict[str, str] = field(default_factory=dict)
    label_colors: Dict[str, str] = field(default_factory=dict)
    editing_enabled: bool = False
    tool: str = "brush"
    brush_radius: int = 3
    edit_all_timepoints: bool = True
    working_labels_3d: Optional[np.ndarray] = None
    working_labels_4d: Optional[np.ndarray] = None
    working_source: str = ""
    dirty: bool = False
    mode: str = "input"
    input_source: str = "original"
    import_path: str = ""
    threshold_scalar: str = "pcmra"
    threshold_value: Any = field(default_factory=lambda: {"mode": "manual", "min_percent": 10.0, "max_percent": 100.0})
    threshold_keep_largest_cc: bool = True
    threshold_min_component_volume_mm3: float = 0.0
    threshold_closing: bool = True
    threshold_opening: bool = False
    auto_backend: str = "nnUNet4D"
    auto_model: str = "/nas-data2/ryy/CMR4DFlow2026/Segdata/scripts/nnunet/4D/run_7020_4d_full_ssd_20260824.sh"
    auto_checkpoint: str = "checkpoint_final.pth"
    auto_folds: str = "single"
    auto_device: str = "auto"
    auto_label_map: str = ""
    force_recompute_auto_cache: bool = False
    write_auto_cache: bool = True
    cleanup_4d_components: bool = False
    cleanup_4d_mode: str = "absolute"
    cleanup_4d_min_volume_mm3: float = 50.0

    @staticmethod
    def _coerce_threshold_value(value):
        if isinstance(value, dict):
            mode = str(value.get("mode", "manual") or "manual").strip().lower()
            if mode == "auto":
                return "auto"
            return {
                "mode": "manual",
                "min_percent": float(value.get("min_percent", 10.0)),
                "max_percent": float(value.get("max_percent", 100.0)),
            }
        if isinstance(value, str) and value.strip().lower() == "auto":
            return "auto"
        if value in (None, ""):
            return {"mode": "manual", "min_percent": 10.0, "max_percent": 100.0}
        try:
            return float(value)
        except Exception:
            return {"mode": "manual", "min_percent": 10.0, "max_percent": 100.0}

    def to_dict(self):
        return {
            "original": self.original.to_dict(),
            "imported": self.imported.to_dict(),
            "threshold": self.threshold.to_dict(),
            "auto": self.auto.to_dict(),
            "active_source": self.active_source,
            "visible": bool(self.visible),
            "opacity": float(self.opacity),
            "active_label": int(self.active_label),
            "label_names": copy.deepcopy(self.label_names),
            "label_colors": copy.deepcopy(self.label_colors),
            "editing_enabled": bool(self.editing_enabled),
            "tool": self.tool,
            "brush_radius": int(self.brush_radius),
            "edit_all_timepoints": bool(self.edit_all_timepoints),
            "working_labels_3d": None if self.working_labels_3d is None else np.asarray(self.working_labels_3d, dtype=np.int16).tolist(),
            "working_labels_4d": None if self.working_labels_4d is None else np.asarray(self.working_labels_4d, dtype=np.int16).tolist(),
            "working_source": self.working_source,
            "dirty": bool(self.dirty),
            "mode": self.mode,
            "input_source": self.input_source,
            "import_path": self.import_path,
            "threshold_scalar": self.threshold_scalar,
            "threshold_value": copy.deepcopy(self.threshold_value),
            "threshold_keep_largest_cc": bool(self.threshold_keep_largest_cc),
            "threshold_min_component_volume_mm3": float(self.threshold_min_component_volume_mm3),
            "threshold_closing": bool(self.threshold_closing),
            "threshold_opening": bool(self.threshold_opening),
            "auto_backend": self.auto_backend,
            "auto_model": self.auto_model,
            "auto_checkpoint": self.auto_checkpoint,
            "auto_folds": str(self.auto_folds),
            "auto_device": self.auto_device,
            "auto_label_map": self.auto_label_map,
            "force_recompute_auto_cache": bool(self.force_recompute_auto_cache),
            "write_auto_cache": bool(self.write_auto_cache),
            "cleanup_4d_components": bool(self.cleanup_4d_components),
            "cleanup_4d_mode": str(self.cleanup_4d_mode),
            "cleanup_4d_min_volume_mm3": float(self.cleanup_4d_min_volume_mm3),
        }

    @staticmethod
    def from_dict(d):
        payload = d or {}
        tool = str(payload.get("tool", "brush"))
        if tool not in ("brush", "erase", "relabel"):
            tool = "brush"
        return SegmentationState(
            original=SegmentationVersion.from_dict(payload.get("original", {})),
            imported=SegmentationVersion.from_dict(payload.get("imported", {})),
            threshold=SegmentationVersion.from_dict(payload.get("threshold", {})),
            auto=SegmentationVersion.from_dict(payload.get("auto", {})),
            active_source=str(payload.get("active_source", "")),
            visible=bool(payload.get("visible", True)),
            opacity=float(payload.get("opacity", 0.35)),
            active_label=int(payload.get("active_label", 1)),
            label_names={str(k): str(v) for k, v in dict(payload.get("label_names", {})).items()},
            label_colors={str(k): str(v) for k, v in dict(payload.get("label_colors", {})).items()},
            editing_enabled=bool(payload.get("editing_enabled", False)),
            tool=tool,
            brush_radius=int(payload.get("brush_radius", 3)),
            edit_all_timepoints=bool(payload.get("edit_all_timepoints", True)),
            working_labels_3d=None if payload.get("working_labels_3d") is None else np.asarray(payload.get("working_labels_3d"), dtype=np.int16),
            working_labels_4d=None if payload.get("working_labels_4d") is None else np.asarray(payload.get("working_labels_4d"), dtype=np.int16),
            working_source=str(payload.get("working_source", "")),
            dirty=bool(payload.get("dirty", False)),
            mode=str(payload.get("mode", "input")),
            input_source=str(payload.get("input_source", "original")),
            import_path=str(payload.get("import_path", "")),
            threshold_scalar=str(payload.get("threshold_scalar", "pcmra")),
            threshold_value=SegmentationState._coerce_threshold_value(payload.get("threshold_value")),
            threshold_keep_largest_cc=bool(payload.get("threshold_keep_largest_cc", True)),
            threshold_min_component_volume_mm3=float(payload.get("threshold_min_component_volume_mm3", 0.0)),
            threshold_closing=bool(payload.get("threshold_closing", True)),
            threshold_opening=bool(payload.get("threshold_opening", False)),
            auto_backend=str(payload.get("auto_backend", "nnUNet4D")),
            auto_model=str(payload.get("auto_model", "/nas-data2/ryy/CMR4DFlow2026/Segdata/scripts/nnunet/4D/run_7020_4d_full_ssd_20260824.sh")),
            auto_checkpoint=str(payload.get("auto_checkpoint", "checkpoint_final.pth")),
            auto_folds=str(payload.get("auto_folds", "single") or "single"),
            auto_device=str(payload.get("auto_device", "auto")),
            auto_label_map=str(payload.get("auto_label_map", "")),
            force_recompute_auto_cache=bool(payload.get("force_recompute_auto_cache", False)),
            write_auto_cache=bool(payload.get("write_auto_cache", True)),
            cleanup_4d_components=bool(payload.get("cleanup_4d_components", False)),
            cleanup_4d_mode=str(payload.get("cleanup_4d_mode", "absolute") or "absolute"),
            cleanup_4d_min_volume_mm3=float(payload.get("cleanup_4d_min_volume_mm3", 50.0) or 50.0),
        )


@dataclass
class Workspace:
    paths: PathsState = field(default_factory=PathsState)
    pipeline: PipelineFlags = field(default_factory=PipelineFlags)
    loader_params: LoaderParams = field(default_factory=LoaderParams)
    phase_unwrap_params: PhaseUnwrappingConfig = field(default_factory=PhaseUnwrappingConfig)
    preprocess_params: PreprocessParams = field(default_factory=PreprocessParams)
    skeleton_params: SkeletonParams = field(default_factory=SkeletonParams)
    label_params: LabelParams = field(default_factory=LabelParams)
    plane_gen_params: PlaneGenerationParams = field(default_factory=PlaneGenerationParams)
    streamline_params: StreamlineParams = field(default_factory=StreamlineParams)
    derived_params: DerivedMetricsParams = field(default_factory=DerivedMetricsParams)
    pwv_params: PwvParams = field(default_factory=PwvParams)
    input_state: InputState = field(default_factory=InputState)
    segmentation: SegmentationState = field(default_factory=SegmentationState)

    resolution: np.ndarray = field(default_factory=lambda: np.array([1., 1., 1.]))
    origin: np.ndarray = field(default_factory=lambda: np.array([0., 0., 0.]))
    spatial_order: List[str] = field(default_factory=lambda: ["FH", "AP", "LR"])
    venc_order: List[str] = field(default_factory=lambda: ["FH", "AP", "LR"])
    venc: np.ndarray = field(default_factory=lambda: np.array([1., 1., 1.]))
    rr: float = 1000.0

    segmask_raw: Optional[np.ndarray] = None
    segmask_labels: Optional[np.ndarray] = None
    segmask_labels_3d: Optional[np.ndarray] = None
    segmask_binary: Optional[np.ndarray] = None
    segmask_3d: Optional[np.ndarray] = None
    group_order: List[str] = field(default_factory=list)
    multilabel_groups: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    mag_raw: Optional[np.ndarray] = None
    source_sigma: Optional[np.ndarray] = None
    source_tke_array: Optional[np.ndarray] = None
    correction_raw: Optional[np.ndarray] = None
    correction_high_raw: Optional[np.ndarray] = None

    skeleton_points: Optional[np.ndarray] = None
    skeleton_mask: Optional[np.ndarray] = None

    graph: GraphData = field(default_factory=GraphData)
    branch_labels: Optional[np.ndarray] = None
    centerline_paths: List[np.ndarray] = field(default_factory=list)
    centerline_node_paths: List[List[int]] = field(default_factory=list)
    centerline_paths_smooth: List[np.ndarray] = field(default_factory=list)
    path_info: List[Dict[str, Any]] = field(default_factory=list)
    forks: List[Dict[str, Any]] = field(default_factory=list)
    planes: List[PlaneData] = field(default_factory=list)

    flow_raw: Optional[np.ndarray] = None
    flow_input: Optional[np.ndarray] = None
    phase_wrapped: Optional[np.ndarray] = None
    phase_wrapped_high: Optional[np.ndarray] = None
    phase_unwrap_result: Dict[str, Any] = field(default_factory=dict)
    streamline_seeds: Optional[np.ndarray] = None
    streamline_cache: Dict[int, Any] = field(default_factory=dict)
    streamline_active: bool = False
    pathline_cache: Dict[int, Dict[int, Any]] = field(default_factory=dict)
    pathline_seed_cache: Dict[int, np.ndarray] = field(default_factory=dict)
    active_pathline_plane_indices: List[int] = field(default_factory=list)
    pathline_colors: Dict[int, str] = field(default_factory=dict)
    derived: DerivedResults = field(default_factory=DerivedResults)
    render_settings: Dict[str, Any] = field(default_factory=dict)

    scene_objects: Dict[str, SceneObject] = field(default_factory=dict)
    current_t: int = 0
    data_loaded: bool = False

    ortho_cursor: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0], dtype=int))
    selected_path_index: int = -1
    _preprocess_signature: Any = field(default=None, repr=False, compare=False)

    def _segmentation_version(self, source):
        if source not in ("original", "imported", "threshold", "auto"):
            return None
        return getattr(self.segmentation, source)

    def time_count(self):
        if self.flow_raw is not None and self.flow_raw.ndim == 5:
            return int(self.flow_raw.shape[3])
        if self.segmask_raw is not None and self.segmask_raw.ndim == 4:
            return int(self.segmask_raw.shape[3])
        return 1

    def segmentation_source_names(self):
        names = []
        for source in ["original", "imported", "threshold", "auto"]:
            version = self._segmentation_version(source)
            if version is not None and version.data is not None:
                names.append(source)
        return names

    def get_segmentation_source(self, source):
        version = self._segmentation_version(source)
        if version is None or version.data is None:
            return None
        return np.asarray(version.data, dtype=np.int16)

    def set_segmentation_source(self, source, data, provenance=None):
        version = self._segmentation_version(source)
        if version is None:
            raise ValueError(f"unknown segmentation source: {source}")
        version.data = None if data is None else np.asarray(data, dtype=np.int16).copy()
        version.provenance = copy.deepcopy(provenance or {})
        if version.data is None and self.segmentation.active_source == source:
            self.segmentation.active_source = ""
            self.segmask_raw = None

    def get_active_segmentation(self):
        if self.segmentation.active_source:
            data = self.get_segmentation_source(self.segmentation.active_source)
            if data is not None:
                return data
        return None if self.segmask_raw is None else np.asarray(self.segmask_raw, dtype=np.int16)

    def get_active_segmentation_provenance(self):
        version = self._segmentation_version(self.segmentation.active_source)
        if version is None:
            return {}
        return copy.deepcopy(version.provenance)

    def activate_segmentation_source(self, source):
        if not source:
            self.segmentation.active_source = ""
            self.segmask_raw = None
            self.clear_working_segmentation()
            return False
        data = self.get_segmentation_source(source)
        if data is None:
            return False
        self.segmentation.active_source = str(source)
        self.segmask_raw = np.asarray(data, dtype=np.int16).copy()
        self.clear_working_segmentation()
        return True

    def clear_working_segmentation(self):
        self.segmentation.working_labels_3d = None
        self.segmentation.working_labels_4d = None
        self.segmentation.working_source = ""
        self.segmentation.dirty = False

    def segmentation_display_4d(self):
        if self.segmentation.working_labels_4d is not None:
            return np.asarray(self.segmentation.working_labels_4d, dtype=np.int16)
        if self.segmentation.working_labels_3d is not None:
            labels = np.asarray(self.segmentation.working_labels_3d, dtype=np.int16)
            active = self.get_active_segmentation()
            nt = int(active.shape[3]) if active is not None and active.ndim == 4 else self.time_count()
            nt = max(1, nt)
            return np.repeat(labels[..., None], nt, axis=3).astype(np.int16)
        active = self.get_active_segmentation()
        return None if active is None else np.asarray(active, dtype=np.int16)

    def segmentation_display_3d(self):
        display = self.segmentation_display_4d()
        if display is None:
            return None
        if display.ndim == 3:
            return np.asarray(display, dtype=np.int16)
        return np.max(display, axis=3).astype(np.int16)

    def display_unique_labels(self):
        display = self.segmentation_display_4d()
        if display is None:
            return []
        return sorted(int(x) for x in np.unique(display) if int(x) != 0)

    def has_flow(self):
        return self.flow_raw is not None

    def has_segmentation(self):
        return self.segmask_raw is not None

    def unique_labels(self):
        if self.segmask_raw is None:
            return []
        return sorted(int(x) for x in np.unique(self.segmask_raw) if x != 0)

    def add_object(self, name, kind, data_key, **kw):
        uid = str(uuid.uuid4())
        self.scene_objects[uid] = SceneObject(uid=uid, name=name, kind=kind, data_key=data_key, **kw)
        return uid

    def remove_object(self, uid):
        return self.scene_objects.pop(uid, None)

    def remove_object_by_data_key(self, data_key):
        to_del = [u for u, o in self.scene_objects.items() if o.data_key == data_key]
        return [self.scene_objects.pop(u) for u in to_del]

    def remove_objects_by_prefix(self, prefix):
        to_del = [u for u, o in self.scene_objects.items() if o.data_key.startswith(prefix)]
        return [self.scene_objects.pop(u) for u in to_del]

    def set_object_visible_by_data_key(self, data_key, visible):
        for o in self.scene_objects.values():
            if o.data_key == data_key:
                o.visible = visible

    def clear_streamlines(self):
        self.streamline_seeds = None
        self.streamline_cache.clear()
        self.streamline_active = False
        self.remove_object_by_data_key("streamlines_live")

    def clear_pathlines(self):
        self.pathline_cache.clear()
        self.pathline_seed_cache.clear()
        self.active_pathline_plane_indices = []
        self.remove_objects_by_prefix("pathline_")

    def pathline_color_for_plane(self, plane_idx):
        plane_idx = int(plane_idx)
        color = str(self.pathline_colors.get(plane_idx, "") or "")
        if color:
            return color
        mode = str(getattr(self.streamline_params, "pathline_color_mode", "per_plane") or "per_plane").strip().lower()
        if mode == "uniform":
            return str(self.streamline_params.pathline_color or "deepskyblue")
        if mode == "per_group":
            plane_groups = [str(getattr(plane, "group_name", "") or "Global") for plane in self.planes]
            group_order = list(dict.fromkeys(
                [str(name or "Global") for name in self.group_order] + plane_groups
            ))
            group_name = plane_groups[plane_idx] if 0 <= plane_idx < len(plane_groups) else "Global"
            color_index = group_order.index(group_name) if group_name in group_order else 0
        else:
            color_index = plane_idx
        return pathline_categorical_color(color_index)

    def set_pathline_color_for_plane(self, plane_idx, color):
        self.pathline_colors[int(plane_idx)] = str(color or self.streamline_params.pathline_color or "deepskyblue")

    def clear_plane_streamlines(self):
        self.clear_pathlines()

    def reset_segmentation_results(self):
        self._preprocess_signature = None
        for attr in [
            "segmask_labels",
            "segmask_labels_3d",
            "segmask_binary",
            "segmask_3d",
            "skeleton_points",
            "skeleton_mask",
            "branch_labels",
        ]:
            setattr(self, attr, None)
        self.group_order = []
        self.multilabel_groups = {}
        self.graph = GraphData()
        self.centerline_paths = []
        self.centerline_node_paths = []
        self.centerline_paths_smooth = []
        self.path_info = []
        self.forks = []
        self.planes = []
        self.selected_path_index = -1
        self.pipeline.reset()
        # Keep loaded flow and phase data intact; unwrapping is an independent
        # optional stage and can be rerun after segmentation edits.
        self.clear_streamlines()
        self.clear_pathlines()
        self.derived = DerivedResults()
        self.remove_object_by_data_key("segmask_pre_surface")
        self.remove_object_by_data_key("skeleton_points")
        self.remove_object_by_data_key("skeleton_mask_surface")
        self.remove_object_by_data_key("segmask_3d_surface")
        self.remove_object_by_data_key("graph_lines")
        self.remove_object_by_data_key("fork_markers")
        self.remove_objects_by_prefix("segmask_group_")
        self.remove_objects_by_prefix("skeleton_")
        self.remove_objects_by_prefix("graph_")
        self.remove_objects_by_prefix("forks_")
        self.remove_object_by_data_key("wss_surface_live")
        self.remove_object_by_data_key("tke_volume")
        self.remove_object_by_data_key("pressure_gradient_volume")
        self.remove_object_by_data_key("relative_pressure_volume")
        self.remove_object_by_data_key("pwv_planes")
        self.remove_object_by_data_key("derived_streamlines_live")
        self.remove_objects_by_prefix("plane_")
        self.remove_objects_by_prefix("path_")
        self.remove_objects_by_prefix("smooth_path_")
        self.remove_objects_by_prefix("path_arrow_")

    def reset_all(self):
        for attr, default in [
            ("paths", PathsState()),
            ("pipeline", PipelineFlags()),
            ("loader_params", LoaderParams()),
            ("phase_unwrap_params", PhaseUnwrappingConfig()),
            ("preprocess_params", PreprocessParams()),
            ("skeleton_params", SkeletonParams()),
            ("label_params", LabelParams()),
            ("plane_gen_params", PlaneGenerationParams()),
            ("streamline_params", StreamlineParams()),
            ("derived_params", DerivedMetricsParams()),
            ("pwv_params", PwvParams()),
            ("input_state", InputState()),
            ("segmentation", SegmentationState()),
        ]:
            setattr(self, attr, default)
        self.resolution = np.array([1., 1., 1.])
        self.origin = np.array([0., 0., 0.])
        self.venc = np.array([1., 1., 1.])
        self.rr = 1000.0
        for attr in ["segmask_raw", "segmask_labels", "segmask_labels_3d", "segmask_binary", "segmask_3d",
                      "skeleton_points", "skeleton_mask", "branch_labels", "flow_raw", "flow_input",
                      "phase_wrapped", "phase_wrapped_high",
                      "streamline_seeds", "mag_raw", "source_sigma", "source_tke_array", "correction_raw", "correction_high_raw"]:
            setattr(self, attr, None)
        self.group_order = []
        self.multilabel_groups = {}
        self.graph = GraphData()
        self.centerline_paths = []
        self.centerline_node_paths = []
        self.centerline_paths_smooth = []
        self.path_info = []
        self.forks = []
        self.planes = []
        self.streamline_cache = {}
        self.streamline_active = False
        self.pathline_cache = {}
        self.pathline_seed_cache = {}
        self.active_pathline_plane_indices = []
        self.pathline_colors = {}
        self.derived = DerivedResults()
        self.scene_objects = {}
        self.current_t = 0
        self.data_loaded = False
        self.ortho_cursor = np.array([0, 0, 0], dtype=int)
        self.selected_path_index = -1
        self._preprocess_signature = None
        self.phase_unwrap_result = {}

    def snapshot_dict(self):
        def arr(v):
            return None if v is None else np.asarray(v).tolist()
        group_payload = {}
        for group_name, state in self.multilabel_groups.items():
            graph_state = state.get("graph", GraphData())
            group_payload[str(group_name)] = {
                "labels": [int(x) for x in state.get("labels", [])],
                "browser_color": str(state.get("browser_color", "") or ""),
                "scene_color": str(state.get("scene_color", "") or ""),
                "segmask_binary": arr(state.get("segmask_binary")),
                "segmask_3d": arr(state.get("segmask_3d")),
                "clean_mask_3d": arr(state.get("clean_mask_3d")),
                "skeleton_points": arr(state.get("skeleton_points")),
                "skeleton_mask": arr(state.get("skeleton_mask")),
                "graph": {
                    "points": arr(graph_state.points if isinstance(graph_state, GraphData) else graph_state.get("points")),
                    "edges": arr(graph_state.edges if isinstance(graph_state, GraphData) else graph_state.get("edges")),
                },
                "branch_labels": arr(state.get("branch_labels")),
                "centerline_paths": [arr(x) for x in state.get("centerline_paths", [])],
                "centerline_node_paths": [list(map(int, x)) for x in state.get("centerline_node_paths", [])],
                "centerline_paths_smooth": [arr(x) for x in state.get("centerline_paths_smooth", [])],
                "path_info": copy.deepcopy(state.get("path_info", [])),
                "forks": copy.deepcopy(state.get("forks", [])),
                "planes": [{"center": arr(p.center), "normal": arr(p.normal), "label": int(p.label),
                            "segmentation_label": int(getattr(p, "segmentation_label", 0) or 0),
                            "path_index": int(p.path_index), "distance": float(p.distance), "group_name": str(p.group_name),
                            "metrics": copy.deepcopy(p.metrics)} for p in state.get("planes", [])],
                "path_index_offset": int(state.get("path_index_offset", 0)),
                "plane_index_offset": int(state.get("plane_index_offset", 0)),
            }
        return {
            "paths": {"segmask_path": self.paths.segmask_path, "flow_path": self.paths.flow_path,
                      "workspace_path": self.paths.workspace_path, "output_dir": self.paths.output_dir},
            "pipeline": {"completed": dict(self.pipeline.completed), "skipped": dict(self.pipeline.skipped)},
            "loader_params": self.loader_params.to_dict(),
            "phase_unwrap_params": self.phase_unwrap_params.to_dict(),
            "preprocess_params": self.preprocess_params.to_dict(),
            "skeleton_params": self.skeleton_params.to_dict(),
            "label_params": self.label_params.to_dict(),
            "plane_gen_params": self.plane_gen_params.to_dict(),
            "streamline_params": self.streamline_params.to_dict(),
            "derived_params": self.derived_params.to_dict(),
            "pwv_params": self.pwv_params.to_dict(),
            "input_state": self.input_state.to_dict(),
            "segmentation": self.segmentation.to_dict(),
            "resolution": arr(self.resolution),
            "origin": arr(self.origin),
            "spatial_order": list(self.spatial_order),
            "venc_order": list(self.venc_order),
            "venc": arr(self.venc),
            "rr": float(self.rr),
            "segmask_raw": arr(self.segmask_raw),
            "segmask_labels": arr(self.segmask_labels),
            "segmask_binary": arr(self.segmask_binary),
            "segmask_3d": arr(self.segmask_3d),
            "group_order": [str(x) for x in self.group_order],
            "multilabel_groups": group_payload,
            "mag_raw": arr(self.mag_raw),
            "source_sigma": arr(self.source_sigma),
            "source_tke_array": arr(self.source_tke_array),
            "correction_raw": arr(self.correction_raw),
            "correction_high_raw": arr(self.correction_high_raw),
            "skeleton_points": arr(self.skeleton_points),
            "skeleton_mask": arr(self.skeleton_mask),
            "graph": {"points": arr(self.graph.points), "edges": arr(self.graph.edges)},
            "branch_labels": arr(self.branch_labels),
            "centerline_paths": [arr(x) for x in self.centerline_paths],
            "centerline_node_paths": [list(map(int, x)) for x in self.centerline_node_paths],
            "centerline_paths_smooth": [arr(x) for x in self.centerline_paths_smooth],
            "path_info": copy.deepcopy(self.path_info),
            "forks": copy.deepcopy(self.forks),
            "planes": [{"center": arr(p.center), "normal": arr(p.normal), "label": int(p.label),
                        "segmentation_label": int(getattr(p, "segmentation_label", 0) or 0),
                        "path_index": int(p.path_index), "distance": float(p.distance),
                        "group_name": str(p.group_name), "metrics": copy.deepcopy(p.metrics)} for p in self.planes],
            "flow_raw": arr(self.flow_raw),
            "flow_input": arr(self.flow_input),
            "phase_wrapped": arr(self.phase_wrapped),
            "phase_wrapped_high": arr(self.phase_wrapped_high),
            "phase_unwrap_result": {
                str(k): (arr(v) if isinstance(v, np.ndarray) else copy.deepcopy(v))
                for k, v in dict(self.phase_unwrap_result or {}).items()
            },
            "streamline_seeds": arr(self.streamline_seeds),
            "streamline_active": self.streamline_active,
            "active_pathline_plane_indices": [int(x) for x in self.active_pathline_plane_indices],
            "pathline_colors": {str(int(k)): str(v) for k, v in self.pathline_colors.items()},
            "derived": {
                "plane_metrics": copy.deepcopy(self.derived.plane_metrics),
                "plane_qc": copy.deepcopy(self.derived.plane_qc),
                "plane_pixelwise_file": str(self.derived.plane_pixelwise_file or ""),
                "centerline_pressure_profiles": copy.deepcopy(self.derived.centerline_pressure_profiles),
                "pressure_gradient_dt_s": self.derived.pressure_gradient_dt_s,
                "pressure_gradient_temporal_scheme": str(self.derived.pressure_gradient_temporal_scheme or ""),
                "pwv_results": copy.deepcopy(self.derived.pwv_results),
                "pwv_planes": copy.deepcopy(self.derived.pwv_planes),
                "pwv_file": str(self.derived.pwv_file or ""),
                "pwv_json_file": str(self.derived.pwv_json_file or ""),
                "pwv_h5_file": str(self.derived.pwv_h5_file or ""),
                "artifact_signatures": copy.deepcopy(self.derived.artifact_signatures),
            },
            "render_settings": copy.deepcopy(self.render_settings),
            "scene_objects": [
                {"uid": o.uid, "name": o.name, "kind": o.kind.value, "data_key": o.data_key,
                 "group_name": o.group_name, "browser_color": o.browser_color,
                 "visible": o.visible, "opacity": o.opacity, "color": o.color,
                 "scalars": o.scalars, "cmap": o.cmap,
                 "clim": list(o.clim) if o.clim else None,
                 "point_size": o.point_size, "line_width": o.line_width,
                 "tube_radius": o.tube_radius, "show_scalar_bar": o.show_scalar_bar,
                 "scalar_bar_title": o.scalar_bar_title, "scalar_bar_cfg": copy.deepcopy(o.scalar_bar_cfg), "dynamic": o.dynamic}
                for o in self.scene_objects.values()],
            "current_t": self.current_t, "data_loaded": self.data_loaded,
            "selected_path_index": int(self.selected_path_index),
        }

    def restore_dict(self, d):
        self.paths = PathsState(**{k: d.get("paths", {}).get(k, "") for k in ["segmask_path", "flow_path", "workspace_path", "output_dir"]})
        self.pipeline = PipelineFlags(completed=dict(d.get("pipeline", {}).get("completed", {})),
                                      skipped=dict(d.get("pipeline", {}).get("skipped", {})))
        self.loader_params = LoaderParams.from_dict(d.get("loader_params", {}))
        self.phase_unwrap_params = PhaseUnwrappingConfig.from_dict(d.get("phase_unwrap_params", {}))
        self.preprocess_params = PreprocessParams.from_dict(d.get("preprocess_params", {}))
        self.skeleton_params = SkeletonParams.from_dict(d.get("skeleton_params", {}))
        self.label_params = LabelParams.from_dict(d.get("label_params", {}))
        self.plane_gen_params = PlaneGenerationParams.from_dict(d.get("plane_gen_params", {}))
        self.streamline_params = StreamlineParams.from_dict(d.get("streamline_params", {}))
        self.derived_params = DerivedMetricsParams.from_dict(d.get("derived_params", {}))
        self.pwv_params = PwvParams.from_dict(d.get("pwv_params", {}), label_map=self.label_params.label_map)
        self.input_state = InputState.from_dict(d.get("input_state", {}))
        self.segmentation = SegmentationState.from_dict(d.get("segmentation", {}))
        self.resolution = np.asarray(d.get("resolution", [1, 1, 1]), dtype=float)
        self.origin = np.asarray(d.get("origin", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
        self.spatial_order = [str(x) for x in d.get("spatial_order", ["FH", "AP", "LR"])]
        self.venc_order = [str(x) for x in d.get("venc_order", ["FH", "AP", "LR"])]
        self.venc = np.asarray(d.get("venc", [1, 1, 1]), dtype=float)
        self.rr = float(d.get("rr", 1000.0))

        def nparr(k, dt=np.float64):
            v = d.get(k)
            return None if v is None else np.asarray(v, dtype=dt)

        self.segmask_raw = nparr("segmask_raw", np.int16)
        self.segmask_labels = nparr("segmask_labels", np.int16)
        self.segmask_binary = None if d.get("segmask_binary") is None else np.asarray(d["segmask_binary"], dtype=bool)
        self.segmask_3d = None if d.get("segmask_3d") is None else np.asarray(d["segmask_3d"], dtype=bool)
        self.group_order = [str(x) for x in d.get("group_order", [])]
        self.multilabel_groups = {}
        for group_name, state in dict(d.get("multilabel_groups", {})).items():
            graph_state = state.get("graph", {}) if isinstance(state, dict) else {}
            planes = []
            for p in state.get("planes", []):
                planes.append(PlaneData(
                    center=np.asarray(p.get("center", [0.0, 0.0, 0.0]), dtype=float),
                    normal=np.asarray(p.get("normal", [1.0, 0.0, 0.0]), dtype=float),
                    label=int(p.get("label", 1)),
                    segmentation_label=int(p.get("segmentation_label", 0) or 0),
                    path_index=int(p.get("path_index", 0)),
                    distance=float(p.get("distance", 0.0)),
                    group_name=str(p.get("group_name", group_name) or group_name),
                    metrics=copy.deepcopy(p.get("metrics", {})),
                ))
            self.multilabel_groups[str(group_name)] = {
                "labels": [int(x) for x in state.get("labels", [])],
                "browser_color": str(state.get("browser_color", "") or ""),
                "scene_color": str(state.get("scene_color", "") or ""),
                "segmask_binary": None if state.get("segmask_binary") is None else np.asarray(state.get("segmask_binary"), dtype=bool),
                "segmask_3d": None if state.get("segmask_3d") is None else np.asarray(state.get("segmask_3d"), dtype=bool),
                "clean_mask_3d": None if state.get("clean_mask_3d") is None else np.asarray(state.get("clean_mask_3d"), dtype=bool),
                "skeleton_points": None if state.get("skeleton_points") is None else np.asarray(state.get("skeleton_points"), dtype=float),
                "skeleton_mask": None if state.get("skeleton_mask") is None else np.asarray(state.get("skeleton_mask"), dtype=bool),
                "graph": GraphData(
                    points=np.asarray(graph_state.get("points", []), dtype=float).reshape(-1, 3) if graph_state.get("points") else np.empty((0, 3)),
                    edges=np.asarray(graph_state.get("edges", []), dtype=int).reshape(-1, 2) if graph_state.get("edges") else np.empty((0, 2), dtype=int),
                ),
                "branch_labels": None if state.get("branch_labels") is None else np.asarray(state.get("branch_labels"), dtype=np.int16),
                "centerline_paths": [np.asarray(x, dtype=float) for x in state.get("centerline_paths", [])],
                "centerline_node_paths": [list(map(int, x)) for x in state.get("centerline_node_paths", [])],
                "centerline_paths_smooth": [np.asarray(x, dtype=float) for x in state.get("centerline_paths_smooth", [])],
                "path_info": copy.deepcopy(state.get("path_info", [])),
                "forks": copy.deepcopy(state.get("forks", [])),
                "planes": planes,
                "path_index_offset": int(state.get("path_index_offset", 0)),
                "plane_index_offset": int(state.get("plane_index_offset", 0)),
            }
        if self.segmentation.original.data is None and self.segmask_raw is not None:
            self.set_segmentation_source("original", self.segmask_raw, provenance={"source": "original"})
            if not self.segmentation.active_source:
                self.segmentation.active_source = "original"
        self.mag_raw = nparr("mag_raw")
        self.source_sigma = nparr("source_sigma")
        self.source_tke_array = nparr("source_tke_array")
        self.correction_raw = nparr("correction_raw", np.float32)
        self.correction_high_raw = nparr("correction_high_raw", np.float32)
        self.skeleton_points = nparr("skeleton_points")
        self.skeleton_mask = nparr("skeleton_mask")
        gd = d.get("graph", {})
        self.graph = GraphData(
            points=np.asarray(gd.get("points", []), dtype=float).reshape(-1, 3) if gd.get("points") else np.empty((0, 3)),
            edges=np.asarray(gd.get("edges", []), dtype=int).reshape(-1, 2) if gd.get("edges") else np.empty((0, 2), dtype=int))
        self.branch_labels = nparr("branch_labels")
        self.centerline_paths = [np.asarray(x, dtype=float) for x in d.get("centerline_paths", [])]
        self.centerline_node_paths = [list(map(int, x)) for x in d.get("centerline_node_paths", [])]
        self.centerline_paths_smooth = [np.asarray(x, dtype=float) for x in d.get("centerline_paths_smooth", [])]
        self.path_info = copy.deepcopy(d.get("path_info", []))
        self.forks = copy.deepcopy(d.get("forks", []))
        self.planes = []
        for p in d.get("planes", []):
            self.planes.append(PlaneData(
                center=np.asarray(p["center"], dtype=float), normal=np.asarray(p["normal"], dtype=float),
                label=int(p.get("label", 1)), segmentation_label=int(p.get("segmentation_label", 0) or 0), path_index=int(p.get("path_index", 0)),
                distance=float(p.get("distance", 0.0)), group_name=str(p.get("group_name", "") or ""),
                metrics=copy.deepcopy(p.get("metrics", {}))))
        self.flow_raw = nparr("flow_raw")
        self.flow_input = nparr("flow_input")
        self.phase_wrapped = nparr("phase_wrapped", np.float32)
        self.phase_wrapped_high = nparr("phase_wrapped_high", np.float32)
        self.phase_unwrap_result = copy.deepcopy(d.get("phase_unwrap_result", {}))
        self.streamline_seeds = nparr("streamline_seeds")
        self.streamline_cache = {}
        self.streamline_active = bool(d.get("streamline_active", False))
        self.pathline_cache = {}
        self.pathline_seed_cache = {}
        self.active_pathline_plane_indices = [int(x) for x in d.get("active_pathline_plane_indices", [])]
        self.pathline_colors = {int(k): str(v) for k, v in d.get("pathline_colors", {}).items()}
        derived_state = d.get("derived", {}) if isinstance(d.get("derived", {}), dict) else {}
        self.derived = DerivedResults(
            plane_metrics=copy.deepcopy(derived_state.get("plane_metrics", [])),
            plane_qc=copy.deepcopy(derived_state.get("plane_qc", {})),
            plane_pixelwise_file=str(derived_state.get("plane_pixelwise_file", "") or ""),
            centerline_pressure_profiles=copy.deepcopy(derived_state.get("centerline_pressure_profiles", [])),
            pressure_gradient_dt_s=(
                None
                if derived_state.get("pressure_gradient_dt_s") is None
                else float(derived_state.get("pressure_gradient_dt_s"))
            ),
            pressure_gradient_temporal_scheme=str(
                derived_state.get("pressure_gradient_temporal_scheme", "") or ""
            ),
            pwv_results=copy.deepcopy(derived_state.get("pwv_results", [])),
            pwv_planes=copy.deepcopy(derived_state.get("pwv_planes", [])),
            pwv_file=str(derived_state.get("pwv_file", "") or ""),
            pwv_json_file=str(derived_state.get("pwv_json_file", "") or ""),
            pwv_h5_file=str(derived_state.get("pwv_h5_file", "") or ""),
            artifact_signatures={
                str(k): str(v)
                for k, v in dict(derived_state.get("artifact_signatures", {})).items()
            },
        )
        self.render_settings = copy.deepcopy(d.get("render_settings", {}))
        self.scene_objects = {}
        for it in d.get("scene_objects", []):
            uid = it["uid"]
            self.scene_objects[uid] = SceneObject(
                uid=uid, name=it["name"], kind=ObjectKind(it["kind"]), data_key=it["data_key"],
                group_name=str(it.get("group_name", "") or ""), browser_color=str(it.get("browser_color", "") or ""),
                visible=bool(it.get("visible", True)), opacity=float(it.get("opacity", 1.0)),
                color=it.get("color", "white"), scalars=it.get("scalars"),
                cmap=it.get("cmap", "turbo"),
                clim=tuple(it["clim"]) if it.get("clim") else None,
                point_size=int(it.get("point_size", 8)), line_width=int(it.get("line_width", 2)),
                tube_radius=float(it.get("tube_radius", 0.0)),
                show_scalar_bar=bool(it.get("show_scalar_bar", False)),
                scalar_bar_title=it.get("scalar_bar_title"),
                scalar_bar_cfg=copy.deepcopy(it.get("scalar_bar_cfg", {})),
                dynamic=bool(it.get("dynamic", False)))
        self.current_t = int(d.get("current_t", 0))
        self.data_loaded = bool(d.get("data_loaded", False))
        self.selected_path_index = int(d.get("selected_path_index", -1))
