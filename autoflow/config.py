import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional

from .core.models import (
    DerivedMetricsParams,
    LabelParams,
    LoaderParams,
    PlaneGenerationParams,
    PwvParams,
    SegmentationState,
    SkeletonParams,
    StreamlineParams,
    Workspace,
)

DEFAULT_WSS_BAR_CFG = {
    "position_x": 0.75,
    "position_y": 0.2,
    "height": 0.22,
    "width": 0.05,
    "title_font_size": 40,
    "label_font_size": 32,
}

DEFAULT_TKE_BAR_CFG = {
    "position_x": 0.75,
    "position_y": 0.2,
    "height": 0.22,
    "width": 0.05,
    "title_font_size": 40,
    "label_font_size": 32,
}

DEFAULT_STREAMLINE_BAR_CFG = {
    "position_x": 0.75,
    "position_y": 0.2,
    "height": 0.22,
    "width": 0.05,
    "title_font_size": 40,
    "label_font_size": 32,
}

DEFAULT_PRESSURE_GRADIENT_BAR_CFG = {
    "position_x": 0.75,
    "position_y": 0.2,
    "height": 0.22,
    "width": 0.05,
    "title_font_size": 40,
    "label_font_size": 32,
}

DEFAULT_RELATIVE_PRESSURE_BAR_CFG = {
    "position_x": 0.75,
    "position_y": 0.2,
    "height": 0.22,
    "width": 0.05,
    "title_font_size": 40,
    "label_font_size": 32,
}

DEFAULT_SHARED_COLORBAR_CFG = {
    "show": True,
    "bar_cfg": {
        "position_x": 0.87,
        "position_y": 0.15,
        "height": 0.65,
        "width": 0.08,
        "title_font_size": 14,
        "label_font_size": 11,
    },
}

DEFAULT_PLANE_LABEL_CFG = {
    "prefix": "planeidx=",
    "font_size": 28,
    "text_color": "black",
    "shape_color": "yellow",
    "shape_opacity": 0.85,
}

DEFAULT_PLANE_RENDER_CFG = {
    "default": {
        "plane_color": "yellow",
        "plane_opacity": 0.75,
    },
    "groups": {},
}

DEFAULT_PLANE_VIDEO_CFG = {
    "show_skeleton": True,
    "skeleton_point_size": 10.0,
    "label": copy.deepcopy(DEFAULT_PLANE_LABEL_CFG),
    "default": {
        "skeleton_color": "",
        "plane_size": None,
        "plane_color": "yellow",
        "plane_opacity": 0.75,
    },
    "groups": {},
}

DEFAULT_SKELETON_LABEL_MAP = {
    "background": 0,
    "AAO": 1,
    "ARCH": 3,
    "DAO": 4,
    "RBCT": 5,
    "CCA": 6,
    "LBCT": 7,
    "MPA": 2,
    "RPA": 8,
    "LPA": 9,
    "HA": 10,
    "SMA": 11,
    "LRA": 12,
    "RRA": 13,
    "PV": 14,
    "SMV": 15,
    "SV": 16,
    "LICA": 17,
    "LVA": 18,
    "RVA": 19,
    "RICA": 20,
    "BA": 21,
    "LTS": 22,
    "SSS": 23,
    "RTS": 24,
    "StrS": 25,
    "LPCA": 26,
    "RPCA": 27,
    "LMCA": 28,
    "RMCA": 29,
    "ACA": 30,
    "LIJV": 31,
    "RIJV": 32,
}

DEFAULT_SPECIAL_CONTACT_LABELS = ["RBCT", "CCA", "LBCT"]

DEFAULT_SKELETON_LABEL_GROUPS = {
    "aorta_systemic_branches": {
        "labels": ["AAO", "ARCH", "DAO", "RBCT", "CCA", "LBCT", "HA", "SMA", "LRA", "RRA"],
        "browser_color": "#c92a2a",
        "skeleton_color": "#c92a2a",
        "graph_color": "#e03131",
        "path_color": "#f76707",
        "plane_color": "#ffd43b",
        "preprocess": {"gaussian_enabled": True, "gaussian_sigma": 0.5, "dilation_iters": 0, "erosion_iters": 0, "opening_iters": 0, "closing_iters": 1},
    },
    "pulmonary_arteries": {
        "labels": ["MPA", "RPA", "LPA"],
        "browser_color": "#1971c2",
        "skeleton_color": "#1971c2",
        "graph_color": "#1c7ed6",
        "path_color": "#4dabf7",
        "plane_color": "#a5d8ff",
        "preprocess": {},
    },
    "portal_splenic_venous": {
        "labels": ["PV", "SMV", "SV"],
        "browser_color": "#2b8a3e",
        "skeleton_color": "#2b8a3e",
        "graph_color": "#37b24d",
        "path_color": "#69db7c",
        "plane_color": "#b2f2bb",
        "preprocess": {},
    },
    "carotid_arteries": {
        "labels": ["LICA", "RICA"],
        "browser_color": "#862e9c",
        "skeleton_color": "#862e9c",
        "graph_color": "#9c36b5",
        "path_color": "#be4bdb",
        "plane_color": "#e599f7",
        "preprocess": {},
    },
    "vertebral_arteries": {
        "labels": ["LVA", "RVA"],
        "browser_color": "#5f3dc4",
        "skeleton_color": "#5f3dc4",
        "graph_color": "#7048e8",
        "path_color": "#9775fa",
        "plane_color": "#d0bfff",
        "preprocess": {},
    },
    "basilar_artery": {
        "labels": ["BA"],
        "browser_color": "#364fc7",
        "skeleton_color": "#364fc7",
        "graph_color": "#4263eb",
        "path_color": "#748ffc",
        "plane_color": "#bac8ff",
        "preprocess": {},
    },
    "dural_sinuses": {
        "labels": ["StrS", "RTS", "SSS", "LTS"],
        "browser_color": "#0b7285",
        "skeleton_color": "#0b7285",
        "graph_color": "#1098ad",
        "path_color": "#3bc9db",
        "plane_color": "#99e9f2",
        "preprocess": {},
    },
    "jugular_veins": {
        "labels": ["LIJV", "RIJV"],
        "browser_color": "#495057",
        "skeleton_color": "#495057",
        "graph_color": "#868e96",
        "path_color": "#adb5bd",
        "plane_color": "#dee2e6",
        "preprocess": {},
    },
    "intracranial_arterial_branches": {
        "labels": ["LPCA", "RPCA", "LMCA", "RMCA", "ACA"],
        "browser_color": "#e67700",
        "skeleton_color": "#e67700",
        "graph_color": "#f08c00",
        "path_color": "#ffa94d",
        "plane_color": "#ffec99",
        "preprocess": {},
    },
}

DEFAULT_CONFIG_BUNDLE: Dict[str, Dict[str, Any]] = {
    "batch": {
        "output_dir": "./results",
        "skip_derived": False,
        "skip_wss": False,
        "skip_tke": False,
        "skip_pressure_gradient": False,
        "skip_plane_metrics": False,
        "use_multithread": True,
        "reuse_planes": "",
    },
    "ui": {
        "background_color": "#000000",
    },
    "loader": {
        "background_phase_correction": {
            "enabled": False,
            "method": "wrls_arto",
            "corr_fit_order": 3,
            "threshold": 0.1,
            "wrls_lambda": 5.0,
            "wrls_magnitude_threshold": 0.04,
            "wrls_mid_fov_fraction": 0.5,
            "wrls_mid_slice_fraction": 0.65,
            "wrls_arto_iterations": 2,
            "wrls_tau": 3.0,
            "wrls_delta": 2.0,
            "wrls_central_probability": 0.5,
            "wrls_fista_iterations": 5000,
            "wrls_gmm_iterations": 1000,
            "dual_venc_ratio1": 0.0,
            "dual_venc_ratio2": 0.0,
            "force_recompute": False,
            "write_cache": True,
        },
        "dicom_parameter_overrides": {},
        "dicom_read_workers": 1,
        "ignore_embedded_segmentation": False,
    },
    "phase_unwrapping": {
        "mask_source": "segmentation",
        "device": "auto",
        "tfc": True,
        "lap4d_ts": 2.0,
        "nprs_upsampling_factor": 2,
        "nprs_pi_unwrap": True,
        "nprs_auto_crop": True,
        "write_output": True,
    },
    "skeleton": {
        "remove_small_cc": True,
        "separate_special_label_contacts": True,
        "special_contact_labels": list(DEFAULT_SPECIAL_CONTACT_LABELS),
        "min_cc_volume_mm3": 50.0,
        "cc_filter_mode": "hybrid",
        "cc_rel_min_ratio": 0.01,
        "do_closing": True,
        "do_opening": False,
        "gaussian_sigma": 0.5,
        "gaussian_enabled": True,
        "dilation_iters": 0,
        "erosion_iters": 0,
        "opening_iters": 0,
        "closing_iters": 0,
    },
    "labels": {
        "single_label_group_name": "single_label",
        "single_label_browser_color": "#d9480f",
        "default_group_browser_color": "#1c7ed6",
        "label_map": dict(DEFAULT_SKELETON_LABEL_MAP),
        "label_groups": copy.deepcopy(DEFAULT_SKELETON_LABEL_GROUPS),
    },
    "planes": {
        "plane_mode": "fixed_step",
        "plane_count": 3,
        "cross_section_distance": 5.0,
        "start_distance": 0.0,
        "end_distance": 0.0,
        "anchor": "center",
        "anchor_offset_mm": 5.0,
        "direction": "both",
        "spacing_mode": "fraction",
        "spacing_ratio": 0.25,
        "segmentation_filter": True,
        "smoothing_window": 15,
        "smoothing_polyorder": 2,
        "inter_time": 10,
        "render": copy.deepcopy(DEFAULT_PLANE_RENDER_CFG),
    },
    "streamlines": {
        "seed_ratio": 0.02,
        "max_steps": 2000,
        "min_seeds": 50,
        "terminal_speed": 0.01,
        "rng_seed": 0,
        "tube_radius": 0.25,
        "render": {
            "clim": None,
            "show_scalar_bar": True,
            "bar_cfg": dict(DEFAULT_STREAMLINE_BAR_CFG),
        },
    },
    "pathlines": {
        "seed_ratio": 0.2,
        "max_steps": 200,
        "min_seeds": 50,
        "seed_mode": "fixed",
        "seed_count": 250,
        "terminal_speed": 0.01,
        "rng_seed": 0,
        "tube_radius": 0.25,
        "color": "deepskyblue",
        "color_mode": "per_plane",
        "temporal_cache_mb": 512.0,
    },
    "derived": {
    },
    "fluid": {
        "rho": 1060.0,
        "viscosity": 4.0,
    },
    "wss": {
        "smoothing_iteration": 200,
        "inward_distance": "auto",
        "parabolic_fitting": True,
        "no_slip_condition": False,
        "render": {
            "clim": [0.0, 10.0],
            "show_scalar_bar": True,
            "bar_cfg": dict(DEFAULT_WSS_BAR_CFG),
        },
    },
    "tke": {
        "render": {
            "clim": [0.0, 100.0],
            "show_scalar_bar": True,
            "bar_cfg": dict(DEFAULT_TKE_BAR_CFG),
        },
    },
    "pressure_gradient": {
        "method": "least_squares",
        "smoothing_sigma": 0.0,
        "support_erosion_iters": 1,
        "layer_opacity": 0.6,
        "relative_pressure_opacity": 0.6,
        "use_convective_acceleration": True,
        "render": {
            "clim": None,
            "show_scalar_bar": True,
            "bar_cfg": dict(DEFAULT_PRESSURE_GRADIENT_BAR_CFG),
        },
    },
    "vortex": {
        "smoothing_sigma": 0.0,
        "support_erosion_iters": 1,
    },
    "pwv": {
        "enabled": False,
        "groups": [],
        "plane_interval_mm": 10.0,
        "start_distance": 0.0,
        "end_distance": 0.0,
        "smoothing_window": 15,
        "smoothing_polyorder": 2,
        "inter_time": 10,
        "waveform_key": "flowrate_mL_s",
        "transit_time_method": "foot_to_foot",
        "foot_method": "tangent",
        "foot_savgol_window": 5,
        "foot_savgol_polyorder": 2,
        "foot_threshold_percent": 10.0,
        "xcorr_window": "full",
        "xcorr_interp_factor": 10,
        "allow_cycle_wrap": True,
        "minimum_valid_planes": 2,
        "scene_visible": True,
        "scene_color": "#ffd43b",
        "plot_color": "#2b8a3e",
        "fit_color": "#f08c00",
        "plot_dpi": 160,
    },
    "segmentation": {
        "visible": True,
        "opacity": 0.35,
        "active_label": 1,
        "editing_enabled": False,
        "tool": "brush",
        "brush_radius": 3,
        "edit_all_timepoints": True,
        "cleanup_4d_components": False,
        "cleanup_4d_mode": "absolute",
        "cleanup_4d_min_volume_mm3": 50.0,
        "mode": "input",
        "input_source": "original",
        "import_path": "",
        "threshold_scalar": "pcmra",
        "threshold_value": {
            "mode": "manual",
            "min_percent": 10.0,
            "max_percent": 100.0,
        },
        "threshold_keep_largest_cc": True,
        "threshold_min_component_volume_mm3": 0.0,
        "threshold_closing": True,
        "threshold_opening": False,
        "auto_backend": "nnUNet4D",
        "auto_model": "/nas-data2/ryy/CMR4DFlow2026/Segdata/scripts/nnunet/4D/run_7020_4d_full_ssd_20260824.sh",
        "auto_checkpoint": "checkpoint_final.pth",
        "auto_folds": "single",
        "auto_device": "auto",
        "auto_label_map": "",
    },
    "colorbar": copy.deepcopy(DEFAULT_SHARED_COLORBAR_CFG),
    "video_exporting": {
        "fps": 12,
        "plane_rotation_frames": 180,
        "make_plane_video": False,
        "make_wss_video": False,
        "make_pressure_gradient_video": False,
        "make_streamlines_video": False,
        "make_tke_video": False,
        "camera_view": "right",
        "camera_distance_scale": 1.5,
        "rotate_dynamic_video": True,
        "dynamic_rotation_frames": 180,
        "dynamic_rotation_elevation_deg": 10.0,
        "dynamic_time_repeat": 3,
        "add_plane_idx": True,
        "add_path_idx": False,
        "window_size": [1600, 1200],
        "plane_video": copy.deepcopy(DEFAULT_PLANE_VIDEO_CFG),
    },
}

CONFIG_MODULES = tuple(DEFAULT_CONFIG_BUNDLE.keys())


def default_config_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "configs"


def resolve_config_dir(config_dir: Optional[str] = None, *, require_exists: bool = False) -> Path:
    if config_dir:
        path = Path(config_dir).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        path = path.resolve()
        if require_exists and not path.exists():
            raise FileNotFoundError(f"config directory not found: {path}")
        return path
    return default_config_dir()


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = copy.deepcopy(base)
        for key, value in override.items():
            merged[key] = _deep_merge(merged[key], value) if key in merged else copy.deepcopy(value)
        return merged
    return copy.deepcopy(override)


def load_config_module(module_name: str, config_dir: Optional[str] = None) -> Dict[str, Any]:
    if module_name not in DEFAULT_CONFIG_BUNDLE:
        raise KeyError(f"unknown config module: {module_name}")
    config_root = resolve_config_dir(config_dir, require_exists=bool(config_dir))
    config_path = config_root / f"{module_name}.json"
    defaults = copy.deepcopy(DEFAULT_CONFIG_BUNDLE[module_name])
    if not config_path.exists():
        if module_name == "video_exporting":
            legacy_path = config_root / "rendering.json"
            if legacy_path.exists():
                config_path = legacy_path
            else:
                return defaults
        else:
            return defaults
    with config_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"config file must contain a JSON object: {config_path}")
    return _deep_merge(defaults, payload)


def load_config_bundle(config_dir: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    bundle = {name: load_config_module(name, config_dir=config_dir) for name in CONFIG_MODULES}
    # Older user config directories put pathline values in streamlines.json.
    # Preserve those values when no dedicated pathlines.json exists yet.
    if config_dir:
        config_root = resolve_config_dir(config_dir, require_exists=True)
        if not (config_root / "pathlines.json").exists():
            legacy = bundle.get("streamlines", {})
            pathlines = bundle.setdefault("pathlines", {})
            legacy_map = {
                "pathline_seed_ratio": "seed_ratio",
                "pathline_seed_mode": "seed_mode",
                "pathline_max_seeds": "seed_count",
                "pathline_max_steps": "max_steps",
                "pathline_min_seeds": "min_seeds",
                "pathline_terminal_speed": "terminal_speed",
                "pathline_rng_seed": "rng_seed",
                "pathline_tube_radius": "tube_radius",
                "pathline_color": "color",
                "pathline_color_mode": "color_mode",
                "pathline_temporal_cache_mb": "temporal_cache_mb",
            }
            for old_key, new_key in legacy_map.items():
                if old_key in legacy:
                    pathlines[new_key] = copy.deepcopy(legacy[old_key])
            for old_key, new_key in (("seed_ratio", "seed_ratio"), ("max_steps", "max_steps"), ("min_seeds", "min_seeds"), ("terminal_speed", "terminal_speed"), ("rng_seed", "rng_seed"), ("tube_radius", "tube_radius")):
                if old_key in legacy:
                    pathlines[new_key] = copy.deepcopy(legacy[old_key])
    return bundle


def _pathline_model_payload(config_bundle: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    path_cfg = dict(config_bundle.get("pathlines", {}) or {})
    legacy = dict(config_bundle.get("streamlines", {}) or {})

    def pick(new_key, legacy_key, default):
        if new_key in path_cfg:
            return path_cfg[new_key]
        if legacy_key in legacy:
            return legacy[legacy_key]
        return default

    return {
        "pathline_seed_ratio": pick("seed_ratio", "pathline_seed_ratio", 0.2),
        "pathline_max_steps": pick("max_steps", "pathline_max_steps", 200),
        "pathline_min_seeds": pick("min_seeds", "pathline_min_seeds", 50),
        "pathline_seed_mode": pick("seed_mode", "pathline_seed_mode", "fixed"),
        "pathline_max_seeds": pick("seed_count", "pathline_max_seeds", 250),
        "pathline_terminal_speed": pick("terminal_speed", "pathline_terminal_speed", 0.01),
        "pathline_rng_seed": pick("rng_seed", "pathline_rng_seed", 0),
        "pathline_tube_radius": pick("tube_radius", "pathline_tube_radius", 0.25),
        "pathline_color": pick("color", "pathline_color", "deepskyblue"),
        "pathline_color_mode": pick("color_mode", "pathline_color_mode", "per_plane"),
        "pathline_temporal_cache_mb": pick("temporal_cache_mb", "pathline_temporal_cache_mb", 512.0),
    }


def _coerce_render_clim(value: Any) -> Optional[tuple[float, float]]:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        return None
    if len(value) < 2:
        return None
    return (float(value[0]), float(value[1]))


def _coerce_window_size(value: Any) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return (1600, 1200)
    return (max(int(value[0]), 1), max(int(value[1]), 1))


def _feature_render_cfg(module_cfg: Dict[str, Any]) -> Dict[str, Any]:
    render_cfg = module_cfg.get("render", {}) if isinstance(module_cfg, dict) else {}
    return dict(render_cfg) if isinstance(render_cfg, dict) else {}


def _plane_render_cfg(render_cfg: Dict[str, Any]) -> Dict[str, Any]:
    resolved = copy.deepcopy(DEFAULT_PLANE_RENDER_CFG)
    cfg = render_cfg if isinstance(render_cfg, dict) else {}
    default_cfg = cfg.get("default", {}) if isinstance(cfg.get("default"), dict) else {}
    group_cfgs = cfg.get("groups", {}) if isinstance(cfg.get("groups"), dict) else {}

    for key in ("plane_color", "plane_opacity"):
        if key in default_cfg:
            resolved["default"][key] = copy.deepcopy(default_cfg[key])
    for group_name, raw_group_cfg in group_cfgs.items():
        if not isinstance(raw_group_cfg, dict):
            continue
        group_cfg = {}
        for key in ("plane_color", "plane_opacity"):
            if key in raw_group_cfg:
                group_cfg[key] = copy.deepcopy(raw_group_cfg[key])
        if group_cfg:
            resolved["groups"][str(group_name)] = group_cfg
    return resolved


def _legacy_plane_video_cfg(render_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = render_cfg if isinstance(render_cfg, dict) else {}
    legacy = {}
    if "show_skeleton" in cfg:
        legacy["show_skeleton"] = copy.deepcopy(cfg.get("show_skeleton"))
    if "skeleton_point_size" in cfg:
        legacy["skeleton_point_size"] = copy.deepcopy(cfg.get("skeleton_point_size"))
    if isinstance(cfg.get("label"), dict):
        legacy["label"] = copy.deepcopy(cfg.get("label"))
    default_cfg = cfg.get("default", {}) if isinstance(cfg.get("default"), dict) else {}
    legacy_default = {}
    for key in ("skeleton_color", "plane_size", "plane_color", "plane_opacity"):
        if key in default_cfg:
            legacy_default[key] = copy.deepcopy(default_cfg[key])
    if legacy_default:
        legacy["default"] = legacy_default
    group_cfgs = cfg.get("groups", {}) if isinstance(cfg.get("groups"), dict) else {}
    legacy_groups = {}
    for group_name, raw_group_cfg in group_cfgs.items():
        if not isinstance(raw_group_cfg, dict):
            continue
        group_cfg = {}
        for key in ("skeleton_color", "plane_size", "plane_color", "plane_opacity"):
            if key in raw_group_cfg:
                group_cfg[key] = copy.deepcopy(raw_group_cfg[key])
        if group_cfg:
            legacy_groups[str(group_name)] = group_cfg
    if legacy_groups:
        legacy["groups"] = legacy_groups
    return legacy


def _resolve_clim(primary: Any, fallback: Any, default: Optional[tuple[float, float]]) -> Optional[tuple[float, float]]:
    value = _coerce_render_clim(primary)
    if value is not None:
        return value
    value = _coerce_render_clim(fallback)
    if value is not None:
        return value
    return default


def _resolve_bar_cfg(primary: Any, fallback: Any, default: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(default)
    if isinstance(fallback, dict):
        cfg.update(copy.deepcopy(fallback))
    if isinstance(primary, dict):
        cfg.update(copy.deepcopy(primary))
    return cfg


def resolve_render_settings(config_bundle: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    video_exporting_cfg = dict(config_bundle.get("video_exporting", config_bundle.get("rendering", {})))
    plane_cfg = dict(config_bundle.get("planes", {}))
    wss_cfg = dict(config_bundle.get("wss", {}))
    tke_cfg = dict(config_bundle.get("tke", {}))
    pressure_gradient_cfg = dict(config_bundle.get("pressure_gradient", {}))
    vortex_cfg = dict(config_bundle.get("vortex", {}))
    streamline_cfg = dict(config_bundle.get("streamlines", {}))
    colorbar_cfg = dict(config_bundle.get("colorbar", {}))

    raw_plane_render_cfg = _feature_render_cfg(plane_cfg)
    plane_render_cfg = _plane_render_cfg(raw_plane_render_cfg)
    wss_render_cfg = _feature_render_cfg(wss_cfg)
    tke_render_cfg = _feature_render_cfg(tke_cfg)
    pressure_gradient_render_cfg = _feature_render_cfg(pressure_gradient_cfg)
    streamline_render_cfg = _feature_render_cfg(streamline_cfg)
    shared_colorbar_bar_cfg = _resolve_bar_cfg(colorbar_cfg.get("bar_cfg", None), video_exporting_cfg.get("shared_colorbar_bar_cfg", None), DEFAULT_SHARED_COLORBAR_CFG["bar_cfg"])
    shared_colorbar_show = bool(colorbar_cfg.get("show", video_exporting_cfg.get("shared_colorbar_show", True)))

    plane_video_cfg = copy.deepcopy(DEFAULT_PLANE_VIDEO_CFG)
    configured_plane_video_cfg = video_exporting_cfg.get("plane_video", {})
    if isinstance(configured_plane_video_cfg, dict) and configured_plane_video_cfg:
        plane_video_cfg = _deep_merge(plane_video_cfg, configured_plane_video_cfg)
    else:
        legacy_plane_video_cfg = _legacy_plane_video_cfg(raw_plane_render_cfg)
        if legacy_plane_video_cfg:
            plane_video_cfg = _deep_merge(plane_video_cfg, legacy_plane_video_cfg)

    return {
        "fps": int(video_exporting_cfg.get("fps", 12)),
        "plane_rotation_frames": int(video_exporting_cfg.get("plane_rotation_frames", 180)),
        "make_plane_video": bool(video_exporting_cfg.get("make_plane_video", False)),
        "make_wss_video": bool(video_exporting_cfg.get("make_wss_video", False)),
        "make_pressure_gradient_video": bool(video_exporting_cfg.get("make_pressure_gradient_video", False)),
        "make_streamlines_video": bool(video_exporting_cfg.get("make_streamlines_video", False)),
        "make_tke_video": bool(video_exporting_cfg.get("make_tke_video", False)),
        "camera_view": str(video_exporting_cfg.get("camera_view", "right")),
        "camera_distance_scale": float(video_exporting_cfg.get("camera_distance_scale", 1.5)),
        "rotate_dynamic_video": bool(video_exporting_cfg.get("rotate_dynamic_video", True)),
        "dynamic_rotation_frames": int(video_exporting_cfg.get("dynamic_rotation_frames", 180)),
        "dynamic_rotation_elevation_deg": video_exporting_cfg.get("dynamic_rotation_elevation_deg", 10.0),
        "dynamic_time_repeat": int(video_exporting_cfg.get("dynamic_time_repeat", 3)),
        "add_plane_idx": bool(video_exporting_cfg.get("add_plane_idx", True)),
        "add_path_idx": bool(video_exporting_cfg.get("add_path_idx", False)),
        "plane_render_cfg": plane_render_cfg,
        "plane_video_cfg": plane_video_cfg,
        "window_size": _coerce_window_size(video_exporting_cfg.get("window_size", video_exporting_cfg.get("figsize", [1600, 1200]))),
        "shared_colorbar_show": shared_colorbar_show,
        "shared_colorbar_bar_cfg": shared_colorbar_bar_cfg,
        "wss_clim": _resolve_clim(wss_render_cfg.get("clim", None), video_exporting_cfg.get("wss_clim", None), (0.0, 10.0)),
        "wss_show_scalar_bar": bool(wss_render_cfg.get("show_scalar_bar", video_exporting_cfg.get("wss_show_scalar_bar", True))),
        "wss_bar_cfg": _resolve_bar_cfg(wss_render_cfg.get("bar_cfg", None), video_exporting_cfg.get("wss_bar_cfg", None), DEFAULT_WSS_BAR_CFG),
        "tke_clim": _resolve_clim(tke_render_cfg.get("clim", None), video_exporting_cfg.get("tke_clim", None), (0.0, 100.0)),
        "tke_show_scalar_bar": bool(tke_render_cfg.get("show_scalar_bar", video_exporting_cfg.get("tke_show_scalar_bar", True))),
        "tke_bar_cfg": _resolve_bar_cfg(tke_render_cfg.get("bar_cfg", None), video_exporting_cfg.get("tke_bar_cfg", None), DEFAULT_TKE_BAR_CFG),
        "pressure_gradient_clim": _resolve_clim(pressure_gradient_render_cfg.get("clim", None), video_exporting_cfg.get("pressure_gradient_clim", None), None),
        "pressure_gradient_show_scalar_bar": bool(pressure_gradient_render_cfg.get("show_scalar_bar", video_exporting_cfg.get("pressure_gradient_show_scalar_bar", True))),
        "pressure_gradient_bar_cfg": _resolve_bar_cfg(pressure_gradient_render_cfg.get("bar_cfg", None), video_exporting_cfg.get("pressure_gradient_bar_cfg", None), DEFAULT_PRESSURE_GRADIENT_BAR_CFG),
        "relative_pressure_clim": _resolve_clim(pressure_gradient_render_cfg.get("relative_pressure_clim", None), video_exporting_cfg.get("relative_pressure_clim", None), None),
        "relative_pressure_show_scalar_bar": bool(pressure_gradient_render_cfg.get("relative_pressure_show_scalar_bar", video_exporting_cfg.get("show_relative_pressure_scalar_bar", video_exporting_cfg.get("pressure_gradient_show_scalar_bar", True)))),
        "relative_pressure_bar_cfg": _resolve_bar_cfg(pressure_gradient_render_cfg.get("relative_pressure_bar_cfg", None), video_exporting_cfg.get("relative_pressure_bar_cfg", None), DEFAULT_RELATIVE_PRESSURE_BAR_CFG),
        "streamline_clim": _resolve_clim(streamline_render_cfg.get("clim", None), video_exporting_cfg.get("streamline_clim", None), None),
        "streamline_show_scalar_bar": bool(streamline_render_cfg.get("show_scalar_bar", video_exporting_cfg.get("streamline_show_scalar_bar", True))),
        "streamline_bar_cfg": _resolve_bar_cfg(streamline_render_cfg.get("bar_cfg", None), video_exporting_cfg.get("streamline_bar_cfg", None), DEFAULT_STREAMLINE_BAR_CFG),
    }


def _build_derived_metrics_config(config_bundle: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    legacy_cfg = dict(config_bundle.get("derived", {}))
    fluid_cfg = dict(config_bundle.get("fluid", {}))
    wss_cfg = dict(config_bundle.get("wss", {}))
    tke_cfg = dict(config_bundle.get("tke", {}))
    pressure_gradient_cfg = dict(config_bundle.get("pressure_gradient", {}))
    vortex_cfg = dict(config_bundle.get("vortex", {}))
    shared_viscosity = float(fluid_cfg.get("viscosity", legacy_cfg.get("viscosity", 4.0)))
    shared_rho = float(fluid_cfg.get("rho", legacy_cfg.get("rho", 1060.0)))
    return {
        "wss_smoothing_iteration": int(wss_cfg.get("smoothing_iteration", legacy_cfg.get("wss_smoothing_iteration", legacy_cfg.get("smoothing_iteration", 200)))),
        "wss_viscosity": float(wss_cfg.get("viscosity", legacy_cfg.get("wss_viscosity", shared_viscosity))),
        "wss_inward_distance": wss_cfg.get("inward_distance", legacy_cfg.get("wss_inward_distance", legacy_cfg.get("inward_distance", "auto"))),
        "wss_parabolic_fitting": bool(wss_cfg.get("parabolic_fitting", legacy_cfg.get("wss_parabolic_fitting", legacy_cfg.get("parabolic_fitting", True)))),
        "wss_no_slip_condition": bool(wss_cfg.get("no_slip_condition", legacy_cfg.get("wss_no_slip_condition", legacy_cfg.get("no_slip_condition", False)))),
        "tke_rho": float(tke_cfg.get("rho", legacy_cfg.get("tke_rho", shared_rho))),
        "pressure_gradient_rho": float(pressure_gradient_cfg.get("rho", legacy_cfg.get("pressure_gradient_rho", shared_rho))),
        "pressure_gradient_viscosity": float(pressure_gradient_cfg.get("viscosity", legacy_cfg.get("pressure_gradient_viscosity", shared_viscosity))),
        "pressure_gradient_smoothing_sigma": float(pressure_gradient_cfg.get("smoothing_sigma", legacy_cfg.get("pressure_gradient_smoothing_sigma", 0.0))),
        "pressure_gradient_support_erosion_iters": int(pressure_gradient_cfg.get("support_erosion_iters", legacy_cfg.get("pressure_gradient_support_erosion_iters", 1))),
        "pressure_gradient_layer_opacity": float(pressure_gradient_cfg.get("layer_opacity", legacy_cfg.get("pressure_gradient_layer_opacity", 0.6))),
        "relative_pressure_layer_opacity": float(pressure_gradient_cfg.get("relative_pressure_opacity", legacy_cfg.get("relative_pressure_layer_opacity", 0.6))),
        "pressure_gradient_use_convective_acceleration": bool(pressure_gradient_cfg.get("use_convective_acceleration", legacy_cfg.get("pressure_gradient_use_convective_acceleration", True))),
        "pressure_method": str(pressure_gradient_cfg.get("method", legacy_cfg.get("pressure_method", legacy_cfg.get("pressure_gradient_method", "least_squares"))) or "least_squares"),
        "vortex_smoothing_sigma": max(float(vortex_cfg.get("smoothing_sigma", legacy_cfg.get("vortex_smoothing_sigma", 0.0))), 0.0),
        "vortex_support_erosion_iters": max(int(vortex_cfg.get("support_erosion_iters", legacy_cfg.get("vortex_support_erosion_iters", 1))), 0),
        "step_size": int(legacy_cfg.get("step_size", 5)),
        "tube_radius": float(legacy_cfg.get("tube_radius", 0.1)),
    }


def apply_config_bundle_to_workspace(workspace: Workspace, config_bundle: Dict[str, Dict[str, Any]]) -> Workspace:
    skeleton_cfg = config_bundle.get("skeleton", {})
    labels_cfg = dict(config_bundle.get("labels", {}))
    for key in [
        "label_map",
        "label_groups",
        "single_label_group_name",
        "single_label_browser_color",
        "default_group_browser_color",
    ]:
        if key not in labels_cfg and key in skeleton_cfg:
            labels_cfg[key] = copy.deepcopy(skeleton_cfg[key])
    workspace.loader_params = LoaderParams.from_dict(config_bundle.get("loader", {}))
    from .case_types import PhaseUnwrappingConfig
    workspace.phase_unwrap_params = PhaseUnwrappingConfig.from_dict(config_bundle.get("phase_unwrapping", {}))
    workspace.skeleton_params = SkeletonParams.from_dict(skeleton_cfg)
    workspace.label_params = LabelParams.from_dict(labels_cfg)
    workspace.skeleton_params.label_map = copy.deepcopy(workspace.label_params.label_map)
    workspace.skeleton_params.label_groups = copy.deepcopy(workspace.label_params.label_groups)
    workspace.skeleton_params.single_label_group_name = str(workspace.label_params.single_label_group_name)
    workspace.skeleton_params.single_label_browser_color = str(workspace.label_params.single_label_browser_color)
    workspace.skeleton_params.default_group_browser_color = str(workspace.label_params.default_group_browser_color)
    workspace.plane_gen_params = PlaneGenerationParams.from_dict(config_bundle.get("planes", {}))
    streamline_payload = dict(config_bundle.get("streamlines", {}) or {})
    streamline_payload.update(_pathline_model_payload(config_bundle))
    workspace.streamline_params = StreamlineParams.from_dict(streamline_payload)
    workspace.derived_params = DerivedMetricsParams.from_dict(_build_derived_metrics_config(config_bundle))
    workspace.pwv_params = PwvParams.from_dict(config_bundle.get("pwv", {}), label_map=workspace.label_params.label_map)
    workspace.render_settings = resolve_render_settings(config_bundle)
    workspace.derived_params.use_multithread = bool(
        config_bundle.get("batch", {}).get("use_multithread", workspace.derived_params.use_multithread)
    )
    workspace.segmentation = SegmentationState.from_dict(config_bundle.get("segmentation", {}))
    return workspace


def bundle_to_autoflow_kwargs(config_bundle: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    batch_cfg = config_bundle.get("batch", {})
    loader_cfg = config_bundle.get("loader", {})
    bpc_cfg = loader_cfg.get("background_phase_correction", {})
    unwrap_cfg = config_bundle.get("phase_unwrapping", {}) or {}
    plane_cfg = config_bundle.get("planes", {})
    skeleton_cfg = config_bundle.get("skeleton", {})
    streamline_cfg = config_bundle.get("streamlines", {})
    pathline_cfg = _pathline_model_payload(config_bundle)
    render_settings = copy.deepcopy(resolve_render_settings(config_bundle))
    render_settings.pop("plane_render_cfg", None)
    derived_cfg = _build_derived_metrics_config(config_bundle)
    plane_count = int(plane_cfg.get("plane_count", 1) or 1)
    if plane_count != -1:
        plane_count = max(1, plane_count)
    return {
        "output_dir": str(batch_cfg.get("output_dir", "./results")),
        "skip_derived": bool(batch_cfg.get("skip_derived", False)),
        "skip_wss": bool(batch_cfg.get("skip_wss", False)),
        "skip_tke": bool(batch_cfg.get("skip_tke", False)),
        "skip_pressure_gradient": bool(batch_cfg.get("skip_pressure_gradient", False)),
        "skip_plane_metrics": bool(batch_cfg.get("skip_plane_metrics", False)),
        "use_multithread": bool(batch_cfg.get("use_multithread", True)),
        "reuse_planes": str(batch_cfg.get("reuse_planes", "")),
        "background_phase_correction": bool(bpc_cfg.get("enabled", False)),
        "background_phase_method": str(bpc_cfg.get("method", "wrls_arto") or "wrls_arto"),
        "background_phase_corr_fit_order": int(bpc_cfg.get("corr_fit_order", 3)),
        "background_phase_threshold": float(bpc_cfg.get("threshold", 0.1)),
        "background_phase_wrls_lambda": float(bpc_cfg.get("wrls_lambda", 5.0)),
        "background_phase_wrls_magnitude_threshold": float(bpc_cfg.get("wrls_magnitude_threshold", 0.04)),
        "background_phase_wrls_mid_fov_fraction": float(bpc_cfg.get("wrls_mid_fov_fraction", 0.5)),
        "background_phase_wrls_mid_slice_fraction": float(bpc_cfg.get("wrls_mid_slice_fraction", 0.65)),
        "background_phase_wrls_arto_iterations": int(bpc_cfg.get("wrls_arto_iterations", 2)),
        "background_phase_wrls_tau": float(bpc_cfg.get("wrls_tau", 3.0)),
        "background_phase_wrls_delta": float(bpc_cfg.get("wrls_delta", 2.0)),
        "background_phase_wrls_central_probability": float(bpc_cfg.get("wrls_central_probability", 0.5)),
        "background_phase_wrls_fista_iterations": int(bpc_cfg.get("wrls_fista_iterations", 5000)),
        "background_phase_wrls_gmm_iterations": int(bpc_cfg.get("wrls_gmm_iterations", 1000)),
        "dual_venc_ratio1": float(bpc_cfg.get("dual_venc_ratio1", 0.0)),
        "dual_venc_ratio2": float(bpc_cfg.get("dual_venc_ratio2", 0.0)),
        "force_recompute_corr": bool(bpc_cfg.get("force_recompute", False)),
        "background_phase_write_cache": bool(bpc_cfg.get("write_cache", True)),
        "phase_unwrap_enabled": bool(unwrap_cfg.get("enabled", False)),
        "phase_unwrap_method": str(unwrap_cfg.get("method", "none") or "none"),
        "phase_unwrap_mask": str(unwrap_cfg.get("mask_source", "segmentation") or "segmentation"),
        "phase_unwrap_device": str(unwrap_cfg.get("device", "auto") or "auto"),
        "phase_unwrap_tfc": bool(unwrap_cfg.get("tfc", True)),
        "phase_unwrap_lap4d_ts": float(unwrap_cfg.get("lap4d_ts", 2.0)),
        "phase_unwrap_nprs_upsampling_factor": int(unwrap_cfg.get("nprs_upsampling_factor", 2)),
        "phase_unwrap_nprs_pi_unwrap": bool(unwrap_cfg.get("nprs_pi_unwrap", True)),
        "phase_unwrap_nprs_auto_crop": bool(unwrap_cfg.get("nprs_auto_crop", True)),
        "dicom_read_workers": int(loader_cfg.get("dicom_read_workers", 1)),
        "ignore_embedded_segmentation": bool(loader_cfg.get("ignore_embedded_segmentation", False)),
        "plane_mode": str(plane_cfg.get("plane_mode", "fixed_step") or "fixed_step"),
        "plane_count": plane_count,
        "cross_section_dist": float(plane_cfg.get("cross_section_distance", 5.0)),
        "start_dist": float(plane_cfg.get("start_distance", 0.0)),
        "end_dist": float(plane_cfg.get("end_distance", 0.0)),
        "plane_anchor": str(plane_cfg.get("anchor", "center") or "center"),
        "plane_offset_mm": float(plane_cfg.get("anchor_offset_mm", 5.0)),
        "plane_direction": str(plane_cfg.get("direction", "both") or "both"),
        "plane_spacing_mode": str(plane_cfg.get("spacing_mode", "fraction") or "fraction"),
        "plane_spacing_ratio": float(plane_cfg.get("spacing_ratio", 0.25)),
        "segmentation_filter": bool(plane_cfg.get("segmentation_filter", True)),
        "use_center_plane": plane_cfg.get("use_center_plane", None),
        "remove_small_cc": bool(skeleton_cfg.get("remove_small_cc", True)),
        "separate_special_label_contacts": bool(skeleton_cfg.get("separate_special_label_contacts", skeleton_cfg.get("separate_label_contacts", True))),
        "special_contact_labels": copy.deepcopy(skeleton_cfg.get("special_contact_labels", DEFAULT_SPECIAL_CONTACT_LABELS)),
        "min_cc_volume": float(skeleton_cfg.get("min_cc_volume_mm3", 50.0)),
        "cc_filter_mode": str(skeleton_cfg.get("cc_filter_mode", "hybrid") or "hybrid"),
        "cc_rel_min_ratio": float(skeleton_cfg.get("cc_rel_min_ratio", 0.01)),
        "seed_ratio": float(streamline_cfg.get("seed_ratio", 0.02)),
        "max_steps": int(streamline_cfg.get("max_steps", 2000)),
        "min_seeds": int(streamline_cfg.get("min_seeds", 50)),
        "terminal_speed": float(streamline_cfg.get("terminal_speed", 0.01)),
        "rng_seed": int(streamline_cfg.get("rng_seed", 0)),
        "tube_radius": float(streamline_cfg.get("tube_radius", 0.25)),
        "pathline_seed_mode": ("ratio" if str(pathline_cfg["pathline_seed_mode"] or "fixed").strip().lower() == "ratio" else "fixed"),
        "pathline_max_seeds": max(1, int(pathline_cfg["pathline_max_seeds"])),
        "pathline_seed_ratio": float(pathline_cfg["pathline_seed_ratio"]),
        "pathline_max_steps": int(pathline_cfg["pathline_max_steps"]),
        "pathline_min_seeds": int(pathline_cfg["pathline_min_seeds"]),
        "pathline_terminal_speed": float(pathline_cfg["pathline_terminal_speed"]),
        "pathline_rng_seed": int(pathline_cfg["pathline_rng_seed"]),
        "pathline_tube_radius": float(pathline_cfg["pathline_tube_radius"]),
        "pathline_color": str(pathline_cfg["pathline_color"] or "deepskyblue"),
        "pathline_color_mode": (str(pathline_cfg["pathline_color_mode"] or "per_plane").strip().lower() if str(pathline_cfg["pathline_color_mode"] or "per_plane").strip().lower() in {"uniform", "per_plane", "per_group"} else "per_plane"),
        "pathline_temporal_cache_mb": max(0.0, float(pathline_cfg["pathline_temporal_cache_mb"])),
        "pressure_method": str(derived_cfg.get("pressure_method", "least_squares") or "least_squares"),
        "force_recompute_seg": bool(config_bundle.get("segmentation", {}).get("force_recompute_auto_cache", False)),
        "write_segmentation_cache": bool(config_bundle.get("segmentation", {}).get("write_auto_cache", True)),
        "autoseg_backend": str(config_bundle.get("segmentation", {}).get("auto_backend", "nnUNet") or "nnUNet"),
        "autoseg_model": str(config_bundle.get("segmentation", {}).get("auto_model", "") or ""),
        "autoseg_checkpoint": str(config_bundle.get("segmentation", {}).get("auto_checkpoint", "checkpoint_final.pth") or "checkpoint_final.pth"),
        "autoseg_folds": str(config_bundle.get("segmentation", {}).get("auto_folds", "single") or "single"),
        "autoseg_device": str(config_bundle.get("segmentation", {}).get("auto_device", "auto") or "auto"),
        "autoseg_label_map": str(config_bundle.get("segmentation", {}).get("auto_label_map", "") or ""),
        **copy.deepcopy(render_settings),
    }
