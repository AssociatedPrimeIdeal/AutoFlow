import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional

from .core.models import (
    DerivedMetricsParams,
    LoaderParams,
    PlaneGenerationParams,
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
    "loader": {
        "background_phase_correction": {
            "enabled": False,
            "corr_fit_order": 3,
            "threshold": 0.1,
            "dual_venc_ratio1": 0.0,
            "dual_venc_ratio2": 0.0,
        },
        "dicom_parameter_overrides": {},
        "dicom_read_workers": 1,
    },
    "skeleton": {
        "remove_small_cc": True,
        "min_cc_volume_mm3": 50.0,
        "do_closing": True,
        "do_opening": False,
        "gaussian_sigma": 0.5,
        "gaussian_enabled": True,
        "dilation_iters": 0,
        "erosion_iters": 0,
        "opening_iters": 0,
        "closing_iters": 0,
        "single_label_group_name": "single_label",
        "single_label_browser_color": "#d9480f",
        "default_group_browser_color": "#1c7ed6",
        "label_map": dict(DEFAULT_SKELETON_LABEL_MAP),
        "label_groups": copy.deepcopy(DEFAULT_SKELETON_LABEL_GROUPS),
    },
    "planes": {
        "use_center_plane": True,
        "cross_section_distance": 5.0,
        "start_distance": 5.0,
        "end_distance": 0.0,
        "smoothing_window": 15,
        "smoothing_polyorder": 2,
        "inter_time": 10,
    },
    "streamlines": {
        "seed_ratio": 0.02,
        "max_steps": 2000,
        "min_seeds": 50,
        "terminal_speed": 0.01,
        "rng_seed": 0,
        "tube_radius": 0.05,
        "pathline_color": "deepskyblue",
    },
    "derived": {
        "smoothing_iteration": 200,
        "viscosity": 4.0,
        "inward_distance": "auto",
        "parabolic_fitting": True,
        "no_slip_condition": False,
        "step_size": 5,
        "tube_radius": 0.1,
        "rho": 1060.0,
        "pressure_gradient_smoothing_sigma": 0.0,
        "pressure_gradient_use_convective_acceleration": True,
    },
    "segmentation": {
        "visible": True,
        "opacity": 0.35,
        "active_label": 1,
        "editing_enabled": False,
        "tool": "brush",
        "brush_radius": 3,
        "edit_all_timepoints": True,
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
        "auto_backend": "nnUNet",
        "auto_model": "autoflow/segmodel/nnUNetTrainer_500epochs__nnUNetPlans__3d_fullres_iso1mm",
        "auto_checkpoint": "checkpoint_final.pth",
        "auto_device": "auto",
        "auto_label_map": "",
    },
    "rendering": {
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
        "add_plane_idx": False,
        "add_path_idx": False,
        "wss_clim": [0.0, 10.0],
        "wss_bar_cfg": dict(DEFAULT_WSS_BAR_CFG),
        "tke_clim": [0.0, 100.0],
        "tke_bar_cfg": dict(DEFAULT_TKE_BAR_CFG),
        "streamline_clim": [0.0, 1.0],
        "streamline_bar_cfg": dict(DEFAULT_STREAMLINE_BAR_CFG),
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
    config_path = resolve_config_dir(config_dir, require_exists=bool(config_dir)) / f"{module_name}.json"
    defaults = copy.deepcopy(DEFAULT_CONFIG_BUNDLE[module_name])
    if not config_path.exists():
        return defaults
    with config_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"config file must contain a JSON object: {config_path}")
    return _deep_merge(defaults, payload)


def load_config_bundle(config_dir: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    return {name: load_config_module(name, config_dir=config_dir) for name in CONFIG_MODULES}


def apply_config_bundle_to_workspace(workspace: Workspace, config_bundle: Dict[str, Dict[str, Any]]) -> Workspace:
    workspace.loader_params = LoaderParams.from_dict(config_bundle.get("loader", {}))
    workspace.skeleton_params = SkeletonParams.from_dict(config_bundle.get("skeleton", {}))
    workspace.plane_gen_params = PlaneGenerationParams.from_dict(config_bundle.get("planes", {}))
    workspace.streamline_params = StreamlineParams.from_dict(config_bundle.get("streamlines", {}))
    workspace.derived_params = DerivedMetricsParams.from_dict(config_bundle.get("derived", {}))
    workspace.derived_params.use_multithread = bool(
        config_bundle.get("batch", {}).get("use_multithread", workspace.derived_params.use_multithread)
    )
    workspace.segmentation = SegmentationState.from_dict(config_bundle.get("segmentation", {}))
    return workspace


def bundle_to_autoflow_kwargs(config_bundle: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    batch_cfg = config_bundle.get("batch", {})
    loader_cfg = config_bundle.get("loader", {})
    bpc_cfg = loader_cfg.get("background_phase_correction", {})
    plane_cfg = config_bundle.get("planes", {})
    skeleton_cfg = config_bundle.get("skeleton", {})
    streamline_cfg = config_bundle.get("streamlines", {})
    rendering_cfg = config_bundle.get("rendering", {})
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
        "background_phase_corr_fit_order": int(bpc_cfg.get("corr_fit_order", 3)),
        "background_phase_threshold": float(bpc_cfg.get("threshold", 0.1)),
        "dual_venc_ratio1": float(bpc_cfg.get("dual_venc_ratio1", 0.0)),
        "dual_venc_ratio2": float(bpc_cfg.get("dual_venc_ratio2", 0.0)),
        "dicom_read_workers": int(loader_cfg.get("dicom_read_workers", 1)),
        "use_center_plane": bool(plane_cfg.get("use_center_plane", True)),
        "cross_section_dist": float(plane_cfg.get("cross_section_distance", 5.0)),
        "start_dist": float(plane_cfg.get("start_distance", 5.0)),
        "end_dist": float(plane_cfg.get("end_distance", 0.0)),
        "remove_small_cc": bool(skeleton_cfg.get("remove_small_cc", True)),
        "min_cc_volume": float(skeleton_cfg.get("min_cc_volume_mm3", 50.0)),
        "seed_ratio": float(streamline_cfg.get("seed_ratio", 0.02)),
        "max_steps": int(streamline_cfg.get("max_steps", 2000)),
        "min_seeds": int(streamline_cfg.get("min_seeds", 50)),
        "terminal_speed": float(streamline_cfg.get("terminal_speed", 0.01)),
        "rng_seed": int(streamline_cfg.get("rng_seed", 0)),
        "tube_radius": float(streamline_cfg.get("tube_radius", 0.05)),
        "pathline_color": str(streamline_cfg.get("pathline_color", streamline_cfg.get("plane_pathline_color", "deepskyblue")) or "deepskyblue"),
        "fps": int(rendering_cfg.get("fps", 12)),
        "plane_rotation_frames": int(rendering_cfg.get("plane_rotation_frames", 180)),
        "make_plane_video": bool(rendering_cfg.get("make_plane_video", False)),
        "make_wss_video": bool(rendering_cfg.get("make_wss_video", False)),
        "make_pressure_gradient_video": bool(rendering_cfg.get("make_pressure_gradient_video", False)),
        "make_streamlines_video": bool(rendering_cfg.get("make_streamlines_video", False)),
        "make_tke_video": bool(rendering_cfg.get("make_tke_video", False)),
        "camera_view": str(rendering_cfg.get("camera_view", "right")),
        "camera_distance_scale": float(rendering_cfg.get("camera_distance_scale", 1.5)),
        "rotate_dynamic_video": bool(rendering_cfg.get("rotate_dynamic_video", True)),
        "dynamic_rotation_frames": int(rendering_cfg.get("dynamic_rotation_frames", 180)),
        "dynamic_rotation_elevation_deg": rendering_cfg.get("dynamic_rotation_elevation_deg", 10.0),
        "dynamic_time_repeat": int(rendering_cfg.get("dynamic_time_repeat", 3)),
        "add_plane_idx": bool(rendering_cfg.get("add_plane_idx", False)),
        "add_path_idx": bool(rendering_cfg.get("add_path_idx", False)),
        "wss_clim": tuple(float(x) for x in list(rendering_cfg.get("wss_clim", [0.0, 10.0]))[:2]),
        "wss_bar_cfg": copy.deepcopy(rendering_cfg.get("wss_bar_cfg", DEFAULT_WSS_BAR_CFG)),
        "tke_clim": tuple(float(x) for x in list(rendering_cfg.get("tke_clim", [0.0, 100.0]))[:2]),
        "tke_bar_cfg": copy.deepcopy(rendering_cfg.get("tke_bar_cfg", DEFAULT_TKE_BAR_CFG)),
        "streamline_clim": tuple(float(x) for x in list(rendering_cfg.get("streamline_clim", [0.0, 1.0]))[:2]),
        "streamline_bar_cfg": copy.deepcopy(rendering_cfg.get("streamline_bar_cfg", DEFAULT_STREAMLINE_BAR_CFG)),
    }
