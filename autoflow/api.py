import copy
from dataclasses import dataclass, field
import json
import os
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .case_types import InputCase
from .config import (
    DEFAULT_PRESSURE_GRADIENT_BAR_CFG,
    DEFAULT_PLANE_VIDEO_CFG,
    DEFAULT_RELATIVE_PRESSURE_BAR_CFG,
    DEFAULT_SHARED_COLORBAR_CFG,
    DEFAULT_STREAMLINE_BAR_CFG,
    DEFAULT_TKE_BAR_CFG,
    DEFAULT_WSS_BAR_CFG,
    apply_config_bundle_to_workspace,
    bundle_to_autoflow_kwargs,
    load_config_bundle,
)
from .core.models import Workspace
from .plane_io import resolve_reuse_plane_file
from .processing import collect_input_items, process_single


@dataclass
class AutoFlowConfig:
    inputs: Sequence[str] = field(default_factory=list)
    config_dir: Optional[str] = None
    output_dir: str = "./results"

    skip_derived: bool = False
    skip_wss: bool = False
    skip_tke: bool = False
    skip_pressure_gradient: bool = False
    skip_plane_metrics: bool = False
    use_multithread: bool = True
    reuse_planes: str = ""
    plane_import_mode: str = "world"
    export_planes: str = ""
    background_phase_correction: bool = False
    background_phase_method: str = "wrls_arto"
    background_phase_corr_fit_order: int = 3
    background_phase_threshold: float = 0.1
    background_phase_wrls_lambda: float = 5.0
    background_phase_wrls_magnitude_threshold: float = 0.04
    background_phase_wrls_mid_fov_fraction: float = 0.5
    background_phase_wrls_mid_slice_fraction: float = 0.65
    background_phase_wrls_arto_iterations: int = 2
    background_phase_wrls_tau: float = 3.0
    background_phase_wrls_delta: float = 2.0
    background_phase_wrls_central_probability: float = 0.5
    background_phase_wrls_fista_iterations: int = 5000
    background_phase_wrls_gmm_iterations: int = 1000
    dual_venc_ratio1: float = 0.0
    dual_venc_ratio2: float = 0.0
    force_recompute_corr: bool = False
    background_phase_write_cache: bool = True
    dicom_read_workers: int = 1
    ignore_embedded_segmentation: bool = False
    phase_unwrap_enabled: bool = False
    phase_unwrap_method: str = "none"
    phase_unwrap_mask: str = "segmentation"
    phase_unwrap_device: str = "auto"
    phase_unwrap_tfc: bool = True
    phase_unwrap_lap4d_ts: float = 2.0
    phase_unwrap_nprs_upsampling_factor: int = 2
    phase_unwrap_nprs_pi_unwrap: bool = True
    phase_unwrap_nprs_auto_crop: bool = True

    plane_mode: str = "fixed_step"
    plane_count: int = 3
    cross_section_dist: float = 5.0
    # Legacy distance/uniform modes historically trimmed 5 mm at the start;
    # fixed-step center layouts do not rely on this advanced trim.
    start_dist: float = 5.0
    end_dist: float = 0.0
    plane_anchor: str = "center"
    plane_offset_mm: float = 5.0
    plane_direction: str = "both"
    plane_spacing_mode: str = "fraction"
    plane_spacing_ratio: float = 0.25
    segmentation_filter: bool = True
    use_center_plane: Optional[bool] = None

    remove_small_cc: bool = True
    min_cc_volume: float = 50.0
    cc_filter_mode: str = "hybrid"
    cc_rel_min_ratio: float = 0.01

    seed_ratio: float = 0.02
    max_steps: int = 2000
    min_seeds: int = 50
    pathline_seed_ratio: float = 0.2
    pathline_max_steps: int = 200
    pathline_min_seeds: int = 50
    pathline_seed_mode: str = "fixed"
    pathline_max_seeds: int = 250
    pathline_terminal_speed: float = 0.01
    pathline_rng_seed: int = 0
    pathline_tube_radius: float = 0.25
    terminal_speed: float = 0.01
    rng_seed: int = 0
    tube_radius: float = 0.25
    pathline_color: Optional[str] = None
    plane_pathline_color: Optional[str] = None
    pathline_color_mode: str = "per_plane"
    pathline_temporal_cache_mb: float = 512.0
    pressure_method: str = "least_squares"

    autoseg: bool = False
    autoseg_backend: str = "nnUNet4D"
    autoseg_model: str = "/nas-data2/ryy/CMR4DFlow2026/Segdata/scripts/nnunet/4D/run_7020_4d_full_ssd_20260824.sh"
    autoseg_checkpoint: str = "checkpoint_final.pth"
    autoseg_folds: str = "single"
    autoseg_device: str = "auto"
    autoseg_label_map: str = ""
    force_recompute_seg: bool = False
    write_segmentation_cache: bool = True
    segmentation_only: bool = False

    requested_metrics: Sequence[str] = field(default_factory=list)
    requested_videos: Sequence[str] = field(default_factory=list)

    fps: int = 12
    plane_rotation_frames: int = 180
    make_plane_video: bool = False
    make_wss_video: bool = False
    make_pressure_gradient_video: bool = False
    make_streamlines_video: bool = False
    make_tke_video: bool = False

    camera_view: str = "right"
    camera_distance_scale: float = 1.5
    rotate_dynamic_video: bool = True
    dynamic_rotation_frames: int = 180
    dynamic_rotation_elevation_deg: Optional[float] = 10.0
    dynamic_time_repeat: int = 3

    add_plane_idx: bool = True
    add_path_idx: bool = False
    plane_video_cfg: Dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULT_PLANE_VIDEO_CFG))
    window_size: Tuple[int, int] = (1600, 1200)
    shared_colorbar_show: bool = True
    shared_colorbar_bar_cfg: Dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULT_SHARED_COLORBAR_CFG["bar_cfg"]))

    wss_clim: Tuple[float, float] = (0.0, 10.0)
    wss_show_scalar_bar: bool = True
    wss_bar_cfg: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_WSS_BAR_CFG))
    tke_clim: Tuple[float, float] = (0.0, 100.0)
    tke_show_scalar_bar: bool = True
    tke_bar_cfg: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_TKE_BAR_CFG))
    pressure_gradient_clim: Optional[Tuple[float, float]] = None
    pressure_gradient_show_scalar_bar: bool = True
    pressure_gradient_bar_cfg: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_PRESSURE_GRADIENT_BAR_CFG))
    relative_pressure_clim: Optional[Tuple[float, float]] = None
    relative_pressure_show_scalar_bar: bool = True
    relative_pressure_bar_cfg: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_RELATIVE_PRESSURE_BAR_CFG))
    streamline_clim: Optional[Tuple[float, float]] = None
    streamline_show_scalar_bar: bool = True
    streamline_bar_cfg: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_STREAMLINE_BAR_CFG))

    @classmethod
    def from_config_dir(cls, config_dir: Optional[str] = None, **overrides):
        bundle = load_config_bundle(config_dir)
        kwargs = bundle_to_autoflow_kwargs(bundle)
        kwargs["config_dir"] = config_dir
        kwargs.update(overrides)
        return cls(**kwargs)


DEFAULT_WSS_BAR_CFG = DEFAULT_WSS_BAR_CFG
DEFAULT_TKE_BAR_CFG = DEFAULT_TKE_BAR_CFG
DEFAULT_PRESSURE_GRADIENT_BAR_CFG = DEFAULT_PRESSURE_GRADIENT_BAR_CFG
DEFAULT_STREAMLINE_BAR_CFG = DEFAULT_STREAMLINE_BAR_CFG


def build_workspace(config: Optional[AutoFlowConfig] = None) -> Workspace:
    cfg = config or AutoFlowConfig.from_config_dir()
    ws = Workspace()
    apply_config_bundle_to_workspace(ws, load_config_bundle(cfg.config_dir))
    ws.loader_params.background_phase_correction.enabled = bool(cfg.background_phase_correction)
    ws.loader_params.background_phase_correction.method = str(cfg.background_phase_method)
    ws.loader_params.background_phase_correction.corr_fit_order = int(cfg.background_phase_corr_fit_order)
    ws.loader_params.background_phase_correction.threshold = float(cfg.background_phase_threshold)
    ws.loader_params.background_phase_correction.wrls_lambda = float(cfg.background_phase_wrls_lambda)
    ws.loader_params.background_phase_correction.wrls_magnitude_threshold = float(cfg.background_phase_wrls_magnitude_threshold)
    ws.loader_params.background_phase_correction.wrls_mid_fov_fraction = float(cfg.background_phase_wrls_mid_fov_fraction)
    ws.loader_params.background_phase_correction.wrls_mid_slice_fraction = float(cfg.background_phase_wrls_mid_slice_fraction)
    ws.loader_params.background_phase_correction.wrls_arto_iterations = int(cfg.background_phase_wrls_arto_iterations)
    ws.loader_params.background_phase_correction.wrls_tau = float(cfg.background_phase_wrls_tau)
    ws.loader_params.background_phase_correction.wrls_delta = float(cfg.background_phase_wrls_delta)
    ws.loader_params.background_phase_correction.wrls_central_probability = float(cfg.background_phase_wrls_central_probability)
    ws.loader_params.background_phase_correction.wrls_fista_iterations = int(cfg.background_phase_wrls_fista_iterations)
    ws.loader_params.background_phase_correction.wrls_gmm_iterations = int(cfg.background_phase_wrls_gmm_iterations)
    ws.loader_params.background_phase_correction.dual_venc_ratio1 = float(cfg.dual_venc_ratio1)
    ws.loader_params.background_phase_correction.dual_venc_ratio2 = float(cfg.dual_venc_ratio2)
    ws.loader_params.background_phase_correction.force_recompute = bool(cfg.force_recompute_corr)
    ws.loader_params.background_phase_correction.write_cache = bool(cfg.background_phase_write_cache)
    ws.loader_params.dicom_read_workers = int(cfg.dicom_read_workers)
    ws.loader_params.ignore_embedded_segmentation = bool(cfg.ignore_embedded_segmentation)
    ws.phase_unwrap_params.enabled = bool(cfg.phase_unwrap_enabled)
    ws.phase_unwrap_params.method = str(cfg.phase_unwrap_method or "none")
    ws.phase_unwrap_params.mask_source = str(cfg.phase_unwrap_mask or "segmentation")
    ws.phase_unwrap_params.device = str(cfg.phase_unwrap_device or "auto")
    ws.phase_unwrap_params.tfc = bool(cfg.phase_unwrap_tfc)
    ws.phase_unwrap_params.lap4d_ts = float(cfg.phase_unwrap_lap4d_ts)
    ws.phase_unwrap_params.nprs_upsampling_factor = int(cfg.phase_unwrap_nprs_upsampling_factor)
    ws.phase_unwrap_params.nprs_pi_unwrap = bool(cfg.phase_unwrap_nprs_pi_unwrap)
    ws.phase_unwrap_params.nprs_auto_crop = bool(cfg.phase_unwrap_nprs_auto_crop)
    ws.segmentation.force_recompute_auto_cache = bool(cfg.force_recompute_seg)
    ws.segmentation.write_auto_cache = bool(cfg.write_segmentation_cache)
    ws.segmentation.auto_backend = str(cfg.autoseg_backend or "nnUNet")
    ws.segmentation.auto_model = str(cfg.autoseg_model or "")
    ws.segmentation.auto_checkpoint = str(cfg.autoseg_checkpoint or "checkpoint_final.pth")
    ws.segmentation.auto_folds = str(cfg.autoseg_folds or "single")
    ws.segmentation.auto_device = str(cfg.autoseg_device or "auto")
    ws.segmentation.auto_label_map = str(cfg.autoseg_label_map or "")
    ws.plane_gen_params.plane_mode = str(getattr(cfg, "plane_mode", "fixed_step") or "fixed_step")
    configured_plane_count = int(getattr(cfg, "plane_count", 3) or 3)
    ws.plane_gen_params.plane_count = configured_plane_count if configured_plane_count == -1 else max(1, configured_plane_count)
    ws.plane_gen_params.cross_section_distance = float(cfg.cross_section_dist)
    configured_start = float(cfg.start_dist)
    # Fixed-step layouts are centered on the usable path by default; retain the
    # historical 5 mm trim only for legacy distance/count modes.
    if str(getattr(cfg, "plane_mode", "fixed_step") or "fixed_step").strip().lower() == "fixed_step" and abs(configured_start - 5.0) < 1e-12:
        configured_start = 0.0
    ws.plane_gen_params.start_distance = configured_start
    ws.plane_gen_params.end_distance = float(cfg.end_dist)
    ws.plane_gen_params.anchor = str(getattr(cfg, "plane_anchor", "center") or "center")
    ws.plane_gen_params.anchor_offset_mm = float(getattr(cfg, "plane_offset_mm", 5.0))
    ws.plane_gen_params.direction = str(getattr(cfg, "plane_direction", "both") or "both")
    ws.plane_gen_params.spacing_mode = str(getattr(cfg, "plane_spacing_mode", "fraction") or "fraction")
    ws.plane_gen_params.spacing_ratio = float(getattr(cfg, "plane_spacing_ratio", 0.25))
    ws.plane_gen_params.segmentation_filter = bool(getattr(cfg, "segmentation_filter", True))
    if getattr(cfg, "use_center_plane", None) is not None:
        if bool(cfg.use_center_plane):
            ws.plane_gen_params.plane_mode = "count"
            ws.plane_gen_params.plane_count = 1
        elif str(getattr(cfg, "plane_mode", "") or "").strip() == "":
            ws.plane_gen_params.plane_mode = "distance"
    ws.skeleton_params.remove_small_cc = bool(cfg.remove_small_cc)
    ws.skeleton_params.min_cc_volume_mm3 = float(cfg.min_cc_volume)
    ws.skeleton_params.cc_filter_mode = str(cfg.cc_filter_mode or "hybrid")
    ws.skeleton_params.cc_rel_min_ratio = float(cfg.cc_rel_min_ratio)
    ws.streamline_params.seed_ratio = float(cfg.seed_ratio)
    ws.streamline_params.max_steps = int(cfg.max_steps)
    ws.streamline_params.min_seeds = int(cfg.min_seeds)
    ws.streamline_params.pathline_seed_ratio = float(cfg.pathline_seed_ratio)
    ws.streamline_params.pathline_max_steps = int(cfg.pathline_max_steps)
    ws.streamline_params.pathline_min_seeds = int(cfg.pathline_min_seeds)
    ws.streamline_params.pathline_seed_mode = "ratio" if str(cfg.pathline_seed_mode or "fixed").strip().lower() == "ratio" else "fixed"
    ws.streamline_params.pathline_max_seeds = max(1, int(cfg.pathline_max_seeds))
    ws.streamline_params.pathline_terminal_speed = float(cfg.pathline_terminal_speed)
    ws.streamline_params.pathline_rng_seed = int(cfg.pathline_rng_seed)
    ws.streamline_params.pathline_tube_radius = float(cfg.pathline_tube_radius)
    ws.streamline_params.terminal_speed = float(cfg.terminal_speed)
    ws.streamline_params.rng_seed = int(cfg.rng_seed)
    ws.streamline_params.tube_radius = float(cfg.tube_radius)
    ws.streamline_params.pathline_color = str(cfg.pathline_color or cfg.plane_pathline_color or "deepskyblue")
    color_mode = str(cfg.pathline_color_mode or "per_plane").strip().lower()
    ws.streamline_params.pathline_color_mode = color_mode if color_mode in {"uniform", "per_plane", "per_group"} else "per_plane"
    ws.streamline_params.pathline_temporal_cache_mb = max(0.0, float(cfg.pathline_temporal_cache_mb))
    pressure_method = str(getattr(cfg, "pressure_method", "least_squares") or "least_squares").strip().lower()
    if pressure_method not in {"least_squares", "ppe"}:
        pressure_method = "least_squares"
    ws.derived_params.pressure_method = pressure_method
    ws.derived_params.use_multithread = bool(cfg.use_multithread)
    return ws


def run_case(
    input_path: Union[str, InputCase],
    output_dir: Optional[str] = None,
    config: Optional[AutoFlowConfig] = None,
    workspace: Optional[Workspace] = None,
) -> Dict[str, Any]:
    cfg = config or AutoFlowConfig.from_config_dir()
    base_ws = workspace if workspace is not None else build_workspace(cfg)
    source = input_path
    if isinstance(input_path, InputCase):
        source = input_path
    case_dir = output_dir
    if not case_dir:
        if isinstance(source, InputCase):
            case_name = source.output_name or os.path.splitext(os.path.basename(source.input_path))[0]
        else:
            case_name = os.path.splitext(os.path.basename(str(source)))[0]
        case_dir = os.path.join(cfg.output_dir, case_name)
    return process_single(
        source,
        case_dir,
        workspace=base_ws,
        skip_derived=cfg.skip_derived,
        skip_wss=cfg.skip_wss,
        skip_tke=cfg.skip_tke,
        skip_pressure_gradient=cfg.skip_pressure_gradient,
        skip_plane_metrics=cfg.skip_plane_metrics,
        use_multithread=cfg.use_multithread,
        reuse_planes_path=cfg.reuse_planes,
        plane_import_mode=cfg.plane_import_mode,
        export_planes_path=cfg.export_planes,
        autoseg=cfg.autoseg,
        autoseg_backend=cfg.autoseg_backend,
        autoseg_model=cfg.autoseg_model,
        autoseg_checkpoint=cfg.autoseg_checkpoint,
        autoseg_folds=cfg.autoseg_folds,
        autoseg_device=cfg.autoseg_device,
        autoseg_label_map=cfg.autoseg_label_map,
        segmentation_only=cfg.segmentation_only,
        phase_unwrap_enabled=cfg.phase_unwrap_enabled,
        phase_unwrap_method=cfg.phase_unwrap_method,
        phase_unwrap_mask=cfg.phase_unwrap_mask,
        phase_unwrap_device=cfg.phase_unwrap_device,
        phase_unwrap_tfc=cfg.phase_unwrap_tfc,
        phase_unwrap_lap4d_ts=cfg.phase_unwrap_lap4d_ts,
        phase_unwrap_nprs_upsampling_factor=cfg.phase_unwrap_nprs_upsampling_factor,
        phase_unwrap_nprs_pi_unwrap=cfg.phase_unwrap_nprs_pi_unwrap,
        phase_unwrap_nprs_auto_crop=cfg.phase_unwrap_nprs_auto_crop,
        requested_metrics=list(cfg.requested_metrics),
        requested_videos=list(cfg.requested_videos),
        fps=cfg.fps,
        plane_rotation_frames=cfg.plane_rotation_frames,
        rotate_dynamic_video=cfg.rotate_dynamic_video,
        dynamic_rotation_frames=cfg.dynamic_rotation_frames,
        dynamic_rotation_elevation_deg=cfg.dynamic_rotation_elevation_deg,
        make_plane_video=cfg.make_plane_video,
        make_wss_video=cfg.make_wss_video,
        make_pressure_gradient_video=cfg.make_pressure_gradient_video,
        make_streamlines_video=cfg.make_streamlines_video,
        make_tke_video=cfg.make_tke_video,
        camera_view=cfg.camera_view,
        camera_distance_scale=cfg.camera_distance_scale,
        add_plane_idx=cfg.add_plane_idx,
        add_path_idx=cfg.add_path_idx,
        plane_video_cfg=copy.deepcopy(cfg.plane_video_cfg),
        window_size=cfg.window_size,
        shared_colorbar_show=cfg.shared_colorbar_show,
        shared_colorbar_bar_cfg=copy.deepcopy(cfg.shared_colorbar_bar_cfg),
        wss_clim=cfg.wss_clim,
        wss_show_scalar_bar=cfg.wss_show_scalar_bar,
        wss_bar_cfg=dict(cfg.wss_bar_cfg),
        tke_clim=cfg.tke_clim,
        tke_show_scalar_bar=cfg.tke_show_scalar_bar,
        tke_bar_cfg=dict(cfg.tke_bar_cfg),
        pressure_gradient_clim=cfg.pressure_gradient_clim,
        pressure_gradient_show_scalar_bar=cfg.pressure_gradient_show_scalar_bar,
        pressure_gradient_bar_cfg=dict(cfg.pressure_gradient_bar_cfg),
        relative_pressure_clim=cfg.relative_pressure_clim,
        relative_pressure_show_scalar_bar=cfg.relative_pressure_show_scalar_bar,
        relative_pressure_bar_cfg=dict(cfg.relative_pressure_bar_cfg),
        streamline_clim=cfg.streamline_clim,
        streamline_show_scalar_bar=cfg.streamline_show_scalar_bar,
        streamline_bar_cfg=dict(cfg.streamline_bar_cfg),
        dynamic_time_repeat=cfg.dynamic_time_repeat,
    )


def run_batch(config: AutoFlowConfig) -> Tuple[List[Dict[str, Any]], str]:
    if not config.inputs:
        raise ValueError("AutoFlowConfig.inputs is empty.")

    input_cases = collect_input_items(list(config.inputs))
    if not input_cases:
        print("No supported H5 or DICOM inputs found.")
        return [], ""

    print(f"Found {len(input_cases)} file(s) to process.")
    if config.export_planes and len(input_cases) > 1 and str(config.export_planes).lower().endswith(".json"):
        raise ValueError("AutoFlowConfig.export_planes must be a directory when processing multiple cases.")
    base_ws = build_workspace(config)
    results: List[Dict[str, Any]] = []
    last_case_out = ""

    for case in input_cases:
        case_name = case.output_name or os.path.splitext(os.path.basename(case.input_path))[0]
        case_out = os.path.join(config.output_dir, case_name)
        reuse_file = resolve_reuse_plane_file(config.reuse_planes, case_name)
        export_file = ""
        if config.export_planes:
            if len(input_cases) == 1 and str(config.export_planes).lower().endswith(".json"):
                export_file = str(config.export_planes)
            else:
                export_file = os.path.join(str(config.export_planes), case_name, "plane_positions.json")

        if config.reuse_planes and not os.path.isfile(reuse_file):
            error = f"reuse plane file not found: {config.reuse_planes}"
            results.append({"file": case.input_path, "case": case.display_name, "status": "error", "error": error})
            print(f"\n[ERROR] {error}")
            continue

        try:
            case_cfg = AutoFlowConfig(
                inputs=[case.input_path],
                config_dir=config.config_dir,
                output_dir=config.output_dir,
                skip_derived=config.skip_derived,
                skip_wss=config.skip_wss,
                skip_tke=config.skip_tke,
                skip_pressure_gradient=config.skip_pressure_gradient,
                skip_plane_metrics=config.skip_plane_metrics,
                use_multithread=config.use_multithread,
                reuse_planes=reuse_file,
                plane_import_mode=config.plane_import_mode,
                export_planes=export_file,
                background_phase_correction=config.background_phase_correction,
                background_phase_method=config.background_phase_method,
                background_phase_corr_fit_order=config.background_phase_corr_fit_order,
                background_phase_threshold=config.background_phase_threshold,
                background_phase_wrls_lambda=config.background_phase_wrls_lambda,
                background_phase_wrls_magnitude_threshold=config.background_phase_wrls_magnitude_threshold,
                background_phase_wrls_mid_fov_fraction=config.background_phase_wrls_mid_fov_fraction,
                background_phase_wrls_mid_slice_fraction=config.background_phase_wrls_mid_slice_fraction,
                background_phase_wrls_arto_iterations=config.background_phase_wrls_arto_iterations,
                background_phase_wrls_tau=config.background_phase_wrls_tau,
                background_phase_wrls_delta=config.background_phase_wrls_delta,
                background_phase_wrls_central_probability=config.background_phase_wrls_central_probability,
                background_phase_wrls_fista_iterations=config.background_phase_wrls_fista_iterations,
                background_phase_wrls_gmm_iterations=config.background_phase_wrls_gmm_iterations,
                dual_venc_ratio1=config.dual_venc_ratio1,
                dual_venc_ratio2=config.dual_venc_ratio2,
                force_recompute_corr=config.force_recompute_corr,
                background_phase_write_cache=config.background_phase_write_cache,
                dicom_read_workers=config.dicom_read_workers,
                ignore_embedded_segmentation=config.ignore_embedded_segmentation,
                phase_unwrap_enabled=config.phase_unwrap_enabled,
                phase_unwrap_method=config.phase_unwrap_method,
                phase_unwrap_mask=config.phase_unwrap_mask,
                phase_unwrap_device=config.phase_unwrap_device,
                phase_unwrap_tfc=config.phase_unwrap_tfc,
                phase_unwrap_lap4d_ts=config.phase_unwrap_lap4d_ts,
                phase_unwrap_nprs_upsampling_factor=config.phase_unwrap_nprs_upsampling_factor,
                phase_unwrap_nprs_pi_unwrap=config.phase_unwrap_nprs_pi_unwrap,
                phase_unwrap_nprs_auto_crop=config.phase_unwrap_nprs_auto_crop,
                write_segmentation_cache=config.write_segmentation_cache,
                plane_mode=config.plane_mode,
                plane_count=config.plane_count,
                cross_section_dist=config.cross_section_dist,
                start_dist=config.start_dist,
                end_dist=config.end_dist,
                plane_anchor=config.plane_anchor,
                plane_offset_mm=config.plane_offset_mm,
                plane_direction=config.plane_direction,
                plane_spacing_mode=config.plane_spacing_mode,
                plane_spacing_ratio=config.plane_spacing_ratio,
                segmentation_filter=config.segmentation_filter,
                use_center_plane=config.use_center_plane,
                remove_small_cc=config.remove_small_cc,
                min_cc_volume=config.min_cc_volume,
                cc_filter_mode=config.cc_filter_mode,
                cc_rel_min_ratio=config.cc_rel_min_ratio,
                seed_ratio=config.seed_ratio,
                max_steps=config.max_steps,
                min_seeds=config.min_seeds,
                pathline_seed_ratio=config.pathline_seed_ratio,
                pathline_max_steps=config.pathline_max_steps,
                pathline_min_seeds=config.pathline_min_seeds,
                pathline_seed_mode=config.pathline_seed_mode,
                pathline_max_seeds=config.pathline_max_seeds,
                pathline_terminal_speed=config.pathline_terminal_speed,
                pathline_rng_seed=config.pathline_rng_seed,
                pathline_tube_radius=config.pathline_tube_radius,
                terminal_speed=config.terminal_speed,
                rng_seed=config.rng_seed,
                tube_radius=config.tube_radius,
                pathline_color=config.pathline_color,
                plane_pathline_color=config.plane_pathline_color,
                pathline_color_mode=config.pathline_color_mode,
                pathline_temporal_cache_mb=config.pathline_temporal_cache_mb,
                pressure_method=config.pressure_method,
                autoseg=config.autoseg,
                autoseg_backend=config.autoseg_backend,
                autoseg_model=config.autoseg_model,
                autoseg_checkpoint=config.autoseg_checkpoint,
                autoseg_folds=config.autoseg_folds,
                autoseg_device=config.autoseg_device,
                autoseg_label_map=config.autoseg_label_map,
                segmentation_only=config.segmentation_only,
                requested_metrics=list(config.requested_metrics),
                requested_videos=list(config.requested_videos),
                fps=config.fps,
                plane_rotation_frames=config.plane_rotation_frames,
                make_plane_video=config.make_plane_video,
                make_wss_video=config.make_wss_video,
                make_pressure_gradient_video=config.make_pressure_gradient_video,
                make_streamlines_video=config.make_streamlines_video,
                make_tke_video=config.make_tke_video,
                camera_view=config.camera_view,
                camera_distance_scale=config.camera_distance_scale,
                rotate_dynamic_video=config.rotate_dynamic_video,
                dynamic_rotation_frames=config.dynamic_rotation_frames,
                dynamic_rotation_elevation_deg=config.dynamic_rotation_elevation_deg,
                dynamic_time_repeat=config.dynamic_time_repeat,
                add_plane_idx=config.add_plane_idx,
                add_path_idx=config.add_path_idx,
                plane_video_cfg=copy.deepcopy(config.plane_video_cfg),
                window_size=config.window_size,
                shared_colorbar_show=config.shared_colorbar_show,
                shared_colorbar_bar_cfg=copy.deepcopy(config.shared_colorbar_bar_cfg),
                wss_clim=config.wss_clim,
                wss_show_scalar_bar=config.wss_show_scalar_bar,
                wss_bar_cfg=dict(config.wss_bar_cfg),
                tke_clim=config.tke_clim,
                tke_show_scalar_bar=config.tke_show_scalar_bar,
                tke_bar_cfg=dict(config.tke_bar_cfg),
                pressure_gradient_clim=config.pressure_gradient_clim,
                pressure_gradient_show_scalar_bar=config.pressure_gradient_show_scalar_bar,
                pressure_gradient_bar_cfg=dict(config.pressure_gradient_bar_cfg),
                relative_pressure_clim=config.relative_pressure_clim,
                relative_pressure_show_scalar_bar=config.relative_pressure_show_scalar_bar,
                relative_pressure_bar_cfg=dict(config.relative_pressure_bar_cfg),
                streamline_clim=config.streamline_clim,
                streamline_show_scalar_bar=config.streamline_show_scalar_bar,
                streamline_bar_cfg=dict(config.streamline_bar_cfg),
            )
            summary = run_case(case, output_dir=case_out, config=case_cfg, workspace=base_ws)
            results.append({"file": case.input_path, "case": case.display_name, "status": "ok", "summary": summary})
            last_case_out = case_out
        except Exception:
            print(f"\n[ERROR] Failed: {case.display_name or case.input_path}")
            print(traceback.format_exc())
            results.append({"file": case.input_path, "case": case.display_name, "status": "error", "error": traceback.format_exc()})

    os.makedirs(config.output_dir, exist_ok=True)
    batch_report = os.path.join(config.output_dir, "batch_report.json")
    with open(batch_report, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    n_ok = sum(1 for item in results if item.get("status") == "ok")
    times_sec = []
    for item in results:
        if item.get("status") == "ok":
            val = item.get("summary", {}).get("total_time_sec")
            if val is not None:
                times_sec.append(float(val))

    if times_sec:
        arr = np.asarray(times_sec, dtype=float)
        mean_sec = float(arr.mean())
        std_sec = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        time_text = f"{mean_sec:.2f} ± {std_sec:.2f} s"
        print(f"\nCase time: {time_text}")
        with open(os.path.join(config.output_dir, "time_summary.txt"), "w", encoding="utf-8") as f:
            f.write(f"n = {len(arr)}\n")
            f.write(f"mean_sec = {mean_sec:.6f}\n")
            f.write(f"std_sec = {std_sec:.6f}\n")
            f.write(f"formatted = {time_text}\n")

    print(f"\nDone: {n_ok}/{len(results)} succeeded. Report: {batch_report}")
    return results, last_case_out
