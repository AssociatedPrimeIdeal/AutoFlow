from dataclasses import dataclass, field
import json
import os
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .case_types import InputCase
from .config import (
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
    background_phase_correction: bool = False
    background_phase_corr_fit_order: int = 3
    background_phase_threshold: float = 0.1
    dual_venc_ratio1: float = 0.0
    dual_venc_ratio2: float = 0.0
    dicom_read_workers: int = 1

    use_center_plane: bool = True
    cross_section_dist: float = 5.0
    start_dist: float = 5.0
    end_dist: float = 0.0

    remove_small_cc: bool = True
    min_cc_volume: float = 50.0

    seed_ratio: float = 0.02
    max_steps: int = 2000
    min_seeds: int = 50
    terminal_speed: float = 0.01
    rng_seed: int = 0
    tube_radius: float = 0.05
    pathline_color: Optional[str] = None
    plane_pathline_color: Optional[str] = None

    autoseg: bool = False
    autoseg_backend: str = "nnUNet"
    autoseg_model: str = ""
    autoseg_checkpoint: str = "checkpoint_final.pth"
    autoseg_device: str = "auto"
    autoseg_label_map: str = ""

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

    add_plane_idx: bool = False
    add_path_idx: bool = False

    wss_clim: Tuple[float, float] = (0.0, 10.0)
    wss_bar_cfg: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_WSS_BAR_CFG))
    tke_clim: Tuple[float, float] = (0.0, 100.0)
    tke_bar_cfg: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_TKE_BAR_CFG))
    streamline_clim: Tuple[float, float] = (0.0, 1.0)
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
DEFAULT_STREAMLINE_BAR_CFG = DEFAULT_STREAMLINE_BAR_CFG


def build_workspace(config: Optional[AutoFlowConfig] = None) -> Workspace:
    cfg = config or AutoFlowConfig.from_config_dir()
    ws = Workspace()
    apply_config_bundle_to_workspace(ws, load_config_bundle(cfg.config_dir))
    ws.loader_params.background_phase_correction.enabled = bool(cfg.background_phase_correction)
    ws.loader_params.background_phase_correction.corr_fit_order = int(cfg.background_phase_corr_fit_order)
    ws.loader_params.background_phase_correction.threshold = float(cfg.background_phase_threshold)
    ws.loader_params.background_phase_correction.dual_venc_ratio1 = float(cfg.dual_venc_ratio1)
    ws.loader_params.background_phase_correction.dual_venc_ratio2 = float(cfg.dual_venc_ratio2)
    ws.loader_params.dicom_read_workers = int(cfg.dicom_read_workers)
    ws.plane_gen_params.use_center_plane = bool(cfg.use_center_plane)
    ws.plane_gen_params.cross_section_distance = float(cfg.cross_section_dist)
    ws.plane_gen_params.start_distance = float(cfg.start_dist)
    ws.plane_gen_params.end_distance = float(cfg.end_dist)
    ws.skeleton_params.remove_small_cc = bool(cfg.remove_small_cc)
    ws.skeleton_params.min_cc_volume_mm3 = float(cfg.min_cc_volume)
    ws.streamline_params.seed_ratio = float(cfg.seed_ratio)
    ws.streamline_params.max_steps = int(cfg.max_steps)
    ws.streamline_params.min_seeds = int(cfg.min_seeds)
    ws.streamline_params.terminal_speed = float(cfg.terminal_speed)
    ws.streamline_params.rng_seed = int(cfg.rng_seed)
    ws.streamline_params.tube_radius = float(cfg.tube_radius)
    ws.streamline_params.pathline_color = str(cfg.pathline_color or cfg.plane_pathline_color or "deepskyblue")
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
        autoseg=cfg.autoseg,
        autoseg_backend=cfg.autoseg_backend,
        autoseg_model=cfg.autoseg_model,
        autoseg_checkpoint=cfg.autoseg_checkpoint,
        autoseg_device=cfg.autoseg_device,
        autoseg_label_map=cfg.autoseg_label_map,
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
        wss_clim=cfg.wss_clim,
        wss_bar_cfg=dict(cfg.wss_bar_cfg),
        tke_clim=cfg.tke_clim,
        tke_bar_cfg=dict(cfg.tke_bar_cfg),
        streamline_clim=cfg.streamline_clim,
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
    base_ws = build_workspace(config)
    results: List[Dict[str, Any]] = []
    last_case_out = ""

    for case in input_cases:
        case_name = case.output_name or os.path.splitext(os.path.basename(case.input_path))[0]
        case_out = os.path.join(config.output_dir, case_name)
        reuse_file = resolve_reuse_plane_file(config.reuse_planes, case_name)

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
                background_phase_correction=config.background_phase_correction,
                background_phase_corr_fit_order=config.background_phase_corr_fit_order,
                background_phase_threshold=config.background_phase_threshold,
                dual_venc_ratio1=config.dual_venc_ratio1,
                dual_venc_ratio2=config.dual_venc_ratio2,
                dicom_read_workers=config.dicom_read_workers,
                use_center_plane=config.use_center_plane,
                cross_section_dist=config.cross_section_dist,
                start_dist=config.start_dist,
                end_dist=config.end_dist,
                remove_small_cc=config.remove_small_cc,
                min_cc_volume=config.min_cc_volume,
                seed_ratio=config.seed_ratio,
                max_steps=config.max_steps,
                min_seeds=config.min_seeds,
                terminal_speed=config.terminal_speed,
                rng_seed=config.rng_seed,
                tube_radius=config.tube_radius,
                pathline_color=config.pathline_color,
                plane_pathline_color=config.plane_pathline_color,
                autoseg=config.autoseg,
                autoseg_backend=config.autoseg_backend,
                autoseg_model=config.autoseg_model,
                autoseg_checkpoint=config.autoseg_checkpoint,
                autoseg_device=config.autoseg_device,
                autoseg_label_map=config.autoseg_label_map,
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
                wss_clim=config.wss_clim,
                wss_bar_cfg=dict(config.wss_bar_cfg),
                tke_clim=config.tke_clim,
                tke_bar_cfg=dict(config.tke_bar_cfg),
                streamline_clim=config.streamline_clim,
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
