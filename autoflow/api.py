from dataclasses import dataclass, field
import json
import os
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from models import Workspace
from utils_demo import collect_h5_files, process_single, resolve_reuse_plane_file

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


@dataclass
class AutoFlowConfig:
    inputs: Sequence[str] = field(default_factory=list)
    output_dir: str = "./results"

    skip_derived: bool = False
    skip_plane_metrics: bool = False
    use_multithread: bool = True
    reuse_planes: str = ""

    use_center_plane: bool = True
    cross_section_dist: float = 5.0
    start_dist: float = 5.0
    end_dist: float = 0.0

    remove_small_cc: bool = True
    min_cc_volume: float = 50.0

    seed_ratio: float = 0.02
    tube_radius: float = 0.05

    fps: int = 12
    plane_rotation_frames: int = 180
    make_plane_video: bool = False
    make_wss_video: bool = True
    make_streamlines_video: bool = True
    make_tke_video: bool = True

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
    streamline_clim: Tuple[float, float] = (0.0, 1)
    streamline_bar_cfg: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_STREAMLINE_BAR_CFG))


def build_workspace(config: Optional[AutoFlowConfig] = None) -> Workspace:
    cfg = config or AutoFlowConfig()
    ws = Workspace()
    ws.plane_gen_params.use_center_plane = bool(cfg.use_center_plane)
    ws.plane_gen_params.cross_section_distance = float(cfg.cross_section_dist)
    ws.plane_gen_params.start_distance = float(cfg.start_dist)
    ws.plane_gen_params.end_distance = float(cfg.end_dist)
    ws.skeleton_params.remove_small_cc = bool(cfg.remove_small_cc)
    ws.skeleton_params.min_cc_volume_mm3 = float(cfg.min_cc_volume)
    ws.streamline_params.max_steps = 2000
    ws.streamline_params.min_seeds = 50
    ws.streamline_params.seed_ratio = float(cfg.seed_ratio)
    ws.streamline_params.tube_radius = float(cfg.tube_radius)
    return ws


def run_case(
    input_path: str,
    output_dir: Optional[str] = None,
    config: Optional[AutoFlowConfig] = None,
    workspace: Optional[Workspace] = None,
) -> Dict[str, Any]:
    cfg = config or AutoFlowConfig()
    base_ws = workspace if workspace is not None else build_workspace(cfg)
    case_dir = output_dir
    if not case_dir:
        case_name = os.path.splitext(os.path.basename(input_path))[0]
        case_dir = os.path.join(cfg.output_dir, case_name)
    return process_single(
        input_path,
        case_dir,
        workspace=base_ws,
        skip_derived=cfg.skip_derived,
        skip_plane_metrics=cfg.skip_plane_metrics,
        use_multithread=cfg.use_multithread,
        reuse_planes_path=cfg.reuse_planes,
        fps=cfg.fps,
        plane_rotation_frames=cfg.plane_rotation_frames,
        rotate_dynamic_video=cfg.rotate_dynamic_video,
        dynamic_rotation_frames=cfg.dynamic_rotation_frames,
        dynamic_rotation_elevation_deg=cfg.dynamic_rotation_elevation_deg,
        make_plane_video=cfg.make_plane_video,
        make_wss_video=cfg.make_wss_video,
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

    h5_files = collect_h5_files(list(config.inputs))
    if not h5_files:
        print("No H5 files found.")
        return [], ""

    print(f"Found {len(h5_files)} file(s) to process.")
    base_ws = build_workspace(config)
    results: List[Dict[str, Any]] = []
    last_case_out = ""

    for path in h5_files:
        case_name = os.path.splitext(os.path.basename(path))[0]
        case_out = os.path.join(config.output_dir, case_name)
        reuse_file = resolve_reuse_plane_file(config.reuse_planes, case_name)

        if config.reuse_planes and not os.path.isfile(reuse_file):
            error = f"reuse plane file not found: {config.reuse_planes}"
            results.append({"file": path, "status": "error", "error": error})
            print(f"\n[ERROR] {error}")
            continue

        try:
            case_cfg = AutoFlowConfig(
                inputs=[path],
                output_dir=config.output_dir,
                skip_derived=config.skip_derived,
                skip_plane_metrics=config.skip_plane_metrics,
                use_multithread=config.use_multithread,
                reuse_planes=reuse_file,
                use_center_plane=config.use_center_plane,
                cross_section_dist=config.cross_section_dist,
                start_dist=config.start_dist,
                end_dist=config.end_dist,
                remove_small_cc=config.remove_small_cc,
                min_cc_volume=config.min_cc_volume,
                seed_ratio=config.seed_ratio,
                tube_radius=config.tube_radius,
                fps=config.fps,
                plane_rotation_frames=config.plane_rotation_frames,
                make_plane_video=config.make_plane_video,
                make_wss_video=config.make_wss_video,
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
            summary = run_case(path, output_dir=case_out, config=case_cfg, workspace=base_ws)
            results.append({"file": path, "status": "ok", "summary": summary})
            last_case_out = case_out
        except Exception:
            print(f"\n[ERROR] Failed: {path}")
            print(traceback.format_exc())
            results.append({"file": path, "status": "error", "error": traceback.format_exc()})

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
