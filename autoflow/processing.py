import copy
import glob
import json
import os
import traceback

import numpy as np

from .core.models import StepId, Workspace
from .core.pipeline import PipelineEngine
from .algorithms import compute_derived_metrics
from .plane_io import (
    load_plane_positions,
    project_planes_to_workspace,
    resolve_reuse_plane_file,
    save_plane_positions,
)
from .rendering import (
    render_plane_rotation_video,
    render_streamlines_video,
    render_tke_video,
    render_wss_video,
)
from .reporting import load_metrics_from_output, print_metrics_summary, print_qc_summary


def process_single(
    h5_path,
    out_dir,
    workspace=None,
    skip_derived=False,
    skip_plane_metrics=False,
    use_multithread=False,
    reuse_planes_path="",
    fps=24,
    plane_rotation_frames=180,
    rotate_dynamic_video=False,
    dynamic_rotation_frames=180,
    dynamic_rotation_elevation_deg=None,
    make_plane_video=True,
    make_wss_video=True,
    make_streamlines_video=True,
    make_tke_video=True,
    camera_view="iso",
    camera_distance_scale=1.0,
    add_plane_idx=False,
    add_path_idx=False,
    wss_clim=None,
    wss_bar_cfg=None,
    tke_clim=None,
    tke_bar_cfg=None,
    streamline_clim=None,
    streamline_bar_cfg=None,
    dynamic_time_repeat=1,
):
    print(f"\n{'=' * 60}")
    print(f"Processing: {h5_path}")
    print(f"Output dir: {out_dir}")
    print(f"{'=' * 60}")

    os.makedirs(out_dir, exist_ok=True)
    ws = copy.deepcopy(workspace) if workspace is not None else Workspace()
    ws.paths.segmask_path = h5_path
    ws.paths.flow_path = h5_path
    ws.paths.output_dir = out_dir
    ws.derived_params.use_multithread = use_multithread
    engine = PipelineEngine()
    logger = lambda msg: None
    import time as _time

    t_total_start = _time.time()

    print("[1/7] Loading data...")
    engine.load_data(ws, logger)

    print("[2/7] Generate Skeleton...")
    result = engine.run_step(ws, StepId.GENERATE_SKELETON, logger)
    print(f"  -> {result.message}")

    print("[3/7] Generate Graph (+ branches/forks)...")
    result = engine.run_step(ws, StepId.GENERATE_GRAPH, logger)
    print(f"  -> {result.message}")

    print("[4/7] Generate Planes...")
    result = engine.run_step(ws, StepId.GENERATE_PLANES, logger)
    print(f"  -> {result.message}")

    if reuse_planes_path:
        print(f"[5/7] Reuse Plane Positions: {reuse_planes_path}")
        plane_items = load_plane_positions(reuse_planes_path)
        ws.planes = project_planes_to_workspace(plane_items, ws)
        planes_json = engine._save_planes_json(ws)
        print(f"  -> Reused {len(ws.planes)} planes saved={planes_json}")
    else:
        print("[5/7] Use generated planes")

    if skip_plane_metrics:
        print("[6/7] Skipped plane metrics")
    else:
        print("[6/7] Calculate & Save Metrics...")
        _, _, metric_msg = engine._compute_plane_metrics_internal(
            ws,
            save=True,
            use_multithread=use_multithread,
        )
        print(f"  -> {metric_msg}")
        try:
            engine._save_planes_json(ws)
        except Exception:
            pass

    pixelwise_result = {}
    if not skip_derived:
        print("[7/7] Compute Derived Metrics (WSS/TKE)...")
        dp = ws.derived_params
        engine.preprocess(ws)
        loaded_tke = ws.derived.tke_array

        derived = compute_derived_metrics(
            flow=ws.flow_raw * ws.segmask_binary[..., None],
            mask4d=ws.segmask_binary,
            spacing=ws.resolution,
            origin=ws.origin,
            smoothing_iteration=dp.smoothing_iteration,
            viscosity=dp.viscosity,
            inward_distance=dp.inward_distance,
            parabolic_fitting=dp.parabolic_fitting,
            no_slip_condition=dp.no_slip_condition,
            step_size=dp.step_size,
            tube_radius=dp.tube_radius,
            rho=dp.rho,
            save_pixelwise=True,
            tke_array=loaded_tke,
        )
        ws.derived.wss_surfaces = derived["wss_surfaces"]
        ws.derived.wss_volume = derived.get("wss_volume")
        ws.derived.tke_volume = derived["tke_volume"]
        ws.derived.tke_array = derived.get("tke_array")
        ws.derived.pixelwise_export = derived.get("pixelwise_export", {})
        pixelwise_result = ws.derived.pixelwise_export
        pixel_path = os.path.join(out_dir, "derived_metrics_pixelwise.npz")
        if pixelwise_result:
            np.savez_compressed(pixel_path, **pixelwise_result)
            print(f"  -> Saved pixelwise: {pixel_path}")
        ws.pipeline.mark_done(StepId.COMPUTE_DERIVED_METRICS)
        print(f"  -> Derived: Nt={len(ws.derived.wss_surfaces)}")
    else:
        print("[7/7] Skipped derived metrics (WSS/TKE)")

    total_time_sec = _time.time() - t_total_start
    print(f"  => Total pipeline took {total_time_sec:.2f}s")

    plane_positions_path = save_plane_positions(ws, os.path.join(out_dir, "plane_positions.json"), source_path=h5_path)
    print(f"Plane positions saved: {plane_positions_path}")

    video_paths = {}
    if make_plane_video:
        try:
            video_paths["planes"] = render_plane_rotation_video(
                ws,
                out_dir,
                fps=fps,
                n_frames=plane_rotation_frames,
                smoothing_iteration=ws.derived_params.smoothing_iteration,
                distance_scale=camera_distance_scale,
                add_plane_idx=add_plane_idx,
                add_path_idx=add_path_idx,
            )
            if video_paths["planes"]:
                print(f"Plane video saved: {video_paths['planes']}")
        except Exception:
            print("[WARN] Plane video failed")
            print(traceback.format_exc())
            video_paths["planes"] = ""

    if make_streamlines_video:
        try:
            video_paths["streamlines"] = render_streamlines_video(
                ws,
                out_dir,
                fps=fps,
                smoothing_iteration=ws.derived_params.smoothing_iteration,
                view=camera_view,
                distance_scale=camera_distance_scale,
                streamline_clim=streamline_clim,
                streamline_bar_cfg=streamline_bar_cfg,
                rotate=rotate_dynamic_video,
                rotation_frames=dynamic_rotation_frames,
                elevation_deg=dynamic_rotation_elevation_deg,
                time_repeat=dynamic_time_repeat,
            )
            if video_paths["streamlines"]:
                print(f"Streamlines video saved: {video_paths['streamlines']}")
        except Exception:
            print("[WARN] Streamlines video failed")
            print(traceback.format_exc())
            video_paths["streamlines"] = ""

    if not skip_derived and make_wss_video:
        try:
            video_paths["wss"] = render_wss_video(
                ws,
                out_dir,
                fps=fps,
                smoothing_iteration=ws.derived_params.smoothing_iteration,
                view=camera_view,
                distance_scale=camera_distance_scale,
                wss_clim=wss_clim,
                wss_bar_cfg=wss_bar_cfg,
                rotate=rotate_dynamic_video,
                rotation_frames=dynamic_rotation_frames,
                elevation_deg=dynamic_rotation_elevation_deg,
                time_repeat=dynamic_time_repeat,
            )
            if video_paths["wss"]:
                print(f"WSS video saved: {video_paths['wss']}")
        except Exception:
            print("[WARN] WSS video failed")
            print(traceback.format_exc())
            video_paths["wss"] = ""

    if not skip_derived and make_tke_video:
        try:
            video_paths["tke"] = render_tke_video(
                ws,
                out_dir,
                fps=fps,
                smoothing_iteration=ws.derived_params.smoothing_iteration,
                view=camera_view,
                distance_scale=camera_distance_scale,
                tke_clim=tke_clim,
                tke_bar_cfg=tke_bar_cfg,
                rotate=rotate_dynamic_video,
                rotation_frames=dynamic_rotation_frames,
                elevation_deg=dynamic_rotation_elevation_deg,
                time_repeat=dynamic_time_repeat,
            )
            if video_paths["tke"]:
                print(f"TKE video saved: {video_paths['tke']}")
        except Exception:
            print("[WARN] TKE video failed")
            print(traceback.format_exc())
            video_paths["tke"] = ""

    table_rows, raw_metrics, qc_data = (None, None, None)
    if not skip_plane_metrics:
        table_rows, raw_metrics, qc_data = load_metrics_from_output(out_dir)

    if table_rows:
        print("\n  === Plane Metrics Summary ===")
        print_metrics_summary(table_rows)

    if qc_data:
        print("\n  === Fork QC Summary ===")
        print_qc_summary(qc_data, ws.forks)

    summary = {
        "input": h5_path,
        "output_dir": out_dir,
        "resolution": ws.resolution.tolist(),
        "origin": np.asarray(ws.origin, dtype=float).reshape(3).tolist(),
        "rr": ws.rr,
        "total_time_sec": float(total_time_sec),
        "n_planes": len(ws.planes),
        "n_skeleton_pts": len(ws.skeleton_points) if ws.skeleton_points is not None else 0,
        "n_graph_nodes": len(ws.graph.points),
        "n_graph_edges": len(ws.graph.edges),
        "n_paths": len(ws.centerline_paths_smooth),
        "n_forks": len(ws.forks),
        "path_info": ws.path_info,
        "forks": ws.forks,
        "plane_metrics": ws.derived.plane_metrics,
        "plane_qc": ws.derived.plane_qc,
        "plane_positions_file": plane_positions_path,
        "reused_planes_file": reuse_planes_path,
        "videos": video_paths,
        "pixelwise_export": {k: list(np.asarray(v).shape) for k, v in pixelwise_result.items()} if pixelwise_result else {},
    }
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSummary saved: {summary_path}")
    return summary


def collect_h5_files(inputs):
    files = []
    for inp in inputs:
        if os.path.isfile(inp) and inp.lower().endswith((".h5", ".hdf5")):
            files.append(inp)
        elif os.path.isdir(inp):
            files.extend(sorted(glob.glob(os.path.join(inp, "**", "*.h5"), recursive=True)))
            files.extend(sorted(glob.glob(os.path.join(inp, "**", "*.hdf5"), recursive=True)))
    return sorted(dict.fromkeys(files))


def build_base_workspace():
    ws = Workspace()
    ws.plane_gen_params.use_center_plane = globals().get("USE_CENTER_PLANE", True)
    ws.plane_gen_params.cross_section_distance = globals().get("CROSS_SECTION_DIST", 5.0)
    ws.plane_gen_params.start_distance = globals().get("START_DIST", 5.0)
    ws.plane_gen_params.end_distance = globals().get("END_DIST", 0.0)
    ws.skeleton_params.remove_small_cc = globals().get("REMOVE_SMALL_CC", True)
    ws.skeleton_params.min_cc_volume_mm3 = globals().get("MIN_CC_VOLUME", 50.0)
    ws.streamline_params.max_steps = 2000
    ws.streamline_params.min_seeds = 50
    ws.streamline_params.seed_ratio = globals().get("SEED_RATIO", 0.02)
    ws.streamline_params.tube_radius = globals().get("TUBE_RADIUS", 0.05)
    return ws


def run_batch():
    inputs = globals().get("INPUT", None)
    if inputs is None:
        raise ValueError("INPUT is not defined.")
    dynamic_time_repeat = globals().get("DYNAMIC_TIME_REPEAT", 1)
    output_dir = globals().get("OUTPUT_DIR", "./batch_output")
    skip_derived = globals().get("SKIP_DERIVED", False)
    use_multithread = globals().get("USE_MULTITHREAD", True)
    reuse_planes = globals().get("REUSE_PLANES", "")
    fps = globals().get("FPS", 12)
    plane_rotation_frames = globals().get("PLANE_ROTATION_FRAMES", 180)
    make_plane_video = globals().get("MAKE_PLANE_VIDEO", True)
    make_wss_video = globals().get("MAKE_WSS_VIDEO", True)
    make_streamlines_video = globals().get("MAKE_STREAMLINES_VIDEO", True)
    make_tke_video = globals().get("MAKE_TKE_VIDEO", True)
    camera_view = globals().get("CAMERA_VIEW", "posterior")
    camera_distance_scale = globals().get("CAMERA_DISTANCE_SCALE", 1.5)
    skip_plane_metrics = globals().get("SKIP_PLANE_METRICS", False)
    rotate_dynamic_video = globals().get("ROTATE_DYNAMIC_VIDEO", False)
    dynamic_rotation_frames = globals().get("DYNAMIC_ROTATION_FRAMES", 180)
    dynamic_rotation_elevation_deg = globals().get("DYNAMIC_ROTATION_ELEVATION_DEG", None)
    add_plane_idx = globals().get("ADD_PLANE_IDX", False)
    add_path_idx = globals().get("ADD_PATH_IDX", False)

    wss_clim = globals().get("WSS_CLIM", (0, 5))
    wss_bar_cfg = globals().get(
        "WSS_BAR_CFG",
        {"position_x": 0.75, "position_y": 0.2, "height": 0.22, "width": 0.05, "title_font_size": 40, "label_font_size": 32},
    )
    tke_clim = globals().get("TKE_CLIM", (0, 2))
    tke_bar_cfg = globals().get(
        "TKE_BAR_CFG",
        {"position_x": 0.75, "position_y": 0.2, "height": 0.22, "width": 0.05, "title_font_size": 40, "label_font_size": 32},
    )
    streamline_clim = globals().get("STREAMLINE_CLIM", (0, 0.6))
    streamline_bar_cfg = globals().get(
        "STREAMLINE_BAR_CFG",
        {"position_x": 0.75, "position_y": 0.2, "height": 0.22, "width": 0.05, "title_font_size": 40, "label_font_size": 32},
    )

    h5_files = collect_h5_files(inputs)
    if not h5_files:
        print("No H5 files found.")
        return [], ""

    print(f"Found {len(h5_files)} file(s) to process.")
    base_ws = build_base_workspace()
    results = []
    case_out = ""

    for path in h5_files:
        name = os.path.splitext(os.path.basename(path))[0]
        case_out = os.path.join(output_dir, name)
        reuse_file = resolve_reuse_plane_file(reuse_planes, name)

        if reuse_planes and not os.path.isfile(reuse_file):
            results.append({"file": path, "status": "error", "error": f"reuse plane file not found: {reuse_planes}"})
            print(f"\n[ERROR] Reuse plane file not found: {reuse_planes}")
            continue

        try:
            summary = process_single(
                path,
                case_out,
                workspace=base_ws,
                skip_derived=skip_derived,
                use_multithread=use_multithread,
                reuse_planes_path=reuse_file,
                fps=fps,
                plane_rotation_frames=plane_rotation_frames,
                make_plane_video=make_plane_video,
                make_wss_video=make_wss_video,
                make_streamlines_video=make_streamlines_video,
                make_tke_video=make_tke_video,
                camera_view=camera_view,
                camera_distance_scale=camera_distance_scale,
                add_plane_idx=add_plane_idx,
                add_path_idx=add_path_idx,
                wss_clim=wss_clim,
                wss_bar_cfg=wss_bar_cfg,
                tke_clim=tke_clim,
                tke_bar_cfg=tke_bar_cfg,
                streamline_clim=streamline_clim,
                streamline_bar_cfg=streamline_bar_cfg,
                skip_plane_metrics=skip_plane_metrics,
                rotate_dynamic_video=rotate_dynamic_video,
                dynamic_rotation_frames=dynamic_rotation_frames,
                dynamic_rotation_elevation_deg=dynamic_rotation_elevation_deg,
                dynamic_time_repeat=dynamic_time_repeat,
            )
            results.append({"file": path, "status": "ok", "summary": summary})
        except Exception:
            print(f"\n[ERROR] Failed: {path}")
            print(traceback.format_exc())
            results.append({"file": path, "status": "error", "error": traceback.format_exc()})

    os.makedirs(output_dir, exist_ok=True)
    batch_report = os.path.join(output_dir, "batch_report.json")
    with open(batch_report, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    n_ok = sum(1 for r in results if r["status"] == "ok")

    times_sec = []
    for r in results:
        if r.get("status") == "ok":
            total_time_sec = r.get("summary", {}).get("total_time_sec", None)
            if total_time_sec is not None:
                times_sec.append(float(total_time_sec))

    if times_sec:
        arr = np.asarray(times_sec, dtype=float)
        mean_sec = arr.mean()
        std_sec = arr.std(ddof=1) if len(arr) > 1 else 0.0
        time_text = f"{mean_sec:.2f} ± {std_sec:.2f} s"
        print(f"\nCase time: {time_text}")

        with open(os.path.join(output_dir, "time_summary.txt"), "w", encoding="utf-8") as f:
            f.write(f"n = {len(arr)}\n")
            f.write(f"mean_sec = {mean_sec:.6f}\n")
            f.write(f"std_sec = {std_sec:.6f}\n")
            f.write(f"formatted = {time_text}\n")
    print(f"\nDone: {n_ok}/{len(results)} succeeded. Report: {batch_report}")
    return results, case_out
