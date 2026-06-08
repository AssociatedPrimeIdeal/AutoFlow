import copy
import glob
import json
import os
import traceback
import time

import numpy as np

from .core.models import StepId, Workspace
from .core.pipeline import PipelineEngine
from .algorithms import collect_input_cases, resolve_input_case
from .algorithms.segmentation import (
    generate_nnunet_auto_segmentation,
    resolve_auto_segmentation_device,
    resolve_nnunet_model_folder,
    save_segmentation_file,
)
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


def _has_cached_derived_metrics(ws):
    has_wss = ws.derived.wss_volume is not None and np.size(ws.derived.wss_volume) > 0
    has_pg = ws.derived.pressure_gradient_array is not None and np.size(ws.derived.pressure_gradient_array) > 0
    has_tke = ws.derived.tke_array is not None or ws.derived.tke_volume is not None
    source_tke = ws.source_tke_array
    source_sigma = ws.source_sigma if ws.input_state.capabilities.has_complex_source else None
    return has_wss and has_pg and (has_tke or (source_tke is None and source_sigma is None))


def _build_cached_pixelwise_export(ws):
    if not _has_cached_derived_metrics(ws):
        return {}

    pixelwise = {
        "wss": np.asarray(ws.derived.wss_volume, dtype=np.float32),
        "spacing": np.asarray(ws.resolution, dtype=np.float32),
        "origin": np.asarray(ws.origin, dtype=np.float32),
        "pressure_gradient": np.asarray(ws.derived.pressure_gradient_array, dtype=np.float32),
        "pressure_gradient_mag": np.asarray(ws.derived.pressure_gradient_magnitude, dtype=np.float32),
        "pressure_gradient_peak": np.asarray(ws.derived.pressure_gradient_peak, dtype=np.float32),
        "pressure_gradient_support_mask": np.asarray(ws.derived.pressure_gradient_support_mask, dtype=np.uint8),
    }
    if ws.derived.tke_array is not None:
        tke_time = np.asarray(ws.derived.tke_array, dtype=np.float32)
        pixelwise["tke"] = np.asarray(np.max(tke_time, axis=3), dtype=np.float32)
        pixelwise["tke_time"] = tke_time
    return pixelwise


def _format_timing_parts(parts):
    return " ".join(f"{name}={seconds:.2f}s" for name, seconds in parts if seconds is not None)


def _default_segmentation_sidecar_path(ws, out_dir, source="auto"):
    base = os.path.splitext(os.path.basename(ws.paths.flow_path or ws.paths.segmask_path or "segmentation"))[0]
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{base}_{source}_segmentation.h5")


def _make_cli_autoseg_progress_handler():
    state = {"last_key": None}

    def _handler(payload):
        if not isinstance(payload, dict):
            message = str(payload or "").strip()
            if message:
                print(f"  [autoseg] {message}")
            return
        stage = str(payload.get("stage", "") or "")
        message = str(payload.get("message", "") or stage or "auto segmentation")
        elapsed_sec = payload.get("elapsed_sec")
        if elapsed_sec is not None:
            message = f"{message} | elapsed={float(elapsed_sec):.2f}s"
        key = (stage, payload.get("current"), payload.get("detail_current"), message)
        if key == state["last_key"]:
            return
        state["last_key"] = key
        print(f"  [autoseg] {message}")

    return _handler


def _run_cli_auto_segmentation(ws, out_dir, *, backend, model_folder, checkpoint_name, device, auto_label_map):
    if ws.mag_raw is None or ws.flow_raw is None:
        raise ValueError("auto segmentation requires loaded mag and flow data")
    resolved_model = resolve_nnunet_model_folder(model_folder)
    resolved_device = resolve_auto_segmentation_device(device)
    print(f"  [autoseg] backend={backend} model={resolved_model} checkpoint={checkpoint_name} device={resolved_device}")
    t_start = time.perf_counter()
    seg, provenance = generate_nnunet_auto_segmentation(
        mag=ws.mag_raw,
        flow=ws.flow_raw,
        resolution=ws.resolution,
        origin=ws.origin,
        model_folder=resolved_model,
        backend=backend,
        checkpoint_name=checkpoint_name,
        device=resolved_device,
        auto_label_map=auto_label_map,
        progress_callback=_make_cli_autoseg_progress_handler(),
    )
    infer_elapsed = time.perf_counter() - t_start
    ws.set_segmentation_source("auto", seg, provenance=provenance)
    ws.activate_segmentation_source("auto")
    sidecar = _default_segmentation_sidecar_path(ws, out_dir, source="auto")
    t_save = time.perf_counter()
    save_segmentation_file(
        sidecar,
        ws.get_active_segmentation(),
        resolution=ws.resolution,
        origin=ws.origin,
        provenance=ws.get_active_segmentation_provenance(),
    )
    save_elapsed = time.perf_counter() - t_save
    print(f"  [autoseg] sidecar saved: {sidecar} | save={save_elapsed:.2f}s total={infer_elapsed + save_elapsed:.2f}s")
    return sidecar


def process_single(
    input_source,
    out_dir,
    workspace=None,
    skip_derived=False,
    skip_wss=False,
    skip_tke=False,
    skip_pressure_gradient=False,
    skip_plane_metrics=False,
    use_multithread=False,
    reuse_planes_path="",
    autoseg=False,
    autoseg_backend="nnUNet",
    autoseg_model="",
    autoseg_checkpoint="checkpoint_final.pth",
    autoseg_device="auto",
    autoseg_label_map="",
    fps=24,
    plane_rotation_frames=180,
    rotate_dynamic_video=False,
    dynamic_rotation_frames=180,
    dynamic_rotation_elevation_deg=None,
    make_plane_video=True,
    make_wss_video=True,
    make_pressure_gradient_video=True,
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
    case = resolve_input_case(input_source)
    input_label = case.display_name or case.input_path

    print(f"\n{'=' * 60}")
    print(f"Processing: {input_label}")
    print(f"Output dir: {out_dir}")
    print(f"{'=' * 60}")

    os.makedirs(out_dir, exist_ok=True)
    ws = copy.deepcopy(workspace) if workspace is not None else Workspace()
    ws.paths.segmask_path = case.input_path
    ws.paths.flow_path = case.input_path
    ws.paths.output_dir = out_dir
    ws.derived_params.use_multithread = use_multithread
    engine = PipelineEngine()
    logger = lambda msg: None
    import time as _time

    stage_times = {}
    t_total_start = _time.perf_counter()

    print("[1/7] Loading data...")
    t_stage = _time.perf_counter()
    engine.load_data(ws, logger, input_source=case)
    elapsed = _time.perf_counter() - t_stage
    stage_times["load"] = float(elapsed)
    print(f"  -> load={elapsed:.2f}s")

    if ws.segmask_raw is None and autoseg:
        print("[1.5/7] Auto segmentation...")
        t_stage = _time.perf_counter()
        sidecar = _run_cli_auto_segmentation(
            ws,
            out_dir,
            backend=autoseg_backend,
            model_folder=autoseg_model,
            checkpoint_name=autoseg_checkpoint,
            device=autoseg_device,
            auto_label_map=autoseg_label_map,
        )
        elapsed = _time.perf_counter() - t_stage
        stage_times["autoseg"] = float(elapsed)
        print(f"  -> auto segmentation ready: {sidecar} | time={elapsed:.2f}s")

    print("[2/7] Generate Skeleton...")
    t_stage = _time.perf_counter()
    result = engine.run_step(ws, StepId.GENERATE_SKELETON, logger)
    elapsed = _time.perf_counter() - t_stage
    stage_times["skeleton"] = float(elapsed)
    print(f"  -> {result.message} | time={elapsed:.2f}s")

    print("[3/7] Generate Graph (+ branches/forks)...")
    t_stage = _time.perf_counter()
    result = engine.run_step(ws, StepId.GENERATE_GRAPH, logger)
    elapsed = _time.perf_counter() - t_stage
    stage_times["graph"] = float(elapsed)
    print(f"  -> {result.message} | time={elapsed:.2f}s")

    print("[4/7] Generate Planes...")
    t_stage = _time.perf_counter()
    result = engine.run_step(ws, StepId.GENERATE_PLANES, logger)
    elapsed = _time.perf_counter() - t_stage
    stage_times["planes"] = float(elapsed)
    print(f"  -> {result.message} | time={elapsed:.2f}s")

    if reuse_planes_path:
        print(f"[5/7] Reuse Plane Positions: {reuse_planes_path}")
        t_stage = _time.perf_counter()
        plane_items = load_plane_positions(reuse_planes_path)
        ws.planes = project_planes_to_workspace(plane_items, ws)
        planes_json = engine._save_planes_json(ws)
        elapsed = _time.perf_counter() - t_stage
        stage_times["reuse_planes"] = float(elapsed)
        print(f"  -> Reused {len(ws.planes)} planes saved={planes_json} | time={elapsed:.2f}s")
    else:
        print("[5/7] Use generated planes")

    if skip_plane_metrics:
        print("[6/7] Skipped plane metrics")
    else:
        print("[6/7] Calculate & Save Metrics...")
        t_step = _time.perf_counter()
        step_parts = []
        if not skip_derived and ws.segmask_raw is not None and not ws.derived.pixelwise_export:
            t_part = _time.perf_counter()
            engine._ensure_derived_metrics(ws, save_pixelwise=True, refresh_scene_objects=False)
            step_parts.append(("derived_prep", _time.perf_counter() - t_part))
        t_part = _time.perf_counter()
        _, _, metric_msg = engine._compute_plane_metrics_internal(
            ws,
            save=True,
            use_multithread=use_multithread,
        )
        step_parts.append(("plane_metrics", _time.perf_counter() - t_part))
        try:
            t_part = _time.perf_counter()
            engine._save_planes_json(ws)
            step_parts.append(("planes_json", _time.perf_counter() - t_part))
        except Exception:
            pass
        elapsed = _time.perf_counter() - t_step
        stage_times["plane_metrics"] = float(elapsed)
        print(f"  -> {metric_msg} | {_format_timing_parts(step_parts + [('total', elapsed)])}")

    pixelwise_result = {}
    if not skip_derived:
        if ws.segmask_raw is None:
            print("[7/7] Skipped derived metrics (no segmentation)")
        else:
            print("[7/7] Compute Derived Metrics (WSS/TKE/Pressure Gradient)...")
            t_step = _time.perf_counter()
            step_parts = []
            if not ws.derived.pixelwise_export:
                if not _has_cached_derived_metrics(ws):
                    t_part = _time.perf_counter()
                    engine._ensure_derived_metrics(ws, save_pixelwise=True, refresh_scene_objects=False)
                    step_parts.append(("derived_compute", _time.perf_counter() - t_part))
                else:
                    t_part = _time.perf_counter()
                    ws.derived.pixelwise_export = _build_cached_pixelwise_export(ws)
                    step_parts.append(("pixelwise_build", _time.perf_counter() - t_part))
            pixelwise_result = ws.derived.pixelwise_export
            pixel_path = os.path.join(out_dir, "derived_metrics_pixelwise.npz")
            if pixelwise_result:
                t_part = _time.perf_counter()
                np.savez_compressed(pixel_path, **pixelwise_result)
                step_parts.append(("npz_write", _time.perf_counter() - t_part))
                print(f"  -> Saved pixelwise: {pixel_path}")
            ws.pipeline.mark_done(StepId.COMPUTE_DERIVED_METRICS)
            tke_suffix = "" if ws.derived.tke_array is not None or ws.derived.tke_volume is not None else " tke=unavailable"
            elapsed = _time.perf_counter() - t_step
            stage_times["derived_export"] = float(elapsed)
            print(f"  -> Derived: Nt={len(ws.derived.wss_surfaces)}{tke_suffix} | {_format_timing_parts(step_parts + [('total', elapsed)])}")
    else:
        print("[7/7] Skipped derived metrics (WSS/TKE/Pressure Gradient)")

    total_time_sec = _time.perf_counter() - t_total_start
    timing_summary = _format_timing_parts([(name, seconds) for name, seconds in stage_times.items()])
    if timing_summary:
        print(f"  => Stage timing: {timing_summary}")
    print(f"  => Total pipeline took {total_time_sec:.2f}s")

    plane_positions_path = save_plane_positions(ws, os.path.join(out_dir, "plane_positions.json"), source_path=case.input_path)
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

    if ws.derived.pwv_results:
        print("\n  === PWV Summary ===")
        for result in ws.derived.pwv_results:
            name = str(result.get("name", "pwv") or "pwv")
            status = str(result.get("status", "unknown") or "unknown")
            valid = int(result.get("valid_plane_count", 0) or 0)
            total = int(result.get("plane_count", 0) or 0)
            pwv = result.get("pwv_m_s")
            message = str(result.get("message", "") or "")
            line = f"  - {name}: status={status} valid_planes={valid}/{total}"
            if pwv is not None:
                line += f" pwv={float(pwv):.4g} m/s"
            if message:
                line += f" | {message}"
            print(line)

    pwv_plot_files = {}
    for result in list(ws.derived.pwv_results or []):
        plot_file = str(result.get("plot_file", "") or "")
        if plot_file:
            pwv_plot_files[str(result.get("name", "pwv") or "pwv")] = plot_file
    summary = {
        "input": case.input_path,
        "input_kind": case.input_kind,
        "input_display_name": case.display_name,
        "output_dir": out_dir,
        "resolution": ws.resolution.tolist(),
        "origin": np.asarray(ws.origin, dtype=float).reshape(3).tolist(),
        "rr": ws.rr,
        "source_format": ws.input_state.source_format,
        "source_group": ws.input_state.source_group,
        "capabilities": ws.input_state.capabilities.to_dict(),
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
        "pwv_results": ws.derived.pwv_results,
        "pwv_file": ws.derived.pwv_file,
        "pwv_plot_files": pwv_plot_files,
        "plane_positions_file": plane_positions_path,
        "reused_planes_file": reuse_planes_path,
        "videos": video_paths,
        "pixelwise_export": {k: list(np.asarray(v).shape) for k, v in pixelwise_result.items()} if pixelwise_result else {},
        "plane_pixelwise_file": ws.derived.plane_pixelwise_file,
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


def collect_input_items(inputs):
    return collect_input_cases(inputs)


def build_base_workspace():
    ws = Workspace()
    ws.plane_gen_params.use_center_plane = globals().get("USE_CENTER_PLANE", True)
    ws.plane_gen_params.cross_section_distance = globals().get("CROSS_SECTION_DIST", 5.0)
    ws.plane_gen_params.start_distance = globals().get("START_DIST", 5.0)
    ws.plane_gen_params.end_distance = globals().get("END_DIST", 0.0)
    ws.skeleton_params.remove_small_cc = globals().get("REMOVE_SMALL_CC", True)
    ws.skeleton_params.min_cc_volume_mm3 = globals().get("MIN_CC_VOLUME", 50.0)
    ws.streamline_params.max_steps = globals().get("MAX_STEPS", 2000)
    ws.streamline_params.min_seeds = globals().get("MIN_SEEDS", 50)
    ws.streamline_params.seed_ratio = globals().get("SEED_RATIO", 0.02)
    ws.streamline_params.terminal_speed = globals().get("TERMINAL_SPEED", 0.01)
    ws.streamline_params.rng_seed = globals().get("RNG_SEED", 0)
    ws.streamline_params.tube_radius = globals().get("TUBE_RADIUS", 0.05)
    ws.streamline_params.pathline_color = globals().get("PATHLINE_COLOR", globals().get("PLANE_PATHLINE_COLOR", "deepskyblue"))
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

    input_cases = collect_input_items(inputs)
    if not input_cases:
        print("No supported H5 or DICOM inputs found.")
        return [], ""

    print(f"Found {len(input_cases)} file(s) to process.")
    base_ws = build_base_workspace()
    results = []
    case_out = ""

    for case in input_cases:
        name = case.output_name or os.path.splitext(os.path.basename(case.input_path))[0]
        case_out = os.path.join(output_dir, name)
        reuse_file = resolve_reuse_plane_file(reuse_planes, name)

        if reuse_planes and not os.path.isfile(reuse_file):
            results.append(
                {
                    "file": case.input_path,
                    "case": case.display_name,
                    "status": "error",
                    "error": f"reuse plane file not found: {reuse_planes}",
                }
            )
            print(f"\n[ERROR] Reuse plane file not found: {reuse_planes}")
            continue

        try:
            summary = process_single(
                case,
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
            results.append({"file": case.input_path, "case": case.display_name, "status": "ok", "summary": summary})
        except Exception:
            print(f"\n[ERROR] Failed: {case.display_name or case.input_path}")
            print(traceback.format_exc())
            results.append({"file": case.input_path, "case": case.display_name, "status": "error", "error": traceback.format_exc()})

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
