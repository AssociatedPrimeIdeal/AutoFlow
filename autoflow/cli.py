import argparse

from .api import AutoFlowConfig, run_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the AutoFlow batch pipeline without the Qt GUI.")
    parser.add_argument("inputs", nargs="+", help="H5/HDF5 files, DICOM files, or directories to process.")
    parser.add_argument(
        "--config-dir",
        default=None,
        help="Directory containing per-module JSON configs used to build CLI defaults. When omitted, AutoFlow uses repo-level configs/ if present.",
    )
    parser.add_argument("--output-dir", default=None, help="Root output directory. Overrides configs/batch.json.")
    parser.add_argument(
        "--import-planes",
        "--reuse-planes",
        dest="reuse_planes",
        default=None,
        help="Import a plane coordinate JSON file, or a directory containing per-case plane_positions.json files.",
    )
    parser.add_argument(
        "--plane-import-mode",
        choices=["world", "local", "path_relative"],
        default=None,
        help="Map imported planes by canonical world-mm coordinates, local physical coordinates, or relative centerline position.",
    )
    parser.add_argument(
        "--export-planes",
        default=None,
        help="Also export plane coordinates to this JSON file. For a multi-case run, provide a directory.",
    )

    parser.add_argument("--skip-derived", action="store_true", help="Skip all WSS/TKE/relative-pressure derived metrics.")
    parser.add_argument("--skip-wss", action="store_true", help="Skip WSS computation, export, and derived summaries.")
    parser.add_argument("--skip-tke", action="store_true", help="Skip TKE computation, export, and derived summaries.")
    parser.add_argument("--skip-pressure-gradient", action="store_true", help="Skip relative-pressure reconstruction, centerline pressure-drop outputs, and pressure-gradient-derived summaries.")
    parser.add_argument("--skip-plane-metrics", action="store_true", help="Skip plane metric export.")
    parser.add_argument("--single-thread", dest="use_multithread", action="store_false", help="Disable multithreaded plane metric calculation.")
    parser.add_argument("--bgc", dest="background_phase_correction", action="store_true", help="Enable background phase offset correction during loading.")
    parser.add_argument(
        "--bgc-method",
        choices=["msac", "wrls_arto"],
        default=None,
        help="Background phase correction algorithm. WRLS + ARTO is the default.",
    )
    parser.add_argument(
        "--bgc-fit-order",
        "--background-phase-fit-order",
        dest="background_phase_fit_order",
        type=int,
        default=None,
        help="Polynomial fit order used by MSAC background phase correction when --bgc is enabled.",
    )
    parser.add_argument(
        "--bgc-threshold",
        "--background-phase-threshold",
        dest="background_phase_threshold",
        type=float,
        default=None,
        help="MSAC threshold in venc units for background phase correction when --bgc is enabled.",
    )
    parser.add_argument("--bgc-wrls-lambda", type=float, default=None, help="WRLS+ARTO L1 regularization strength.")
    parser.add_argument("--bgc-wrls-magnitude-threshold", type=float, default=None, help="WRLS+ARTO reference-magnitude mask fraction.")
    parser.add_argument("--bgc-wrls-mid-fov-fraction", type=float, default=None, help="WRLS+ARTO middle-FOV initialization fraction.")
    parser.add_argument("--bgc-wrls-mid-slice-fraction", type=float, default=None, help="WRLS+ARTO through-plane initialization fraction.")
    parser.add_argument("--bgc-wrls-arto-iterations", type=int, default=None, help="WRLS+ARTO exclusion/refit iteration count.")
    parser.add_argument("--bgc-wrls-tau", type=float, default=None, help="WRLS+ARTO central-Gaussian exclusion width in standard deviations.")
    parser.add_argument("--bgc-wrls-delta", type=float, default=None, help="WRLS+ARTO minimum side-Gaussian separation.")
    parser.add_argument("--bgc-wrls-central-probability", type=float, default=None, help="WRLS+ARTO minimum central-Gaussian prior.")
    parser.add_argument("--bgc-wrls-fista-iterations", type=int, default=None, help="WRLS+ARTO FISTA iteration count per fit.")
    parser.add_argument("--bgc-wrls-gmm-iterations", type=int, default=None, help="WRLS+ARTO maximum GMM EM iterations per ARTO pass.")
    parser.add_argument(
        "--dual-venc-ratio1",
        type=float,
        default=None,
        help="Dual-venc alias threshold ratio1 used when loading legacy Nv=7 complex H5 inputs.",
    )
    parser.add_argument(
        "--dual-venc-ratio2",
        type=float,
        default=None,
        help="Dual-venc alias threshold ratio2 used when loading legacy Nv=7 complex H5 inputs.",
    )
    parser.add_argument(
        "--force-recompute-corr",
        action="store_true",
        help="Ignore reusable H5 background phase correction caches and recompute them before optionally overwriting the cache.",
    )
    parser.add_argument(
        "--no-cache-write",
        dest="background_phase_write_cache",
        action="store_false",
        help="Do not write newly computed correction or automatic-segmentation caches back to the input H5 (useful for read-only benchmarks).",
    )
    parser.add_argument("--dicom-read-workers", type=int, default=None, help="Worker count for direct DICOM loading; use 0 to pick an automatic thread count.")
    parser.set_defaults(use_multithread=None)
    parser.set_defaults(background_phase_correction=None)
    parser.set_defaults(background_phase_write_cache=None)

    parser.add_argument("--plane-mode", choices=["uniform", "fixed_step", "count", "distance", "anchored_offset"], default=None, help="Plane placement mode.")
    parser.add_argument("--plane-count", type=int, default=None, help="Plane count; -1 places every position that fits. Symmetric even counts omit the center plane.")
    parser.add_argument("--plane-anchor", choices=["start", "center", "end", "junction"], default=None, help="Anchor for fixed_step placement.")
    parser.add_argument("--plane-direction", choices=["toward_start", "toward_end", "both"], default=None, help="Direction from the anchor.")
    parser.add_argument("--plane-spacing-mode", choices=["distance", "fraction"], default=None, help="Interpret fixed-step spacing as millimetres or a fraction of path length.")
    parser.add_argument("--plane-spacing-ratio", type=float, default=None, help="Fixed-step spacing as a fraction of path length (for fraction mode).")
    parser.add_argument("--segmentation-filter", dest="segmentation_filter", action="store_true", help="Restrict each path and its planes/metrics to its topology-aware segmentation label (default).")
    parser.add_argument("--no-segmentation-filter", dest="segmentation_filter", action="store_false", help="Disable topology-aware segmentation path filtering.")
    parser.set_defaults(segmentation_filter=None)
    parser.add_argument("--plane-offset-mm", type=float, default=None, help="First distance in mm from a graph junction in anchored_offset mode.")
    parser.add_argument("--plane-by-distance", dest="use_center_plane", action="store_false", help="Deprecated compatibility flag. Equivalent to --plane-mode distance.")
    parser.add_argument("--cross-section-dist", type=float, default=None, help="Plane spacing in mm when using distance spacing mode.")
    parser.add_argument("--start-dist", type=float, default=None, help="Distance from path start before the first plane.")
    parser.add_argument("--end-dist", type=float, default=None, help="Distance from path end to stop placing planes.")
    parser.set_defaults(use_center_plane=None)

    parser.add_argument("--remove-small-cc", action="store_true", help="Remove small connected components before skeletonization.")
    parser.add_argument("--min-cc-volume", type=float, default=None, help="Minimum component volume in mm^3 when removal is enabled.")
    parser.add_argument("--cc-filter-mode", choices=["absolute", "relative", "hybrid", "largest"], default=None, help="Connected-component filtering mode used during skeleton preprocessing.")
    parser.add_argument("--cc-rel-min-ratio", type=float, default=None, help="Relative component threshold ratio against the largest connected component when using relative or hybrid filtering.")

    parser.add_argument("--seed-ratio", type=float, default=None, help="Seed ratio for streamline rendering.")
    parser.add_argument("--tube-radius", type=float, default=None, help="Tube radius used in streamline rendering.")
    parser.add_argument(
        "--pressure-method",
        choices=["least_squares", "ppe"],
        default=None,
        help="Relative-pressure reconstruction method used by the pg metric/video outputs.",
    )

    parser.add_argument("--autoseg", action="store_true", help="If the loaded case has no segmentation, run auto segmentation before segmentation-dependent steps.")
    parser.add_argument("--autoseg-backend", default=None, help="Auto segmentation backend. Default comes from configs/segmentation.json or falls back to nnUNet.")
    parser.add_argument("--autoseg-model", default=None, help="Auto segmentation model folder. If omitted, AutoFlow uses the bundled default nnUNet model when present.")
    parser.add_argument("--autoseg-checkpoint", default=None, help="Auto segmentation checkpoint name.")
    parser.add_argument(
        "--autoseg-folds",
        default=None,
        help="Auto segmentation folds: single, all (ensemble), or comma-separated fold IDs (default: single).",
    )
    parser.add_argument("--autoseg-device", default=None, help="Auto segmentation device: auto, cpu, or cuda.")
    parser.add_argument("--autoseg-label-map", default=None, help="Optional JSON label remap passed to auto segmentation.")
    parser.add_argument(
        "--force-recompute-seg",
        action="store_true",
        help="Ignore H5 auto-segmentation caches tagged as AutoFlow-generated and rerun auto segmentation. Original or imported segmentations are not bypassed.",
    )
    parser.add_argument(
        "--ignore-embedded-segmentation",
        action="store_true",
        help="Ignore any segmentation embedded in the input (including original labels) without modifying the source file.",
    )
    parser.add_argument(
        "--segmentation-only",
        action="store_true",
        help="Stop after loading or generating segmentation; skip skeleton, planes, metrics, and videos.",
    )
    parser.add_argument("--phase-unwrap-method", choices=["none", "gc3D", "lap4D", "nprs"], default=None, help="Optional traditional phase-unwrapping method (disabled by default).")
    parser.add_argument("--phase-unwrap-mask", choices=["segmentation", "all"], default=None, help="Mask used for phase unwrapping.")
    parser.add_argument("--phase-unwrap-device", choices=["auto", "cpu", "cuda"], default=None, help="Device for phase unwrapping; lap4D uses CUDA when available.")

    parser.add_argument(
        "--with",
        dest="requested_metrics",
        default=None,
        help="Comma-separated optional computations to enable. Supported: pwv,wss,tke,pg,vortex. Default computes only plane metrics.",
    )
    parser.add_argument(
        "--video",
        dest="requested_videos",
        default=None,
        help="Comma-separated videos to export. Supported: plane,wss,tke,pg,streamlines.",
    )
    parser.add_argument("--fps", type=int, default=None, help="Output video FPS.")
    parser.add_argument("--plane-rotation-frames", type=int, default=None, help="Frame count for plane rotation video.")

    parser.add_argument("--camera-view", default=None, help="Camera preset used for rendered videos.")
    parser.add_argument("--camera-distance-scale", type=float, default=None, help="Camera distance scale for rendered videos.")
    parser.add_argument("--rotate-dynamic-video", dest="rotate_dynamic_video", action="store_true", help="Rotate dynamic videos while sweeping time.")
    parser.add_argument("--no-rotate-dynamic-video", dest="rotate_dynamic_video", action="store_false", help="Disable dynamic video rotation.")
    parser.add_argument("--dynamic-rotation-frames", type=int, default=None, help="Rotation frame count for dynamic videos.")
    parser.add_argument("--dynamic-time-repeat", type=int, default=None, help="Repeat each time frame this many times in dynamic videos.")
    parser.add_argument("--dynamic-rotation-elevation-deg", type=float, default=None, help="Optional elevation override for dynamic rotation.")

    parser.add_argument("--add-plane-idx", action="store_true", help="Annotate plane indices in the plane video.")
    parser.add_argument("--add-path-idx", dest="add_path_idx", action="store_true", help="Annotate path indices in the plane video.")
    parser.add_argument("--no-path-idx", dest="add_path_idx", action="store_false", help="Disable path index annotations in the plane video.")
    parser.set_defaults(add_path_idx=None, rotate_dynamic_video=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = AutoFlowConfig.from_config_dir(args.config_dir, inputs=args.inputs)

    overrides = {
        "output_dir": args.output_dir,
        "reuse_planes": args.reuse_planes,
        "plane_import_mode": args.plane_import_mode,
        "export_planes": args.export_planes,
        "use_multithread": args.use_multithread,
        "background_phase_correction": args.background_phase_correction,
        "background_phase_method": args.bgc_method,
        "background_phase_corr_fit_order": args.background_phase_fit_order,
        "background_phase_threshold": args.background_phase_threshold,
        "background_phase_wrls_lambda": args.bgc_wrls_lambda,
        "background_phase_wrls_magnitude_threshold": args.bgc_wrls_magnitude_threshold,
        "background_phase_wrls_mid_fov_fraction": args.bgc_wrls_mid_fov_fraction,
        "background_phase_wrls_mid_slice_fraction": args.bgc_wrls_mid_slice_fraction,
        "background_phase_wrls_arto_iterations": args.bgc_wrls_arto_iterations,
        "background_phase_wrls_tau": args.bgc_wrls_tau,
        "background_phase_wrls_delta": args.bgc_wrls_delta,
        "background_phase_wrls_central_probability": args.bgc_wrls_central_probability,
        "background_phase_wrls_fista_iterations": args.bgc_wrls_fista_iterations,
        "background_phase_wrls_gmm_iterations": args.bgc_wrls_gmm_iterations,
        "dual_venc_ratio1": args.dual_venc_ratio1,
        "dual_venc_ratio2": args.dual_venc_ratio2,
        "force_recompute_corr": args.force_recompute_corr,
        "background_phase_write_cache": args.background_phase_write_cache,
        "write_segmentation_cache": (
            False if args.background_phase_write_cache is False else None
        ),
        "dicom_read_workers": args.dicom_read_workers,
        "plane_mode": args.plane_mode,
        "plane_count": args.plane_count,
        "use_center_plane": args.use_center_plane,
        "cross_section_dist": args.cross_section_dist,
        "start_dist": args.start_dist,
        "end_dist": args.end_dist,
        "plane_anchor": args.plane_anchor,
        "plane_offset_mm": args.plane_offset_mm,
        "plane_direction": args.plane_direction,
        "plane_spacing_mode": args.plane_spacing_mode,
        "plane_spacing_ratio": args.plane_spacing_ratio,
        "segmentation_filter": args.segmentation_filter,
        "min_cc_volume": args.min_cc_volume,
        "cc_filter_mode": args.cc_filter_mode,
        "cc_rel_min_ratio": args.cc_rel_min_ratio,
        "seed_ratio": args.seed_ratio,
        "tube_radius": args.tube_radius,
        "pressure_method": args.pressure_method,
        "autoseg_backend": args.autoseg_backend,
        "autoseg_model": args.autoseg_model,
        "autoseg_checkpoint": args.autoseg_checkpoint,
        "autoseg_folds": args.autoseg_folds,
        "autoseg_device": args.autoseg_device,
        "autoseg_label_map": args.autoseg_label_map,
        "force_recompute_seg": args.force_recompute_seg,
        "ignore_embedded_segmentation": args.ignore_embedded_segmentation,
        "segmentation_only": args.segmentation_only,
        "phase_unwrap_enabled": (None if args.phase_unwrap_method is None else args.phase_unwrap_method != "none"),
        "phase_unwrap_method": args.phase_unwrap_method,
        "phase_unwrap_mask": args.phase_unwrap_mask,
        "phase_unwrap_device": args.phase_unwrap_device,
        "requested_metrics": [] if args.requested_metrics is None else [args.requested_metrics],
        "requested_videos": [] if args.requested_videos is None else [args.requested_videos],
        "fps": args.fps,
        "plane_rotation_frames": args.plane_rotation_frames,
        "camera_view": args.camera_view,
        "camera_distance_scale": args.camera_distance_scale,
        "rotate_dynamic_video": args.rotate_dynamic_video,
        "dynamic_rotation_frames": args.dynamic_rotation_frames,
        "dynamic_time_repeat": args.dynamic_time_repeat,
        "dynamic_rotation_elevation_deg": args.dynamic_rotation_elevation_deg,
        "add_path_idx": args.add_path_idx,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(config, key, value)

    if args.skip_derived:
        config.skip_derived = True
    if args.skip_wss:
        config.skip_wss = True
    if args.skip_tke:
        config.skip_tke = True
    if args.skip_pressure_gradient:
        config.skip_pressure_gradient = True
    if args.skip_plane_metrics:
        config.skip_plane_metrics = True
    if args.remove_small_cc:
        config.remove_small_cc = True
    if args.add_plane_idx:
        config.add_plane_idx = True
    if args.autoseg:
        config.autoseg = True

    run_batch(config)


if __name__ == "__main__":
    main()
