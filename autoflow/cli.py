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
    parser.add_argument("--reuse-planes", default=None, help="Plane positions file or directory to reuse.")

    parser.add_argument("--skip-derived", action="store_true", help="Skip all WSS/TKE/relative-pressure derived metrics.")
    parser.add_argument("--skip-wss", action="store_true", help="Skip WSS computation, export, and derived summaries.")
    parser.add_argument("--skip-tke", action="store_true", help="Skip TKE computation, export, and derived summaries.")
    parser.add_argument("--skip-pressure-gradient", action="store_true", help="Skip relative-pressure reconstruction, centerline pressure-drop outputs, and pressure-gradient-derived summaries.")
    parser.add_argument("--skip-plane-metrics", action="store_true", help="Skip plane metric export.")
    parser.add_argument("--single-thread", dest="use_multithread", action="store_false", help="Disable multithreaded plane metric calculation.")
    parser.add_argument("--bgc", dest="background_phase_correction", action="store_true", help="Enable background phase offset correction during loading.")
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
    parser.add_argument("--dicom-read-workers", type=int, default=None, help="Worker count for direct DICOM loading; use 0 to pick an automatic thread count.")
    parser.set_defaults(use_multithread=None)
    parser.set_defaults(background_phase_correction=None)

    parser.add_argument("--plane-mode", choices=["count", "distance", "anchored_offset"], default=None, help="Plane placement mode.")
    parser.add_argument("--plane-count", type=int, default=None, help="Number of evenly spaced planes when using count mode. count=1 is the center-plane default.")
    parser.add_argument("--plane-anchor", choices=["start", "end"], default=None, help="Anchor used by anchored_offset mode.")
    parser.add_argument("--plane-offset-mm", type=float, default=None, help="Offset in mm from the selected anchor when using anchored_offset mode.")
    parser.add_argument("--plane-by-distance", dest="use_center_plane", action="store_false", help="Deprecated compatibility flag. Equivalent to --plane-mode distance.")
    parser.add_argument("--cross-section-dist", type=float, default=None, help="Plane spacing in mm when using distance mode.")
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
    parser.add_argument("--autoseg-device", default=None, help="Auto segmentation device: auto, cpu, or cuda.")
    parser.add_argument("--autoseg-label-map", default=None, help="Optional JSON label remap passed to auto segmentation.")
    parser.add_argument(
        "--force-recompute-seg",
        action="store_true",
        help="Ignore H5 auto-segmentation caches tagged as AutoFlow-generated and rerun auto segmentation. Original or imported segmentations are not bypassed.",
    )
    parser.add_argument(
        "--segmentation-only",
        action="store_true",
        help="Stop after loading or generating segmentation; skip skeleton, planes, metrics, and videos.",
    )

    parser.add_argument(
        "--with",
        dest="requested_metrics",
        default=None,
        help="Comma-separated optional computations to enable. Supported: pwv,wss,tke,pg. Default computes only plane metrics.",
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
        "use_multithread": args.use_multithread,
        "background_phase_correction": args.background_phase_correction,
        "background_phase_corr_fit_order": args.background_phase_fit_order,
        "background_phase_threshold": args.background_phase_threshold,
        "dual_venc_ratio1": args.dual_venc_ratio1,
        "dual_venc_ratio2": args.dual_venc_ratio2,
        "force_recompute_corr": args.force_recompute_corr,
        "dicom_read_workers": args.dicom_read_workers,
        "plane_mode": args.plane_mode,
        "plane_count": args.plane_count,
        "use_center_plane": args.use_center_plane,
        "cross_section_dist": args.cross_section_dist,
        "start_dist": args.start_dist,
        "end_dist": args.end_dist,
        "plane_anchor": args.plane_anchor,
        "plane_offset_mm": args.plane_offset_mm,
        "min_cc_volume": args.min_cc_volume,
        "cc_filter_mode": args.cc_filter_mode,
        "cc_rel_min_ratio": args.cc_rel_min_ratio,
        "seed_ratio": args.seed_ratio,
        "tube_radius": args.tube_radius,
        "pressure_method": args.pressure_method,
        "autoseg_backend": args.autoseg_backend,
        "autoseg_model": args.autoseg_model,
        "autoseg_checkpoint": args.autoseg_checkpoint,
        "autoseg_device": args.autoseg_device,
        "autoseg_label_map": args.autoseg_label_map,
        "force_recompute_seg": args.force_recompute_seg,
        "segmentation_only": args.segmentation_only,
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
