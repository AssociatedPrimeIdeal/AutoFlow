import argparse

from .api import AutoFlowConfig, run_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the AutoFlow batch pipeline without the Qt GUI.")
    parser.add_argument("inputs", nargs="+", help="H5/HDF5 files or directories to process.")
    parser.add_argument("--output-dir", default="./results", help="Root output directory.")
    parser.add_argument("--reuse-planes", default="", help="Plane positions file or directory to reuse.")

    parser.add_argument("--skip-derived", action="store_true", help="Skip WSS/TKE derived metrics.")
    parser.add_argument("--skip-plane-metrics", action="store_true", help="Skip plane metric export.")
    parser.add_argument("--single-thread", dest="use_multithread", action="store_false", help="Disable multithreaded plane metric calculation.")
    parser.set_defaults(use_multithread=True)

    parser.add_argument("--plane-by-distance", dest="use_center_plane", action="store_false", help="Generate evenly spaced planes instead of a single center plane.")
    parser.add_argument("--cross-section-dist", type=float, default=5.0, help="Plane spacing in mm when using distance mode.")
    parser.add_argument("--start-dist", type=float, default=5.0, help="Distance from path start before the first plane.")
    parser.add_argument("--end-dist", type=float, default=0.0, help="Distance from path end to stop placing planes.")
    parser.set_defaults(use_center_plane=True)

    parser.add_argument("--remove-small-cc", action="store_true", help="Remove small connected components before skeletonization.")
    parser.add_argument("--min-cc-volume", type=float, default=50.0, help="Minimum component volume in mm^3 when removal is enabled.")

    parser.add_argument("--seed-ratio", type=float, default=0.02, help="Seed ratio for streamline rendering.")
    parser.add_argument("--tube-radius", type=float, default=0.05, help="Tube radius used in streamline rendering.")

    parser.add_argument("--fps", type=int, default=12, help="Output video FPS.")
    parser.add_argument("--plane-rotation-frames", type=int, default=180, help="Frame count for plane rotation video.")
    parser.add_argument("--plane-video", dest="make_plane_video", action="store_true", help="Enable plane rotation video.")
    parser.add_argument("--no-plane-video", dest="make_plane_video", action="store_false", help="Disable plane rotation video.")
    parser.add_argument("--wss-video", dest="make_wss_video", action="store_true", help="Enable WSS video.")
    parser.add_argument("--no-wss-video", dest="make_wss_video", action="store_false", help="Disable WSS video.")
    parser.add_argument("--streamlines-video", dest="make_streamlines_video", action="store_true", help="Enable streamline video.")
    parser.add_argument("--no-streamlines-video", dest="make_streamlines_video", action="store_false", help="Disable streamline video.")
    parser.add_argument("--tke-video", dest="make_tke_video", action="store_true", help="Enable TKE video.")
    parser.add_argument("--no-tke-video", dest="make_tke_video", action="store_false", help="Disable TKE video.")
    parser.set_defaults(
        make_plane_video=False,
        make_wss_video=True,
        make_streamlines_video=True,
        make_tke_video=True,
    )

    parser.add_argument("--camera-view", default="right", help="Camera preset used for rendered videos.")
    parser.add_argument("--camera-distance-scale", type=float, default=1.5, help="Camera distance scale for rendered videos.")
    parser.add_argument("--rotate-dynamic-video", dest="rotate_dynamic_video", action="store_true", help="Rotate dynamic videos while sweeping time.")
    parser.add_argument("--no-rotate-dynamic-video", dest="rotate_dynamic_video", action="store_false", help="Disable dynamic video rotation.")
    parser.add_argument("--dynamic-rotation-frames", type=int, default=180, help="Rotation frame count for dynamic videos.")
    parser.add_argument("--dynamic-time-repeat", type=int, default=3, help="Repeat each time frame this many times in dynamic videos.")
    parser.add_argument("--dynamic-rotation-elevation-deg", type=float, default=10.0, help="Optional elevation override for dynamic rotation.")

    parser.add_argument("--add-plane-idx", action="store_true", help="Annotate plane indices in the plane video.")
    parser.add_argument("--add-path-idx", dest="add_path_idx", action="store_true", help="Annotate path indices in the plane video.")
    parser.add_argument("--no-path-idx", dest="add_path_idx", action="store_false", help="Disable path index annotations in the plane video.")
    parser.set_defaults(add_path_idx=False, rotate_dynamic_video=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = AutoFlowConfig(
        inputs=args.inputs,
        output_dir=args.output_dir,
        skip_derived=args.skip_derived,
        skip_plane_metrics=args.skip_plane_metrics,
        use_multithread=args.use_multithread,
        reuse_planes=args.reuse_planes,
        use_center_plane=args.use_center_plane,
        cross_section_dist=args.cross_section_dist,
        start_dist=args.start_dist,
        end_dist=args.end_dist,
        remove_small_cc=args.remove_small_cc,
        min_cc_volume=args.min_cc_volume,
        seed_ratio=args.seed_ratio,
        tube_radius=args.tube_radius,
        fps=args.fps,
        plane_rotation_frames=args.plane_rotation_frames,
        make_plane_video=args.make_plane_video,
        make_wss_video=args.make_wss_video,
        make_streamlines_video=args.make_streamlines_video,
        make_tke_video=args.make_tke_video,
        camera_view=args.camera_view,
        camera_distance_scale=args.camera_distance_scale,
        rotate_dynamic_video=args.rotate_dynamic_video,
        dynamic_rotation_frames=args.dynamic_rotation_frames,
        dynamic_rotation_elevation_deg=args.dynamic_rotation_elevation_deg,
        dynamic_time_repeat=args.dynamic_time_repeat,
        add_plane_idx=args.add_plane_idx,
        add_path_idx=args.add_path_idx,
    )
    run_batch(config)


if __name__ == "__main__":
    main()
