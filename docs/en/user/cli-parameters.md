# CLI parameter reference

All arguments accepted by `autoflow-run`. Omitted overrides inherit the module bundle; argparse `None` is a sentinel. See [CLI workflows](cli.md) and [effective configuration defaults](parameters.md). Flags with two spellings are aliases. Positive/negative Boolean switches act as described by their help text. Learned unwrapping choices pudip/gust are accepted only when their optional backends are installed; they are included here to keep the reference independent of the documentation-build environment.

| Flag / argument | Type / allowed values | Omitted parser value | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `-h` / `--help` | switch | `"==SUPPRESS=="` | CLI | show this help message and exit | `autoflow/cli.py; autoflow/api.py` |
| `inputs` | string | inherited / omitted | CLI | H5/HDF5 files, H5 batch directories, or DICOM acquisition directories. | `autoflow/cli.py; autoflow/api.py` |
| `--config-dir` | string | inherited / omitted | CLI | Directory containing per-module JSON configs used to build CLI defaults. When omitted, AutoFlow uses repo-level configs/ if present. | `autoflow/cli.py; autoflow/api.py` |
| `--output-dir` | string | inherited / omitted | CLI; `batch.output_dir` | Root output directory. Overrides configs/batch.json. | `autoflow/cli.py; autoflow/api.py` |
| `--import-planes` / `--reuse-planes` | string | inherited / omitted | CLI; `batch.reuse_planes` | Import a plane coordinate JSON file, or a directory containing per-case plane_positions.json files. | `autoflow/cli.py; autoflow/api.py` |
| `--plane-import-mode` | string; `["world", "local", "path_relative"]` | inherited / omitted | CLI | Map imported planes by canonical world-mm coordinates, local physical coordinates, or relative centerline position. | `autoflow/cli.py; autoflow/api.py` |
| `--export-planes` | string | inherited / omitted | CLI | Also export plane coordinates to this JSON file. For a multi-case run, provide a directory. | `autoflow/cli.py; autoflow/api.py` |
| `--skip-derived` | switch | `false` | CLI; `batch.skip_derived` | Skip all WSS/TKE/relative-pressure derived metrics. | `autoflow/cli.py; autoflow/api.py` |
| `--skip-wss` | switch | `false` | CLI; `batch.skip_wss` | Skip WSS computation, export, and derived summaries. | `autoflow/cli.py; autoflow/api.py` |
| `--skip-tke` | switch | `false` | CLI; `batch.skip_tke` | Skip TKE computation, export, and derived summaries. | `autoflow/cli.py; autoflow/api.py` |
| `--skip-pressure-gradient` | switch | `false` | CLI; `batch.skip_pressure_gradient` | Skip relative-pressure reconstruction, centerline pressure-drop outputs, and pressure-gradient-derived summaries. | `autoflow/cli.py; autoflow/api.py` |
| `--skip-plane-metrics` | switch | `false` | CLI; `batch.skip_plane_metrics` | Skip plane metric export. | `autoflow/cli.py; autoflow/api.py` |
| `--single-thread` | switch | inherited / omitted | CLI; `batch.use_multithread` | Disable parallel plane metric calculation. | `autoflow/cli.py; autoflow/api.py` |
| `--bgc` | switch | inherited / omitted | CLI; `loader.background_phase_correction.enabled` | Run background phase correction after loading, before segmentation. | `autoflow/cli.py; autoflow/api.py` |
| `--bgc-method` | string; `["msac", "wrls_arto"]` | inherited / omitted | CLI | Background phase correction algorithm. WRLS + ARTO is the default. | `autoflow/cli.py; autoflow/api.py` |
| `--bgc-fit-order` / `--background-phase-fit-order` | int | inherited / omitted | CLI | Polynomial fit order used by MSAC background phase correction when --bgc is enabled. | `autoflow/cli.py; autoflow/api.py` |
| `--bgc-threshold` / `--background-phase-threshold` | float | inherited / omitted | CLI; `loader.background_phase_correction.threshold` | MSAC threshold in venc units for background phase correction when --bgc is enabled. | `autoflow/cli.py; autoflow/api.py` |
| `--bgc-wrls-lambda` | float | inherited / omitted | CLI | WRLS+ARTO L1 regularization strength. | `autoflow/cli.py; autoflow/api.py` |
| `--bgc-wrls-magnitude-threshold` | float | inherited / omitted | CLI | WRLS+ARTO reference-magnitude mask fraction. | `autoflow/cli.py; autoflow/api.py` |
| `--bgc-wrls-mid-fov-fraction` | float | inherited / omitted | CLI | WRLS+ARTO middle-FOV initialization fraction. | `autoflow/cli.py; autoflow/api.py` |
| `--bgc-wrls-mid-slice-fraction` | float | inherited / omitted | CLI | WRLS+ARTO through-plane initialization fraction. | `autoflow/cli.py; autoflow/api.py` |
| `--bgc-wrls-arto-iterations` | int | inherited / omitted | CLI | WRLS+ARTO exclusion/refit iteration count. | `autoflow/cli.py; autoflow/api.py` |
| `--bgc-wrls-tau` | float | inherited / omitted | CLI | WRLS+ARTO central-Gaussian exclusion width in standard deviations. | `autoflow/cli.py; autoflow/api.py` |
| `--bgc-wrls-delta` | float | inherited / omitted | CLI | WRLS+ARTO minimum side-Gaussian separation. | `autoflow/cli.py; autoflow/api.py` |
| `--bgc-wrls-central-probability` | float | inherited / omitted | CLI | WRLS+ARTO minimum central-Gaussian prior. | `autoflow/cli.py; autoflow/api.py` |
| `--bgc-wrls-fista-iterations` | int | inherited / omitted | CLI | WRLS+ARTO FISTA iteration count per fit. | `autoflow/cli.py; autoflow/api.py` |
| `--bgc-wrls-gmm-iterations` | int | inherited / omitted | CLI | WRLS+ARTO maximum GMM EM iterations per ARTO pass. | `autoflow/cli.py; autoflow/api.py` |
| `--dual-venc-ratio1` | float | inherited / omitted | CLI; `loader.background_phase_correction.dual_venc_ratio1` | Dual-venc alias threshold ratio1 used when loading legacy Nv=7 complex H5 inputs. | `autoflow/cli.py; autoflow/api.py` |
| `--dual-venc-ratio2` | float | inherited / omitted | CLI; `loader.background_phase_correction.dual_venc_ratio2` | Dual-venc alias threshold ratio2 used when loading legacy Nv=7 complex H5 inputs. | `autoflow/cli.py; autoflow/api.py` |
| `--force-recompute-corr` | switch | `false` | CLI; `loader.background_phase_correction.force_recompute` | Ignore reusable H5 background phase correction caches and recompute them before optionally overwriting the cache. | `autoflow/cli.py; autoflow/api.py` |
| `--no-cache-write` | switch | inherited / omitted | CLI; `loader.background_phase_correction.write_cache` | Do not write newly computed correction or automatic-segmentation caches back to the input H5 (useful for read-only benchmarks). | `autoflow/cli.py; autoflow/api.py` |
| `--dicom-backend` | string; `["dicom2h5"]` | inherited / omitted | CLI; `loader.dicom_backend` | Compatibility setting: DICOM directories always convert through Dicom2H5. Legacy API/config native values migrate to dicom2h5. | `autoflow/cli.py; autoflow/api.py` |
| `--dicom-h5-dir` | string | inherited / omitted | CLI; `loader.dicom_h5_dir` | Directory for new Dicom2H5 outputs; defaults to OUTPUT_DIR/_dicom_h5. Existing H5 files are never replaced. | `autoflow/cli.py; autoflow/api.py` |
| `--plane-mode` | string; `["uniform", "fixed_step", "count", "distance", "anchored_offset"]` | inherited / omitted | CLI; `planes.plane_mode` | Plane placement mode. | `autoflow/cli.py; autoflow/api.py` |
| `--plane-count` | int | inherited / omitted | CLI; `planes.plane_count` | Plane count; -1 places every position that fits. Symmetric even counts omit the center plane. | `autoflow/cli.py; autoflow/api.py` |
| `--plane-anchor` | string; `["start", "center", "end", "junction"]` | inherited / omitted | CLI; `planes.anchor` | Anchor for fixed_step placement. | `autoflow/cli.py; autoflow/api.py` |
| `--plane-direction` | string; `["toward_start", "toward_end", "both"]` | inherited / omitted | CLI; `planes.direction` | Direction from the anchor. | `autoflow/cli.py; autoflow/api.py` |
| `--plane-spacing-mode` | string; `["distance", "fraction"]` | inherited / omitted | CLI; `planes.spacing_mode` | Interpret fixed-step spacing as millimetres or a fraction of path length. | `autoflow/cli.py; autoflow/api.py` |
| `--plane-spacing-ratio` | float | inherited / omitted | CLI; `planes.spacing_ratio` | Fixed-step spacing as a fraction of path length (for fraction mode). | `autoflow/cli.py; autoflow/api.py` |
| `--segmentation-filter` | switch | inherited / omitted | CLI; `planes.segmentation_filter` | Restrict each path and its planes/metrics to its topology-aware segmentation label (default). | `autoflow/cli.py; autoflow/api.py` |
| `--no-segmentation-filter` | switch | inherited / omitted | CLI; `planes.segmentation_filter` | Disable topology-aware segmentation path filtering. | `autoflow/cli.py; autoflow/api.py` |
| `--plane-offset-mm` | float | inherited / omitted | CLI; `planes.anchor_offset_mm` | First distance in mm from a graph junction in anchored_offset mode. | `autoflow/cli.py; autoflow/api.py` |
| `--plane-by-distance` | switch | inherited / omitted | CLI; `planes.use_center_plane` | Deprecated compatibility flag. Equivalent to --plane-mode distance. | `autoflow/cli.py; autoflow/api.py` |
| `--cross-section-dist` | float | inherited / omitted | CLI; `planes.cross_section_distance` | Plane spacing in mm when using distance spacing mode. | `autoflow/cli.py; autoflow/api.py` |
| `--start-dist` | float | inherited / omitted | CLI; `planes.start_distance` | Distance from path start before the first plane. | `autoflow/cli.py; autoflow/api.py` |
| `--end-dist` | float | inherited / omitted | CLI; `planes.end_distance` | Distance from path end to stop placing planes. | `autoflow/cli.py; autoflow/api.py` |
| `--remove-small-cc` | switch | `false` | CLI; `skeleton.remove_small_cc` | Remove small connected components before skeletonization. | `autoflow/cli.py; autoflow/api.py` |
| `--special-handling` | string; `["three_pass_merge", "contact_surface"]` | inherited / omitted | CLI; `skeleton.special_handling` | Special-label skeleton handling: three-pass merge (default) or legacy contact-surface separation. | `autoflow/cli.py; autoflow/api.py` |
| `--separate-special-label-contacts` / `--separate-label-contacts` | switch | inherited / omitted | CLI; `skeleton.separate_special_label_contacts` | Separate contacts between configured special labels before skeletonization (default). | `autoflow/cli.py; autoflow/api.py` |
| `--no-separate-special-label-contacts` / `--no-separate-label-contacts` | switch | inherited / omitted | CLI; `skeleton.separate_special_label_contacts` | Disable special-label contact separation. | `autoflow/cli.py; autoflow/api.py` |
| `--min-cc-volume` | float | inherited / omitted | CLI; `skeleton.min_cc_volume_mm3` | Minimum component volume in mm^3 when removal is enabled. | `autoflow/cli.py; autoflow/api.py` |
| `--cc-filter-mode` | string; `["absolute", "relative", "hybrid", "largest"]` | inherited / omitted | CLI; `skeleton.cc_filter_mode` | Connected-component filtering mode used during skeleton preprocessing. | `autoflow/cli.py; autoflow/api.py` |
| `--cc-rel-min-ratio` | float | inherited / omitted | CLI; `skeleton.cc_rel_min_ratio` | Relative component threshold ratio against the largest connected component when using relative or hybrid filtering. | `autoflow/cli.py; autoflow/api.py` |
| `--min-edge-points` / `--min-edge-count` | int | inherited / omitted | CLI; `skeleton.min_edge_points` | Minimum number of graph edge segments in a terminal branch; shorter endpoint spurs are removed (default: 3). | `autoflow/cli.py; autoflow/api.py` |
| `--seed-ratio` | float | inherited / omitted | CLI; `streamlines.seed_ratio` | Seed ratio for streamline rendering. | `autoflow/cli.py; autoflow/api.py` |
| `--tube-radius` | float | inherited / omitted | CLI; `streamlines.tube_radius` | Tube radius used in streamline rendering. | `autoflow/cli.py; autoflow/api.py` |
| `--pressure-method` | string; `["ppe", "ste", "least_squares"]` | inherited / omitted | CLI; `pressure_gradient.method` | `ppe` uses the merged Cartesian pressure solver; `ste` uses the staggered-grid Stokes estimator; `least_squares` is a compatibility alias for `ppe`. | `autoflow/cli.py; autoflow/api.py` |
| `--autoseg` | switch | `false` | CLI | If the loaded case has no segmentation, run auto segmentation before segmentation-dependent steps. | `autoflow/cli.py; autoflow/api.py` |
| `--autoseg-backend` | string | inherited / omitted | CLI; `segmentation.auto_backend` | Auto segmentation backend: nnUNet4D for temporal prediction or nnUNet for static 3D prediction copied across time. | `autoflow/cli.py; autoflow/api.py` |
| `--autoseg-model` | string | inherited / omitted | CLI; `segmentation.auto_model` | Auto segmentation model folder, or 'auto' for the backend-specific Dataset7010/Dataset7020 profile. | `autoflow/cli.py; autoflow/api.py` |
| `--autoseg-checkpoint` | string | inherited / omitted | CLI; `segmentation.auto_checkpoint` | Checkpoint name, or 'auto' for Dataset7010 final / Dataset7020 best. | `autoflow/cli.py; autoflow/api.py` |
| `--autoseg-folds` | string | inherited / omitted | CLI; `segmentation.auto_folds` | Auto segmentation folds: single, all (ensemble), or comma-separated fold IDs (default: single). | `autoflow/cli.py; autoflow/api.py` |
| `--autoseg-device` | string | inherited / omitted | CLI; `segmentation.auto_device` | Auto segmentation device: auto, cpu, or cuda. | `autoflow/cli.py; autoflow/api.py` |
| `--autoseg-label-map` | string | inherited / omitted | CLI; `segmentation.auto_label_map` | Optional JSON label remap passed to auto segmentation. | `autoflow/cli.py; autoflow/api.py` |
| `--force-recompute-seg` | switch | `false` | CLI; `segmentation.force_recompute_auto_cache` | Ignore H5 auto-segmentation caches tagged as AutoFlow-generated and rerun auto segmentation. Original or imported segmentations are not bypassed. | `autoflow/cli.py; autoflow/api.py` |
| `--ignore-embedded-segmentation` | switch | `false` | CLI; `loader.ignore_embedded_segmentation` | Ignore any segmentation embedded in the input (including original labels) without modifying the source file. | `autoflow/cli.py; autoflow/api.py` |
| `--segmentation-only` | switch | `false` | CLI | Stop after loading or generating segmentation; skip skeleton, planes, metrics, and videos. | `autoflow/cli.py; autoflow/api.py` |
| `--phase-unwrap-method` | string; `["none", "gc3D", "lap4D", "nprs", "pudip", "gust"]` | inherited / omitted | CLI; `phase_unwrapping.method` | Optional phase-unwrapping method (pudip and gust require their git-submodule backends). | `autoflow/cli.py; autoflow/api.py` |
| `--generate-pcmra` | switch | inherited / omitted | CLI; API `generate_pcmra` | Generate PC-MRA after requested correction steps; loading does not calculate it | `autoflow/cli.py`, `autoflow/core/pipeline.py` |
| `--correction` | switch | false | CLI; API `correction_all` | Run Background Correction → Noise Removal → Unwrap Phase → Generate PC-MRA before segmentation; default LAP4D/none | `autoflow/cli.py`, `autoflow/processing.py` |
| `--noise-removal` | switch | inherited / omitted | CLI; `noise_removal.enabled` | Build a PC-MRA display mask only | `autoflow/cli.py`, `autoflow/processing.py` |
| `--noise-removal-method` | string; `magnitude_temporal`, `magnitude` | inherited / omitted | CLI; `noise_removal.method` | Magnitude with optional temporal velocity SD | `autoflow/algorithms/noise_removal.py` |
| `--noise-magnitude-fraction` | float, 0–1 | inherited / `0.05` | CLI; `noise_removal.magnitude_fraction` | Fraction of maximum temporal-mean magnitude; lower retains more; 0 keeps positive signal | `autoflow/algorithms/noise_removal.py` |
| `--noise-velocity-std-max` | float, 0–1 | inherited / `0.80` | CLI; `noise_removal.velocity_std_max` | Upper temporal speed SD fraction of maximum SD, not VENC; higher retains more; 0 disables | `autoflow/algorithms/noise_removal.py` |
| `--phase-unwrap-mask` | string; `auto`, `none`, `segmask`, `pcmra_std`, `pcmra_mean` | inherited / omitted | CLI; `phase_unwrapping.mask_source` | Mask source: auto selects none for gc3D/lap4D/NPRS and pcmra_std for PUDIP/GUST. Traditional methods allow none/segmask; learned methods also allow pcmra_std/pcmra_mean. segmask requires an active segmentation. | `autoflow/cli.py`, `autoflow/core/pipeline.py` |
| `--phase-unwrap-device` | string; `["auto", "cpu", "cuda"]` | inherited / omitted | CLI; `phase_unwrapping.device` | Device for phase unwrapping; lap4D uses CUDA when available. | `autoflow/cli.py; autoflow/api.py` |
| `--with` | string | inherited / omitted | CLI | Comma-separated optional computations to enable. Supported: pwv,wss,tke,pg,vortex. Default computes only plane metrics. | `autoflow/cli.py; autoflow/api.py` |
| `--video` | string | inherited / omitted | CLI | Comma-separated videos to export. Supported: plane,wss,tke,pg,streamlines. | `autoflow/cli.py; autoflow/api.py` |
| `--fps` | int | inherited / omitted | CLI; `video_exporting.fps` | Output video FPS. | `autoflow/cli.py; autoflow/api.py` |
| `--plane-rotation-frames` | int | inherited / omitted | CLI; `video_exporting.plane_rotation_frames` | Frame count for plane rotation video. | `autoflow/cli.py; autoflow/api.py` |
| `--camera-view` | string | inherited / omitted | CLI; `video_exporting.camera_view` | Camera preset used for rendered videos. | `autoflow/cli.py; autoflow/api.py` |
| `--camera-distance-scale` | float | inherited / omitted | CLI; `video_exporting.camera_distance_scale` | Camera distance scale for rendered videos. | `autoflow/cli.py; autoflow/api.py` |
| `--rotate-dynamic-video` | switch | inherited / omitted | CLI; `video_exporting.rotate_dynamic_video` | Rotate dynamic videos while sweeping time. | `autoflow/cli.py; autoflow/api.py` |
| `--no-rotate-dynamic-video` | switch | inherited / omitted | CLI; `video_exporting.rotate_dynamic_video` | Disable dynamic video rotation. | `autoflow/cli.py; autoflow/api.py` |
| `--dynamic-rotation-frames` | int | inherited / omitted | CLI; `video_exporting.dynamic_rotation_frames` | Rotation frame count for dynamic videos. | `autoflow/cli.py; autoflow/api.py` |
| `--dynamic-time-repeat` | int | inherited / omitted | CLI; `video_exporting.dynamic_time_repeat` | Repeat each time frame this many times in dynamic videos. | `autoflow/cli.py; autoflow/api.py` |
| `--dynamic-rotation-elevation-deg` | float | inherited / omitted | CLI; `video_exporting.dynamic_rotation_elevation_deg` | Optional elevation override for dynamic rotation. | `autoflow/cli.py; autoflow/api.py` |
| `--add-plane-idx` | switch | `false` | CLI; `video_exporting.add_plane_idx` | Annotate plane indices in the plane video. | `autoflow/cli.py; autoflow/api.py` |
| `--add-path-idx` | switch | inherited / omitted | CLI; `video_exporting.add_path_idx` | Annotate path indices in the plane video. | `autoflow/cli.py; autoflow/api.py` |
| `--no-path-idx` | switch | inherited / omitted | CLI; `video_exporting.add_path_idx` | Disable path index annotations in the plane video. | `autoflow/cli.py; autoflow/api.py` |
