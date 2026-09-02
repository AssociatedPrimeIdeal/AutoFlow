# Outputs

## Common Output Files

| Output | Created when | Meaning | Main code |
| --- | --- | --- | --- |
| `planes.json` | planes exist | serialized plane geometry, `label_name`, and attached summaries | `autoflow/core/pipeline.py` |
| `planes.h5` | planes exist | one root group per plane with geometry, `label_name`, path metadata, metrics, and a `payload_json` mirror | `autoflow/plane_io.py`, `autoflow/core/pipeline.py` |
| `plane_positions.json` | planes exist | portable v2 plane coordinates with world/local centers, normal, source geometry, group, path rank, and relative path position | `autoflow/plane_io.py` |
| `plane_metrics.json` | plane metrics run | time-resolved plane metrics with explicit `plane_index` values | `autoflow/core/pipeline.py` |
| `plane_qc.json` | plane metrics run | fork and path QC | `autoflow/core/pipeline.py` |
| `plane_metrics_pixelwise.h5` | plane metrics run | per-plane pixelwise derived samples | `autoflow/core/pipeline.py` |
| `pwv.json` | PWV runs successfully | PWV fit results for each configured PWV group | `autoflow/core/pipeline.py`, `autoflow/algorithms/pwv.py` |
| `pwv_<group>.png` | PWV plotting succeeds | per-group two-panel plot with time-to-foot fit and all plane flowrate waveforms | `autoflow/algorithms/pwv.py` |
| `derived_metrics_pixelwise.npz` | derived metrics run | whole-volume WSS, pressure-gradient intermediates, reconstructed relative pressure, optional TKE, and requested vortex-kinematics arrays (uncompressed NPZ for faster export) | `autoflow/processing.py` |
| `quality_report.json` | every processed case | staged automated QC with `pass`, `warn`, `fail`, or `not_run` checks and suggested review actions | `autoflow/quality.py`, `autoflow/processing.py` |
| `summary.json` | every processed case; also updated by GUI video export when present | single-case summary including request flags plus stage and video timings | `autoflow/processing.py`, `autoflow/ui/app.py` |
| `phase_unwrap.npz` | optional phase-unwrapping run | wrapped/unwrapped phase, flow, signed `wrap_count`, `wrap_mask`, and `mask_used` | `autoflow/processing.py` |
| `batch_report.json` | batch run | batch summary report | `autoflow/api.py`, `autoflow/processing.py` |
| `time_summary.txt` | batch run | timing summary | `autoflow/api.py`, `autoflow/processing.py` |

## Video Outputs

| Output | Created when | Meaning |
| --- | --- | --- |
| `planes_rotate.mp4` | `plane` video export requested from CLI, Python API, or GUI | rotating plane overview with configurable index labels that default to `planeidx=<index>`; MP4 only |
| `streamlines_video.mp4` or `streamlines_rotate.mp4` | `streamlines` video export requested from CLI, Python API, or GUI | time-resolved streamline movie |
| `wss_video.mp4` or `wss_rotate.mp4` | `wss` video export requested from CLI, Python API, or GUI | WSS movie |
| `tke_video.mp4` or `tke_rotate.mp4` | `tke` video export requested and TKE available | TKE movie |
| `pressure_gradient_video.mp4` or `pressure_gradient_rotate.mp4` | `pg` video export requested from CLI, Python API, or GUI | pressure-gradient movie |
| `relative_pressure_video.mp4` or `relative_pressure_rotate.mp4` | `pg` video export requested from CLI, Python API, or GUI | relative-pressure movie |

## Segmentation Sidecars

| Output | Created when | Meaning |
| --- | --- | --- |
| `*_threshold_segmentation.h5` | threshold segmentation succeeds | saved threshold segmentation |
| source input H5 `segmask` dataset | auto segmentation succeeds on an H5 input | embedded reusable auto segmentation cache written back into the source H5 |
| `*_auto_segmentation.nii.gz` | auto segmentation succeeds | saved predicted segmentation NIfTI used by the run in the same `HF/AP/RL` geometry convention used by the nnUNet training/export scripts |
| `*_auto_segmentation_feature_*.nii.gz` | static 3D auto segmentation succeeds | saved nnUNet feature-channel NIfTI inputs; grouped 4D preprocessing keeps temporary source maps out of the output directory |
| user-selected H5, NPY, or NPZ file | `Segmentation` dock `Source -> Save...` | manual save of the active segmentation |

## Notes

- `derived_metrics_pixelwise.npz` stores whole-volume derived arrays
- When `vortex` is requested, `derived_metrics_pixelwise.npz` includes `vorticity`, `vorticity_magnitude`, `q_criterion`, `swirling_strength`, their phase-peak arrays, and `vortex_support_mask`.
- `plane_metrics_pixelwise.h5` stores per-plane slice-cellwise samples
- TKE outputs remain optional and are absent for inputs that do not carry TKE
- `pwv.json` is also mirrored into `summary.json` as `pwv_results`, `pwv_file`, and `pwv_plot_files`
- `summary.json` records `requested_metrics`, `requested_videos`, `stage_times_sec`, `video_times_sec`, `pressure_method`, `centerline_pressure_profiles`, and the generated quality report
- `summary.json` also records optional `phase_unwrap` method/device, skip reason, elapsed time, and wrap statistics
- multi-group H5 inputs write separate case subdirectories whose names include the H5 data-group path
- GUI `Export Videos...` also updates `summary.json` with refreshed `videos`, refreshed `video_times_sec`, and a `gui_video_export` record for that export run
- `planes.json` and `planes.h5` store per-plane `label_name` from `label_map` plus `path_info` when path metadata is available; grouped multi-label workflows can therefore expose both readable plane labels and `path_info.group_name` in those files
- `planes.h5` stores each plane as a root group such as `plane_0000`, `plane_0001`, and mirrors the full record again as `payload_json` inside that group
- GUI scene objects are not saved files, but grouped workflows keep stable internal keys such as `segmask_group_<group>`, `skeleton_<group>`, `graph_<group>`, `smooth_path_<group>_<path_idx>`, `plane_<group>_<plane_idx>`, and `pathline_<group>_<plane_idx>` while showing shorter browser-visible names like `plane 5` and `pathline 5`
- PWV scene planes are represented as one grouped browser object named `PWV planes`
- `plane_positions.json` uses schema `autoflow.plane_positions.v2`; `center_world_mm` is meaningful across cases only when their AutoFlow canonical world-mm frames are registered
- offline video export writes MP4 only; no GIF fallback is generated
