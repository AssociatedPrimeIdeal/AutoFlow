# Outputs

## Common Output Files

| Output | Created when | Meaning | Main code |
| --- | --- | --- | --- |
| `planes.json` | planes exist | serialized plane geometry and summaries | `autoflow/core/pipeline.py` |
| `plane_positions.json` | planes exist | saved plane positions for reuse | `autoflow/plane_io.py` |
| `plane_metrics.json` | plane metrics run | time-resolved plane metrics | `autoflow/core/pipeline.py` |
| `plane_qc.json` | plane metrics run | fork and path QC | `autoflow/core/pipeline.py` |
| `plane_metrics_pixelwise.h5` | plane metrics run | per-plane pixelwise derived samples | `autoflow/core/pipeline.py` |
| `pwv.json` | PWV is enabled and PWV runs | PWV fit results for each configured PWV group | `autoflow/core/pipeline.py`, `autoflow/algorithms/pwv.py` |
| `pwv_<group>.png` | PWV plotting succeeds | per-group two-panel plot with time-to-foot fit and all plane flowrate waveforms | `autoflow/algorithms/pwv.py` |
| `derived_metrics_pixelwise.npz` | derived metrics run | whole-volume WSS, pressure-gradient intermediates, reconstructed relative pressure, and optional TKE arrays | `autoflow/processing.py` |
| `summary.json` | every processed case; also updated by GUI video export when present | single-case summary including request flags plus stage and video timings | `autoflow/processing.py`, `autoflow/ui/app.py` |
| `batch_report.json` | batch run | batch summary report | `autoflow/api.py`, `autoflow/processing.py` |
| `time_summary.txt` | batch run | timing summary | `autoflow/api.py`, `autoflow/processing.py` |

## Video Outputs

| Output | Created when | Meaning |
| --- | --- | --- |
| `planes_rotate.mp4` | `plane` video export requested from CLI, Python API, or GUI | rotating plane overview |
| `streamlines_video.mp4` or `streamlines_rotate.mp4` | `streamlines` video export requested from CLI, Python API, or GUI | time-resolved streamline movie |
| `wss_video.mp4` or `wss_rotate.mp4` | `wss` video export requested from CLI, Python API, or GUI | WSS movie |
| `tke_video.mp4` or `tke_rotate.mp4` | `tke` video export requested and TKE available | TKE movie |
| `pressure_gradient_video.mp4` or `pressure_gradient_rotate.mp4` | `pg` video export requested from CLI, Python API, or GUI | pressure-gradient movie |
| `relative_pressure_video.mp4` or `relative_pressure_rotate.mp4` | `pg` video export requested from CLI, Python API, or GUI | relative-pressure movie |

## Segmentation Sidecars

| Output | Created when | Meaning |
| --- | --- | --- |
| `*_threshold_segmentation.h5` | threshold segmentation succeeds | saved threshold segmentation |
| `*_auto_segmentation.h5` | auto segmentation succeeds | saved auto segmentation |
| user-selected H5, NPY, or NPZ file | `Save Active Segmentation...` | manual save of the active segmentation |

## Notes

- `derived_metrics_pixelwise.npz` stores whole-volume derived arrays
- `plane_metrics_pixelwise.h5` stores per-plane slice-cellwise samples
- TKE outputs remain optional and are absent for inputs that do not carry TKE
- `pwv.json` is also mirrored into `summary.json` as `pwv_results`, `pwv_file`, and `pwv_plot_files`
- `summary.json` records `requested_metrics`, `requested_videos`, `stage_times_sec`, `video_times_sec`, `pressure_method`, and `centerline_pressure_profiles`
- GUI `Export Videos...` also updates `summary.json` with refreshed `videos`, refreshed `video_times_sec`, and a `gui_video_export` record for that export run
- `planes.json` stores per-plane `path_info` when path metadata is available; grouped multi-label workflows can therefore expose `path_info.group_name` in that file
- GUI scene objects are not saved files, but grouped workflows keep stable internal keys such as `segmask_group_<group>`, `skeleton_<group>`, `graph_<group>`, `smooth_path_<group>_<path_idx>`, `plane_<group>_<plane_idx>`, and `pathline_<group>_<plane_idx>` while showing shorter browser-visible names like `plane 5` and `pathline 5`
- PWV scene planes are represented as one grouped browser object named `PWV planes`
- `plane_positions.json` stays geometry-focused and does not store browser colors or GUI-only group-title styling
