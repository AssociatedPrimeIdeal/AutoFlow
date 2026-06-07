# Outputs

## Common Output Files

| Output | Created when | Meaning | Main code |
| --- | --- | --- | --- |
| `planes.json` | planes exist | serialized plane geometry and summaries | `autoflow/core/pipeline.py` |
| `plane_positions.json` | planes exist | saved plane positions for reuse | `autoflow/plane_io.py` |
| `plane_metrics.json` | plane metrics run | time-resolved plane metrics | `autoflow/core/pipeline.py` |
| `plane_qc.json` | plane metrics run | fork and path QC | `autoflow/core/pipeline.py` |
| `plane_metrics_pixelwise.h5` | plane metrics run | per-plane pixelwise derived samples | `autoflow/core/pipeline.py` |
| `derived_metrics_pixelwise.npz` | derived metrics run | whole-volume WSS, pressure gradient, and optional TKE arrays | `autoflow/processing.py` |
| `summary.json` | every processed case | single-case summary | `autoflow/processing.py` |
| `batch_report.json` | batch run | batch summary report | `autoflow/api.py`, `autoflow/processing.py` |
| `time_summary.txt` | batch run | timing summary | `autoflow/api.py`, `autoflow/processing.py` |

## Video Outputs

| Output | Created when | Meaning |
| --- | --- | --- |
| `planes_rotate.mp4` | `--plane-video` | rotating plane overview |
| `streamlines_video.mp4` or `streamlines_rotate.mp4` | `--streamlines-video` | time-resolved streamline movie |
| `wss_video.mp4` or `wss_rotate.mp4` | `--wss-video` | WSS movie |
| `tke_video.mp4` or `tke_rotate.mp4` | `--tke-video` and TKE available | TKE movie |

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
- `planes.json` stores per-plane `path_info` when path metadata is available; grouped multi-label workflows can therefore expose `path_info.group_name` in that file
- GUI scene objects are not saved files, but grouped workflows name them with stable prefixes such as `segmask_group_<group>`, `skeleton_<group>`, `graph_<group>`, `smooth_path_<group>_<path_idx>`, `plane_<group>_<plane_idx>`, and `pathline_<group>_<plane_idx>`
- `plane_positions.json` stays geometry-focused and does not store browser colors or GUI-only group-title styling
