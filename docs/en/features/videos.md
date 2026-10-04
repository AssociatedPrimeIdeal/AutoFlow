# Feature: Offline Videos

## Status

| Entry point | Status | Notes |
| --- | --- | --- |
| CLI | Supported | main export path |
| Python API | Supported | same rendering path as CLI |
| GUI | Supported | selective export through `Export > Export Videos...` |

## What It Does
Offline video rendering exports rotating or time-resolved MP4 files for planes, WSS, TKE, pressure gradient, relative pressure, and streamlines. In grouped vessel workflows, plane videos color each rendered centerline path with the configured group path color from `configs/labels.json`. Plane rotation videos annotate `planeidx=<index>` by default so the rendered label matches the saved plane index, and `configs/video_exporting.json -> plane_video.label` controls the label prefix, font size, text color, and label background styling. Frames are encoded as they are rendered; exports do not retain the full RGB frame sequence. Complete MP4s replace the destination atomically, retaining an existing complete video on encoding failure or cancellation. Dynamic exports still cache per-timepoint streamlines, tube geometry, and sampled TKE or pressure meshes. Streamline phases are prepared in parallel, identical pressure-support phases share one smoothed surface, and VTK actors stay attached while their mapper input and camera are updated for each frame.

## When To Use It
- use it for reports, demos, or review packages
- use it after the required upstream data exists
- use CLI or Python API for repeatable exports

## Quick Use

### CLI

```bash
autoflow-run case.h5 \
  --output-dir results/case \
  --with wss,pg \
  --video plane,wss,pg
```

### Python API

```python
from autoflow import AutoFlowConfig, run_case
config = AutoFlowConfig(
    output_dir="./results/case",
    requested_metrics=["wss", "pg"],
    requested_videos=["plane", "wss", "pg"],
)
summary = run_case("case.h5", config=config)
```

### GUI
Use `Export > Export Videos...`, choose an output directory, then select any combination of `plane`, `wss`, `tke`, `pg`, and `streamlines`. The GUI computes missing derived data in a background task, then runs VTK/FFmpeg in an isolated rendering process. Progress shows completed videos and rendered frames while the activity dots continue independently. × stops that task process and its encoder; the workspace unlocks after exit. Completed videos are published from a temporary directory only after successful rendering. Plane videos read per-group skeleton color plus plane size, color, and opacity from `configs/video_exporting.json -> plane_video.groups`. Plane-video index label prefix, font size, text color, and label background styling come from `configs/video_exporting.json -> plane_video.label`. Live GUI plane objects read `configs/planes.json -> render`.

## Inputs

| Input | Required | Meaning |
| --- | --- | --- |
| planes | for plane video | plane objects to render |
| derived metrics | for WSS, TKE, pressure-gradient, or relative-pressure videos | computed derived fields |
| flow and segmentation | for streamline video | streamline source data |

## Parameters

See [video exporting parameters](../user/parameters.md#video_exporting), [colorbar parameters](../user/parameters.md#colorbar), [CLI flags](../user/cli-parameters.md), and [API fields](../user/api-parameters.md) for complete type/default/unit/effect/owner tables. Dictionary controls are expanded in [Structured parameters](../user/parameter-schemas.md).

## Outputs

| Output file | Created when | Meaning |
| --- | --- | --- |
| `planes_rotate.mp4` | `plane` video requested | rotating plane overview with per-group skeleton and plane styling, per-group centerline colors, and configurable index labels that default to `planeidx=<index>`; MP4 only |
| `wss_video.mp4` or `wss_rotate.mp4` | `wss` video requested | WSS movie |
| `pressure_gradient_video.mp4` or `pressure_gradient_rotate.mp4` | `pg` video requested | pressure-gradient movie |
| `relative_pressure_video.mp4` or `relative_pressure_rotate.mp4` | `pg` video requested | relative-pressure movie |
| `streamlines_video.mp4` or `streamlines_rotate.mp4` | `streamlines` video requested | streamline movie |
| `tke_video.mp4` or `tke_rotate.mp4` | `tke` video requested and TKE exists | TKE movie |
| `summary.json` updates | GUI export runs in an output directory that already has results | refreshed `videos`, refreshed `video_times_sec`, and a `gui_video_export` record |

## Limitations
- videos only render successfully if upstream data exists
- offline export writes MP4 only through ImageIO FFmpeg; no GIF fallback is generated when MP4 encoding fails
- TKE video stays unavailable when TKE is unavailable
- pathlines are not currently exported as a separate batch video feature
- dynamic geometry/field caches remain alive for the duration of one export, using additional memory and up to eight streamline preparation workers to reduce repeated geometry and sampling work

GUI process isolation and task cleanup are owned by `autoflow/rendering/jobs.py`, `autoflow/task_control.py`, and `autoflow/ui/progress.py`. Shared streaming encoding and renderer generators remain in `autoflow/rendering/videos.py`.

## Where To Change Code

| Change you want | Edit here | Also check | Tests |
| --- | --- | --- | --- |
| video rendering logic | `autoflow/rendering/videos.py` | `autoflow/processing.py` | `tests/test_smoke_phantoms.py` |
| GUI video export wiring | `autoflow/ui/app.py` | `autoflow/rendering/videos.py`, `autoflow/config.py` | GUI manual verification |
| CLI or API video flags | `autoflow/cli.py`, `autoflow/api.py` | `autoflow/config.py` | `tests/test_smoke_phantoms.py` |

## Tests

- `/home/renyuyang/miniconda3/envs/autoflow311/bin/python -m pytest tests/test_smoke_phantoms.py -q`

## Common Problems

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| no video file is produced | the corresponding item was not included in `--video`, `requested_videos`, or the GUI export selection; or MP4 encoding failed | request or select the video explicitly and verify the host can write MP4 through ImageIO FFmpeg instead of a still-image plugin |
| `extract_surface()` got an unexpected keyword argument `algorithm` | the installed `PyVista` release does not support the newer `extract_surface(algorithm=...)` keyword | use the updated build, which falls back to the older `extract_surface()` call automatically |
| repeated `vtkEGLRenderWindow ... Unable to eglMakeCurrent: 12290` lines during GUI export | off-screen export tried to create a second local render context while a display-backed GUI VTK context was already active | rerun with the updated build; if the host still prefers display-backed export, launch `autoflow-gui` with `AUTOFLOW_OFFSCREEN_MODE=display` |
| TKE video is missing | no TKE data | expected for many inputs |
