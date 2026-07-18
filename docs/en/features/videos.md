# Feature: Offline Videos

## Status

| Entry point | Status | Notes |
| --- | --- | --- |
| CLI | Supported | main export path |
| Python API | Supported | same rendering path as CLI |
| GUI | Supported | selective export through `Export > Export Videos...` |

## What It Does
Offline video rendering exports rotating or time-resolved MP4 files for planes, WSS, TKE, pressure gradient, relative pressure, and streamlines. In grouped vessel workflows, plane videos color each rendered centerline path with the configured group path color from `configs/labels.json`. Plane rotation videos annotate `planeidx=<index>` by default so the rendered label matches the saved plane index, and `configs/video_exporting.json -> plane_video.label` controls the label prefix, font size, text color, and label background styling.

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
Use `Export > Export Videos...`, choose an output directory, then select any combination of `plane`, `wss`, `tke`, `pg`, and `streamlines`. The GUI computes missing derived data for WSS, TKE, and pressure-analysis exports before rendering. Plane videos read per-group skeleton color plus plane size, color, and opacity from `configs/video_exporting.json -> plane_video.groups`. Plane-video index label prefix, font size, text color, and label background styling come from `configs/video_exporting.json -> plane_video.label`. Live GUI plane objects read `configs/planes.json -> render`.

## Inputs

| Input | Required | Meaning |
| --- | --- | --- |
| planes | for plane video | plane objects to render |
| derived metrics | for WSS, TKE, pressure-gradient, or relative-pressure videos | computed derived fields |
| flow and segmentation | for streamline video | streamline source data |

## Parameters

| Parameter | Type | Default | Where set | Effect |
| --- | --- | --- | --- | --- |
| `requested_videos` / `--video` | csv | empty | command line or `AutoFlowConfig` | enable one or more of `plane`, `wss`, `tke`, `pg`, `streamlines` |
| `Export > Export Videos...` | GUI action | none | GUI menu | choose output directory and selected video items interactively |
| `plane_video.show_skeleton` | bool | `True` | `configs/video_exporting.json` | show or hide skeleton points in plane videos |
| `plane_video.skeleton_point_size` | float | `10.0` | `configs/video_exporting.json` | skeleton point size in plane videos |
| `plane_video.default.skeleton_color` | string | empty | `configs/video_exporting.json` | fallback skeleton color when a group override is missing |
| `plane_video.default.plane_size` | float or null | `null` | `configs/video_exporting.json` | fallback plane size; `null` uses the automatic scene-based size |
| `plane_video.default.plane_color` | string | `yellow` | `configs/video_exporting.json` | fallback plane color when a group override is missing |
| `plane_video.default.plane_opacity` | float | `0.75` | `configs/video_exporting.json` | fallback plane opacity |
| `plane_video.label.prefix` | string | `planeidx=` | `configs/video_exporting.json` | plane-video index label prefix before the plane number |
| `plane_video.label.font_size` | int | `28` | `configs/video_exporting.json` | plane-video index label font size |
| `plane_video.label.text_color` | string | `black` | `configs/video_exporting.json` | plane-video index label text color |
| `plane_video.label.shape_color` | string | `yellow` | `configs/video_exporting.json` | plane-video index label background color |
| `plane_video.label.shape_opacity` | float | `0.85` | `configs/video_exporting.json` | plane-video index label background opacity |
| `plane_video.groups.<group>.skeleton_color` | string | group fallback | `configs/video_exporting.json` | per-group skeleton color in the plane video |
| `plane_video.groups.<group>.plane_size` | float or null | `null` | `configs/video_exporting.json` | per-group plane size in the plane video |
| `plane_video.groups.<group>.plane_color` | string | group fallback | `configs/video_exporting.json` | per-group plane color in the plane video |
| `plane_video.groups.<group>.plane_opacity` | float | `0.75` | `configs/video_exporting.json` | per-group plane opacity in the plane video |
| `fps` / `--fps` | int | `12` | `configs/video_exporting.json` | output frame rate |
| `plane_rotation_frames` / `--plane-rotation-frames` | int | `180` | `configs/video_exporting.json` | plane video rotation length |
| `window_size` | list[int, int] | `[1600, 1200]` | `configs/video_exporting.json` | output render size; this is the main replacement for a Matplotlib-style `figsize` |
| `camera_view` / `--camera-view` | string | `right` | `configs/video_exporting.json` | camera preset |
| `camera_distance_scale` / `--camera-distance-scale` | float | `1.5` | `configs/video_exporting.json` | scale camera distance |
| `rotate_dynamic_video` | bool | `True` | `configs/video_exporting.json` | rotate dynamic videos; set `False` to keep WSS, TKE, pressure-gradient, relative-pressure, and streamline videos fixed |
| `dynamic_rotation_frames` | int | `180` | `configs/video_exporting.json` | rotation frame count for dynamic videos |
| `dynamic_rotation_elevation_deg` | float | `10.0` | `configs/video_exporting.json` | dynamic rotation elevation |
| `dynamic_time_repeat` | int | `3` | `configs/video_exporting.json` | repeat each time frame |
| `add_plane_idx` | bool | `True` | `configs/video_exporting.json` | annotate plane indices as `planeidx=<index>` in the plane video |
| `add_path_idx` | bool | `False` | `configs/video_exporting.json` | annotate path indices |
| `wss.render.show_scalar_bar` | bool | `True` | `configs/wss.json` | show or hide the WSS colorbar |
| `tke.render.show_scalar_bar` | bool | `True` | `configs/tke.json` | show or hide the TKE colorbar |
| `pressure_gradient.render.clim` | list[float, float] or `null` | `null` | `configs/pressure_gradient.json` | explicit pressure-gradient color range; `null` keeps the auto range |
| `pressure_gradient.render.show_scalar_bar` | bool | `True` | `configs/pressure_gradient.json` | show or hide the pressure-gradient colorbar |
| `pressure_gradient.render.relative_pressure_clim` | list[float, float] or `null` | `null` | `configs/pressure_gradient.json` | explicit relative-pressure color range; `null` keeps the symmetric auto range |
| `pressure_gradient.render.relative_pressure_show_scalar_bar` | bool | `True` | `configs/pressure_gradient.json` | show or hide the relative-pressure colorbar |
| `streamlines.render.show_scalar_bar` | bool | `True` | `configs/streamlines.json` | show or hide the streamline colorbar |

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

## Where To Change Code

| Change you want | Edit here | Also check | Tests |
| --- | --- | --- | --- |
| video rendering logic | `autoflow/rendering/videos.py` | `autoflow/processing.py` | `tests/test_smoke_phantoms.py` |
| GUI video export wiring | `autoflow/ui/app.py` | `autoflow/rendering/videos.py`, `autoflow/config.py` | GUI manual verification |
| CLI or API video flags | `autoflow/cli.py`, `autoflow/api.py` | `autoflow/config.py` | `tests/test_smoke_phantoms.py` |

## Tests

- `~/miniconda3/envs/ryy/bin/python -m pytest tests/test_smoke_phantoms.py -q`

## Common Problems

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| no video file is produced | the corresponding item was not included in `--video`, `requested_videos`, or the GUI export selection; or MP4 encoding failed | request or select the video explicitly and verify the host can write MP4 through ImageIO FFmpeg instead of a still-image plugin |
| `extract_surface()` got an unexpected keyword argument `algorithm` | the installed `PyVista` release does not support the newer `extract_surface(algorithm=...)` keyword | use the updated build, which falls back to the older `extract_surface()` call automatically |
| repeated `vtkEGLRenderWindow ... Unable to eglMakeCurrent: 12290` lines during GUI export | off-screen export tried to create a second local render context while a display-backed GUI VTK context was already active | rerun with the updated build; if the host still prefers display-backed export, launch `autoflow-gui` with `AUTOFLOW_OFFSCREEN_MODE=display` |
| TKE video is missing | no TKE data | expected for many inputs |
