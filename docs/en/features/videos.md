# Feature: Offline Videos

## Status

| Entry point | Status | Notes |
| --- | --- | --- |
| CLI | Supported | main export path |
| Python API | Supported | same rendering path as CLI |
| GUI | Supported | selective export through `Export > Export Videos...` |

## What It Does
Offline video rendering exports rotating or time-resolved MP4 files for planes, WSS, TKE, pressure gradient, and streamlines. In grouped vessel workflows, plane videos color each rendered centerline path with the configured group path color from `configs/labels.json`.

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
Use `Export > Export Videos...`, choose an output directory, then select any combination of `plane`, `wss`, `tke`, `pg`, and `streamlines`. The GUI computes missing derived data for WSS, TKE, and pressure-gradient exports before rendering. Plane videos and live GUI plane objects read per-group skeleton color plus plane size, color, and opacity from `configs/planes.json -> render.groups`.

## Inputs

| Input | Required | Meaning |
| --- | --- | --- |
| planes | for plane video | plane objects to render |
| derived metrics | for WSS, TKE, or pressure-gradient videos | computed derived fields |
| flow and segmentation | for streamline video | streamline source data |

## Parameters

| Parameter | Type | Default | Where set | Effect |
| --- | --- | --- | --- | --- |
| `requested_videos` / `--video` | csv | empty | command line or `AutoFlowConfig` | enable one or more of `plane`, `wss`, `tke`, `pg`, `streamlines` |
| `Export > Export Videos...` | GUI action | none | GUI menu | choose output directory and selected video items interactively |
| `planes.render.show_skeleton` | bool | `True` | `configs/planes.json` | show or hide skeleton points in plane videos |
| `planes.render.skeleton_point_size` | float | `10.0` | `configs/planes.json` | skeleton point size in plane videos |
| `planes.render.default.skeleton_color` | string | empty | `configs/planes.json` | fallback skeleton color when a group override is missing |
| `planes.render.default.plane_size` | float or null | `null` | `configs/planes.json` | fallback plane size; `null` uses the automatic scene-based size |
| `planes.render.default.plane_color` | string | `yellow` | `configs/planes.json` | fallback plane color when a group override is missing |
| `planes.render.default.plane_opacity` | float | `0.75` | `configs/planes.json` | fallback plane opacity |
| `planes.render.groups.<group>.skeleton_color` | string | group fallback | `configs/planes.json` | per-group skeleton color in the plane video |
| `planes.render.groups.<group>.plane_size` | float or null | `null` | `configs/planes.json` | per-group plane size in the plane video |
| `planes.render.groups.<group>.plane_color` | string | group fallback | `configs/planes.json` | per-group plane color in the plane video and GUI planes |
| `planes.render.groups.<group>.plane_opacity` | float | `0.75` | `configs/planes.json` | per-group plane opacity in the plane video and GUI planes |
| `fps` / `--fps` | int | `12` | `configs/rendering.json` | output frame rate |
| `plane_rotation_frames` / `--plane-rotation-frames` | int | `180` | `configs/rendering.json` | plane video rotation length |
| `window_size` | list[int, int] | `[1600, 1200]` | `configs/rendering.json` | output render size; this is the main replacement for a Matplotlib-style `figsize` |
| `camera_view` / `--camera-view` | string | `right` | `configs/rendering.json` | camera preset |
| `camera_distance_scale` / `--camera-distance-scale` | float | `1.5` | `configs/rendering.json` | scale camera distance |
| `rotate_dynamic_video` | bool | `True` | `configs/rendering.json` | rotate dynamic videos; set `False` to keep WSS, TKE, pressure-gradient, and streamline videos fixed |
| `dynamic_rotation_frames` | int | `180` | `configs/rendering.json` | rotation frame count for dynamic videos |
| `dynamic_rotation_elevation_deg` | float | `10.0` | `configs/rendering.json` | dynamic rotation elevation |
| `dynamic_time_repeat` | int | `3` | `configs/rendering.json` | repeat each time frame |
| `add_plane_idx` | bool | `False` | `configs/rendering.json` | annotate plane indices |
| `add_path_idx` | bool | `False` | `configs/rendering.json` | annotate path indices |
| `wss.render.show_scalar_bar` | bool | `True` | `configs/wss.json` | show or hide the WSS colorbar |
| `tke.render.show_scalar_bar` | bool | `True` | `configs/tke.json` | show or hide the TKE colorbar |
| `pressure_gradient.render.clim` | list[float, float] | `[0.0, 500.0]` | `configs/pressure_gradient.json` | explicit pressure-gradient color range |
| `pressure_gradient.render.show_scalar_bar` | bool | `True` | `configs/pressure_gradient.json` | show or hide the pressure-gradient colorbar |
| `streamlines.render.show_scalar_bar` | bool | `True` | `configs/streamlines.json` | show or hide the streamline colorbar |

## Outputs

| Output file | Created when | Meaning |
| --- | --- | --- |
| `planes_rotate.mp4` | `plane` video requested | rotating plane overview with per-group skeleton and plane styling plus per-group centerline colors when grouped paths exist |
| `wss_video.mp4` or `wss_rotate.mp4` | `wss` video requested | WSS movie |
| `pressure_gradient_video.mp4` or `pressure_gradient_rotate.mp4` | `pg` video requested | pressure-gradient movie |
| `streamlines_video.mp4` or `streamlines_rotate.mp4` | `streamlines` video requested | streamline movie |
| `tke_video.mp4` or `tke_rotate.mp4` | `tke` video requested and TKE exists | TKE movie |
| `summary.json` updates | GUI export runs in an output directory that already has results | refreshed `videos`, refreshed `video_times_sec`, and a `gui_video_export` record |

## Limitations
- videos only render successfully if upstream data exists
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
| no video file is produced | the corresponding item was not included in `--video`, `requested_videos`, or the GUI export selection | request or select the video explicitly |
| repeated `vtkEGLRenderWindow ... Unable to eglMakeCurrent: 12290` lines during GUI export | off-screen export tried to create a second local render context while a display-backed GUI VTK context was already active | rerun with the updated build; if the host still prefers display-backed export, launch `autoflow-gui` with `AUTOFLOW_OFFSCREEN_MODE=display` |
| TKE video is missing | no TKE data | expected for many inputs |
