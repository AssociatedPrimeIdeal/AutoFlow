# Feature: Offline Videos

## Status

| Entry point | Status | Notes |
| --- | --- | --- |
| CLI | Supported | main export path |
| Python API | Supported | same rendering path as CLI |
| GUI | Partial | GUI visualizes content but is not the primary video-export workflow |

## What It Does
Offline video rendering exports rotating or time-resolved MP4 files for planes, WSS, TKE, and streamlines.

## When To Use It
- use it for reports, demos, or review packages
- use it after the required upstream data exists
- use CLI or Python API for repeatable exports

## Quick Use

### CLI

```bash
autoflow-run case.h5 \
  --output-dir results/case \
  --plane-video \
  --wss-video \
  --streamlines-video
```

### Python API

```python
from autoflow import AutoFlowConfig, run_case
config = AutoFlowConfig(
    output_dir="./results/case",
    make_plane_video=True,
    make_wss_video=True,
)
summary = run_case("case.h5", config=config)
```

### GUI
Use the GUI to inspect views before exporting with CLI or Python API.

## Inputs

| Input | Required | Meaning |
| --- | --- | --- |
| planes | for plane video | plane objects to render |
| derived metrics | for WSS or TKE videos | computed derived fields |
| flow and segmentation | for streamline video | streamline source data |

## Parameters

| Parameter | Type | Default | Where set | Effect |
| --- | --- | --- | --- | --- |
| `make_plane_video` / `--plane-video` | bool | `False` | `configs/rendering.json` | enable plane video |
| `make_wss_video` / `--wss-video` | bool | `False` | `configs/rendering.json` | enable WSS video |
| `make_streamlines_video` / `--streamlines-video` | bool | `False` | `configs/rendering.json` | enable streamline video |
| `make_tke_video` / `--tke-video` | bool | `False` | `configs/rendering.json` | enable TKE video |
| `fps` / `--fps` | int | `12` | `configs/rendering.json` | output frame rate |
| `plane_rotation_frames` / `--plane-rotation-frames` | int | `180` | `configs/rendering.json` | plane video rotation length |
| `camera_view` / `--camera-view` | string | `right` | `configs/rendering.json` | camera preset |
| `camera_distance_scale` / `--camera-distance-scale` | float | `1.5` | `configs/rendering.json` | scale camera distance |
| `rotate_dynamic_video` | bool | `True` | `configs/rendering.json` | rotate dynamic videos |
| `dynamic_rotation_frames` | int | `180` | `configs/rendering.json` | rotation frame count for dynamic videos |
| `dynamic_rotation_elevation_deg` | float | `10.0` | `configs/rendering.json` | dynamic rotation elevation |
| `dynamic_time_repeat` | int | `3` | `configs/rendering.json` | repeat each time frame |
| `add_plane_idx` | bool | `False` | `configs/rendering.json` | annotate plane indices |
| `add_path_idx` | bool | `False` | `configs/rendering.json` | annotate path indices |

## Outputs

| Output file | Created when | Meaning |
| --- | --- | --- |
| `planes_rotate.mp4` | plane video enabled | rotating plane overview |
| `wss_video.mp4` or `wss_rotate.mp4` | WSS video enabled | WSS movie |
| `streamlines_video.mp4` or `streamlines_rotate.mp4` | streamline video enabled | streamline movie |
| `tke_video.mp4` or `tke_rotate.mp4` | TKE video enabled and TKE exists | TKE movie |

## Limitations
- videos only render successfully if upstream data exists
- TKE video stays unavailable when TKE is unavailable
- pathlines are not currently exported as a separate batch video feature

## Where To Change Code

| Change you want | Edit here | Also check | Tests |
| --- | --- | --- | --- |
| video rendering logic | `autoflow/rendering/videos.py` | `autoflow/processing.py` | `tests/test_smoke_phantoms.py` |
| CLI or API video flags | `autoflow/cli.py`, `autoflow/api.py` | `autoflow/config.py` | `tests/test_smoke_phantoms.py` |

## Tests

- `~/miniconda3/envs/ryy/bin/python -m pytest tests/test_smoke_phantoms.py -q`

## Common Problems

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| no video file is produced | corresponding `make_*_video` flag is off | enable the flag |
| TKE video is missing | no TKE data | expected for many inputs |
