# Feature: Streamlines And Pathlines

## Status

| Entry point | Status | Notes |
| --- | --- | --- |
| GUI streamlines | Supported | live streamline scene objects |
| GUI pathlines | Supported | time-resolved plane-launched pathlines |
| CLI streamlines video | Partial | offline streamline video export only |
| CLI pathlines | Not implemented | no public batch pathline export |
| Python API | Partial | offline streamline video through batch config |

## What It Does
Streamlines show instantaneous flow trajectories. Pathlines show time-resolved particle travel launched from planes in the GUI. In grouped multi-label workflows, each generated pathline keeps the plane group name internally, while the browser shows short visible names such as `pathline 5`.

## When To Use It
- use streamlines for qualitative inspection of instantaneous flow patterns
- use pathlines for time-resolved trajectories from a plane in the GUI
- use CLI or Python API when you need offline streamline videos

## Quick Use

### GUI
1. load a segmented case
2. click `Generate Streamlines` for live streamlines
3. select a plane and click `Pathlines` to generate one grouped pathline object per plane
4. use the left browser to show or hide a whole group of pathlines, or right-click an individual pathline and use `Set Pathline Color` to change only that pathline

### CLI

```bash
autoflow-run case.h5 --output-dir results/case --video streamlines
```

### Python API

```python
from autoflow import AutoFlowConfig, run_case
config = AutoFlowConfig(
    output_dir="./results/case",
    requested_videos=["streamlines"],
)
summary = run_case("case.h5", config=config)
```

## Inputs

| Input | Required | Meaning |
| --- | --- | --- |
| flow | yes | velocity field |
| segmentation | yes | seed-mask source |
| planes | for pathlines | selected launch planes |
| RR interval | for pathlines | time step scaling |

## Parameters

| Parameter | Type | Default | Where set | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `seed_ratio` | float | `0.02` | `configs/streamlines.json` | streamline seed density | `autoflow/core/models.py` |
| `max_steps` | int | `2000` | `configs/streamlines.json` | integration steps | `autoflow/core/models.py` |
| `min_seeds` | int | `50` | `configs/streamlines.json` | minimum number of seeds | `autoflow/core/models.py` |
| `terminal_speed` | float | `0.01` | `configs/streamlines.json` | stop threshold | `autoflow/core/models.py` |
| `rng_seed` | int | `0` | `configs/streamlines.json` | deterministic seed generation | `autoflow/core/models.py` |
| `tube_radius` | float | `0.05` | `configs/streamlines.json` | rendered tube thickness | `autoflow/core/models.py` |
| `pathline_color` | string | `deepskyblue` | `configs/streamlines.json` or GUI | default color for newly generated pathlines; each pathline can later be recolored individually in the browser | `autoflow/ui/app.py` |
| `streamlines.render.clim` | list[float, float] | `[0.0, 1.0]` | `configs/streamlines.json` | streamline display range in GUI and videos | `autoflow/rendering/videos.py`, `autoflow/core/pipeline.py` |
| `streamlines.render.show_scalar_bar` | bool | `True` | `configs/streamlines.json` | show or hide the streamline colorbar in GUI and videos | `autoflow/rendering/videos.py`, `autoflow/core/pipeline.py` |

## Outputs

| Output file or object | Created when | Meaning |
| --- | --- | --- |
| live streamline scene object | GUI streamline step succeeds | dynamic streamline object |
| live pathline scene objects | GUI pathline step succeeds | per-plane grouped pathline objects, each with its own visibility and color |
| `streamlines_video.mp4` or `streamlines_rotate.mp4` | streamlines video export runs | offline streamline movie |

## Limitations
- segmentation is required
- public batch pathline export is not implemented
- pathlines are currently a GUI-centered workflow

## Where To Change Code

| Change you want | Edit here | Also check | Tests |
| --- | --- | --- | --- |
| streamline generation | `autoflow/algorithms/streamlines.py` | `autoflow/core/pipeline.py` | `tests/test_smoke_phantoms.py` |
| pathline behavior | `autoflow/algorithms/streamlines.py`, `autoflow/ui/app.py` | `autoflow/ui/viewer.py` | manual verification plus smoke and phantom regression only |
| streamline video export | `autoflow/rendering/videos.py` | `autoflow/processing.py` | `tests/test_smoke_phantoms.py` |

## Tests

- `~/miniconda3/envs/ryy/bin/python -m pytest tests/test_smoke_phantoms.py -q`
- GUI pathline behavior outside this retained regression suite requires manual verification

## Common Problems

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| streamlines are skipped | no segmentation or no flow | load a segmented flow case |
| pathlines are skipped | no planes | generate planes first |
| pathlines look mixed across vessels in the browser | grouped labels were not configured as expected | check `configs/labels.json -> label_groups` and regenerate planes and pathlines |
