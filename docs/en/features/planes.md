# Feature: Planes

## Status

| Entry point | Status | Notes |
| --- | --- | --- |
| GUI | Supported | includes live drag editing |
| CLI | Supported | center-plane and distance-plane modes |
| Python API | Supported | through config and pipeline execution |

## What It Does
Plane generation creates cross-sectional analysis planes along vessel paths.
In grouped multi-label workflows, every plane keeps the group name of the path it came from, so plane and pathline objects can stay grouped in the GUI.

## When To Use It
- use it after graph and path generation
- use center-plane mode for one representative plane per path
- use distance-plane mode for repeated analysis along a path
- use GUI drag editing when a generated plane needs manual adjustment

## Quick Use

### GUI
1. run `Generate Graph`
2. click `Generate Planes`
3. select a grouped plane in the browser or 3D view
4. drag the center or normal widgets if manual correction is needed

### CLI

```bash
autoflow-run case.h5 --output-dir results/case --plane-by-distance --cross-section-dist 15
```

### Python API

```python
from autoflow import AutoFlowConfig, run_case
config = AutoFlowConfig(
    output_dir="./results/case",
    use_center_plane=False,
    cross_section_dist=15.0,
)
summary = run_case("case.h5", config=config)
```

## Inputs

| Input | Required | Meaning |
| --- | --- | --- |
| graph and paths | yes | geometry used to place planes |
| resolution | yes | spacing-aware geometry operations |

## Parameters

| Parameter | Type | Default | Where set | Effect |
| --- | --- | --- | --- | --- |
| `use_center_plane` | bool | `True` | `configs/planes.json` or CLI/API | one plane per path when true |
| `cross_section_distance` / `cross_section_dist` | float mm | `5.0` | `configs/planes.json` or CLI/API | plane spacing in distance mode |
| `start_distance` / `start_dist` | float mm | `5.0` | `configs/planes.json` or CLI/API | offset from path start |
| `end_distance` / `end_dist` | float mm | `0.0` | `configs/planes.json` or CLI/API | offset from path end |
| `smoothing_window` | int | `15` | `configs/planes.json` | path smoothing window |
| `smoothing_polyorder` | int | `2` | `configs/planes.json` | smoothing polynomial order |
| `inter_time` | int | `10` | `configs/planes.json` | interpolation time setting used by plane code |

## Outputs

| Output file or object | Created when | Meaning |
| --- | --- | --- |
| `planes.json` | planes exist | serialized plane geometry |
| `plane_positions.json` | planes exist | reusable plane position file |
| plane scene objects | GUI or pipeline plane step succeeds | grouped plane objects such as `plane_aorta_systemic_branches_5` |

## Limitations
- depends on graph and path quality
- distance-plane placement is only meaningful when path geometry is stable
- plane editing is GUI-only

## Where To Change Code

| Change you want | Edit here | Also check | Tests |
| --- | --- | --- | --- |
| plane generation algorithm | `autoflow/algorithms/planes.py` | `autoflow/core/pipeline.py` | `tests/test_smoke_phantoms.py`, `tests/test_pressure_gradient_phantom.py` |
| plane serialization | `autoflow/plane_io.py` | `autoflow/core/pipeline.py` | `tests/test_pressure_gradient_phantom.py` |
| plane drag editing | `autoflow/ui/app.py` | `autoflow/ui/ortho_viewer.py` | GUI smoke coverage |

## Tests

- `~/miniconda3/envs/ryy/bin/python -m pytest tests/test_smoke_phantoms.py -q`
- `~/miniconda3/envs/ryy/bin/python -m pytest tests/test_pressure_gradient_phantom.py -q`

## Common Problems

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| no planes are created | graph or paths are missing | run skeleton and graph first |
| plane positions look poor | path geometry is noisy | improve graph quality or adjust planes manually in the GUI |
| planes are grouped differently than expected | grouped label configuration does not match the label mask | review `configs/skeleton.json -> label_groups` and regenerate graph and planes |
