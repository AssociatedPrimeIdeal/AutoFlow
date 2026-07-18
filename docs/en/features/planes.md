# Feature: Planes

## Status

| Entry point | Status | Notes |
| --- | --- | --- |
| GUI | Supported | includes live drag editing plus count, distance, and anchored-offset placement |
| CLI | Supported | count, distance, and anchored-offset modes |
| Python API | Supported | through config and pipeline execution |

## What It Does
Plane generation creates cross-sectional analysis planes along vessel paths.
In grouped multi-label workflows, every plane keeps the group name of the path it came from, so plane and pathline objects can stay grouped in the GUI while still showing short browser-visible names such as `plane 5`. Saved plane outputs now include both `planes.json` and `planes.h5`, and the H5 file stores one root group per plane with the same per-plane payload.

## When To Use It
- use it after graph and path generation
- use count mode with `plane_count=1` for one representative plane per path
- use count mode with `plane_count>1` for evenly spaced repeated analysis along a path
- use distance mode for fixed-mm spacing along a path
- use anchored-offset mode when the plane must stay a fixed distance from the path start or end
- use GUI drag editing when a generated plane needs manual adjustment

## Quick Use

### GUI
1. run `Generate Graph`
2. click `Generate Planes`
3. select a grouped plane in the browser or 3D view
4. drag the center or normal widgets if manual correction is needed

### CLI

```bash
autoflow-run case.h5 --output-dir results/case --plane-mode distance --cross-section-dist 15
```

### Python API

```python
from autoflow import AutoFlowConfig, run_case
config = AutoFlowConfig(
    output_dir="./results/case",
    plane_mode="distance",
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
| `plane_mode` | string | `count` | `configs/planes.json` or CLI/API | choose `count`, `distance`, or `anchored_offset` placement |
| `plane_count` | int | `1` | `configs/planes.json` or CLI/API | evenly spaced plane count in count mode; `1` gives the center-style default |
| `cross_section_distance` / `cross_section_dist` | float mm | `5.0` | `configs/planes.json` or CLI/API | plane spacing in distance mode |
| `start_distance` / `start_dist` | float mm | `5.0` | `configs/planes.json` or CLI/API | trim from path start before count or distance placement |
| `end_distance` / `end_dist` | float mm | `0.0` | `configs/planes.json` or CLI/API | offset from path end |
| `anchor` / `plane_anchor` | string | `end` | `configs/planes.json` or CLI/API | choose `start` or `end` anchor in anchored-offset mode |
| `anchor_offset_mm` / `plane_offset_mm` | float mm | `5.0` | `configs/planes.json` or CLI/API | offset from the chosen anchor in anchored-offset mode |
| `smoothing_window` | int | `15` | `configs/planes.json` | path smoothing window |
| `smoothing_polyorder` | int | `2` | `configs/planes.json` | smoothing polynomial order |
| `inter_time` | int | `10` | `configs/planes.json` | interpolation time setting used by plane code |

## Outputs

| Output file or object | Created when | Meaning |
| --- | --- | --- |
| `planes.json` | planes exist | serialized plane geometry, `label_name`, and attached summaries when metrics exist |
| `planes.h5` | planes exist | one root group per plane with geometry, `label_name`, path metadata, metrics, and `payload_json` |
| `plane_positions.json` | planes exist | reusable plane position file |
| plane scene objects | GUI or pipeline plane step succeeds | grouped plane objects with stable internal keys such as `plane_aorta_systemic_branches_5`, shown in the browser as `plane 5` |

## Limitations
- depends on graph and path quality
- distance-plane placement is only meaningful when path geometry is stable
- anchored-offset placement uses oriented path start/end, not a post-hoc hemodynamic flow classification
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
| planes are grouped differently than expected | grouped label configuration does not match the label mask | review `configs/labels.json -> label_groups` and regenerate graph and planes |
