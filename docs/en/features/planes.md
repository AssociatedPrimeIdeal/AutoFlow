# Feature: Planes

## Status

| Entry point | Status | Notes |
| --- | --- | --- |
| GUI | Supported | includes path-constrained generated planes, free manual planes, and direct 3D editing |
| CLI | Supported | uniform and fixed-step layouts, with topology-aware segmentation filtering |
| Python API | Supported | through config and pipeline execution |

## What It Does
Plane generation creates cross-sectional analysis planes along vessel paths.
In grouped multi-label workflows, every plane keeps the group name of the path it came from, so plane and pathline objects can stay grouped in the GUI while still showing short browser-visible names such as `plane 5`. Saved plane outputs now include both `planes.json` and `planes.h5`, and the H5 file stores one root group per plane with the same per-plane payload.

## When To Use It
- use it after graph and path generation
- use evenly-spaced mode with `plane_count=1` for one representative plane per path
- use evenly-spaced mode with `plane_count>1` for repeated analysis distributed across a path
- use fixed-step mode with an anchor, direction, and physical or fractional spacing
- use junction-spacing mode to step away from the nearest graph junction along its branch
- use GUI editing when a generated plane needs path-position or orientation correction
- use `Add Plane` when the measurement plane must be placed independently of every generated path

## Quick Use

### GUI
1. run `Generate Graph`, then click `Generate Planes`, or click `Add Plane` to create a free plane at the current ortho cursor
2. select a plane in the browser or 3D view
3. click `Edit Plane`
4. drag the cyan center handle to move it; generated planes stay on their associated path, while manually added planes can move anywhere
5. drag the orange or yellow in-plane-axis handle to rotate the plane around the other local axis
6. click `Finish Plane Edit`; the ortho view, active pathline, metrics, and saved plane outputs are updated after the interaction ends
7. use `Export -> Export Plane Coordinates...` to export selected Browser planes, or all planes when none are selected
8. load a target case, use `File -> Import Plane Coordinates...`, then choose world, local, or relative-centerline mapping and Replace or Append

The Generate Planes panel exposes `Plane Mode`, `Plane Count`, `Anchor`,
`Direction`, `Spacing Mode`, `Spacing Ratio`, and the enabled-by-default
`Segmentation Filter` checkbox.

### CLI

```bash
autoflow-run case.h5 --output-dir results/case --plane-mode fixed_step \
  --plane-anchor center --plane-direction both --plane-count 3 \
  --plane-spacing-mode fraction --plane-spacing-ratio 0.25
```

Cross-case transfer:

```bash
autoflow-run target.h5 --output-dir results/target \
  --import-planes results/source/plane_positions.json \
  --plane-import-mode path_relative \
  --export-planes results/target/transferred_planes.json
```

### Python API

```python
from autoflow import AutoFlowConfig, run_case
config = AutoFlowConfig(
    output_dir="./results/case",
    plane_mode="fixed_step",
    plane_anchor="center",
    plane_direction="both",
    plane_count=3,
    plane_spacing_mode="fraction",
    plane_spacing_ratio=0.25,
)
summary = run_case("case.h5", config=config)
```

## Inputs

| Input | Required | Meaning |
| --- | --- | --- |
| graph and paths | for generated planes | geometry used to place and constrain generated planes |
| loaded volume and ortho cursor | for manual planes | defines the initial free-plane position |
| resolution | yes | spacing-aware geometry operations |

## Parameters

| Parameter | Type | Default | Where set | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `plane_mode` | string | `fixed_step` | `configs/planes.json` or CLI/API | `uniform` distributes planes; `fixed_step` uses anchor, direction, and spacing | `autoflow/algorithms/planes.py` |
| `plane_count` | int | `3` | `configs/planes.json` or CLI/API | requested planes; `-1` fills positions that fit. Even symmetric counts omit the center plane | `autoflow/algorithms/planes.py` |
| `cross_section_distance` / `cross_section_dist` | float mm | `5.0` | `configs/planes.json` or CLI/API | fixed-step spacing when spacing mode is distance | `autoflow/algorithms/planes.py` |
| `spacing_mode` / `plane_spacing_mode` | string | `fraction` | `configs/planes.json` or CLI/API | use `distance` (mm) or `fraction` of the usable centerline length | `autoflow/algorithms/planes.py` |
| `spacing_ratio` / `plane_spacing_ratio` | float | `0.25` | `configs/planes.json` or CLI/API | fraction of the current filtered centerline length per step | `autoflow/algorithms/planes.py` |
| `direction` / `plane_direction` | string | `both` | `configs/planes.json` or CLI/API | `toward_start`, `toward_end`, or `both` | `autoflow/algorithms/planes.py` |
| `segmentation_filter` | bool | `true` | `configs/planes.json` or CLI/API | topology-aware owner label selection, path clipping, and metric mask filtering | `autoflow/algorithms/planes.py`, `autoflow/algorithms/metrics.py` |
| `start_distance` / `start_dist` | float mm | `0.0` | `configs/planes.json` or CLI/API | advanced trim from path start before placement | `autoflow/algorithms/planes.py` |
| `end_distance` / `end_dist` | float mm | `0.0` | `configs/planes.json` or CLI/API | offset from path end | `autoflow/algorithms/planes.py` |
| `anchor` / `plane_anchor` | string | `center` | `configs/planes.json` or CLI/API | `start`, `center`, `end`, or `junction` placement anchor | `autoflow/algorithms/planes.py` |
| `anchor_offset_mm` / `plane_offset_mm` | float mm | `5.0` | `configs/planes.json` or CLI/API | first distance from the junction in `anchored_offset` mode | `autoflow/algorithms/planes.py` |
| `smoothing_window` | int | `15` | `configs/planes.json` | path smoothing window | `autoflow/core/models.py` |
| `smoothing_polyorder` | int | `2` | `configs/planes.json` | smoothing polynomial order | `autoflow/core/models.py` |
| `inter_time` | int | `10` | `configs/planes.json` | interpolation time setting used by plane code | `autoflow/core/models.py` |
| `--import-planes` / `reuse_planes` | path | empty | CLI or `AutoFlowConfig` | load a v2 or legacy plane-coordinate file before metrics | `autoflow/plane_io.py` |
| `--plane-import-mode` / `plane_import_mode` | string | `world` | CLI or `AutoFlowConfig` | choose world, local, or relative-centerline cross-case mapping | `autoflow/plane_io.py` |
| `--export-planes` / `export_planes` | path | empty | CLI or `AutoFlowConfig` | write an additional portable coordinate file | `autoflow/processing.py` |

## Outputs

| Output file or object | Created when | Meaning |
| --- | --- | --- |
| `planes.json` | planes exist | serialized plane geometry, `segmentation_label`, `placement_mode`, and attached summaries when metrics exist |
| `planes.h5` | planes exist | one root group per plane with geometry, `placement_mode`, `label_name`, path metadata, metrics, and `payload_json` |
| `plane_positions.json` | planes exist | v2 portable coordinate file containing world/local centers and relative path mapping hints |
| `plane_qc.json` | plane generation or metrics run | owner label, confidence, original/retained path lengths, and requested/actual plane counts |
| plane scene objects | GUI or pipeline plane step succeeds | grouped plane objects with stable internal keys such as `plane_aorta_systemic_branches_5`, shown in the browser as `plane 5` |

`PlaneData.center` and the saved `center` field are local physical millimetres. `center_world` is `center + origin`. Sampling converts local centers to world space only when calling VTK, so a non-zero image origin changes placement in the world scene without changing the sampled cross-section.

For cross-case import, `world` preserves `center_world_mm` and requires registered cases; `local` preserves `center_local_mm`; `path_relative` matches group plus path rank and places the plane at the same fractional distance on the target centerline. Relative mapping recalculates the normal from the target path tangent. Manual planes have no centerline fraction and therefore retain world coordinates.

## Limitations
- depends on graph and path quality
- distance-plane placement is only meaningful when path geometry is stable
- segmentation filtering cannot infer a vessel identity if the source segmentation labels the entire short branch as its parent vessel; inspect `plane_qc.json` for low owner-label confidence
- junction-spacing uses the closest graph fork attached to the path; paths without a fork use the configured fallback endpoint
- plane editing is GUI-only
- generated plane centers remain constrained to their associated path; use `Add Plane` for unrestricted placement
- plane metrics are recomputed only when a drag finishes, so the metric panel can briefly show the previous values during an active drag
- AutoFlow world coordinates are canonical physical millimetres based on origin and spacing, not a general DICOM registration transform

## Where To Change Code

| Change you want | Edit here | Also check | Tests |
| --- | --- | --- | --- |
| plane generation algorithm | `autoflow/algorithms/planes.py` | `autoflow/core/pipeline.py` | `tests/test_smoke_phantoms.py`, `tests/test_pressure_gradient_phantom.py` |
| plane serialization | `autoflow/plane_io.py` | `autoflow/core/pipeline.py` | `tests/test_pressure_gradient_phantom.py` |
| plane geometry editing | `autoflow/ui/app.py` | `autoflow/ui/ortho_viewer.py`, `autoflow/algorithms/paths.py` | GUI manual verification plus smoke coverage |

## Tests

- `~/miniconda3/envs/ryy/bin/python -m pytest tests/test_smoke_phantoms.py -q`
- `~/miniconda3/envs/ryy/bin/python -m pytest tests/test_pressure_gradient_phantom.py -q`

## Common Problems

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| no planes are created | graph or paths are missing | run skeleton and graph first |
| plane positions look poor | path geometry is noisy | improve graph quality or adjust planes manually in the GUI |
| planes are grouped differently than expected | grouped label configuration does not match the label mask | review `configs/labels.json -> label_groups` and regenerate graph and planes |
