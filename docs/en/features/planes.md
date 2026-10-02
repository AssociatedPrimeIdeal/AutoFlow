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
2. select a plane in the Browser, or right-click its visible 3-D wireframe; the
   3-D picker checks the plane foreground layer before the PC-MRA volume
3. click `Edit Plane`
4. drag the cyan center handle to move it; generated planes stay on their associated path, while manually added planes can move anywhere
5. drag the orange or yellow in-plane-axis handle to rotate the plane around the other local axis
6. selecting a plane automatically replaces the three vertically stacked standard views with `U × V`, `V × N`, and `U × N` views centered on `Plane.center`; clearing the selection restores axial, coronal, and sagittal views
7. inspect the `U × V` view, which is the selected plane itself. Its colored overlay is the active segmentation sampled with nearest-neighbour interpolation, and its cyan outline is the display-smoothed current-time ROI used for area and flow metrics. A saved manual ROI is shown with a dashed yellow outline
8. select `Through-plane Flow (cm/s)` in the main `Content` menu to display velocity projected onto the selected plane normal in all three plane-orthogonal views
9. use `Settings -> Ortho Viewer Display` to tune display-only contour smoothing and set the default three-view FOV (50% means a 2x view-only zoom); changes do not alter metric areas or crop data
10. select `Edit contour` and draw on `U × V`: a closed stroke creates a contour when none exists; with an existing boundary, a stroke crossing it twice replaces the shorter local boundary arc. Endpoints snap automatically, the stroke is smoothed, and outward/inward routes expand/shrink the selected region without an add/remove mode. Each frame has its own contour operation; invalid joins, multiple crossings, and self-intersections are rejected. Submitting an edit recomputes only that frame's plane metrics; TKE, WSS, pressure, and other derived volumes are not recomputed. `Undo`, `Redo`, and `Reset frame` are available
11. scalar images use linear interpolation, labels use nearest-neighbour sampling, and each viewer gets an automatic robust window/level from its displayed slice. Right-click zooms back out
12. click `Finish Plane Edit`; the ortho view, active pathline, metrics, and saved plane outputs are updated after the interaction ends
13. adjust a selected plane's `Opacity` with the Browser slider (or its context menu), then use `Export -> Export Plane Coordinates...` to export selected Browser planes, or all planes when none are selected
14. load a target case, use `File -> Import Plane Coordinates...`, then choose world, local, or relative-centerline mapping and Replace or Append

When `Calculate && Save Metrics` runs in the GUI, its progress dialog advances
after each plane completes and shows the current plane count.

The Generate Planes panel exposes `Plane Mode`, `Plane Count`, `Anchor`,
`Direction`, `Spacing Mode`, `Spacing Ratio`, and the enabled-by-default
`Segmentation Filter` checkbox.

With the filter enabled, each graph path is densified and sampled against the
majority-voted 3-D label volume.  Non-zero samples are grouped into contiguous
physical-length runs. Graph forks are established from nodes with degree
`>= 3`; flow is then sampled on the filtered path geometry to orient the path.
For a path marked `outgoing` at a fork, runs on the free end half are
considered; for an `incoming` path, runs on the free start half are considered.
The longest eligible run becomes `owner_label`, the path is clipped to that
run, and plane metrics are masked to the same numeric label. This is a
path/plane selection rule, not a skeleton edit, and it cannot correct a source
segmentation that labels a branch as its parent vessel.

After plane placement, generated planes are checked against the rasterized
branch volume. A plane with no cells for its requested branch is omitted from
the plane list and recorded in `plane_qc.json` with
`reason=no_branch_support`. This check uses geometric cell support, not a
numeric flow threshold, so genuinely low-flow planes are not discarded. Path
and graph topology are retained even when all planes for a path are omitted;
the path is reported with `path_status=no_valid_planes`.

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

See [planes parameters](../user/parameters.md#planes), [labels parameters](../user/parameters.md#labels), [CLI flags](../user/cli-parameters.md), and [API fields](../user/api-parameters.md) for complete type/default/unit/effect/owner tables. Dictionary controls are expanded in [Structured parameters](../user/parameter-schemas.md).

## Outputs

| Output file or object | Created when | Meaning |
| --- | --- | --- |
| `planes.json` | planes exist | serialized plane geometry, optional `roi_polygon_uv_mm`, `segmentation_label`, `placement_mode`, and attached summaries when metrics exist |
| `planes.h5` | planes exist | one root group per plane with geometry, optional manual ROI in `payload_json`, `placement_mode`, `label_name`, path metadata, and metrics |
| `plane_positions.json` | planes exist | v2 portable coordinate file containing world/local centers, optional plane-local ROI polygons, and relative path mapping hints |
| `plane_qc.json` | plane generation or metrics run | owner label, confidence, original/retained path lengths, requested/actual counts, and per-plane branch-support decisions |
| plane scene objects | GUI or pipeline plane step succeeds | grouped plane objects with stable internal keys such as `plane_aorta_systemic_branches_5`, shown in the browser as `plane 5` |

`PlaneData.center` and the saved `center` field are local physical millimetres. `center_world` is `center + origin`. Sampling converts local centers to world space only when calling VTK, so a non-zero image origin changes placement in the world scene without changing the sampled cross-section.

For cross-case import, `world` preserves `center_world_mm` and requires registered cases; `local` preserves `center_local_mm`; `path_relative` matches group plus path rank and places the plane at the same fractional distance on the target centerline. Relative mapping recalculates the normal from the target path tangent. Manual planes have no centerline fraction and therefore retain world coordinates.

## Limitations
- depends on graph and path quality
- distance-plane placement is only meaningful when path geometry is stable
- segmentation filtering cannot infer a vessel identity if the source segmentation labels the entire short branch as its parent vessel; inspect `plane_qc.json` for low owner-label confidence
- planes with no branch-supported cross-section are intentionally omitted; if every plane on a path is omitted, the graph path remains for topology but has no plane metrics
- junction-spacing uses the closest graph fork attached to the path; paths without a fork use the configured fallback endpoint
- plane editing is GUI-only
- generated plane centers remain constrained to their associated path; use `Add Plane` for unrestricted placement
- plane metrics are recomputed only when a drag finishes, so the metric panel can briefly show the previous values during an active drag
- selected-plane zoom is display-only; it never changes segmentation or the metric ROI
- the selected-plane overlay is sampled from the active mask at the selected time frame with nearest-neighbour interpolation; the cyan metric-ROI contour may apply display smoothing, but the underlying calculation mask is unchanged
- AutoFlow world coordinates are canonical physical millimetres based on origin and spacing, not a general DICOM registration transform

## Where To Change Code

| Change you want | Edit here | Also check | Tests |
| --- | --- | --- | --- |
| plane generation algorithm | `autoflow/algorithms/planes.py` | `autoflow/core/pipeline.py` | `tests/test_smoke_phantoms.py`, `tests/test_pressure_gradient_phantom.py` |
| plane serialization | `autoflow/plane_io.py` | `autoflow/core/pipeline.py` | `tests/test_pressure_gradient_phantom.py` |
| plane geometry editing | `autoflow/ui/app.py` | `autoflow/ui/ortho_viewer.py`, `autoflow/algorithms/paths.py` | GUI manual verification plus smoke coverage |

## Tests

- `/home/renyuyang/miniconda3/envs/autoflow311/bin/python -m pytest tests/test_smoke_phantoms.py -q`
- `/home/renyuyang/miniconda3/envs/autoflow311/bin/python -m pytest tests/test_pressure_gradient_phantom.py -q`

## Common Problems

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| no planes are created | graph or paths are missing | run skeleton and graph first |
| plane positions look poor | path geometry is noisy | improve graph quality or adjust planes manually in the GUI |
| planes are grouped differently than expected | grouped label configuration does not match the label mask | review `configs/labels.json -> label_groups` and regenerate graph and planes |
