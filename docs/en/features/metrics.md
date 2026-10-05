# Feature: Plane Metrics

## Status

| Entry point | Status | Notes |
| --- | --- | --- |
| GUI | Supported | `Calculate && Save Metrics`, `Plane Curve`, and automatic refresh after plane edits |
| CLI | Supported | part of the default batch order |
| Python API | Supported | through `run_case()` and `run_batch()` |

## What It Does

General acquisition, retrospective plane-flow analysis and validation guidance are listed in [Scientific references: 4D flow](../references/index.md#general-4d-flow-guidance). The consensus statements are context, not validation of AutoFlow's numerical outputs.
Plane metrics compute time-resolved cross-sectional measurements for each plane and save them into `plane_metrics.json`. The same per-plane payload is also mirrored into `planes.json` and `planes.h5`, so one plane index has one consistent set of geometry and summaries across outputs. The GUI plane-metric step computes basic flow, area, and velocity without implicitly starting WSS, TKE, or pressure work. If derived arrays already exist, it reuses them. Batch runs compute only the derived fields explicitly requested through `--with` or `requested_metrics`.

For each unique segmentation phase, plane metrics build one thresholded VTK support mesh and reuse it across all planes. Each plane still has its own slice and connectivity selection, while repeated cardiac phases reuse the resulting slice specification. Large parallel jobs use independent loky worker processes with memory-mapped arrays because concurrently slicing a shared VTK dataset from Python threads is unsafe. Plane centers remain local physical coordinates and are shifted by `origin` only for VTK slicing.

Support meshes are constructed inside the occupied mask's bounding box. They retain the full image's voxel identifiers and physical coordinates, so plane sampling and contour edits still address the original volume while avoiding background grid construction.

Generated planes are filtered before metric integration when their requested
branch has no cells in the segmentation-filtered cross-section. The exclusion
criterion is missing geometric support (`area_mm2 == 0` after branch
selection), not a zero numerical flow value. This keeps real low-flow or
forward/reverse-cancelling planes. A path with no remaining valid planes stays
in the graph topology but has no path metric; its path IC is `null`/undefined,
and a fork that is missing a path metric is also reported as incomplete rather
than treating the missing path as zero flow.

## When To Use It
- use it after planes exist
- use it when you need per-plane flow, area, and velocity curves
- use it when you need explicit `plane_index` values that line up with `planes.json`, `planes.h5`, and the plane-rotation video labels
- add `--with wss,tke,pg` if you also want derived per-plane summaries; `--with vortex` computes whole-volume vortex fields and is not a plane summary
- do not expect it to run without segmentation

## Quick Use

### GUI
1. generate planes
2. click `Calculate && Save Metrics`
3. inspect the selected plane in the selection panel, ortho viewer, and `Plane Curve` mode
4. when plane geometry needs correction, use `Edit Plane`; releasing a handle recomputes only that plane and updates the metric and plane records
5. click `Calculate && Save Metrics` after interactive edits when a fully regenerated `plane_metrics_pixelwise.h5` is required

### CLI

```bash
autoflow-run case.h5 --output-dir results/case
```

```bash
autoflow-run case.h5 --output-dir results/case --with pwv,wss,pg
```

### Python API
Use the standard `run_case()` or `run_batch()` flow. Plane metrics run unless `skip_plane_metrics=True`.

## Inputs

| Input | Required | Meaning |
| --- | --- | --- |
| flow | yes | time-resolved velocity field |
| segmentation | yes | vessel mask for valid plane sampling |
| planes | yes | analysis planes |

## Parameters

See [batch parameters](../user/parameters.md#batch), [planes parameters](../user/parameters.md#planes), [CLI flags](../user/cli-parameters.md), and [API fields](../user/api-parameters.md) for complete type/default/unit/effect/owner tables. Dictionary controls are expanded in [Structured parameters](../user/parameter-schemas.md).

Derived plane sampling reuses a slice only for the same support phase and ROI state. Each timepoint uses its own contour edits, including when the volume mask is unchanged. Label-specific support meshes also share geometry across phases when that label's filtered mask is identical.

Low-level Python users can call `augment_plane_metrics_with_derived(...)` with:

| Parameter | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `use_multithread` | bool | `False` | function keyword; pipeline uses its plane-computation setting | Automatically use up to four isolated processes at 1920 plane-phases or more | `autoflow/algorithms/metrics/plane_derived.py` |
| `max_workers` | int or None | `None` | function keyword | Override derived-sampling process count; one selects serial | same |
| `progress_callback` | callable or None | `None` | function keyword | Receive completed-plane dictionaries on the calling thread | same |


## Outputs

Plane pressure statistics use the valid pressure-derivative support, excluding zero padding outside it. Unavailable phases are JSON `null`; valid measured zero remains `0`. Cycle summaries average available phases, and `pressure_gradient_valid_cell_count_t` / `relative_pressure_valid_cell_count_t` expose per-phase sample counts. Plane pixelwise H5 adds `pressure_gradient_valid` and `relative_pressure_valid` flags; invalid pressure samples are NaN.


| Output file or object | Created when | Meaning |
| --- | --- | --- |
| `plane_metrics.json` | plane metrics run | one metric record per plane with explicit `plane_index` and time-resolved fields |
| `plane_qc.json` | plane metrics run | QC for paths and forks |
| `plane_metrics_pixelwise.h5` | plane metrics run | per-plane slice-cellwise derived samples |
| `planes.json` | planes exist | geometry, `placement_mode`, `label_name`, and attached plane summaries |
| `planes.h5` | planes exist | one root group per plane such as `plane_0000`, with `placement_mode`, `label_name`, the same per-plane payload, and a `payload_json` mirror |
| `pwv.json` | PWV runs successfully | saved PWV results for configured label groups |
| `pwv_<group>.png` | PWV plotting succeeds | saved PWV plots |

### Main metric fields

| Field | Type or unit | Meaning | Where to see it |
| --- | --- | --- | --- |
| `plane_index` | int | stable plane number for this record; matches `planes.json`, `planes.h5`, and plane-video labels | `plane_metrics.json`, `planes.json`, `planes.h5` |
| `center`, `normal` | 3-element arrays | plane center and plane normal in workspace coordinates | `plane_metrics.json`, `planes.json`, `planes.h5` |
| `path_index` | int | centerline path that generated this plane | `plane_metrics.json`, `planes.json`, `planes.h5`, GUI selection panel |
| `distance` | mm | cumulative distance from the path start to the plane location | `plane_metrics.json`, `planes.json`, `planes.h5` |
| `peakv_cm_s` | cm/s | peak absolute through-plane velocity over the whole cardiac cycle | `plane_metrics.json`, GUI selection panel, ortho viewer |
| `flowrate_mL_s` | array, mL/s | raw per-timepoint flow-rate curve through the plane | `plane_metrics.json`, `Plane Curve` mode, `planes.h5` |
| `netflow_mL_beat` | mL/beat | beat-integrated absolute flow derived from the mean of `flowrate_mL_s` and `RR` | `plane_metrics.json`, GUI selection panel, `planes.h5` |
| `area_mm2` | array, mm^2 | segmented cross-sectional area per timepoint | `plane_metrics.json`, `Plane Curve` mode, `planes.h5` |
| `meanv_cm_s`, `meanv_cm_s_t` | cm/s | mean through-plane velocity summary and its per-timepoint curve | `plane_metrics.json`, ortho viewer, `Plane Curve` mode, `planes.h5` |
| `flowrate_forward_mL_s`, `flowrate_reverse_mL_s` | arrays, mL/s | forward and reverse components after AutoFlow resolves the forward direction along the local path | `plane_metrics.json`, `Plane Curve` mode, `planes.h5` |
| `meanv_signed_cm_s`, `meanv_signed_cm_s_t` | cm/s | signed mean velocity aligned to the resolved forward direction | `plane_metrics.json`, `planes.h5` |
| `path_ic`, `fork_ic` | unitless or `null` | internal-consistency checks along a path and across forks; paths with fewer than two valid planes and forks missing a path metric are reported as `null`/undefined rather than a misleading perfect score or zero | `plane_metrics.json`, `plane_qc.json`, GUI `Internal Consistency` mode |
| `forward_sign`, `forward_sign_source` | int and string | how the forward direction was resolved for the plane | `plane_metrics.json`, `planes.h5` |
| `local_path_tangent`, `local_path_direction`, `normal_tangent_cos` | vector, text, scalar | relationship between the plane normal and the local centerline tangent | `plane_metrics.json`, `planes.h5` |
| `tke_*` | J/m^3 | derived TKE summaries; present only when TKE is available and requested | `plane_metrics.json`, `planes.h5`, `Plane Curve` mode |
| `pressure_gradient_*` | Pa/m | derived pressure-gradient summaries; present only when pressure analysis is requested | `plane_metrics.json`, `planes.h5`, `Plane Curve` mode |
| `relative_pressure_*` | Pa | derived relative-pressure summaries; present only when pressure analysis is requested | `plane_metrics.json`, `planes.h5`, `Plane Curve` mode |
| `wss_wall_*` | Pa | derived wall-shear summaries; present only when WSS is requested | `plane_metrics.json`, `planes.h5`, `Plane Curve` mode |
| `label_name` | string | readable plane label resolved from `configs/labels.json -> label_map` | `planes.json`, `planes.h5` |
| `path_info` | object | saved path metadata such as direction text, endpoints, and fork linkage | `planes.json`, `planes.h5` |

### Naming rules
- fields ending in `_t` are time-resolved arrays in cardiac-phase order
- fields ending in `_mean`, `_peak`, or `_p95` are scalar summaries over the sampled plane data
- `forward` and `reverse` use the resolved local forward direction; `signed` keeps that sign convention in one curve

Process parallelism now accounts for temporal workload: below 128 planes, 1920 or more plane-phase evaluations use up to four processes; smaller workloads remain serial. At least 128 planes retain up to eight processes. Shared input memmaps avoid copying full arrays into each worker. The controlled 96-plane/20-phase comparison preserved metric and QC results; see [Performance](../developer/performance.md).

Plane progress callbacks report every completed plane on the calling thread, including when several worker markers are collected in one poll.

Plane pixelwise H5 output is written privately and replaces the previous file only after a complete write. Cancelling or failing that export retains the previous complete file; its schema is unchanged.

## Limitations
- segmentation is required
- plane metrics depend on valid plane placement
- derived summaries attached to planes depend on opting in to the corresponding derived metrics
- requesting one derived metric does not implicitly compute the others; for example, `--with pg` attaches pressure summaries without running WSS
- `WSS / TKE / Pressure / Vortex` augments existing GUI plane metrics with WSS, TKE, and pressure summaries after computing missing derived families; vortex fields are whole-volume only and do not add plane summaries
- interactive plane edits defer the complete pixelwise H5 resampling pass until the explicit metric-save step
- support-mesh reuse assumes identical mask bytes represent identical geometry; changing segmentation content creates a new support mesh
- the compatibility setting is still named `use_multithread`, but large jobs use processes to isolate VTK state; process startup can make small jobs slower, so small plane-phase workloads remain serial

## Where To Change Code

| Change you want | Edit here | Also check | Tests |
| --- | --- | --- | --- |
| plane flow, area, velocity and process dispatch | `autoflow/algorithms/metrics/planes.py` | `autoflow/core/pipeline.py` | `tests/test_smoke_phantoms.py`, `tests/test_pressure_gradient_phantom.py` |
| support geometry, ROI selection and sampling caches | `autoflow/algorithms/metrics/sampling.py` | `autoflow/ui/ortho_viewer.py` | `tests/test_smoke_phantoms.py` |
| path, label and fork internal consistency | `autoflow/algorithms/metrics/consistency.py` | `autoflow/quality.py` | `tests/test_smoke_phantoms.py` |
| derived per-plane summaries and sample payloads | `autoflow/algorithms/metrics/plane_derived.py` | `autoflow/algorithms/metrics/derived.py` | `tests/test_smoke_phantoms.py`, `tests/test_pressure_gradient_phantom.py` |
| pixelwise H5 publication and metric-table loading | `autoflow/algorithms/metrics/export.py` | `autoflow/core/pipeline.py` | `tests/test_smoke_phantoms.py` |
| plane metric save format | `autoflow/core/pipeline.py`, `autoflow/plane_io.py` | `autoflow/reporting.py` | `tests/test_pressure_gradient_phantom.py` |
| GUI plane metric refresh and PWV dock | `autoflow/ui/app.py`, `autoflow/ui/ortho_viewer.py` | `autoflow/core/pipeline.py`, `autoflow/algorithms/pwv.py` | GUI manual verification |

## Tests

The existing `autoflow.algorithms.metrics` import path remains supported. Numerical implementations live in the modules above; helper instrumentation should patch the owning module, such as `metrics.sampling` or `metrics.wss`.

- `/home/renyuyang/miniconda3/envs/autoflow311/bin/python -m pytest tests/test_smoke_phantoms.py -q`
- `/home/renyuyang/miniconda3/envs/autoflow311/bin/python -m pytest tests/test_pressure_gradient_phantom.py -q`

## Common Problems

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| metrics step is skipped | no segmentation or no flow | load or create segmentation and verify input data |
| saved metrics do not match moved planes | plane edits were not finalized | finish drag interaction and let the GUI recompute metrics |
| it is hard to match a metric row to a rendered plane | plane labels were not inspected together with saved outputs | use `plane_index` in `plane_metrics.json`, `planes.json`, `planes.h5`, and the the plane-video index labels in `planes_rotate.mp4`, which default to `planeidx=<index>` and can be restyled in `configs/video_exporting.json -> plane_video.label` |
