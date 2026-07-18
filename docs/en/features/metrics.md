# Feature: Plane Metrics

## Status

| Entry point | Status | Notes |
| --- | --- | --- |
| GUI | Supported | `Calculate && Save Metrics` step plus live refresh after plane edits |
| CLI | Supported | part of the default batch order |
| Python API | Supported | through `run_case()` and `run_batch()` |

## What It Does
Plane metrics compute time-resolved cross-sectional measurements for each plane and save them into `plane_metrics.json`. The same per-plane payload is also mirrored into `planes.json` and `planes.h5`, so one plane index has one consistent set of geometry and summaries across outputs.

## When To Use It
- use it after planes exist
- use it when you need per-plane flow, area, and velocity curves
- use it when you need explicit `plane_index` values that line up with `planes.json`, `planes.h5`, and the plane-rotation video labels
- add `--with wss,tke,pg` if you also want derived per-plane summaries
- do not expect it to run without segmentation

## Quick Use

### GUI
1. generate planes
2. click `Calculate && Save Metrics`
3. inspect the selected plane in the selection panel, ortho viewer, and `Plane Curve` mode

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

| Parameter | Type | Default | Where set | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `skip_plane_metrics` | bool | `False` | batch config or CLI/API | disable the whole plane-metric export step | `autoflow/processing.py` |
| `use_multithread` | bool | `True` | `configs/batch.json` | multithread plane metric calculation | `autoflow/core/pipeline.py` |

## Outputs

| Output file or object | Created when | Meaning |
| --- | --- | --- |
| `plane_metrics.json` | plane metrics run | one metric record per plane with explicit `plane_index` and time-resolved fields |
| `plane_qc.json` | plane metrics run | QC for paths and forks |
| `plane_metrics_pixelwise.h5` | plane metrics run | per-plane slice-cellwise derived samples |
| `planes.json` | planes exist | geometry, `label_name`, and attached plane summaries |
| `planes.h5` | planes exist | one root group per plane such as `plane_0000`, with `label_name`, the same per-plane payload, and a `payload_json` mirror |
| `pwv.json` | PWV is enabled | saved PWV results for configured label groups |
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
| `path_ic`, `fork_ic` | unitless | internal-consistency checks along a path and across forks | `plane_metrics.json`, `plane_qc.json`, GUI `Internal Consistency` mode |
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

## Limitations
- segmentation is required
- plane metrics depend on valid plane placement
- derived summaries attached to planes depend on opting in to the corresponding derived metrics

## Where To Change Code

| Change you want | Edit here | Also check | Tests |
| --- | --- | --- | --- |
| plane metric computation | `autoflow/algorithms/metrics.py` | `autoflow/core/pipeline.py` | `tests/test_smoke_phantoms.py`, `tests/test_pressure_gradient_phantom.py` |
| plane metric save format | `autoflow/core/pipeline.py`, `autoflow/plane_io.py` | `autoflow/reporting.py` | `tests/test_pressure_gradient_phantom.py` |
| GUI plane metric refresh and PWV dock | `autoflow/ui/app.py`, `autoflow/ui/ortho_viewer.py` | `autoflow/core/pipeline.py`, `autoflow/algorithms/pwv.py` | GUI manual verification |

## Tests

- `~/miniconda3/envs/ryy/bin/python -m pytest tests/test_smoke_phantoms.py -q`
- `~/miniconda3/envs/ryy/bin/python -m pytest tests/test_pressure_gradient_phantom.py -q`

## Common Problems

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| metrics step is skipped | no segmentation or no flow | load or create segmentation and verify input data |
| saved metrics do not match moved planes | plane edits were not finalized | finish drag interaction and let the GUI recompute metrics |
| it is hard to match a metric row to a rendered plane | plane labels were not inspected together with saved outputs | use `plane_index` in `plane_metrics.json`, `planes.json`, `planes.h5`, and the the plane-video index labels in `planes_rotate.mp4`, which default to `planeidx=<index>` and can be restyled in `configs/video_exporting.json -> plane_video.label` |
