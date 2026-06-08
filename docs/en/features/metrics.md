# Feature: Plane Metrics

## Status

| Entry point | Status | Notes |
| --- | --- | --- |
| GUI | Supported | `Calculate && Save Metrics` step |
| CLI | Supported | part of default batch order |
| Python API | Supported | through batch and case APIs |

## What It Does
Plane metrics compute time-resolved cross-sectional measurements for each plane and save them to JSON and HDF5 outputs.

When `configs/pwv.json -> enabled` is true, the plane-metrics step also computes PWV for each configured PWV group and writes PWV outputs.

## When To Use It
- use it after planes exist
- use it when you need flow, area, mean velocity, peak velocity, net flow, and derived plane summaries
- do not expect it to run without segmentation

## Quick Use

### GUI
1. generate planes
2. click `Calculate && Save Metrics`
3. inspect the selected plane in the selection panel and ortho viewer

### CLI

```bash
autoflow-run case.h5 --output-dir results/case
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

| Parameter | Type | Default | Where set | Effect |
| --- | --- | --- | --- | --- |
| `skip_plane_metrics` | bool | `False` | batch config or CLI/API | disable the whole plane-metric export step |
| `use_multithread` | bool | `True` | `configs/batch.json` | multithread plane metric calculation |

## Outputs

| Output file or object | Created when | Meaning |
| --- | --- | --- |
| `plane_metrics.json` | plane metrics run | time-resolved plane metrics |
| `plane_qc.json` | plane metrics run | QC for forks and paths |
| `plane_metrics_pixelwise.h5` | plane metrics run | per-plane slice-cellwise derived samples |
| plane summaries in `planes.json` | plane metrics run | geometry plus attached summaries |
| `pwv.json` | PWV is enabled | saved PWV results for configured label groups |
| `pwv_<group>.png` | PWV plotting succeeds | saved PWV plots |

## Limitations
- segmentation is required
- plane metrics depend on valid plane placement
- derived summaries attached to planes depend on derived-metric availability

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
