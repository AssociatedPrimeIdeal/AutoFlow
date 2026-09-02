# Feature: Quality Control

## Status

| Entry point | Status | Notes |
| --- | --- | --- |
| GUI | Supported | interactive staged check table and JSON export in `Review & Export` |
| CLI | Supported | every processed case writes `quality_report.json` |
| Python API | Supported | batch summaries include the report; direct report helpers are also available |

## What It Does

Quality control converts pipeline state into explicit `pass`, `warn`, `fail`, or `not_run` checks. It currently screens physical geometry, finite velocity values, VENC saturation, background-phase provenance, segmentation content and temporal stability, graph topology, plane alignment, plane sampling, flow internal consistency, and PWV fit quality.

The report is for engineering review prioritization. It does not establish clinical validity and must not be interpreted as a diagnostic conclusion.

## When To Use It

- refresh QC after loading to review input and segmentation checks
- refresh it after editing segmentation, centerlines, or planes
- use it before exporting results to find incomplete or suspicious stages
- save the JSON beside research outputs to record which checks were available and their values

## Quick Use

### GUI

1. load and process a case
2. open `Review & Export`
3. click `Refresh QC`
4. inspect warnings and failures; hover the result for its threshold and suggested action
5. click `Export QC Report`

### CLI

```bash
autoflow-run case.h5 --output-dir results/case --with pwv,wss,pg
```

The case directory receives `quality_report.json`; the same payload and its file path are recorded in `summary.json`.

### Python API

```python
from autoflow.quality import build_quality_report, save_quality_report

report = build_quality_report(workspace, source_path="case.h5")
save_quality_report(workspace, "results/case/quality_report.json", report=report)
```

`run_case()` and `run_batch()` generate the report automatically.

## Inputs

| Input | Required | Meaning |
| --- | --- | --- |
| loaded flow and physical metadata | yes | input geometry, finite-value, and VENC checks |
| active segmentation | for segmentation QC | foreground labels, components, volume, and temporal stability |
| graph and paths | for topology QC | connected components, endpoints, branch nodes, isolated nodes, and cycles |
| planes and plane metrics | for plane/hemodynamic QC | plane alignment, sampling, area stability, and internal consistency |
| PWV results | when PWV runs | successful groups and minimum fit R2 |

## Parameters

The first report schema uses conservative screening thresholds owned by code. They are not clinical acceptance limits.

| Parameter | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| finite velocity pass threshold | fraction | `0.999` | code | warns or fails when sampled flow contains non-finite values | `autoflow/quality.py` |
| VENC saturation warning/fail thresholds | fraction | `0.001` / `0.01` | code | screens component values reaching 95% of VENC | `autoflow/quality.py` |
| segmentation temporal CV thresholds | fraction | `0.05` / `0.15` | code | screens abrupt time-varying foreground volume | `autoflow/quality.py` |
| plane normal angle warning/fail thresholds | degrees | `20` / `45` | code | compares generated-plane normals with local centerline tangents | `autoflow/quality.py` |
| internal-consistency pass/warn thresholds | score | `0.8` / `0.6` | code | screens path and fork flow consistency | `autoflow/quality.py` |
| PWV minimum fit R2 | score | `0.8` | code | warns on weak arrival-time fits | `autoflow/quality.py` |

## Outputs

| Output | Created when | Meaning |
| --- | --- | --- |
| `quality_report.json` | CLI or Python case processing completes | schema, overall state, counts, checks, values, thresholds, actions, and run context |
| `summary.json -> quality_report` | case processing completes | embedded report for batch aggregation |
| GUI QC table | `Refresh QC` runs | current workspace checks, including unsaved interactive edits |

The overall state is `not_ready` when a required check fails, `needs_review` when it warns, `incomplete` when required stages have not run, and `ready` when all required available checks pass.

## Limitations

- thresholds are screening defaults, not population-derived clinical limits
- VENC screening samples the array for responsiveness and does not perform phase unwrapping
- graph cycles and disconnected segmentation components can be intentional in some anatomies and therefore require review
- pressure solver residuals, WSS sensitivity intervals, uncertainty maps, and registration QC are not yet available in the v1 schema
- a `ready` report means the implemented automated checks passed; it does not prove ground-truth accuracy

## Where To Change Code

| Change you want | Edit here | Also check | Tests |
| --- | --- | --- | --- |
| add or change a check | `autoflow/quality.py` | `docs/en/features/quality-control.md` | `tests/test_smoke_phantoms.py` |
| change automatic CLI export | `autoflow/processing.py` | `autoflow/api.py`, `docs/en/user/outputs.md` | `tests/test_smoke_phantoms.py`, `tests/test_pressure_gradient_phantom.py` |
| change GUI presentation | `autoflow/ui/app.py` | `docs/en/user/gui.md` | manual GUI verification plus smoke coverage |

## Tests

- `~/miniconda3/envs/ryy/bin/python -m pytest tests/test_smoke_phantoms.py -q`
- `~/miniconda3/envs/ryy/bin/python -m pytest tests/test_pressure_gradient_phantom.py -q`
- manually refresh the GUI table before and after changing a plane or segmentation

## Common Problems

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| overall state is `incomplete` | centerline, planes, or metrics have not run | finish the missing workflow stage and refresh QC |
| VENC saturation warns | velocities approach the encoded limit | inspect aliasing and acquisition metadata before accepting quantitative results |
| plane geometry warns after cross-case import | cases are unregistered or target paths differ | use relative-centerline mapping or register cases before world-coordinate import |
| flow consistency fails | segmentation, orientation, branch ownership, or plane placement is inconsistent | inspect the named paths/forks and their time curves |
| GUI report differs from saved batch report | the workspace was edited after the batch run | refresh and export the GUI report again |
