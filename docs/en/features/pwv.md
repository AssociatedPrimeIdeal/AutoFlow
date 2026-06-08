# Feature: PWV

## Status

| Entry point | Status | Notes |
| --- | --- | --- |
| GUI | Supported | plot shown in the `PWV` dock after plane metrics run |
| CLI | Supported | config-driven through `configs/pwv.json` |
| Python API | Supported | config-driven through `AutoFlowConfig.from_config_dir()` |

## What it does
PWV computes pulse-wave velocity from one or more configured label groups.

For each PWV group, AutoFlow:

1. merges the configured labels into one mask
2. keeps only the largest connected component
3. skeletonizes that grouped mask
4. builds a graph and finds the longest path
5. places planes along that path at the configured spacing in mm
6. computes plane metrics for those PWV planes
7. extracts a waveform foot time at each plane
8. fits time-to-foot versus slice position
9. reports PWV and saves a plot

## When to use it
- use it when you want a config-defined PWV measurement along a vessel tree
- use it when a vessel is represented by several segmentation labels that should be merged before PWV
- use it after segmentation and flow are available

## Quick use

### GUI
1. enable PWV groups in `configs/pwv.json`
2. load a case with segmentation and flow
3. run `Calculate && Save Metrics`
4. inspect the `PWV` dock and the grouped `PWV planes` browser item

### CLI

```bash
autoflow-run case.h5 --config-dir ./configs --output-dir results/case
```

### Python API

```python
from autoflow import AutoFlowConfig, run_case

config = AutoFlowConfig.from_config_dir("./configs")
summary = run_case("case.h5", config=config)
```

## Inputs

| Input | Required | Meaning |
| --- | --- | --- |
| flow | yes | time-resolved velocity field |
| segmentation | yes | label mask used to define PWV groups |
| `configs/pwv.json` | yes | PWV groups and measurement parameters |
| `configs/labels.json` | for symbolic labels | resolves label names such as `AAO` or `PV` |

## Parameters

| Parameter | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `enabled` | bool | `False` | `configs/pwv.json` | enable or disable PWV computation | `autoflow/core/pipeline.py` |
| `groups` | list | `[]` | `configs/pwv.json` | each item defines one PWV label combination | `autoflow/core/models.py` |
| `groups[].name` | string | generated name | `configs/pwv.json` | display name in results and GUI | `autoflow/core/models.py` |
| `groups[].labels` | list[int or symbol] | required per group | `configs/pwv.json` | labels merged into one PWV mask | `autoflow/core/models.py` |
| `plane_interval_mm` | float | `10.0` | `configs/pwv.json` | spacing between PWV planes | `autoflow/algorithms/pwv.py` |
| `start_distance` | float | `0.0` | `configs/pwv.json` | offset from the path start | `autoflow/algorithms/pwv.py` |
| `end_distance` | float | `0.0` | `configs/pwv.json` | offset from the path end | `autoflow/algorithms/pwv.py` |
| `waveform_key` | string | `flowrate_signed_mL_s` | `configs/pwv.json` | metric waveform used for foot detection | `autoflow/algorithms/pwv.py` |
| `foot_savgol_window` | int | `5` | `configs/pwv.json` | waveform smoothing window | `autoflow/algorithms/pwv.py` |
| `foot_savgol_polyorder` | int | `2` | `configs/pwv.json` | waveform smoothing polynomial order | `autoflow/algorithms/pwv.py` |
| `minimum_valid_planes` | int | `2` | `configs/pwv.json` | minimum valid planes required to fit PWV | `autoflow/algorithms/pwv.py` |

## Outputs

| Output file or object | Created when | Meaning |
| --- | --- | --- |
| `pwv.json` | PWV is enabled and plane metrics run | one result block per PWV group |
| `pwv_<group>.png` | plotting succeeds | per-group fit plot |
| `summary.json -> pwv_results` | summary export runs | PWV result summary |
| `PWV planes` scene object | GUI or pipeline PWV succeeds | one grouped browser item that controls all PWV planes |

## Limitations
- PWV depends on segmentation quality, graph quality, and the longest-path heuristic
- there are no dedicated CLI flags yet; configuration is file-driven
- if too few valid planes survive waveform foot detection, the group is skipped

## Where to change code

| Change you want | Edit here | Also check | Tests |
| --- | --- | --- | --- |
| PWV algorithm | `autoflow/algorithms/pwv.py` | `autoflow/core/pipeline.py` | manual verification plus retained smoke/phantom suite |
| PWV scene registration | `autoflow/core/pipeline.py` | `autoflow/ui/viewer.py` | GUI manual verification |
| PWV dock and plot | `autoflow/ui/app.py` | `autoflow/algorithms/pwv.py` | GUI manual verification |

## Tests
- `~/miniconda3/envs/ryy/bin/python -m pytest tests/test_smoke_phantoms.py tests/test_pressure_gradient_phantom.py -q`

## Common problems

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| no PWV result is written | `configs/pwv.json -> enabled` is false or no groups are configured | enable PWV and define groups |
| a PWV group is skipped | the grouped mask is empty or too few valid planes survived | check segmentation labels and waveform quality |
| PWV planes do not appear individually in the browser | expected behavior | use the single `PWV planes` browser item to control visibility |
