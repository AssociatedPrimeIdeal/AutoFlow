# Feature: PWV

## Status

| Entry point | Status | Notes |
| --- | --- | --- |
| GUI | Supported | PWV is shown in the `Analysis` dock after the `Compute PWV` step runs; the same dock also exposes selection-driven plane curves and internal-consistency views |
| CLI | Supported | opt-in through `--with pwv` plus `configs/pwv.json` |
| Python API | Supported | config-driven through `AutoFlowConfig.from_config_dir()` |

## What it does
PWV computes pulse-wave velocity from one or more configured label groups.

For each PWV group, AutoFlow:

1. merges the configured labels into one mask
2. applies the active skeleton connected-component cleanup rule to that grouped mask
3. skeletonizes that grouped mask
4. builds a graph and finds the longest endpoint-to-endpoint path across all degree-1 nodes in the PWV group graph
5. places planes along that path at the configured spacing in mm
6. computes plane metrics for those PWV planes
7. extracts a waveform foot time at each plane
8. fits time-to-foot versus slice position
9. converts per-plane timing into arrival times using the selected method and reports PWV

## When to use it
- use it when you want a config-defined PWV measurement along a vessel tree
- use it when a vessel is represented by several segmentation labels that should be merged before PWV
- use it after segmentation and flow are available

## Quick use

### GUI
1. load a case with segmentation and flow
2. open `PWV Parameters` in the main parameter panel
3. enable PWV, define one or more groups, and adjust spacing or waveform settings if needed
4. run `Calculate && Save Metrics`, then `Compute PWV`
5. inspect `Analysis -> PWV` and the grouped `PWV planes` browser item

### CLI

```bash
autoflow-run case.h5 --config-dir ./configs --output-dir results/case --with pwv
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
| `enabled` | bool | `False` | `configs/pwv.json` or GUI `PWV Parameters` | enable or disable PWV computation | `autoflow/core/pipeline.py` |
| `groups` | list | `[]` | `configs/pwv.json` or GUI `PWV Parameters` | each item defines one PWV label combination | `autoflow/core/models.py` |
| `groups[].name` | string | generated name | `configs/pwv.json` or GUI `PWV Parameters` | display name in results and GUI | `autoflow/core/models.py` |
| `groups[].labels` | list[int or symbol] | required per group | `configs/pwv.json` or GUI `PWV Parameters` | labels merged into one PWV mask | `autoflow/core/models.py` |
| `plane_interval_mm` | float | `10.0` | `configs/pwv.json` or GUI `PWV Parameters` | spacing between PWV planes | `autoflow/algorithms/pwv.py` |
| `start_distance` | float | `0.0` | `configs/pwv.json` or GUI `PWV Parameters` | offset from the path start | `autoflow/algorithms/pwv.py` |
| `end_distance` | float | `0.0` | `configs/pwv.json` or GUI `PWV Parameters` | offset from the path end | `autoflow/algorithms/pwv.py` |
| `smoothing_window` | int | `15` | `configs/pwv.json` or GUI `PWV Parameters` | path smoothing window before PWV planes are generated | `autoflow/algorithms/pwv.py` |
| `smoothing_polyorder` | int | `2` | `configs/pwv.json` or GUI `PWV Parameters` | path smoothing polyorder before PWV planes are generated | `autoflow/algorithms/pwv.py` |
| `inter_time` | int | `10` | `configs/pwv.json` or GUI `PWV Parameters` | path interpolation multiplier before PWV plane generation | `autoflow/algorithms/pwv.py` |
| `waveform_key` | string | `flowrate_mL_s` | `configs/pwv.json` or GUI `PWV Parameters` | metric waveform used for foot-to-foot detection and as the default waveform family for PWV timing | `autoflow/algorithms/pwv.py` |
| `transit_time_method` | string | `foot_to_foot` | `configs/pwv.json` or GUI `PWV Parameters` | choose `foot_to_foot` or `cross_correlation` transit-time estimation | `autoflow/algorithms/pwv.py` |
| `foot_method` | string | `tangent` | `configs/pwv.json` or GUI `PWV Parameters` | choose `tangent` or `threshold` foot definition for `foot_to_foot` mode | `autoflow/algorithms/pwv.py` |
| `foot_savgol_window` | int | `5` | `configs/pwv.json` or GUI `PWV Parameters` | waveform smoothing window | `autoflow/algorithms/pwv.py` |
| `foot_savgol_polyorder` | int | `2` | `configs/pwv.json` or GUI `PWV Parameters` | waveform smoothing polynomial order | `autoflow/algorithms/pwv.py` |
| `foot_threshold_percent` | float | `10.0` | `configs/pwv.json` or GUI `PWV Parameters` | threshold percentage used when `foot_method=threshold` | `autoflow/algorithms/pwv.py` |
| `xcorr_window` | string | `full` | `configs/pwv.json` or GUI `PWV Parameters` | choose `full` waveform or `upstroke` window for cross-correlation | `autoflow/algorithms/pwv.py` |
| `xcorr_interp_factor` | int | `10` | `configs/pwv.json` or GUI `PWV Parameters` | cyclic waveform interpolation factor used before cross-correlation delay estimation | `autoflow/algorithms/pwv.py` |
| `allow_cycle_wrap` | bool | `True` | `configs/pwv.json` or GUI `PWV Parameters` | allow one-cycle wrap correction and cycle-aware foot detection when the upstroke crosses the frame boundary | `autoflow/algorithms/pwv.py` |
| `minimum_valid_planes` | int | `2` | `configs/pwv.json` or GUI `PWV Parameters` | minimum valid planes required to fit PWV | `autoflow/algorithms/pwv.py` |
| `scene_visible` | bool | `True` | `configs/pwv.json` or GUI `PWV Parameters` | show or hide the grouped `PWV planes` scene object by default | `autoflow/core/pipeline.py` |
| `scene_color` | string | `#ffd43b` | `configs/pwv.json` or GUI `PWV Parameters` | PWV plane scene color | `autoflow/core/pipeline.py` |
| `plot_color` | string | `#2b8a3e` | `configs/pwv.json` or GUI `PWV Parameters` | scatter color in saved PWV plots and `Analysis -> PWV` | `autoflow/algorithms/pwv.py` |
| `fit_color` | string | `#f08c00` | `configs/pwv.json` or GUI `PWV Parameters` | fit line color in saved PWV plots and `Analysis -> PWV` | `autoflow/algorithms/pwv.py` |
| `plot_dpi` | int | `160` | `configs/pwv.json` or GUI `PWV Parameters` | PNG export resolution for saved PWV plots | `autoflow/algorithms/pwv.py` |

## Outputs

| Output file or object | Created when | Meaning |
| --- | --- | --- |
| `pwv.json` | PWV is enabled and PWV runs | one result block per PWV group |
| `pwv_<group>.png` | plotting succeeds | per-group two-panel plot with PWV fit plus all plane flowrate waveforms |
| `summary.json -> pwv_results` | summary export runs | PWV result summary |
| `PWV planes` scene object | GUI or pipeline PWV succeeds | one grouped browser item that controls all PWV planes |

## Limitations
- PWV depends on segmentation quality and graph quality; the centerline is chosen as the longest endpoint-to-endpoint path among all degree-1 graph nodes in the PWV group
- CLI must opt in with `--with pwv`, and PWV group definitions still come from `configs/pwv.json`
- if too few valid planes survive waveform foot detection, the group is skipped
- `cross_correlation` now upsamples the waveform in time before alignment, so low phase counts can still yield sub-frame delay estimates
- with `allow_cycle_wrap=true`, `tangent` and `threshold` foot detection search across the cycle boundary instead of forcing early-systolic peaks near frame 0 to foot time `0 ms`

## Where to change code

| Change you want | Edit here | Also check | Tests |
| --- | --- | --- | --- |
| PWV algorithm | `autoflow/algorithms/pwv.py` | `autoflow/core/pipeline.py` | manual verification plus retained smoke/phantom suite |
| PWV scene registration | `autoflow/core/pipeline.py` | `autoflow/ui/viewer.py` | GUI manual verification |
| Analysis dock, PWV panel, plane curves, and internal-consistency views | `autoflow/ui/app.py` | `autoflow/algorithms/pwv.py`, `autoflow/algorithms/metrics.py` | GUI manual verification |

## Tests
- `~/miniconda3/envs/ryy/bin/python -m pytest tests/test_smoke_phantoms.py tests/test_pressure_gradient_phantom.py -q`

## Common problems

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| no PWV result is written | PWV is disabled or no groups are configured in `configs/pwv.json` or the GUI `PWV Parameters` panel | enable PWV and define groups |
| a PWV group is skipped | the grouped mask is empty or too few valid planes survived | check segmentation labels and waveform quality |
| PWV planes do not appear individually in the browser | expected behavior | use the single `PWV planes` browser item to control visibility |
