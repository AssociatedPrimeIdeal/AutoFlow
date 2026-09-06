# Feature: Skeleton

## Status

| Entry point | Status | Notes |
| --- | --- | --- |
| GUI | Supported | interactive skeleton editing is still limited to single-group cases |
| CLI | Supported | batch generation only |
| Python API | Supported | through pipeline execution |

## What it does
Skeleton generation reduces the vessel mask to a centerline-style structure that seeds graph construction.

For grouped multi-label segmentations, AutoFlow now:

1. reduces 4D labels to 3D by majority vote along time
2. removes small connected components per label when enabled
3. merges labels into configured groups from `configs/labels.json`
4. filters grouped components with the configured connected-component rule, which defaults to `hybrid = max(min_cc_volume_mm3, cc_rel_min_ratio * largest_component_volume_mm3)`
5. applies per-group preprocessing
6. skeletonizes each group separately

Graph paths derived from the skeleton are split at graph nodes with degree
`>= 3`, so fork existence does not depend on noisy or locally ambiguous flow
directions. Flow is sampled on the segmentation-filtered path geometry to
orient each path and assign incoming/outgoing roles after topology has been
established.

## When to use it
- use it after segmentation is available
- use it before graph and plane generation
- use it when you want grouped vessel trees instead of one merged binary tree

## Quick use

### GUI
1. load or create segmentation
2. click `Generate Skeleton`
3. inspect grouped skeleton points in the browser and 3D view
4. optionally click `Edit Skeleton` when exactly one segmentation group is active

Edit mode shows only the skeleton in the 3D view and replaces the step buttons
with operation instructions. Click a point, drag its orange sphere to move it,
press Delete/Backspace to remove it, then click `Save Changes` to invalidate and
rebuild dependent graph/path/plane data. `Cancel` or Esc restores the prior
visibility and discards the edit.

### CLI

```bash
autoflow-run case.h5 --output-dir results/case
```

### Python API

```python
from autoflow import AutoFlowConfig, run_case

summary = run_case("case.h5", config=AutoFlowConfig(output_dir="./results/case"))
```

## Inputs

| Input | Required | Meaning |
| --- | --- | --- |
| segmentation | yes | binary mask or label mask used to derive the skeleton |
| resolution | yes | voxel spacing for volume-aware cleanup and preprocessing |
| `configs/labels.json` | for grouped label workflows | label map, label groups, colors, and per-group preprocessing |

## Parameters

| Parameter | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `remove_small_cc` | bool | `True` | `configs/skeleton.json` | remove small connected components before grouped preprocessing | `autoflow/core/models.py` |
| `separate_special_label_contacts` | bool | `True` | `configs/skeleton.json`, GUI skeleton parameters, CLI `--separate-special-label-contacts` | separates contacts only between the configured special labels (default: `RBCT`, `CCA`, `LBCT`) | `autoflow/algorithms/preprocess.py` |
| `special_contact_labels` | list[str] | `["RBCT", "CCA", "LBCT"]` | `configs/skeleton.json` | names of labels whose pairwise contacts are cut; all other label pairs are untouched | `autoflow/core/models.py` |
| `min_cc_volume_mm3` | float | `50.0` | `configs/skeleton.json` | component-volume threshold | `autoflow/core/models.py` |
| `cc_filter_mode` | string | `hybrid` | `configs/skeleton.json` | choose `absolute`, `relative`, `hybrid`, or `largest` connected-component filtering | `autoflow/core/models.py` |
| `cc_rel_min_ratio` | float | `0.01` | `configs/skeleton.json` | relative threshold against the largest connected component for `relative` and `hybrid` filtering | `autoflow/core/models.py` |
| `do_closing` | bool | `True` | `configs/skeleton.json` | global closing before skeletonization | `autoflow/algorithms/preprocess.py` |
| `do_opening` | bool | `False` | `configs/skeleton.json` | global opening before skeletonization | `autoflow/algorithms/preprocess.py` |
| `gaussian_sigma` | float | `0.5` | `configs/skeleton.json` | global smoothing strength | `autoflow/algorithms/preprocess.py` |
| `gaussian_enabled` | bool | `True` | `configs/skeleton.json` | enable or disable global Gaussian smoothing | `autoflow/algorithms/preprocess.py` |
| `label_map` | mapping | built-in vessel defaults | `configs/labels.json` | maps symbolic vessel names to integer label values | `autoflow/config.py` |
| `label_groups` | mapping | built-in vessel groups | `configs/labels.json` | merges labels into named groups and defines colors plus preprocessing overrides | `autoflow/core/models.py` |
| `label_groups.<group>.preprocess` | mapping | `{}` | `configs/labels.json` | per-group preprocessing overrides before skeletonization | `autoflow/algorithms/preprocess.py` |
| `single_label_group_name` | string | `single_label` | `configs/labels.json` | fallback group name for binary or one-label inputs | `autoflow/core/models.py` |

Fork detection is topology-based and therefore remains stable when flow near a
junction is weak. Path direction is still flow-informed, but uses the path
after segmentation filtering so adjacent labels do not dominate the direction
score.

## Outputs

| Output file or object | Created when | Meaning |
| --- | --- | --- |
| workspace skeleton points | skeleton step succeeds | centerline-like skeleton representation |
| skeleton scene object | GUI skeleton step succeeds | grouped skeleton object such as `skeleton_aorta_systemic_branches` |

## Limitations
- segmentation is required
- quality depends directly on segmentation quality
- special-label contact separation changes only the skeleton/graph mask; the original label mask used by flow and metrics is preserved
- interactive skeleton editing is only available when exactly one segmentation group is active

## Where to change code

| Change you want | Edit here | Also check | Tests |
| --- | --- | --- | --- |
| preprocessing before skeletonization | `autoflow/algorithms/preprocess.py` | `autoflow/core/models.py`, `autoflow/config.py` | `tests/test_smoke_phantoms.py` |
| grouped skeleton pipeline flow | `autoflow/core/pipeline.py` | `autoflow/algorithms/skeleton.py` | `tests/test_smoke_phantoms.py` |
| graph fork detection and flow-based path orientation | `autoflow/algorithms/branch.py`, `autoflow/algorithms/planes.py` | `autoflow/core/pipeline.py` | `tests/test_smoke_phantoms.py` |
| interactive skeleton edit behavior | `autoflow/ui/app.py`, `autoflow/ui/editors.py` | `autoflow/core/pipeline.py` | GUI manual verification |

## Tests
- `~/miniconda3/envs/ryy/bin/python -m pytest tests/test_smoke_phantoms.py -q`

## Common problems

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| skeleton step is skipped | no segmentation | create or load segmentation first |
| grouped skeleton is listed in the browser but absent from the 3D view | an older viewer did not resolve grouped `skeleton_<group>` data keys | install the current editable build and run `Generate Skeleton` again |
| skeleton contains many small branches | noisy segmentation | raise cleanup thresholds or improve segmentation |
| one grouped vessel disappears | every component in that group fell below the active cleanup threshold | lower `min_cc_volume_mm3` or `cc_rel_min_ratio`, or improve the segmentation |
| two special labels remain joined | contact separation is disabled or the labels are absent from `special_contact_labels` | enable `separate_special_label_contacts` and check the configured list; all non-special labels are intentionally left unchanged |
| `Edit Skeleton` is unavailable | more than one segmentation group is active | use a single-group case or simplify `configs/labels.json` |
