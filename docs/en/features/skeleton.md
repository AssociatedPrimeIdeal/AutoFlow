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
4. keeps only the largest connected component for each group mask
5. applies per-group preprocessing
6. skeletonizes each group separately

## When to use it
- use it after segmentation is available
- use it before graph and plane generation
- use it when you want grouped vessel trees instead of one merged binary tree

## Quick use

### GUI
1. load or create segmentation
2. click `Generate Skeleton`
3. inspect grouped skeleton objects in the browser
4. optionally click `Edit Skeleton` when exactly one segmentation group is active

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
| `min_cc_volume_mm3` | float | `50.0` | `configs/skeleton.json` | component-volume threshold | `autoflow/core/models.py` |
| `do_closing` | bool | `True` | `configs/skeleton.json` | global closing before skeletonization | `autoflow/algorithms/preprocess.py` |
| `do_opening` | bool | `False` | `configs/skeleton.json` | global opening before skeletonization | `autoflow/algorithms/preprocess.py` |
| `gaussian_sigma` | float | `0.5` | `configs/skeleton.json` | global smoothing strength | `autoflow/algorithms/preprocess.py` |
| `gaussian_enabled` | bool | `True` | `configs/skeleton.json` | enable or disable global Gaussian smoothing | `autoflow/algorithms/preprocess.py` |
| `label_map` | mapping | built-in vessel defaults | `configs/labels.json` | maps symbolic vessel names to integer label values | `autoflow/config.py` |
| `label_groups` | mapping | built-in vessel groups | `configs/labels.json` | merges labels into named groups and defines colors plus preprocessing overrides | `autoflow/core/models.py` |
| `label_groups.<group>.preprocess` | mapping | `{}` | `configs/labels.json` | per-group preprocessing overrides before skeletonization | `autoflow/algorithms/preprocess.py` |
| `single_label_group_name` | string | `single_label` | `configs/labels.json` | fallback group name for binary or one-label inputs | `autoflow/core/models.py` |

## Outputs

| Output file or object | Created when | Meaning |
| --- | --- | --- |
| workspace skeleton points | skeleton step succeeds | centerline-like skeleton representation |
| skeleton scene object | GUI skeleton step succeeds | grouped skeleton object such as `skeleton_aorta_systemic_branches` |

## Limitations
- segmentation is required
- quality depends directly on segmentation quality
- interactive skeleton editing is only available when exactly one segmentation group is active

## Where to change code

| Change you want | Edit here | Also check | Tests |
| --- | --- | --- | --- |
| preprocessing before skeletonization | `autoflow/algorithms/preprocess.py` | `autoflow/core/models.py`, `autoflow/config.py` | `tests/test_smoke_phantoms.py` |
| grouped skeleton pipeline flow | `autoflow/core/pipeline.py` | `autoflow/algorithms/skeleton.py` | `tests/test_smoke_phantoms.py` |
| interactive skeleton edit behavior | `autoflow/ui/app.py`, `autoflow/ui/editors.py` | `autoflow/core/pipeline.py` | GUI manual verification |

## Tests
- `~/miniconda3/envs/ryy/bin/python -m pytest tests/test_smoke_phantoms.py -q`

## Common problems

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| skeleton step is skipped | no segmentation | create or load segmentation first |
| skeleton contains many small branches | noisy segmentation | raise cleanup thresholds or improve segmentation |
| one grouped vessel disappears | only a tiny disconnected component existed after grouping | review the segmentation or adjust grouping so the desired vessel is in the largest connected component |
| `Edit Skeleton` is unavailable | more than one segmentation group is active | use a single-group case or simplify `configs/labels.json` |
