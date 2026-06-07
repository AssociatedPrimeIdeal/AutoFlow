# Feature: Skeleton

## Status

| Entry point | Status | Notes |
| --- | --- | --- |
| GUI | Supported | includes interactive skeleton editing for single-group cases |
| CLI | Supported | batch generation only |
| Python API | Supported | through pipeline execution |

## What It Does
Skeleton generation reduces the vessel mask to a centerline-style structure that seeds graph construction.
For grouped multi-label segmentations, AutoFlow first reduces 4D labels to 3D by majority vote along time, removes small connected components per label, merges labels into configured groups, applies per-group preprocessing, and then skeletonizes each group separately.

## When To Use It
- use it after segmentation is available
- use it before graph and plane generation
- do not use it when no segmentation is loaded

## Quick Use

### GUI
1. load or create segmentation
2. click `Generate Skeleton`
3. inspect grouped skeleton objects such as `skeleton_aorta_systemic_branches` in the browser
4. optionally click `Edit Skeleton` for interactive correction when exactly one segmentation group is active

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

| Input | Required | Meaning | Example |
| --- | --- | --- | --- |
| segmentation | yes | binary or label mask used to derive a skeleton | original, imported, threshold, or auto segmentation |
| resolution | yes | voxel spacing for volume-aware preprocessing | from loader |

## Parameters

| Parameter | Type | Default | Where set | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `remove_small_cc` | bool | `True` | `configs/skeleton.json` | remove small connected components | `autoflow/core/models.py` |
| `min_cc_volume_mm3` | float | `50.0` | `configs/skeleton.json` | component-volume threshold | `autoflow/core/models.py` |
| `do_closing` | bool | `True` | `configs/skeleton.json` | morphological closing before skeletonization | `autoflow/algorithms/preprocess.py` |
| `do_opening` | bool | `False` | `configs/skeleton.json` | morphological opening before skeletonization | `autoflow/algorithms/preprocess.py` |
| `gaussian_sigma` | float | `0.5` | `configs/skeleton.json` | smoothing strength | `autoflow/algorithms/preprocess.py` |
| `gaussian_enabled` | bool | `True` | `configs/skeleton.json` | enable or disable Gaussian smoothing | `autoflow/algorithms/preprocess.py` |
| `dilation_iters` | int | `0` | `configs/skeleton.json` | global dilation iterations before skeletonization | `autoflow/algorithms/preprocess.py` |
| `erosion_iters` | int | `0` | `configs/skeleton.json` | global erosion iterations before skeletonization | `autoflow/algorithms/preprocess.py` |
| `opening_iters` | int | `0` | `configs/skeleton.json` | global opening iterations before skeletonization | `autoflow/algorithms/preprocess.py` |
| `closing_iters` | int | `0` | `configs/skeleton.json` | global closing iterations before skeletonization | `autoflow/algorithms/preprocess.py` |
| `label_map` | mapping | built-in vessel defaults | `configs/skeleton.json` | maps label names to integer label values | `autoflow/config.py` |
| `label_groups` | mapping | built-in vessel groups | `configs/skeleton.json` | merges labels into named groups and defines colors plus preprocessing overrides | `autoflow/core/models.py` |
| `single_label_group_name` | string | `single_label` | `configs/skeleton.json` | fallback group name for binary or one-label inputs | `autoflow/core/models.py` |
| `single_label_browser_color` | string | `#d9480f` | `configs/skeleton.json` | browser title color for the single-label fallback | `autoflow/core/models.py` |
| `default_group_browser_color` | string | `#1c7ed6` | `configs/skeleton.json` | browser title color when a group has no explicit color | `autoflow/core/models.py` |

## Outputs

| Output file or object | Created when | Meaning |
| --- | --- | --- |
| workspace skeleton points | skeleton step succeeds | centerline-like skeleton representation |
| skeleton scene object | GUI skeleton step succeeds | grouped skeleton object such as `skeleton_aorta_systemic_branches` |

## Limitations
- segmentation is required
- quality depends directly on segmentation quality
- there is no standalone skeleton file export documented as a public output
- interactive skeleton editing is only available when exactly one segmentation group is active

## Where To Change Code

| Change you want | Edit here | Also check | Tests |
| --- | --- | --- | --- |
| preprocessing before skeletonization | `autoflow/algorithms/preprocess.py` | `autoflow/core/models.py` | `tests/test_smoke_phantoms.py` |
| skeleton extraction | `autoflow/algorithms/skeleton.py` | `autoflow/core/pipeline.py` | `tests/test_smoke_phantoms.py` |
| interactive skeleton edit behavior | `autoflow/ui/app.py`, `autoflow/ui/editors.py` | `autoflow/core/pipeline.py` | GUI smoke coverage |

## Tests

- `~/miniconda3/envs/ryy/bin/python -m pytest tests/test_smoke_phantoms.py -q`

## Common Problems

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| skeleton step is skipped | no segmentation | create or load segmentation first |
| skeleton contains many small branches | noisy segmentation | raise cleanup thresholds or improve segmentation |
| `Edit Skeleton` is unavailable | more than one segmentation group is active | use a single-label or single-group case, or change group configuration |
