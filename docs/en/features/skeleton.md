# Feature: Skeleton

## Status

| Entry point | Status | Notes |
| --- | --- | --- |
| GUI | Supported | interactive skeleton editing is still limited to single-group cases |
| CLI | Supported | batch generation only |
| Python API | Supported | through pipeline execution |

## What it does
Skeleton generation reduces the vessel mask to a centerline-style structure that seeds graph construction.

See [Scientific references: centerline](../references/index.md#centerline-skeleton-and-paths) for Lee 3-D thinning and subsequent Savitzky–Golay path smoothing. Group handling and graph cleanup are AutoFlow-specific; this is not a VMTK centerline workflow.

For grouped multi-label segmentations, AutoFlow now:

1. reduces 4D labels to 3D by majority vote along time
2. removes small connected components per label when enabled
3. merges labels into configured groups from `configs/labels.json`
4. filters grouped components with the configured connected-component rule, which defaults to `hybrid = max(min_cc_volume_mm3, cc_rel_min_ratio * largest_component_volume_mm3)`
5. applies per-group preprocessing
6. applies the configured special-label handling for groups containing at least two special labels
7. skeletonizes each group separately

The built-in label groups keep the non-cranial workflows unchanged. Cranial
labels are organized as `intracranial_anterior_arteries` (`LICA`, `RICA`,
`ACA`, `LMCA`, `RMCA`), `vertebrobasilar_arteries` (`LVA`, `RVA`, `BA`,
`LPCA`, `RPCA`), `intracranial_veins` (`LTS`, `SSS`, `RTS`, `StrS`), and
`jugular_veins` (`LIJV`, `RIJV`).

After graph generation, AutoFlow performs a conservative Willis-ring check on
the two arterial masks. A detected cycle containing labels from both arterial
groups is exposed as a derived `willis_ring` graph overlay. The overlay does
not replace source labels, ordinary groups, planes, or metrics. Cases without
a reliable mixed arterial cycle are reported as `not_detected` and keep their
anterior and vertebrobasilar graphs separate.

`special_handling` selects the special-label strategy. `three_pass_merge` (the
default) skeletonizes `A+B`, `A+C`, and so on, then merges nearby points.
`contact_surface` keeps the previous contact-surface separation algorithm.

Graph paths derived from the skeleton are split at graph nodes with degree
`>= 3`, so fork existence does not depend on noisy or locally ambiguous flow
directions. Flow is sampled on the segmentation-filtered path geometry to
orient each path and assign incoming/outgoing roles after topology has been
established.

During graph generation, terminal branches shorter than `min_edge_points` graph
edge segments are removed. The default is `3`; set it to `0` or `1` to disable
this cleanup. This filters graph edges and derived paths while leaving the
original skeleton mask and points available for inspection.

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

See [skeleton parameters](../user/parameters.md#skeleton), [labels parameters](../user/parameters.md#labels), [CLI flags](../user/cli-parameters.md), and [API fields](../user/api-parameters.md) for complete type/default/unit/effect/owner tables. Dictionary controls are expanded in [Structured parameters](../user/parameter-schemas.md).

Fork detection is topology-based and therefore remains stable when flow near a
junction is weak. Path direction is still flow-informed, but uses the path
after segmentation filtering so adjacent labels do not dominate the direction
score.

## Outputs

| Output file or object | Created when | Meaning |
| --- | --- | --- |
| workspace skeleton points | skeleton step succeeds | centerline-like skeleton representation |
| skeleton scene object | GUI skeleton step succeeds | grouped skeleton object such as `skeleton_aorta_systemic_branches` |
| `willis_ring_status` and optional `willis_ring_graph` | graph step succeeds for the cranial arterial groups | topology-only Willis-ring status and overlay; source labels and metrics remain unchanged |
| `summary.json -> willis_ring` | CLI/batch processing completes | persisted status, cycle rank, component details, and member label IDs |

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
| Willis-ring topology check | `autoflow/algorithms/intracranial.py`, `autoflow/core/pipeline.py` | `autoflow/core/models.py`, `autoflow/ui/viewer.py` | `tests/test_smoke_phantoms.py` |
| short terminal branch cleanup | `autoflow/algorithms/graph.py` | `autoflow/core/pipeline.py`, `autoflow/core/models.py` | `tests/test_smoke_phantoms.py` |
| graph fork detection and flow-based path orientation | `autoflow/algorithms/branch.py`, `autoflow/algorithms/planes.py` | `autoflow/core/pipeline.py` | `tests/test_smoke_phantoms.py` |
| interactive skeleton edit behavior | `autoflow/ui/app.py`, `autoflow/ui/editors.py` | `autoflow/core/pipeline.py` | GUI manual verification |

## Tests
- `/home/renyuyang/miniconda3/envs/autoflow311/bin/python -m pytest tests/test_smoke_phantoms.py -q`

## Common problems

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| skeleton step is skipped | no segmentation | create or load segmentation first |
| grouped skeleton is listed in the browser but absent from the 3D view | an older viewer did not resolve grouped `skeleton_<group>` data keys | install the current editable build and run `Generate Skeleton` again |
| skeleton contains many small branches | noisy segmentation | raise cleanup thresholds or improve segmentation |
| one grouped vessel disappears | every component in that group fell below the active cleanup threshold | lower `min_cc_volume_mm3` or `cc_rel_min_ratio`, or improve the segmentation |
| special-label result is unexpected | the selected strategy or special-label list does not match the group | set `special_handling` to `three_pass_merge` or `contact_surface`, then check `special_contact_labels` |
| `willis_ring` is not shown | no reliable mixed anterior/posterior arterial cycle was found, or one cranial arterial group is missing | inspect the source labels and graph connectivity; `not_detected` does not modify the three fixed cranial groups |
| `Edit Skeleton` is unavailable | more than one segmentation group is active | use a single-group case or simplify `configs/labels.json` |
