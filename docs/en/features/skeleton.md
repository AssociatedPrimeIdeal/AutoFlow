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
2. merges labels into configured groups from `configs/labels.json`
3. applies per-group dilation, erosion, closing, opening, and Gaussian smoothing when enabled; closing is off by default (`do_closing=false`, `closing_iters=0`, including the aorta override)
4. applies optional special-label contact separation before component selection
5. filters connected components last: the default `hybrid` rule retains every merged-group component with volume at least `max(50 mm³, 0.10 × largest component volume)`
6. applies the same volume rule independently to every 4D group frame for phase-dependent metrics, without skeleton morphology or cropping to the static majority mask
7. skeletonizes each group separately

Component connectivity uses the existing 6-neighbour (face-connected) rule. Attached small labels survive as part of the retained group. Disconnected left/right vessels both survive when each meets the volume threshold; components below it are removed. Explicit `cc_filter_mode=largest` still retains only one component. Source labels are never expanded into voxels added by morphology. Component decisions are projected back onto their original source support, so ordinary erosion does not silently shrink the labels used to classify paths. The optional segmentation-stage `cleanup_4d_components` setting remains a separate, explicitly enabled per-label cleanup.

Skeletonization follows preprocessing, and graph construction follows skeletonization. Morphology changes the group binary mask, while label attribution uses the source label volume. A skeleton junction on a newly filled voxel can therefore be connected in the graph yet have source label `0`; clipping paths to one label can lose that junction and create apparent extra roots. Moving component filtering last cannot remove such a voxel if it belongs to a retained component. Closing is disabled by default; no graph connections are added to repair these junctions.

Preprocessing records every input label in `Workspace.segmentation_preprocess_qc`, including labels that disappear before groups are built. [QC](quality-control.md#temporal-label-voting-and-preprocessing) distinguishes optional 4D cleanup, temporal voting, merged-group component removal, and skeleton-mask preprocessing. Expand `Temporal label voting` by label and phase to inspect partial-loss fractions, overlap, gains, and winning labels. Phase loss divides by that phase's voting-input label count; union loss divides by the label's spatial union, counting each location once. Absent-phase fractions are `N/A`. These phase-to-static-topology differences can reflect motion and are separate from actual losses in the downstream temporal group masks.

The built-in label groups keep the non-cranial workflows unchanged. Cranial
labels are organized as `intracranial_anterior_arteries` (`LICA`, `RICA`,
`ACA`, `LMCA`, `RMCA`), `vertebrobasilar_arteries` (`LVA`, `RVA`, `BA`,
`LPCA`, `RPCA`), `intracranial_veins` (`LTS`, `SSS`, `RTS`, `StrS`), and
`jugular_veins` (`LIJV`, `RIJV`).

Graph generation applies compact junction-cluster routing after isolated triangle contraction, then configurable Flow-C path filtering inside one positive source label, then short terminal-branch pruning. Junction routing preserves original point coordinates and uses only existing same-label edges; compact mixed clusters retain protected cross-label interfaces. Cross-label and unknown-node edges are excluded from this filter; the default connectivity guard retains each label component, its endpoints and interface connectivity. Source segmentation and skeleton masks/points are unchanged; inspect the graph and `graph_hierarchy.json` cleanup provenance after regeneration. See [Graph and paths](graph-paths.md) for the exact policy and limits.

After graph generation, AutoFlow performs a conservative Willis-ring check on
the two arterial masks. A detected cycle containing labels from both arterial
groups is exposed as a derived `willis_ring` graph overlay. The overlay does
not replace source labels, ordinary groups, planes, or metrics. Cases without
a reliable mixed arterial cycle are reported as `not_detected` and keep their
anterior and vertebrobasilar graphs separate.

`special_handling` selects the special-label strategy. `three_pass_merge` (the
default) skeletonizes `A+B`, `A+C`, and so on, then merges nearby points. The legacy strategy name remains `three_pass_merge` even when more than three branches are configured. Its default protected list is `RBCT`, `CCA`, `LBCT`, `HA`, `SMA`, `LRA`, `RRA`; the shared base normally contains `AAO`, `ARCH`, `DAO`.
Each pass excludes the other configured special labels after morphology and
before its final component filter. This prevents closing or dilation from growing back into a
neighbouring special branch and creating false loops, short tails, or a lost
branch endpoint during graph cleanup. Morphology still applies to the shared
trunk and background; the original segmentation labels remain unchanged.
`contact_surface` keeps the previous contact-surface separation algorithm. Graph construction also excludes direct contacts sampling two different configured independent branch labels, so merging the skeleton point sets cannot reconnect HA/SMA/LRA/RRA to each other through those contacts. Trunk-to-branch connections remain eligible. This filter is active for `three_pass_merge`, and for `contact_surface` when contact separation is enabled.

Graph paths derived from the skeleton are split at graph nodes with degree
`>= 3`, so fork existence does not depend on noisy or locally ambiguous flow
directions. Flow is sampled on the segmentation-filtered path geometry to
orient each path and assign incoming/outgoing roles after topology has been
established.

Before path extraction, an isolated triangular junction with at least three
distinct external neighbors is contracted to the physical-coordinate centroid
of its three graph points, regardless of their source labels. Mixed-label or
unknown-label triangles are not excluded. Existing external edges are remapped to that point;
there is no new neighbor search. This replaces the earlier triangle-edge
deletion that could create a local gap despite a long alternate route. Original
skeleton points and masks are preserved. Overlapping triangles, closed rings
and contractions that would merge distinct external edges are retained; see
[Graph and paths](graph-paths.md) for the scope and centroid limitations.

During graph generation, terminal branches shorter than `min_edge_points` graph
edge segments are removed. The default is `3`: arms with one or two edges
are removed, while a three-edge arm is retained. Set it to `0` or `1` to disable
this cleanup. This filters graph edges and derived paths while leaving the
original skeleton mask and points available for inspection.

## When to use it

Regenerating the skeleton clears the previous graph, paths, planes, plane measurements, PWV, and plane pathlines from the workspace. Generate the graph and planes again before measuring them. This applies to GUI, CLI pipeline orchestration, and Python API step calls.
- use it after segmentation is available
- use it before graph and plane generation
- use it when you want grouped vessel trees instead of one merged binary tree

## Quick use

### GUI
1. load or create segmentation
2. right-click `Generate Skeleton` for cleanup and smoothing settings, then Save and run. Filter thresholds and Gaussian sigma appear only when applicable
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
| label and phase loss provenance | `autoflow/algorithms/preprocess_qc.py` | `autoflow/core/pipeline.py`, `autoflow/quality_exclusions.py` | `tests/test_smoke_phantoms.py` |
| grouped skeleton pipeline flow | `autoflow/core/pipeline.py` | `autoflow/algorithms/skeleton.py` | `tests/test_smoke_phantoms.py` |
| Willis-ring topology check | `autoflow/algorithms/intracranial.py`, `autoflow/core/pipeline.py` | `autoflow/core/models.py`, `autoflow/ui/viewer.py` | `tests/test_smoke_phantoms.py` |
| short terminal branch cleanup | `autoflow/algorithms/graph.py` | `autoflow/core/pipeline.py`, `autoflow/core/models.py` | `tests/test_smoke_phantoms.py` |
| graph fork detection and flow-based path orientation | `autoflow/algorithms/branch.py`, `autoflow/algorithms/planes.py` | `autoflow/core/pipeline.py` | `tests/test_smoke_phantoms.py` |
| interactive skeleton edit behavior | `autoflow/ui/app.py`, `autoflow/ui/editors.py` | `autoflow/core/pipeline.py` | GUI manual verification |

## Tests
- `/home/renyuyang/miniconda3/envs/autoflow311/bin/python -m pytest tests/test_smoke_phantoms.py -q`
- The close-branch phantom checks that closing and dilation cannot reintroduce
  excluded special labels into a per-pass skeleton input, and that the source
  segmentation remains unchanged.

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
