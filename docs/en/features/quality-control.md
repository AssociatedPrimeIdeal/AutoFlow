# Feature: Quality Control

## Status

| Entry point | Status | Notes |
| --- | --- | --- |
| GUI | Supported | interactive staged check table and JSON export in `Review & Export` |
| CLI | Supported | every processed case writes `quality_report.json` |
| Python API | Supported | batch summaries include the report; direct report helpers are also available |

## What It Does

Quality control converts pipeline state into explicit `pass`, `warn`, `fail`, or `not_run` checks. It currently screens physical geometry, finite velocity values, VENC saturation, background-phase provenance, segmentation content and temporal stability, graph topology, plane alignment, plane sampling, flow internal consistency, and PWV fit quality.

Exclusion provenance is available before plane metrics run. Expand temporal label voting by label and phase to see partial or complete support loss, its fraction, and which labels won those voxels. Separate checks report optional 4D cleanup, merged-group component filtering, skeleton-mask preprocessing, and plane selection/exclusion decisions.

`Per-phase group-mask exclusions` shows actual downstream temporal-mask removal separately from phase-to-static voting differences. Check summaries and expandable `details` link to the complete canonical data in `exclusions` through `provenance_path`.

The report contains the topology computed during Generate Graph. Each group is divided into connected subgroups. Each subgroup's inlet arms start at G1; splits/merges continue at one plus the highest upstream grade. Labels annotate paths without resetting this grade. Every parent and directed-loop region remains in `graph_hierarchy`; the metric flow tree uses group/subgroup containers and a deterministic display forest with path labels, grades, and ids. Each path still reports mean ± population SD across its planes for net flow, peak flow rate, mean velocity, and peak velocity. Old workspaces without topology metadata retain the legacy label-named metric tree until Generate Graph runs again.

The report is for engineering review prioritization. It does not establish clinical validity and must not be interpreted as a diagnostic conclusion.

`graph_hierarchy.groups[].cycle_cleanup` and `joint_systems[].cycle_cleanup` record the automatic positive-single-label cleanup before short-terminal pruning. The v2 audit includes the Flow-C cutoff and connectivity guard, pre/post cycle ranks, protected-edge and unknown-node counts, per-label decisions, removed/rejected path geometry and C values, and input-node indices. Missing or zero valid flow produces `flow_signal_unavailable`; it is not treated as evidence for path deletion. Radius is not used. Cross-label/unknown-node edges are protected; the default connectivity guard retains label-interface reachability and original endpoints. Disabling the guard permits component/path loss and is recorded explicitly. `L?` can remain after cleanup when velocity direction or an upstream relationship is still unresolved. Cleanup provenance is preserved in workspace saves and the canonical topology exports; these removals do not edit the segmentation or source skeleton.

`graph_hierarchy.groups[].junction_cluster_cleanup`, `joint_systems[].junction_cluster_cleanup` and the Willis report record the earlier junction stage as `autoflow.junction_cluster_cleanup.v1`. They include the enabled/extent/cycle settings, detected and eligible cluster counts, bounds and input geometry, source-label composition, boundary ports, same-label routing decisions, removed-edge geometry, pre/post cycle ranks and explicit skip reasons. Input-node ids refer to this stage before compaction, not final Graph ids. Mixed-label clusters can retain protected cross-label cycles. The subsequent Flow audit starts from the routed graph; read the two stages in order.

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
4. inspect warnings and failures; expand `Temporal label voting` into labels and phases, and `Plane selection and exclusions` into labels and candidates; hover results for the full text
5. inspect the collapsed level-0 vessel rows; expand a row for downstream branches and expand `Planes` for individual measurements
6. click `Export QC Report`

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
| path topology, flow direction, and segmentation labels | for the vessel hierarchy | parent/child order, readable branch names, and label-named junction equations |
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
| `planes.fov_boundary_filter` | bool | `true` | `configs/planes.json`; Generate Planes settings; API | applies FOV filtering to generated planes, independently of branch filtering | `autoflow/algorithms/metrics/fov.py` |
| `planes.fov_boundary_max_fraction` | fraction | `0.25` | `configs/planes.json`; GUI; CLI `--fov-boundary-max-fraction`; API | excludes a generated plane when FOV contact length / outer ROI perimeter reaches the threshold | `autoflow/algorithms/metrics/fov.py` |

## Outputs

| Output | Created when | Meaning |
| --- | --- | --- |
| `quality_report.json` | CLI or Python case processing completes | schema, overall state, counts, expandable checks, run context, `flow_hierarchy`, and complete `exclusions` provenance |
| `summary.json -> quality_report` | case processing completes | embedded report for batch aggregation |
| GUI QC check table | `Refresh QC` runs | current workspace checks, including unsaved interactive edits |
| GUI vessel flow tree | `Refresh QC` runs | expandable display roots, group grades, branches, conservation equations, and plane details |
| `graph_hierarchy` | graph step has run | complete group graph analysis, all parents, null unresolved grades, directed loops, local label depths, and optional joint cerebral view |

`flow_hierarchy` uses schema `autoflow.flow_hierarchy.v1`. Its `labels` array contains per-label aggregate statistics. `junctions` contains the incoming and outgoing path ids, display names, equation, and consistency score. `roots` contains the recursive path tree; every path includes its statistics, path internal consistency, related junctions, child paths, and distance-sorted planes.

`graph_hierarchy` uses schema `autoflow.graph_hierarchy.v1` and is independent of plane metrics. `centerline.hierarchy` warns when native path depths remain undefined or junction directions conflict. `flow_hierarchy.roots[].graph_topology` retains the corresponding path record, including all upstream references and loop ids. The display forest is a navigation aid; it does not remove additional graph parents or physical cycle edges. See [Graph and paths](graph-paths.md) for exact grading, joint analysis, and scan-boundary limitations.

`graph_hierarchy.groups[].subgroups` and `joint_systems[].subgroups` summarize each physical graph component. `flow_hierarchy.subgroups` adds native component summaries with their display-root ids; each recursive path node carries `subgroup_id` and `subgroup_index`. Flat `roots` remain available for existing consumers. The GUI groups these roots under `Group -> Subgroup` before showing downstream paths and planes.

### Temporal label voting and preprocessing

`exclusions.segmentation_preprocess` uses schema `autoflow.segmentation_preprocess_qc.v1`. It is computed while preprocessing and survives workspace saves. Its `labels` array includes every positive input label, including labels completely lost before group construction. Each entry records label id/name, group names, input and voting-input unions, voted 3D support, group-retained support, skeleton preprocessing support, and zero-based `phases`.

| Quantity | Numerator | Denominator / interpretation |
| --- | --- | --- |
| `vote_removed_union_fraction` | union voxels of this label won by another label or background | this label's spatial union over voting-input phases; a coordinate counts once |
| `phases[].vote_removed_fraction` | this phase's label voxels whose winning 3D label differs | this label's voxel count in this phase **after optional 4D cleanup** |
| `phases[].cleanup_removed_fraction` | this phase's label voxels removed before voting | this label's voxel count in the original input phase |
| `component_removed_fraction` | label voxels surviving morphology but removed by merged-group component filtering | this label's voted voxels still present after morphology |
| `phases[].component_removed_fraction` | this phase's label voxels surviving voting and morphology but removed by 3D component filtering | this phase's overlap with the voted label after morphology |
| `phases[].temporal_group_removed_fraction` | this phase's label voxels missing from the downstream temporal group mask | this phase's voting-input label voxels; reports group unavailability and independent per-phase component filtering |
| `morphology_removed_fraction` | voted label voxels absent after morphology, before the component filter | this label's voted 3D support; these are mask voxels, not thinned skeleton points |

An absent label in a phase has a `null` loss fraction (GUI: `N/A`), not 100%. Per-phase `vote_overlap_voxels`, `vote_removed_voxels`, and `vote_added_voxels` distinguish spatial support loss and gain; a net-volume comparison alone cannot do this. `vote_destinations` separates background (`label=0`) from named competing labels and reports their counts and fractions using the same source denominator. Partial changes are reported numerically; complete phase or label losses raise a review warning.

Voting selects the most frequent label at each voxel, including background; ties select the smallest numeric label. It does not require more than half the votes. A phase-to-3D mismatch can reflect normal lumen motion, so it is not automatically a segmentation error. Original input labels are unchanged. Completely lost 3D labels can also become unavailable for downstream group construction; their actual temporal-mask losses are reported separately. `cleanup_4d`, `group_filters` and `preprocessing_order` record the applied settings and morphology-before-component order. Older saved reports retain their own denominator descriptions.

### Plane selection and exclusions

`exclusions.plane_layout` mirrors the complete generation layout from `plane_qc.json` (nested under `plane_layout` there after metrics). Per-label statuses distinguish intentionally disabled rules, omitted labels, input/preprocessing absence, missing paths and incomplete generation. Root selection is automatic; there is no manual `needs_selection` state. `main_path`, `main_path_count`, `plane_count_per_main_path` and total `requested_plane_count` describe the plan. Every item in `selected_main_paths` retains raw-root identity, placement segment, length, direction confidence, selection reason and final retained count after branch/FOV filtering. `topology_candidates` explains why other raw segments were not selected. Multiple valid roots are all retained and receive the configured count independently; other labels need not exist. `selection_warnings` raises review status for missing/unreliable flow or a rootless cycle that required a longest-segment fallback. `topology_gap_links` reports small background-gap connections; `smoothing_label_restored_samples` counts placement samples restored to source-label support. `omitted_positions` includes geometry for centers outside the requested label. `excluded_plane_counts` counts rejection reasons for each label.

Every branch/FOV candidate records local center, normal, path distance, group/path ids, a stable global `plane_index_source`, and its final `plane_index` (`null` when excluded). Rejected candidates therefore cannot be confused with renumbered retained planes. `reason=no_branch_support` means no supported ROI; `reason=fov_boundary_contact` means the configured FOV threshold was reached. Its `fov_boundary` object includes `perimeter_mm`, `contact_length_mm`, `contact_fraction`, touched `faces`, world-mm FOV bounds, threshold, numerical tolerance, and evaluation status.

The FOV probe uses the temporal-union, label/branch-supported connected ROI. Only outer-contour segments lying on the actual image box faces count as contact. Vessel walls, label interfaces inside the image, interior holes, isolated tangent points, and the GUI display rectangle do not count. The box follows the existing cell geometry: `origin` to `origin + shape * spacing`. No voxel-wide proximity margin is used. A missing contour is reported and kept for review. Imported and manually edited planes are not automatically deleted by this generation filter.

A path with no valid planes remains in the graph for topology, but its path IC is undefined and any fork depending on its missing metrics is reported as incomplete.

The overall state is `not_ready` when a required check fails, `needs_review` when it warns, `incomplete` when required stages have not run, and `ready` when all required available checks pass.

## Limitations

- thresholds are screening defaults, not population-derived clinical limits
- VENC screening samples the array for responsiveness and does not perform phase unwrapping
- graph cycles and disconnected segmentation components can be intentional in some anatomies and therefore require review
- pressure solver residuals, WSS sensitivity intervals, uncertainty maps, and registration QC are not yet available in the v1 schema
- a `ready` report means the implemented automated checks passed; it does not prove ground-truth accuracy
- dropped branch-unsupported planes are not treated as zero flow; review their entries in `plane_qc.json` when a path has fewer planes than requested
- the FOV threshold is an engineering screening default and uses temporal-union geometry, not a per-phase exclusion vote; review individual phase contours when motion is material
- legacy workspaces without recorded preprocessing provenance need preprocessing rerun to populate label-loss details
- when topology produces more than one disconnected level-0 path for the same label, the highest-flow root keeps the bare label and additional roots use `-R2`, `-R3`, and so on
- branch numbering is run-specific because it follows measured mean net flow; it is stable for unchanged topology and metrics, but it is not a permanent anatomical identifier across independently processed cases

## Where To Change Code

| Change you want | Edit here | Also check | Tests |
| --- | --- | --- | --- |
| add or change a check | `autoflow/quality.py`, `autoflow/quality_exclusions.py` | `docs/en/features/quality-control.md` | `tests/test_smoke_phantoms.py` |
| temporal label-loss counts and denominators | `autoflow/algorithms/preprocess_qc.py` | `autoflow/core/pipeline.py`, `autoflow/core/models.py` | `tests/test_smoke_phantoms.py` |
| change automatic CLI export | `autoflow/processing.py` | `autoflow/api.py`, `docs/en/user/outputs.md` | `tests/test_smoke_phantoms.py`, `tests/test_pressure_gradient_phantom.py` |
| change GUI presentation | `autoflow/ui/app.py` | `docs/en/user/gui.md` | manual GUI verification plus smoke coverage |

## Tests

- `/home/renyuyang/miniconda3/envs/autoflow311/bin/python -m pytest tests/test_smoke_phantoms.py -q`
- `/home/renyuyang/miniconda3/envs/autoflow311/bin/python -m pytest tests/test_pressure_gradient_phantom.py -q`
- manually refresh the GUI table before and after changing a plane or segmentation

## Common Problems

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| overall state is `incomplete` | centerline, planes, or metrics have not run | finish the missing workflow stage and refresh QC |
| VENC saturation warns | velocities approach the encoded limit | inspect aliasing and acquisition metadata before accepting quantitative results |
| plane geometry warns after cross-case import | cases are unregistered or target paths differ | use relative-centerline mapping or register cases before world-coordinate import |
| flow consistency fails | segmentation, orientation, branch ownership, or plane placement is inconsistent | inspect the named paths/forks and their time curves |
| GUI report differs from saved batch report | the workspace was edited after the batch run | refresh and export the GUI report again |
