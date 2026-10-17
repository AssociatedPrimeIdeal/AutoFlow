# Feature: Graph And Paths

## Status

| Entry point | Status | Notes |
| --- | --- | --- |
| GUI | Supported | includes interactive graph editing for single-group cases |
| CLI | Supported | generated in batch order |
| Python API | Supported | through pipeline execution |

## What It Does
Graph and path generation converts the skeleton into nodes, edges, branches, forks, and path objects used by planes and downstream metrics.
The thinning/smoothing sources and AutoFlow-specific graph logic are distinguished in [Scientific references: centerline](../references/index.md#centerline-skeleton-and-paths).
When segmentation is grouped, graph generation runs per group, then combines path endpoints, fork nodes, fork roles, and path indices into one workspace-wide numbering while preserving each path `group_name`.

Skeleton graph adjacency is configurable: `graph_connectivity=26` (default) connects face, edge and corner neighbors; `6` connects face neighbors only. Coordinates are compared in voxel units, so anisotropic spacing is respected. This changes graph edges, not thinning or preprocessing's component filtering. Face-only adjacency can fragment diagonal skeletons, increasing subgroups and per-component plane counts. Triangle-junction contraction and terminal pruning run after adjacency construction.

An isolated triangle with at least three distinct external neighbors is replaced by one junction at the arithmetic mean of its three physical-coordinate points. The three original graph points and internal triangle edges are replaced; their existing external edges are remapped to the new point, and graph indices are compacted. Adjacency is not searched again after this move: a retained edge can exceed the original neighbor radius once its endpoint moves to the centroid. This avoids deleting a local connection simply because a long alternate route exists elsewhere in the graph. Original skeleton points and masks remain unchanged for inspection. GUI Generate Graph, CLI batches and Python pipeline calls all use this cleanup; regenerate graph and planes to apply it to an existing workspace.

Graph arms are analyzed before final path geometry is materialized. Every graph edge is retained in the arm decomposition, including closed components without an endpoint. Each anatomical group is divided into subgroups by connected components of its physical graph. Each subgroup is graded independently. Labels annotate an arm and never split connectivity or restart the subgroup grade.

Observed inlet arms receive level 1 within their subgroup. A split or merge gives downstream arms `1 + max(upstream levels)`. Multiple inlets remain in one subgroup when they share physical connectivity. An unresolved or inlet-free subgroup does not affect other subgroups. All parent/child references and inlet origins are preserved. At a merge, the Browser uses one deterministic display parent and exposes the other connections as upstream references. This is topological depth within the scanned connected component, not a clinical anatomical branch name.

The exported `subgroups` list identifies every component and records its path ids, label names, inlet/outlet ids, loop ids and depth summary. Paths, junctions and loops carry `subgroup_id` and zero-based `subgroup_index`. `subgroup_level` is the path depth; `group_level` remains an equal-valued compatibility field. Component numbering is deterministic for the generated graph, but can change when the graph is regenerated or edited.

Newly generated path ids follow anatomical group order, then subgroup order, then ascending level (G1, G2, etc.). Unresolved G? paths follow the ranked paths of their subgroup; ties retain graph traversal order. Native path ids remain globally unique and zero-based across groups. Joint cerebral analysis applies the same ordering with its separate ids. Reordering happens before branch rasterization, fork construction and plane generation, so geometry arrays, branch labels, parent/child ids, planes and exports use the same numbering. Existing saved workspaces retain their ids until Generate Graph runs again; regeneration also clears their previous plane layout.

Directed cycles are condensed into loop regions only for ranking. Their original arms and physical edges remain intact. A loop reached from an inlet has one region level; downstream arms continue from that level. A closed loop without an inlet has an undefined level. An undirected anatomical cycle may become a directed acyclic graph after flow orientation and therefore does not necessarily form a directed loop region.

Direction is the signed tangent/velocity alignment integrated over arc length and cardiac phases. Missing/zero/ambiguous flow remains unoriented; affected paths or unresolved upstream relationships display `G?`. A physical graph endpoint with reliable incoming velocity is an observed inlet; scan boundaries and segmentation fragments can introduce additional observed endpoints. No longest-path substitute is used for grading.

`label_local_levels` is an additional annotation on the same connectivity: the first label-bearing arm has local depth 0, followed by 1, 2, and so on. One arm may carry several labels. These local values do not create new anatomical labels; in particular, PV children are not automatically renamed RPV/LPV.

When both configured Willis roles (`anterior`, `posterior`) are present, AutoFlow also constructs a joint cerebral arterial analysis from their connected mask union. Disconnected mask components are skeletonized separately, so nearby disconnected vessels are not joined. The joint view preserves cross-group connectivity without changing source groups, ordinary paths, plane placement, or measurements.

GUI edits now invalidate dependent artifacts explicitly: skeleton edits clear
graph, paths, planes, and metrics; graph edits clear paths, planes, and metrics.
Run the next centerline action after finishing an edit.

## When To Use It

Regenerating the graph clears the previous plane layout, smoothed display paths, plane measurements, PWV, and plane pathlines. Generate planes again against the new graph.
- use it after a valid skeleton exists
- use it before plane generation
- use GUI editing when automatic graph structure needs correction

## Quick Use

### GUI
1. run `Generate Skeleton`
2. right-click `Generate Graph` to set Minimum terminal edge count, then Save and run. This pruning setting is no longer under Skeleton
3. inspect grouped graph, fork, and path objects in the browser and 3D view
4. if needed, click `Edit Graph` when exactly one segmentation group is active

The Browser shows `Group -> Paths -> Subgroup -> downstream paths` immediately after Generate Graph, with label names, `G1`, `G2`, etc., loop containers, and navigable `Upstream Path` references. G grades restart independently in each subgroup. Plane objects appear under their owning path after generation. Hover a path or select it to inspect direction confidence, subgroup membership, all parents, children, label-local depths and loop membership. Subgroup checkboxes control only their component's displayed objects; connection references do not duplicate visibility controls.

For cranial cases, `Joint cerebral arterial topology` is a separate analysis tree, also divided into connected subgroups. Its branches are initially hidden in 3D; enable their checkboxes to inspect the combined geometry. Selecting a joint branch highlights that analysis geometry and displays its attributes. Joint branches have their own analysis ids and do not own ordinary measurement planes.

Graph edit mode hides unrelated scene layers and shows the graph controls in the
step area. Select and drag nodes with the orange sphere; click an edge and press
Delete/Backspace to remove it. Press E, or click `Edge Mode`, then click two
nodes to add or remove their edge. `Save Changes` clears dependent paths,
planes, and metrics; `Cancel` or Esc leaves the graph unchanged.

### CLI

```bash
autoflow-run case.h5 --output-dir results/case
```

### Python API

Use the standard batch or single-case pipeline. Graph generation is part of the normal order.

```python
from autoflow.core.models import StepId
from autoflow.core.pipeline import PipelineEngine

PipelineEngine().run_step(workspace, StepId.GENERATE_GRAPH, print)
hierarchy = workspace.graph_hierarchy
for group in hierarchy["groups"]:
    arms = {arm["path_index"]: arm for arm in group["paths"]}
    for subgroup in group["subgroups"]:
        for index in subgroup["path_indices"]:
            arm = arms[index]
            print(subgroup["subgroup_id"], index, arm["subgroup_level"], arm["parent_path_indices"])
```

## Inputs

| Input | Required | Meaning |
| --- | --- | --- |
| skeleton | yes | centerline input for graph construction |
| segmentation | indirectly yes | needed to build the skeleton first |

## Parameters

See [skeleton parameters](../user/parameters.md#skeleton), [labels parameters](../user/parameters.md#labels), [CLI flags](../user/cli-parameters.md), and [API fields](../user/api-parameters.md) for complete type/default/unit/effect/owner tables. Dictionary controls are expanded in [Structured parameters](../user/parameter-schemas.md).

| Parameter | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `graph_connectivity` / `--graph-connectivity` | int enum | 26 | `configs/skeleton.json`; API; GUI Generate Graph | 6 = face neighbors; 26 = face/edge/corner neighbors. Group overrides allowed; joint cerebral/Willis/PWV graphs use the shared setting | `autoflow/algorithms/graph.py`; `autoflow/core/pipeline.py` |
| `labels.label_groups` | mapping | configured anatomical groups | `configs/labels.json` | Defines anatomical containers; connected graph subgroups are detected automatically within each container | `autoflow/algorithms/hierarchy.py` |
| `willis_ring_role` | group string | anterior/posterior for the two cerebral arterial groups | per group in `labels.json` | Enables the joint cerebral analysis when both roles are present | same |
| `direction_confidence_threshold` | float | 0.05 | algorithm constant, reported in JSON | Lower absolute signed alignment leaves direction/depth unresolved | same |

## Outputs

| Output file or object | Created when | Meaning |
| --- | --- | --- |
| graph data in workspace | graph step succeeds | nodes and edges |
| branches and paths in workspace | graph step succeeds | branch and path topology; nearest-path labels are assigned in one vectorized operation, and grouped endpoints, fork nodes, and fork roles use global ids |
| graph scene objects | GUI graph step succeeds | grouped graph objects such as `graph_aorta_systemic_branches` and `smooth_path_aorta_systemic_branches_3` |
| `graph_hierarchy.json` | graph step succeeds | `autoflow.graph_hierarchy.v1`: groups, connected subgroups with independently graded paths, split/merge junctions, inlet/outlet ids, loops, local label depths, and optional joint systems |
| `Workspace.graph_hierarchy` | graph step succeeds | same analysis, preserved by workspace save/restore; individual records also appear under `path_info[].hierarchy` |
| `quality_report.json -> graph_hierarchy` / `summary.json -> graph_hierarchy` | report/batch summary is saved | same canonical topology; unresolved grades produce the `centerline.hierarchy` review check |

## Limitations
- depends on usable skeleton quality
- Triangle contraction is limited to isolated triangle junctions. Overlapping triangles, triangles with fewer than three distinct external neighbors, and contractions that would merge distinct external edges into one are retained. Larger rings and separate external routes are not pruned by this cleanup. The centroid is not constrained to a voxel center or guaranteed to lie inside the segmentation; this is graph cleanup, not a lumen or segmentation repair.
- GUI: right-click Generate Graph → Skeleton graph connectivity; CLI: `--graph-connectivity 26`; API: `AutoFlowConfig(graph_connectivity=26)`. Regenerate graph and planes after changing it.
- interactive graph editing is only available when exactly one segmentation group is active
- Grades depend on scan coverage, segmentation connectivity and reliable flow direction. They are not guaranteed anatomical generations.
- Unresolved paths keep a null grade; closed unrooted loops do not acquire an artificial inlet.
- Joint cerebral branches are analysis geometry with separate ids. They do not replace ordinary path/plane ids or generate extra planes.

## Where To Change Code

| Change you want | Edit here | Also check | Tests |
| --- | --- | --- | --- |
| graph construction | `autoflow/algorithms/graph.py` | `autoflow/core/pipeline.py` | `tests/test_smoke_phantoms.py` |
| branches or paths logic | `autoflow/algorithms/branch.py`, `autoflow/algorithms/paths.py` | `autoflow/core/pipeline.py` | `tests/test_smoke_phantoms.py` |
| topology depth, joins, loops or label-local relationships | `autoflow/algorithms/hierarchy.py` | `autoflow/core/models.py`, `autoflow/quality.py`, `autoflow/ui/app.py`, `autoflow/ui/viewer.py` | `tests/test_smoke_phantoms.py` |
| interactive graph editing | `autoflow/ui/app.py`, `autoflow/ui/editors.py` | `autoflow/ui/viewer.py` | GUI smoke coverage |

## Tests

- `/home/renyuyang/miniconda3/envs/autoflow311/bin/python -m pytest tests/test_smoke_phantoms.py -q`
- Triangle-junction phantoms check the physical centroid, external-edge remapping without another distance search, input immutability, node/edge permutation, preserved external rings and disconnected components, and conservative handling of closed, overlapping or parallel-connection triangles.
- Phantom coverage includes level-ordered path ids and branch/fork references, independently graded disconnected subgroups, multiple inlets within one subgroup, unequal-depth merging, labels crossing one arm, rooted/unrooted directed cycles, missing direction, joint cerebral connectivity, workspace restoration, Browser references and analysis geometry.

## Common Problems

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| graph step is skipped | no skeleton | run `Generate Skeleton` first |
| graph and fork objects are listed but absent from the 3D view while paths remain visible | an older viewer did not resolve grouped `graph_<group>` and `forks_<group>` data keys | install the current editable build and run `Generate Graph` again |
| path structure looks wrong | skeleton topology is noisy | improve segmentation or edit graph manually |
| `Edit Graph` is unavailable | more than one segmentation group is active | use a single-group case, or change group configuration |
| path shows `G?` | direction is unreliable, upstream flow is unresolved, or the component is an unrooted loop | inspect the path tooltip and `centerline.hierarchy` QC; review velocity/segmentation instead of assuming an anatomical generation |
| several rows appear to feed the same downstream path | a genuine merge has several parents | inspect upstream references; the display tree keeps one canonical path row and preserves all graph parents |
