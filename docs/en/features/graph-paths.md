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

Skeleton graph adjacency is configurable: `graph_connectivity=26` (default) connects face, edge and corner neighbors; `6` connects face neighbors only. Coordinates are compared in voxel units, so anisotropic spacing is respected. This changes graph edges, not thinning or preprocessing's component filtering. Face-only adjacency can fragment diagonal skeletons, increasing subgroups and per-component plane counts. The automatic order is adjacency construction and label-independent triangle-junction contraction, then compact junction-cluster routing, then single-label Flow-C filtering, then short-terminal pruning, then path extraction and grading.

An isolated triangle with at least three distinct external neighbors is replaced by one junction at the arithmetic mean of its three physical-coordinate points. Source labels do not restrict this pass: mixed-label and unknown-label triangles can contract when the geometric/topological checks pass. The three original graph points and internal triangle edges are replaced; their existing external edges are remapped to the new point, and graph indices are compacted. Adjacency is not searched again after this move: a retained edge can exceed the original neighbor radius once its endpoint moves to the centroid. This avoids deleting a local connection simply because a long alternate route exists elsewhere in the graph. Original skeleton points and masks remain unchanged for inspection. GUI Generate Graph, CLI batches and Python pipeline calls all use this cleanup; regenerate graph and planes to apply it to an existing workspace.

Compact junction-cluster routing runs automatically after isolated triangle contraction and before Flow-C filtering. It finds connected components of adjacent nodes with degree at least three, with at least three nodes. A candidate must have internal cycle rank E-V+1 at least 2 by default and an extent of at most 8 voxels along every axis (physical coordinates divided by spacing, including anisotropic spacing). These are graph-regularization criteria, not proof of an anatomical error. Long or low-cycle junction regions are reported and retained.

Inside each candidate, the algorithm handles each connected same-positive-label edge patch independently. Existing cross-label edges, edges with unknown endpoint labels, and edges traversing another positive label are protected. Every node connected to the outside of its patch is an interface port. With at least two retained ports, it chooses the existing node nearest the patch's physical centroid, builds a shortest-path tree using existing physical edge lengths, and keeps the union of routes to all ports and old endpoints. An unnecessary root spur is pruned. Other internal owned edges and unused internal points are removed; points never move and no new edge is added. Fewer than two ports leaves that patch unchanged. Mixed-label candidates can therefore be partially simplified while retaining cross-label cycles. Global and within-label component connectivity, external connections, original endpoints and interface ports are checked before returning the graph.

The same stage is used by native groups, the joint cerebral view and the Willis overlay. Native groups support per-group overrides; joint cerebral/Willis use the global values. Source segmentation and source skeleton points remain unchanged. Routing does not use lumen radius or Flow C as edge costs. Regenerate Graph and Planes to apply a changed setting; newly generated final paths and planes use the routed graph.

Single-label Flow-C path filtering runs automatically from GUI, CLI and Python graph generation. Only existing edges with the same positive source label at both endpoint nodes and no other positive label sampled along the edge are eligible for removal. Cross-label edges, edges with unknown endpoint labels, and edges crossing another positive label are protected. Interior background samples on an existing digital diagonal connection retain its endpoint label; attribution creates no new edges across background gaps. Interface nodes split compressed branches and remain present. With the default connectivity guard, each original within-label component stays connected and retains its endpoints, so paths between its label interfaces remain reachable; cross-label ring topology, including a Willis candidate spanning labels, is preserved. No whole cerebral-label exemption is used: a local self-contact loop inside LICA can be cleaned in either a neck or brain case.

Flow filtering temporarily compresses degree-2 connections into paths, splitting at every label-interface port. It evaluates C = abs(sum(v dot dl) / sum(norm(v) * norm(dl))) over valid velocity/tangent pairs and cardiac phases using the existing path-direction sampler. Paths with C strictly below `cycle_flow_threshold=0.3` are considered from lowest C upwards; canonical physical geometry breaks ties. A reverse-aligned path is retained because C uses the absolute alignment. Missing, zero or invalid masked flow is skipped. Lumen radius is not used, and there is no radius-based fallback to force a forest. Set the threshold to 0 to disable these deletions.

With `cycle_preserve_connectivity=true`, a proposed removal must retain each ORIGINAL same-label connected component, every original terminal endpoint and every label-interface port. Checking only the global component count is insufficient. A remaining singleton interface port can still connect through protected edges; a lone terminal point without a retained vessel edge cannot replace its component. An isolated closed ring is therefore retained if removing its only path would erase that component. High-C cycles and low-C paths blocked by these guards can remain. Setting the guard to false allows low-C paths to split or remove label components and terminal paths; protected cross-label/unknown edges themselves are still never deleted, but routes between interfaces are no longer guaranteed.

The GUI exposes both controls under right-click Generate Graph. CLI examples use `--cycle-flow-threshold 0.3 --cycle-preserve-connectivity`; Python uses `AutoFlowConfig(cycle_flow_threshold=0.3, cycle_preserve_connectivity=True)`. Both persist in `skeleton.json` and workspace saves. Native graphs accept the existing per-group preprocessing overrides; joint cerebral and Willis analysis use the global settings. Source segmentation, skeleton points and masks remain unchanged. Short-terminal pruning runs afterwards using its existing edge-count threshold, so its removals are separate from the Flow-filter guarantees.

Special-branch contact protection is applied during adjacency construction, before Flow filtering: the default independent branches are RBCT, CCA, LBCT, HA, SMA, LRA and RRA. Candidate edges sampling two distinct configured branch labels are excluded, preserving trunk-to-branch contacts. This prevents separately generated skeleton passes from rejoining through direct branch-to-branch adjacency. It is separate from the cross-label protection of the subsequent Flow filter.

Graph arms are analyzed before final path geometry is materialized. Every edge of the cleaned graph is retained in the arm decomposition, including protected closed components without an endpoint. Each anatomical group is divided into subgroups by connected components of its physical graph. Each subgroup is graded independently. Labels annotate an arm and never split connectivity or restart the subgroup grade.

Observed inlet arms receive level 1 within their subgroup. A split or merge gives downstream arms `1 + max(upstream levels)`. Multiple inlets remain in one subgroup when they share physical connectivity. An unresolved or inlet-free subgroup does not affect other subgroups. All parent/child references and inlet origins are preserved. At a merge, the Browser uses one deterministic display parent and exposes the other connections as upstream references. This is topological depth within the scanned connected component, not a clinical anatomical branch name.

The exported `subgroups` list identifies every component and records its path ids, label names, inlet/outlet ids, loop ids and depth summary. Paths, junctions and loops carry `subgroup_id` and zero-based `subgroup_index`. `subgroup_level` is the path depth; `group_level` remains an equal-valued compatibility field. Component numbering is deterministic for the generated graph, but can change when the graph is regenerated or edited.

Newly generated path ids follow anatomical group order, then subgroup order, then ascending level (G1, G2, etc.). Unresolved G? paths follow the ranked paths of their subgroup; ties retain graph traversal order. Native path ids remain globally unique and zero-based across groups. Joint cerebral analysis applies the same ordering with its separate ids. Reordering happens before branch rasterization, fork construction and plane generation, so geometry arrays, branch labels, parent/child ids, planes and exports use the same numbering. Existing saved workspaces retain their ids until Generate Graph runs again; regeneration also clears their previous plane layout.

Directed cycles remaining after the single-label cleanup are condensed into loop regions only for ranking. Their arms and physical edges remain intact. A loop reached from an inlet has one region level; downstream arms continue from that level. A closed loop without an inlet has an undefined level. An undirected anatomical cycle may become a directed acyclic graph after flow orientation and therefore does not necessarily form a directed loop region.

Direction is the signed tangent/velocity alignment integrated over arc length and cardiac phases. Missing/zero/ambiguous flow remains unoriented; affected paths or unresolved upstream relationships display `G?`. A physical graph endpoint with reliable incoming velocity is an observed inlet; scan boundaries and segmentation fragments can introduce additional observed endpoints. No longest-path substitute is used for grading.

`label_local_levels` is an additional annotation on the same connectivity: the first label-bearing arm has local depth 0, followed by 1, 2, and so on. One arm may carry several labels. These local values do not create new anatomical labels; in particular, PV children are not automatically renamed RPV/LPV.

When both configured Willis roles (`anterior`, `posterior`) are present, AutoFlow also constructs a joint cerebral arterial analysis from their connected mask union. Disconnected mask components are skeletonized separately, so nearby disconnected vessels are not joined. Joint analysis and the Willis overlay use the same positive-single-label cleanup while preserving cross-label interfaces. The joint view preserves cross-group connectivity without changing source groups, ordinary paths, plane placement, or measurements.

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
| `min_edge_points` / `--min-edge-count` | int | 3 | `configs/skeleton.json`; API; GUI Generate Graph | Remove terminal arms with fewer than three graph edges; retain three-edge arms and junction-to-junction links. `0` or `1` disables this cleanup. | `autoflow/algorithms/graph.py`; `autoflow/core/pipeline.py` |
| `graph_connectivity` / `--graph-connectivity` | int enum | 26 | `configs/skeleton.json`; API; GUI Generate Graph | 6 = face neighbors; 26 = face/edge/corner neighbors. Group overrides allowed; joint cerebral/Willis/PWV graphs use the shared setting | `autoflow/algorithms/graph.py`; `autoflow/core/pipeline.py` |
| `labels.label_groups` | mapping | configured anatomical groups | `configs/labels.json` | Defines anatomical containers; connected graph subgroups are detected automatically within each container | `autoflow/algorithms/hierarchy.py` |
| `willis_ring_role` | group string | anterior/posterior for the two cerebral arterial groups | per group in `labels.json` | Enables the joint cerebral analysis when both roles are present | same |
| `direction_confidence_threshold` | float | 0.05 | algorithm constant, reported in JSON | Lower absolute signed alignment leaves direction/depth unresolved | same |
| `junction_cluster_enabled` / `--junction-cluster-cleanup` | bool | true | `skeleton.json`; API; GUI Generate Graph | Enable compact junction routing between triangle contraction and Flow filtering. `--no-junction-cluster-cleanup` disables it. | `autoflow/algorithms/graph_junctions.py` |
| `junction_cluster_max_extent_voxels` | float | 8.0 | same; `--junction-cluster-max-extent-voxels` | Maximum candidate extent along each voxel axis; finite and at least 1. This bounds candidates, not a merge radius. | same |
| `junction_cluster_min_cycle_rank` | int | 2 | same; `--junction-cluster-min-cycle-rank` | Minimum internal independent-cycle count E-V+1; positive integer. | same |
| `cycle_flow_threshold` | float | 0.3 | `skeleton.json`, Generate Graph settings, CLI/API | Flow C cutoff in [0, 1]: delete only paths strictly below it; 0 disables deletions. Missing/zero valid flow is skipped; radius is not used. | `autoflow/algorithms/graph_cycles.py` |
| `cycle_preserve_connectivity` | bool | true | same | Protect each original within-label component, label interfaces and endpoints. False allows low-C paths to split/remove components; cross-label/unknown edges remain excluded from removal. | same |

## Outputs

| Output file or object | Created when | Meaning |
| --- | --- | --- |
| graph data in workspace | graph step succeeds | nodes and edges |
| branches and paths in workspace | graph step succeeds | branch and path topology; nearest-path labels are assigned in one vectorized operation, and grouped endpoints, fork nodes, and fork roles use global ids |
| graph scene objects | GUI graph step succeeds | grouped graph objects such as `graph_aorta_systemic_branches` and `smooth_path_aorta_systemic_branches_3` |
| `graph_hierarchy.json` | graph step succeeds | `autoflow.graph_hierarchy.v1`: groups, connected subgroups with independently graded paths, split/merge junctions, inlet/outlet ids, loops, local label depths, optional joint systems, and `junction_cluster_cleanup` / `cycle_cleanup` provenance for native/joint graphs |
| `Workspace.graph_hierarchy` | graph step succeeds | same analysis, preserved by workspace save/restore; individual records also appear under `path_info[].hierarchy` |
| `quality_report.json -> graph_hierarchy` / `summary.json -> graph_hierarchy` | report/batch summary is saved | same canonical topology; unresolved grades produce the `centerline.hierarchy` review check |

## Limitations
- depends on usable skeleton quality
- Triangle contraction is label-independent and limited to isolated triangle junctions. Overlapping triangles, triangles with fewer than three distinct external neighbors, and contractions that would merge distinct external edges into one are retained by that first pass. External graph connections are remapped to the centroid even across a label interface. Later junction routing and Flow filtering protect label interfaces in their own input graphs; those protections do not restrict the earlier triangle contraction. The centroid is not constrained to a voxel center or guaranteed to lie inside the segmentation, and its label is subsequently sampled from the unchanged source volume; this is graph cleanup, not a lumen or segmentation repair.
- Single-label cleanup preserves cross-label cycle topology; it does not identify anatomical Willis components. A genuine ring entirely assigned to one label has no anatomical exemption; flow and the selected connectivity guard determine whether its paths are retained. Connections with unknown endpoint labels remain protected and can leave unresolved cycles. Loop removal also does not guarantee reliable velocity direction or defined grades.
- GUI: right-click Generate Graph → Skeleton graph connectivity; CLI: `--graph-connectivity 26`; API: `AutoFlowConfig(graph_connectivity=26)`. Regenerate graph and planes after changing it.
- interactive graph editing is only available when exactly one segmentation group is active
- Grades depend on scan coverage, segmentation connectivity and reliable flow direction. They are not guaranteed anatomical generations.
- Unresolved paths keep a null grade; closed unrooted loops do not acquire an artificial inlet.
- Joint cerebral branches are analysis geometry with separate ids. They do not replace ordinary path/plane ids or generate extra planes.

## Where To Change Code

| Change you want | Edit here | Also check | Tests |
| --- | --- | --- | --- |
| graph construction | `autoflow/algorithms/graph.py` | `autoflow/core/pipeline.py` | `tests/test_smoke_phantoms.py` |
| compact junction candidates and local routing | `autoflow/algorithms/graph_junctions.py` | `autoflow/core/pipeline.py`, `autoflow/algorithms/hierarchy.py`, `autoflow/algorithms/intracranial.py`, `autoflow/core/models.py` | junction-cluster phantoms in `tests/test_smoke_phantoms.py` |
| single-label cycle cleanup and interface preservation | `autoflow/algorithms/graph_cycles.py` | `autoflow/core/pipeline.py`, `autoflow/algorithms/hierarchy.py`, `autoflow/algorithms/intracranial.py` | same-label loops, cross-label ring and pre-pruning order phantoms in `tests/test_smoke_phantoms.py` |
| branches or paths logic | `autoflow/algorithms/branch.py`, `autoflow/algorithms/paths.py` | `autoflow/core/pipeline.py` | `tests/test_smoke_phantoms.py` |
| topology depth, joins, loops or label-local relationships | `autoflow/algorithms/hierarchy.py` | `autoflow/core/models.py`, `autoflow/quality.py`, `autoflow/ui/app.py`, `autoflow/ui/viewer.py` | `tests/test_smoke_phantoms.py` |
| interactive graph editing | `autoflow/ui/app.py`, `autoflow/ui/editors.py` | `autoflow/ui/viewer.py` | GUI smoke coverage |

## Tests

- `/home/renyuyang/miniconda3/envs/autoflow311/bin/python -m pytest tests/test_smoke_phantoms.py -q`
- Triangle-junction phantoms check the physical centroid, external-edge remapping without another distance search, input and source-label immutability, mixed-label contraction, node/edge permutation, preserved external rings and disconnected components, and conservative handling of closed, overlapping or parallel-connection triangles.
- Junction-cluster phantoms check fixed geometry and existing-edge subsets, outside ports/endpoints, mixed-label interface preservation, retained cross-label cycles, oversized/closed/unknown/low-cycle patches, anisotropic spacing/origin, input permutations, stage order, GUI/CLI/config parameters and workspace/audit persistence.
- Single-label Flow phantoms check strict configurable C thresholds, reverse alignment, missing/zero/masked signal, connectivity enabled/disabled, retained isolated rings, cross-label/unknown interfaces, deterministic geometry under permutation and anisotropic spacing/origin, stage order, and parameter/audit persistence. The special-branch phantom rejects abdominal branch-to-branch contacts while retaining each trunk interface.
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

A dense portal-system knot can be a cluster of adjacent junction voxels rather than one ordinary branch. [Fiji AnalyzeSkeleton](https://imagej.net/plugins/analyze-skeleton/#table-of-results) represents neighboring junction voxels as one actual junction; [VMTK centerlines](https://vmtk.github.io/tutorials/Centerlines.html) offer a separate seed-constrained surface/Voronoi workflow. AutoFlow contracts eligible isolated triangles and then routes eligible compact junction clusters with the documented bounds and label-interface protections. Large, low-cycle, unowned or insufficient-interface patches remain available for review; neither routing nor Flow filtering guarantees removal of every anatomical or discretization loop. Inspect `junction_cluster_cleanup.clusters[]` for each candidate and skip reason.
