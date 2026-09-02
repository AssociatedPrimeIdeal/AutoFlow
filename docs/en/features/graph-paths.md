# Feature: Graph And Paths

## Status

| Entry point | Status | Notes |
| --- | --- | --- |
| GUI | Supported | includes interactive graph editing for single-group cases |
| CLI | Supported | generated in batch order |
| Python API | Supported | through pipeline execution |

## What It Does
Graph and path generation converts the skeleton into nodes, edges, branches, forks, and path objects used by planes and downstream metrics.
When segmentation is grouped, graph generation runs per group, then combines path endpoints, fork nodes, fork roles, and path indices into one workspace-wide numbering while preserving each path `group_name`.

GUI edits now invalidate dependent artifacts explicitly: skeleton edits clear
graph, paths, planes, and metrics; graph edits clear paths, planes, and metrics.
Run the next centerline action after finishing an edit.

## When To Use It
- use it after a valid skeleton exists
- use it before plane generation
- use GUI editing when automatic graph structure needs correction

## Quick Use

### GUI
1. run `Generate Skeleton`
2. click `Generate Graph`
3. inspect grouped graph, fork, and path objects in the browser and 3D view
4. if needed, click `Edit Graph` when exactly one segmentation group is active

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

## Inputs

| Input | Required | Meaning |
| --- | --- | --- |
| skeleton | yes | centerline input for graph construction |
| segmentation | indirectly yes | needed to build the skeleton first |

## Parameters

| Parameter | Type | Default | Where set | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| graph step trigger | step | fixed | GUI step or batch order | build nodes, edges, branches, and paths | `autoflow/core/pipeline.py` |
| edit mode | GUI mode | off | GUI only | drag nodes, toggle edges, delete nodes or edges | `autoflow/ui/app.py` |

## Outputs

| Output file or object | Created when | Meaning |
| --- | --- | --- |
| graph data in workspace | graph step succeeds | nodes and edges |
| branches and paths in workspace | graph step succeeds | branch and path topology; nearest-path labels are assigned in one vectorized operation, and grouped endpoints, fork nodes, and fork roles use global ids |
| graph scene objects | GUI graph step succeeds | grouped graph objects such as `graph_aorta_systemic_branches` and `smooth_path_aorta_systemic_branches_3` |

## Limitations
- depends on usable skeleton quality
- there is no dedicated public CLI parameter family just for graph tuning in the current docs set
- interactive graph editing is only available when exactly one segmentation group is active

## Where To Change Code

| Change you want | Edit here | Also check | Tests |
| --- | --- | --- | --- |
| graph construction | `autoflow/algorithms/graph.py` | `autoflow/core/pipeline.py` | `tests/test_smoke_phantoms.py` |
| branches or paths logic | `autoflow/algorithms/branch.py`, `autoflow/algorithms/paths.py` | `autoflow/core/pipeline.py` | `tests/test_smoke_phantoms.py` |
| interactive graph editing | `autoflow/ui/app.py`, `autoflow/ui/editors.py` | `autoflow/ui/viewer.py` | GUI smoke coverage |

## Tests

- `~/miniconda3/envs/ryy/bin/python -m pytest tests/test_smoke_phantoms.py -q`

## Common Problems

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| graph step is skipped | no skeleton | run `Generate Skeleton` first |
| graph and fork objects are listed but absent from the 3D view while paths remain visible | an older viewer did not resolve grouped `graph_<group>` and `forks_<group>` data keys | install the current editable build and run `Generate Graph` again |
| path structure looks wrong | skeleton topology is noisy | improve segmentation or edit graph manually |
| `Edit Graph` is unavailable | more than one segmentation group is active | use a single-group case, or change group configuration |
