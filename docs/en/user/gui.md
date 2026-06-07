# GUI Guide

## Status
`autoflow-gui` is supported for interactive loading, segmentation review, grouped vessel processing, step execution, and visualization.

Entry files:

- `pyproject.toml`
- `autoflow/ui/launcher.py`
- `autoflow/ui/app.py`

## Start The GUI

```bash
autoflow-gui
```

Or load defaults from a custom config directory:

```bash
autoflow-gui --config-dir ./configs
```

## Main Layout

The main window contains:

- menu bar
- left browser
- central 3D view
- steps area
- parameter panels
- right ortho viewer
- bottom timeline
- bottom selection panel
- bottom log
- right segmentation dock

The left browser is group-aware. When the loaded segmentation produces multiple vessel groups, the browser shows one top-level section per group so you can toggle an entire group at once.

## File Menu

| Menu item | What it does | Main code |
| --- | --- | --- |
| `Open H5` | open an H5 or HDF5 case | `autoflow/ui/app.py` |
| `Import DICOM Directory` | scan a DICOM directory and choose a case | `autoflow/ui/app.py`, `autoflow/ui/dicom_confirm.py` |
| `Clear Workspace` | clear loaded data and restore config defaults in the UI | `autoflow/ui/app.py` |
| `Exit` | close the GUI | `autoflow/ui/app.py` |

## Standard Workflow

1. start the GUI
2. load a case through `Open H5` or `Import DICOM Directory`
3. for DICOM, confirm or edit resolution, venc, spatial order, venc order, and RR
4. check loader parameters in `Input / Background Correction`, including dual-venc ratios for legacy `Nv=7` H5 when needed
5. if no segmentation exists, use the segmentation menu or dock
6. if the segmentation is a label mask, AutoFlow will reduce 4D labels to 3D by time majority vote, remove small connected components per label, merge labels by configured groups, and then run grouped skeleton, graph, and plane generation
7. run steps individually or click `Run All`
8. inspect group sections in the browser, 3D view, ortho viewer, and selection panel
9. right-click an individual pathline if you want to change only that pathline color
10. save the active segmentation if you want to reuse it later

## Step Buttons

| Step | Purpose | Notes |
| --- | --- | --- |
| `Generate Skeleton` | skeletonize the active segmentation | binary masks run as one group; label masks run per configured group |
| `Generate Graph` | build graph, branches, and paths | depends on skeleton and preserves group names |
| `Generate Planes` | create center or distance-based planes | depends on graph and keeps grouped path ownership |
| `Calculate && Save Metrics` | compute plane metrics and save outputs | requires segmentation and flow |
| `WSS / TKE / Pressure Gradient` | compute derived volumes and refresh scene objects | TKE stays optional |
| `Generate Streamlines` | enable live streamlines from the combined segmentation | requires segmentation |
| `Pathlines` | launch time-resolved pathlines from planes | grouped plane names are preserved |
| `Edit Skeleton` | interactive skeleton correction | available only when exactly one segmentation group is active |
| `Edit Graph` | interactive graph correction | available only when exactly one segmentation group is active |
| `Run All` | run the standard order | does not create segmentation automatically |

Current `Run All` order:

1. `Generate Skeleton`
2. `Generate Graph`
3. `Generate Planes`
4. `Calculate && Save Metrics`
5. `WSS / TKE / Pressure Gradient`

## Parameter Panels

| Panel | Main purpose | Main code |
| --- | --- | --- |
| `Input / Background Correction` | loader and DICOM settings | `autoflow/ui/app.py`, `autoflow/config.py` |
| `Generate Skeleton Parameters` | cleanup and morphology controls; grouped label maps and group colors are loaded from `configs/skeleton.json` | `autoflow/ui/app.py`, `autoflow/config.py` |
| `Generate Planes Parameters` | center-plane or distance-plane generation | `autoflow/ui/app.py` |
| `Streamline Parameters` | seed density, steps, terminal speed, colors | `autoflow/ui/app.py` |
| `WSS Parameters` | WSS-specific controls | `autoflow/ui/app.py` |
| `Flow / TKE / Pressure Gradient Parameters` | derived metrics controls | `autoflow/ui/app.py` |

## Segmentation Workflow

The GUI segmentation system supports:

- original segmentation
- imported segmentation
- threshold segmentation
- automatic segmentation through `nnUNet`
- manual editing in the segmentation dock

Important behavior:

- `Run All` does not auto-start segmentation
- segmentation must already exist when segmentation-dependent steps run
- auto segmentation can be launched through `Configure Segmentation...`
- GUI auto segmentation opens a modal progress dialog and keeps the main case state locked until inference and sidecar save complete
- automatic and threshold segmentations save sidecar H5 files after a successful run

Grouped label-mask behavior:

- binary masks and single-label masks are treated as one group
- 4D label masks are reduced to 3D labels by majority vote along time before vessel steps run
- connected-component cleanup runs per label value before labels are merged into groups
- label grouping, browser colors, and per-group preprocessing come from `configs/skeleton.json`
- each group can apply its own Gaussian smoothing and morphology overrides before skeletonization

## Selection, Browser, And Timeline

- the browser creates one top-level row per segmentation group and one `Global` row for non-grouped scene objects
- checking or unchecking a group row shows or hides every object in that group
- group title colors use `label_groups.<group>.browser_color` from `configs/skeleton.json`, with fallback colors for unmatched or single-label cases
- grouped objects use names such as `segmask_group_aorta_systemic_branches`, `skeleton_aorta_systemic_branches`, `graph_aorta_systemic_branches`, `smooth_path_aorta_systemic_branches_3`, `plane_aorta_systemic_branches_5`, and `pathline_aorta_systemic_branches_5`
- right-clicking an individual grouped pathline in the browser opens `Set Pathline Color`, which changes only that pathline
- selecting a plane updates selection info and the ortho viewer
- selecting a path shows path-level information
- the timeline controls time-resolved scene objects and ortho slices

## Ortho Viewer

The ortho viewer supports:

- flow components
- magnitude
- PC-MRA
- speed
- WSS
- TKE
- pressure-gradient components and magnitude
- segmentation painting while editing is enabled
- jump-to-plane-center behavior

## When To Use The GUI

Use the GUI when you need:

- interactive segmentation review or correction
- grouped visibility control for multi-label vessel workflows
- skeleton or graph editing in single-group cases
- plane dragging and immediate metric recomputation
- time navigation through 3D and ortho views

Use the CLI when you need:

- unattended batch processing
- repeated runs over many cases
- offline video export pipelines
