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
- right analysis dock

The left browser is group-aware. When the loaded segmentation produces multiple vessel groups, the browser shows one top-level section per group, then type sections such as `Paths`, `Planes`, and `Pathlines`, so you can toggle a whole group or one object class at once.

## File Menu

| Menu item | What it does | Main code |
| --- | --- | --- |
| `Open H5` | open an H5 or HDF5 case | `autoflow/ui/app.py` |
| `Import DICOM Directory` | scan a DICOM directory and choose a case | `autoflow/ui/app.py`, `autoflow/ui/dicom_confirm.py` |
| `Clear Workspace` | clear loaded data and restore config defaults in the UI | `autoflow/ui/app.py` |
| `Exit` | close the GUI | `autoflow/ui/app.py` |

## Export Menu

| Menu item | What it does | Main code |
| --- | --- | --- |
| `Export Videos...` | choose an output directory and export selected plane, WSS, TKE, pressure-gradient, or streamline videos | `autoflow/ui/app.py`, `autoflow/rendering/videos.py` |

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
10. use `Export > Export Videos...` when you want selected offline MP4 exports
11. save the active segmentation if you want to reuse it later

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
5. `Compute PWV`
6. `WSS / TKE / Pressure Gradient`

## Parameter Panels

| Panel | Main purpose | Main code |
| --- | --- | --- |
| `Input / Background Correction` | loader and DICOM settings | `autoflow/ui/app.py`, `autoflow/config.py` |
| `Generate Skeleton Parameters` | cleanup and morphology controls; grouped label maps and group colors are loaded from `configs/labels.json` | `autoflow/ui/app.py`, `autoflow/config.py` |
| `Generate Planes Parameters` | center-plane or distance-plane generation | `autoflow/ui/app.py` |
| `PWV Parameters` | enable PWV, edit groups, waveform selection, spacing, and plot styling | `autoflow/ui/app.py` |
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
- each grouped vessel mask is reduced to its largest connected component before skeletonization
- label grouping, browser colors, and per-group preprocessing come from `configs/labels.json`
- each group can apply its own Gaussian smoothing and morphology overrides before skeletonization

PWV behavior:

- the `PWV Parameters` panel exposes the same core settings as `configs/pwv.json`, including `enabled`, groups, waveform key, spacing, transit-time method, foot definition, cross-correlation window, cross-correlation interpolation factor, cycle-wrap correction, smoothing, minimum valid planes, and plot colors. The default foot-to-foot waveform is `flowrate_mL_s`
- when `Allow Cycle Wrap` is enabled, foot detection can follow an upstroke that starts at the end of the cardiac cycle and peaks in the first frames instead of pinning those planes to `0 ms`
- group labels can be entered as symbolic names from `configs/labels.json` such as `AAO, ARCH, DAO` or as integer ids
- step buttons use the current GUI PWV values immediately after `Load`, `Clear Workspace`, or manual edits; the values do not write back to `configs/pwv.json` automatically
- the GUI shows one `Analysis` dock with a `Display` selector. `PWV` keeps the group selector and two-panel PWV plot. `Plane Curve` shows the selected plane's chosen metric across cardiac phase. `Internal Consistency` shows the selected path or plane's path-level and branch-level internal consistency
- PWV planes appear in the browser as one grouped item named `PWV planes`; the GUI does not list every PWV plane separately


## Video Export

Use `Export > Export Videos...` for interactive offline export.

- choose an output directory
- select any combination of `plane`, `wss`, `tke`, `pg`, and `streamlines`
- `plane` is checked by default
- WSS, TKE, and pressure-gradient exports compute missing derived data before rendering
- GUI video export reads shared camera and sizing defaults from `configs/rendering.json`
- GUI live planes read `configs/planes.json -> render`
- GUI live WSS, TKE, pressure-gradient, and streamline objects read the corresponding `*.json -> render` block, including optional colorbar visibility
- if `summary.json` already exists in the output directory, the GUI updates `videos`, `video_times_sec`, and adds `gui_video_export`

`Run All` does not export videos automatically. Video export is a separate top-level menu action.

## Selection, Browser, And Timeline

- the browser creates one top-level row per segmentation group and one `Global` row for non-grouped scene objects
- checking or unchecking a group row shows or hides every object in that group, and checking or unchecking a type row such as `Planes` or `Pathlines` controls that whole class inside the group
- group title colors use `label_groups.<group>.browser_color` from `configs/labels.json`, with fallback colors for unmatched or single-label cases
- grouped objects keep stable internal data keys such as `smooth_path_aorta_systemic_branches_3`, `plane_aorta_systemic_branches_5`, and `pathline_aorta_systemic_branches_5`
- browser-visible names are intentionally shorter, for example `path 3`, `plane 5`, and `pathline 5`
- right-clicking an individual grouped pathline in the browser opens `Set Pathline Color`, which changes only that pathline
- selecting a plane updates selection info, the ortho viewer, and the `Analysis` plane/path views
- selecting a path shows path-level information and updates `Analysis -> Internal Consistency`
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
- interactive selection of which offline videos to export

Use the CLI when you need:

- unattended batch processing
- repeated runs over many cases
- unattended or repeated offline video export pipelines
