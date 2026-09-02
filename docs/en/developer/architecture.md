# Architecture

## Purpose
This page explains where the major runtime responsibilities live so maintainers can find the right layer quickly.

## Main Layers

| Layer | Directory | Responsibility |
| --- | --- | --- |
| public entry points | `autoflow/` | CLI, GUI launcher, public Python API |
| pipeline orchestration | `autoflow/core` | workspace state and step orchestration |
| algorithms | `autoflow/algorithms` | loading, preprocessing, graph, planes, metrics, segmentation, streamlines |
| GUI | `autoflow/ui` | PySide6 window, docks, viewers, dialogs, and editors; pyqtgraph owns interactive orthogonal slices |
| rendering | `autoflow/rendering` | offline video export |
| reporting and portable geometry | `autoflow/quality.py`, `autoflow/plane_io.py`, `autoflow/reporting.py` | staged QC, cross-case plane coordinates, and human-readable summaries |
| tests | `tests` | behavior coverage |
| configuration | `configs`, `autoflow/config.py` | per-module JSON defaults and loading |

## Primary Execution Paths

### CLI batch path
1. `autoflow/cli.py` parses flags
2. `autoflow/api.py` builds `AutoFlowConfig`
3. `autoflow/processing.py:process_single()` runs the batch order
4. `autoflow/core/pipeline.py` executes concrete steps

### GUI path
1. `autoflow/ui/launcher.py` starts the app
2. `autoflow/ui/app.py` manages the main window and UI state
3. `autoflow/core/pipeline.py` runs steps against the workspace
4. `autoflow/ui/viewer.py` renders 3D state through the local Qt VTK widget or the forwarded-X11 EGL adapter in `autoflow/ui/remote_plotter.py`; `autoflow/ui/ortho_viewer.py` and `autoflow/ui/slice_view.py` render interactive 2D state
5. `autoflow/ui/app.py` exports Labeler exchange features, launches the optional SpatioTemporal Labeler process, and imports a saved label sequence after the process exits

### Python API path
1. `autoflow/api.py` exposes `AutoFlowConfig`, `run_case()`, `run_batch()`, and `build_workspace()`
2. config defaults come from `autoflow/config.py`
3. execution still flows through `autoflow/processing.py` and `autoflow/core/pipeline.py`

## Core Data Flow

1. loader returns `LoadedCase`
2. workspace stores normalized source arrays and metadata
3. segmentation availability controls downstream step eligibility
4. skeleton feeds graph and paths
5. graph and paths feed planes
6. planes feed metrics
7. segmentation and flow feed derived metrics
8. rendering consumes planes, metrics, and derived arrays
9. quality reporting inspects the current workspace without mutating algorithm results

### Temporal segmentation path

`autoflow/algorithms/segmentation.py:generate_nnunet_4d_auto_segmentation()`
constructs one nnUNet sample per cardiac frame from the Dataset7020 temporal
channels, invokes the checkpoint once, and restores an `XYZT` label volume.
`auto_folds=single` selects one checkpoint; `all` passes every available fold
to nnUNet's ensemble predictor. Topology preprocessing uses a 3D majority
vote, while `Workspace.segmask_binary` remains temporal for phase-wise metrics.

## Design Constraints To Keep
- `LoadedCase` normalization is the contract between loaders and the rest of the system
- downstream arrays use spatial and vector-component order `LR/AP/FH`; source-order metadata is audit information, while 3D and 2D viewers must label the normalized array order
- workspace centerlines and planes use local physical millimetres; array lookup divides by spacing without subtracting origin, while VTK/world export adds origin exactly once
- TKE stays optional
- segmentation is an interface with multiple valid sources
- external segmentation edits must be imported through the existing segmentation-source invalidation path
- GUI and CLI share pipeline logic as much as possible

## Derived Artifact Cache

`DerivedResults.artifact_signatures` owns independent `wss`, `tke`, and `pressure` signatures. `PipelineEngine._ensure_derived_metrics()` must compare the current signature before reusing any derived array. New parameters or algorithm changes must be added to the affected signature payload and its algorithm-version token. Do not use array presence alone as cache validity.

Plane metrics cache thresholded support geometry per unique mask phase and slice specifications per plane and representative phase. WSS caches the extracted and smoothed base surface per unique mask phase, then copies that geometry before attaching phase-specific arrays. Geometry caches must never share mutable phase-specific scalar data.

Long-running compute-oriented GUI pipeline steps are dispatched by `_PipelineTaskWorker` in `autoflow/ui/app.py`. The worker mutates the active workspace while a window-modal progress dialog prevents competing user actions; scene cache invalidation and VTK actor refresh happen on the GUI thread after completion. Interactive editors and live streamline/pathline actions are deliberately excluded from this worker path.
