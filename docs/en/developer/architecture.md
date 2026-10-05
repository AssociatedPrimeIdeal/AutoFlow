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

1. DICOM directories convert through Dicom2H5; the H5 loader returns `LoadedCase`
2. workspace stores normalized source arrays and metadata
3. optional Correction runs background correction → display noise mask → phase unwrap → PC-MRA generation before segmentation; segmentation availability then controls downstream eligibility
4. skeleton feeds graph and paths
5. graph and paths feed planes
6. planes feed metrics
7. segmentation and flow feed derived metrics
8. rendering consumes planes, metrics, and derived arrays
9. quality reporting inspects the current workspace without mutating algorithm results

### Temporal segmentation path

`autoflow/algorithms/segmentation/nnunet_temporal.py:generate_nnunet_4d_auto_segmentation()`
constructs one nnUNet sample per cardiac frame from the Dataset7020 temporal
channels, invokes the checkpoint once, and restores an `XYZT` label volume.
`auto_folds=single` selects one checkpoint; `all` passes every available fold
to nnUNet's ensemble predictor. Topology preprocessing uses a 3D majority
vote, while `Workspace.segmask_binary` remains temporal for phase-wise metrics.

## Metric implementation boundaries

`autoflow/algorithms/metrics/` separates numerical families and plane work by responsibility. `__init__.py` preserves existing `from autoflow.algorithms.metrics import ...` entry points, including the legacy helper imports used by the GUI and pressure phantom suite. Function signatures, numerical formulas, optional-family selection and output schemas are unchanged.

`sampling.py` owns support geometry, ROI selection, representative-mask lookup and slice caches shared by `planes.py`, `plane_derived.py` and `wss.py`. `consistency.py` owns path, label and fork QC. WSS, TKE, pressure and vortex calculations live in their respective modules; `derived.py` combines only the requested families. `export.py` owns pixelwise H5 publication and metric-table loading. `_common.py` normalizes array shapes and `_parallel.py` transports cancellation/progress for isolated plane workers.

Implementation modules import their dependencies directly, without importing the package entry point. This keeps the dependency graph acyclic and leaves geometry caches with one owner. When instrumenting a helper in tests, patch its implementation module; changing a re-export does not replace the helper used inside another module. See [Modules and code ownership](feature-to-code-map.md) for the file-level map.

## Loader and segmentation boundaries

`autoflow/algorithms/inputs.py` handles input dispatch. It sends DICOM directories to the pinned Dicom2H5 adapter and loads the resulting H5 cases; vendor decoding is owned by that dependency. `data/` separates H5 metadata/discovery, canonical array normalization, coordinate conversion, correction-cache IO and dual-VENC decoding. Existing `autoflow.algorithms.data` imports remain available.

`segmentation/` separates label IO and thresholding from nnUNet models, channel preparation, runtime commands and static/temporal/grouped inference. Its package entry point preserves existing segmentation imports. Backend code uses direct implementation imports. Tests patch helpers at the call site in their implementation module rather than replacing package re-exports. Checkout-relative model/runtime paths retain their original roots after the move.

## Phase-unwrapping boundaries

`autoflow/algorithms/phase_unwrapping/__init__.py` preserves the original imports, signatures and result dictionary. `engine.py` validates the request, dispatches components/backends and builds diagnostics. `backends.py` owns method aliases, allowed mask sources, optional dependency checks and CPU/CUDA selection; `_common.py` owns canonical array and learned-weight normalization.

`pudip.py` and `gust.py` adapt their upstream layouts and convert recovered velocity to phase. `laplacian.py`, `nprs.py` and `graphcut.py` own both CPU and Torch implementations of each traditional algorithm. Shared Fourier helpers live in `fourier.py`, PUMA in `_puma.py`, total-field correction in `_common.py`, and the legacy local-gradient method in `brute.py`. `cpu.py` owns `unwrap_data` dispatch and its low-level return contract. All callers use the unified owners; the former standalone traditional package is removed. Each implementation imports its dependencies directly, so the package facade is not part of the internal dependency graph. Patch an implementation helper where it is called when instrumenting manual checks.

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

Long-running compute-oriented GUI pipeline steps are dispatched by `_PipelineTaskWorker` in `autoflow/ui/app.py`. The worker computes each step in a detached copy of mutable workspace containers, sharing read-only numerical inputs. It publishes only completed successful steps. An application-modal progress dialog prevents competing user actions until the thread stops; scene cache invalidation and VTK actor refresh happen on the GUI thread after completion. Interactive editors and live streamlines use their existing scene paths; pathlines have a separate cancellable worker. H5 loading, Dicom2H5 conversion, segmentation preparation and Labeler exchange use the shared background function runner. Case loading retains the current workspace until the replacement load succeeds.

`autoflow/task_control.py` owns thread-local cancellation/progress scopes and task-owned subprocess termination. `autoflow/ui/progress.py` owns application modality, input filtering, animated activity, elapsed/update age, and the close-to-cancel lifecycle. A progress window is dismissed only after worker termination. × requests cancellation; Escape does not dismiss the task lock. Native calculations and file writes stop at safe boundaries.

GUI video preparation uses detached workspace state; `autoflow/rendering/jobs.py` launches an isolated renderer with its own VTK context and streamed frame progress. The child renders into a private directory and the parent publishes completed files after success. CLI/API renderers share the replayable frame generators and atomic MP4 encoder.

Segmentation updates call `PipelineEngine.refresh_segmentation_dependents()`: retain all artifacts only for identical processed 4D masks; retain geometry but invalidate all numerical families/trajectories for temporal-only changes with identical processed 3D topology and configuration; otherwise reset segmentation dependents.

## Correction state and reruns

`PipelineEngine.load_data` remembers the original selected input and disables new correction fitting during load. GUI passes `reuse_existing_corr=True`, so the loader applies existing structurally valid fields read-only; missing or invalid caches leave their encoding unchanged. This mode uses cache metadata rather than assigning the current UI fitting settings to old fields. GUI Background Correction passes `force_recompute_corr=True` to `run_step`; it clears cache-only loading from remembered kwargs and rereads the original source/group with current fitting parameters, preserving true complex and dual-VENC processing and avoiding repeated subtraction. It replaces working correction fields and, when enabled, their H5 caches. Normal CLI/API steps keep their configured cache-reuse policy. It updates working velocity and the unwrap baseline without resetting downstream state. Unwrap always starts from stored wrapped phase and preserves working velocity outside the selected mask. Noise Removal writes only `pcmra_render_mask`, read by the 3D PC-MRA renderer. GUI refreshes affected display layers without invoking segmentation or metric recomputation. CLI `process_single` executes the group before segmentation; standalone unavailable-segmask unwrap can be deferred until automatic segmentation.

`Workspace.pcmra_array` is an explicitly generated `XYZT` snapshot of magnitude times working speed. It is unset on input load, computed frame by frame by `StepId.GENERATE_PCMRA`, saved/restored in workspaces and consumed by both 3D rendering and the PC-MRA Content entry. Noise Removal adds a `noise_region` review surface from the excluded mask. `OrthoViewer` filters Content by available arrays/results and stores stable field IDs in combo item data; never interpret row positions as field identities.
