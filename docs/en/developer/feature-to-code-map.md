# Modules and code ownership

Find a feature owner here, then use its feature page for algorithms, inputs, outputs, parameters and regression/manual checks. Internal helpers are documented at their owning module; the supported configurable/public parameters are exhaustively listed in the [parameter references](../user/parameters.md).

## Python modules

| Module | Responsibility |
| --- | --- |
| `autoflow/api.py` | Public configuration dataclass, workspace construction, case/batch API and GUI launcher. |
| `autoflow/cli.py` | Argument parser, explicit CLI overrides and batch command entry. |
| `autoflow/config.py` | Per-module defaults, recursive merging, compatibility and API/workspace/render mappings. |
| `autoflow/case_types.py` | Normalized LoadedCase/InputCase contracts, capabilities and correction/DICOM metadata types. |
| `autoflow/processing.py` | Case execution order, optional family selection, timing, batch discovery and exports. |
| `autoflow/nnunet_runtime.py` | Inference child-runtime helpers and guarded CPU/GPU transfer/copy optimizations. |
| `autoflow/quality.py` | Staged QC records, path hierarchy, branch conservation and user-facing review statistics. |
| `autoflow/reporting.py` | Summary and metric reporting/serialization. |
| `autoflow/plane_io.py` | Plane coordinate formats, import mapping, compatibility and export records. |
| `autoflow/io_utils.py` | Reusable input/output serialization helpers. |
| `autoflow/task_control.py` | Cooperative task cancellation, thread-local progress and task-owned subprocess cleanup. |
| `autoflow/utils.py` | Shared numerical/coordinate utility functions. |
| `autoflow/core/models.py` | Workspace state, parameters, segmentation versions, planes, scene objects and saved-state contracts. |
| `autoflow/core/pipeline.py` | Step orchestration, geometry/metrics dispatch, dependencies, derived signatures and scene registration. |
| `autoflow/algorithms/data/__init__.py` | Compatible H5 loader, normalization and orientation entry points. |
| `autoflow/algorithms/data/h5_loader.py` | H5 layout dispatch and normalized case loading. |
| `autoflow/algorithms/data/orientation.py` | Spatial-axis and vector-component direction normalization. |
| `autoflow/algorithms/data/normalization.py` | LoadedCase array layouts, time broadcasting and optional sigma/TKE fields. |
| `autoflow/algorithms/data/h5_metadata.py` | Key lookup, metadata coercion and supported acquisition groups. |
| `autoflow/algorithms/data/discovery.py` | H5 case discovery and embedded-capability inspection. |
| `autoflow/algorithms/data/correction_cache.py` | Background-phase cache validation/publication and loader progress. |
| `autoflow/algorithms/data/venc.py` | VENC parsing and dual-VENC alias helpers. |
| `autoflow/algorithms/data/dual_venc.py` | Legacy complex dual-VENC loading and concurrent correction. |
| `autoflow/algorithms/dicom_conversion.py` | Optional pinned Dicom2H5 conversion, calibrated H5 validation, safe publication and source provenance. |
| `autoflow/algorithms/inputs.py` | H5 case discovery and DICOM directory dispatch through Dicom2H5; all loading uses H5. |
| `autoflow/algorithms/phase_correction.py` | MSAC and WRLS+ARTO background phase correction, robust fitting and CUDA fallback. |
| `autoflow/algorithms/noise_removal.py` | Magnitude/temporal-velocity PC-MRA display mask; no quantitative data changes. |
| `autoflow/algorithms/phase_unwrapping/__init__.py` | Compatible phase-unwrapping entry points and legacy helper exports. |
| `autoflow/algorithms/phase_unwrapping/backends.py` | Method/mask policies, optional dependency checks and device selection. |
| `autoflow/algorithms/phase_unwrapping/_common.py` | Phase/VENC validation, mask weights, learned parameter extraction and total-field correction. |
| `autoflow/algorithms/phase_unwrapping/engine.py` | Backend/component dispatch, phase-to-velocity conversion and wrap diagnostics. |
| `autoflow/algorithms/phase_unwrapping/pudip.py` | PUDIP-Flow parameters and array-layout/velocity adapter. |
| `autoflow/algorithms/phase_unwrapping/gust.py` | GUST-Flow confidence input, layout adapter and CUDA failure diagnostics. |
| `autoflow/algorithms/phase_unwrapping/laplacian.py` | CPU 3D/4D and Torch 4D Laplacian phase recovery. |
| `autoflow/algorithms/phase_unwrapping/nprs.py` | CPU/Torch Fourier resampling with the original NPRS skimage unwrap core. |
| `autoflow/algorithms/phase_unwrapping/graphcut.py` | CPU/Torch masked graph discovery and 3D/4D PUMA phase recovery. |
| `autoflow/algorithms/phase_unwrapping/_puma.py` | PUMA energy minimization with optional PyMaxflow. |
| `autoflow/algorithms/phase_unwrapping/fourier.py` | Single copies of the shared Fourier phase-shift and pad/crop helpers. |
| `autoflow/algorithms/phase_unwrapping/brute.py` | Legacy discontinuity/local-gradient recovery. |
| `autoflow/algorithms/phase_unwrapping/cpu.py` | Low-level CPU method dispatch and wrap-count/velocity output. |
| `autoflow/algorithms/preprocess.py` | Label cleanup, morphology, Gaussian masks and connected-component filtering. |
| `autoflow/algorithms/segmentation/__init__.py` | Compatible segmentation source and backend entry points. |
| `autoflow/algorithms/segmentation/_common.py` | Label-volume normalization, broadcasting and timestamps. |
| `autoflow/algorithms/segmentation/io.py` | Segmentation import/export, NIfTI features and reusable H5 masks. |
| `autoflow/algorithms/segmentation/threshold.py` | Reference scalar, automatic threshold selection and cleanup. |
| `autoflow/algorithms/segmentation/channels.py` | Static/temporal nnUNet channels and coordinate preparation. |
| `autoflow/algorithms/segmentation/models.py` | Model profiles/paths, folds, checkpoints and pipeline-script metadata. |
| `autoflow/algorithms/segmentation/runtime.py` | Cancellable subprocesses, device selection and GPU preprocessing. |
| `autoflow/algorithms/segmentation/nnunet_static.py` | Static inference and segmentation-backend dispatch. |
| `autoflow/algorithms/segmentation/nnunet_temporal.py` | Phase-resolved nnUNet inference and restored 4D labels. |
| `autoflow/algorithms/segmentation/nnunet_grouped.py` | Optional grouped Dataset7020 preparation and inference adapter. |
| `autoflow/algorithms/skeleton.py` | Skeleton extraction and special-contact/merge handling. |
| `autoflow/algorithms/graph.py` | Skeleton graph connectivity and graph cleanup. |
| `autoflow/algorithms/branch.py` | Branch ownership, junctions and branch labels. |
| `autoflow/algorithms/paths.py` | Ordered/smoothed centreline paths, tangents and flow-direction interpretation. |
| `autoflow/algorithms/intracranial.py` | Intracranial group topology and Circle of Willis diagnostics. |
| `autoflow/algorithms/planes.py` | Plane placement, centreline spacing, support validation and contour geometry. |
| `autoflow/algorithms/metrics/__init__.py` | Compatible metric imports; re-exports the existing entry points and legacy helpers. |
| `autoflow/algorithms/metrics/sampling.py` | Shared segmentation support meshes, phase/ROI-aware plane geometry and field sampling. |
| `autoflow/algorithms/metrics/planes.py` | Plane flow/area/velocity integration and isolated-process dispatch. |
| `autoflow/algorithms/metrics/consistency.py` | Path, segmentation-label and fork internal-consistency summaries. |
| `autoflow/algorithms/metrics/plane_derived.py` | Per-plane derived summaries, pixelwise sample payloads and isolated-process sampling. |
| `autoflow/algorithms/metrics/wss.py` | Tangential wall-velocity derivatives, WSS surfaces and volume maps. |
| `autoflow/algorithms/metrics/tke.py` | Optional sigma-derived TKE arrays and summaries. |
| `autoflow/algorithms/metrics/pressure.py` | Pressure gradients, Cartesian/Stokes reconstruction and centreline profiles. |
| `autoflow/algorithms/metrics/vortex.py` | Vorticity, Q criterion and swirling strength. |
| `autoflow/algorithms/metrics/derived.py` | Optional-family selection and combined derived result payloads. |
| `autoflow/algorithms/metrics/export.py` | Atomic plane pixelwise H5 export and metric-table loading. |
| `autoflow/algorithms/metrics/_common.py` | Shared mask/velocity shape normalization. |
| `autoflow/algorithms/metrics/_parallel.py` | Plane worker cancellation markers and parent-thread progress collection. |
| `autoflow/algorithms/pwv.py` | PWV group paths/planes, waveform timing, transit fitting and plots. |
| `autoflow/algorithms/streamlines.py` | Instantaneous streamlines, plane seeds, temporal VTK pathlines and bounded frame caches. |
| `autoflow/algorithms/surfaces.py` | VTK grids, velocity/field conversion, segmentation surfaces and connected support meshes. |
| `autoflow/ui/progress.py` | Application-modal animated progress, input lock and close-to-cancel lifecycle. |
| `autoflow/ui/app.py` | Main Qt window, menus/docks, actions, worker coordination and user workflow. |
| `autoflow/ui/viewer.py` | 3D dataset/actor management, scalar ranges, PC-MRA, geometry/trajectory caches and rendering. |
| `autoflow/ui/ortho_viewer.py` | Orthogonal image/derived-map slices, overlays, content selection and phase display. |
| `autoflow/ui/slice_view.py` | Reusable slice widgets, coordinates and mouse interactions. |
| `autoflow/ui/segmentation.py` | Segmentation dock, run/import/edit/review controls and model configuration. |
| `autoflow/ui/labeler_exchange.py` | SpatioTemporal Labeler feature/mask exchange, content-based reuse and saved-edit import. |
| `autoflow/ui/input_dialogs.py` | H5 group and dual-VENC source selection dialogs. |
| `autoflow/ui/editors.py` | Dedicated geometry/editor controls and interactive revision workflows. |
| `autoflow/ui/contour_edit.py` | Cross-section contour editing and local support updates. |
| `autoflow/ui/launcher.py` | GUI startup, environment checks and application entry point. |
| `autoflow/ui/remote_plotter.py` | Remote/off-screen VTK setup and rendering fallbacks. |
| `autoflow/ui/theme.py` | Qt appearance and colour/theme helpers. |
| `autoflow/rendering/jobs.py` | Isolated GUI VTK/FFmpeg task, progress transport and completed-video publication. |
| `autoflow/rendering/videos.py` | Offline geometry/metric/streamline videos, cameras, labels, colourbars and encoding. |

Package `__init__.py` files in `autoflow/`, `algorithms/`, `algorithms/metrics/`, `algorithms/data/`, `algorithms/segmentation/`, `algorithms/phase_unwrapping/`, `core/`, `ui/`, and `rendering/` define their package/export boundaries rather than separate algorithm controls.

## Configuration modules

| Module | Configuration and explanations | Main consumer |
| --- | --- | --- |
| Batch execution | [`batch.json`](../user/parameters.md#batch) | `autoflow/config.py` mapping and the owner listed in its parameter table |
| 3D display | [`ui.json`](../user/parameters.md#ui) | `autoflow/config.py` mapping and the owner listed in its parameter table |
| Inputs/correction | [`loader.json`](../user/parameters.md#loader) | `autoflow/config.py` mapping and the owner listed in its parameter table |
| PC-MRA noise mask | [`noise_removal.json`](../user/parameters.md#noise_removal) | `autoflow/algorithms/noise_removal.py`, `autoflow/ui/viewer.py` |
| Unwrapping | [`phase_unwrapping.json`](../user/parameters.md#phase_unwrapping) | `autoflow/config.py` mapping and the owner listed in its parameter table |
| Skeleton preprocessing | [`skeleton.json`](../user/parameters.md#skeleton) | `autoflow/config.py` mapping and the owner listed in its parameter table |
| Labels/groups | [`labels.json`](../user/parameters.md#labels) | `autoflow/config.py` mapping and the owner listed in its parameter table |
| Plane placement | [`planes.json`](../user/parameters.md#planes) | `autoflow/config.py` mapping and the owner listed in its parameter table |
| Streamlines | [`streamlines.json`](../user/parameters.md#streamlines) | `autoflow/config.py` mapping and the owner listed in its parameter table |
| Pathlines | [`pathlines.json`](../user/parameters.md#pathlines) | `autoflow/config.py` mapping and the owner listed in its parameter table |
| Legacy compatibility | [`derived.json`](../user/parameters.md#derived) | `autoflow/config.py` mapping and the owner listed in its parameter table |
| Fluid units | [`fluid.json`](../user/parameters.md#fluid) | `autoflow/config.py` mapping and the owner listed in its parameter table |
| WSS | [`wss.json`](../user/parameters.md#wss) | `autoflow/config.py` mapping and the owner listed in its parameter table |
| TKE | [`tke.json`](../user/parameters.md#tke) | `autoflow/config.py` mapping and the owner listed in its parameter table |
| Pressure | [`pressure_gradient.json`](../user/parameters.md#pressure_gradient) | `autoflow/config.py` mapping and the owner listed in its parameter table |
| Vortex | [`vortex.json`](../user/parameters.md#vortex) | `autoflow/config.py` mapping and the owner listed in its parameter table |
| PWV | [`pwv.json`](../user/parameters.md#pwv) | `autoflow/config.py` mapping and the owner listed in its parameter table |
| Segmentation | [`segmentation.json`](../user/parameters.md#segmentation) | `autoflow/config.py` mapping and the owner listed in its parameter table |
| Colourbars | [`colorbar.json`](../user/parameters.md#colorbar) | `autoflow/config.py` mapping and the owner listed in its parameter table |
| Videos | [`video_exporting.json`](../user/parameters.md#video_exporting) | `autoflow/config.py` mapping and the owner listed in its parameter table |

## Changing behaviour

Follow [Change recipes](change-recipes.md). Edit the owning algorithm and pipeline wiring, update that feature's explanation and parameter metadata, and run the retained [smoke/phantom coverage](testing.md). GUI interaction and performance use manual checks. Do not create a targeted pytest file per module.
