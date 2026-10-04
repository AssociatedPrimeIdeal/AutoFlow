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
| `autoflow/algorithms/data.py` | Legacy/normalized H5 parsing, orientation, dual-VENC flow, optional sigma/TKE and correction cache. |
| `autoflow/algorithms/dicom_conversion.py` | Optional pinned Dicom2H5 conversion, calibrated H5 validation, safe publication and source provenance. |
| `autoflow/algorithms/dicom.py` | DICOM scan, case detection/preview, component/geometry decoding and normalized loading. |
| `autoflow/algorithms/phase_correction.py` | MSAC and WRLS+ARTO background phase correction, robust fitting and CUDA fallback. |
| `autoflow/algorithms/noise_removal.py` | Magnitude/temporal-velocity PC-MRA display mask; no quantitative data changes. |
| `autoflow/algorithms/phase_unwrapping.py` | Backend adapters, masks/devices, traditional/learned unwrapping and wrap diagnostics. |
| `autoflow/algorithms/preprocess.py` | Label cleanup, morphology, Gaussian masks and connected-component filtering. |
| `autoflow/algorithms/segmentation.py` | Threshold, static/4D nnUNet input preparation, inference, cleanup and reusable outputs. |
| `autoflow/algorithms/skeleton.py` | Skeleton extraction and special-contact/merge handling. |
| `autoflow/algorithms/graph.py` | Skeleton graph connectivity and graph cleanup. |
| `autoflow/algorithms/branch.py` | Branch ownership, junctions and branch labels. |
| `autoflow/algorithms/paths.py` | Ordered/smoothed centreline paths, tangents and flow-direction interpretation. |
| `autoflow/algorithms/intracranial.py` | Intracranial group topology and Circle of Willis diagnostics. |
| `autoflow/algorithms/planes.py` | Plane placement, centreline spacing, support validation and contour geometry. |
| `autoflow/algorithms/metrics.py` | Plane velocity/flow/area, derived plane samples, WSS, TKE, pressure, vortex and process metrics. |
| `autoflow/algorithms/pwv.py` | PWV group paths/planes, waveform timing, transit fitting and plots. |
| `autoflow/algorithms/streamlines.py` | Instantaneous streamlines, plane seeds, temporal VTK pathlines and bounded frame caches. |
| `autoflow/algorithms/surfaces.py` | VTK grids, velocity/field conversion, segmentation surfaces and connected support meshes. |
| `autoflow/algorithms/traditional/flowunwrap.py` | Bundled traditional phase-recovery algorithms and graph/FFT helpers. |
| `autoflow/algorithms/traditional/fourierOperators.py` | Fourier differentiation, inversion and resampling operators for phase recovery. |
| `autoflow/ui/progress.py` | Application-modal animated progress, input lock and close-to-cancel lifecycle. |
| `autoflow/ui/app.py` | Main Qt window, menus/docks, actions, worker coordination and user workflow. |
| `autoflow/ui/viewer.py` | 3D dataset/actor management, scalar ranges, PC-MRA, geometry/trajectory caches and rendering. |
| `autoflow/ui/ortho_viewer.py` | Orthogonal image/derived-map slices, overlays, content selection and phase display. |
| `autoflow/ui/slice_view.py` | Reusable slice widgets, coordinates and mouse interactions. |
| `autoflow/ui/segmentation.py` | Segmentation dock, run/import/edit/review controls and model configuration. |
| `autoflow/ui/labeler_exchange.py` | SpatioTemporal Labeler feature/mask exchange, content-based reuse and saved-edit import. |
| `autoflow/ui/dicom_confirm.py` | DICOM case preview and acquisition metadata confirmation dialog. |
| `autoflow/ui/editors.py` | Dedicated geometry/editor controls and interactive revision workflows. |
| `autoflow/ui/contour_edit.py` | Cross-section contour editing and local support updates. |
| `autoflow/ui/launcher.py` | GUI startup, environment checks and application entry point. |
| `autoflow/ui/remote_plotter.py` | Remote/off-screen VTK setup and rendering fallbacks. |
| `autoflow/ui/theme.py` | Qt appearance and colour/theme helpers. |
| `autoflow/rendering/jobs.py` | Isolated GUI VTK/FFmpeg task, progress transport and completed-video publication. |
| `autoflow/rendering/videos.py` | Offline geometry/metric/streamline videos, cameras, labels, colourbars and encoding. |

Package `__init__.py` files in `autoflow/`, `algorithms/`, `algorithms/traditional/`, `core/`, `ui/`, and `rendering/` define their package/export boundaries rather than separate algorithm controls.

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
