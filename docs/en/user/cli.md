# CLI Guide

## Status
`autoflow-run` is the main batch entry point. It is supported.

Entry files:

- `pyproject.toml`
- `autoflow/cli.py`
- `autoflow/api.py`
- `autoflow/processing.py`

## Task-First Usage

### Run one H5 case

```bash
autoflow-run ./data/demo_data.h5 --output-dir ./results/demo
```

### Run a DICOM root

```bash
autoflow-run /path/to/dicom_root --output-dir ./results/dicom_batch
```

### Import and export plane coordinates

```bash
autoflow-run ./data/demo_data.h5 \
  --output-dir ./results/demo \
  --import-planes ./results/old_case/plane_positions.json \
  --plane-import-mode world \
  --export-planes ./results/transferred_planes.json
```

Use `world` only for cases registered in the same AutoFlow canonical world-mm frame. Use `local` for identical cropped geometry with different origins. Use `path_relative` for unregistered cases whose generated path groups and path ranks correspond; it maps each plane by fractional centerline distance and uses the target path tangent. `--reuse-planes` remains an alias for `--import-planes`.

### Enable auto segmentation for cases without segmentation

```bash
autoflow-run ./data/demo_data.h5 \
  --output-dir ./results/demo \
  --autoseg
```

### Batch auto segmentation, streamline videos, and segmentation NIfTI packaging

`segonly.sh` processes every H5/HDF5 file below `ROOT`, generates or refreshes
the automatic segmentation, and exports a streamline video for each case. It
passes `--skip-derived` and `--skip-plane-metrics` so WSS, TKE, pressure, and
plane metrics are excluded; skeleton, graph, and plane generation remain in the
current batch pipeline before video export. The per-case `summary.json` records
the rendered video as `videos.streamlines`, and `_run_status.tsv` includes the
same path in its `streamline_video` column. A missing streamline MP4 marks the
case as failed. When all cases succeed, the script also writes
`$OUTROOT/segmentation_nifti.zip`. The archive contains only
`*_auto_segmentation.nii` and `*_auto_segmentation.nii.gz`; nnUNet feature-channel
NIfTI files are excluded. Each file retains its path relative to `OUTROOT`.

With the default `configs/video_exporting.json`, the output is
`streamlines_rotate.mp4`; it is `streamlines_video.mp4` when dynamic rotation is
disabled in that config.

```bash
ROOT=/path/to/h5_cases \
OUTROOT=/path/to/results \
SEGMENTATION_ZIP=/path/to/segmentation_nifti.zip \
./segonly.sh
```

| Environment variable | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `ROOT` | path | script default | environment | source H5/HDF5 root scanned by `segonly.sh` | `segonly.sh` |
| `OUTROOT` | path | script default | environment | per-case output root and the base for archived relative paths | `segonly.sh` |
| `SEGMENTATION_ZIP` | path | `$OUTROOT/segmentation_nifti.zip` | environment | destination ZIP containing only automatic segmentation NIfTI outputs | `segonly.sh` |

### Distance-based planes

```bash
autoflow-run ./data/demo_data.h5 \
  --output-dir ./results/demo \
  --plane-mode distance \
  --cross-section-dist 15
```

### Center fixed-step planes (default)

```bash
autoflow-run ./data/demo_data.h5 --output-dir ./results/demo \
  --plane-mode fixed_step --plane-anchor center --plane-direction both \
  --plane-count 3 --plane-spacing-mode fraction --plane-spacing-ratio 0.25
```

The default segmentation filter assigns each graph path an owner label from
topology-aware contiguous label runs and clips plane placement and metrics to
that label. Use `--no-segmentation-filter` for a binary-mask workflow.

### Anchored-offset planes

When the path meets a graph junction, placement begins at that junction and proceeds along the branch. `--plane-anchor` is used only for a path without a junction.

```bash
autoflow-run ./data/demo_data.h5 \
  --output-dir ./results/demo \
  --plane-mode anchored_offset \
  --plane-count 3 \
  --cross-section-dist 10 \
  --plane-anchor end \
  --plane-offset-mm 10
```

### Opt in to optional computations

Phase unwrapping is opt-in and disabled by default. For single-VENC wrapped phase:

```bash
autoflow-run ./data/demo_data.h5 --phase-unwrap-method lap4D --phase-unwrap-device auto
```

Choose `gc3D`, `lap4D`, or `nprs`; add `--phase-unwrap-mask all` to ignore the segmentation mask. `lap4D` uses CUDA FFT, `nprs` uses CUDA for Fourier resampling while retaining the CPU reliability solver, and `gc3D` uses CUDA only for graph construction. Dual-VENC inputs report a skip.

```bash
autoflow-run ./data/demo_data.h5 \
  --output-dir ./results/demo \
  --with pwv,wss,vortex
```

### Export selected offline videos

```bash
autoflow-run ./data/demo_data.h5 \
  --output-dir ./results/demo \
  --with wss,pg \
  --video plane,wss,pg
```

Plane videos annotate `planeidx=<index>` by default so the rendered label matches `plane_index` in the saved plane outputs. Change the plane-video label prefix, font size, text color, and label background styling in `configs/video_exporting.json -> plane_video.label`.

## Execution Order

The current batch order in `process_single()` is:

1. `load_data`
2. optional CLI auto segmentation when `--autoseg` is enabled and no segmentation is loaded; for H5 inputs the predicted `segmask` is cached back into the source file; `--segmentation-only` stops successfully here
3. `Generate Skeleton`
4. `Generate Graph`
5. `Generate Planes`
6. plane metrics; requested WSS, TKE, or pressure fields are prepared once before plane sampling so their summaries can be attached
7. optional PWV plus export of the already computed derived metrics selected by `--with`
8. optional video export selected by `--video`

Behavior details:

- if segmentation is missing, segmentation-dependent steps are skipped
- if `--autoseg` is enabled and the case has no segmentation, auto segmentation runs before skeleton and graph steps; H5 inputs then reuse the cached `segmask` on later runs
- CLI auto segmentation prints backend/model/device details, stage progress, and per-case timing for inference plus source H5 cache write when the input is H5
- `tools/benchmark_pipeline.py` defaults to the registered DV H5 validation case, cold-start correction/segmentation semantics, and no source-H5 writes; use `--autoseg-folds all` to benchmark a future 5-fold ensemble
- directory inputs scan only top-level H5/HDF5 files, so nested output folders such as `autoflow_out/` are skipped during H5 batch discovery
- default CLI runs only through plane metrics
- `--with pwv,wss,tke,pg,vortex` enables one or more optional computations
- WSS, TKE, and pressure analysis are computed independently; requesting `pg` alone does not also compute WSS or TKE
- plane metrics reuse requested derived fields prepared for the same run instead of recomputing them during the later pixelwise export
- if TKE is unavailable, WSS and pressure gradient still run when possible and TKE stays unavailable
- PWV requires `--with pwv`, at least one configured PWV group, and (for CLI/API batch compatibility) `configs/pwv.json -> enabled=true`
- each selected stage and each rendered video writes elapsed seconds into `summary.json`

## Parameter Tables

### Input and output

| CLI flag | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `inputs` | paths | required | command line | files or directories to process | `autoflow/cli.py`, `autoflow/processing.py` |
| `--output-dir` | path | `./results` | `AutoFlowConfig.output_dir` | root output directory | `autoflow/api.py` |
| `--config-dir` | path | repo `configs/` when present | command line | load per-module JSON defaults | `autoflow/config.py` |
| `--import-planes` / `--reuse-planes` | path | empty | command line or `AutoFlowConfig.reuse_planes` | import saved plane coordinates; a directory resolves per-case files | `autoflow/plane_io.py` |
| `--plane-import-mode` | choice | `world` | command line or `AutoFlowConfig.plane_import_mode` | choose `world`, `local`, or `path_relative` cross-case mapping | `autoflow/plane_io.py` |
| `--export-planes` | path | empty | command line or `AutoFlowConfig.export_planes` | write an additional plane-coordinate JSON; use a directory for multi-case runs | `autoflow/processing.py`, `autoflow/api.py` |

### Batch and skip behavior

| CLI flag | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `--with` | csv | empty | command line | opt in to `pwv`, `wss`, `tke`, `pg`, and/or `vortex` | `autoflow/cli.py`, `autoflow/processing.py` |
| `--skip-derived` | bool | `False` | `configs/batch.json` | remove WSS, TKE, and relative-pressure work from the requested set | `autoflow/processing.py` |
| `--skip-plane-metrics` | bool | `False` | `configs/batch.json` | skip plane metric export | `autoflow/processing.py` |
| `--single-thread` | bool | multithread on | `configs/batch.json` | disable multithreaded plane metrics | `autoflow/core/pipeline.py` |

### Loading and background phase correction

| CLI flag | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `--bgc` | bool | `False` | `configs/loader.json` | enable background phase correction; H5 inputs reuse or write a compatible `corr` cache | `autoflow/algorithms/phase_correction.py`, `autoflow/algorithms/data.py` |
| `--bgc-method` | choice | `wrls_arto` | `configs/loader.json` | choose `msac` or `wrls_arto` | `autoflow/algorithms/phase_correction.py` |
| `--bgc-fit-order` | int | `3` | `configs/loader.json` | polynomial fit order for correction | `autoflow/algorithms/phase_correction.py` |
| `--bgc-threshold` | float | `0.1` | `configs/loader.json` | MSAC venc-space threshold for the stationary-tissue mask | `autoflow/algorithms/phase_correction.py` |
| `--bgc-wrls-lambda` | float | `5.0` | `configs/loader.json` | WRLS L1 regularization strength | `autoflow/algorithms/phase_correction.py` |
| `--bgc-wrls-magnitude-threshold` | float | `0.04` | `configs/loader.json` | per-slice reference-magnitude fraction used to form WRLS candidates | `autoflow/algorithms/phase_correction.py` |
| `--bgc-wrls-mid-fov-fraction` | float | `0.5` | `configs/loader.json` | middle in-plane FOV fraction used by the first-order initialization | `autoflow/algorithms/phase_correction.py` |
| `--bgc-wrls-mid-slice-fraction` | float | `0.65` | `configs/loader.json` | middle through-plane fraction used by the initialization | `autoflow/algorithms/phase_correction.py` |
| `--bgc-wrls-arto-iterations` | int | `2` | `configs/loader.json` | ARTO exclusion and refit count | `autoflow/algorithms/phase_correction.py` |
| `--bgc-wrls-tau` | float | `3.0` | `configs/loader.json` | central-Gaussian inclusion width in standard deviations | `autoflow/algorithms/phase_correction.py` |
| `--bgc-wrls-delta` | float | `2.0` | `configs/loader.json` | minimum side-Gaussian separation | `autoflow/algorithms/phase_correction.py` |
| `--bgc-wrls-central-probability` | float | `0.5` | `configs/loader.json` | minimum central-Gaussian prior | `autoflow/algorithms/phase_correction.py` |
| `--bgc-wrls-fista-iterations` | int | `5000` | `configs/loader.json` | maximum FISTA iterations per WRLS fit | `autoflow/algorithms/phase_correction.py` |
| `--bgc-wrls-gmm-iterations` | int | `1000` | `configs/loader.json` | maximum GMM EM iterations per ARTO pass | `autoflow/algorithms/phase_correction.py` |
| `--dual-venc-ratio1` | float | `0.0` | `configs/loader.json` | first dual-venc alias window ratio for legacy `Nv=7` H5 | `autoflow/algorithms/data.py` |
| `--dual-venc-ratio2` | float | `0.0` | `configs/loader.json` | second dual-venc alias window ratio for legacy `Nv=7` H5 | `autoflow/algorithms/data.py` |
| `--dicom-read-workers` | int | `1` | `configs/loader.json` | DICOM read worker count; `0` means auto selection inside loader | `autoflow/algorithms/dicom.py` |

WRLS+ARTO automatically uses CUDA for its ARTO GMM stage when the installed PyTorch build reports a usable CUDA device. There is no background-correction device flag; unavailable or failed CUDA execution falls back to CPU.

### Plane generation

| CLI flag | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `--plane-mode` | string | `fixed_step` | `configs/planes.json` | choose `uniform` or composable `fixed_step` placement (legacy modes remain accepted) | `autoflow/algorithms/planes.py` |
| `--plane-count` | int | `3` | `configs/planes.json` | requested planes; even symmetric counts omit center; `-1` fills positions that fit | `autoflow/algorithms/planes.py` |
| `--cross-section-dist` | float mm | `5.0` | `configs/planes.json` | fixed-step spacing in mm when distance spacing is selected | `autoflow/algorithms/planes.py` |
| `--plane-spacing-mode` | string | `fraction` | `configs/planes.json` | `distance` or `fraction` of the usable centerline | `autoflow/algorithms/planes.py` |
| `--plane-spacing-ratio` | float | `0.25` | `configs/planes.json` | fractional fixed-step spacing | `autoflow/algorithms/planes.py` |
| `--plane-anchor` | string | `center` | `configs/planes.json` | `start`, `center`, `end`, or `junction` | `autoflow/algorithms/planes.py` |
| `--plane-direction` | string | `both` | `configs/planes.json` | `toward_start`, `toward_end`, or `both` | `autoflow/algorithms/planes.py` |
| `--segmentation-filter` / `--no-segmentation-filter` | bool | `true` | `configs/planes.json` | enable/disable topology-aware path and metric filtering | `autoflow/algorithms/planes.py` |
| `--start-dist` | float mm | `0.0` (fixed-step) | `configs/planes.json` | advanced trim from path start before placement | `autoflow/algorithms/planes.py` |
| `--end-dist` | float mm | `0.0` | `configs/planes.json` | stop offset near path end | `autoflow/algorithms/planes.py` |
| `--plane-offset-mm` | float mm | `5.0` | `configs/planes.json` | first offset from the graph junction in anchored-offset mode | `autoflow/algorithms/planes.py` |
| `--plane-by-distance` | bool | unset | CLI compatibility flag | deprecated alias for `--plane-mode distance` | `autoflow/cli.py` |

### PWV

PWV is opt-in from the CLI and still uses `configs/pwv.json` for PWV group definitions.

- pass `--with pwv`
- enable it in `configs/pwv.json`
- define one or more PWV groups in `configs/pwv.json -> groups`

WSS, TKE, and pressure-analysis compute defaults are now split by metric:

- `configs/fluid.json` for shared fluid properties such as `rho` and `viscosity`
- `configs/wss.json` for WSS computation
- `configs/tke.json` for TKE density
- `configs/pressure_gradient.json` for pressure-gradient estimation, relative-pressure reconstruction, and centerline-pressure settings
- `configs/vortex.json` for vorticity, Q-criterion, and swirling-strength smoothing and support erosion
- `configs/planes.json` for plane render styling
- `configs/wss.json`, `configs/tke.json`, `configs/pressure_gradient.json`, and `configs/streamlines.json` for metric-specific render ranges and optional colorbars
- `configs/pathlines.json` for GUI-only pathline launch and rendering defaults; the CLI still does not export pathlines
- `configs/video_exporting.json` for shared video controls such as `window_size`, `rotate_dynamic_video`, and camera behavior

### Skeleton preprocessing

| CLI flag | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `--remove-small-cc` | bool | `True` in config defaults | `configs/skeleton.json` | remove small connected components before skeletonization | `autoflow/algorithms/preprocess.py` |
| `--min-cc-volume` | float mm^3 | `50.0` | `configs/skeleton.json` | component-volume threshold | `autoflow/algorithms/preprocess.py` |
| `--cc-filter-mode` | string | `hybrid` | `configs/skeleton.json` | choose `absolute`, `relative`, `hybrid`, or `largest` component filtering | `autoflow/algorithms/preprocess.py` |
| `--cc-rel-min-ratio` | float | `0.01` | `configs/skeleton.json` | relative threshold against the largest component for `relative` and `hybrid` filtering | `autoflow/algorithms/preprocess.py` |

### Streamlines and pathline styling

| CLI flag | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `--seed-ratio` | float | `0.02` | `configs/streamlines.json` | streamline seed density | `autoflow/algorithms/streamlines.py` |
| `--tube-radius` | float | `0.25` | `configs/streamlines.json` | streamline tube radius in mm | `autoflow/rendering/videos.py` |
| `--pressure-method` | string | `least_squares` | `configs/pressure_gradient.json` | choose `least_squares` or `ppe` relative-pressure reconstruction; both use SciPy sparse solvers | `autoflow/algorithms/metrics.py` |

### Auto segmentation

| CLI flag | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `--autoseg` | bool | `False` | command line | run auto segmentation only when the loaded case has no segmentation | `autoflow/processing.py` |
| `--autoseg-backend` | string | `nnUNet4D` in `configs/segmentation.json` | `AutoFlowConfig` | select `nnUNet4D` temporal or `nnUNet` static automatic segmentation | `autoflow/algorithms/segmentation.py` |
| `--autoseg-model` | path | Dataset7020 `.sh` in `configs/segmentation.json` | `AutoFlowConfig` | override the 4D model folder or orchestration script; static `nnUNet` accepts a model folder | `autoflow/algorithms/segmentation.py` |
| `--autoseg-checkpoint` | string | `checkpoint_final.pth` | `AutoFlowConfig` | choose nnUNet checkpoint | `autoflow/algorithms/segmentation.py` |
| `--autoseg-folds` | string | `single` | `AutoFlowConfig` | choose `single`, `all`/`ensemble`, or explicit fold IDs such as `0,1,2,3,4` | `autoflow/algorithms/segmentation.py` |
| `--autoseg-device` | string | `auto` | `AutoFlowConfig` | choose `auto`, `cpu`, or `cuda` | `autoflow/algorithms/segmentation.py` |
| `--autoseg-label-map` | JSON string | empty | `AutoFlowConfig` | remap predicted labels after inference | `autoflow/algorithms/segmentation.py` |
| `--force-recompute-seg` | bool | `False` | command line | ignore an AutoFlow-generated H5 segmentation cache and rerun automatic segmentation | `autoflow/algorithms/data.py`, `autoflow/processing.py` |
| `--ignore-embedded-segmentation` | bool | `False` | command line | ignore every embedded segmentation source for a cold start without changing the input H5 | `autoflow/algorithms/data.py`, `autoflow/core/pipeline.py` |
| `--no-cache-write` | bool | `False` | command line | keep newly computed correction and automatic-segmentation caches out of the source H5; pair with `--ignore-embedded-segmentation` for read-only timing | `autoflow/algorithms/data.py`, `autoflow/processing.py`, `autoflow/cli.py` |
| `--segmentation-only` | bool | `False` | command line | stop after loading or generating segmentation; skip skeleton, planes, metrics, and videos | `autoflow/processing.py` |

Note:

- the CLI currently takes these auto-segmentation defaults from `AutoFlowConfig`
- the GUI segmentation dialog uses the segmentation config bundle for its initial auto-segmentation fields

### Video export

| CLI flag or config key | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `--video` | csv | empty | command line | export one or more of `plane`, `wss`, `tke`, `pg`, `streamlines` | `autoflow/cli.py`, `autoflow/processing.py` |
| `--fps` | int | `12` | `configs/video_exporting.json` | output frame rate | `autoflow/rendering/videos.py` |
| `--plane-rotation-frames` | int | `180` | `configs/video_exporting.json` | frame count for plane rotation video | `autoflow/rendering/videos.py` |
| `window_size` | list[int, int] | `[1600, 1200]` | `configs/video_exporting.json` | output render size; use this instead of a Matplotlib-style `figsize` | `autoflow/rendering/videos.py` |
| `--camera-view` | string | `right` | `configs/video_exporting.json` | camera preset for dynamic videos | `autoflow/rendering/videos.py` |
| `--camera-distance-scale` | float | `1.5` | `configs/video_exporting.json` | scale camera distance | `autoflow/rendering/videos.py` |
| `--rotate-dynamic-video` / `--no-rotate-dynamic-video` | bool | `True` | `configs/video_exporting.json` | rotate or keep fixed dynamic videos | `autoflow/rendering/videos.py` |
| `--dynamic-rotation-frames` | int | `180` | `configs/video_exporting.json` | dynamic rotation frame count | `autoflow/rendering/videos.py` |
| `--dynamic-time-repeat` | int | `3` | `configs/video_exporting.json` | repeat each time frame in dynamic videos | `autoflow/rendering/videos.py` |
| `--dynamic-rotation-elevation-deg` | float | `10.0` | `configs/video_exporting.json` | dynamic rotation elevation override | `autoflow/rendering/videos.py` |
| `--add-plane-idx` | bool | `True` | `configs/video_exporting.json` | show or hide plane index labels in the plane video | `autoflow/rendering/videos.py` |
| `--add-path-idx` / `--no-path-idx` | bool | `False` | `configs/video_exporting.json` | annotate path indices in plane video | `autoflow/rendering/videos.py` |
| `planes.render.default.plane_color` | string | `yellow` | `configs/planes.json` | default plane color in the GUI | `autoflow/ui/app.py`, `autoflow/rendering/videos.py` |
| `planes.render.default.plane_opacity` | float | `0.75` | `configs/planes.json` | default plane opacity in the GUI | `autoflow/ui/app.py`, `autoflow/rendering/videos.py` |
| `plane_video.label.prefix` | string | `planeidx=` | `configs/video_exporting.json` | plane-video index label prefix before the plane number | `autoflow/rendering/videos.py` |
| `plane_video.label.font_size` | int | `28` | `configs/video_exporting.json` | plane-video index label font size | `autoflow/rendering/videos.py` |
| `plane_video.label.text_color` | string | `black` | `configs/video_exporting.json` | plane-video index label text color | `autoflow/rendering/videos.py` |
| `plane_video.label.shape_color` | string | `yellow` | `configs/video_exporting.json` | plane-video index label background color | `autoflow/rendering/videos.py` |
| `plane_video.label.shape_opacity` | float | `0.85` | `configs/video_exporting.json` | plane-video index label background opacity | `autoflow/rendering/videos.py` |
| `wss.render.show_scalar_bar` | bool | `True` | `configs/wss.json` | show or hide the WSS colorbar in GUI and exported videos | `autoflow/rendering/videos.py` |
| `tke.render.show_scalar_bar` | bool | `True` | `configs/tke.json` | show or hide the TKE colorbar in GUI and exported videos | `autoflow/rendering/videos.py` |
| `pressure_gradient.render.clim` | list[float, float] or `null` | `null` | `configs/pressure_gradient.json` | explicit pressure-gradient display range; `null` keeps the auto range | `autoflow/rendering/videos.py` |
| `pressure_gradient.render.show_scalar_bar` | bool | `True` | `configs/pressure_gradient.json` | show or hide the pressure-gradient colorbar in GUI and exported videos | `autoflow/rendering/videos.py` |
| `pressure_gradient.render.relative_pressure_clim` | list[float, float] or `null` | `null` | `configs/pressure_gradient.json` | explicit relative-pressure display range; `null` keeps the symmetric auto range | `autoflow/rendering/videos.py` |
| `pressure_gradient.render.relative_pressure_show_scalar_bar` | bool | `True` | `configs/pressure_gradient.json` | show or hide the relative-pressure colorbar in GUI and exported videos | `autoflow/rendering/videos.py` |
| `streamlines.render.clim` | list[float, float] or `null` | `null` | `configs/streamlines.json` | explicit streamline velocity range; `null` uses `0` to the all-phase P99 velocity inside the segmentation | `autoflow/algorithms/streamlines.py`, `autoflow/rendering/videos.py` |
| `streamlines.render.show_scalar_bar` | bool | `True` | `configs/streamlines.json` | show or hide the streamline colorbar in GUI and exported videos | `autoflow/rendering/videos.py` |

## Outputs

Typical CLI outputs are documented in [Outputs](outputs.md).

## Code References

- parser: `autoflow/cli.py`
- public config and API: `autoflow/api.py`
- batch orchestration: `autoflow/processing.py`
- pipeline steps: `autoflow/core/pipeline.py`
