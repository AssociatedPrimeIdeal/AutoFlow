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

### Reuse existing planes

```bash
autoflow-run ./data/demo_data.h5 \
  --output-dir ./results/demo \
  --reuse-planes ./results/old_case/plane_positions.json
```

### Enable auto segmentation for cases without segmentation

```bash
autoflow-run ./data/demo_data.h5 \
  --output-dir ./results/demo \
  --autoseg
```

### Distance-based planes

```bash
autoflow-run ./data/demo_data.h5 \
  --output-dir ./results/demo \
  --plane-mode distance \
  --cross-section-dist 15
```

### Anchored-offset planes

```bash
autoflow-run ./data/demo_data.h5 \
  --output-dir ./results/demo \
  --plane-mode anchored_offset \
  --plane-anchor end \
  --plane-offset-mm 10
```

### Opt in to optional computations

```bash
autoflow-run ./data/demo_data.h5 \
  --output-dir ./results/demo \
  --with pwv,wss
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
6. plane metrics
7. optional PWV or derived metrics selected by `--with`
8. optional video export selected by `--video`

Behavior details:

- if segmentation is missing, segmentation-dependent steps are skipped
- if `--autoseg` is enabled and the case has no segmentation, auto segmentation runs before skeleton and graph steps; H5 inputs then reuse the cached `segmask` on later runs
- CLI auto segmentation prints backend/model/device details, stage progress, and per-case timing for inference plus source H5 cache write when the input is H5
- directory inputs scan only top-level H5/HDF5 files, so nested output folders such as `autoflow_out/` are skipped during H5 batch discovery
- default CLI runs only through plane metrics
- `--with pwv,wss,tke,pg` enables one or more optional computations
- if TKE is unavailable, WSS and pressure gradient still run when possible and TKE stays unavailable
- PWV still requires `configs/pwv.json -> enabled` plus at least one configured PWV group
- each selected stage and each rendered video writes elapsed seconds into `summary.json`

## Parameter Tables

### Input and output

| CLI flag | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `inputs` | paths | required | command line | files or directories to process | `autoflow/cli.py`, `autoflow/processing.py` |
| `--output-dir` | path | `./results` | `AutoFlowConfig.output_dir` | root output directory | `autoflow/api.py` |
| `--config-dir` | path | repo `configs/` when present | command line | load per-module JSON defaults | `autoflow/config.py` |
| `--reuse-planes` | path | empty | `configs/batch.json` or flag | reuse saved plane positions | `autoflow/plane_io.py` |

### Batch and skip behavior

| CLI flag | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `--with` | csv | empty | command line | opt in to `pwv`, `wss`, `tke`, and/or `pg` | `autoflow/cli.py`, `autoflow/processing.py` |
| `--skip-derived` | bool | `False` | `configs/batch.json` | remove WSS, TKE, and relative-pressure work from the requested set | `autoflow/processing.py` |
| `--skip-plane-metrics` | bool | `False` | `configs/batch.json` | skip plane metric export | `autoflow/processing.py` |
| `--single-thread` | bool | multithread on | `configs/batch.json` | disable multithreaded plane metrics | `autoflow/core/pipeline.py` |

### Loading and background phase correction

| CLI flag | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `--bgc` | bool | `False` | `configs/loader.json` | enable background phase correction; H5 inputs reuse or write a compatible `corr` cache | `autoflow/algorithms/phase_correction.py`, `autoflow/algorithms/data.py` |
| `--bgc-fit-order` | int | `3` | `configs/loader.json` | polynomial fit order for correction | `autoflow/algorithms/phase_correction.py` |
| `--bgc-threshold` | float | `0.1` | `configs/loader.json` | venc-space threshold for correction mask | `autoflow/algorithms/phase_correction.py` |
| `--dual-venc-ratio1` | float | `0.0` | `configs/loader.json` | first dual-venc alias window ratio for legacy `Nv=7` H5 | `autoflow/algorithms/data.py` |
| `--dual-venc-ratio2` | float | `0.0` | `configs/loader.json` | second dual-venc alias window ratio for legacy `Nv=7` H5 | `autoflow/algorithms/data.py` |
| `--dicom-read-workers` | int | `1` | `configs/loader.json` | DICOM read worker count; `0` means auto selection inside loader | `autoflow/algorithms/dicom.py` |

### Plane generation

| CLI flag | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `--plane-mode` | string | `count` | `configs/planes.json` | choose `count`, `distance`, or `anchored_offset` plane placement | `autoflow/algorithms/planes.py` |
| `--plane-count` | int | `1` | `configs/planes.json` | evenly spaced plane count in count mode; `1` gives the center-style default | `autoflow/algorithms/planes.py` |
| `--cross-section-dist` | float mm | `5.0` | `configs/planes.json` | spacing between planes in distance mode | `autoflow/algorithms/planes.py` |
| `--start-dist` | float mm | `5.0` | `configs/planes.json` | trim from path start before count or distance placement | `autoflow/algorithms/planes.py` |
| `--end-dist` | float mm | `0.0` | `configs/planes.json` | stop offset near path end | `autoflow/algorithms/planes.py` |
| `--plane-anchor` | string | `end` | `configs/planes.json` | choose `start` or `end` anchor in anchored-offset mode | `autoflow/algorithms/planes.py` |
| `--plane-offset-mm` | float mm | `5.0` | `configs/planes.json` | offset from the chosen anchor in anchored-offset mode | `autoflow/algorithms/planes.py` |
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
- `configs/planes.json` for plane render styling
- `configs/wss.json`, `configs/tke.json`, `configs/pressure_gradient.json`, and `configs/streamlines.json` for metric-specific render ranges and optional colorbars
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
| `--tube-radius` | float | `0.05` | `configs/streamlines.json` | streamline tube radius | `autoflow/rendering/videos.py` |
| `--pressure-method` | string | `least_squares` | `configs/pressure_gradient.json` | choose `least_squares` or `ppe` relative-pressure reconstruction; both use SciPy sparse solvers | `autoflow/algorithms/metrics.py` |

### Auto segmentation

| CLI flag | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `--autoseg` | bool | `False` | command line | run auto segmentation only when the loaded case has no segmentation | `autoflow/processing.py` |
| `--autoseg-backend` | string | `nnUNet` | `AutoFlowConfig` | select automatic segmentation backend | `autoflow/algorithms/segmentation.py` |
| `--autoseg-model` | path | empty string, then resolved to bundled default `autoflow/segmodel/nnUNetTrainerPartBalanced__nnUNetPlans__3d_fullres_iso1mm` if present | `AutoFlowConfig` | choose nnUNet model folder | `autoflow/algorithms/segmentation.py` |
| `--autoseg-checkpoint` | string | `checkpoint_final.pth` | `AutoFlowConfig` | choose nnUNet checkpoint | `autoflow/algorithms/segmentation.py` |
| `--autoseg-device` | string | `auto` | `AutoFlowConfig` | choose `auto`, `cpu`, or `cuda` | `autoflow/algorithms/segmentation.py` |
| `--autoseg-label-map` | JSON string | empty | `AutoFlowConfig` | remap predicted labels after inference | `autoflow/algorithms/segmentation.py` |
| `--force-recompute-seg` | bool | `False` | command line | ignore an AutoFlow-generated H5 segmentation cache and rerun automatic segmentation | `autoflow/algorithms/data.py`, `autoflow/processing.py` |
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
| `streamlines.render.show_scalar_bar` | bool | `True` | `configs/streamlines.json` | show or hide the streamline colorbar in GUI and exported videos | `autoflow/rendering/videos.py` |

## Outputs

Typical CLI outputs are documented in [Outputs](outputs.md).

## Code References

- parser: `autoflow/cli.py`
- public config and API: `autoflow/api.py`
- batch orchestration: `autoflow/processing.py`
- pipeline steps: `autoflow/core/pipeline.py`
