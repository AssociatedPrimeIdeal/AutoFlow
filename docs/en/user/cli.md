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
  --plane-by-distance \
  --cross-section-dist 15
```

### Export offline videos

```bash
autoflow-run ./data/demo_data.h5 \
  --output-dir ./results/demo \
  --plane-video \
  --wss-video \
  --streamlines-video
```

## Execution Order

The current batch order in `process_single()` is:

1. `load_data`
2. optional CLI auto segmentation when `--autoseg` is enabled and no segmentation is loaded
3. `Generate Skeleton`
4. `Generate Graph`
5. `Generate Planes`
6. plane metrics
7. derived metrics export
8. enabled video export

Behavior details:

- if segmentation is missing, segmentation-dependent steps are skipped
- if `--autoseg` is enabled and the case has no segmentation, auto segmentation runs before skeleton and graph steps
- CLI auto segmentation prints backend/model/device details, stage progress, and per-case timing for inference plus sidecar save
- if TKE is unavailable, WSS and pressure gradient still run when possible and TKE stays unavailable
- if `configs/pwv.json -> enabled` is true, the plane-metrics stage also computes PWV and writes `pwv.json` plus per-group PNG plots

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
| `--skip-derived` | bool | `False` | `configs/batch.json` | skip WSS, TKE, and pressure-gradient export | `autoflow/processing.py` |
| `--skip-plane-metrics` | bool | `False` | `configs/batch.json` | skip plane metric export | `autoflow/processing.py` |
| `--single-thread` | bool | multithread on | `configs/batch.json` | disable multithreaded plane metrics | `autoflow/core/pipeline.py` |

### Loading and background phase correction

| CLI flag | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `--bgc` | bool | `False` | `configs/loader.json` | enable background phase correction | `autoflow/algorithms/phase_correction.py` |
| `--bgc-fit-order` | int | `3` | `configs/loader.json` | polynomial fit order for correction | `autoflow/algorithms/phase_correction.py` |
| `--bgc-threshold` | float | `0.1` | `configs/loader.json` | venc-space threshold for correction mask | `autoflow/algorithms/phase_correction.py` |
| `--dual-venc-ratio1` | float | `0.0` | `configs/loader.json` | first dual-venc alias window ratio for legacy `Nv=7` H5 | `autoflow/algorithms/data.py` |
| `--dual-venc-ratio2` | float | `0.0` | `configs/loader.json` | second dual-venc alias window ratio for legacy `Nv=7` H5 | `autoflow/algorithms/data.py` |
| `--dicom-read-workers` | int | `1` | `configs/loader.json` | DICOM read worker count; `0` means auto selection inside loader | `autoflow/algorithms/dicom.py` |

### Plane generation

| CLI flag | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `--plane-by-distance` | bool | center plane mode on | `configs/planes.json` | switch to evenly spaced planes | `autoflow/algorithms/planes.py` |
| `--cross-section-dist` | float mm | `5.0` | `configs/planes.json` | spacing between planes in distance mode | `autoflow/algorithms/planes.py` |
| `--start-dist` | float mm | `5.0` | `configs/planes.json` | offset from path start | `autoflow/algorithms/planes.py` |
| `--end-dist` | float mm | `0.0` | `configs/planes.json` | stop offset near path end | `autoflow/algorithms/planes.py` |

### PWV

PWV is currently config-driven rather than flag-driven.

- enable it in `configs/pwv.json`
- define one or more PWV groups in `configs/pwv.json -> groups`
- batch runs then compute PWV during the plane-metrics stage

### Skeleton preprocessing

| CLI flag | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `--remove-small-cc` | bool | `True` in config defaults | `configs/skeleton.json` | remove small connected components before skeletonization | `autoflow/algorithms/preprocess.py` |
| `--min-cc-volume` | float mm^3 | `50.0` | `configs/skeleton.json` | component-volume threshold | `autoflow/algorithms/preprocess.py` |

### Streamlines and pathline styling

| CLI flag | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `--seed-ratio` | float | `0.02` | `configs/streamlines.json` | streamline seed density | `autoflow/algorithms/streamlines.py` |
| `--tube-radius` | float | `0.05` | `configs/streamlines.json` | streamline tube radius | `autoflow/rendering/videos.py` |

### Auto segmentation

| CLI flag | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `--autoseg` | bool | `False` | command line | run auto segmentation only when the loaded case has no segmentation | `autoflow/processing.py` |
| `--autoseg-backend` | string | `nnUNet` | `AutoFlowConfig` | select automatic segmentation backend | `autoflow/algorithms/segmentation.py` |
| `--autoseg-model` | path | empty string, then resolved to bundled default model if present | `AutoFlowConfig` | choose nnUNet model folder | `autoflow/algorithms/segmentation.py` |
| `--autoseg-checkpoint` | string | `checkpoint_final.pth` | `AutoFlowConfig` | choose nnUNet checkpoint | `autoflow/algorithms/segmentation.py` |
| `--autoseg-device` | string | `auto` | `AutoFlowConfig` | choose `auto`, `cpu`, or `cuda` | `autoflow/algorithms/segmentation.py` |
| `--autoseg-label-map` | JSON string | empty | `AutoFlowConfig` | remap predicted labels after inference | `autoflow/algorithms/segmentation.py` |

Note:

- the CLI currently takes these auto-segmentation defaults from `AutoFlowConfig`
- the GUI segmentation dialog uses the segmentation config bundle for its initial auto-segmentation fields

### Video export

| CLI flag | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `--plane-video` | bool | `False` | `configs/rendering.json` | export `planes_rotate.mp4` | `autoflow/rendering/videos.py` |
| `--wss-video` | bool | `False` | `configs/rendering.json` | export WSS video | `autoflow/rendering/videos.py` |
| `--streamlines-video` | bool | `False` | `configs/rendering.json` | export streamline video | `autoflow/rendering/videos.py` |
| `--tke-video` | bool | `False` | `configs/rendering.json` | export TKE video when TKE exists | `autoflow/rendering/videos.py` |
| `--fps` | int | `12` | `configs/rendering.json` | output frame rate | `autoflow/rendering/videos.py` |
| `--plane-rotation-frames` | int | `180` | `configs/rendering.json` | frame count for plane rotation video | `autoflow/rendering/videos.py` |
| `--camera-view` | string | `right` | `configs/rendering.json` | camera preset for dynamic videos | `autoflow/rendering/videos.py` |
| `--camera-distance-scale` | float | `1.5` | `configs/rendering.json` | scale camera distance | `autoflow/rendering/videos.py` |
| `--rotate-dynamic-video` / `--no-rotate-dynamic-video` | bool | `True` | `configs/rendering.json` | rotate or keep fixed dynamic videos | `autoflow/rendering/videos.py` |
| `--dynamic-rotation-frames` | int | `180` | `configs/rendering.json` | dynamic rotation frame count | `autoflow/rendering/videos.py` |
| `--dynamic-time-repeat` | int | `3` | `configs/rendering.json` | repeat each time frame in dynamic videos | `autoflow/rendering/videos.py` |
| `--dynamic-rotation-elevation-deg` | float | `10.0` | `configs/rendering.json` | dynamic rotation elevation override | `autoflow/rendering/videos.py` |
| `--add-plane-idx` | bool | `False` | `configs/rendering.json` | annotate plane indices in plane video | `autoflow/rendering/videos.py` |
| `--add-path-idx` / `--no-path-idx` | bool | `False` | `configs/rendering.json` | annotate path indices in plane video | `autoflow/rendering/videos.py` |

## Outputs

Typical CLI outputs are documented in [Outputs](outputs.md).

## Code References

- parser: `autoflow/cli.py`
- public config and API: `autoflow/api.py`
- batch orchestration: `autoflow/processing.py`
- pipeline steps: `autoflow/core/pipeline.py`
