# Python API

## Status
The public Python API is supported through `autoflow/__init__.py` and `autoflow/api.py`.

Public entry points:

- `AutoFlowConfig`
- `build_workspace()`
- `run_case()`
- `run_batch()`
- `launch_gui()`

## Quick Use

### Run one case

```python
from autoflow import AutoFlowConfig, run_case

config = AutoFlowConfig(output_dir="./results/demo")
summary = run_case("./data/demo_data.h5", config=config)
```

### Run a batch

```python
from autoflow import AutoFlowConfig, run_batch

config = AutoFlowConfig(
    inputs=["./data/demo_data.h5"],
    output_dir="./results/demo",
    make_plane_video=False,
)
results, last_case_out = run_batch(config)
```

### Build a workspace from config defaults

```python
from autoflow import AutoFlowConfig, build_workspace

config = AutoFlowConfig.from_config_dir("./configs")
workspace = build_workspace(config)
```

### Launch the GUI from Python

```python
from autoflow import launch_gui

launch_gui(config_dir="./configs")
```

## Main Config Fields

| Field | Type | Default | Effect | Main code |
| --- | --- | --- | --- | --- |
| `inputs` | list of paths | empty | batch inputs for `run_batch()` | `autoflow/api.py` |
| `output_dir` | path | `./results` | root output directory | `autoflow/api.py` |
| `skip_derived` | bool | `False` | skip derived metrics export | `autoflow/processing.py` |
| `skip_plane_metrics` | bool | `False` | skip plane metrics | `autoflow/processing.py` |
| `background_phase_correction` | bool | `False` | enable loader correction | `autoflow/algorithms/data.py`, `autoflow/algorithms/dicom.py` |
| `dual_venc_ratio1` | float | `0.0` | first dual-venc alias window ratio for legacy `Nv=7` H5 | `autoflow/algorithms/data.py` |
| `dual_venc_ratio2` | float | `0.0` | second dual-venc alias window ratio for legacy `Nv=7` H5 | `autoflow/algorithms/data.py` |
| `use_center_plane` | bool | `True` | use one center plane per path | `autoflow/algorithms/planes.py` |
| `cross_section_dist` | float | `5.0` | distance-plane spacing | `autoflow/algorithms/planes.py` |
| `remove_small_cc` | bool | `True` | drop small components before skeletonization | `autoflow/algorithms/preprocess.py` |
| `seed_ratio` | float | `0.02` | streamline seed density | `autoflow/algorithms/streamlines.py` |
| `autoseg` | bool | `False` | run auto segmentation if no segmentation exists | `autoflow/processing.py` |
| `autoseg_model` | string | empty | choose explicit nnUNet model folder | `autoflow/algorithms/segmentation.py` |
| `make_plane_video` | bool | `False` | export plane video | `autoflow/rendering/videos.py` |
| `make_wss_video` | bool | `False` | export WSS video | `autoflow/rendering/videos.py` |
| `make_streamlines_video` | bool | `False` | export streamline video | `autoflow/rendering/videos.py` |
| `make_tke_video` | bool | `False` | export TKE video when available | `autoflow/rendering/videos.py` |

## Config Directory Support

`AutoFlowConfig.from_config_dir()` loads the per-module JSON files from a config directory and converts them to public API defaults.

Main code:

- `autoflow/config.py`
- `autoflow/api.py:AutoFlowConfig.from_config_dir()`

## Current API Boundaries

- batch processing flows through `run_case()` and `run_batch()`
- interactive GUI editing is not exposed as a stable batch API
- GUI pathlines are interactive behavior, not a public batch-processing API
- offline videos are supported through config flags in `run_case()` and `run_batch()`
