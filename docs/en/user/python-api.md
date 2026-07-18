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
    requested_metrics=["pwv", "wss"],
    requested_videos=["plane", "wss"],
)
results, last_case_out = run_batch(config)
```

### Build a workspace from config defaults

```python
from autoflow import AutoFlowConfig, build_workspace

config = AutoFlowConfig.from_config_dir("./configs")
workspace = build_workspace(config)
```

### Enable PWV through config files

```python
from autoflow import AutoFlowConfig, run_case

config = AutoFlowConfig.from_config_dir("./configs")
summary = run_case("case.h5", config=config)
```

PWV group definitions, timing method selection, and plotting defaults still come from `configs/pwv.json`, while `AutoFlowConfig.requested_metrics=["pwv"]` decides whether the batch run executes PWV.

For the other derived metrics, `AutoFlowConfig.from_config_dir()` now reads:

- `configs/fluid.json` for shared fluid properties such as `rho` and `viscosity`
- `configs/wss.json` for WSS compute parameters
- `configs/tke.json` for TKE density
- `configs/pressure_gradient.json` for pressure-gradient estimation, relative-pressure reconstruction, and centerline-pressure outputs
- `configs/planes.json` for GUI plane styling, plus `configs/video_exporting.json` for plane-video styling
- `configs/wss.json`, `configs/tke.json`, `configs/pressure_gradient.json`, and `configs/streamlines.json` for metric-specific render ranges and optional colorbars
- `configs/video_exporting.json` for shared video controls such as `window_size`, camera behavior, and rotation

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
| `requested_metrics` | list[str] | empty | opt in to `pwv`, `wss`, `tke`, and/or `pg` | `autoflow/processing.py` |
| `skip_derived` | bool | `False` | remove requested WSS, TKE, pressure-gradient, relative-pressure, and centerline-pressure work | `autoflow/processing.py` |
| `pressure_method` | string | `least_squares` | choose `least_squares` or `ppe` relative-pressure reconstruction; both use SciPy sparse solvers | `autoflow/algorithms/metrics.py` |
| `skip_plane_metrics` | bool | `False` | skip plane metrics | `autoflow/processing.py` |
| `background_phase_correction` | bool | `False` | enable loader correction; H5 inputs reuse or write a compatible `corr` cache | `autoflow/algorithms/data.py`, `autoflow/algorithms/dicom.py` |
| `dual_venc_ratio1` | float | `0.0` | first dual-venc alias window ratio for legacy `Nv=7` H5 | `autoflow/algorithms/data.py` |
| `dual_venc_ratio2` | float | `0.0` | second dual-venc alias window ratio for legacy `Nv=7` H5 | `autoflow/algorithms/data.py` |
| `plane_mode` | string | `count` | choose `count`, `distance`, or `anchored_offset` plane placement | `autoflow/algorithms/planes.py` |
| `plane_count` | int | `1` | evenly spaced plane count in count mode; `1` gives the center-style default | `autoflow/algorithms/planes.py` |
| `cross_section_dist` | float | `5.0` | distance-plane spacing | `autoflow/algorithms/planes.py` |
| `start_dist` | float | `5.0` | trim from path start before count or distance placement | `autoflow/algorithms/planes.py` |
| `end_dist` | float | `0.0` | trim from path end | `autoflow/algorithms/planes.py` |
| `plane_anchor` | string | `end` | choose `start` or `end` anchor in anchored-offset mode | `autoflow/algorithms/planes.py` |
| `plane_offset_mm` | float | `5.0` | offset from the chosen anchor in anchored-offset mode | `autoflow/algorithms/planes.py` |
| `remove_small_cc` | bool | `True` | drop small components before skeletonization | `autoflow/algorithms/preprocess.py` |
| `min_cc_volume` | float | `50.0` | absolute component threshold in mm^3 | `autoflow/algorithms/preprocess.py` |
| `cc_filter_mode` | string | `hybrid` | choose `absolute`, `relative`, `hybrid`, or `largest` component filtering | `autoflow/algorithms/preprocess.py` |
| `cc_rel_min_ratio` | float | `0.01` | relative threshold against the largest component for `relative` and `hybrid` filtering | `autoflow/algorithms/preprocess.py` |
| `seed_ratio` | float | `0.02` | streamline seed density | `autoflow/algorithms/streamlines.py` |
| `autoseg` | bool | `False` | run auto segmentation if no segmentation exists | `autoflow/processing.py` |
| `autoseg_model` | string | empty, then resolved to bundled `autoflow/segmodel/nnUNetTrainerPartBalanced__nnUNetPlans__3d_fullres_iso1mm` if present | choose explicit nnUNet model folder | `autoflow/algorithms/segmentation.py` |
| `requested_videos` | list[str] | empty | export any of `plane`, `wss`, `tke`, `pg`, `streamlines` | `autoflow/rendering/videos.py` |
| `add_plane_idx` | bool | `True` | show or hide plane index labels in the plane video | `autoflow/rendering/videos.py` |
| `plane_video_cfg["default"]["plane_color"]` | string | `yellow` | fallback plane color for plane videos | `autoflow/rendering/videos.py`, `autoflow/ui/app.py` |
| `plane_video_cfg["default"]["plane_opacity"]` | float | `0.75` | fallback plane opacity for plane videos | `autoflow/rendering/videos.py`, `autoflow/ui/app.py` |
| `plane_video_cfg["label"]["prefix"]` | string | `planeidx=` | plane-video index label prefix before the plane number | `autoflow/rendering/videos.py` |
| `plane_video_cfg["label"]["font_size"]` | int | `28` | plane-video index label font size | `autoflow/rendering/videos.py` |
| `plane_video_cfg["label"]["text_color"]` | string | `black` | plane-video index label text color | `autoflow/rendering/videos.py` |
| `window_size` | tuple[int, int] | `(1600, 1200)` | output render size for exported videos | `autoflow/rendering/videos.py` |
| `pressure_gradient_clim` | tuple[float, float] or `None` | `None` | explicit pressure-gradient video display range; `None` keeps the auto range | `autoflow/rendering/videos.py` |
| `relative_pressure_clim` | tuple[float, float] or `None` | `None` | explicit relative-pressure video display range; `None` keeps the symmetric auto range | `autoflow/rendering/videos.py` |
| `wss_show_scalar_bar`, `tke_show_scalar_bar`, `pressure_gradient_show_scalar_bar`, `relative_pressure_show_scalar_bar`, `streamline_show_scalar_bar` | bool | `True` | show or hide the matching GUI and video colorbar | `autoflow/rendering/videos.py`, `autoflow/core/pipeline.py` |

## Config Directory Support

`AutoFlowConfig.from_config_dir()` loads the per-module JSON files from a config directory and converts them to public API defaults.

Main code:

- `autoflow/config.py`
- `autoflow/api.py:AutoFlowConfig.from_config_dir()`

## Current API Boundaries

- batch processing flows through `run_case()` and `run_batch()`
- interactive GUI editing is not exposed as a stable batch API
- GUI pathlines are interactive behavior, not a public batch-processing API
- offline videos are supported through `requested_videos` in `run_case()` and `run_batch()`
- PWV is available through the config bundle loaded by `AutoFlowConfig.from_config_dir()` and `build_workspace()`
