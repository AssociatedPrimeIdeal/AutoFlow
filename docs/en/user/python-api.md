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

Phase unwrapping is opt-in:

```python
config = AutoFlowConfig(
    output_dir="./results/demo",
    phase_unwrap_method="lap4D",
    phase_unwrap_device="auto",
)
```

The returned summary includes `phase_unwrap` statistics and, when enabled, a `phase_unwrap_file` NPZ. Dual-VENC inputs report a skipped phase-unwrapping step.

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
- `configs/vortex.json` for vorticity, Q-criterion, and swirling-strength smoothing and support erosion
- `configs/planes.json` for GUI plane styling, plus `configs/video_exporting.json` for plane-video styling
- `configs/wss.json`, `configs/tke.json`, `configs/pressure_gradient.json`, and `configs/streamlines.json` for metric-specific render ranges and optional colorbars
- `configs/pathlines.json` for GUI pathline launch, color, tube, and temporal-cache defaults
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
| `reuse_planes` | path | empty | import plane coordinates before metrics | `autoflow/plane_io.py`, `autoflow/processing.py` |
| `plane_import_mode` | string | `world` | map imported planes by `world`, `local`, or `path_relative` coordinates | `autoflow/plane_io.py` |
| `export_planes` | path | empty | write an additional coordinate JSON; multi-case runs require a directory | `autoflow/api.py`, `autoflow/processing.py` |
| `requested_metrics` | list[str] | empty | opt in to `pwv`, `wss`, `tke`, `pg`, and/or `vortex` | `autoflow/processing.py` |
| `skip_derived` | bool | `False` | remove requested WSS, TKE, pressure-gradient, relative-pressure, and centerline-pressure work | `autoflow/processing.py` |
| `pressure_method` | string | `least_squares` | choose `least_squares` or `ppe` relative-pressure reconstruction; large least-squares systems use AMG-preconditioned CG when available | `autoflow/algorithms/metrics.py` |
| `skip_plane_metrics` | bool | `False` | skip plane metrics | `autoflow/processing.py` |
| `background_phase_correction` | bool | `False` | enable loader correction; H5 inputs reuse or write a compatible `corr` cache | `autoflow/algorithms/data.py`, `autoflow/algorithms/dicom.py` |
| `background_phase_method` | string | `wrls_arto` | choose `msac` or `wrls_arto` | `autoflow/algorithms/phase_correction.py` |
| `background_phase_corr_fit_order` | int | `3` | polynomial fit order for the selected correction method | `autoflow/algorithms/phase_correction.py` |
| `background_phase_threshold` | float | `0.1` | MSAC stationary-tissue threshold in venc units | `autoflow/algorithms/phase_correction.py` |
| `background_phase_wrls_lambda` | float | `5.0` | WRLS L1 regularization strength | `autoflow/algorithms/phase_correction.py` |
| `background_phase_wrls_magnitude_threshold` | float | `0.04` | per-slice reference-magnitude fraction for WRLS candidates | `autoflow/algorithms/phase_correction.py` |
| `background_phase_wrls_mid_fov_fraction` | float | `0.5` | middle in-plane FOV fraction used for WRLS initialization | `autoflow/algorithms/phase_correction.py` |
| `background_phase_wrls_mid_slice_fraction` | float | `0.65` | middle through-plane fraction used for WRLS initialization | `autoflow/algorithms/phase_correction.py` |
| `background_phase_wrls_arto_iterations` | int | `2` | ARTO exclusion and refit count | `autoflow/algorithms/phase_correction.py` |
| `background_phase_wrls_tau` | float | `3.0` | central-Gaussian inclusion width | `autoflow/algorithms/phase_correction.py` |
| `background_phase_wrls_delta` | float | `2.0` | minimum side-Gaussian separation | `autoflow/algorithms/phase_correction.py` |
| `background_phase_wrls_central_probability` | float | `0.5` | minimum central-Gaussian prior | `autoflow/algorithms/phase_correction.py` |
| `background_phase_wrls_fista_iterations` | int | `5000` | maximum FISTA iterations per WRLS fit | `autoflow/algorithms/phase_correction.py` |
| `background_phase_wrls_gmm_iterations` | int | `1000` | maximum GMM EM iterations per ARTO pass | `autoflow/algorithms/phase_correction.py` |
| `dual_venc_ratio1` | float | `0.0` | first dual-venc alias window ratio for legacy `Nv=7` H5 | `autoflow/algorithms/data.py` |
| `dual_venc_ratio2` | float | `0.0` | second dual-venc alias window ratio for legacy `Nv=7` H5 | `autoflow/algorithms/data.py` |
| `plane_mode` | string | `fixed_step` | `uniform` distributes planes over a path; `fixed_step` composes anchor, direction, and spacing | `autoflow/algorithms/planes.py` |
| `plane_count` | int | `3` | requested planes; symmetric even counts omit center; `-1` fills positions that fit | `autoflow/algorithms/planes.py` |
| `cross_section_dist` | float | `5.0` | fixed-step spacing in mm when `plane_spacing_mode="distance"` | `autoflow/algorithms/planes.py` |
| `plane_spacing_mode` | string | `fraction` | interpret spacing as physical mm (`distance`) or path-length fraction | `autoflow/algorithms/planes.py` |
| `plane_spacing_ratio` | float | `0.25` | fractional fixed-step spacing | `autoflow/algorithms/planes.py` |
| `plane_direction` | string | `both` | `toward_start`, `toward_end`, or `both` from the anchor | `autoflow/algorithms/planes.py` |
| `segmentation_filter` | bool | `True` | select a topology-aware owner label, crop placement to it, and filter plane metrics to it | `autoflow/algorithms/planes.py`, `autoflow/algorithms/metrics.py` |
| `start_dist` | float | `0.0` | advanced trim from path start before placement | `autoflow/algorithms/planes.py` |
| `end_dist` | float | `0.0` | trim from path end before placement | `autoflow/algorithms/planes.py` |
| `plane_anchor` | string | `center` | `start`, `center`, `end`, or `junction` placement anchor | `autoflow/algorithms/planes.py` |
| `plane_offset_mm` | float | `5.0` | first offset from the graph junction in anchored-offset mode | `autoflow/algorithms/planes.py` |
| `remove_small_cc` | bool | `True` | drop small components before skeletonization | `autoflow/algorithms/preprocess.py` |
| `separate_special_label_contacts` | bool | `True` | separate contacts only between configured special labels (default: `RBCT`, `CCA`, `LBCT`) | `autoflow/algorithms/preprocess.py` |
| `special_contact_labels` | list[str] | `["RBCT", "CCA", "LBCT"]` | names of labels whose pairwise contacts are separated | `autoflow/core/models.py` |
| `min_cc_volume` | float | `50.0` | absolute component threshold in mm^3 | `autoflow/algorithms/preprocess.py` |
| `cc_filter_mode` | string | `hybrid` | choose `absolute`, `relative`, `hybrid`, or `largest` component filtering | `autoflow/algorithms/preprocess.py` |
| `cc_rel_min_ratio` | float | `0.01` | relative threshold against the largest component for `relative` and `hybrid` filtering | `autoflow/algorithms/preprocess.py` |
| `seed_ratio` | float | `0.02` | streamline seed density | `autoflow/algorithms/streamlines.py` |
| `pathline_seed_ratio` | float | `0.2` | `configs/pathlines.json` ratio-mode cross-section seed density | `autoflow/algorithms/streamlines.py` |
| `pathline_min_seeds` | int | `50` | `configs/pathlines.json` ratio-mode lower seed bound | `autoflow/algorithms/streamlines.py` |
| `pathline_seed_mode` | string | `fixed` | `configs/pathlines.json`: choose `fixed` for the configured count per plane or `ratio` for area-dependent sampling | `autoflow/algorithms/streamlines.py`, `autoflow/core/models.py` |
| `pathline_max_seeds` | int | `250` | `configs/pathlines.json` fixed launch count or ratio-mode cap; t=0 plane seeds are retained with the trajectory | `autoflow/algorithms/streamlines.py`, `autoflow/core/models.py` |
| `pathline_max_steps` | int | `200` | `configs/pathlines.json` maximum VTK cardiac-frame updates per pathline | `autoflow/algorithms/streamlines.py`, `autoflow/ui/app.py` |
| `pathline_terminal_speed` | float | `0.01` | `configs/pathlines.json` VTK pathline stop threshold in m/s | `autoflow/algorithms/streamlines.py` |
| `pathline_rng_seed` | int | `0` | `configs/pathlines.json` deterministic seed selection | `autoflow/algorithms/streamlines.py` |
| `pathline_tube_radius` | float | `0.25` | `configs/pathlines.json` pathline tube radius in mm | `autoflow/ui/viewer.py` |
| `pathline_color_mode` | string | `per_plane` | `configs/pathlines.json`: stable `uniform`, `per_plane`, or `per_group` pathline colors | `autoflow/core/models.py`, `autoflow/ui/app.py` |
| `pathline_color` | string | `deepskyblue` | `configs/pathlines.json` uniform-mode pathline color | `autoflow/core/models.py`, `autoflow/ui/app.py` |
| `pathline_temporal_cache_mb` | float | `512.0` | `configs/pathlines.json` cap for VTK frame reuse during all-plane GUI pathlines; `0` uses rolling frames only | `autoflow/algorithms/streamlines.py`, `autoflow/ui/app.py` |
| `autoseg` | bool | `False` | run auto segmentation if no segmentation exists | `autoflow/processing.py` |
| `autoseg_model` | string | Dataset7020 `.sh` from the shipped config | override the 4D model folder/script or static nnUNet model folder | `autoflow/algorithms/segmentation.py` |
| `autoseg_backend` | string | `nnUNet4D` from the shipped config | choose `nnUNet4D` temporal or `nnUNet` static inference | `autoflow/algorithms/segmentation.py` |
| `autoseg_folds` | string | `single` | select one fold, all folds for an ensemble, or explicit fold IDs | `autoflow/algorithms/segmentation.py` |
| `force_recompute_corr` | bool | `False` | ignore reusable background-phase correction cache | `autoflow/algorithms/data.py` |
| `background_phase_write_cache` | bool | `True` | write newly computed correction back to source H5 | `autoflow/algorithms/data.py` |
| `ignore_embedded_segmentation` | bool | `False` | ignore all segmentation embedded in the input for a cold run | `autoflow/algorithms/data.py` |
| `force_recompute_seg` | bool | `False` | ignore AutoFlow-generated segmentation cache | `autoflow/algorithms/data.py`, `autoflow/processing.py` |
| `write_segmentation_cache` | bool | `True` | write generated segmentation back to source H5 | `autoflow/processing.py` |
| `segmentation_only` | bool | `False` | stop after loading/generating segmentation | `autoflow/processing.py` |
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
| `streamline_clim` | tuple[float, float] or `None` | `None` | explicit streamline velocity range; `None` uses `0` to the all-phase P99 velocity inside the segmentation | `autoflow/algorithms/streamlines.py`, `autoflow/rendering/videos.py` |
| `wss_show_scalar_bar`, `tke_show_scalar_bar`, `pressure_gradient_show_scalar_bar`, `relative_pressure_show_scalar_bar`, `streamline_show_scalar_bar` | bool | `True` | show or hide the matching GUI and video colorbar | `autoflow/rendering/videos.py`, `autoflow/core/pipeline.py` |

WRLS+ARTO has no public device field. Its ARTO GMM stage uses CUDA automatically when PyTorch reports an available device and otherwise runs on CPU.

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
