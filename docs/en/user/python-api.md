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

Use `phase_unwrap_method="pudip"` or `"gust"` for the optional learned
backends after initializing their git submodules and installing
`pip install ".[pu]"`. Put their training options in
`configs/phase_unwrapping.json` under `backend_params`, or pass the same mapping as
`AutoFlowConfig(phase_unwrap_backend_params={...})`.

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

WSS now defaults to zero wall velocity (`no_slip_condition=True`) and computes signed shear vectors in Pa with analytic wall derivatives. `cal_wss_from_surf()` accepts an optional `support_grid` whose `wss_lumen` cell scalars identify valid sampling segments. Invalid surface values are NaN with `wss_valid=0`. Pressure support always requires a valid spatial and temporal stencil; `support_erosion_iters=0` only disables the additional erosion margin. See [WSS and pressure](../features/pressure.md) for units, reconstruction methods, and migration of old results.

### Launch the GUI from Python

```python
from autoflow import launch_gui

launch_gui(config_dir="./configs")
```

## Parameter reference

Every supported field is explained in [Python API parameters](api-parameters.md). For shipped JSON values, see [Configuration parameters](parameters.md).

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

## Function contracts

| Function | Parameters | Result / behaviour |
| --- | --- | --- |
| `build_workspace(config=None)` | Optional AutoFlowConfig | Creates configured Workspace without loading a case |
| `run_case(input_path, output_dir=None, config=None, workspace=None)` | File/InputCase; optional output path/config/template workspace | Summary dictionary; creates per-case outputs. The supplied workspace is deep-copied and is not the loaded/result workspace |
| `run_batch(config)` | AutoFlowConfig including inputs | (summaries, last_case_output_dir); collects H5 groups/DICOM cases and processes them |
| `launch_gui(config_dir=None)` | Optional module config directory | Starts the Qt GUI; see GUI workflow |

## Optional DICOM conversion

Set `AutoFlowConfig.dicom_backend="dicom2h5"` and `dicom_h5_dir` for preserved conversion during batch collection. `run_batch()` processes all converted H5 groups; `run_case()` requires one group or an explicit H5 `InputCase`. Use `convert_dicom_input()` for conversion without analysis. Installation, signatures, output contracts and failure behavior are documented in [DICOM loading and conversion](../features/dicom-loading.md).
