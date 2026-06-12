# AutoFlow

AutoFlow is a 4D flow MRI processing toolkit with three entry points:

- a batch CLI: `autoflow-run`
- a desktop GUI: `autoflow-gui`
- a Python API built around `AutoFlowConfig`, `run_case()`, and `run_batch()`

It currently supports:

- legacy complex H5 input
- normalized H5 input
- direct DICOM directory input
- segmentation from embedded masks, imported masks, thresholding, and nnUNet auto segmentation
- grouped multi-label segmentation with config-driven label maps, per-group preprocessing, and grouped visualization
- skeleton, graph, branch, path, and plane generation
- plane metrics, config-driven PWV, WSS, optional TKE, pressure-gradient fields, relative-pressure maps, centerline pressure-drop analysis, streamlines, and GUI pathlines
- offline result export to JSON, NPZ, H5, PNG, and video

![Demo](https://github.com/user-attachments/assets/e2c17a9e-6a47-4f85-ba0d-c35b622802b1)

## Installation

Base install:

```bash
git clone https://github.com/AssociatedPrimeIdeal/AutoFlow.git
cd AutoFlow
pip install .
```

GUI install:

```bash
pip install ".[gui]"
```

Test dependencies:

```bash
pip install -e ".[test]"
```

Known working pytest environment in this repo:

```bash
~/miniconda3/envs/ryy/bin/python -m pytest tests/test_smoke_phantoms.py tests/test_pressure_gradient_phantom.py -q
```

## Quick Start

CLI:

```bash
autoflow-run ./data/demo_data.h5 --output-dir ./results/demo
```

Optional metrics and videos are opt-in:

```bash
autoflow-run ./data/demo_data.h5 \
  --output-dir ./results/demo \
  --with pwv,wss,pg \
  --video plane,wss,pg
```

This runs the standard batch order:

1. load data
2. generate skeleton
3. generate graph
4. generate planes
5. calculate plane metrics
6. optionally calculate PWV, WSS, TKE, pressure gradient, relative pressure, and centerline pressure drop when requested
7. optionally export requested videos

Typical outputs under `./results/demo/<case_name>/`:

- `planes.json`
- `plane_positions.json`
- `plane_metrics.json`
- `plane_qc.json`
- `pwv.json` when PWV is enabled
- `pwv_<group>.png` when PWV plotting succeeds
- `summary.json` with `stage_times_sec`, `video_times_sec`, and request flags

GUI:

```bash
autoflow-gui
```

Use `Export > Export Videos...` for interactive selective export of `plane`, `wss`, `tke`, `pg`, and `streamlines` videos.

Python API:

```python
from autoflow import AutoFlowConfig, run_batch

config = AutoFlowConfig(
    inputs=["./data/demo_data.h5"],
    output_dir="./results/demo",
)

results, case_out = run_batch(config)
```

## Configuration

AutoFlow keeps default hyperparameters in per-module JSON files under `configs/`:

- `batch.json`
- `loader.json`
- `skeleton.json`
- `labels.json`
- `planes.json`
- `streamlines.json`
- `derived.json`
- `fluid.json`
- `wss.json`
- `tke.json`
- `pressure_gradient.json`
- `pwv.json`
- `segmentation.json`
- `rendering.json`
  includes only shared video and camera defaults such as `window_size`, `rotate_dynamic_video`, `dynamic_rotation_frames`, `camera_view`, and `make_*_video`

Metric config split:

- `fluid.json` owns shared fluid properties such as `rho` and `viscosity`.
- `wss.json` owns WSS compute parameters.
- `wss.json -> render` owns WSS display range and optional colorbar settings for GUI and offline videos.
- `tke.json` only needs metric-specific overrides when you want to override the shared fluid density.
- `tke.json -> render` owns TKE display range and optional colorbar settings for GUI and offline videos.
- `pressure_gradient.json` owns pressure-analysis parameters: pressure-gradient estimation, relative-pressure reconstruction, and centerline pressure-drop sampling inputs.
- `pressure_gradient.json -> render` owns pressure-gradient display settings, plus optional `relative_pressure_*` overrides for the reconstructed relative-pressure map.
- `planes.json -> render` owns plane skeleton and plane styling for GUI display and plane videos.
- `streamlines.json -> render` owns streamline display range and optional colorbar settings for GUI and offline videos.
- `derived.json` is now just a legacy compatibility placeholder.
- `rendering.json` stays the central place for shared video sizing, rotation, camera, and output toggles.

Important grouped-mask note:

- `skeleton.json` owns skeleton cleanup and morphology defaults.
- `labels.json` owns label maps, label groups, browser colors, and per-group preprocessing used by grouped multi-label vessel workflows.
- `pwv.json` owns PWV groups, spacing, waveform selection, and plot styling.

Typical usage:

```bash
autoflow-run ./data/demo_data.h5 --config-dir ./configs
autoflow-gui --config-dir ./configs
```

Python API:

```python
from autoflow import AutoFlowConfig

config = AutoFlowConfig.from_config_dir(
    "./configs",
    inputs=["./data/demo_data.h5"],
)
```

Override rules:

- CLI loads defaults from the config directory, then applies explicit CLI flags on top.
- GUI loads panel defaults from the config directory, and you can still edit them interactively.
- If `--config-dir` is omitted, AutoFlow uses the repo `configs/` directory when available.

## Documentation

Detailed documentation now lives under `docs/en/`.

### English

**User Guide**

- [Quickstart](docs/en/user/quickstart.md)
- [CLI](docs/en/user/cli.md)
- [GUI](docs/en/user/gui.md)
- [Python API](docs/en/user/python-api.md)
- [Inputs](docs/en/user/inputs.md)
- [Outputs](docs/en/user/outputs.md)
- [Troubleshooting](docs/en/user/troubleshooting.md)

**Feature Guides**

- [Segmentation](docs/en/features/segmentation.md)
- [Skeleton](docs/en/features/skeleton.md)
- [Graph and Paths](docs/en/features/graph-paths.md)
- [Planes](docs/en/features/planes.md)
- [Plane Metrics](docs/en/features/metrics.md)
- [PWV](docs/en/features/pwv.md)
- [WSS, TKE, Pressure Gradient, and Relative Pressure](docs/en/features/wss-tke-pressure.md)
- [Streamlines and Pathlines](docs/en/features/streamlines.md)
- [Videos](docs/en/features/videos.md)

**Developer Guide**

- [Architecture](docs/en/developer/architecture.md)
- [Feature-to-Code Map](docs/en/developer/feature-to-code-map.md)
- [Change Recipes](docs/en/developer/change-recipes.md)
- [Config System](docs/en/developer/config-system.md)
- [Testing](docs/en/developer/testing.md)

## Current Status Highlights

- input loading supports legacy complex H5, normalized H5, and direct DICOM directories
- loader output is normalized around `LoadedCase` with required `mag`, `flow`, `resolution`, `origin`, `venc`, and `rr`
- grouped multi-label segmentations can be reduced from 4D labels to 3D by time majority vote, cleaned per label, reduced to the largest connected component per group before skeletonization, and rendered back as grouped skeleton, graph, path, plane, and pathline objects
- TKE is optional; mag/flow-only inputs must not synthesize fake TKE
- auto segmentation is currently executable in both CLI and GUI when the nnUNet backend and model folder are available
- the GUI Browser can show and hide a whole segmentation group at once, and group title colors come from `configs/labels.json`
- when PWV is enabled, the GUI uses one `Analysis` dock for PWV, plane cardiac-phase curves, and path/branch internal consistency, and still exposes PWV planes as one `PWV planes` browser item
- `Run All` in the GUI runs `Generate Skeleton -> Generate Graph -> Generate Planes -> Calculate && Save Metrics -> Compute PWV -> WSS / TKE / Pressure`
- offline videos can be exported from CLI, Python batch, or GUI `Export > Export Videos...`; CLI and Python remain the repeatable batch path

## Documentation Checklist For PRs

A PR must update documentation when it changes any of the following:

- CLI flags, defaults, or command examples
- GUI menus, buttons, panels, shortcuts, or workflow
- Python public API behavior
- input formats or loader normalization behavior
- output files or JSON, H5, NPZ, or video structure
- pipeline step order or skip behavior
- segmentation behavior or available backends
- config keys or defaults
- tests that define expected behavior
