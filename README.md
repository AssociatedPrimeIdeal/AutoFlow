# AutoFlow

AutoFlow is a 4D flow MRI processing toolkit with three entry points:

- a batch CLI: `autoflow-run`
- a desktop GUI: `autoflow-gui`
- a Python API built around `AutoFlowConfig`, `run_case()`, and `run_batch()`

It currently supports:

- legacy complex H5 input
- normalized H5 input
- nested-group H5 input when one case payload lives below the root and loader keys vary by case
- direct DICOM directory input
- segmentation from embedded masks, imported masks, thresholding, static nnUNet, and temporal nnUNet4D auto segmentation
- grouped multi-label segmentation with config-driven label maps, per-group preprocessing, and grouped visualization
- skeleton, graph, branch, path, and plane generation
- plane metrics, config-driven PWV, WSS, optional TKE, pressure-gradient fields, relative-pressure maps, vortex kinematics (vorticity, Q-criterion, and swirling strength), centerline pressure-drop analysis, streamlines, and GUI pathlines
- offline result export to JSON, NPZ, H5, PNG, and video

![Demo](https://github.com/user-attachments/assets/e2c17a9e-6a47-4f85-ba0d-c35b622802b1)

## Installation

Base install:

```bash
git clone https://github.com/AssociatedPrimeIdeal/AutoFlow.git
cd AutoFlow
pip install .
```

GUI install, including automatic segmentation:

```bash
pip install ".[gui]"
```

Install the optional SpatioTemporal Labeler bridge as well:

```bash
git submodule update --init --recursive
pip install ".[gui,labeler]"
```

Normal and editable installs include the bundled nnUNet `dataset.json`, `plans.json`, and final checkpoint as package data. The `gui` extra also installs the nnUNet inference runtime; no separate automatic-segmentation extra or model download is required. The `labeler` extra installs the pinned SpatioTemporal Labeler `v0.4.0` interface; its upstream source remains a separate GPL-3.0 submodule.

## Quick Start

CLI:

```bash
autoflow-run ./data/demo_data.h5 --output-dir ./results/demo
```

Optional metrics and videos are opt-in:

```bash
autoflow-run ./data/demo_data.h5 \
  --output-dir ./results/demo \
  --with pwv,wss,pg,vortex \
  --video plane,wss,pg
```

Directory inputs scan only top-level H5/HDF5 files. Nested output folders such as `autoflow_out/` are not recursed into for H5 batch discovery. Multi-group H5 files create one case output per H5 data-group path.

This runs the standard batch order:

1. load data
2. generate skeleton
3. generate graph
4. generate planes
5. calculate plane metrics
6. optionally calculate PWV, WSS, TKE, pressure gradient, relative pressure, vortex kinematics, and centerline pressure drop when requested
7. optionally export requested videos

Typical outputs under `./results/demo/<case_name>/`:

- `planes.json`
- `planes.h5`
- `plane_positions.json`
- `plane_metrics.json`
- `plane_qc.json`
- `quality_report.json` with staged input, segmentation, topology, plane, flow-consistency, and PWV checks
- `pwv.json` when PWV runs
- `pwv_<group>.png` when PWV plotting succeeds
- `summary.json` with `stage_times_sec`, `video_times_sec`, and request flags
- source H5 `segmask` is updated in place for reuse, plus `*_auto_segmentation.nii.gz` and `*_auto_segmentation_feature_*.nii.gz` when `--autoseg` runs on an H5 input

The shipped automatic-segmentation config uses the Dataset7020 temporal
`nnUNet4D` model (`run_7020_4d_full_ssd_20260824.sh`). Use
`--autoseg-folds single` for the current `fold_all` checkpoint, or `all` / an
explicit list such as `0,1,2,3,4` when five folds are available. For a
read-only cold benchmark, use `tools/benchmark_pipeline.py`, which defaults to
the registered DV validation H5 and ignores embedded correction and segmentation
caches.

GUI:

```bash
autoflow-gui
```

Use `Export > Export Videos...` for interactive selective export of `plane`, `wss`, `tke`, `pg`, and `streamlines` videos.
The GUI is organized as `Input & QC`, `Segmentation`, `Phase Unwrapping`, `Centerline & Planes`, `Hemodynamics`, and `Review & Export`. Phase unwrapping is optional and dual-VENC inputs skip it automatically. Use the top-level `Settings` menu to configure the display-only 3D axis orientation.
The 3-D Browser also provides a phase-resolved 4D PC-MRA volume backdrop and
per-object opacity controls; the ortho viewer has an independent
segmentation-overlay opacity slider.

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
- `pathlines.json`
- `derived.json`
- `fluid.json`
- `wss.json`
- `tke.json`
- `pressure_gradient.json`
- `vortex.json`
- `pwv.json`
- `segmentation.json`
- `colorbar.json`
- `video_exporting.json`
  includes only shared video and camera defaults such as `window_size`, `rotate_dynamic_video`, `dynamic_rotation_frames`, `camera_view`, and `make_*_video`

Metric config split:

- `fluid.json` owns shared fluid properties such as `rho` and `viscosity`.
- `wss.json` owns WSS compute parameters.
- `wss.json -> render` owns WSS display range and optional colorbar settings for GUI and offline videos.
- `tke.json` only needs metric-specific overrides when you want to override the shared fluid density.
- `tke.json -> render` owns TKE display range and optional colorbar settings for GUI and offline videos.
- `pressure_gradient.json` owns pressure-analysis parameters: pressure-gradient estimation, relative-pressure reconstruction, and centerline pressure-drop sampling inputs.
- `pressure_gradient.json -> render` owns pressure-gradient display settings, plus optional `relative_pressure_*` overrides for the reconstructed relative-pressure map.
- `vortex.json` owns spatial smoothing and valid-support erosion for vorticity, Q-criterion, and swirling-strength calculations.
- `planes.json -> render` owns GUI plane color and opacity. `video_exporting.json -> plane_video` owns plane-video skeleton, plane size, plane color, plane opacity, and label styling.
- `streamlines.json -> render` owns streamline display range; `clim: null` automatically uses the all-phase P99 segmented velocity so isolated dual-VENC outliers do not flatten the useful color range, while two numeric limits select a fixed range. Metric-specific `bar_cfg` values can still override the shared GUI colorbar defaults and are also used by offline videos.
- `pathlines.json` owns GUI pathline launch, color, tube, and temporal-cache defaults. It is separate from `streamlines.json`, which remains responsible for live streamline and video settings.
- `colorbar.json` owns the shared GUI colorbar visibility, size, position, and font defaults.
- `derived.json` is now just a legacy compatibility placeholder.
- `video_exporting.json` stays the central place for shared video sizing, rotation, camera, and output toggles.

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

### Build the Material for MkDocs site

Install the documentation extra and start a local site:

```bash
pip install -e ".[docs]"
mkdocs serve
```

The site entry point is `mkdocs.yml`; the landing page is `docs/index.md`.
The site provides Chinese and English user guides. The Aorta example uses a
real H5 path for demonstration but stores only derived documentation images,
not the medical source file.

The published site is intended to live at:

```text
https://associatedprimeideal.github.io/autoflow-docs
```

The workflow in `.github/workflows/docs.yml` builds the site and syncs it to
`AssociatedPrimeIdeal/AssociatedPrimeIdeal.github.io/autoflow-docs`. Configure
a repository secret named `DOCS_REPO_TOKEN` with write access to that Pages
repository before enabling the cross-repository publish job.

### English

**User Guide**

- [中文用户指南 / Chinese User Guide](docs/zh/user-guide.md)
- [English User Guide](docs/en/user-guide.md)
- [中文示例 / Chinese Example](docs/zh/example.md)
- [English Example](docs/en/example.md)
- [中文输入与输出 / Chinese Input & Output](docs/zh/input-output.md)
- [English Input & Output](docs/en/input-output.md)
- [Quickstart](docs/en/user/quickstart.md)
- [CLI](docs/en/user/cli.md)
- [GUI](docs/en/user/gui.md)
- [Python API](docs/en/user/python-api.md)
- [Inputs](docs/en/user/inputs.md)
- [Outputs](docs/en/user/outputs.md)
- [Troubleshooting](docs/en/user/troubleshooting.md)

**Feature Guides**

- [Background Phase Correction](docs/en/features/background-phase-correction.md)
- [Segmentation](docs/en/features/segmentation.md)
- [Phase Unwrapping](docs/en/features/phase-unwrapping.md)
- [Skeleton](docs/en/features/skeleton.md)
- [Graph and Paths](docs/en/features/graph-paths.md)
- [Planes](docs/en/features/planes.md)
- [Plane Metrics](docs/en/features/metrics.md)
- [Quality Control](docs/en/features/quality-control.md)
- [PWV](docs/en/features/pwv.md)
- [WSS, TKE, Pressure Gradient, and Relative Pressure](docs/en/features/wss-tke-pressure.md)
- [Vortex Kinematics](docs/en/features/vortex-kinematics.md)
- [Streamlines and Pathlines](docs/en/features/streamlines.md)
- [PC-MRA Volume Rendering](docs/en/features/pcmra-volume-rendering.md)
- [Videos](docs/en/features/videos.md)

**Developer Guide**

- [Architecture](docs/en/developer/architecture.md)
- [Feature-to-Code Map](docs/en/developer/feature-to-code-map.md)
- [Change Recipes](docs/en/developer/change-recipes.md)
- [Config System](docs/en/developer/config-system.md)
- [Testing](docs/en/developer/testing.md)

## Current Status Highlights

- input loading supports legacy complex H5, normalized H5, real-valued `img[..., 0:4]` as `mag + flow_xyz`, complex-valued `img[..., 0:4]` as legacy complex input, and direct DICOM directories
- H5 loader keys are matched case-insensitively and can be resolved from a single nested case group instead of only the file root
- loader output is normalized around `LoadedCase` with required `mag`, `flow`, `resolution`, `origin`, `venc`, and `rr`
- grouped multi-label segmentations can be reduced from 4D labels to 3D by time majority vote, cleaned per label, filtered per group with the skeleton connected-component rule before skeletonization, and rendered back as grouped skeleton, graph, path, plane, and pathline objects
- TKE is optional; mag/flow-only inputs must not synthesize fake TKE
- auto segmentation is currently executable in both CLI and GUI when the nnUNet backend and model folder are available
- the GUI Browser can show and hide a whole segmentation group at once, and group title colors come from `configs/labels.json`
- the GUI Browser supports Ctrl/Shift multi-selection of planes; the main `Pathlines` action runs all planes, while a plane context menu can target one plane or the selected subset
- after `Compute PWV` runs, the GUI uses one `Analysis` dock for PWV, plane cardiac-phase curves, and path/branch internal consistency, and still exposes PWV planes as one `PWV planes` browser item
- GUI `Run All` is stage-scoped: Centerline runs skeleton, graph, and planes; Hemodynamics runs plane metrics, derived metrics, live streamlines, and all-plane pathlines. `Compute PWV` remains explicit.
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
