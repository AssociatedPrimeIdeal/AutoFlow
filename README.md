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
- plane metrics, WSS, optional TKE, pressure-gradient, streamlines, and GUI pathlines
- offline result export to JSON, NPZ, H5, and video

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

This runs the standard batch order:

1. load data
2. generate skeleton
3. generate graph
4. generate planes
5. calculate plane metrics
6. calculate derived metrics
7. export enabled videos

Typical outputs under `./results/demo/<case_name>/`:

- `planes.json`
- `plane_positions.json`
- `plane_metrics.json`
- `plane_qc.json`
- `summary.json`

GUI:

```bash
autoflow-gui
```

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
- `planes.json`
- `streamlines.json`
- `derived.json`
- `segmentation.json`
- `rendering.json`

Important grouped-mask note:

- `skeleton.json` owns the default label map, label-group definitions, browser colors, and per-group preprocessing used by grouped multi-label vessel workflows.

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

Detailed documentation now lives under `docs/` in English and Chinese.

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
- [WSS, TKE, Pressure Gradient](docs/en/features/wss-tke-pressure.md)
- [Streamlines and Pathlines](docs/en/features/streamlines.md)
- [Videos](docs/en/features/videos.md)

**Developer Guide**

- [Architecture](docs/en/developer/architecture.md)
- [Feature-to-Code Map](docs/en/developer/feature-to-code-map.md)
- [Change Recipes](docs/en/developer/change-recipes.md)
- [Config System](docs/en/developer/config-system.md)
- [Testing](docs/en/developer/testing.md)

### 中文

**用户文档**

- [快速开始](docs/zh/user/quickstart.md)
- [CLI 指南](docs/zh/user/cli.md)
- [GUI 指南](docs/zh/user/gui.md)
- [Python API](docs/zh/user/python-api.md)
- [输入](docs/zh/user/inputs.md)
- [输出](docs/zh/user/outputs.md)
- [故障排查](docs/zh/user/troubleshooting.md)

**功能文档**

- [分割](docs/zh/features/segmentation.md)
- [骨架](docs/zh/features/skeleton.md)
- [图与路径](docs/zh/features/graph-paths.md)
- [平面](docs/zh/features/planes.md)
- [平面指标](docs/zh/features/metrics.md)
- [WSS / TKE / 压力梯度](docs/zh/features/wss-tke-pressure.md)
- [流线与路径线](docs/zh/features/streamlines.md)
- [视频](docs/zh/features/videos.md)

**开发者文档**

- [架构](docs/zh/developer/architecture.md)
- [功能到代码映射](docs/zh/developer/feature-to-code-map.md)
- [变更配方](docs/zh/developer/change-recipes.md)
- [配置系统](docs/zh/developer/config-system.md)
- [测试](docs/zh/developer/testing.md)

## Current Status Highlights

- input loading supports legacy complex H5, normalized H5, and direct DICOM directories
- loader output is normalized around `LoadedCase` with required `mag`, `flow`, `resolution`, `origin`, `venc`, and `rr`
- grouped multi-label segmentations can be reduced from 4D labels to 3D by time majority vote, cleaned per label, merged by config groups, and rendered back as grouped skeleton, graph, path, plane, and pathline objects
- TKE is optional; mag/flow-only inputs must not synthesize fake TKE
- auto segmentation is currently executable in both CLI and GUI when the nnUNet backend and model folder are available
- the GUI Browser can show and hide a whole segmentation group at once, and group title colors come from `configs/skeleton.json`
- `Run All` in the GUI runs `Generate Skeleton -> Generate Graph -> Generate Planes -> Calculate && Save Metrics -> WSS / TKE / Pressure Gradient`
- offline videos are driven mainly by CLI and Python batch workflows

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
