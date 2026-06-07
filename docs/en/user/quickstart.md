# Quickstart

## Goal
Use this page when you want the shortest path to a successful AutoFlow run.

## Install

Full local install for GUI and tests:

```bash
pip install -e ".[gui,test]"
```

CLI-only install:

```bash
pip install .
```

Known working test environment in this repo:

```bash
~/miniconda3/envs/ryy/bin/python -m pytest tests/test_smoke_phantoms.py tests/test_pressure_gradient_phantom.py -q
```

## Run One H5 Case From The CLI

```bash
autoflow-run ./data/demo_data.h5 --output-dir ./results/demo
```

What this does:

1. load data
2. generate skeleton
3. generate graph
4. generate planes
5. calculate plane metrics
6. calculate derived metrics
7. export any enabled videos

Typical outputs under `./results/demo/<case_name>/`:

- `planes.json`
- `plane_positions.json`
- `plane_metrics.json`
- `plane_qc.json`
- `summary.json`

## Run One DICOM Root From The CLI

```bash
autoflow-run /path/to/dicom_root --output-dir ./results/dicom_batch
```

## Open The GUI

```bash
autoflow-gui
```

Typical GUI flow:

1. `File > Open H5` or `File > Import DICOM Directory`
2. inspect the case in the 3D view and ortho viewer
3. if needed, choose or generate segmentation
4. click `Run All`
5. inspect planes, metrics, and derived volumes

## Run From Python

```python
from autoflow import AutoFlowConfig, run_batch

config = AutoFlowConfig(
    inputs=["./data/demo_data.h5"],
    output_dir="./results/demo",
)
results, last_case_out = run_batch(config)
```

## Common First Steps

### Batch planes by distance

```bash
autoflow-run ./data/demo_data.h5 \
  --output-dir ./results/demo \
  --plane-by-distance \
  --cross-section-dist 15 \
  --start-dist 5 \
  --end-dist 0
```

### Auto segmentation when no segmentation is present

```bash
autoflow-run ./data/demo_data.h5 \
  --output-dir ./results/demo \
  --autoseg
```

### Export videos

```bash
autoflow-run ./data/demo_data.h5 \
  --output-dir ./results/demo \
  --plane-video \
  --wss-video \
  --streamlines-video
```

## Next Pages

- [CLI](cli.md)
- [GUI](gui.md)
- [Python API](python-api.md)
- [Inputs](inputs.md)
- [Outputs](outputs.md)
- [Troubleshooting](troubleshooting.md)
