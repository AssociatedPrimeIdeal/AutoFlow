# Quickstart

## Goal
Use this page when you want the shortest path to a successful AutoFlow run.

## Install

Full local install for GUI, automatic segmentation, and tests:

```bash
pip install -e ".[gui,test]"
```

Install the learned PUDIP-Flow and GUST-Flow phase-unwrapping backends as well:

```bash
pip install -e ".[gui,test,pu]"
```

To install every optional group in one step, use `pip install -e ".[all]"`.

CLI-only install:

```bash
pip install .
```

For DICOM directories, initialize `third_party/4DFlow_Dicom2H5` and install `.[dicom]` before importing; conversion saves a reusable H5. H5-only analysis needs no converter.

Both normal and editable installs include the bundled nnUNet final checkpoint and its required metadata. The `gui` extra installs both the GUI and nnUNet inference dependencies; there is no separate automatic-segmentation extra.

To enable the optional SpatioTemporal Labeler `v0.4.7` exchange workflow, initialize the pinned source submodule and install its extra:

```bash
git submodule update --init --recursive
pip install -e ".[gui,labeler]"
```

Known working environment in this repo (activate before CLI, GUI or tests):

```bash
conda activate autoflow311
python -m pytest tests/test_smoke_phantoms.py tests/test_pressure_gradient_phantom.py -q
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
6. requested WSS/TKE/pressure fields are prepared before derived plane sampling and reused in volume export; PWV runs after plane metrics
7. only export videos when requested

Typical outputs under `./results/demo/<case_name>/`:

- multi-group H5 files create one output folder per H5 data-group path, for example `stem__StudyA_Series1`
- `planes.json`
- `planes.h5`
- `plane_positions.json`
- `plane_metrics.json`
- `plane_qc.json`
- `quality_report.json`
- `summary.json`
- source H5 `segmask` is updated in place for reuse, plus `*_auto_segmentation.nii.gz` and `*_auto_segmentation_feature_*.nii.gz` when auto segmentation runs on an H5 input

## Run One DICOM Root From The CLI

```bash
autoflow-run /path/to/dicom_root --output-dir ./results/dicom_batch
```

## Open The GUI

```bash
autoflow-gui
```

Typical GUI flow:

1. `File > Load H5` or `File > Load DICOM` (the single DICOM action directly below `Load H5`; converts to a new H5 first)
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
  --plane-mode distance \
  --cross-section-dist 15 \
  --start-dist 5 \
  --end-dist 0
```

### Default center fixed-step planes

Generated planes default to three symmetric planes at 25% of the usable
centerline length on either side of the center (`fixed_step`, `center`, `both`).
The segmentation filter is enabled by default; disable it with
`--no-segmentation-filter` when working with a binary mask.

### Auto segmentation when no segmentation is present

```bash
autoflow-run ./data/demo_data.h5 \
  --output-dir ./results/demo \
  --autoseg
```

### Opt in to extra metrics and videos

```bash
autoflow-run ./data/demo_data.h5 \
  --output-dir ./results/demo \
  --with pwv,wss,tke,pg,vortex \
  --video plane,wss,tke,pg
```

`vortex` computes whole-volume vorticity, Q-criterion, and swirling strength. It is not a plane-video family, so it is intentionally omitted from `--video`.

## Next Pages

- [CLI](cli.md)
- [GUI](gui.md)
- [Python API](python-api.md)
- [Inputs](inputs.md)
- [Outputs](outputs.md)
- [Troubleshooting](troubleshooting.md)

[Functional workflow](../features/index.md) explains each module. [Configuration parameters](parameters.md) explains all defaults.

## Optional Correction group

In the GUI, load in Input & QC, then use **Correction > Run All** for Background Correction → Noise Removal → Unwrap Phase → Generate PC-MRA. Default unwrap is LAP4D/none. Input loading itself creates no PC-MRA. Continue to Segmentation afterward.

```bash
autoflow-run case.h5 --correction --autoseg --no-cache-write
# PC-MRA generation alone, for a case with segmentation:
autoflow-run case.h5 --generate-pcmra
```

The left Noise Region layer reviews excluded voxels and is hidden by default; its binary maximum-intensity rendering does not accumulate opacity with depth. Noise Removal, alone or in Correction Run All, automatically enables the ortho viewer's **Noise mask** overlay for review. Its checkbox and opacity slider show only excluded voxels in red; zero opacity disables it. Right-side Content lists acquired fields and computed results only.
