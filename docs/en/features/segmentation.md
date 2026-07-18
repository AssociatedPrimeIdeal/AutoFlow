# Feature: Segmentation

## Status

| Entry point | Status | Notes |
| --- | --- | --- |
| GUI | Supported | original, imported, threshold, auto, and manual edit flows |
| CLI | Supported | `--autoseg` only runs when no segmentation is already loaded |
| Python API | Partial | supported through `AutoFlowConfig` and workspace state, not a standalone high-level editing API |

## What It Does
Segmentation gives AutoFlow the lumen mask needed for skeletons, graphs, planes, plane metrics, WSS, streamlines, and pathlines.

## When To Use It
- use it whenever the input has no usable vessel mask
- use imported segmentation when the mask already exists externally
- use threshold segmentation for quick mask generation from `mag`, `pcmra`, or `pcmra_std`
- use automatic segmentation when nnUNet is available and you want the result cached back into the source H5 for reuse
- do not expect fake TKE reconstruction from segmentation alone

## Quick Use

### GUI
1. load a case
2. open `Segmentation > Configure Segmentation...`
3. choose `input`, `threshold`, or `auto`
4. apply the configuration and wait for the auto-segmentation progress dialog to finish when using `auto`
5. if needed, refine the result in the segmentation dock and click `Apply`

### CLI

```bash
autoflow-run case.h5 --output-dir results/case --autoseg
```

### Python API

```python
from autoflow import AutoFlowConfig, run_case

config = AutoFlowConfig(
    output_dir="./results/case",
    autoseg=True,
    autoseg_model="/path/to/model",
)
summary = run_case("case.h5", config=config)
```

## Inputs

| Input | Required | Meaning | Example |
| --- | --- | --- | --- |
| loaded `mag` | for threshold or auto | magnitude input for segmentation generation | normalized H5 or DICOM case |
| loaded `flow` | for auto | velocity channels used to prepare nnUNet inputs | normalized H5 or DICOM case |
| embedded segmentation | optional | original segmentation source | legacy H5 |
| external segmentation file | optional | imported segmentation source | `.h5`, `.npy`, `.npz` |

## Parameters

| Parameter | Type | Default | Where set | Effect |
| --- | --- | --- | --- | --- |
| `mode` | string | `input` | GUI segmentation config | choose `input`, `threshold`, or `auto` |
| `input_source` | string | `original` | GUI segmentation config | choose original or imported source |
| `threshold_scalar` | string | `pcmra` | `configs/segmentation.json` | scalar used for threshold segmentation |
| `threshold_value` | object | manual `10%..100%` percent window | `configs/segmentation.json` | threshold mode and values |
| `threshold_keep_largest_cc` | bool | `True` | `configs/segmentation.json` | keep largest connected component |
| `threshold_min_component_volume_mm3` | float | `0.0` | `configs/segmentation.json` | remove components below threshold |
| `threshold_closing` | bool | `True` | `configs/segmentation.json` | morphological closing |
| `threshold_opening` | bool | `False` | `configs/segmentation.json` | morphological opening |
| `auto_backend` / `--autoseg-backend` | string | `nnUNet` | GUI config or CLI/API | automatic backend |
| `auto_model` / `--autoseg-model` | path | bundled `autoflow/segmodel/nnUNetTrainerPartBalanced__nnUNetPlans__3d_fullres_iso1mm` in GUI config; empty string in CLI/API config until resolved | GUI config or CLI/API | nnUNet model folder |
| `auto_checkpoint` / `--autoseg-checkpoint` | string | `checkpoint_final.pth` | GUI config or CLI/API | checkpoint name |
| `auto_device` / `--autoseg-device` | string | `auto` | GUI config or CLI/API | choose device |
| `auto_label_map` / `--autoseg-label-map` | JSON | empty | GUI config or CLI/API | remap predicted labels |
| `edit_all_timepoints` | bool | `True` | segmentation dock | edit a 3D-all-frames mask instead of current-frame 4D editing |
| `colorbar.show` | bool | `True` | `configs/colorbar.json` | show or hide the shared GUI colorbar used by segmentation labels |
| `colorbar.bar_cfg` | object | built-in default | `configs/colorbar.json` | shared GUI colorbar placement and font settings used by segmentation labels |

## Outputs

| Output file or object | Created when | Meaning |
| --- | --- | --- |
| active segmentation source in workspace | every segmentation change | current segmentation used by downstream steps |
| `*_threshold_segmentation.h5` | threshold segmentation succeeds | reusable threshold sidecar |
| source input H5 `segmask` dataset | auto segmentation succeeds on an H5 input | reusable auto segmentation cache written back into the source H5 by both GUI and CLI flows |
| `*_auto_segmentation.nii.gz` | auto segmentation succeeds | saved predicted segmentation NIfTI for review and reuse in the same `HF/AP/RL` geometry convention used by the nnUNet training/export scripts |
| `*_auto_segmentation_feature_*.nii.gz` | auto segmentation succeeds | saved nnUNet feature-channel NIfTI inputs used for the prediction, written in the same `HF/AP/RL` geometry convention used by the nnUNet training/export scripts |
| saved H5, NPY, or NPZ chosen by user | `Save Active Segmentation...` | manual export of active segmentation |
| provenance metadata | segmentation source changes | backend, model, and save provenance |

## Limitations
- the implemented automatic backend is `nnUNet`
- automatic segmentation requires loaded `mag` and `flow`
- GUI auto segmentation runs in a background worker and shows a progress dialog until inference and H5 cache write finish when the input is H5
- CLI auto segmentation only runs when the loaded case has no segmentation
- `Run All` in the GUI does not auto-start segmentation
- TKE availability is independent of segmentation availability

## Where To Change Code

| Change you want | Edit here | Also check | Tests |
| --- | --- | --- | --- |
| add a new segmentation source | `autoflow/core/models.py`, `autoflow/ui/app.py` | `autoflow/ui/segmentation.py`, `autoflow/algorithms/segmentation.py` | `tests/test_smoke_phantoms.py` |
| change threshold behavior | `autoflow/algorithms/segmentation.py` | `autoflow/ui/segmentation.py` | `tests/test_smoke_phantoms.py` |
| change nnUNet auto segmentation | `autoflow/algorithms/segmentation.py` | `autoflow/processing.py`, `autoflow/ui/app.py` | manual verification plus smoke and phantom regression only |
| change segmentation dock behavior | `autoflow/ui/segmentation.py`, `autoflow/ui/app.py` | `autoflow/core/models.py` | `tests/test_smoke_phantoms.py` |

## Tests

- `~/miniconda3/envs/ryy/bin/python -m pytest tests/test_smoke_phantoms.py -q`
- `~/miniconda3/envs/ryy/bin/python -m pytest tests/test_pressure_gradient_phantom.py -q`
- segmentation-specific changes beyond this retained regression suite require manual verification

## Common Problems

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| auto segmentation says model folder is missing | no explicit model and no bundled default folder found | set the model folder explicitly |
| segmentation-dependent steps are skipped | no active segmentation | choose, import, threshold, or auto-generate a segmentation first |
| switching segmentation source is blocked | uncommitted segmentation edits exist | click `Apply` or `Cancel` first |
