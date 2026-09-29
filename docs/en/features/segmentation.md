# Feature: Segmentation

## Status

| Entry point | Status | Notes |
| --- | --- | --- |
| GUI | Supported | 3D/4D nnUNet generation plus original/imported review and external Labeler flows; threshold generation is not exposed in the GUI |
| CLI | Supported | `--autoseg` only runs when no segmentation is already loaded |
| Python API | Partial | supported through `AutoFlowConfig` and workspace state, not a standalone high-level editing API |

4D connected-component cleanup is supported in the GUI and pipeline.
For a 4D prediction, topology uses a 3D majority-vote mask while phase-wise
plane metrics and derived fields keep the original dynamic `XYZT` mask.
The 3D vote is used only for topology and plane placement; a plane's
segmentation label is selected from the corresponding 4D frame during metric
sampling.

## What It Does
Segmentation gives AutoFlow the lumen mask needed for skeletons, graphs, planes, plane metrics, WSS, streamlines, and pathlines.

## When To Use It
- use it whenever the input has no usable vessel mask
- use imported segmentation when the mask already exists externally
- use the GUI's 3D or 4D nnUNet mode when generating a model segmentation
- use automatic segmentation when nnUNet is available and you want the result cached back into the source H5 for reuse
- do not expect fake TKE reconstruction from segmentation alone

## Quick Use

### GUI
1. load a case
2. open the `Segmentation` workflow stage; loading never starts automatic segmentation
3. in the `Segmentation Parameters` panel, choose `3D` or `4D`, then set the model path, checkpoint, folds, device, and label map
4. click `Run Automatic Segmentation` to start nnUNet
5. wait for the auto-segmentation progress dialog to finish
6. click `Open in SpatioTemporal Labeler`; AutoFlow exports six exchange files one feature per progress step and opens the separately installed GPL-3.0 editor
7. in Labeler, save `segmentation.nii` with `Ctrl+S` before closing; AutoFlow detects the changed file and offers to apply it as the imported segmentation source

The right dock opens on `Segmentation`. Its first `Source` section contains active-source switching, visibility, opacity, provenance, `Import...`, and `Save...`; its `Edit` section retains the automatic-run shortcut, review, and cleanup actions. Segmentation generation settings are kept with the other workflow parameters in the main window. The top menu bar does not duplicate these commands in a separate `Segmentation` menu.

The right ortho viewer's `Overlay` slider controls the opacity of the
segmentation label overlay on the `Content` slices. The 3-D Browser has a
separate per-object opacity control for segmentation surfaces and other scene
layers such as planes and streamlines.

During a manual stroke, only the active slice overlay is redrawn. Linked views refresh when the stroke ends, and the label browser uses incrementally maintained counts instead of repeatedly scanning the complete XYZT label array.

### CLI

```bash
autoflow-run case.h5 --output-dir results/case --autoseg
```

For the default temporal Dataset7020 model, select the 4D backend and leave the
model and checkpoint settings at `auto`. `single` uses `fold_all` (or the first
available fold); `all` averages all available folds, including a future
five-fold export:

When both `fold_all` and numeric folds are present, `single` deliberately keeps
the full-data `fold_all` branch while `all` selects the numeric folds for the
cross-validation ensemble.

```bash
autoflow-run case.h5 --output-dir results/case --autoseg \
  --autoseg-backend nnUNet4D \
  --autoseg-model auto --autoseg-checkpoint auto \
  --autoseg-folds single
```

For one static Dataset7010 prediction copied to every cardiac phase:

```bash
autoflow-run case.h5 --output-dir results/case --autoseg \
  --autoseg-backend nnUNet \
  --autoseg-model auto --autoseg-checkpoint auto \
  --autoseg-folds single
```

Use `--autoseg-folds all` (or `0,1,2,3,4`) when five trained folds are present:

```bash
autoflow-run case.h5 --output-dir results/case --autoseg \
  --autoseg-backend nnUNet4D --autoseg-folds all
```

For a cold timing run that ignores embedded `corr` and `seg` and leaves the H5
unchanged, add `--force-recompute-corr --force-recompute-seg
--ignore-embedded-segmentation --no-cache-write`.

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
| embedded `segmask`, `segmentation`, or `seg` | optional | original segmentation source | legacy or grouped H5 |
| external segmentation file | optional | imported segmentation source | `.h5`, `.npy`, `.npz`, `.nii`, or `.nii.gz` |

## Parameters

| Parameter | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `mode` | string | `auto` | workspace/API | internal source mode; the GUI's Mode selector chooses `auto_backend` | `autoflow/core/models.py` |
| `threshold_scalar` | string | `pcmra` | configs/API | scalar used by the non-GUI threshold backend | `autoflow/algorithms/segmentation.py` |
| `threshold_value` | object | manual `10%..100%` percent window | `configs/segmentation.json` or API | threshold mode and values for the non-GUI backend | `autoflow/algorithms/segmentation.py` |
| `threshold_keep_largest_cc` | bool | `True` | `configs/segmentation.json` or API | keep largest connected component for the non-GUI backend | `autoflow/algorithms/segmentation.py` |
| `threshold_min_component_volume_mm3` | float | `0.0` | `configs/segmentation.json` or API | remove components below threshold for the non-GUI backend | `autoflow/algorithms/segmentation.py` |
| `threshold_closing` | bool | `True` | `configs/segmentation.json` or API | morphological closing for the non-GUI backend | `autoflow/algorithms/segmentation.py` |
| `threshold_opening` | bool | `False` | `configs/segmentation.json` or API | morphological opening for the non-GUI backend | `autoflow/algorithms/segmentation.py` |
| `auto_backend` / `--autoseg-backend` | string | `nnUNet4D` in the shipped config | Segmentation Parameters panel or CLI/API | automatic backend; `nnUNet4D` consumes temporal channels and returns XYZT labels; choose `nnUNet` for static 3D models | `autoflow/ui/app.py`, `autoflow/algorithms/segmentation.py` |
| `auto_model` / `--autoseg-model` | path or `auto` | `auto` | Segmentation Parameters panel or CLI/API | GUI presets expose the detected Dataset7010 3D or Dataset7020 4D model; `auto` selects the backend-specific default; explicit 4D values also accept an orchestration `.sh`, while 3D accepts a model folder | `autoflow/algorithms/segmentation.py`, `autoflow/ui/app.py` |
| `auto_checkpoint` / `--autoseg-checkpoint` | string or `auto` | `auto` | Segmentation Parameters panel or CLI/API | `auto` selects `checkpoint_final.pth` for Dataset7010 and `checkpoint_best.pth` for Dataset7020 | `autoflow/ui/app.py`, `autoflow/algorithms/segmentation.py` |
| `auto_folds` / `--autoseg-folds` | string | `single` | Segmentation Parameters panel or CLI/API | choose `single`, `all`/`ensemble`, or a comma-separated fold list | `autoflow/ui/app.py`, `autoflow/algorithms/segmentation.py` |
| `auto_device` / `--autoseg-device` | string | `auto` | Segmentation Parameters panel or CLI/API | choose the inference device; CUDA also enables GPU input and probability-export resampling by default | `autoflow/ui/app.py`, `autoflow/algorithms/segmentation.py` |
| `auto_label_map` / `--autoseg-label-map` | JSON | empty | Segmentation Parameters panel or CLI/API | remap predicted labels | `autoflow/ui/app.py`, `autoflow/algorithms/segmentation.py` |
| `AUTOFLOW_NNUNET4D_GROUPED` | environment string | `auto` | process environment | control grouped in-process temporal preprocessing; `0`/`false`/`subprocess` forces the standard subprocess path, while `1` enables grouped mode (the GUI overrides this to stay isolated) | `autoflow/algorithms/segmentation.py` |
| `AUTOFLOW_NNUNET_GPU_PREPROCESSING` | environment string | `cuda` | process environment | GPU input resampling is the default when inference uses CUDA; use `0`, `false`, `off`, or `cpu` to keep nnUNet input resampling on CPU | `autoflow/algorithms/segmentation.py` |
| `cleanup_4d_components` | bool | `False` | Segmentation dock / workspace | enable per-label, per-frame connected-component cleanup before majority voting | `autoflow/core/models.py`, `autoflow/algorithms/preprocess.py` |
| `cleanup_4d_mode` | string | `absolute` | Segmentation dock / workspace | remove below volume or keep largest per label and frame | `autoflow/algorithms/preprocess.py` |
| `cleanup_4d_min_volume_mm3` | float | `50.0` | Segmentation dock / workspace | minimum physical component volume for absolute mode | `autoflow/algorithms/preprocess.py` |
| `colorbar.show` | bool | `True` | `configs/colorbar.json` | show or hide the shared 3D GUI colorbar used by segmentation labels | `autoflow/ui/viewer.py` |
| `colorbar.bar_cfg` | object | built-in default | `configs/colorbar.json` | configure the shared 3D GUI colorbar placement and fonts | `autoflow/ui/viewer.py` |

## Outputs

| Output file or object | Created when | Meaning |
| --- | --- | --- |
| active segmentation source in workspace | every segmentation change | current segmentation used by downstream steps |
| `*_threshold_segmentation.h5` | threshold segmentation succeeds | reusable threshold sidecar |
| source input H5 `segmask` dataset | auto segmentation succeeds on an H5 input | reusable auto segmentation cache written back into the source H5 by both GUI and CLI flows |
| `*_auto_segmentation.nii.gz` | auto segmentation succeeds | saved predicted segmentation NIfTI for review and reuse in the same `HF/AP/RL` geometry convention used by the nnUNet training/export scripts |
| `*_auto_segmentation_feature_*.nii.gz` | 3D auto segmentation succeeds | saved nnUNet feature-channel NIfTI inputs used for the prediction |
| saved H5, NPY, or NPZ chosen by user | dock `Source -> Save...` | manual export of active segmentation |
| saved NIfTI chosen by user | dock `Source -> Save...` or external-editor exchange | label sequence for external review/editing |
| provenance metadata | segmentation source changes | backend, model, save provenance, requested/actual preprocessing device, resampler, and GPU fallback details |
| imported `int16 XYZT` label source | Labeler result is applied | edited copy committed to `imported`; embedded `original` remains unchanged |

## Limitations
- automatic backends are `nnUNet` (3D/static) and `nnUNet4D` (Dataset7020 temporal-channel model)
- the `nnUNet` backend predicts one 3D label volume from Dataset7010's 12 summary channels and broadcasts that unchanged volume across the input time count
- the `nnUNet4D` backend predicts one label volume per frame from Dataset7020's 37 global and temporal channels and returns the stacked dynamic `XYZT` labels
- automatic segmentation requires loaded `mag` and `flow`
- `auto` or an empty model setting selects the matching local Dataset7010/Dataset7020 profile when available; static 3D falls back to the model bundled in the installed package, and the legacy Dataset7020 orchestration script remains the 4D fallback
- pip wheels and the Windows standalone build include only `fold_all/checkpoint_final.pth` plus `dataset.json` and `plans.json`; training logs, debug output, `checkpoint_best.pth`, and `checkpoint_latest.pth` are not packaged
- the nnUNet model was trained with `HF/AP/RL` feature geometry, so AutoFlow performs one required conversion from its normalized `LR/AP/FH` arrays before inference and restores the predicted labels once afterward; removing either conversion would change model inputs or downstream geometry
- when inference uses CUDA, AutoFlow creates a temporary model plan that uses nnUNet's `resample_torch_fornnunet` implementation for floating-point input channels; the source model and its `plans.json` are not modified
- CUDA input and prediction-probability resampling use nnUNet's linear torch interpolation instead of the default cubic SciPy interpolation; label export remains unchanged
- if the CUDA resampling prediction attempt fails, AutoFlow automatically retries once with the unmodified model plan and CPU input resampling; provenance records the requested and actual preprocessing devices and the fallback reason
- `nnUNet4D` writes all cardiac phases and loads the model once per case. The GUI explicitly runs nnUNet in an isolated subprocess so a native CUDA failure cannot terminate the Qt application. Set `AUTOFLOW_NNUNET4D_GROUPED=1` (or pass `grouped_preprocessing=True` from a controlled Python job) to use grouped in-process preprocessing outside the GUI; if that helper is unavailable, AutoFlow falls back to the standard subprocess and records the reason in provenance
- the supplied `run_7020_4d_full_ssd_20260824.sh` is a training/batch orchestration script. AutoFlow parses its Dataset7020 result location when used as `--autoseg-model`; it does not rerun training during a case analysis
- when that script contains an existing `PYTHON_BIN` executable, AutoFlow uses it for the nnUNet subprocess so the custom Dataset7020 trainer/resampler remains importable; otherwise it uses the active AutoFlow Python environment
- `auto_folds=all` delegates fold averaging to nnUNet. Use `single` while validating a checkpoint, then switch to `all` when `fold_0`...`fold_4` are present
- GUI auto segmentation runs in a background worker and uses the same closeable window-modal progress dialog as other long-running GUI tasks; closing it hides progress permanently for that run without cancelling inference, foreground-label validation, H5 cache writing, or workspace refresh
- AutoFlow uses the nnUNet predictor available in the active environment or on its existing `PATH`; it does not search or switch to another Conda environment
- an all-background nnUNet prediction is reported as a failure instead of being applied as an invisible segmentation
- CLI auto segmentation only runs when the loaded case has no segmentation
- `Run All` in the GUI does not auto-start segmentation
- TKE availability is independent of segmentation availability
- the optional SpatioTemporal Labeler bridge launches a separate process and exchanges six uncompressed 4D NIfTI files; it is not embedded into the AutoFlow Qt process. It uses the checked-out `third_party/SpatioTemporalLabeler` source with AutoFlow's Python/Qt/VTK runtime, so install `pip install ".[gui,labeler]"` in that environment
- when AutoFlow is running through forwarded X11, it clears AutoFlow's EGL/off-screen VTK environment variables only for the separate Labeler process. Labeler's unmodified embedded VTK view then uses native X11/GLX rendering instead of opening with a blank 3D panel
- the exchange directory is stable for a loaded case. When its source path, geometry, and frame count still match, the saved feature files and `segmentation.nii` are reused on the next open so Labeler edits remain available
- the export progress dialog has one step each for `mag`, `flow_x`, `flow_y`, `flow_z`, `pcmra`, and `segmentation`; export intentionally runs in the GUI thread
- automatic re-import requires saving the original `segmentation.nii` in Labeler; `Save As` to another directory requires the normal `Import...` action
- the pinned upstream `v0.4.7` source is available as the `third_party/SpatioTemporalLabeler` git submodule; install the integration only with `pip install .[gui,labeler]`

## Where To Change Code

| Change you want | Edit here | Also check | Tests |
| --- | --- | --- | --- |
| add a new segmentation source | `autoflow/core/models.py`, `autoflow/ui/app.py` | `autoflow/ui/segmentation.py`, `autoflow/algorithms/segmentation.py` | `tests/test_smoke_phantoms.py` |
| change threshold behavior | `autoflow/algorithms/segmentation.py`, `autoflow/ui/app.py` | `autoflow/core/models.py` | `tests/test_smoke_phantoms.py` |
| change nnUNet auto segmentation | `autoflow/algorithms/segmentation.py` | `autoflow/processing.py`, `autoflow/ui/app.py` | manual verification plus smoke and phantom regression only |
| change segmentation dock behavior | `autoflow/ui/segmentation.py`, `autoflow/ui/app.py` | `autoflow/core/models.py` | `tests/test_smoke_phantoms.py` |
| change external editor exchange | `autoflow/algorithms/segmentation.py`, `autoflow/ui/app.py` | `autoflow/ui/segmentation.py` | NIfTI round-trip plus GUI manual verification |

## Tests

- `~/miniconda3/envs/ryy/bin/python -m pytest tests/test_smoke_phantoms.py -q`
- `~/miniconda3/envs/ryy/bin/python -m pytest tests/test_pressure_gradient_phantom.py -q`
- segmentation-specific changes beyond this retained regression suite require manual verification

## Common Problems

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| auto segmentation says the bundled model folder is missing | the installed package does not contain the default model | reinstall a complete build or set the model folder explicitly |
| auto segmentation finishes without a result | inference failed or returned only background | read the failure dialog details; fix the current environment or verify that the case is suitable for the model |
| segmentation-dependent steps are skipped | no active segmentation | choose, import, threshold, or auto-generate a segmentation first |
| Labeler does not open | optional `labeler` package is missing from AutoFlow's Python environment | install `pip install ".[gui,labeler]"` in the environment used to launch `autoflow-gui` |
| Labeler closes unexpectedly | Labeler process or graphics-stack error | inspect the AutoFlow log for `[Labeler stderr]` lines captured from the child process |
| Labeler result is not imported | `segmentation.nii` was not saved | use `Ctrl+S` on the original exchange segmentation before closing; `Save As` requires `Import...` |
