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

The nnU-Net framework citation and distinction from local checkpoint/4-D validation are recorded in [Scientific references: segmentation](../references/index.md#segmentation).

## When To Use It
- use it whenever the input has no usable vessel mask
- use imported segmentation when the mask already exists externally
- use the GUI's 3D or 4D nnUNet mode when generating a model segmentation
- use automatic segmentation when nnUNet is available and you want the result cached back into the source H5 for reuse
- do not expect fake TKE reconstruction from segmentation alone

## Quick Use

### GUI

The GUI respects `configs/segmentation.json -> write_auto_cache`: false preserves the source H5 while retaining prediction sidecars for review.
1. load a case
2. open the `Segmentation` workflow stage; loading never starts automatic segmentation
3. in the `Segmentation Parameters` panel, choose `3D` or `4D`, then set the model path, checkpoint, folds, device, and label map
4. click `Run Automatic Segmentation` to start nnUNet
5. wait for the auto-segmentation progress dialog to finish
6. click `Open in SpatioTemporal Labeler`; AutoFlow prepares six exchange files in background workers and opens the separately installed GPL-3.0 editor
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

See [segmentation parameters](../user/parameters.md#segmentation), [labels parameters](../user/parameters.md#labels), [CLI flags](../user/cli-parameters.md), and [API fields](../user/api-parameters.md) for complete type/default/unit/effect/owner tables. Dictionary controls are expanded in [Structured parameters](../user/parameter-schemas.md).

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
| `spatiotemporal_labeler/<case>/exchange.json` | Labeler exchange is prepared | source/geometry metadata, image digests, and active seed digest for safe reuse |
| `segmentation.previous.<timestamp_ns>.nii` | a different or legacy exchange mask must be replaced | recoverable copy of previous Labeler work |

Changing the active segmentation compares the processed masks and skeleton/group configuration. Identical processed 4D labels retain geometry and numerical results. Changed temporal labels with identical voted 3D labels and per-group topology retain centerlines and planes, while clearing all segmentation-dependent metrics, PWV and trajectories for explicit recomputation. Changed 3D topology or preprocessing/group configuration clears downstream geometry. No phase-local pressure or PWV shortcut is applied.

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
- GUI automatic segmentation uses application-modal task progress. × requests cancellation of its inference child process; the window stays open and other actions stay locked until the task stops. A cancelled prediction is not applied. Animated dots and prediction-file counts show activity; an unchanged count is not fabricated inference progress.
- AutoFlow uses the nnUNet predictor available in the active environment or on its existing `PATH`; it does not search or switch to another Conda environment
- an all-background nnUNet prediction is reported as a failure instead of being applied as an invisible segmentation
- CLI auto segmentation only runs when the loaded case has no segmentation
- `Run All` in the GUI does not auto-start segmentation
- TKE availability is independent of segmentation availability
- the optional SpatioTemporal Labeler bridge launches a separate process and exchanges six uncompressed 4D NIfTI files; it is not embedded into the AutoFlow Qt process. It uses the checked-out `third_party/SpatioTemporalLabeler` source with AutoFlow's Python/Qt/VTK runtime, so install `pip install ".[gui,labeler]"` in that environment
- when AutoFlow is running through forwarded X11, it clears AutoFlow's EGL/off-screen VTK environment variables only for the separate Labeler process. Labeler's unmodified embedded VTK view then uses native X11/GLX rendering instead of opening with a blank 3D panel
- the exchange directory is stable for a loaded case. Image reuse checks source metadata, geometry, and digests of the actual magnitude/flow arrays. An unchanged active seed retains Labeler's saved working mask; applying that exact edited mask also retains its file and label definitions. A different active segmentation refreshes only `segmentation.nii`, avoiding a full image export and stale labels after another automatic run
- Labeler export progress advances as files finish. Export and digest calculation run outside the GUI thread, with at most two concurrent NIfTI writers; no worker accesses Qt widgets. × stops at a safe digest/file boundary and waits for active writes; the editor is not launched after cancellation. Files are replaced only after complete writes. The five features and segmentation remain uncompressed `.nii`, with unchanged float32/int16 values and spatial affines.
- a saved Labeler result is offered for import only after active task/dialog locks are released. Results from a previous loaded case are retained on disk instead of being applied to a replacement case
- before replacing a different or legacy exchange mask, AutoFlow preserves it as `segmentation.previous.<timestamp_ns>.nii`; saved manual revisions remain recoverable. Existing schema-2 workspaces refresh their feature manifest once
- GUI 4D nnUNet preparation encodes each distinct feature map once and links the usual per-frame channel filenames to it, falling back to byte copies when hard links are unavailable. For 20 phases and 37 channels this encodes 112 maps rather than 740. Channel order, per-sample cropping/normalization, folds, checkpoint, sliding-window settings, TTA, and precision are unchanged; the CLI grouped path remains separate
- the standard Python 4D subprocess moves unpadded tensors to CUDA before nnUNet's original padding and inference when enough memory is free, and skips an inspected preprocessing copy whose pinned result is discarded. CPU inference, CPU accumulation, low-memory devices, and CUDA allocation failures use the original padding path. Model arithmetic and labels are unchanged; the standalone frozen predictor retains its existing path
- independent CUDA launches can exhibit small label differences in the existing nnUNet runtime. Equality validation must fix the input and execution settings; see the [segmentation and Labeler performance validation](../developer/segmentation-performance.md)
- automatic re-import requires saving the original `segmentation.nii` in Labeler; `Save As` to another directory requires the normal `Import...` action
- the pinned upstream `v0.4.7` source is available as the `third_party/SpatioTemporalLabeler` git submodule; install the integration only with `pip install .[gui,labeler]`

## Where To Change Code

| Change you want | Edit here | Also check | Tests |
| --- | --- | --- | --- |
| add a new segmentation source | `autoflow/core/models.py`, `autoflow/ui/app.py` | `autoflow/ui/segmentation.py`, `autoflow/algorithms/segmentation/io.py` | `tests/test_smoke_phantoms.py` |
| change threshold behavior | `autoflow/algorithms/segmentation/threshold.py`, `autoflow/ui/app.py` | `autoflow/core/models.py` | `tests/test_smoke_phantoms.py` |
| change static nnUNet inference | `autoflow/algorithms/segmentation/nnunet_static.py`, `autoflow/nnunet_runtime.py` | `autoflow/processing.py`, `autoflow/ui/app.py` | manual verification plus smoke and phantom regression only |
| change temporal/grouped nnUNet inference | `autoflow/algorithms/segmentation/nnunet_temporal.py`, `autoflow/algorithms/segmentation/nnunet_grouped.py` | `autoflow/algorithms/segmentation/channels.py` | smoke/phantom coverage and manual inference checks |
| change models, channels or subprocess behavior | `autoflow/algorithms/segmentation/models.py`, `autoflow/algorithms/segmentation/channels.py`, `autoflow/algorithms/segmentation/runtime.py` | static and temporal inference modules | `tests/test_smoke_phantoms.py` |
| change segmentation import/export | `autoflow/algorithms/segmentation/io.py`, `autoflow/algorithms/segmentation/_common.py` | `autoflow/ui/labeler_exchange.py` | `tests/test_smoke_phantoms.py` |
| change segmentation dock behavior | `autoflow/ui/segmentation.py`, `autoflow/ui/app.py` | `autoflow/core/models.py` | `tests/test_smoke_phantoms.py` |
| change external editor exchange | `autoflow/ui/labeler_exchange.py`, `autoflow/ui/app.py` | `autoflow/algorithms/segmentation/io.py`, `third_party/SpatioTemporalLabeler/src/spatiotemporal_labeler/io/nrrd_sequence.py` | NIfTI round-trip plus GUI manual verification |

## Tests

- `conda activate autoflow311`
- `pytest tests/test_smoke_phantoms.py tests/test_pressure_gradient_phantom.py -q`
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
