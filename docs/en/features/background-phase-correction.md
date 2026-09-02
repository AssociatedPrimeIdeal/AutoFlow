# Background Phase Correction

## Status

Supported. MSAC and WRLS+ARTO are available from the GUI, CLI, and Python API. WRLS+ARTO is the default; `wrls` remains an alias.

## What it does

Background phase correction estimates a smooth, time-invariant phase-offset field for each velocity direction and removes it before flow reconstruction. AutoFlow supports two methods:

- `msac` finds stationary tissue with a robust random-sample fit, then fits the requested polynomial correction.
- `wrls_arto` follows the OSU-MR weighted robust least-squares and automatic residual tissue optimization workflow: middle-FOV first-order initialization, L1 WRLS fitting, constrained three-component GMM exclusion, and repeated polynomial refitting.

For dual-venc legacy H5 inputs, low- and high-venc corrections are computed independently and concurrently before alias reconstruction.

## When to use it

Use background correction when static tissue has a non-zero velocity offset or a smooth spatial phase bias. WRLS+ARTO is the default; use MSAC when you need the established robust-sample fit for comparison.

Do not assume the two methods are interchangeable. They can estimate materially different correction fields and may change dual-venc alias decisions near wrapping boundaries.

## Quick use

### GUI

When an H5 case has no reusable correction cache, enable correction in the first prompt and select `MSAC` or `WRLS + ARTO` from the second prompt. The selected method is reflected in `Input / Background Correction -> Correction Method`. Existing caches skip both prompts and select their recorded method automatically.

### CLI

```bash
autoflow-run input.h5 --bgc --bgc-method msac
autoflow-run input.h5 --bgc --bgc-method wrls_arto
```

There is no correction-device flag. When WRLS+ARTO runs, AutoFlow automatically uses CUDA for its ARTO GMM stage if PyTorch reports a usable CUDA device; otherwise it falls back to CPU.

### Python API

```python
from autoflow import AutoFlowConfig, run_case

config = AutoFlowConfig(
    background_phase_correction=True,
    background_phase_method="wrls_arto",
)
result = run_case("input.h5", config=config)
```

## Inputs

The correction layer accepts legacy complex H5 data and normalized `mag + flow` inputs. Complex data uses one reference encode plus three velocity encodes. Normalized velocity inputs are converted to an equivalent synthetic complex representation for correction and converted back afterward.

WRLS+ARTO requires at least two time frames because it estimates temporal standard deviation. Its magnitude mask uses the configured fraction of each slice's maximum time-averaged reference magnitude.

## Parameters

| Parameter or flag | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `method`, `--bgc-method` | string | `wrls_arto` | `configs/loader.json`, GUI, CLI, Python API | choose `msac` or `wrls_arto` | `autoflow/algorithms/phase_correction.py` |
| `corr_fit_order`, `--bgc-fit-order` | int | `3` | `configs/loader.json`, GUI, CLI, Python API | polynomial order of the final correction | `autoflow/algorithms/phase_correction.py` |
| `threshold`, `--bgc-threshold` | float | `0.1` | `configs/loader.json`, GUI, CLI, Python API | MSAC stationary-tissue threshold in venc units | `autoflow/algorithms/phase_correction.py` |
| `wrls_lambda`, `--bgc-wrls-lambda` | float | `5.0` | `configs/loader.json`, CLI, Python API | L1 regularization strength | `autoflow/algorithms/phase_correction.py` |
| `wrls_magnitude_threshold`, `--bgc-wrls-magnitude-threshold` | float | `0.04` | `configs/loader.json`, CLI, Python API | per-slice reference-magnitude fraction | `autoflow/algorithms/phase_correction.py` |
| `wrls_mid_fov_fraction`, `--bgc-wrls-mid-fov-fraction` | float | `0.5` | `configs/loader.json`, CLI, Python API | middle in-plane FOV fraction for initialization | `autoflow/algorithms/phase_correction.py` |
| `wrls_mid_slice_fraction`, `--bgc-wrls-mid-slice-fraction` | float | `0.65` | `configs/loader.json`, CLI, Python API | middle through-plane fraction for initialization | `autoflow/algorithms/phase_correction.py` |
| `wrls_arto_iterations`, `--bgc-wrls-arto-iterations` | int | `2` | `configs/loader.json`, CLI, Python API | ARTO exclusion and refit count | `autoflow/algorithms/phase_correction.py` |
| `wrls_tau`, `--bgc-wrls-tau` | float | `3.0` | `configs/loader.json`, CLI, Python API | central-Gaussian inclusion width | `autoflow/algorithms/phase_correction.py` |
| `wrls_delta`, `--bgc-wrls-delta` | float | `2.0` | `configs/loader.json`, CLI, Python API | minimum side-Gaussian separation | `autoflow/algorithms/phase_correction.py` |
| `wrls_central_probability`, `--bgc-wrls-central-probability` | float | `0.5` | `configs/loader.json`, CLI, Python API | minimum central-Gaussian prior | `autoflow/algorithms/phase_correction.py` |
| `wrls_fista_iterations`, `--bgc-wrls-fista-iterations` | int | `5000` | `configs/loader.json`, CLI, Python API | maximum FISTA iterations per fit | `autoflow/algorithms/phase_correction.py` |
| `wrls_gmm_iterations`, `--bgc-wrls-gmm-iterations` | int | `1000` | `configs/loader.json`, CLI, Python API | maximum GMM EM iterations per ARTO pass | `autoflow/algorithms/phase_correction.py` |
| `force_recompute`, `--force-recompute-corr` | bool | `False` | `configs/loader.json`, CLI, Python API | ignore a compatible H5 correction cache | `autoflow/algorithms/data.py` |

## Outputs

The corrected flow becomes the loader's normalized `flow`. Writable H5 sources also receive a reusable `corr` dataset, or `corr_low` and `corr_high` for dual-venc data. Cache attributes record the correction method, algorithm version, fit order, source group, and every method-specific parameter used to compute the field.

The GUI retains correction fields as optional workspace arrays and exposes their LR/AP/FH components in the right-side Content selector (radians). The Corr entries remain disabled until a correction was applied or reused. Dual-venc data expose both normalized `Corr Low LR/AP/FH` and `Corr High LR/AP/FH` fields.

Runtime metadata records whether correction was applied, whether a cache was reused, stationary-voxel counts, and the compute device used by WRLS+ARTO. Device metadata is diagnostic only and is not a user setting.

## Limitations

- WRLS+ARTO needs at least two time frames.
- Automatic CUDA acceleration currently covers the dominant ARTO GMM stage; the small WRLS coefficient systems remain on CPU.
- PyTorch is optional. Missing CUDA support or a CUDA execution failure causes a transparent CPU fallback.
- A correction cache is reusable only when its shape and complete algorithm metadata match.
- Without a ground-truth phase-offset field, method comparison relies on static-tissue residuals and downstream measurement review.

## Where to change code

- Algorithm implementations and automatic CUDA fallback: `autoflow/algorithms/phase_correction.py`
- H5 cache validation, dual-venc concurrency, and cache writing: `autoflow/algorithms/data.py`
- Public configuration types: `autoflow/case_types.py`, `autoflow/api.py`, and `autoflow/config.py`
- CLI flags: `autoflow/cli.py`
- GUI enable and method prompts: `autoflow/ui/app.py`

## Tests

Smoke coverage is in `tests/test_smoke_phantoms.py`. It covers MSAC progress and cache reuse, WRLS polynomial recovery, method-specific cache isolation, and concurrent dual-venc correction. Run the supported suite with:

```bash
~/miniconda3/envs/ryy/bin/python -m pytest \
  tests/test_smoke_phantoms.py \
  tests/test_pressure_gradient_phantom.py -q
```

## Common problems

- WRLS+ARTO runs on CPU: confirm that the active Python environment has a CUDA-enabled PyTorch build and that `torch.cuda.is_available()` is true. No AutoFlow device option is required.
- A cache is recomputed: inspect the H5 correction attributes for a method, fit-order, version, or WRLS-parameter mismatch.
- The WRLS threshold cannot be edited in the GUI: `MSAC Threshold` is method-specific; WRLS tuning values are read from `configs/loader.json`.
- Dual-venc output changes substantially in a few voxels: review those locations near wrap boundaries because different background fields can change the selected alias branch.
