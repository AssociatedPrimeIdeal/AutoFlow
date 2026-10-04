# Background Phase Correction

## Status

Supported. MSAC and WRLS+ARTO are available from the GUI, CLI, and Python API. WRLS+ARTO is the default; `wrls` remains an alias.

## What it does

Background phase correction estimates a smooth, time-invariant phase-offset field for each velocity direction and removes it before flow reconstruction. AutoFlow supports two methods:

- `msac` finds stationary tissue with a robust random-sample fit, then fits the requested polynomial correction.
- `wrls_arto` follows the OSU-MR weighted regularized least-squares and automatic rejection of temporally invariant outliers workflow: middle-FOV first-order initialization, L1 WRLS fitting, constrained three-component GMM exclusion, and repeated polynomial refitting.

See [Scientific references: background phase correction](../references/index.md#background-phase-correction) for Pruitt et al.'s WRLS+ARTO method and AutoFlow's volumetric adaptations.

For dual-venc legacy H5 inputs, low- and high-venc corrections are computed independently and concurrently before alias reconstruction.

## When to use it

Use background correction when static tissue has a non-zero velocity offset or a smooth spatial phase bias. WRLS+ARTO is the default; use MSAC when you need the established robust-sample fit for comparison.

Do not assume the two methods are interchangeable. They can estimate materially different correction fields and may change dual-venc alias decisions near wrapping boundaries.

## Quick use

### GUI

Open the input in **Input & QC**, then choose **Correction > Background Correction**. Set `MSAC` or `WRLS + ARTO` and click **Background Correction**, or use **Run All** for background correction, noise masking, unwrapping and PC-MRA generation. Opening H5 does not prompt for or apply correction, including when a cache exists. Compatible correction caches are checked only when the action runs.

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

See [loader parameters](../user/parameters.md#loader), [CLI flags](../user/cli-parameters.md), and [API fields](../user/api-parameters.md) for complete type/default/unit/effect/owner tables. Dictionary controls are expanded in [Structured parameters](../user/parameter-schemas.md).

## Outputs

The explicit Correction action updates working `flow_raw`, its pre-unwrapping `flow_input` baseline and stored wrapped phase. New segmentation uses this working velocity. Existing segmentation, geometry, metrics and trajectories are retained until manually rerun. The low-level loader still supports `correction_config` for direct callers; pipeline loading itself disables correction. Writable H5 sources also receive a reusable `corr` dataset, or `corr_low` and `corr_high` for dual-venc data. Cache attributes record the correction method, algorithm version, fit order, source group, and every method-specific parameter used to compute the field.

The GUI retains correction fields as optional workspace arrays and exposes their LR/AP/FH components in the right-side Content selector (radians). The Corr entries remain disabled until a correction was applied or reused. Dual-venc data expose both normalized `Corr Low LR/AP/FH` and `Corr High LR/AP/FH` fields.

Runtime metadata records whether correction was applied, whether a cache was reused, stationary-voxel counts, and the compute device used by WRLS+ARTO. Device metadata is diagnostic only and is not a user setting.

For the shipped WRLS fit orders, CUDA constructs the volume-scale normal equations and evaluates the corrected volume; the small float64 FISTA coefficient loop runs on CPU. The GPU initial solution and Lipschitz constant, configured iteration count, regularization and ARTO rounds are preserved. Fits with more than 64 polynomial coefficients retain the CUDA loop. See the controlled comparison in [Performance](../developer/performance.md).

## Limitations

- WRLS+ARTO needs at least two time frames.
- Automatic CUDA acceleration currently covers the dominant ARTO GMM stage; the small WRLS coefficient systems remain on CPU.
- PyTorch is optional. Missing CUDA support or a CUDA execution failure causes a transparent CPU fallback.
- A correction cache is reusable only when its shape and complete algorithm metadata match.
- Without a ground-truth phase-offset field, method comparison relies on static-tissue residuals and downstream measurement review.

## Where to change code

- Algorithm implementations and automatic CUDA fallback: `autoflow/algorithms/phase_correction.py`
- H5 cache validation, dual-venc concurrency, and cache writing: `autoflow/algorithms/data/h5_loader.py`, `autoflow/algorithms/data/correction_cache.py`, `autoflow/algorithms/data/dual_venc.py`
- Public configuration types: `autoflow/case_types.py`, `autoflow/api.py`, and `autoflow/config.py`
- CLI flags: `autoflow/cli.py`
- Explicit pipeline action and original-source reloading: `autoflow/core/pipeline.py`
- GUI action and method controls: `autoflow/ui/app.py`

## Tests

Smoke coverage is in `tests/test_smoke_phantoms.py`. It covers MSAC progress and cache reuse, WRLS polynomial recovery, method-specific cache isolation, and concurrent dual-venc correction. Run the supported suite with:

```bash
/home/renyuyang/miniconda3/envs/autoflow311/bin/python -m pytest \
  tests/test_smoke_phantoms.py \
  tests/test_pressure_gradient_phantom.py -q
```

## Common problems

- WRLS+ARTO runs on CPU: confirm that the active Python environment has a CUDA-enabled PyTorch build and that `torch.cuda.is_available()` is true. No AutoFlow device option is required.
- A cache is recomputed: inspect the H5 correction attributes for a method, fit-order, version, or WRLS-parameter mismatch.
- The WRLS threshold cannot be edited in the GUI: `MSAC Threshold` is method-specific; WRLS tuning values are read from `configs/loader.json`.
- Dual-venc output changes substantially in a few voxels: review those locations near wrap boundaries because different background fields can change the selected alias branch.

Dual-VENC low/high correction jobs run concurrently. Progress callbacks are delivered on the calling loader thread; GUI workers forward payloads through Qt signals to the main thread. With `force_recompute=true`, running Background Correction ignores embedded correction caches. Reruns read the original selected source and group, preventing repeated subtraction of correction from working flow.
