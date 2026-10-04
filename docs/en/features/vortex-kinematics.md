# Feature: Vortex Kinematics

## Status

| Entry point | Status | Notes |
| --- | --- | --- |
| GUI | Supported | runs with the derived-metrics action and displays maps in the ortho viewer |
| CLI | Supported | opt in with `--with vortex` |
| Python API | Supported | opt in with `requested_metrics=["vortex"]` |

## What it does

See [Scientific references: vortex kinematics](../references/index.md#vortex-kinematics) for velocity-gradient vortex identifiers and swirling strength. The cited `lambda_2` literature does not mean that AutoFlow implements that criterion.

Vortex kinematics derives spatial velocity-gradient fields from the normalized 4D flow velocity: vorticity vector and magnitude, Q-criterion, and swirling strength (`λci`). Velocity is converted from cm/s to m/s and spacing from mm to m, so vorticity and `λci` are in `s^-1` and Q is in `s^-2`.

The feature creates a separate `vortex_support_mask`. It is an eroded copy of the lumen mask used only for these spatial derivatives; it never changes the loaded segmentation or WSS surface.

Mask preparation and optional smoothing operate inside the union vessel bounding box with a one-voxel halo. Velocity-gradient tensors, Q, and eigenvalue calculations then run only at valid support voxels. The output arrays retain the full image shape and are zero outside support, reducing background computation and temporary memory without changing the derivative stencil or metric definitions.

## When to use it

Use it to inspect rotational flow structures after a segmentation is available. Use the support mask and smoothing setting when comparing cases, and avoid interpreting a small isolated signal at the vessel boundary as a vortex.

## Quick use

### GUI

1. Load or create a segmentation.
2. In the visible `Vortex Kinematics Parameters` panel, choose `Gaussian Sigma (vox)` and `Support Erosion (vox)`.
3. Click `WSS / TKE / Pressure / Vortex` in the Hemodynamics workflow.
4. In the left `Global` browser group, toggle `Vorticity Magnitude`, `Q-Criterion`, or `Swirling Strength` for the 3D scene; the ortho viewer `Content` menu provides the same three maps.

### CLI

```bash
autoflow-run case.h5 --output-dir results/case --with vortex
```

`vortex` may be combined with other optional metrics, for example `--with wss,pg,vortex`.

### Python API

```python
from autoflow import AutoFlowConfig, run_case

config = AutoFlowConfig(output_dir="results/case", requested_metrics=["vortex"])
summary = run_case("case.h5", config=config)
```

## Inputs

| Input | Required | Meaning |
| --- | --- | --- |
| normalized `flow` | yes | XYZTV velocity field with three components in cm/s |
| segmentation | yes | lumen support used to reject boundary derivatives |
| resolution | yes | three spatial spacings in mm |

## Parameters

See [vortex parameters](../user/parameters.md#vortex), [CLI flags](../user/cli-parameters.md), and [API fields](../user/api-parameters.md) for complete type/default/unit/effect/owner tables. Dictionary controls are expanded in [Structured parameters](../user/parameter-schemas.md).

## Outputs

`derived_metrics_pixelwise.npz` contains these arrays when `vortex` is requested:

| Array | Shape | Meaning |
| --- | --- | --- |
| `vorticity` | XYZTV3 | vorticity vector (`s^-1`) |
| `vorticity_magnitude` / `vorticity_magnitude_peak` | XYZT / XYZ | magnitude per cardiac phase and phase maximum (`s^-1`) |
| `q_criterion` / `q_criterion_peak` | XYZT / XYZ | Q-criterion per phase and phase maximum (`s^-2`) |
| `swirling_strength` / `swirling_strength_peak` | XYZT / XYZ | swirling strength per phase and phase maximum (`s^-1`) |
| `vortex_support_mask` | XYZT | valid derivative support (`uint8`) |

`summary.json -> requested_metrics.vortex` records whether the batch run requested this family.

## Limitations

- The results are spatial derivatives, so they amplify velocity noise and are sensitive to resolution, VENC, background correction, and segmentation edges.
- A one-voxel support erosion is the default because a central difference needs neighbours on both sides of a voxel. Small vessels can have little or no valid support; reduce the parameter to `0` only if the additional boundary uncertainty is acceptable.
- Q, `λci`, and `λ2` are distinct vortex identifiers. This feature provides Q and `λci`; it does not currently compute `λ2`.
- No confidence interval or clinical threshold is supplied.

## Where to change code

| Change you want | Edit here | Also check | Tests |
| --- | --- | --- | --- |
| velocity-gradient calculation and units | `autoflow/algorithms/metrics/vortex.py` | `autoflow/algorithms/data/h5_loader.py` | `tests/test_smoke_phantoms.py` |
| cache, workspace arrays, and NPZ wiring | `autoflow/core/models.py`, `autoflow/core/pipeline.py`, `autoflow/processing.py` | `autoflow/api.py`, `autoflow/cli.py` | `tests/test_pressure_gradient_phantom.py` |
| GUI parameters and maps | `autoflow/ui/app.py`, `autoflow/ui/ortho_viewer.py` | `configs/vortex.json` | manual GUI verification |

## Tests

- `conda activate autoflow311`, then `python -m pytest tests/test_smoke_phantoms.py tests/test_pressure_gradient_phantom.py -q`
- The smoke suite checks rigid-body rotation (`|ω|=2ω0`, `Q=ω0²`, `λci=ω0`) and verifies that pure shear has zero Q and `λci`.

## Common problems

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| all maps are zero | no segmentation, an empty support mask, or a constant velocity field | verify the segmentation and reduce support erosion only if necessary |
| only the vessel edge is bright | derivative boundary artefact | keep erosion at `1` or increase smoothing after checking the raw velocity |
| maps are noisy | low spatial resolution or noisy velocity | use a documented smoothing setting and compare the support mask across cases |
