# Pressure gradient and relative pressure

## Status

Supported in GUI, CLI and Python.

## What it does

Estimates `grad(p) = -rho*(dv/dt + (v dot grad)v) + mu*laplacian(v)` from velocity, then reconstructs relative pressure and samples centreline pressure/drop profiles. Input velocity (cm/s), spacing (mm), RR (ms) and viscosity (mPa s) are converted to SI; outputs are Pa/m and Pa.

Temporal acceleration uses a periodic central difference over the cardiac cycle. Valid support requires a complete six-neighbour spatial stencil, finite velocities and both temporal neighbours inside segmentation. Optional 26-neighbour erosion adds a margin. Velocity is not zeroed outside the vessel before differentiation; optional smoothing is normalized by valid-mask weights. Work is cropped to the vessel bounding box plus one stencil voxel, then embedded in the original shape.

Least squares fits `(p_j-p_i)/h` to face-average PG. PPE uses a finite-volume negative Laplacian and matching boundary flux. Their discrete equations are equivalent on this Cartesian grid. A hard zero gauge per connected component uses symmetric row/column elimination. Large systems use cached AMG-preconditioned CG, with a Jacobi fallback if AMG is unavailable. Family signatures and derived plane samples avoid repeated calculations.

## When to use it

Use to study spatial pressure variation and pressure drops after reviewing segmentation, velocity and RR.

## Quick use

GUI: select pressure reconstruction in **WSS / TKE / Pressure / Vortex**; inspect support, PG and relative pressure, and centreline profiles. The 3D pressure layer colours the smoothed valid support surface.

CLI:

```bash
autoflow-run case.h5 --output-dir results/case --with pg
```

Python:

```python
from autoflow import AutoFlowConfig, run_case
config = AutoFlowConfig.from_config_dir('./configs', requested_metrics=['pg'], pressure_method='least_squares')
summary = run_case('case.h5', config=config)
```

## Inputs

Calibrated velocity, segmentation, spacing/origin and RR. Phase timing assumes a uniformly sampled complete cardiac cycle; a single phase is steady flow.

## Parameters

[All pressure parameters](../user/parameters.md#pressure_gradient) and [shared fluid properties](../user/parameters.md#fluid). Methods are `least_squares` and `ppe`; default extra erosion is one iteration. `support_erosion_iters=0` disables only the additional margin. Display ranges/opacity do not change numerical results.

## Outputs

Volume PG vector/magnitude, valid support, relative-pressure arrays, centreline profiles and drops; `derived_metrics_pixelwise.npz` and augmented plane JSON/H5. `summary.json` records `pressure_gradient_dt_s` and `pressure_gradient_temporal_scheme=periodic_central_difference`. See [Outputs](../user/outputs.md) for keys and units.

## Limitations

Each disconnected support component has its own gauge; pressure levels across disconnected components are not comparable. Support erosion excludes boundary regions. AMG hierarchies consume additional memory. No uncertainty interval is produced. Old pressure exports had sign/scaling/boundary errors; recompute them rather than applying a global correction factor.

## Where to change code

`autoflow/algorithms/metrics.py` (`compute_pressure_gradient_metrics`, `reconstruct_relative_pressure_map`, assembly and solver helpers); signatures/profiles: `autoflow/core/pipeline.py`; export: `autoflow/processing.py`; views: `autoflow/ui/ortho_viewer.py`, `autoflow/ui/viewer.py`.

## Tests

Pressure phantoms in `tests/test_pressure_gradient_phantom.py` cover analytic constant/quadratic pressure at isotropic/anisotropic spacing, disconnected gauges, periodic acceleration and valid boundary/smoothing support.

```bash
conda activate autoflow311
PYVISTA_OFF_SCREEN=true python -m pytest tests/test_smoke_phantoms.py tests/test_pressure_gradient_phantom.py -q
```

## Common problems

Trimmed edges: inspect the support mask; excluded boundary stencils are expected. Large historical pressure differences: regenerate the case with the corrected algorithm. Unrealistic gradients: check RR, spacing, component orientation, phase order and velocity calibration.
