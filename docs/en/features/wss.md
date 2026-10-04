# Wall shear stress

## Status

Supported in GUI, CLI and Python.

## What it does

Computes wall shear vectors in Pa and their magnitude from tangential velocity derivatives at the segmented vessel wall. The cell-centred velocity field is interpolated to points for continuous probing. Each tangential component uses wall-normal samples at 0, h and 2h; quadratic mode evaluates the derivative analytically at the wall, while linear mode uses the first slope. Default no-slip sets the wall velocity to zero.

See [Scientific references: WSS](../references/index.md#wall-shear-stress) for MRI wall-shear estimation and Taubin smoothing. AutoFlow's probe-based derivative is not the original B-spline estimator.

Taubin smoothing reduces surface shrinkage. Byte-identical segmentation phases reuse prepared geometry; velocity and attached WSS values remain independent per phase. Finite probes and intermediate lumen checks identify invalid sample segments. Family signatures include source arrays, geometry, parameters and algorithm version; cached arrays are reused by plane summaries and exports.

## When to use it

Use after reviewing segmentation and velocity calibration, when wall shear magnitude or direction is required.

## Quick use

GUI: load segmentation, open **WSS / TKE / Pressure / Vortex**, and inspect WSS surfaces / orthogonal views. **Calculate && Save Metrics** also samples requested WSS into plane summaries.

CLI:

```bash
autoflow-run case.h5 --output-dir results/case --with wss
```

Python:

```python
from autoflow import AutoFlowConfig, run_case
config = AutoFlowConfig.from_config_dir('./configs', requested_metrics=['wss'])
summary = run_case('case.h5', config=config)
```

## Inputs

Velocity `XYZT3` in cm/s; segmentation `XYZ(T)`; spacing/origin in mm; dynamic viscosity in mPa s. Sampling distance `auto` is resolved from voxel spacing.

## Parameters

[All WSS parameters](../user/parameters.md#wss), [shared viscosity](../user/parameters.md#fluid), and [colourbars](../user/parameters.md#colorbar). Defaults include 200 Taubin iterations, quadratic fitting and no-slip. Adjust `inward_distance` for lumen width; numerical choices change the result.

## Outputs

`derived_metrics_pixelwise.npz` contains the wall volume. Workspace surfaces carry `wss` (Pa), signed `wss_vectors` (Pa), and uint8 `wss_valid`. Invalid samples are NaN; volume voxels away from the rasterized wall are zero. A valid wall point sharing a voxel can replace its invalid value. Plane summaries include wall mean, peak and percentile values; see [Outputs](../user/outputs.md).

## Limitations

Segmentation, smoothing and sample distance affect estimates. A sample reaching outside the lumen/grid is invalid. Artificial inlet/outlet caps are not physiological vessel walls. No statistical confidence interval is produced. Older exports must be recomputed: the corrected derivative/vector algorithm invalidates old workspace signatures.

## Where to change code

`autoflow/algorithms/metrics.py` (`calculate_gradient`, `cal_wss_from_surf`, `compute_wss_metrics`); caching and plane wiring: `autoflow/core/pipeline.py`; views: `autoflow/ui/viewer.py`, `autoflow/ui/ortho_viewer.py`.

## Tests

Smoke phantoms cover analytic linear/quadratic vector derivatives, interpolation, invalid probes and phase independence. Optional `python validate_wss_ground_truth.py` evaluates the production function on six linear fields and fails above 0.01% relative error.

```bash
conda activate autoflow311
PYVISTA_OFF_SCREEN=true python -m pytest tests/test_smoke_phantoms.py tests/test_pressure_gradient_phantom.py -q
```

## Common problems

Blank wall regions: inspect `wss_valid`, boundary placement and segmentation, then reduce sampling distance if it crosses a narrow lumen. Changed historical values: recompute and replace old exports.
