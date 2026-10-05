# Wall shear stress

## Status

Supported in GUI, CLI and Python.

## What it does

Computes wall shear vectors in Pa and their magnitude from tangential velocity derivatives at the segmented vessel wall. Velocity samples are placed directly at their original voxel centres and continuously interpolated there, avoiding the extra averaging of cell-to-corner conversion. Each tangential component uses wall-normal samples at 0, h and 2h; quadratic mode evaluates the derivative analytically at the wall, while linear mode uses the first slope. Default no-slip sets the wall velocity to zero.

See [Scientific references: WSS](../references/index.md#wall-shear-stress) for MRI wall-shear estimation and Taubin smoothing. AutoFlow's probe-based derivative is not the original B-spline estimator.

A padded signed-distance field reconstructs a closed numerical wall between voxel centres. With smoothing enabled, it uses a Gaussian physical scale of half the smallest voxel spacing followed by Taubin smoothing. Zero smoothing iterations disables both smoothing steps. Lumen membership is evaluated against this same final wall with a cached signed-distance locator, avoiding disagreement between a smoothed wall and a separate stair-step binary boundary. Short forward/backward probes correct unambiguous local normal reversals, including cavity walls.

Sampling uses the configured fixed h and 2h; it is not shortened automatically. Finite/grid-valid velocity probes and intermediate lumen checks remain required. Identical segmentation phases reuse both prepared wall geometry and its support locator; velocity and WSS values remain independent per phase. Algorithm signatures invalidate older cached results.

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

GUI and videos read shared materials from `configs/render_style.json`, shared colourbar layout from `configs/colorbar.json`, and layer styles from the metric JSON `render` groups. See [Rendering configuration layout](../developer/config-system.md#rendering-configuration-layout) and the complete [parameter tables](../user/parameters.md). Restart the GUI or start a new run after editing the selected JSON files.


[All WSS parameters](../user/parameters.md#wss), [shared viscosity](../user/parameters.md#fluid), and [colourbars](../user/parameters.md#colorbar). Defaults include 200 Taubin iterations, quadratic fitting and no-slip. Adjust `inward_distance` for lumen width; numerical choices change the result.

## Outputs

`derived_metrics_pixelwise.npz` contains the wall volume. Workspace surfaces carry `wss` (Pa), signed `wss_vectors` (Pa), uint8 `wss_valid`, and uint8 `wss_normal_flipped` (local direction corrections). Invalid samples are NaN; the GUI and movies show them as transparent gaps using `wss_valid`, while finite samples retain their measured values. Both use an unlit `turbo` colour scale so lighting cannot whiten the WSS colours. Volume voxels away from the rasterized wall are zero. A valid wall point sharing a voxel can replace its invalid value. Plane summaries include wall mean, peak and percentile values; see [Outputs](../user/outputs.md).

## Limitations

Segmentation, reconstruction, smoothing and sample distance affect estimates. Closed geometry and higher valid coverage do not establish physiological accuracy. Narrow branches, insufficient velocity-grid support, or a normal segment crossing a wall still produce NaNs. Artificial inlet/outlet caps are not physiological vessel walls. No statistical confidence interval is produced. Recompute older exports after this change; cached workspace signatures use `wss-continuous-wall-v4`.

Comparison with the [upstream MRI WSS calculator](https://github.com/EdwardFerdian/wss_mri_calculator/tree/5819969357d45c317ac7e6b88e27daf0016b06f5):

| Item | Upstream commit `5819969` | AutoFlow |
| --- | --- | --- |
| Default sampling step | 0.6 mm | Minimum voxel spacing (`auto`) |
| Wall geometry | 500 Laplacian iterations | Closed signed-distance reconstruction, half-voxel Gaussian scale and 200 Taubin iterations |
| Velocity probing | Cell-data grid | Continuous interpolation at original voxel centres |
| Wall derivative | Tangential-speed fit and numerical endpoint gradient | Analytic derivative of each tangential-vector component |
| Invalid samples | No grid/lumen rejection | Finite/grid-valid probes and intermediate support against the same numerical wall |
| `wss_vectors` | First inward tangential velocity | Wall shear vectors in Pa |

Manual validation on 2026-10-05 used noise-free Poiseuille tubes, including different radii, resolutions, oblique directions and anisotropic voxels. Of nine cases, eight improved mean absolute relative error against the preceding implementation; one coarse 7-mm-radius case increased from 8.9% to 9.5%. All evaluated central lateral wall points were valid. Errors across the new cases ranged from 7.0% to 19.4%, so this is an improvement in tested conditions, not a claim of exact WSS. The compared surfaces have different vertices and valid subsets.

For a 10-mm-radius tube with peak velocity 0.5 m/s, viscosity 4 mPa s and 2.5-mm isotropic voxels, true lateral WSS is 0.4 Pa. The new median is 0.369 Pa (7.8% low), with mean absolute relative error 12.6%, versus 17.1% in the preceding implementation. The reproduced upstream algorithm at h=2.5 mm gave mean absolute relative error about 29%; its default h=0.6 mm gave about 101%. These upstream checks use current PyVista API renames, not its original PyVista 0.24 runtime, and do not validate real acquisitions.

On phase 0 of the default DV validation case at h=2.5 mm, valid wall points increased from 56.6% to 87.5%, while open edges decreased from 109 to zero. This describes numerical support, not ground-truth accuracy. The previous unsmoothed-vertex anchoring experiment was rejected because curved-wall error increased, despite higher coverage. Stronger Gaussian smoothing was also rejected after worsening some coarse-vessel phantoms.

## Where to change code

`autoflow/algorithms/metrics/wss.py` (`calculate_gradient`, `cal_wss_from_surf`, `compute_wss_metrics`); caching and plane wiring: `autoflow/core/pipeline.py`; views: `autoflow/ui/viewer.py`, `autoflow/ui/ortho_viewer.py`; shared colour defaults: `autoflow/config.py`; invalid-sample display and finite limits: `autoflow/rendering/style.py`; movies: `autoflow/rendering/videos.py`.

## Tests

Smoke phantoms cover analytic vector derivatives, local normals, interpolation, invalid probes, phase reuse, closed-wall cavities/separate lumens, and end-to-end axial and anisotropic Poiseuille WSS. Optional `python validate_wss_ground_truth.py` evaluates the production function on six linear fields and fails above 0.01% relative error.

```bash
conda activate autoflow311
PYVISTA_OFF_SCREEN=true python -m pytest tests/test_smoke_phantoms.py tests/test_pressure_gradient_phantom.py -q
```

## Common problems

Blank wall regions: recompute old results, inspect `wss_valid`, boundary placement and segmentation, then compare a smaller sampling distance if 2h crosses a narrow lumen. Changing h changes the estimator; confirm stability rather than treating a larger valid fraction as proof of accuracy. Changed historical values: recompute and replace old exports.
