# Feature: WSS, TKE, Pressure Gradient, And Relative Pressure

## Status

| Entry point | Status | Notes |
| --- | --- | --- |
| GUI | Supported | derived volumes and scene refresh |
| CLI | Supported | derived export and optional videos |
| Python API | Supported | through case and batch APIs |
| TKE specifically | Partial | only available when the input carries TKE or complex-derived support |

## What It Does
This feature computes wall shear stress, pressure-gradient fields, reconstructed relative-pressure maps, centerline pressure-drop profiles, and optional TKE outputs from the loaded velocity data and segmentation.

Rendering controls are now split by metric: `configs/wss.json -> render`, `configs/tke.json -> render`, and `configs/pressure_gradient.json -> render`. These values load as GUI defaults, and the GUI can override `clim` plus colorbar layout at runtime for the live scene and later offline video export in the same session.

For pressure-gradient and relative-pressure 3D views, the GUI and exported videos now color a smoothed pressure-support surface derived from `pressure_gradient_support_mask`. This keeps the 3D pressure view aligned with the valid pressure solve region.

## When To Use It
- use it after segmentation and planes exist
- use it when WSS, pressure-gradient, or relative-pressure maps are needed
- use it when TKE is present in the source or already materialized
- do not expect TKE to appear for every case

## Quick Use

### GUI
1. load a segmented case
2. run `Calculate && Save Metrics` if you want plane summaries too
3. click `WSS / TKE / Pressure`
4. inspect the ortho viewer or browser objects

### CLI

```bash
autoflow-run case.h5 --output-dir results/case --with wss,tke,pg
```

### Python API
Use `run_case()` or `run_batch()` with `requested_metrics=["wss", "tke", "pg"]` or any subset.

## Inputs

| Input | Required | Meaning |
| --- | --- | --- |
| flow | yes | velocity field |
| segmentation | yes | mask for valid derived computations |
| `sigma` or `tke_array` | optional | source needed for TKE |
| resolution and origin | yes | spatial calibration |

## Parameters

| Parameter | Type | Default | Where set | Effect |
| --- | --- | --- | --- | --- |
| `smoothing_iteration` | int | `200` | `configs/wss.json` | smoothing for WSS processing |
| `viscosity` | float | shared from `configs/fluid.json` | `configs/fluid.json` or `configs/wss.json` | viscosity value for WSS |
| `inward_distance` | float or `auto` | `auto` | `configs/wss.json` | near-wall sample distance |
| `parabolic_fitting` | bool | `True` | `configs/wss.json` | WSS fitting mode |
| `no_slip_condition` | bool | `False` | `configs/wss.json` | WSS no-slip toggle |
| `rho` | float | shared from `configs/fluid.json` | `configs/fluid.json` or `configs/tke.json` | density used by TKE |
| `rho` | float | shared from `configs/fluid.json` | `configs/fluid.json` or `configs/pressure_gradient.json` | density used by pressure-gradient and relative-pressure logic |
| `viscosity` | float | shared from `configs/fluid.json` | `configs/fluid.json` or `configs/pressure_gradient.json` | viscosity used by pressure-gradient and relative-pressure logic |
| `method` | string | `least_squares` | `configs/pressure_gradient.json` or GUI panel | choose relative-pressure reconstruction method: `least_squares` or `ppe`; both use SciPy sparse solvers |
| `smoothing_sigma` | float | `0.0` | `configs/pressure_gradient.json` | pressure-gradient smoothing before relative-pressure reconstruction |
| `support_erosion_iters` | int | `1` | `configs/pressure_gradient.json` or GUI panel | how many voxels to erode from the segmentation before the pressure solve and pressure-support display are defined |
| `layer_opacity` | float | `0.6` | `configs/pressure_gradient.json` or GUI panel | opacity of the pressure-gradient 3D layer |
| `relative_pressure_opacity` | float | `0.6` | `configs/pressure_gradient.json` or GUI panel | opacity of the relative-pressure 3D layer |
| `use_convective_acceleration` | bool | `True` | `configs/pressure_gradient.json` | include convective acceleration |
| `wss.render.clim` | list[float, float] | `[0.0, 10.0]` | `configs/wss.json` | WSS display range in GUI and videos |
| `wss.render.show_scalar_bar` | bool | `True` | `configs/wss.json` | show or hide the WSS colorbar in GUI and videos |
| `tke.render.clim` | list[float, float] | `[0.0, 100.0]` | `configs/tke.json` | TKE display range in GUI and videos |
| `tke.render.show_scalar_bar` | bool | `True` | `configs/tke.json` | show or hide the TKE colorbar in GUI and videos |
| `pressure_gradient.render.clim` | list[float, float] or `null` | `null` | `configs/pressure_gradient.json` | pressure-gradient display range in GUI and videos; `null` keeps the auto range |
| `pressure_gradient.render.show_scalar_bar` | bool | `True` | `configs/pressure_gradient.json` | show or hide the pressure-gradient colorbar in GUI and videos |
| `pressure_gradient.render.relative_pressure_clim` | list[float, float] or `null` | `null` | `configs/pressure_gradient.json` | relative-pressure display range in GUI and videos; `null` keeps the symmetric auto range |
| `pressure_gradient.render.relative_pressure_show_scalar_bar` | bool | `True` | `configs/pressure_gradient.json` | show or hide the relative-pressure colorbar in GUI and videos |
| GUI runtime `clim` edits | two floats or `auto` | config default | GUI `Render / Colorbar` panel | update live GUI metric ranges immediately and reuse the same values for video export in the current session |
| GUI runtime colorbar width, height, gap, and x position | float | config default | GUI `Render / Colorbar` panel | update live GUI colorbar layout immediately and reuse the same layout for video export in the current session |
| `requested_metrics` / `--with` | csv | empty | CLI/API | opt in to `wss`, `tke`, and/or `pg` |
| `skip_derived` | bool | `False` | batch config or CLI/API | remove all requested WSS, TKE, pressure-gradient, relative-pressure, and centerline-pressure work |

## Outputs

| Output file or object | Created when | Meaning |
| --- | --- | --- |
| `derived_metrics_pixelwise.npz` | requested derived export runs | whole-volume WSS, pressure-gradient arrays, reconstructed relative pressure, and optional TKE arrays |
| WSS surfaces in workspace | derived step succeeds | WSS scene data |
| pressure-gradient arrays in workspace | derived step succeeds | vector field, magnitude map, support mask, and display range |
| relative-pressure arrays in workspace | derived step succeeds | reconstructed volume arrays and display range |
| centerline pressure profiles in workspace and `summary.json` | relative pressure succeeds | sampled centerline pressure curves plus per-phase pressure drop |
| TKE arrays in workspace | TKE exists | optional TKE output |
| derived plane summaries | plane metrics and derived metrics both exist | per-plane derived summaries |

## Limitations
- segmentation is required
- TKE is optional and must remain optional
- DICOM-derived mag/flow-only inputs must not synthesize fake TKE
- pressure-gradient boundary voxels are excluded by the support mask before reconstruction

## Where To Change Code

| Change you want | Edit here | Also check | Tests |
| --- | --- | --- | --- |
| WSS computation | `autoflow/algorithms/metrics.py` | `autoflow/ui/ortho_viewer.py` | `tests/test_smoke_phantoms.py` |
| TKE handling | `autoflow/algorithms/data.py`, `autoflow/algorithms/metrics.py` | `autoflow/core/pipeline.py` | `tests/test_smoke_phantoms.py` |
| pressure-gradient reconstruction, relative-pressure reconstruction, and centerline pressure drop | `autoflow/algorithms/metrics.py` | `autoflow/ui/app.py`, `autoflow/ui/ortho_viewer.py` | `tests/test_pressure_gradient_phantom.py` |
| derived export wiring | `autoflow/core/pipeline.py`, `autoflow/processing.py` | `autoflow/rendering/videos.py` | `tests/test_pressure_gradient_phantom.py` |

## Tests

- `~/miniconda3/envs/ryy/bin/python -m pytest tests/test_smoke_phantoms.py -q`
- `~/miniconda3/envs/ryy/bin/python -m pytest tests/test_pressure_gradient_phantom.py -q`

## Common Problems

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| TKE output is unavailable | input has no TKE-supporting source | expected for many mag/flow-only cases |
| derived step is skipped | no segmentation | create or load segmentation first |
| relative-pressure map looks trimmed at the vessel edge | support mask removes boundary voxels from the pressure-gradient solve | expected behavior; inspect `pressure_gradient_support_mask` |
