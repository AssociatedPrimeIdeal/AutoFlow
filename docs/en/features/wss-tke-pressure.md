# Feature: WSS, TKE, And Pressure Gradient

## Status

| Entry point | Status | Notes |
| --- | --- | --- |
| GUI | Supported | derived volumes and scene refresh |
| CLI | Supported | derived export and optional videos |
| Python API | Supported | through case and batch APIs |
| TKE specifically | Partial | only available when the input carries TKE or complex-derived support |

## What It Does
This feature computes wall shear stress, pressure-gradient fields, and optional TKE outputs from the loaded velocity data and segmentation.

Rendering controls are now split by metric: `configs/wss.json -> render`, `configs/tke.json -> render`, and `configs/pressure_gradient.json -> render`. The same settings drive offline videos and the live GUI scene objects.

## When To Use It
- use it after segmentation and planes exist
- use it when WSS or pressure-gradient maps are needed
- use it when TKE is present in the source or already materialized
- do not expect TKE to appear for every case

## Quick Use

### GUI
1. load a segmented case
2. run `Calculate && Save Metrics` if you want plane summaries too
3. click `WSS / TKE / Pressure Gradient`
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
| `rho` | float | shared from `configs/fluid.json` | `configs/fluid.json` or `configs/pressure_gradient.json` | density used by pressure-gradient logic |
| `viscosity` | float | shared from `configs/fluid.json` | `configs/fluid.json` or `configs/pressure_gradient.json` | viscosity used by pressure-gradient logic |
| `smoothing_sigma` | float | `0.0` | `configs/pressure_gradient.json` | pressure-gradient smoothing |
| `use_convective_acceleration` | bool | `True` | `configs/pressure_gradient.json` | include convective acceleration |
| `wss.render.clim` | list[float, float] | `[0.0, 10.0]` | `configs/wss.json` | WSS display range in GUI and videos |
| `wss.render.show_scalar_bar` | bool | `True` | `configs/wss.json` | show or hide the WSS colorbar in GUI and videos |
| `tke.render.clim` | list[float, float] | `[0.0, 100.0]` | `configs/tke.json` | TKE display range in GUI and videos |
| `tke.render.show_scalar_bar` | bool | `True` | `configs/tke.json` | show or hide the TKE colorbar in GUI and videos |
| `pressure_gradient.render.clim` | list[float, float] | `[0.0, 500.0]` | `configs/pressure_gradient.json` | pressure-gradient display range in GUI and videos |
| `pressure_gradient.render.show_scalar_bar` | bool | `True` | `configs/pressure_gradient.json` | show or hide the pressure-gradient colorbar in GUI and videos |
| `requested_metrics` / `--with` | csv | empty | CLI/API | opt in to `wss`, `tke`, and/or `pg` |
| `skip_derived` | bool | `False` | batch config or CLI/API | remove all requested WSS, TKE, and pressure-gradient work |

## Outputs

| Output file or object | Created when | Meaning |
| --- | --- | --- |
| `derived_metrics_pixelwise.npz` | requested derived export runs | whole-volume WSS, pressure gradient, and optional TKE arrays |
| WSS surfaces in workspace | derived step succeeds | WSS scene data |
| pressure-gradient arrays in workspace | derived step succeeds | volume arrays and support mask |
| TKE arrays in workspace | TKE exists | optional TKE output |
| derived plane summaries | plane metrics and derived metrics both exist | per-plane derived summaries |

## Limitations
- segmentation is required
- TKE is optional and must remain optional
- DICOM-derived mag/flow-only inputs must not synthesize fake TKE
- pressure-gradient boundary voxels are excluded by the support mask

## Where To Change Code

| Change you want | Edit here | Also check | Tests |
| --- | --- | --- | --- |
| WSS computation | `autoflow/algorithms/metrics.py` | `autoflow/ui/ortho_viewer.py` | `tests/test_smoke_phantoms.py` |
| TKE handling | `autoflow/algorithms/data.py`, `autoflow/algorithms/metrics.py` | `autoflow/core/pipeline.py` | `tests/test_smoke_phantoms.py` |
| pressure-gradient computation | `autoflow/algorithms/metrics.py` | `autoflow/ui/ortho_viewer.py` | `tests/test_pressure_gradient_phantom.py` |
| derived export wiring | `autoflow/core/pipeline.py`, `autoflow/processing.py` | `autoflow/rendering/videos.py` | `tests/test_pressure_gradient_phantom.py` |

## Tests

- `~/miniconda3/envs/ryy/bin/python -m pytest tests/test_smoke_phantoms.py -q`
- `~/miniconda3/envs/ryy/bin/python -m pytest tests/test_pressure_gradient_phantom.py -q`

## Common Problems

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| TKE output is unavailable | input has no TKE-supporting source | expected for many mag/flow-only cases |
| derived step is skipped | no segmentation | create or load segmentation first |
| pressure-gradient looks trimmed at the vessel edge | support mask removes boundary voxels | expected behavior; inspect `pressure_gradient_support_mask` |
