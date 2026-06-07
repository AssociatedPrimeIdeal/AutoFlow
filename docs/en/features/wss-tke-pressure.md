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
autoflow-run case.h5 --output-dir results/case
```

### Python API
Use `run_case()` or `run_batch()` with `skip_derived=False`.

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
| `smoothing_iteration` | int | `200` | `configs/derived.json` | smoothing for WSS processing |
| `viscosity` | float | `4.0` | `configs/derived.json` | viscosity value for WSS |
| `inward_distance` | float or `auto` | `auto` | `configs/derived.json` | near-wall sample distance |
| `parabolic_fitting` | bool | `True` | `configs/derived.json` | WSS fitting mode |
| `no_slip_condition` | bool | `False` | `configs/derived.json` | WSS no-slip toggle |
| `rho` | float | `1060.0` | `configs/derived.json` | density used by pressure-gradient logic |
| `pressure_gradient_smoothing_sigma` | float | `0.0` | `configs/derived.json` | pressure-gradient smoothing |
| `pressure_gradient_use_convective_acceleration` | bool | `True` | `configs/derived.json` | include convective acceleration |
| `skip_derived` | bool | `False` | batch config or CLI/API | disable the entire derived-export step |

## Outputs

| Output file or object | Created when | Meaning |
| --- | --- | --- |
| `derived_metrics_pixelwise.npz` | derived export runs | whole-volume WSS, pressure gradient, and optional TKE arrays |
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
