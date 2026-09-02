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

Each derived family is opt-in and independent. Batch processing computes the requested WSS, TKE, and/or pressure fields once, reuses them for plane summaries and pixelwise output, and does not materialize unrequested families.

Cached WSS, TKE, and pressure families carry independent artifact signatures. A signature includes the relevant source arrays, segmentation, spacing/origin, family parameters, and an algorithm version; pressure also includes RR and centerline paths. A missing or changed signature recomputes only the affected family and removes its stale pixelwise payload. Workspaces saved by older versions have no signatures and therefore recompute derived caches on first use.

Pressure temporal acceleration uses a periodic central difference across the cardiac-cycle boundary. The first phase uses the last and second phases as neighbors, and the last phase uses the penultimate and first phases. `summary.json` records `pressure_gradient_dt_s` and `pressure_gradient_temporal_scheme=periodic_central_difference` for provenance.

When segmentation phases are byte-identical, WSS builds, extracts, and smooths the vessel surface once. Each phase receives a geometry copy before phase-specific velocity sampling and WSS point data are attached, so returned surfaces remain independent without repeating the expensive geometry preparation.

For large `least_squares` pressure systems, AutoFlow builds a smoothed-aggregation AMG preconditioner and reuses it across cardiac phases. This accelerates the conjugate-gradient solves at the cost of retaining the hierarchy in memory. If PyAMG is unavailable or setup fails, the solver falls back to a cached Jacobi preconditioner.

Finite-difference pressure terms are evaluated inside the union vessel bounding box plus one stencil voxel and then embedded into the original image shape. This removes empty-background work without changing the pressure-gradient output contract. GUI and video pressure layers also reuse smoothed support geometry when segmentation phases are identical.

Rendering controls are split by metric: `configs/wss.json -> render`, `configs/tke.json -> render`, and `configs/pressure_gradient.json -> render`. Metric ranges and visibility load as GUI defaults. In the live GUI scene, WSS, TKE, pressure-gradient, and relative-pressure layers share one colorbar slot whose geometry and fonts come from `configs/colorbar.json`; metric `bar_cfg` values remain the defaults for offline video rendering. GUI runtime edits update the live shared slot and are copied to the metric video settings for later exports in the same session.

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
3. click `WSS / TKE / Pressure / Vortex`
4. inspect the ortho viewer or browser objects

The GUI computes only missing derived families. When basic plane metrics already exist, the derived step augments those records and rewrites the derived pixelwise plane samples without rerunning the basic flow/area calculation.

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

| Parameter | Type | Default | Where set | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `smoothing_iteration` | int | `200` | `configs/wss.json` | smoothing for WSS processing | `autoflow/algorithms/metrics.py` |
| `viscosity` | float | shared from `configs/fluid.json` | `configs/fluid.json` or `configs/wss.json` | viscosity value for WSS | `autoflow/algorithms/metrics.py` |
| `inward_distance` | float or `auto` | `auto` | `configs/wss.json` | near-wall sample distance | `autoflow/algorithms/metrics.py` |
| `parabolic_fitting` | bool | `True` | `configs/wss.json` | WSS fitting mode | `autoflow/algorithms/metrics.py` |
| `no_slip_condition` | bool | `False` | `configs/wss.json` | WSS no-slip toggle | `autoflow/algorithms/metrics.py` |
| `rho` | float | shared from `configs/fluid.json` | `configs/fluid.json` or `configs/tke.json` | density used by TKE | `autoflow/algorithms/metrics.py` |
| `rho` | float | shared from `configs/fluid.json` | `configs/fluid.json` or `configs/pressure_gradient.json` | density used by pressure-gradient and relative-pressure logic | `autoflow/algorithms/metrics.py` |
| `viscosity` | float | shared from `configs/fluid.json` | `configs/fluid.json` or `configs/pressure_gradient.json` | viscosity used by pressure-gradient and relative-pressure logic | `autoflow/algorithms/metrics.py` |
| `method` | string | `least_squares` | `configs/pressure_gradient.json` or GUI panel | choose relative-pressure reconstruction method: `least_squares` or `ppe`; large least-squares systems use cached AMG-preconditioned CG with a Jacobi fallback | `autoflow/algorithms/metrics.py` |
| `smoothing_sigma` | float | `0.0` | `configs/pressure_gradient.json` | pressure-gradient smoothing before relative-pressure reconstruction | `autoflow/algorithms/metrics.py` |
| `support_erosion_iters` | int | `1` | `configs/pressure_gradient.json` or GUI panel | how many voxels to erode from the segmentation before the pressure solve and pressure-support display are defined | `autoflow/algorithms/metrics.py` |
| `layer_opacity` | float | `0.6` | `configs/pressure_gradient.json` or GUI panel | opacity of the pressure-gradient 3D layer | `autoflow/ui/viewer.py` |
| `relative_pressure_opacity` | float | `0.6` | `configs/pressure_gradient.json` or GUI panel | opacity of the relative-pressure 3D layer | `autoflow/ui/viewer.py` |
| `use_convective_acceleration` | bool | `True` | `configs/pressure_gradient.json` | include convective acceleration | `autoflow/algorithms/metrics.py` |
| `wss.render.clim` | list[float, float] | `[0.0, 10.0]` | `configs/wss.json` | WSS display range in GUI and videos | `autoflow/ui/viewer.py`, `autoflow/rendering/videos.py` |
| `wss.render.show_scalar_bar` | bool | `True` | `configs/wss.json` | show or hide the WSS colorbar in GUI and videos | `autoflow/ui/viewer.py`, `autoflow/rendering/videos.py` |
| `tke.render.clim` | list[float, float] | `[0.0, 100.0]` | `configs/tke.json` | TKE display range in GUI and videos | `autoflow/ui/viewer.py`, `autoflow/rendering/videos.py` |
| `tke.render.show_scalar_bar` | bool | `True` | `configs/tke.json` | show or hide the TKE colorbar in GUI and videos | `autoflow/ui/viewer.py`, `autoflow/rendering/videos.py` |
| `pressure_gradient.render.clim` | list[float, float] or `null` | `null` | `configs/pressure_gradient.json` | pressure-gradient display range in GUI and videos; `null` keeps the auto range | `autoflow/ui/viewer.py`, `autoflow/rendering/videos.py` |
| `pressure_gradient.render.show_scalar_bar` | bool | `True` | `configs/pressure_gradient.json` | show or hide the pressure-gradient colorbar in GUI and videos | `autoflow/ui/viewer.py`, `autoflow/rendering/videos.py` |
| `pressure_gradient.render.relative_pressure_clim` | list[float, float] or `null` | `null` | `configs/pressure_gradient.json` | relative-pressure display range in GUI and videos; `null` keeps the symmetric auto range | `autoflow/ui/viewer.py`, `autoflow/rendering/videos.py` |
| `pressure_gradient.render.relative_pressure_show_scalar_bar` | bool | `True` | `configs/pressure_gradient.json` | show or hide the relative-pressure colorbar in GUI and videos | `autoflow/ui/viewer.py`, `autoflow/rendering/videos.py` |
| GUI runtime `clim` edits | two floats or `auto` | config default | GUI `Render / Colorbar` panel | update live GUI metric ranges immediately and reuse the same values for video export in the current session | `autoflow/ui/app.py` |
| GUI runtime colorbar width, height, gap, and x position | float | config default | GUI `Render / Colorbar` panel | update live GUI colorbar layout immediately and reuse the same layout for video export in the current session | `autoflow/ui/app.py` |
| `requested_metrics` / `--with` | csv | empty | CLI/API | opt in to `wss`, `tke`, and/or `pg` | `autoflow/api.py`, `autoflow/cli.py` |
| `skip_derived` | bool | `False` | batch config or CLI/API | remove all requested WSS, TKE, pressure-gradient, relative-pressure, and centerline-pressure work | `autoflow/processing.py` |

## Outputs

| Output file or object | Created when | Meaning |
| --- | --- | --- |
| `derived_metrics_pixelwise.npz` | requested derived export runs | whole-volume WSS, pressure-gradient arrays, reconstructed relative pressure, and optional TKE arrays |
| WSS surfaces in workspace | derived step succeeds | WSS scene data |
| pressure-gradient arrays in workspace | derived step succeeds | vector field, magnitude map, support mask, and display range |
| relative-pressure arrays in workspace | derived step succeeds | reconstructed volume arrays and display range |
| centerline pressure profiles in workspace and `summary.json` | relative pressure succeeds | sampled centerline pressure curves plus per-phase pressure drop |
| `pressure_gradient_dt_s`, `pressure_gradient_temporal_scheme` in `summary.json` | pressure analysis succeeds | temporal spacing and periodic derivative provenance |
| TKE arrays in workspace | TKE exists | optional TKE output |
| derived plane summaries | plane metrics and derived metrics both exist | per-plane derived summaries |

## Limitations
- segmentation is required
- TKE is optional and must remain optional
- DICOM-derived mag/flow-only inputs must not synthesize fake TKE
- pressure-gradient boundary voxels are excluded by the support mask before reconstruction
- AMG acceleration retains its hierarchy for repeated cardiac-phase solves, trading additional memory for lower solve time
- WSS and pressure outputs currently do not include statistical confidence intervals; use resolution/segmentation sensitivity runs and phantom validation before interpreting small differences as physiological change

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
