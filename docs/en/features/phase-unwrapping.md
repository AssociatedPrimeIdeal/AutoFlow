# Phase Unwrapping

## Status
Supported as an optional GUI, CLI, and Python workflow step for the bundled backends; PUDIP-Flow and GUST-Flow are Experimental. Disabled by default; `DV` dual-VENC inputs are skipped while `LV` and `HV` selections can be unwrapped.

## What it does

Runs a selected phase-unwrapping backend on wrapped phase (radians), converts the result to velocity, and records signed wrap count `k` plus affected voxels. The bundled `gc3D`, `lap4D`, and `nprs` methods remain available; optional `pudip` and `gust` methods use the `third_party/PUDIP-Flow` and `third_party/GUST-Flow` git submodules.

## When to use it

Use for single-VENC data containing phase wraps or for a dual-VENC case loaded as `LV` or `HV`. `DV` dual-VENC inputs are skipped automatically. Use PUDIP-Flow or GUST-Flow when a learned GPU reconstruction is appropriate and its training cost is acceptable.

## Quick use

GUI: open **Phase Unwrapping**; the method selector defaults to `lap4D`. Choose a method/mask/device and click **Unwrap Phase**. Choosing a method is the opt-in action; the step is never run until that button is pressed. PUDIP-Flow and GUST-Flow training settings are read from `phase_unwrapping.json` under `backend_params`. The Browser exposes **Estimated Wrap Locations** and **Phase Wrap Count**.

CLI:

    autoflow-run case.h5 --phase-unwrap-method lap4D --phase-unwrap-device auto

For the learned backends, initialize the submodules and install the optional backend group:

```bash
git submodule update --init --recursive
pip install -e ".[pu]"
```

`pip install .` does not install optional extras. Use `pip install ".[all]"` to install every optional group; GUST-Flow still requires a compatible CUDA/CuPy runtime.

Python: set `AutoFlowConfig(phase_unwrap_method="lap4D")` and call `run_case`. Omit the method (or leave it empty) to skip the step.

## Inputs

The loader must provide canonical `phase_wrapped` (`X,Y,Z,T,3`). For normalized H5 and DICOM velocity inputs, AutoFlow reconstructs the wrapped phase modulo `2π` from velocity and VENC before this step. The GUI and CLI mask choices are `segmask` and `pcmra_std`. `segmask` is used as the learned-backend weight map and Gaussian-center confidence; `pcmra_std` uses temporal PC-MRA standard deviation for the PUDIP weight map and GUST Gaussian-center initialization. PUDIP-Flow and GUST-Flow receive the converted `[component,time,x,y,z]` layout expected by their upstream packages.

## Parameters

| Parameter | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `phase_unwrap_method` | enum | `lap4D` in GUI; omitted in CLI/API | Config/CLI/GUI | Select `gc3D`, `lap4D`, `nprs`, `pudip`, or `gust`; selecting a method opts in | `autoflow/algorithms/phase_unwrapping.py` |
| `phase_unwrap_device` | enum | `auto` | Config/CLI/GUI | CPU or CUDA (`lap4D` FFT, `nprs` FFT, `gc3D` graph construction, or learned backend training) | same |
| `phase_unwrap_mask` | enum | `segmask` | Config/CLI/GUI | `segmask` or temporal PC-MRA standard deviation (`pcmra_std`) for learned-backend weighting/initialization | `autoflow/core/pipeline.py`, `autoflow/algorithms/phase_unwrapping.py` |
| `lap4d_ts` | float | `2.0` | Config/GUI (shown only for `lap4D`) | Temporal weighting for lap4D | `phase_unwrapping.py` |
| `nprs_upsampling_factor` | int | `2` | Config/GUI (shown only for `nprs`) | NPRS spatial upsampling factor | `phase_unwrapping.py` |
| `nprs_pi_unwrap` | bool | `true` | Config/GUI (shown only for `nprs`) | Enable NPRS π-unwrapping pass | `phase_unwrapping.py` |
| `nprs_auto_crop` | bool | `true` | Config/GUI (shown only for `nprs`) | Crop NPRS FFT padding before returning | `phase_unwrapping.py` |
| `backend_params` | object | `{}` | `configs/phase_unwrapping.json` | Backend-specific PUDIP-Flow or GUST-Flow constructor options such as `num_iter`, `lr`, and `num_primitives` | `phase_unwrapping.py` |

When omitted, the learned backends use the upstream notebook settings: PUDIP-Flow uses `level=4`, `features=128`, `input_depth=128`, `lr=1e-3`, `num_iter=1000`, unit TV weights, `loss_type="l1"`, cosine scheduling, `div_weight=0`, and `reshape_mode="bt_as_channel"`; GUST-Flow uses `num_iter=1000` and `num_primitives=8192`.

## Outputs

`summary.json` records method, device, elapsed time, and wrap statistics. `phase_unwrap.npz` contains wrapped/unwrapped phase, flow, `wrap_count`, `wrap_mask`, and `mask_used`.

## Limitations

`gc3D` still solves max-flow on CPU; CUDA only accelerates masked graph construction, so gains are modest. `nprs` keeps the skimage reliability solver on CPU and accelerates Fourier resampling on CUDA. PUDIP-Flow is iterative and can be slow; GUST-Flow requires Python 3.10+, CUDA, its CuPy runtime, and enough VRAM for its Gaussian primitives. A CUDA/CuPy/PyTorch mismatch or VRAM exhaustion is reported as a backend error; use `lap4D` as the fallback. `pcmra_std` is derived from the temporal PC-MRA volume (`mag × |wrapped phase|`) and is used as the learned-backend confidence/weight map. Wrap locations are estimates, not ground truth.

When the `pu` extra is not installed, PUDIP-Flow and GUST-Flow are hidden from the GUI and are not accepted by the CLI; a direct Python call reports that the optional dependencies must be installed.

## Where to change code

Backend adapter: `autoflow/algorithms/phase_unwrapping.py`; upstream sources: `third_party/PUDIP-Flow` and `third_party/GUST-Flow`; pipeline/state: `autoflow/core/pipeline.py` and `autoflow/core/models.py`; GUI: `autoflow/ui/app.py` and `autoflow/ui/ortho_viewer.py`.

## Tests

`pytest tests/test_smoke_phantoms.py tests/test_pressure_gradient_phantom.py -q`

## Common problems

“wrapped phase unavailable” means an input has neither stored phase nor a reconstructable velocity/VENC pair. Normalized H5 and DICOM velocity inputs now reconstruct a wrapped phase automatically. `DV` dual-VENC input intentionally skips this step; an `LV` or `HV` selection from the GUI keeps its selected single-VENC wrapped phase available.
