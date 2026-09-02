# Phase Unwrapping

## Status
Supported as an optional GUI, CLI, and Python workflow step. Disabled by default; dual-VENC inputs are always skipped.

## What it does

Runs bundled traditional `gc3D`, `lap4D`, or `nprs` on wrapped phase (radians), converts the result to velocity, and records signed wrap count `k` plus affected voxels.

## When to use it

Use for single-VENC data containing phase wraps. Dual-VENC inputs are skipped automatically.

## Quick use

GUI: open **Phase Unwrapping**, choose a method/mask/device, and click **Unwrap Phase**. Choosing a method is the opt-in action; the step is never run until that button is pressed. The Browser exposes **Estimated Wrap Locations** and **Phase Wrap Count**.

CLI:

    autoflow-run case.h5 --phase-unwrap-method lap4D --phase-unwrap-device auto

Python: set `AutoFlowConfig(phase_unwrap_method="lap4D")` and call `run_case`. Omit the method (or leave it empty) to skip the step.

## Inputs

The loader must provide canonical `phase_wrapped` (`X,Y,Z,T,3`). Segmentation is the default mask; `--phase-unwrap-mask all` evaluates every voxel.

## Parameters

| Parameter | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `phase_unwrap_method` | enum | omitted | Config/CLI/GUI | Select `gc3D`, `lap4D`, or `nprs`; selecting a method opts in | `autoflow/algorithms/phase_unwrapping.py` |
| `phase_unwrap_device` | enum | `auto` | Config/CLI/GUI | CPU or CUDA (`lap4D` FFT, `nprs` FFT, or `gc3D` graph construction) | same |
| `phase_unwrap_mask` | enum | `segmentation` | Config/CLI/GUI | Active mask or all voxels | pipeline |
| `lap4d_ts` | float | `2.0` | Config/GUI (shown only for `lap4D`) | Temporal weighting for lap4D | `phase_unwrapping.py` |
| `nprs_upsampling_factor` | int | `2` | Config/GUI (shown only for `nprs`) | NPRS spatial upsampling factor | `phase_unwrapping.py` |
| `nprs_pi_unwrap` | bool | `true` | Config/GUI (shown only for `nprs`) | Enable NPRS π-unwrapping pass | `phase_unwrapping.py` |
| `nprs_auto_crop` | bool | `true` | Config/GUI (shown only for `nprs`) | Crop NPRS FFT padding before returning | `phase_unwrapping.py` |

## Outputs

`summary.json` records method, device, elapsed time, and wrap statistics. `phase_unwrap.npz` contains wrapped/unwrapped phase, flow, `wrap_count`, `wrap_mask`, and `mask_used`.

## Limitations

`gc3D` still solves max-flow on CPU; CUDA only accelerates masked graph construction, so gains are modest. `nprs` keeps the skimage reliability solver on CPU and accelerates Fourier resampling on CUDA. Wrap locations are estimates, not ground truth.

## Where to change code

Backend: `autoflow/algorithms/phase_unwrapping.py`; pipeline/state: `autoflow/core/pipeline.py` and `autoflow/core/models.py`; GUI: `autoflow/ui/app.py` and `autoflow/ui/ortho_viewer.py`.

## Tests

`pytest tests/test_smoke_phantoms.py tests/test_pressure_gradient_phantom.py -q`

## Common problems

“wrapped phase unavailable” means the input contains velocity only. “dual-VENC input” is intentional and skips this step.
