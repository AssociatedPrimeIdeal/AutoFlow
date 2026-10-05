# Phase Unwrapping

## Status
Supported as an optional GUI, CLI, and Python workflow step for the bundled backends; PUDIP-Flow and GUST-Flow are Experimental. Explicit action in GUI; CLI/API remain opt-in, while `--correction` / `correction_all=True` default to `lap4D` with no mask; `DV` dual-VENC inputs are skipped while `LV` and `HV` selections can be unwrapped.

## What it does

Runs a selected phase-unwrapping backend on wrapped phase (radians), converts the result to velocity, and records signed wrap count `k` plus affected voxels. The bundled `gc3D`, `lap4D`, and `nprs` methods remain available; optional `pudip` and `gust` methods use the `third_party/PUDIP-Flow` and `third_party/GUST-Flow` git submodules.

The LAP4D article and upstream implementation are listed in [Scientific references: phase unwrapping](../references/index.md#phase-unwrapping). That citation does not cover every available backend.

## When to use it

Use for single-VENC data containing phase wraps or for a dual-VENC case loaded as `LV` or `HV`. `DV` dual-VENC inputs are skipped automatically. Use PUDIP-Flow or GUST-Flow when a learned GPU reconstruction is appropriate and its training cost is acceptable.

## Quick use

GUI: open **Correction > Phase Unwrapping**; the method defaults to `lap4D` and the mask to `none`. Click **Unwrap Phase** to run only this stage, or **Run All** to run Background Correction → Noise Removal → Unwrap Phase → Generate PC-MRA. PUDIP/GUST training settings come from `phase_unwrapping.json` under `backend_params`. Review **Estimated Wrap Locations** and **Phase Wrap Count** in the Browser.

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

The loader must provide canonical `phase_wrapped` (`X,Y,Z,T,3`). Normalized H5 and DICOM velocity inputs reconstruct wrapped phase modulo `2π` from velocity and VENC. Mask choices depend on the method: `gc3D`, `lap4D` and `nprs` allow `none` (default) or `segmask`; `pudip` and `gust` allow `pcmra_std` (default), `pcmra_mean`, `none` or `segmask`. GUI disables `segmask` until an active segmentation exists; direct calls reject an unavailable mask. PCMRA choices are computed from magnitude and wrapped phase independently of segmentation and the noise display mask. Learned backends receive `[component,time,x,y,z]` layout.

## Parameters

| Parameter / flag | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `mask_source` / `--phase-unwrap-mask` / API `phase_unwrap_mask` | string | `auto` | JSON, GUI, CLI/API | `auto` resolves by method as described above; `none` uses the whole volume | `autoflow/algorithms/phase_unwrapping/backends.py`, `autoflow/core/pipeline.py` |


See [phase unwrapping parameters](../user/parameters.md#phase_unwrapping), [CLI flags](../user/cli-parameters.md), and [API fields](../user/api-parameters.md) for complete type/default/unit/effect/owner tables. Dictionary controls are expanded in [Structured parameters](../user/parameter-schemas.md).

When omitted, the learned backends use the upstream notebook settings: PUDIP-Flow uses `level=4`, `features=128`, `input_depth=128`, `lr=1e-3`, `num_iter=1000`, unit TV weights, `loss_type="l1"`, cosine scheduling, `div_weight=0`, and `reshape_mode="bt_as_channel"`; GUST-Flow uses `num_iter=1000` and `num_primitives=8192`.

## Outputs

`summary.json` records method, selected mask source, device, elapsed time, diagnostic scope and wrap statistics. The 3-D Estimated Wrap Locations and Phase Wrap Count layers extract only nonzero corrected voxels; all zero-count background is absent. Locations include any corrected velocity component. Count colour represents the signed count from the component with largest absolute count, with X/Y/Z tie order; the complete three-component counts remain in the exported arrays. An empty phase removes its actor and returning to an active phase restores it. `phase_unwrap.npz` contains wrapped/unwrapped phase, flow, `wrap_count`, `wrap_mask`, and `mask_used`.

Correction updates `Workspace.flow_raw`, the working velocity used by later segmentation and analysis. Every unwrap run starts from stored wrapped phase, preventing repeated unwrapping of already recovered phase. A masked rerun preserves current velocity outside the mask; exported recovered phase matches that combined velocity, while wrap counts/statistics describe the latest mask (`diagnostic_scope=latest_masked_run`). `flow_input` retains the pre-unwrapping baseline (after saved corr is applied during GUI load or background correction is run), used by **Revert to Pre-Unwrapping Flow**.

Existing segmentation, skeleton, graph, paths, planes, metrics and trajectories are retained after unwrap or revert. They do not automatically recompute: rerun the desired downstream actions yourself. After segmentation, return to Correction and select `segmask` to run a masked refinement.

## Limitations

`gc3D` still solves max-flow on CPU; CUDA only accelerates masked graph construction, so gains are modest. `nprs` keeps the skimage reliability solver on CPU and accelerates Fourier resampling on CUDA. PUDIP-Flow is iterative and can be slow; GUST-Flow requires Python 3.10+, CUDA, its CuPy runtime, and enough VRAM for its Gaussian primitives. A CUDA/CuPy/PyTorch mismatch or VRAM exhaustion is reported as a backend error; use `lap4D` as the fallback. `pcmra_std` is derived from the temporal PC-MRA volume (`mag × |wrapped phase|`) and is used as the learned-backend confidence/weight map. Wrap locations are estimates, not ground truth.

When the `pu` extra is not installed, PUDIP-Flow and GUST-Flow are hidden from the GUI and are not accepted by the CLI; a direct Python call reports that the optional dependencies must be installed.

## Where to change code

Existing imports from `autoflow.algorithms.phase_unwrapping` remain valid. The package entry point re-exports the original functions; numerical work lives in the modules below.

| Change you want | Edit here | Also check | Tests |
| --- | --- | --- | --- |
| method aliases, allowed masks, dependency checks or device selection | `autoflow/algorithms/phase_unwrapping/backends.py` | `autoflow/cli.py`, `autoflow/ui/app.py` | `tests/test_smoke_phantoms.py` |
| phase/VENC shape validation or learned mask weights | `autoflow/algorithms/phase_unwrapping/_common.py` | `autoflow/core/pipeline.py` | smoke/phantom suite and a synthetic array check |
| dispatch, component order or wrap diagnostics | `autoflow/algorithms/phase_unwrapping/engine.py` | `autoflow/core/pipeline.py`, `autoflow/core/models.py` | smoke/phantom suite |
| PUDIP-Flow adapter | `autoflow/algorithms/phase_unwrapping/pudip.py` | `third_party/PUDIP-Flow` | synthetic adapter check; manual real-backend verification |
| GUST-Flow adapter or CUDA error messages | `autoflow/algorithms/phase_unwrapping/gust.py` | `third_party/GUST-Flow` | synthetic adapter check; manual CUDA verification |
| CPU/Torch Laplacian, NPRS resampling or masked graph recovery | `autoflow/algorithms/phase_unwrapping/laplacian.py`, `autoflow/algorithms/phase_unwrapping/nprs.py`, `autoflow/algorithms/phase_unwrapping/graphcut.py` | `autoflow/algorithms/phase_unwrapping/_puma.py`, `autoflow/algorithms/phase_unwrapping/fourier.py` | synthetic phase comparison plus smoke/phantom suite |
| shared total-field correction | `autoflow/algorithms/phase_unwrapping/_common.py` | `autoflow/algorithms/phase_unwrapping/cpu.py`, `autoflow/algorithms/phase_unwrapping/engine.py` | temporal synthetic phase comparison |
| CPU dispatch or local-gradient recovery | `autoflow/algorithms/phase_unwrapping/cpu.py`, `autoflow/algorithms/phase_unwrapping/brute.py` | `autoflow/algorithms/phase_unwrapping/engine.py` | synthetic legacy-mode comparison |
| pipeline state, exports or GUI workflow | `autoflow/core/pipeline.py`, `autoflow/core/models.py`, `autoflow/ui/app.py`, `autoflow/ui/ortho_viewer.py` | `autoflow/processing.py` | `tests/test_smoke_phantoms.py` and manual GUI review |

CPU and Torch implementations now share the same algorithm modules under `autoflow/algorithms/phase_unwrapping/`. `cpu.py` owns `unwrap_data` dispatch. All implementation and imports use this unified package. Numerical formulas are unchanged. `fourier.py` contains one copy of each shared Fourier helper, replacing the identical duplicate definitions in the former `fourierOperators.py`. Implementation modules import their owners directly and do not import the package facade.

For low-level CPU calls, use `from autoflow.algorithms.phase_unwrapping.cpu import unwrap_data`. Low-level modes (`lap3D`, `brute`, `gc4D`) remain available through `unwrap_data`; the GUI/CLI `method` choices remain those listed above. The old standalone traditional package has been removed without forwarding modules.

## Tests

Activate `autoflow311`, then run the retained suite:

```bash
conda activate autoflow311
python -m pytest tests/test_smoke_phantoms.py tests/test_pressure_gradient_phantom.py -q
```

For structural refactors, manually compare wrapped/unwrapped phase, velocity, signed wrap counts, masks and statistics on a small synthetic phase volume. Check learned adapters with lightweight stand-ins; full PUDIP/GUST fitting and actual CUDA execution require separate manual validation. Patch helpers at their implementation call site rather than on package re-exports. Keep automated tests within the two retained smoke/phantom files.

## Common problems

“wrapped phase unavailable” means an input has neither stored phase nor a reconstructable velocity/VENC pair. Normalized H5 and DICOM velocity inputs now reconstruct a wrapped phase automatically. `DV` dual-VENC input intentionally skips this step; an `LV` or `HV` selection from the GUI keeps its selected single-VENC wrapped phase available.

For CLI/API, standalone unwrapping with `segmask` is delayed until requested automatic segmentation has produced a mask when none was loaded. `--correction` requires its mask to exist before the group runs; use its default `none` or a learned PCMRA source for correction before segmentation.

PC-MRA is a stored display computed by the fourth Correction action. Returning to run only Unwrap Phase updates velocity but keeps the existing PC-MRA data until Generate PC-MRA is explicitly rerun. Run All finishes with that generation step.
