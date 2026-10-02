# Temporal pathlines

## Status

GUI only. CLI and public batch Python pathline export are Not implemented.

## What it does

Uses VTK `vtkParticleTracer` to follow particles released from cross-section planes at t=0 through one cardiac cycle. Prepared temporal velocity frames can be shared across planes within a bounded cache. Playback reveals cached trajectory prefixes without reintegrating. Worker threads compute data; scene actors and rendering update on the GUI thread.

## When to use it

Use to review time-dependent transport from one plane, selected planes or all planes.

## Quick use

GUI: generate planes, then choose **Pathlines**, a plane context-menu action, or **Hemodynamics → Run All**. Repeated requests select/reuse existing trajectories; unchanged planes accumulate pathlines. Use timeline playback, Browser visibility/opacity and **Set Pathline Color**.

CLI: no public pathline export command. [Streamline video](streamlines.md) is available for instantaneous trajectories.

Python: configure GUI defaults via `AutoFlowConfig.from_config_dir('./configs')`, then pass the configuration through the GUI workflow. Direct batch `run_case()` does not export pathlines.

## Inputs

Velocity, segmentation, planes, spatial calibration and RR. Plane changes invalidate launch geometry and cached paths.

## Parameters

[Every pathline parameter](../user/parameters.md#pathlines) and [API aliases](../user/api-parameters.md). Fixed mode requests `seed_count` (default 250); ratio mode uses the support fraction with a count ceiling. `terminal_speed` is m/s, tube radius is mm, cache size is MiB. Manual per-object colour overrides uniform/per-plane/per-group modes.

## Outputs

GUI trajectory arrays, phase samples and Browser objects retaining plane/group identity. No public batch file export.

## Limitations

A single cardiac cycle is traced from t=0. Seed support can reduce requested counts. Temporal frame preparation retains memory up to the cache budget. Regenerating/deleting planes clears dependent trajectories.

## Where to change code

`autoflow/algorithms/streamlines.py`, `autoflow/ui/app.py`, `autoflow/ui/viewer.py`; defaults: `autoflow/config.py`, `autoflow/core/models.py`.

## Tests

Use the retained smoke/phantom suite. Manually verify per-plane reuse, all-plane accumulation, timeline prefixes, colour overrides and rendering on the GUI thread.

```bash
conda activate autoflow311
PYVISTA_OFF_SCREEN=true python -m pytest tests/test_smoke_phantoms.py tests/test_pressure_gradient_phantom.py -q
```

## Common problems

No paths: check planes and flow support. Invalid RR/spacing/flow: correct inputs before tracing. Remote OpenGL/thread errors: use the current renderer and validate the local/SSH rendering environment.
