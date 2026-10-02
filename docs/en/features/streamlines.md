# Instantaneous streamlines

## Status

Supported in GUI. Partial in CLI and public Python API: offline streamline video export.

## What it does

Integrates instantaneous velocity trajectories inside the vessel. Live trajectories use unlit velocity-coloured tubes; colour is the magnitude of the same interpolated vector used by the tracer. The GUI uses line-tube rendering and phase caches. Offline videos crop velocity grids to the vessel, prepare phases with up to eight workers, and retain meshes until encoding completes. Automatic colour limits use 0 to the finite vessel-velocity 99th percentile across phases.

## When to use it

Use for qualitative instantaneous flow patterns. Use [pathlines](pathlines.md) for time-dependent particle travel.

## Quick use

GUI: click **Generate Streamlines**, or run **Hemodynamics → Run All**; review the Browser and timeline.

CLI:

```bash
autoflow-run case.h5 --output-dir results/case --video streamlines
```

Python:

```python
from autoflow import AutoFlowConfig, run_case
config = AutoFlowConfig.from_config_dir('./configs', requested_videos=['streamlines'])
summary = run_case('case.h5', config=config)
```

## Inputs

Velocity `XYZT3` in cm/s, segmentation, spacing and origin. Tracer velocity is converted to m/s.

## Parameters

[All streamline controls](../user/parameters.md#streamlines), [video settings](../user/parameters.md#video_exporting), and [colourbar layout](../user/parameters.md#colorbar). `terminal_speed` is m/s; tube radius is mm. Seed randomness is reproducible via `rng_seed`.

## Outputs

Cached GUI streamline polylines/display meshes and requested offline videos. See [Outputs](../user/outputs.md) for video names.

## Limitations

Instantaneous streamlines do not track particles through changing phases. Offline phase meshes consume memory. Geometry and colour remain qualitative unless reviewed against calibrated measurements.

## Where to change code

`autoflow/algorithms/streamlines.py`; workflow/cache: `autoflow/core/pipeline.py`, `autoflow/ui/viewer.py`; export: `autoflow/rendering/videos.py`, `autoflow/processing.py`.

## Tests

Retained smoke/phantom coverage plus manual timeline, unlit colour, radius and camera interaction checks.

```bash
conda activate autoflow311
PYVISTA_OFF_SCREEN=true python -m pytest tests/test_smoke_phantoms.py tests/test_pressure_gradient_phantom.py -q
```

## Common problems

No trajectories: check segmentation and flow. Dark/subpixel tubes: use unlit rendering, a suitable tube radius and automatic limits; explicit broad colour limits can compress most velocities into dark colours. Phase changes should preserve active scalars and tube styling.
