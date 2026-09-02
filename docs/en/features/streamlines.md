# Feature: Streamlines And Pathlines

## Status

| Entry point | Status | Notes |
| --- | --- | --- |
| GUI streamlines | Supported | live streamline scene objects |
| GUI pathlines | Supported | time-resolved plane-launched pathlines |
| CLI streamlines video | Partial | offline streamline video export only |
| CLI pathlines | Not implemented | no public batch pathline export |
| Python API | Partial | offline streamline video through batch config |

## What It Does
Streamlines show instantaneous flow trajectories. Pathlines show time-resolved particle travel launched from planes in the GUI. In grouped multi-label workflows, each generated pathline keeps the plane group name internally, while the browser shows short visible names such as `pathline 5`.

## When To Use It
- use streamlines for qualitative inspection of instantaneous flow patterns
- use pathlines for time-resolved trajectories from a plane in the GUI
- use CLI or Python API when you need offline streamline videos

## Quick Use

### GUI
1. load a segmented case
2. click `Generate Streamlines` for live streamlines, or use `Hemodynamics -> Run All` to include it after metrics
3. click `Pathlines` to generate all planes, or use a plane context menu for one plane or a selected subset; generated plane pathlines accumulate, and repeated requests skip integration for existing planes. Fixed seed mode launches the configured count (default `250`) per plane, while Ratio mode uses `Seed Ratio` with the configured count as a ceiling. Repeating a one-plane request selects the existing pathline in the Browser. `Hemodynamics -> Run All` also generates pathlines for every available plane after enabling live streamlines.
4. use the left browser to show or hide a whole group of pathlines, or right-click an individual pathline and use `Set Pathline Color` to change only that pathline
5. drag the timeline to inspect image and other dynamic layers; pathlines reveal the cached trajectory prefix reached by particles released at `t=0`, without reintegrating them

Pathline generation always releases particles at `t=0` and integrates their
trajectory through one cardiac cycle. Plane seed geometry and per-point phase
samples are stored with that trajectory. Timeline playback only rebuilds the
visible prefix from those cached samples; it does not launch another VTK
integration.
Pathline integration is performed by VTK's `vtkParticleTracer` over a temporal
velocity source; AutoFlow does not implement a separate numerical integrator.
The worker only computes data; pathline actors and the VTK render window are
updated on the GUI thread after completion.
`Pathline Seed Mode` chooses predictable fixed seed counts or a ratio of the
plane cross-section. The count field is the fixed count in `Fixed Count` mode
and a safety ceiling in `Ratio` mode. `Temporal Cache (MiB)` bounds the extra
VTK frame cache used for an all-plane job: when one complete cardiac cycle fits
within the configured budget, all planes reuse those prepared velocity frames.
Otherwise VTK keeps only the bounded rolling cache to avoid an excessive memory
allocation.

`Pathline Color Mode` defaults to `Per Plane`, assigning a stable categorical
color from the plane index. `Per Group` shares a stable color by vessel group,
and `Uniform` uses `Uniform Pathline Color`. A color set manually in the
Browser overrides all three automatic modes for that pathline.

### Benchmark

Run the controlled VTK benchmark from the repository root:

```bash
PYVISTA_OFF_SCREEN=true ~/miniconda3/envs/ryy/bin/python tools/benchmark_pathlines.py
```

It reports three otherwise identical all-plane workloads: 500 fixed seeds with
one temporal source per plane, 250 fixed seeds with one source per plane, and
250 fixed seeds with a shared all-frame source. The reported deltas separate
the seed-count reduction from temporal-frame reuse. Results are hardware and
case dependent; use the real case for a deployment decision. `--json` emits
the same measurements for archival or CI comparison.

Velocity-colored streamlines use unlit tube colors in the live GUI. Each
streamline point is colored by the magnitude of the interpolated velocity
vector used by the tracer, rather than by separately interpolating voxel speed
magnitudes. This keeps the displayed value physically consistent with the
trajectory and prevents tube orientation from darkening the quantitative color.

The default streamline range is `auto`. Auto mode uses `0` to the 99th
percentile of finite velocities inside the segmentation across all cardiac
phases. The GUI and exported video therefore keep one comparable range while
isolated dual-VENC outliers cannot compress most of a case into the darkest end
of the color map.

### CLI

```bash
autoflow-run case.h5 --output-dir results/case --video streamlines
```

### Python API

```python
from autoflow import AutoFlowConfig, run_case
config = AutoFlowConfig(
    output_dir="./results/case",
    requested_videos=["streamlines"],
)
summary = run_case("case.h5", config=config)
```

## Inputs

| Input | Required | Meaning |
| --- | --- | --- |
| flow | yes | velocity field |
| segmentation | yes | seed-mask source |
| planes | for pathlines | generated planes; the main GUI action uses all planes, while the context menu can target one plane or a selected subset |
| RR interval | for pathlines | time step scaling |

## Parameters

| Parameter | Type | Default | Where set | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `seed_ratio` | float | `0.02` | `configs/streamlines.json` | streamline seed density | `autoflow/core/models.py` |
| `max_steps` | int | `2000` | `configs/streamlines.json` | integration steps | `autoflow/core/models.py` |
| `min_seeds` | int | `50` | `configs/streamlines.json` | minimum number of seeds | `autoflow/core/models.py` |
| `pathlines.seed_ratio` | float | `0.2` | `configs/pathlines.json` or GUI | cross-section sampling ratio in `ratio` mode | `autoflow/core/models.py`, `autoflow/algorithms/streamlines.py` |
| `pathlines.min_seeds` | int | `50` | `configs/pathlines.json` or GUI | lower seed bound in `ratio` mode | `autoflow/core/models.py`, `autoflow/algorithms/streamlines.py` |
| `pathlines.seed_mode` | `fixed` or `ratio` | `fixed` | `configs/pathlines.json` or GUI | choose exactly the configured seed count per plane, or derive seed count from the cross-section candidate ratio | `autoflow/core/models.py`, `autoflow/algorithms/streamlines.py` |
| `pathlines.seed_count` | int | `250` | `configs/pathlines.json` or GUI | seed count in `fixed` mode; maximum seed count in `ratio` mode | `autoflow/core/models.py`, `autoflow/algorithms/streamlines.py` |
| `pathlines.max_steps` | int | `200` | `configs/pathlines.json` or GUI | maximum cardiac-frame updates per pathline; values above the available frame count trace the full cycle | `autoflow/algorithms/streamlines.py`, `autoflow/ui/app.py` |
| `pathlines.terminal_speed` | float | `0.01` | `configs/pathlines.json` or GUI | stop threshold in m/s | `autoflow/core/models.py`, `autoflow/algorithms/streamlines.py` |
| `pathlines.rng_seed` | int | `0` | `configs/pathlines.json` or GUI | deterministic plane-seed selection | `autoflow/core/models.py` |
| `pathlines.tube_radius` | float | `0.25` | `configs/pathlines.json` or GUI | rendered pathline tube radius in mm | `autoflow/core/models.py`, `autoflow/ui/viewer.py` |
| `pathlines.color_mode` | `uniform`, `per_plane`, or `per_group` | `per_plane` | `configs/pathlines.json` or GUI | choose one color, deterministic categorical colors by plane, or deterministic colors by vessel group | `autoflow/core/models.py`, `autoflow/ui/app.py` |
| `pathlines.color` | string | `deepskyblue` | `configs/pathlines.json` or GUI | color used by `uniform` mode; each pathline can later be recolored individually in the Browser | `autoflow/ui/app.py` |
| `pathlines.temporal_cache_mb` | float | `512` | `configs/pathlines.json` or GUI | maximum additional VTK temporal-frame cache for reusing velocity frames across an all-plane task; `0` keeps the rolling cache only | `autoflow/algorithms/streamlines.py`, `autoflow/ui/app.py` |
| `streamlines.render.clim` | list[float, float] or `null` | `null` | `configs/streamlines.json` | explicit streamline display range in GUI and videos; `null` uses `0` to the all-phase P99 segmented velocity | `autoflow/algorithms/streamlines.py`, `autoflow/ui/viewer.py`, `autoflow/rendering/videos.py` |
| `streamlines.render.show_scalar_bar` | bool | `True` | `configs/streamlines.json` | show or hide the streamline colorbar in GUI and videos | `autoflow/rendering/videos.py`, `autoflow/core/pipeline.py` |
| GUI runtime streamline `clim` and colorbar layout | floats or `auto` | config default | GUI `Render / Colorbar` panel | update live streamline display immediately and reuse the same values for video export in the current session; the live GUI scene reuses the same shared colorbar slot as other scalar layers, with shared defaults from `configs/colorbar.json` | `autoflow/ui/app.py`, `autoflow/ui/viewer.py` |

## Outputs

| Output file or object | Created when | Meaning |
| --- | --- | --- |
| live streamline scene object | GUI streamline step succeeds | dynamic streamline object |
| live pathline scene objects | GUI pathline step succeeds | one object for each selected launch plane, each with its own visibility and color |
| `streamlines_video.mp4` or `streamlines_rotate.mp4` | streamlines video export runs | offline streamline movie |

## Limitations
- segmentation is required
- public batch pathline export is not implemented
- pathlines are currently a GUI-centered workflow
- initial plane-launched pathline integration runs in a background worker and caches the complete `t=0` trajectory, per-point phase samples, and plane seeds before updating the 3D scene; all-plane work also reuses prepared VTK velocity frames only when the configured memory budget covers a full cycle
- pathlines accumulate while plane geometry is unchanged; regenerating or deleting planes clears them because their launch geometry and indices are no longer stable
- streamline video export crops each vector field to the segmented vessel extent, precomputes cardiac phases with up to eight CPU workers, and retains streamline/tube meshes until encoding finishes; this favors speed over memory
- the GUI caches both raw streamlines and their tube display meshes by cardiac phase, trading memory for fast timeline revisits and consistent styling

## Where To Change Code

| Change you want | Edit here | Also check | Tests |
| --- | --- | --- | --- |
| streamline generation | `autoflow/algorithms/streamlines.py` | `autoflow/core/pipeline.py` | `tests/test_smoke_phantoms.py` |
| pathline behavior | `autoflow/algorithms/streamlines.py`, `autoflow/ui/app.py` | `autoflow/ui/viewer.py` | manual verification plus smoke and phantom regression only |
| streamline video export | `autoflow/rendering/videos.py` | `autoflow/processing.py` | `tests/test_smoke_phantoms.py` |

## Tests

- `~/miniconda3/envs/ryy/bin/python -m pytest tests/test_smoke_phantoms.py -q`
- GUI pathline behavior outside this retained regression suite requires manual verification

## Common Problems

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| streamlines are skipped | no segmentation or no flow | load a segmented flow case |
| streamlines become thin lines after changing the timeline | an older GUI update path replaced tube geometry with raw polylines | reinstall the current build; initial and time-updated streamlines now share the cached tube-display path |
| nonzero streamline regions look gray or black despite a colored velocity scale | tube lighting is enabled, the tube is subpixel-thin, or a fixed range is much wider than the case velocity range | use the current unlit renderer, keep `tube_radius` near its `0.25 mm` default, and leave `streamlines.render.clim` as `null`; set explicit limits only for cross-case comparison |
| the ascending aorta is colored but the arch or descending aorta turns black after changing phase | an older dynamic update bypassed PyVista's active-scalar pipeline, leaving the new tube mesh attached to stale scalar texture state | reinstall the current build; phase changes now reconnect the mapper dataset while preserving the configured velocity range and colorbar |
| pathlines are skipped | no planes | generate planes first |
| generating a pathline reports invalid flow, segmentation, spacing, origin, or RR values | VTK received a malformed temporal input before tracing | regenerate the plane after checking the loaded case metadata and segmentation; the GUI now reports this as a pathline error instead of terminating the process |
| `eglMakeCurrent: 12290`, `Timers cannot be started from another thread`, or a segfault appears after pathline generation | a render-window update ran from the pathline worker thread | use the current build; pathline completion now queues all scene and render-window updates to the GUI thread |
| pathlines look mixed across vessels in the browser | grouped labels were not configured as expected | check `configs/labels.json -> label_groups` and regenerate planes and pathlines |
