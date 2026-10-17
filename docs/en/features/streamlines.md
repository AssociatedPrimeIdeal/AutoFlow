# Instantaneous streamlines

## Status

Supported in GUI. Partial in CLI and public Python API: offline streamline video export.
The two slice overlays described below are **GUI only**.

## What it does

Integrates instantaneous velocity trajectories inside the vessel. Live trajectories use unlit velocity-coloured tubes; colour is the magnitude of the same interpolated vector used by the tracer. Both the GUI and offline movies use screen-space line-tube rendering and phase caches, with the same unlit `turbo` colours, layer opacity and Browser line width (2 px by default). The movie no longer expands a second thick tube mesh. Offline videos crop velocity grids to the vessel, prepare phases with up to eight workers, and retain meshes until encoding completes. Automatic colour limits use 0 to the finite vessel-velocity 99th percentile across phases.

The 2D viewer also supports two independent overlays on any scalar background.
**2D streamlines** integrate the two velocity components tangent to the
displayed slice, using a midpoint integrator with steps measured in mm.
**3D projection** clips existing instantaneous 3D streamline segments
to a slab centered on the slice and projects only those pieces. Arrows indicate
flow direction, using the sampled velocity to orient backward VTK traces.
Both default to total-speed colors in cm/s; fixed cyan/orange colors remain
selectable, as do separate colormaps, manual limits and in-plane speed colors.
Both modes work in standard axial/coronal/sagittal views and
selected-plane `U × V`, `V × N`, and `U × N` views, and follow the cardiac phase.

## When to use it

Use for qualitative instantaneous flow patterns. Use [pathlines](pathlines.md) for time-dependent particle travel.
Use planar streamlines to inspect local tangential flow; use slab projection to
see how nearby 3D trajectories relate to the image. The two overlays can be shown together.

## Quick use

Right-click **Generate Streamlines** for seed ratio, maximum steps and terminal speed (m/s), then Save and run. **Pathlines** has separate settings. Save retains existing results until rerun; Cancel discards draft changes.

GUI: click **Generate Streamlines**, or run **Hemodynamics → Run All**; review the Browser and timeline.

In the 2D viewer, enable **2D streamlines** without generating 3D trajectories.
Open **Display settings → Flow** to adjust planar seed density and appearance. If
segmentation is available, integration stays inside the current frame's mask;
otherwise it uses finite, above-threshold velocities in the displayed domain.
To show **3D projection**, first click **Generate Streamlines**, then enable
the checkbox and set **Full slab thickness** in the dock (mm). The same current-phase
cache is used even if the 3D Browser layer is hidden. Each mode has its own
opacity, line width, arrows and color settings, independent of segmentation
and the image colorbar. Clearing a
plane selection restores overlays on the standard slices. Both checkboxes
start off and are reset when the viewer is reset for another case.

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

CLI and public Python API do not expose or export the 2D viewer overlays.

## Inputs

Velocity `XYZT3` in cm/s, segmentation, spacing and origin. Tracer velocity is converted to m/s.
Planar overlays require velocity and spacing; segmentation is optional. Slab
projection requires generated 3D streamlines. Selected-plane centers are local
physical coordinates; world origin is applied once at the VTK boundary, so
nonzero-origin inputs retain alignment with the slice images.

## Parameters

[All streamline controls](../user/parameters.md#streamlines), [video settings](../user/parameters.md#video_exporting), and [colourbar layout](../user/parameters.md#colorbar). `terminal_speed` is m/s. `tube_radius` is retained for physical tube geometry on secondary / plane streamline objects; the main live layer and movies use screen-space line width. Seed randomness is reproducible via `rng_seed`.

| Parameter / control | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `2D streamlines`, `3D projection` | bool | off | 2D viewer quick toggles or Flow tab | Independent overlays, including simultaneous use | `autoflow/ui/ortho_viewer.py` |
| `color_mode`, `color` | enum, color | speed; fixed cyan for planar, orange for projection | Flow tab, per mode/view | speed colors or a user-chosen uniform color | `autoflow/ui/slice_view.py` |
| `quantity`, `cmap` | enums | total speed, turbo | Flow tab, per mode/view | total or in-plane speed; planar integration still discards the normal component | `autoflow/ui/viewer_display.py`, `autoflow/ui/slice_view.py` |
| `range_mode`, `low`, `high` | enum, floats cm/s | across-phase P99, 0, 100 | Flow tab, per mode/view | fixed across-phase automatic limits or manual limits; legends identify quantity and units | `autoflow/ui/viewer_display.py` |
| `density` | float, 0.25–3 | 1 | Flow tab, planar | seed spacing `4 × min(pixel spacing) / density`; capped at 512 seeds/view | `autoflow/algorithms/slice_streamlines.py` |
| `fraction` | float, 0.01–1 | 1 | Flow tab, projection | display only a fraction of retained 3D pieces; does not generate more trajectories | `autoflow/ui/ortho_viewer.py` |
| `slab` | float, 0.1–100 mm | 3 mm | Flow tab, projection | full thickness centered on each slice; clips segment crossings at both boundaries | `autoflow/algorithms/slice_streamlines.py` |
| `opacity`, `line_width` | float 0–1, float 0.25–8 px | 0.85, 1.5 px | Flow tab, per mode/view | independent overlay opacity and cosmetic line width | `autoflow/ui/slice_view.py` |
| `arrows`, `arrow_size`, `arrow_spacing` | bool, floats mm | true, 1 mm, 8 mm | Flow tab, per mode/view | direction glyph visibility, size and arc-length spacing | `autoflow/ui/slice_view.py` |
| `terminal_speed` | float, cm/s | 1 | Flow tab, planar | minimum tangential speed for 2D integration; main 3D tracer retains its existing m/s parameter | `autoflow/algorithms/slice_streamlines.py` |
| `max_steps`, `max_length` | int 1–800, float mm | 200, 0 (step-limited) | Flow tab, planar | steps per direction and optional displayed arc-length cap | `autoflow/algorithms/slice_streamlines.py`, `autoflow/ui/ortho_viewer.py` |
| `min_length` | float mm | 0 | Flow tab, per mode/view | hide short visible pieces | `autoflow/ui/slice_view.py` |

Overlay controls are saved under `render_settings.viewer_display` in the
workspace; reusable styles can be saved as local defaults. They add no CLI or
public configuration fields. See [Viewer display settings](viewer-display.md).

## Outputs

Cached GUI streamline polylines/display meshes and requested offline videos. See [Outputs](../user/outputs.md) for video names.
Slice-overlay geometry is cached for display only; it creates no extra output files.

Interactive live 3D streamlines and their slice projections share a background
mesh request manager. Missing phases appear after computation finishes; playback
coalesces pending requests and ignores results for obsolete inputs or parameters.
The numerical integration, seeds and output geometry are unchanged. The 2D
integrator evaluates only surviving seeds at each existing RK2 step, preserving
trajectory order and stopping rules. CLI/offline export keeps its existing
synchronous numerical contract.

## Limitations

Instantaneous streamlines do not track particles through changing phases. Offline phase meshes consume memory. Geometry and colour remain qualitative unless reviewed against calibrated measurements.
Planar integration discards the normal component and does not describe particles
remaining in the slice. A projected 3D trajectory need not follow the in-plane
velocity. Fixed colors identify modes; speed colors use independent flow
legends, while the original colorbar still describes the scalar background.
Very short projected pieces may have no arrow. First
display of an uncached 3D phase still requires its normal 3D integration.

## Where to change code

`autoflow/algorithms/streamlines.py`; workflow/cache: `autoflow/core/pipeline.py`, `autoflow/ui/viewer.py`; export: `autoflow/rendering/videos.py`, `autoflow/processing.py`.
Planar integration and slab clipping: `autoflow/algorithms/slice_streamlines.py`;
controls, sampling and phase caches: `autoflow/ui/ortho_viewer.py`; persistent
overlay curves/arrows: `autoflow/ui/slice_view.py`.

Live GUI mesh scheduling and obsolete-result rejection live in `autoflow/ui/streamline_jobs.py`; scene/slice consumers are `autoflow/ui/viewer.py` and `autoflow/ui/ortho_viewer.py`.

## Tests

Retained smoke/phantom coverage plus manual timeline, unlit colour, radius and camera interaction checks.
`tests/test_smoke_phantoms.py` also covers anisotropic planar flow, disconnected
mask domains, circular vortex direction, empty/invalid fields, slab crossings
with both endpoints outside, separated projection pieces, and both Qt modes in
axis/oblique views with nonzero origin, phase changes, opacity and reset.

```bash
conda activate autoflow311
PYVISTA_OFF_SCREEN=true python -m pytest tests/test_smoke_phantoms.py tests/test_pressure_gradient_phantom.py -q
```

## Common problems

No trajectories: check segmentation and flow. Dark/subpixel tubes: use unlit rendering, a suitable Browser line width and automatic limits; explicit broad colour limits can compress most velocities into dark colours. Phase changes should preserve active scalars and tube styling.
For an empty projected overlay, run **Generate Streamlines** first and check the
current phase and slab thickness. For an empty planar overlay, check tangential
speed and the current segmentation: purely through-plane flow has no planar
streamlines. Missing velocity disables both controls. Opening another case
requires enabling the overlays again.
