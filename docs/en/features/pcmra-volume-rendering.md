# Content volumes and PC-MRA

## Status

| Entry point | Status | Notes |
| --- | --- | --- |
| GUI | Supported | Add any available scalar Content to Browser and show it as a volume, maximum-intensity projection or orthogonal planes. |
| CLI | Partial | `--generate-pcmra` or `--correction` creates numerical PC-MRA; interactive Content rendering is GUI only. |
| Python API | Partial | Numerical PC-MRA and workspace persistence are supported; the public API has no interactive renderer settings. |

## What it does

**3D Render** beside **Content** creates a checked **3D Volume** object in the
left Browser. Each Content has its own object, visibility, opacity, colormap,
window and rendering mode. Clicking again reveals/reuses that object. Magnitude,
velocity components, speed and available derived/correction/unwrap fields share
the same scalar source as the 2D viewer. No unavailable metric is synthesized.
Through-plane Flow binds to the selected measurement plane when added.
Opening another input binds all Content rendering to the new case. The button
selects and scrolls to its Browser object; the status bar and Log report success
or a rendering failure.

**Correction → Generate PC-MRA** computes `magnitude_t × ||velocity_t||` in every
cardiac frame. After successful generation the GUI switches Content to PC-MRA,
adds/selects its checked 3D layer, and preserves the visibility of other layers.
Correction **Run All** performs the same handoff after its final generation
stage. Regeneration replaces the numerical PC-MRA and keeps its render settings.
The [scientific reference](../references/index.md#noise-masking-and-pc-mra)
describes the formula and its distinction from static PC-MRA methods.

For the selected Browser object, **Show 3 orthogonal planes** displays LR,
AP and FH cuts. Clicking an axis checkbox or moving its slider from Volume/MIP
mode switches directly to that single cut; the controls remain available.
**Render → Orthogonal planes** also selects the cut display.
Drag the colored axis handles in the 3D viewer to move them;
each handle changes only its own coordinate. The matching sliders provide the
same positioning. Uncheck individual axes to display a single anatomical cut
with colored streamlines, as in a slice-plus-flow figure. Axis handles belong
to the selected visible object and disappear when it is hidden or deleted.
The timeline updates visible scalar layers while retaining their cut positions
and the camera. Display changes leave intensity, velocity and metric arrays intact.
Camera, window/level and slice drags use a temporary lower-cost display preview.
Releasing the mouse restores the original sampling, antialiasing and image size.
These preview settings never resample the stored source arrays or change metrics.
Browser **Selection on top** is enabled by default. Select the streamline layer
to draw it in front of the cuts. Uncheck the option to restore normal depth
occlusion. Selecting a group brings its children forward; opacity and visibility
still apply. Color legends stay above the 3D layers.

## When to use it

Use a magnitude cut for anatomical context, PC-MRA for magnitude-weighted flow
contrast, and scalar field volumes/cuts to inspect metric distributions. Keep
streamlines visible to compare trajectories with anatomy. The slice display
adds no velocity/time plot.

## Quick use

GUI:

1. Load an H5 or DICOM case, choose **Content**, and click **3D Render**.
2. Select the resulting **3D Volume** item in Browser. Check/uncheck it to show/hide it.
3. Below Browser, click **Show 3 orthogonal planes**, or click LR/AP/FH to open
   one cut immediately. **Render** also selects Volume, Maximum intensity or
   Orthogonal planes. Adjust colormap and opacity.
4. Drag the red LR, green AP and blue FH handles in 3D, or move their sliders.
   In cut mode the axis checkboxes choose which cuts remain visible.
5. For a slice-plus-flow composition, keep one cut checked, generate streamlines,
   select their Browser layer with **Selection on top** checked, and adjust
   the 3D background and other Browser layers as desired.
6. Use the existing **Window/Level** sliders below the 3D view. On visible volume
   layers, Shift + left drag also changes window/level. Other mouse gestures
   retain the normal camera controls.

To generate PC-MRA, use **Correction → Generate PC-MRA** or **Run All**. Loading
alone does not compute PC-MRA. Its 2D Content and checked Browser layer are
selected automatically on completion. See [Viewer display settings](viewer-display.md)
for the independent slice-display controls.

CLI: `autoflow-run case.h5 --generate-pcmra`, or `--correction` for all four
correction actions. Normal batch segmentation prerequisites still apply.

Python API: `run_case("case.h5", config=AutoFlowConfig(generate_pcmra=True))`;
for a loaded workspace use
`PipelineEngine().run_step(ws, StepId.GENERATE_PCMRA, print)`.
Interactive rendering uses the GUI; snapshots retain its settings.

## Inputs

| Input | Required | Meaning |
| --- | --- | --- |
| Selected scalar Content | yes | The source must already exist; only available sources appear in the menu. |
| `resolution`, `origin` | yes | Image geometry in mm; cuts use voxel centers and normalized LR/AP/FH axes. |
| `mag_raw`, `flow_raw` | for PC-MRA generation | Static/4D magnitude and normalized XYZT3 velocity in cm/s. |
| Selected measurement plane | for Through-plane Flow | Its index/normal defines the scalar projection. |
| `pcmra_render_mask` | optional | Retained 3D/4D region for PC-MRA rendering only. |

## Parameters

| Parameter / control | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `mode` / Render | enum | composite | Selected Browser 3D Volume | Volume, maximum intensity, or orthogonal planes | `ui/content_controls.py`, `ui/content_rendering.py` |
| `planes` / LR, AP, FH | three bools | all true | Selected Browser 3D Volume | Click in Volume/MIP to open a single cut; in cut mode toggles visibility; at least one remains checked | `ui/content_controls.py` |
| Show 3 orthogonal planes | action | no change until clicked | Selected Browser 3D Volume | Enables all three cuts and 3D drag handles | `ui/content_controls.py` |
| `positions` | three floats 0–1 | `[0.5, 0.5, 0.5]` | Colored 3D handles or axis sliders | From first to last voxel center along each axis; clamps at image limits | `ui/content_rendering.py` |
| `handles` | bool | true | Drag colored axes in 3D | Handles for the selected visible orthogonal-plane object | `ui/content_rendering.py` |
| `cmap` | enum | Content palette; gray for magnitude/PC-MRA | Selected Browser 3D Volume | Scalar palette; independent of 2D palette | `ui/content_controls.py`, `ui/viewer.py` |
| Opacity | float 0–1 | 0.85 | Browser opacity slider / context menu | Plane transparency or volume transfer-function scale | `ui/app.py`, `ui/viewer.py` |
| Window / Level | numeric range | automatic current-frame range; signed fields remain centered on zero | Sliders below 3D viewer / Shift-drag on volumes | Scalar window; Auto restores current-frame range | `ui/app.py`, `ui/viewer.py` |
| `threshold` / Transparent below | float 0–0.99 | 0.05 | Selected Browser 3D Volume | Low-signal transparency; signed fields use absolute value around zero | `ui/viewer.py` |
| `shade` | bool | false | Selected Browser 3D Volume | Volume lighting; orthogonal cuts stay unlit | `ui/viewer.py` |
| `plane_index` | optional int | selected plane at creation | Through-plane Content binding | Plane-normal projection source for this layer | `rendering/content.py`, `ui/content_rendering.py` |
| `--generate-pcmra` / `generate_pcmra` | bool | false | CLI / API; explicit GUI action | Generate after requested corrections | `processing.py`, `core/pipeline.py` |
| PC-MRA automatic range | percentiles | positive finite P5 to P99 of current retained frame | automatic | Foreground window/level | `ui/viewer.py` |
| Volume mapper | enum | smart; fixed-point for software render windows | automatic VTK selection | GPU rendering when supported, CPU fallback for SSH/software rendering | `ui/viewer.py`, `ui/remote_plotter.py` |
| Interaction preview | automatic display policy | enabled during drags | live GUI only | Temporarily skips antialiasing and increases ray sample distance to at least twice the minimum voxel spacing; software mapper uses at least 3-pixel image sampling | `ui/interaction_rendering.py` |
| SSH preview size | automatic pixel cap | longest side 960 pixels during drags | SSH rendering widget | Reduces render/readback/transfer cost; native size and pointer mapping are restored on release | `ui/remote_plotter.py` |
| Selection on top / `selected_layer_on_top` | bool | true | Browser; workspace render settings | Selected layers draw above other layers; unchecked restores ordinary depth occlusion | `ui/scene_layers.py`, `ui/app.py` |
| SSH motion updates | automatic interval | about 30 ms | SSH rendering widget | Coalesces pointer events and applies the final position on release | `ui/remote_plotter.py` |

## Outputs

Browser objects use `magnitude_volume`, `pcmra_volume`, or
`content_volume_<Content ID>`. Their render mode, cut positions and axis
visibility are stored under `render_settings.content_renders.<data_key>`;
scene objects preserve visibility, palette, window and opacity in the existing
workspace snapshot. No extra 4D source copy or numerical export is created by
**3D Render**. Orthogonal modes use scalar-colored cut meshes; volume modes use
VTK volumes. Source frames are cached for display and actors are reused across phases.

Generated, unmasked PC-MRA is stored in `Workspace.pcmra_array`. CLI/API export
writes `pcmra.npz` containing `pcmra`, `resolution`, and `origin`, and records
`summary.json.pcmra_file`. Noise Removal masks only PC-MRA's 3D display, not its
stored scalar Content. It enables the independent 2D rejected-region overlay
for review. See [Noise removal](noise-removal.md).

## Limitations

- Changes to working velocity require explicit PC-MRA regeneration.
- Content rendering and cut controls affect the live GUI, not offline video styles.
- The timeline uses acquired frames without temporal interpolation.
- Only the selected visible orthogonal-plane object has drag handles. Multiple
  render objects can remain visible and retain independent positions.
- WSS volume rendering uses its existing voxelized scalar representation; it
  is not a replacement for the wall surface or quantitative wall analysis.
- Through-plane Flow requires its referenced measurement plane to remain available.
- PC-MRA/noise-mask contrast is not a vessel segmentation; review the mask,
  phase and window before interpreting peripheral signal.

## Where to change code

`autoflow/rendering/content.py` owns shared scalar sources;
`autoflow/ui/ortho_viewer.py` owns Content and its render button;
`autoflow/ui/content_controls.py` owns Browser controls;
`autoflow/ui/content_rendering.py` owns per-object modes, slices and constrained
VTK handles, with cached plane geometry and direct two-neighbor scalar sampling;
`autoflow/ui/interaction_rendering.py` owns temporary drag quality;
`autoflow/ui/scene_layers.py` owns selected-layer depth priority and foreground legends;
`autoflow/ui/viewer.py` owns geometry, transfer functions and actor
lifecycle. `autoflow/ui/app.py` connects Browser selection and the PC-MRA handoff;
`autoflow/core/pipeline.py` generates the numerical PC-MRA.

## Tests

Use `tests/test_smoke_phantoms.py` and `tests/test_pressure_gradient_phantom.py`.
Content phantoms cover signed and unsigned scalar fields, all three render modes,
anisotropic/nonzero-origin cuts, actual VTK drag callbacks, mirrored axes,
phase actor reuse, static magnitude, missing PC-MRA, repeated creation,
hide/delete cleanup, workspace persistence and the GUI PC-MRA handoff.
The GUI input-loading regression opens two distinct H5 cases through the real
load action and clicks 3D Render after each load, checking current-case values,
Browser insertion, status feedback and orthogonal handles.
Additional phantoms compare cut values with independent trilinear interpolation,
exercise direct axis/slider activation, and verify mouse-release restoration of
both smart and software volume mapper settings, image size and pointer coordinates.
Review real-data rendering and mouse interaction in `autoflow311`.

## Common problems

| Problem | Check |
| --- | --- |
| No PC-MRA Content | Run Generate PC-MRA successfully; loading alone leaves it uncomputed. |
| No object after 3D Render | Check the status bar and Log for unavailable data or a render error. |
| No colored handles | Select one checked 3D Volume, choose Orthogonal planes and enable Drag colored axes. |
| LR/AP/FH controls are gray | Restart with the updated GUI, select one 3D Volume, and use Show 3 orthogonal planes; axis clicks now open cuts directly. |
| Only one cut appears | Check the LR/AP/FH cut toggles and camera orientation. |
| Volume obscures flow | Reduce Browser opacity or switch to one orthogonal cut. |
| Dark or empty image | Check scalar window/level, threshold, current phase and PC-MRA retained mask. |
| Metric Content missing | Compute the metric with supported inputs; unavailable TKE is never synthesized. |
