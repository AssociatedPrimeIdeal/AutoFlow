# Viewer display settings

## Status

**GUI only.** The unified dock controls the 2D viewer.
CLI and public Python API do not expose these interactive controls or
export their appearance to videos.

## What it does

**Display settings** beside the 2D viewer opens a dock with Image, Overlays,
Flow and View tabs. **Settings → Viewer Display Settings** opens the
same dock. Changes preview immediately. The scope selector applies slice
settings to all three views or one named view.
Standard and selected-plane U/V/N layouts use the same settings.
Controls marked **shared** apply to the whole viewer, including layout, linked
navigation, shared text/legend geometry and contour sampling/edit settings.

Image styles are remembered by Content source. Flow overlays can use fixed
colors or speed colors with separate legends from the image. Identical flow
quantity, colormap and limits share a legend; differing settings have separate
legends. No original intensity, velocity, segmentation or metric arrays change.

## When to use it

Use this dock to prepare readable overlays, compare cardiac phases with locked
color limits, adjust label visibility, or place grayscale anatomy behind live
3D streamlines. Use [PC-MRA and magnitude context](pcmra-volume-rendering.md)
for the anatomical modes and [Streamlines](streamlines.md) for flow semantics.

## Quick use

GUI:

1. Load a case and click **Display settings**.
2. In **Image**, choose the colormap and per-slice, current-frame, across-phase
   or manual range for the selected Content source.
3. In **Overlays**, configure segmentation, ROI contours and the noise mask.
4. In **Flow**, enable planar streamlines or projection, choose total/in-plane
   speed colors or a fixed color, and adjust density, line width and arrows.
   Projection requires **Generate Streamlines** first.
5. In **View**, adjust background, guides, linked zoom/pan and colorbars.
6. For 3D context, choose **Content → 3D Render**, then configure the selected
   Browser object. See [Content volumes and PC-MRA](pcmra-volume-rendering.md).

CLI: no new flags; use the existing numerical/volume/video workflows.

Python API: no new public fields. `Workspace.snapshot_dict()` / `restore_dict()`
retain the GUI state in the existing `render_settings` payload.

## Inputs

Available scalar Content sources, optional segmentation and noise mask, velocity
for flow overlays, and magnitude or explicitly generated PC-MRA for 3D context.
PC-MRA remains unavailable until generation; selecting a context source never
synthesizes PC-MRA or TKE. All geometry uses normalized array axes and mm.

## Parameters

Values are stored under `render_settings.viewer_display`; `views.<name>` holds
per-view overrides and `images.<Content ID>` holds source-specific styles.
The owning controls live in `autoflow/ui/display_settings.py`; slice application
is in `autoflow/ui/viewer_display.py` and `autoflow/ui/ortho_viewer.py`.

| Parameter / controls | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| Image `cmap`, `interpolation` | enums | content default, nearest | Image tab, by Content | scalar palette and display interpolation | `ui/viewer_display.py`, `ui/slice_view.py` |
| Image `range_mode`, `low`, `high` | enum, floats | auto per slice, 0, 100 | Image tab, by Content | per-slice/current-frame/across-phase/manual limits; manual high must exceed low | `ui/viewer_display.py` |
| Segmentation `visible`, `opacity` | bool, float 0–1 | true, current workspace segmentation opacity | Overlays tab | label fill/boundary visibility and opacity | `ui/ortho_viewer.py` |
| Segmentation `mode`, `labels`, `colors`, `line_width` | enum, comma-separated IDs, mapping, float px | fill, all labels, workspace palette, 1.5 px | Overlays tab | fill/outline/both, displayed IDs, per-label colors and boundary width | `ui/ortho_viewer.py`, `ui/slice_view.py` |
| Contours `visible`, `roi_color`, `manual_color`, `line_width`, `opacity`, `dashed` | bool, colors, float px, float 0–1, bool | true, cyan, yellow, 1.8 px, 1, false | Overlays tab | metric/manual ROI styling | `ui/viewer_display.py` |
| Contours `smoothing`, `minimum_points`, `snap_distance` | float px, int, float mm | existing display preferences, normally 0.8, 8, 0 (auto) | Overlays tab | contour display smoothing/filter; endpoint snap remains an ROI editing control | `ui/ortho_viewer.py` |
| Noise `visible`, `opacity`, `color` | bool, float 0–1, color | off, 0.35, red | Overlays tab; quick mask toggle | rejected-region display; the mask must exist | `ui/ortho_viewer.py` |
| Flow appearance / integration | grouped controls | speed colors, turbo, 85% opacity | Flow tab | independent planar/projected styles and arrows; see [full flow table](streamlines.md#parameters) | `ui/slice_view.py`, `algorithms/slice_streamlines.py` |
| Guides `crosshairs`, `orientation`, `values`, `plane_intersection` | bool | true | View tab | guide and text visibility | `ui/viewer_display.py` |
| Guides `crosshair_color`, `plane_color`, `line_width` | colors, float px | red, yellow, 1 px | View tab | cursor/intersection appearance | `ui/viewer_display.py` |
| View `background`, `zoom`, `linked_zoom`, `order` | color, float 0.1–1, bool, enum | dark, 0.5, false, axial/coronal/sagittal | View tab | background, initial FOV, linked zoom/pan and stack order; title dragging also reorders views | `ui/viewer_display.py`, `ui/ortho_viewer.py` |
| Colorbars `background`, `velocity`, `width`, `font_size` | bool, bool, int px, int pt | true, true, 18, 10 | View tab | image/flow legend visibility and geometry | `ui/viewer_display.py` |

## Outputs

The existing workspace snapshot contains slice-display state under
`render_settings.viewer_display`. **Save defaults** writes reusable styles to local
Qt preferences; manual image/flow windows remain case-specific and are reset to
automatic in defaults. Browser render objects have their own saved settings.
**Restore defaults** resets the dock's display settings. No extra image, video,
NPZ, H5 or JSON export is created by changing the display.

## Limitations

Rendering settings do not recompute derived metrics. A 3D display fraction can
only thin existing trajectories; generating more requires changing the main 3D
seed controls. Across-phase image range calculation visits scalar phases and
can take longer on first use. Per-view zoom links share view-coordinate ranges;
U/V/N coordinates differ from LR/AP/FH coordinates. Endpoint snap settings
affect contour editing rather than only painting. Configure 3D rendering on the
selected Browser object, independently of this dock.

## Where to change code

`autoflow/ui/display_settings.py`: dock fields and serializable defaults.
`autoflow/ui/viewer_display.py`: scope, styles, colors and legends.
`autoflow/ui/ortho_viewer.py`: sampling, layer caches and legacy quick controls.
`autoflow/ui/slice_view.py`: cached colored segments, arrows and boundaries.
`autoflow/rendering/content.py`, `autoflow/ui/content_rendering.py`: 3D Content layers.
`autoflow/ui/app.py`: menus, Browser controls and PC-MRA handoff.

## Tests

Use the retained `tests/test_smoke_phantoms.py` and
`tests/test_pressure_gradient_phantom.py` suites. New smoke/phantom checks cover
speed gradients, fixed colors, independent opacity and legends, per-view guides,
manual limits across phases, label filtering/boundaries, workspace round trips,
Content render modes, draggable cuts and missing PC-MRA. Review dock
layout, camera interaction and real-data overlays manually in `autoflow311`.

## Common problems

| Problem | Check |
| --- | --- |
| No orange/projected trajectories | Generate Streamlines first; check the current phase and slab width. |
| No planar trajectories | Check tangential speed, current segmentation and the minimum speed threshold. |
| Changes affect one view only | Check the scope selector and existing per-view overrides. |
| No anatomy | Choose Magnitude or generated PC-MRA Content, click 3D Render and check its Browser object. |
| Colors change meaning during playback | Select across-phase/automatic flow limits or a manual window. |
| A manual range is rejected | Upper limit must exceed lower limit. |
