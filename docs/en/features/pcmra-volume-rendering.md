# PC-MRA Volume Rendering

## Status

| Entry point | Status | Notes |
| --- | --- | --- |
| GUI | Supported | A dynamic `PC-MRA (4D)` layer is created only by Correction > Generate PC-MRA, the fourth and final Run All action. |
| CLI | Partial | `--generate-pcmra` or `--correction` creates `pcmra.npz`; interactive VTK rendering is GUI-only. |
| Python API | Supported | `generate_pcmra=True` or `correction_all=True` computes/export arrays; workspace action supports later GUI rendering. |

## What it does

The magnitude-weighted speed formula and distinction from static PC-MRA methods are recorded in [Scientific references: PC-MRA](../references/index.md#noise-masking-and-pc-mra).

The explicit generation action computes a phase-resolved phase-contrast MR angiography (PC-MRA) volume
for every cardiac frame: `magnitude_t × ||velocity_t||`. It renders the current
frame with VTK's grayscale composite volume mapper. The automatic display range
uses the positive finite foreground values from that frame: `L=P5` and `H=P99`,
then `WL=(L+H)/2` and `WW=H-L`. The timeline and playback update the 3-D volume
actor in place. The cached image and its raw-frame display range are reused on
later cycles; changing phase, opacity, or window/level keeps the same volume
actor. Scene changes are batched into one render per timeline update.

The normal volume path uses VTK's smart mapper to select GPU rendering when
supported. SSH-forwarded sessions probe EGL off-screen rendering at startup
and use a software render window with the CPU fixed-point mapper if EGL is
unavailable. A static 3-D magnitude image is used for every velocity phase.

The VTK image stores the masked values directly at voxel centers, aligned with
segmentation cells, with a zero-valued border. Nearest-neighbor rendering avoids
cell-to-point averaging across the mask boundary. Rejected samples remain zero
and transparent, including after manual window/level changes.

## When to use it

Use it as a 3-D anatomical backdrop while reviewing segmentation, planes,
streamlines, and pathlines. It is especially useful when a surface-only
segmentation view hides nearby vessels.

The GUI can show PC-MRA, segmentation, and planes together. When processed
label-group surfaces exist, the original aggregate segmentation surface is
hidden to avoid drawing the same segmentation twice. Group surfaces start at
15% opacity. Plane actors use bright, unlit wireframes so they remain visible
through the volume; selecting a plane adds a magenta highlight.

## Quick use

1. Load an H5 or DICOM case with magnitude and flow. Loading leaves PC-MRA uncomputed.
2. Open **Correction** and click **Generate PC-MRA**, or **Run All** to generate it after background correction, noise removal and unwrapping.
3. In the left Browser, select the `PC-MRA (4D)` item under `Global → PC-MRA` and check or uncheck it.
4. Adjust `Opacity` below the Browser (or use the item's right-click `Set Opacity…`).
5. Keep the `Segmentation` and `Planes` Browser entries visible. Use the
   segmentation dock or Browser opacity slider to make segmentation faint, and
   select a plane to highlight the plane of interest.
6. In the central 3-D view, hold `Shift` and the left mouse button and drag
   horizontally to change window width and vertically to change window level.
   All other camera gestures use VTK's default trackball-camera mapping.
7. Use the `Window/Level` sliders below the 3-D view to adjust the range, or
   use `Auto` / `Reset PC-MRA Window/Level` to restore the automatic range.
   With no continuous scalar layer selected, these controls target visible
   PC-MRA, including when segmentation or Noise Region is selected.
   Without `Shift`, all three mouse buttons retain VTK's default camera
   behavior regardless of PC-MRA visibility.

The optional **Correction > Noise Removal** stage restricts this 3D layer to the `True` (retained) voxels in `Workspace.pcmra_render_mask`; its automatic range then uses retained positive values. The complementary red 3-D Noise Region is hidden by default and uses a binary maximum-intensity volume so its opacity does not saturate with thickness. Noise Removal, alone or in GUI Run All, enables the independent 2-D red overlay for review. Only excluded (`False`) voxels are painted above segmentation, without changing source slice values. Resetting that mask restores the full display region. It does not change segmentation features or quantitative data. See [Noise removal](noise-removal.md).

CLI: `autoflow-run case.h5 --generate-pcmra`, or `--correction` for all four actions. Normal batch segmentation prerequisites still apply. Python: `run_case("case.h5", config=AutoFlowConfig(generate_pcmra=True))`; for a loaded workspace, call `PipelineEngine().run_step(ws, StepId.GENERATE_PCMRA, print)`.

## Inputs

| Input | Required | Meaning |
| --- | --- | --- |
| `mag_raw` | yes | magnitude volume, 3-D or 4-D |
| `flow_raw` | yes | normalized `XYZT3` velocity in cm/s |
| `resolution`, `origin` | yes | physical spacing and world origin for the VTK image |

## Parameters

| Parameter | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| Browser `Opacity` | 0–100% slider | 85% for PC-MRA; 75% for planes | GUI, per `SceneObject` | scales the selected layer; for PC-MRA it scales the scalar-opacity transfer function | `autoflow/ui/app.py`, `autoflow/ui/viewer.py` |
| segmentation opacity | 0–100% slider | 15% | Segmentation dock or Browser | controls the active segmentation surfaces; grouped surfaces replace the aggregate surface in the 3-D view | `autoflow/ui/app.py`, `autoflow/core/models.py` |
| ortho `Noise mask` | checkbox and 0–100% slider | off before calculation; on at 35% after first Noise Removal | Ortho Viewer | overlays only excluded PC-MRA mask voxels in red; slider zero disables it, positive opacity enables it | `autoflow/ui/app.py`, `autoflow/ui/ortho_viewer.py`, `autoflow/ui/slice_view.py` |
| `--generate-pcmra` / API `generate_pcmra` | bool | false | CLI/API; explicit GUI action | materialize PC-MRA after requested corrections; no calculation during input load | `autoflow/processing.py`, `autoflow/core/pipeline.py` |
| automatic range | current frame positive finite values | `P5` to `P99` | GUI | derives `WL=(L+H)/2` and `WW=H-L` | `autoflow/ui/viewer.py` |
| `Window` slider | normalized slider | automatic current-frame width | GUI | changes PC-MRA window width | `autoflow/ui/app.py`, `autoflow/ui/viewer.py` |
| `Level` slider | normalized slider | automatic current-frame center | GUI | changes PC-MRA window level | `autoflow/ui/app.py`, `autoflow/ui/viewer.py` |
| volume blend/shading | mapper options | composite, no shading | code | displays the raw grayscale volume without added surface lighting | `autoflow/ui/viewer.py` |
| volume mapper | mapper name | smart; fixed-point on the software render window | automatic, based on VTK render window | selects GPU volume rendering when supported and retains CPU fallback | `autoflow/ui/viewer.py` |
| `VTK_DEFAULT_OPENGL_WINDOW` | environment string | EGL probe with software fallback for SSH | environment before GUI startup | overrides automatic render-window selection, e.g. `vtkOSOpenGLRenderWindow` to force software | `autoflow/ui/launcher.py`, `autoflow/ui/remote_plotter.py` |
| 3-D background color | color string | `#000000` | Settings → 3D Background Color | controls the backdrop independently of PC-MRA transfer functions | `autoflow/config.py`, `autoflow/ui/app.py`, `autoflow/ui/viewer.py` |

## Outputs

`Workspace.pcmra_array` stores the generated, unmasked `XYZT` array and is saved/restored with the workspace. CLI/API generation writes `pcmra.npz` with `pcmra`, `resolution` and `origin`, plus `summary.json.pcmra_file`. No PC-MRA is computed or written by input loading. The Browser stores a `SceneObject` with
`data_key="pcmra_volume"`; workspace save/restore preserves its visibility and
opacity. The 3-D actor is a VTK volume actor, not a polygon surface.

Changes to working velocity do not automatically regenerate the stored PC-MRA; click Generate PC-MRA again after a correction rerun. Noise-mask changes affect the 3D display region without rewriting the generated data. The Content menu exposes PC-MRA only once this array exists.

## Limitations

- The volume is phase-resolved, but it uses the loaded temporal frames directly; no temporal interpolation is performed.
- PC-MRA has no scalar bar by default because it is used as an anatomical backdrop.
- Showing every generated plane at once can be visually dense; select one plane in the Browser for the magenta focus overlay, or hide individual plane objects.
- GPU volume rendering support depends on the local VTK/OpenGL environment.
- A retained region is not a vessel segmentation. Static tissue, coherent motion
  and low-flow phases can leave peripheral signal brighter than small vessels.
  Review the red slice mask and the current phase, and adjust PC-MRA window/level;
  bright peripheral signal alone does not imply an inverted mask.

## Where to change code

| Change you want | Edit here | Also check | Tests |
| --- | --- | --- | --- |
| PC-MRA calculation and VTK image | `autoflow/core/pipeline.py` generation and `autoflow/ui/viewer.py` (`_build_dataset`) | `autoflow/core/models.py` persistence | smoke test and GUI manual check |
| GPU selection and SSH fallback | `autoflow/ui/launcher.py`, `autoflow/ui/remote_plotter.py`, `autoflow/ui/viewer.py` | forwarded X11 and off-screen OpenGL driver | GUI manual check |
| Browser visibility/opacity controls | `autoflow/ui/app.py` | `autoflow/core/models.py` persistence | GUI manual check |
| 2-D segmentation overlay opacity | `autoflow/ui/ortho_viewer.py` | `autoflow/ui/app.py` segmentation dock | GUI manual check |

## Tests

- Activate the known working environment with `conda activate autoflow311`.
- Run `pytest tests/test_smoke_phantoms.py tests/test_pressure_gradient_phantom.py -q`.
- The PC-MRA phantom smoke check covers phase-dependent image values, actor reuse,
  automatic and manual window/level, opacity after LUT range changes, and static
  3-D magnitude input. Check camera interaction and combined volume/mesh display
  manually in the GUI.

## Common problems

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `PC-MRA` is absent | magnitude or flow was not loaded | check the input case and loader metadata |
| the backdrop is too faint or too strong | opacity is low or background intensity is high | select the Browser item and adjust `Opacity` |
| volume rendering fails on a remote display | VTK/OpenGL volume support is unavailable | use the existing SSH off-screen renderer or hide `PC-MRA` and review surfaces |
