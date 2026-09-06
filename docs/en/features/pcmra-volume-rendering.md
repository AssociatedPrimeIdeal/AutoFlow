# PC-MRA Volume Rendering

## Status

| Entry point | Status | Notes |
| --- | --- | --- |
| GUI | Supported | A dynamic `PC-MRA (4D)` layer is added to the Browser after a flow + magnitude case is loaded. |
| CLI | Not implemented | Batch outputs remain numerical; the interactive VTK layer is GUI-only. |
| Python API | Not implemented | Use the GUI for the interactive volume actor. |

## What it does

The GUI computes a phase-resolved phase-contrast MR angiography (PC-MRA) volume
for every cardiac frame: `magnitude_t × ||velocity_t||`. It renders the current
frame with VTK's grayscale composite volume mapper. The automatic display range
uses the positive finite foreground values from that frame: `L=P5` and `H=P99`,
then `WL=(L+H)/2` and `WW=H-L`. The timeline and playback update the 3-D volume
actor.

## When to use it

Use it as a 3-D anatomical backdrop while reviewing segmentation, planes,
streamlines, and pathlines. It is especially useful when a surface-only
segmentation view hides nearby vessels.

## Quick use

1. Load an H5 or DICOM case with magnitude and flow.
2. In the left Browser, select the `PC-MRA (4D)` item under `Global → PC-MRA` and check or uncheck it.
3. Adjust `Opacity` below the Browser (or use the item's right-click `Set Opacity…`).
4. In the central 3-D view, hold `Shift` and the left mouse button and drag
   horizontally to change window width and vertically to change window level.
   All other camera gestures use VTK's default trackball-camera mapping.
5. Use the `Window/Level` sliders below the 3-D view to adjust the range, or
   use `Auto` / `Reset PC-MRA Window/Level` to restore the automatic range.
   Without `Shift`, all three mouse buttons retain VTK's default camera
   behavior regardless of PC-MRA visibility.

## Inputs

| Input | Required | Meaning |
| --- | --- | --- |
| `mag_raw` | yes | magnitude volume, 3-D or 4-D |
| `flow_raw` | yes | normalized `XYZT3` velocity in cm/s |
| `resolution`, `origin` | yes | physical spacing and world origin for the VTK image |

## Parameters

| Parameter | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| Browser `Opacity` | 0–100% slider | 85% | GUI, per `SceneObject` | scales the PC-MRA scalar-opacity transfer function | `autoflow/ui/app.py`, `autoflow/ui/viewer.py` |
| automatic range | current frame positive finite values | `P5` to `P99` | GUI | derives `WL=(L+H)/2` and `WW=H-L` | `autoflow/ui/viewer.py` |
| `Window` slider | normalized slider | automatic current-frame width | GUI | changes PC-MRA window width | `autoflow/ui/app.py`, `autoflow/ui/viewer.py` |
| `Level` slider | normalized slider | automatic current-frame center | GUI | changes PC-MRA window level | `autoflow/ui/app.py`, `autoflow/ui/viewer.py` |
| volume blend/shading | mapper options | composite, no shading | code | displays the raw grayscale volume without added surface lighting | `autoflow/ui/viewer.py` |
| 3-D background color | color string | `#000000` | Settings → 3D Background Color | controls the backdrop independently of PC-MRA transfer functions | `autoflow/config.py`, `autoflow/ui/app.py`, `autoflow/ui/viewer.py` |

## Outputs

No file is created automatically. The Browser stores a `SceneObject` with
`data_key="pcmra_volume"`; workspace save/restore preserves its visibility and
opacity. The 3-D actor is a VTK volume actor, not a polygon surface.

## Limitations

- The volume is phase-resolved, but it uses the loaded temporal frames directly; no temporal interpolation is performed.
- PC-MRA has no scalar bar by default because it is used as an anatomical backdrop.
- GPU volume rendering support depends on the local VTK/OpenGL environment.

## Where to change code

| Change you want | Edit here | Also check | Tests |
| --- | --- | --- | --- |
| PC-MRA calculation and VTK image | `autoflow/ui/viewer.py` (`_build_dataset`) | `autoflow/core/pipeline.py` registration | smoke test and GUI manual check |
| Browser visibility/opacity controls | `autoflow/ui/app.py` | `autoflow/core/models.py` persistence | GUI manual check |
| 2-D segmentation overlay opacity | `autoflow/ui/ortho_viewer.py` | `autoflow/ui/app.py` segmentation dock | GUI manual check |

## Tests

- `~/miniconda3/envs/ryy/bin/python -m pytest tests/test_smoke_phantoms.py -q`
- `~/miniconda3/envs/ryy/bin/python -m pytest tests/test_pressure_gradient_phantom.py -q`

## Common problems

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `PC-MRA` is absent | magnitude or flow was not loaded | check the input case and loader metadata |
| the backdrop is too faint or too strong | opacity is low or background intensity is high | select the Browser item and adjust `Opacity` |
| volume rendering fails on a remote display | VTK/OpenGL volume support is unavailable | use the existing SSH off-screen renderer or hide `PC-MRA` and review surfaces |
