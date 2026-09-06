# User Guide

## The shortest useful mental model

AutoFlow is a three-entry-point workflow:

| Entry point | Best for | What you do there |
| --- | --- | --- |
| GUI | first review and correction | inspect geometry, choose segmentation, edit objects, scrub phases, export selected results |
| CLI | batch processing and provenance | provide one or more H5/DICOM inputs, select stages, write a case output tree |
| Python API | research automation | build `AutoFlowConfig`, call `run_batch`, and consume result paths/JSON |

The same workspace concepts appear in all three: input → segmentation → centerline/graph → planes → metrics → review/export.

## Install and start

```bash
pip install -e ".[gui]"
autoflow-gui --config-dir ./configs
```

For a batch run:

```bash
pip install -e "."
autoflow-run case.h5 --output-dir ./results/case
```

## GUI and CLI side by side

| Stage | GUI | CLI | Main result |
| --- | --- | --- | --- |
| Load | `File > Open H5` or `Import DICOM Directory` | positional input path | normalized `mag`, `flow`, geometry and metadata |
| Background correction | choose it when the case has no reusable correction | `--bgc`, optionally `--bgc-method msac` | corrected velocity and correction report/cache |
| Phase unwrapping | `Phase Unwrapping` step | `--phase-unwrap-method gc3D`, `lap4D`, or `nprs` | unwrapped velocity field when enabled |
| Segmentation | `Segmentation` workspace: import, threshold, auto segment, or edit | `--autoseg`; imported masks/configs can also be supplied through the API/GUI | active mask for downstream geometry |
| Centerline | run `Skeleton`, then `Graph/Paths` | normal batch order | skeleton, graph, branches and paths |
| Planes | configure count, anchor, spacing and direction | `--plane-mode`, `--plane-count`, `--plane-spacing-*` | `planes.json`, `planes.h5`, positions |
| Base metrics | `Hemodynamics` / plane metrics | default unless `--skip-plane-metrics` | time-resolved plane flow/velocity metrics |
| Optional metrics | click the requested analysis in the GUI | `--with pwv,wss,tke,pg,vortex` | metric-specific JSON/NPZ/H5/PNG |
| Dynamic review | timeline, 3D browser, ortho viewer, streamlines/pathlines | `--video plane,wss,tke,pg,streamlines` | MP4 videos and frame exports |
| QC/export | `Review & Export` | inspect output JSON and `summary.json` | quality report, plane QC and reproducible artifacts |

## Feature tour: how to use each part

### Input and QC

Open the case before changing analysis parameters. Check spatial dimensions, cardiac phases, voxel spacing, VENC, axis labels and the presence/absence of an active segmentation. The loader accepts either a complex image representation or normalized magnitude and velocity arrays; see [Input & Output](input-output.md).

### Segmentation

Use an embedded mask only when it is the mask you intend to analyze. Otherwise choose an external file, threshold the PC-MRA/magnitude, run the available nnUNet backend, or edit the result. Treat auto segmentation as an initial result that needs review, especially at branch ostia and vessel boundaries.

### Skeleton, graph and paths

Skeletonization turns the vessel mask into a centerline representation. The graph adds topology; paths are the objects used for planes, pathlines and path-based measurements. In the GUI, inspect the generated objects before running hemodynamics.

### Planes and metrics

Planes are placed along selected paths with an anchor and either distance- or fraction-based spacing. Prefer planes away from bifurcations and mask ends. Plane metrics summarize the velocity crossing each plane over cardiac phase; compare neighboring planes and use plane QC before interpreting a single number.

### Hemodynamics

`PWV` uses configured groups and waveforms. `WSS` needs a wall surface and velocity gradients. `TKE` remains optional and is skipped when the input does not provide the required information. Pressure analysis can produce pressure-gradient and relative-pressure fields. Vortex analysis provides vorticity, Q-criterion and swirling strength.

### Streamlines, pathlines and PC-MRA

Streamlines show an instantaneous field; pathlines follow particles through time. Use the GUI for interactive seed and camera exploration, and use the CLI for repeatable videos. PC-MRA rendering provides a structural backdrop; it should support, not replace, velocity and segmentation review.

## Export a video

Video generation is opt-in:

```bash
autoflow-run case.h5 \
  --output-dir ./results/case \
  --with wss,pg \
  --video plane,wss,pg,streamlines \
  --fps 12 \
  --camera-view right
```

The GUI equivalent is `Export > Export Videos...`, where you select video families and render settings. Video names and availability depend on the requested metric and whether the case has the required segmentation/TKE support.

## Python entry point

```python
from autoflow import AutoFlowConfig, run_batch

config = AutoFlowConfig(
    inputs=["case.h5"],
    output_dir="./results/case",
    requested_metrics=["wss", "pg"],
    requested_videos=["plane", "wss"],
)
results, case_output = run_batch(config)
```

For algorithm choices, configuration ownership and code changes, use the [Developer architecture](developer/architecture.md) and [feature-to-code map](developer/feature-to-code-map.md).
