# AutoFlow

AutoFlow processes 4D flow MRI from H5 or DICOM using a desktop GUI, batch CLI, or Python API.

Start with [Quickstart](en/user/quickstart.md), then follow the [functional workflow](en/features/index.md). Each feature page explains its purpose, entry points, inputs, outputs, limitations, parameters and implementation.

| I want to… | Read |
| --- | --- |
| Run a case or a batch | [Quickstart](en/user/quickstart.md), [CLI](en/user/cli.md), [GUI](en/user/gui.md), [Python API](en/user/python-api.md) |
| Check data shape, coordinates and units | [Inputs](en/user/inputs.md) |
| Correct phase offsets or aliasing | [Background correction](en/features/background-phase-correction.md), [Unwrapping](en/features/phase-unwrapping.md) |
| Prepare segmentation and geometry | [Segmentation](en/features/segmentation.md), [Skeleton](en/features/skeleton.md), [Graph and paths](en/features/graph-paths.md), [Planes](en/features/planes.md) |
| Compute haemodynamic metrics | [Plane metrics](en/features/metrics.md), [WSS](en/features/wss.md), [TKE](en/features/tke.md), [Pressure](en/features/pressure.md), [PWV](en/features/pwv.md), [Vortex kinematics](en/features/vortex-kinematics.md) |
| Review motion and export results | [Streamlines](en/features/streamlines.md), [Pathlines](en/features/pathlines.md), [PC-MRA](en/features/pcmra-volume-rendering.md), [Quality control](en/features/quality-control.md), [Videos](en/features/videos.md), [Outputs](en/user/outputs.md) |
| Understand a parameter | [Configuration](en/user/parameters.md), [Dictionary schemas](en/user/parameter-schemas.md), [CLI flags](en/user/cli-parameters.md), [API fields](en/user/api-parameters.md) |
| Change or optimize code | [Architecture](en/developer/architecture.md), [Modules](en/developer/feature-to-code-map.md), [Performance](en/developer/performance.md), [Testing](en/developer/testing.md) |

TKE is optional. WSS and pressure require reviewed segmentation and calibrated velocity, space and time metadata. See the feature limitations before interpreting results.
