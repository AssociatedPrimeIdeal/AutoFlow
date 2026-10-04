# Functional workflow

Review the input and segmentation before interpreting downstream haemodynamics. Optional derived families run only when requested. Requested WSS/pressure fields are prepared before plane summaries so their volumes can be sampled once and reused during export.

```mermaid
flowchart LR
    A[H5 or DICOM] --> B[Load and normalize]
    B --> C[Optional Correction: background, noise, unwrap, PC-MRA]
    C --> D[Segmentation and review]
    D --> F[Skeleton]
    F --> G[Graph and paths]
    G --> H[Planes]
    H --> I[Requested derived fields]
    I --> J[Plane metrics and QC]
    J --> K[Optional PWV]
    K --> L[Export and videos]
```

| Stage | Feature and purpose | Entry points |
| --- | --- | --- |
| DICOM conversion | [DICOM loading and conversion](dicom-loading.md): Dicom2H5 conversion followed by H5 loading | GUI / CLI / Python; converter optional |
| Input | [Inputs](../user/inputs.md): formats, coordinate mapping, cm/s, mm and RR | GUI / CLI / Python |
| Correction | [Background phase correction](background-phase-correction.md): static-tissue offsets | GUI / CLI / Python |
| Correction | [Noise removal](noise-removal.md): PC-MRA display-region screening | GUI / CLI / Python |
| Recovery | [Phase unwrapping](phase-unwrapping.md): velocity alias correction | GUI / CLI / Python; learned backends experimental |
| Segmentation | [Segmentation](segmentation.md): embedded, imported, threshold or nnUNet masks; review and reuse | GUI / CLI / Python; manual editing in GUI |
| Geometry | [Skeleton](skeleton.md): vessel centre points and cleanup | GUI / CLI / Python |
| Topology | [Graph and paths](graph-paths.md): branches, forks and ordered centreline paths | GUI / CLI / Python |
| Cross-sections | [Planes](planes.md): placement, contour review and coordinate import/export | GUI / CLI / Python |
| Basic haemodynamics | [Plane metrics](metrics.md): area, flow, velocity and reflux | GUI / CLI / Python |
| Wall haemodynamics | [WSS](wss.md): wall shear vectors and magnitude | GUI / CLI / Python |
| Energy | [TKE](tke.md): optional turbulence information from supported source data | GUI / CLI / Python; input dependent |
| Pressure | [Pressure](pressure.md): Navier–Stokes gradient, relative pressure and centreline drop | GUI / CLI / Python |
| Wave propagation | [PWV](pwv.md): timing along a selected vessel group | GUI / CLI / Python |
| Flow structure | [Vortex kinematics](vortex-kinematics.md): vorticity, Q and swirling strength | GUI / CLI / Python |
| Instantaneous trajectories | [Streamlines](streamlines.md) | GUI; CLI/Python offline videos |
| Temporal trajectories | [Pathlines](pathlines.md) | GUI only |
| Anatomical backdrop | [PC-MRA](pcmra-volume-rendering.md): explicitly generated final Correction step | GUI rendering; CLI/API numerical NPZ |
| Review | [Quality control](quality-control.md): input, geometry, planes and flow consistency | GUI / CLI / Python |
| Delivery | [Videos](videos.md) and [output contracts](../user/outputs.md) | GUI / CLI / Python |

Every configuration module is documented in [Configuration parameters](../user/parameters.md). Public controls have separate [CLI](../user/cli-parameters.md) and [Python](../user/api-parameters.md) references; dictionary controls are expanded in [Structured parameters](../user/parameter-schemas.md).

After segmentation, an optional masked unwrapping rerun is available in Correction. It retains existing geometry and measurements; users explicitly rerun the downstream stages they need.
