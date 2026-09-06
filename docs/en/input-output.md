# Input & Output

## Minimum input contract

AutoFlow normalizes every accepted input into a case with `mag`, `flow`, `resolution`, `origin`, `venc` and `rr`. Segmentation, correction, sigma and TKE are optional capabilities.

| Input family | Required data | Typical shape | Notes |
| --- | --- | --- | --- |
| Legacy complex H5 | `img_complex` or complex `img`, `Resolution`, `VENC`, `RR` | `(X,Y,Z,T,4)` | channel 0 is magnitude reference; channels 1–3 are velocity encodes |
| Normalized H5 | `mag`, `flow`, `Resolution`, `Origin`, `VENC`, `RR` | `mag=(X,Y,Z,T)`, `flow=(X,Y,Z,T,3)` | preferred when an upstream converter decoded complex data |
| Real `img` layout | `img` plus metadata | `(X,Y,Z,T,4)` or channel-first equivalent | interpreted as magnitude + three flow components |
| Dual-VENC complex H5 | 7-channel complex `img_complex` plus `VENCOrder` | `(X,Y,Z,T,7)` | two three-axis encodings are paired and reconstructed |
| DICOM directory | readable velocity/magnitude series and DICOM geometry | vendor-dependent | scan a directory; AutoFlow resolves cases and metadata |

The loader is case-insensitive for common dataset names and can resolve a case below an H5 group. Multiple recognizable groups can become multiple cases.

## Legacy complex image

```text
img_complex[..., 0]  -> magnitude reference
img_complex[..., 1]  -> encoded velocity component 1
img_complex[..., 2]  -> encoded velocity component 2
img_complex[..., 3]  -> encoded velocity component 3
```

Complex channels are not yet a ready-to-display signed velocity field. AutoFlow uses VENC and `VENCOrder` to normalize them, then publishes a canonical three-component `flow` array.

## Normalized magnitude and flow

An upstream converter can provide:

```text
mag   : (X, Y, Z, T)
flow  : (X, Y, Z, T, 3)
```

The final dimension of `flow` is the three velocity components. The physical unit must agree with the metadata and conversion procedure; do not infer units from the file extension.

## Geometry and time metadata

| Dataset | Meaning | Why it matters |
| --- | --- | --- |
| `Resolution` | three voxel spacings, usually mm | converts voxels to distances and affects gradients/areas |
| `Origin` | physical origin | maps local arrays to world coordinates |
| `SpatialOrder` | labels such as `FH`, `RL`, `AP` | lets the loader permute/sign-correct spatial axes |
| `VENC` | one or three velocity-encoding limits | sets velocity scale and dual-VENC pairing |
| `VENCOrder` | labels for encoded components | maps channels to canonical components |
| `RR` | R-R interval, commonly ms | provides temporal spacing for phase-resolved analysis |

Triplet metadata may be flat, a singleton row, or a singleton column. Names are matched case-insensitively and separators such as `_` and `-` are tolerated.

## Optional fields and capabilities

- `segmask`, `segmentation` or `seg` activates an embedded segmentation when present.
- `corr` (or `corr_low` and `corr_high`) is a reusable background phase correction cache.
- Complex input can provide sigma information used by TKE-related paths.
- DICOM/normalized `mag + flow` input can still run skeleton, graph, planes, plane metrics, WSS and streamlines.
- TKE remains optional. AutoFlow does not synthesize fake TKE from velocity magnitude.

## Output contract

| Output | Contains | Use it for |
| --- | --- | --- |
| `summary.json` | stage timings and request flags | provenance and batch monitoring |
| `quality_report.json` | staged input, segmentation, topology, plane, flow-consistency and PWV checks | review before interpretation |
| `planes.json` / `planes.h5` | plane geometry and serialized objects | reload and share plane layouts |
| `plane_positions.json` | portable plane coordinates | reuse planes in another run |
| `plane_metrics.json` | cardiac-phase and aggregate plane metrics | tables and plots |
| `plane_qc.json` | plane-level consistency checks | detect suspect planes |
| `pwv.json`, PNG | PWV groups, waveforms and plots | pulse-wave analysis |
| NPZ/H5 derived files | WSS, pressure, vortex and other volumes | downstream numerical analysis |
| MP4/PNG | rendered dynamic or static views | presentations and review |

Output is written below `--output-dir`; multi-group H5 files create one case directory per recognized data group. See the [CLI reference](user/cli.md) for all flags and the [Developer docs](developer/config-system.md) for configuration ownership.
