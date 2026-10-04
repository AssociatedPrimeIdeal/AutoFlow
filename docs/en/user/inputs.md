# Inputs

## Required information

AutoFlow needs anatomy (`mag`), velocity (`flow` or complex encodes), spatial geometry (`resolution`, `origin`, axis/component directions), and velocity/time calibration (`VENC`, RR and phase count). Canonical velocity is cm/s, spacing is mm and RR is ms.

Segmentation and background correction are separate optional inputs. Segmentation defines the vessel analysis domain; correction removes static-tissue velocity offsets. Missing segmentation must be imported or generated before geometry/metrics.

## Cold validation and inspection

Use [the read-only benchmark](../developer/performance.md) to recompute correction and segmentation without source cache writes. A normal CLI run with `--bgc --autoseg` may reuse compatible caches and write H5 cache fields; it is not automatically a cold benchmark. [Worked example](demo-case.md) describes review and output checks.

Inspect H5 dataset names and shapes before processing:

```python
import h5py
with h5py.File("case.h5", "r") as handle:
    def show(name, value):
        if isinstance(value, h5py.Dataset):
            print(name, value.shape, value.dtype)
    handle.visititems(show)
```

Legacy complex data must be channels-last `XYZT4` (single VENC) or `XYZT7` (dual VENC). Real combined `img` is `XYZT4`; normalized `mag` is `XYZT` and `flow` is `XYZT3`. Channels-first layouts are rejected, not silently transposed.

[Loader parameters](parameters.md#loader) and [DICOM override schema](parameter-schemas.md#dicom-metadata-overrides) explain every supported setting.

## Supported Inputs

| Input type | Supported | Typical entry points | Notes |
| --- | --- | --- | --- |
| legacy complex H5 | yes | CLI, GUI, Python API | requires channels-last `XYZT4` or `XYZT7`; can provide segmentation and complex-derived sigma |
| normalized H5 with `mag` and `flow` | yes | CLI, GUI, Python API | preferred normalized path |
| normalized H5 with real-valued `img[..., 0:4]` | yes | CLI, GUI, Python API | requires `XYZT4` and is interpreted as `mag + flow_xyz` |
| DICOM-to-H5 conversion | optional backend | CLI, GUI, Python API | [Pinned converter](../features/dicom-loading.md); preserves normalized H5 groups for reuse |
| direct DICOM directory | yes | CLI, GUI, Python API | scanned into importable cases |
| single DICOM file | yes for case collection | CLI | resolved through case collection logic |

## Legacy Complex H5 Variants

- classic legacy complex H5 uses `img_complex[..., 0]` as the magnitude reference and `img_complex[..., 1:4]` as three velocity encodes
- complex-valued `img[..., 0:4]` is treated the same as `img_complex[..., 0:4]`
- legacy dual-venc H5 with `img_complex.shape[-1] == 7` is supported when the file stores one reference plus six velocity encodes ordered by `VENCOrder`
- for `Nv=7`, `VENC` contains two consecutive three-axis triplets corresponding to channel groups `1:4` and `4:7`; AutoFlow compares the triplets component-wise to identify the low-venc and high-venc groups, keeps each triplet paired with its source channels, and applies background phase correction separately
- the GUI asks whether to load `LV` (low-VENC flow), `HV` (high-VENC flow), or `DV` (dual-VENC alias reconstruction) when a dual-VENC H5 is opened; `DV` remains the default for direct loader calls
- Python callers can pass `dual_venc_mode="lv"`, `"hv"`, or `"dv"` to `load_h5_data()` or `load_input_data()`; the selected source is returned as `LoadedCase.flow` and its matching VENC is returned as `LoadedCase.venc`
- the two `Nv=7` VENC triplets may be stored high-first or low-first; equal triplets or conflicting per-axis ordering are rejected because the low/high channel mapping would be ambiguous
- `DV` uses the high-venc triplet as `LoadedCase.venc`; `LV` and `HV` use their selected triplet
- dual-venc thresholds use `configs/loader.json -> background_phase_correction -> dual_venc_ratio1` and `dual_venc_ratio2`; when both remain `0.0`, AutoFlow derives them from the high/low venc ratio

## H5 Layout Resolution

- AutoFlow first checks the H5 root for data keys
- if the root does not directly contain a supported layout, AutoFlow searches nested groups and uses the shallowest group that contains one supported case layout
- if one H5 file contains multiple recognizable 4D flow data groups at the same shallowest depth, AutoFlow treats them as separate input cases instead of forcing one winner
- each such case is identified by its H5 data-group path, for example `StudyA/Series1`
- H5 dataset names are matched case-insensitively for loader keys such as `img_complex`, `img`, `mag`, `flow`, `segmask`, `segmentation`, `seg`, `Resolution`, `Origin`, `RR`, `VENC`, `SpatialOrder`, and `VENCOrder`
- `SpatialOrder` and `VENCOrder` can be stored either as a three-item text array such as `["FH", "RL", "PA"]` or as one comma-separated string such as `"FH,RL,PA"`
- `Resolution`, `Origin`, and `VENC` can be stored as flat triplets, singleton row vectors such as `[[1.2, 1.3, 1.4]]`, or singleton column vectors such as `[[50], [60], [70]]`; AutoFlow flattens these forms during load
- separators such as `_` and `-` are ignored during key matching, so names like `Spatial_Order` and `venc-order` are accepted
- when metadata like `Origin` is missing inside the selected case group, AutoFlow also checks the H5 root before falling back to defaults

## H5 Background Correction Cache

The GUI/CLI/API pipeline separates acquisition loading from the explicit correction action. Low-level `load_h5_data` / `load_input_data` callers may still request correction during a direct load. Noise removal changes only the PC-MRA display region, while background correction and phase unwrapping update working velocity without overwriting source velocity datasets.

- after selecting an H5 case group, the GUI loads acquisition arrays with correction disabled; run Correction explicitly to apply or reuse a compatible cache
- loading does not prompt for correction; choose its method and run it in the Correction stage
- after loading, an embedded `segmask`, `segmentation`, or `seg` is activated without another prompt; when no segmentation is available, the GUI leaves the case unloaded from segmentation work and directs the user to the `Segmentation` stage's explicit `Run Automatic Segmentation` command
- embedded segmentation remains available for review; changing correction does not automatically regenerate it
- when background phase correction is disabled, the correction stage preserves the same normalized output values while bypassing its extra full-volume copy and synthetic complex conversion
- when background phase correction is enabled for an H5 input, AutoFlow first looks for a reusable `corr` dataset in the selected H5 data group, then at the file root
- if no compatible cache is found, AutoFlow runs the configured background-correction method and writes the resulting correction field back to the original H5 as `corr` when the file is writable
- WRLS + ARTO is the default method in Correction; MSAC is also available
- WRLS+ARTO automatically runs its dominant ARTO GMM stage on CUDA when PyTorch and a usable CUDA device are available, with automatic CPU fallback and no device setting
- the GUI shows its standard progress dialog for enabled H5 background correction, including method-specific fitting and the H5 cache-write stage; closing the dialog only hides progress
- legacy dual-venc H5 stores separate correction caches as `corr_low` and `corr_high`
- after a dual-venc correction is applied or reused, the GUI Content selector exposes both normalized low- and high-venc correction components (`Corr Low LR/AP/FH` and `Corr High LR/AP/FH`)
- when both dual-venc correction fields must be computed, the low- and high-venc correction passes run concurrently; H5 cache writes remain serialized
- a time-invariant correction cache may use shape `XYZ13`; AutoFlow broadcasts its singleton time dimension across every input time frame
- cached corrections are reused only when their shape, method, algorithm version, fit order, and method-specific parameters match the current load configuration
- cold MSAC correction caches the full-volume polynomial design matrix across its fixed random trials; the random seed, sampled indices, threshold, fit order, correction values, and cache format remain unchanged
- for multi-group H5 files, untagged root-level `corr` caches are not reused across different data-group paths
- if the H5 file cannot be opened for writing, loading still succeeds; AutoFlow simply skips writing the cache

## H5 Orientation Normalization

- AutoFlow uses `SpatialOrder` together with `VENCOrder` or `VencOrder` for all supported H5 layouts, including legacy complex H5, normalized `mag` + `flow`, and real-valued `img[..., 0:4]`.
- combined real and complex `img` arrays must use channels-last `XYZTV`; channel-first inputs are rejected during loading instead of being guessed or transposed.
- loaded arrays are normalized to internal spatial order `LR, AP, FH` and velocity-component order `LR, AP, FH` before downstream processing.
- the GUI labels the normalized render axes and ortho views with `LR, AP, FH`; `spatial_order_raw` describes the source layout and must not be interpreted as the post-load array-axis order.
- opposite-direction labels such as `RL`, `PA`, and `HF` trigger spatial flips and velocity sign flips so the final `flow` stays physically consistent after reordering.
- for real-valued `img[..., 1:4]`, AutoFlow treats values near the full `[-pi, pi]` phase range as phase radians and rescales them to physical velocity with `flow / pi * VENC` during load.
- complex single-VENC H5 inputs retain canonical `phase_wrapped` for the optional phase-unwrapping stage; dual-VENC inputs retain low/high wrapped phases for audit, and only the `DV` selection skips unwrapping.
- for real-valued H5 inputs, this normalization reorders spatial axes and flow components but does not rescale the stored magnitude values.
- `LoadedCase.metadata["spatial_order_raw"]` and `LoadedCase.metadata["venc_order_raw"]` preserve the source labels read from the H5 file.

Segmentation imports also accept NIfTI (`.nii` and `.nii.gz`) 3D or 4D label volumes. NIfTI files exported by the GUI use the loaded voxel spacing and origin in the affine and can be round-tripped through external editors.

## Loader Result Contract

AutoFlow normalizes loaders to `LoadedCase`.

### Required fields

| Field | Meaning |
| --- | --- |
| `mag` | magnitude volume |
| `flow` | velocity field |
| `resolution` | voxel spacing |
| `origin` | world-space origin |
| `venc` | velocity encoding |
| `rr` | RR interval |

### Optional fields

| Field | Meaning |
| --- | --- |
| `segmentation` | embedded or loaded segmentation; can be binary, 3D label, or 4D label mask |
| `tke_array` | optional TKE array |
| `sigma` | optional complex-derived sigma |
| `metadata` | source metadata |
| `source_format` | loader source format label |
| `source_group` | selected H5 data-group path when applicable |
| `capabilities` | explicit downstream capability flags |

### Coordinate contract

- `origin` is the world-space position of local physical coordinate `[0, 0, 0]`.
- For the current axis-aligned loader contract, voxel index `ijk` maps to local physical millimetres as `ijk * resolution` and to world space as `origin + ijk * resolution`.
- Workspace centerlines and `PlaneData.center` use local physical millimetres. VTK rendering and slicing add `origin` exactly once at the geometry boundary.
- Saved plane records use `center` for the local coordinate and `center_world` for the world coordinate.
- A restored workspace preserves its saved `origin`; changing only `origin` must not change sampled area, flow, WSS, TKE, or pressure values.
- A full direction-matrix/oblique affine is not yet part of `LoadedCase`; oblique DICOM remains a loader limitation rather than being approximated through `origin`.

### Capability flags

| Capability | Meaning |
| --- | --- |
| `has_segmentation` | segmentation is available |
| `has_tke` | TKE is available |
| `has_complex_source` | complex-source information exists |
| `supports_wss` | WSS can be computed when segmentation exists |
| `supports_plane_metrics` | plane metrics can be computed when planes and segmentation exist |

## DICOM Notes

- direct DICOM loading is handled in `autoflow/algorithms/dicom.py`
- Siemens signed phase velocity channels are rescaled to physical velocity during direct load
- textual through-plane direction tags such as `IN` or `TH` are interpreted as slice-select velocity components during case detection
- DICOM-derived mag and flow inputs do not reconstruct fake TKE

## Segmentation Expectations By Input

| Input type | Embedded segmentation | External segmentation | Auto segmentation |
| --- | --- | --- | --- |
| legacy complex H5 | yes when present | yes | yes |
| normalized H5 | yes when present | yes | yes |
| direct DICOM | usually no | yes | yes |

## Grouped Multi-Label Segmentation Behavior

- downstream vessel steps accept binary masks, 3D label masks, and 4D label masks
- 4D label masks are reduced to a 3D label volume by majority vote along the time axis before skeleton, graph, plane, or pathline generation
- connected-component cleanup runs per label value before groups are built
- label values are merged into groups using `configs/labels.json -> label_map` and `label_groups`
- each grouped vessel mask uses the skeleton connected-component cleanup rule instead of always collapsing to one largest component
- each group can override preprocessing through `configs/labels.json -> label_groups.<group>.preprocess`, including Gaussian smoothing, dilation, erosion, opening, and closing
- when only one foreground label exists, AutoFlow falls back to one group so single-label segmentations still work with the same pipeline

## Code References

- H5 loading: `autoflow/algorithms/data.py`
- DICOM scan and load: `autoflow/algorithms/dicom.py`
- pipeline load step: `autoflow/core/pipeline.py`
- types: `autoflow/case_types.py`

Loading does not compute PC-MRA, even when correction or segmentation caches exist. Generate it explicitly as the final Correction step. Raw magnitude/velocity remain available in the right Content menu immediately after loading; derived entries appear only after their data exists.
