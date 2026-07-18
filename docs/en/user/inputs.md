# Inputs

## Supported Inputs

| Input type | Supported | Typical entry points | Notes |
| --- | --- | --- | --- |
| legacy complex H5 | yes | CLI, GUI, Python API | can provide segmentation and complex-derived sigma |
| normalized H5 with `mag` and `flow` | yes | CLI, GUI, Python API | preferred normalized path |
| normalized H5 with real-valued `img[..., 0:4]` or `img[0:4, ...]` | yes | CLI, GUI, Python API | interpreted as `mag + flow_xyz`; channel-first `img[0:4, ...]` is transposed during load |
| direct DICOM directory | yes | CLI, GUI, Python API | scanned into importable cases |
| single DICOM file | yes for case collection | CLI | resolved through case collection logic |

## Legacy Complex H5 Variants

- classic legacy complex H5 uses `img_complex[..., 0]` as the magnitude reference and `img_complex[..., 1:4]` as three velocity encodes
- complex-valued `img[..., 0:4]` is treated the same as `img_complex[..., 0:4]`
- legacy dual-venc H5 with `img_complex.shape[-1] == 7` is supported when the file stores one reference plus six velocity encodes ordered by `VENCOrder`
- for `Nv=7`, `VENC` contains two consecutive three-axis triplets corresponding to channel groups `1:4` and `4:7`; AutoFlow compares the triplets component-wise to identify the low-venc and high-venc groups, keeps each triplet paired with its source channels, applies background phase correction separately, runs dual-venc alias reconstruction, and publishes the reconstructed flow field as the final `flow`
- the two `Nv=7` VENC triplets may be stored high-first or low-first; equal triplets or conflicting per-axis ordering are rejected because the low/high channel mapping would be ambiguous
- the final loader `venc` for `Nv=7` is the high-venc triplet
- dual-venc thresholds use `configs/loader.json -> background_phase_correction -> dual_venc_ratio1` and `dual_venc_ratio2`; when both remain `0.0`, AutoFlow derives them from the high/low venc ratio

## H5 Layout Resolution

- AutoFlow first checks the H5 root for data keys
- if the root does not directly contain a supported layout, AutoFlow searches nested groups and uses the shallowest group that contains one supported case layout
- if one H5 file contains multiple recognizable 4D flow data groups at the same shallowest depth, AutoFlow treats them as separate input cases instead of forcing one winner
- each such case is identified by its H5 data-group path, for example `StudyA/Series1`
- H5 dataset names are matched case-insensitively for loader keys such as `img_complex`, `img`, `mag`, `flow`, `segmask`, `segmentation`, `Resolution`, `Origin`, `RR`, `VENC`, `SpatialOrder`, and `VENCOrder`
- `SpatialOrder` and `VENCOrder` can be stored either as a three-item text array such as `["FH", "RL", "PA"]` or as one comma-separated string such as `"FH,RL,PA"`
- `Resolution`, `Origin`, and `VENC` can be stored as flat triplets, singleton row vectors such as `[[1.2, 1.3, 1.4]]`, or singleton column vectors such as `[[50], [60], [70]]`; AutoFlow flattens these forms during load
- separators such as `_` and `-` are ignored during key matching, so names like `Spatial_Order` and `venc-order` are accepted
- when metadata like `Origin` is missing inside the selected case group, AutoFlow also checks the H5 root before falling back to defaults

## H5 Background Correction Cache

- when background phase correction is enabled for an H5 input, AutoFlow first looks for a reusable `corr` dataset in the selected H5 data group, then at the file root
- if no compatible cache is found, AutoFlow runs MSAC background phase correction and writes the resulting correction field back to the original H5 as `corr` when the file is writable
- legacy dual-venc H5 stores separate correction caches as `corr_low` and `corr_high`
- a time-invariant correction cache may use shape `XYZ13`; AutoFlow broadcasts its singleton time dimension across every input time frame
- cached corrections are reused only when their shape, algorithm version, `corr_fit_order`, and `threshold` match the current load configuration
- for multi-group H5 files, untagged root-level `corr` caches are not reused across different data-group paths
- if the H5 file cannot be opened for writing, loading still succeeds; AutoFlow simply skips writing the cache

## H5 Orientation Normalization

- AutoFlow uses `SpatialOrder` together with `VENCOrder` or `VencOrder` for all supported H5 layouts, including legacy complex H5, normalized `mag` + `flow`, and real-valued `img[..., 0:4]` or `img[0:4, ...]`.
- for real-valued channel-first `img[0:4, ...]`, AutoFlow first transposes the raw array into internal `XYZT4` or `XYZ4` order before applying the usual spatial-axis and velocity-component normalization.
- loaded arrays are normalized to internal spatial order `LR, AP, FH` and velocity-component order `LR, AP, FH` before downstream processing.
- opposite-direction labels such as `RL`, `PA`, and `HF` trigger spatial flips and velocity sign flips so the final `flow` stays physically consistent after reordering.
- for real-valued `img[..., 1:4]`, AutoFlow treats values near the full `[-pi, pi]` phase range as phase radians and rescales them to physical velocity with `flow / pi * VENC` during load.
- for real-valued H5 inputs, this normalization reorders spatial axes and flow components but does not rescale the stored magnitude values.
- `LoadedCase.metadata["spatial_order_raw"]` and `LoadedCase.metadata["venc_order_raw"]` preserve the source labels read from the H5 file.

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
