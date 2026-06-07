# Inputs

## Supported Inputs

| Input type | Supported | Typical entry points | Notes |
| --- | --- | --- | --- |
| legacy complex H5 | yes | CLI, GUI, Python API | can provide segmentation and complex-derived sigma |
| normalized H5 with `mag` and `flow` | yes | CLI, GUI, Python API | preferred normalized path |
| direct DICOM directory | yes | CLI, GUI, Python API | scanned into importable cases |
| single DICOM file | yes for case collection | CLI | resolved through case collection logic |

## Legacy Complex H5 Variants

- classic legacy complex H5 uses `img_complex[..., 0]` as the magnitude reference and `img_complex[..., 1:4]` as three velocity encodes
- legacy dual-venc H5 with `img_complex.shape[-1] == 7` is supported when the file stores one reference plus six velocity encodes ordered by `VENCOrder`
- for `Nv=7`, AutoFlow interprets channels `1:4` as low-venc encodes and channels `4:7` as high-venc encodes, applies background phase correction to each triplet separately, runs dual-venc alias reconstruction, and publishes the reconstructed flow field as the final `flow`
- the final loader `venc` for `Nv=7` is the high-venc triplet
- dual-venc thresholds use `configs/loader.json -> background_phase_correction -> dual_venc_ratio1` and `dual_venc_ratio2`; when both remain `0.0`, AutoFlow derives them from the high/low venc ratio

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
| `source_group` | top-level H5 group when applicable |
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
- label values are merged into groups using `configs/skeleton.json -> label_map` and `label_groups`
- each group can override preprocessing through `label_groups.<group>.preprocess`, including Gaussian smoothing, dilation, erosion, opening, and closing
- when only one foreground label exists, AutoFlow falls back to one group so single-label segmentations still work with the same pipeline

## Code References

- H5 loading: `autoflow/algorithms/data.py`
- DICOM scan and load: `autoflow/algorithms/dicom.py`
- pipeline load step: `autoflow/core/pipeline.py`
- types: `autoflow/case_types.py`
