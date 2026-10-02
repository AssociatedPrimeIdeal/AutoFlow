# Structured parameters

These tables expand dictionary/list controls that cannot be explained by a scalar default. Every row lists its type, default, configuration location, effect and owner. See [configuration parameters](parameters.md) for surrounding scalar controls.

## Anatomical label map

`labels.label_map` maps a label name to its integer segmentation class. Zero is background. These names are keys used for grouping and do not rename/relabel an imported mask automatically.

| Name | Type | Shipped ID | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `background` | int | 0 | `labels.json: label_map` | Exclude background from vessel groups | `autoflow/core/models.py` |
| `AAO` | int | 1 | `labels.json: label_map` | Select voxels with this class ID for group membership | `autoflow/core/models.py` |
| `ARCH` | int | 3 | `labels.json: label_map` | Select voxels with this class ID for group membership | `autoflow/core/models.py` |
| `DAO` | int | 4 | `labels.json: label_map` | Select voxels with this class ID for group membership | `autoflow/core/models.py` |
| `RBCT` | int | 5 | `labels.json: label_map` | Select voxels with this class ID for group membership | `autoflow/core/models.py` |
| `CCA` | int | 6 | `labels.json: label_map` | Select voxels with this class ID for group membership | `autoflow/core/models.py` |
| `LBCT` | int | 7 | `labels.json: label_map` | Select voxels with this class ID for group membership | `autoflow/core/models.py` |
| `MPA` | int | 2 | `labels.json: label_map` | Select voxels with this class ID for group membership | `autoflow/core/models.py` |
| `RPA` | int | 8 | `labels.json: label_map` | Select voxels with this class ID for group membership | `autoflow/core/models.py` |
| `LPA` | int | 9 | `labels.json: label_map` | Select voxels with this class ID for group membership | `autoflow/core/models.py` |
| `HA` | int | 10 | `labels.json: label_map` | Select voxels with this class ID for group membership | `autoflow/core/models.py` |
| `SMA` | int | 11 | `labels.json: label_map` | Select voxels with this class ID for group membership | `autoflow/core/models.py` |
| `LRA` | int | 12 | `labels.json: label_map` | Select voxels with this class ID for group membership | `autoflow/core/models.py` |
| `RRA` | int | 13 | `labels.json: label_map` | Select voxels with this class ID for group membership | `autoflow/core/models.py` |
| `PV` | int | 14 | `labels.json: label_map` | Select voxels with this class ID for group membership | `autoflow/core/models.py` |
| `SMV` | int | 15 | `labels.json: label_map` | Select voxels with this class ID for group membership | `autoflow/core/models.py` |
| `SV` | int | 16 | `labels.json: label_map` | Select voxels with this class ID for group membership | `autoflow/core/models.py` |
| `LICA` | int | 17 | `labels.json: label_map` | Select voxels with this class ID for group membership | `autoflow/core/models.py` |
| `LVA` | int | 18 | `labels.json: label_map` | Select voxels with this class ID for group membership | `autoflow/core/models.py` |
| `RVA` | int | 19 | `labels.json: label_map` | Select voxels with this class ID for group membership | `autoflow/core/models.py` |
| `RICA` | int | 20 | `labels.json: label_map` | Select voxels with this class ID for group membership | `autoflow/core/models.py` |
| `BA` | int | 21 | `labels.json: label_map` | Select voxels with this class ID for group membership | `autoflow/core/models.py` |
| `LTS` | int | 22 | `labels.json: label_map` | Select voxels with this class ID for group membership | `autoflow/core/models.py` |
| `SSS` | int | 23 | `labels.json: label_map` | Select voxels with this class ID for group membership | `autoflow/core/models.py` |
| `RTS` | int | 24 | `labels.json: label_map` | Select voxels with this class ID for group membership | `autoflow/core/models.py` |
| `StrS` | int | 25 | `labels.json: label_map` | Select voxels with this class ID for group membership | `autoflow/core/models.py` |
| `LPCA` | int | 26 | `labels.json: label_map` | Select voxels with this class ID for group membership | `autoflow/core/models.py` |
| `RPCA` | int | 27 | `labels.json: label_map` | Select voxels with this class ID for group membership | `autoflow/core/models.py` |
| `LMCA` | int | 28 | `labels.json: label_map` | Select voxels with this class ID for group membership | `autoflow/core/models.py` |
| `RMCA` | int | 29 | `labels.json: label_map` | Select voxels with this class ID for group membership | `autoflow/core/models.py` |
| `ACA` | int | 30 | `labels.json: label_map` | Select voxels with this class ID for group membership | `autoflow/core/models.py` |
| `LIJV` | int | 31 | `labels.json: label_map` | Select voxels with this class ID for group membership | `autoflow/core/models.py` |
| `RIJV` | int | 32 | `labels.json: label_map` | Select voxels with this class ID for group membership | `autoflow/core/models.py` |

## Label group definitions

Keys under `labels.label_groups` are arbitrary group names. Their defaults are:

| Group | Labels | Browser / skeleton colour | Graph colour | Path colour | Plane colour | Preprocessing override | Willis role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `aorta_systemic_branches` | AAO, ARCH, DAO, RBCT, CCA, LBCT, HA, SMA, LRA, RRA | `#c92a2a` | `#e03131` | `#f76707` | `#ffd43b` | `{"gaussian_enabled":true,"gaussian_sigma":0.5,"dilation_iters":0,"erosion_iters":0,"opening_iters":0,"closing_iters":1}` | none |
| `pulmonary_arteries` | MPA, RPA, LPA | `#1971c2` | `#1c7ed6` | `#4dabf7` | `#a5d8ff` | `{}` | none |
| `portal_splenic_venous` | PV, SMV, SV | `#2b8a3e` | `#37b24d` | `#69db7c` | `#b2f2bb` | `{}` | none |
| `intracranial_anterior_arteries` | LICA, RICA, LMCA, RMCA, ACA | `#862e9c` | `#9c36b5` | `#be4bdb` | `#e599f7` | `{}` | anterior |
| `vertebrobasilar_arteries` | LVA, RVA, BA, LPCA, RPCA | `#5f3dc4` | `#7048e8` | `#9775fa` | `#d0bfff` | `{}` | posterior |
| `intracranial_veins` | StrS, RTS, SSS, LTS | `#0b7285` | `#1098ad` | `#3bc9db` | `#99e9f2` | `{}` | none |
| `jugular_veins` | LIJV, RIJV | `#495057` | `#868e96` | `#adb5bd` | `#dee2e6` | `{}` | none |

| Parameter | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `labels` | list[string/int] | group entries above | `labels.label_groups.<name>` | Label names resolve through label_map; integers select class IDs directly | `autoflow/core/models.py` |
| `browser_color` | colour string | default_group_browser_color | same | Default group Browser colour | same |
| `skeleton_color`, `graph_color`, `path_color`, `plane_color` | colour strings | group colour fallback | same | Colour for the corresponding scene object; no numerical effect | same |
| `scene_color` | colour string | browser colour fallback | same | Compatibility group scene colour used when a per-kind colour is absent | same |
| `willis_ring_role` | string | absent | same | Mark anterior/posterior groups for Circle of Willis topology analysis | `autoflow/algorithms/intracranial.py` |
| `preprocess` | object | inherit skeleton defaults | same | Per-group preprocessing overrides below | `autoflow/algorithms/preprocess.py` |
| `preprocess.gaussian_enabled` | bool | inherited | same | Enable Gaussian mask smoothing before extraction | same |
| `preprocess.gaussian_sigma` | float | inherited | same | Gaussian width in voxels | same |
| `preprocess.dilation_iters` | int | inherited | same | Grow foreground by morphological iterations | same |
| `preprocess.erosion_iters` | int | inherited | same | Shrink foreground by morphological iterations | same |
| `preprocess.opening_iters` | int | inherited | same | Remove small foreground projections with opening | same |
| `preprocess.closing_iters` | int | inherited | same | Close small gaps with closing | same |

## DICOM metadata overrides

All fields are optional and otherwise use detected DICOM metadata. Values are interpreted before canonical reorientation; they do not rewrite DICOM files.

| Parameter | Type | Default | Where configured | Effect / units | Code owner |
| --- | --- | --- | --- | --- | --- |
| `spatial_order` | three direction strings | detected | `loader.dicom_parameter_overrides` | Physical directions of the three source spatial axes, e.g. LR/AP/FH with their reversals | `autoflow/algorithms/dicom.py` |
| `venc_order` | three direction strings | detected | same | Physical directions/order of source velocity components | same |
| `resolution` | three floats | detected | same | Source voxel spacing in mm | same |
| `venc` | three floats | detected | same | Source component VENC in cm/s | same |
| `rr` | float | detected | same | Cardiac cycle duration in ms | same |

## Learned phase-unwrapping adapters

Put keys inside `phase_unwrapping.backend_params.pudip` or `.gust`. AutoFlow passes the following supported constructor controls; unsupported keys are not forwarded blindly. Input VENC/device come from the case and selected top-level method/device.

| Parameter | Type | Adapter default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `pudip.level` | int | 4 | phase_unwrapping.backend_params | DIP network depth | `autoflow/algorithms/phase_unwrapping.py` |
| `pudip.features` | int | 128 | same | Feature channels in the network | same |
| `pudip.input_depth` | int | 128 | same | Input noise channel count | same |
| `pudip.lr` | float | 0.001 | same | Optimizer learning rate | same |
| `pudip.num_iter` | int | 1000 | same | Maximum training iterations | same |
| `pudip.tv_weights` | four floats | [1,1,1,1] | same | Spatial and temporal total-variation weights | same |
| `pudip.loss_type` | string | l1 | same | Wrapped-gradient loss norm selected by upstream backend | same |
| `pudip.lr_scheduler` | string | cosine | same | Training learning-rate schedule | same |
| `pudip.div_weight` | float | 0 | same | Divergence loss weight | same |
| `pudip.spacing` | three positive floats | [1,1,1] | same | Backend spatial derivative spacing; explicit backend scale, not an automatic replacement of case metadata | same |
| `pudip.reshape_mode` | string | bt_as_channel | same | How component/time dimensions enter the network | same |
| `gust.voxel_spacing` | three positive floats | [1,1,1] | same | Backend spatial derivative scaling | same |
| `gust.num_primitives` | int | 8192 | same | Gaussian primitive count; affects capacity and VRAM | same |
| `gust.num_iter` | int | 1000 | same | Maximum fitting iterations | same |
| `gust.lr` | float | 0.03 | same | Optimizer learning rate | same |
| `gust.seed` | int | 314159 | same | Random initialization seed | same |
| `gust.center_confidence` | XYZ float array | derived mask/confidence | Python backend mapping | Gaussian-centre confidence; GUI pcmra_std support can supply this explicitly | same |

## PWV group objects

`pwv.groups` is a list of `{"name": "...", "labels": [...]}`. The shipped group is `{"name": "Portal Vein", "labels": ["PV"]}`.

| Parameter | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `name` | string | Portal Vein in shipped group | pwv.groups[] | Result/scene/plot group identifier | `autoflow/algorithms/pwv.py` |
| `labels` | list[string/int] | [PV] in shipped group | same | Vessel classes participating in the PWV path | same |

## Plane and video group styles

`planes.render.groups.<name>` overrides GUI plane colours/opacity. Shipped values are expanded in [plane parameters](parameters.md#planes). `video_exporting.plane_video.groups.<name>` is initially empty and overrides `plane_video.default` fields:

| Parameter | Type | Default | Where configured | Effect / units | Code owner |
| --- | --- | --- | --- | --- | --- |
| `skeleton_color` | colour string | empty: resolved group colour | video_exporting.plane_video.groups.<name> | Colour of group skeleton in plane video | `autoflow/rendering/videos.py` |
| `plane_size` | float/null | 10 in shipped plane_video.default | same | Plane marker size in mm; null uses inferred dimensions | same |
| `plane_color` | colour string | yellow | same | Plane marker colour | same |
| `plane_opacity` | float | 0.75 | same | Plane marker opacity, 0–1 | same |

Metric `render.bar_cfg` and shared `colorbar.bar_cfg` accept `position_x`, `position_y`, `width`, `height` as normalized viewport fractions, plus `title_font_size` / `label_font_size` as font sizes. Every default and field is expanded in [the module tables](parameters.md#colorbar). Per-metric bar_cfg governs offline defaults; the GUI uses the shared colourbar slot.

## Optional overrides and compatibility

These accepted controls may be absent from shipped JSON.

| Parameter | Type | Default | Where configured | Effect / units | Code owner |
| --- | --- | --- | --- | --- | --- |
| `wss.viscosity` | float | fluid.viscosity = 4.0 | configs/wss.json | Override shared WSS viscosity in mPa s | `autoflow/config.py; autoflow/algorithms/metrics.py` |
| `tke.rho` | float | fluid.rho = 1060.0 | configs/tke.json | Override shared density in kg/m3 | same |
| `pressure_gradient.rho` | float | fluid.rho = 1060.0 | configs/pressure_gradient.json | Override pressure density in kg/m3 | same |
| `pressure_gradient.viscosity` | float | fluid.viscosity = 4.0 | configs/pressure_gradient.json | Override pressure viscosity in mPa s | same |
| `phase_unwrapping.enabled` | bool | false | configs/phase_unwrapping.json / API | Legacy enable state; method selection is the public opt-in action | `autoflow/config.py; autoflow/api.py` |
| `phase_unwrapping.method` | enum | none | same | none, gc3D, lap4D, nprs, pudip, gust; learned methods require optional dependencies | same |
| `planes.use_center_plane` | bool/null | null | configs/planes.json / API | Legacy centre-plane override; prefer fixed_step placement controls for new workflows | `autoflow/algorithms/planes.py` |

New configurations should use named metric modules rather than `derived.json`. Its historical `wss_*`, `pressure_gradient_*`, `pressure_method`, `vortex_*` and `tke_rho` fields are fallback aliases resolved by `autoflow/config.py:_build_derived_metrics_config`; an explicit modern field takes precedence.
