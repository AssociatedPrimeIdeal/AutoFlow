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
| `aorta_systemic_branches` | AAO, ARCH, DAO, RBCT, CCA, LBCT, HA, SMA, LRA, RRA | `#c92a2a` | `#e03131` | `#f76707` | `#ffd43b` | `{"gaussian_enabled":true,"gaussian_sigma":0.5,"dilation_iters":0,"erosion_iters":0,"opening_iters":0,"closing_iters":0}` | none |
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

## Learned phase-unwrapping adapters

Put keys inside `phase_unwrapping.backend_params.pudip` or `.gust`. AutoFlow passes the following supported constructor controls; unsupported keys are not forwarded blindly. Input VENC/device come from the case and selected top-level method/device.

| Parameter | Type | Adapter default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `pudip.level` | int | 4 | phase_unwrapping.backend_params | DIP network depth | `autoflow/algorithms/phase_unwrapping/pudip.py` |
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

Metric `render.bar_cfg` and shared `colorbar.bar_cfg` accept `position_x`, `position_y`, `width`, `height` as normalized viewport fractions, plus `title_font_size` / `label_font_size` as font sizes. Every default and field is expanded in [the module tables](parameters.md#colorbar). GUI and movies use the shared colourbar slot. Per-metric `bar_cfg` provides fallback fields; `colorbar.bar_cfg` overrides matching fields in both entry points. Text contrast follows `ui.background_color`, and `colorbar.show` is the master visibility switch for both.

## Label plane rules

`planes.plane_mode=by_label` uses a complete measurement list. The default module references `configs/plane_plans/default.json`; use a standalone file with CLI `--plane-config`, GUI `Plane config > Load…`, or API `plane_config`. An omitted label is disabled, exactly like count zero. Lists are never deep-merged with the default labels.

A compact connection plan is:

```json
{
  "label_rules": {
    "AAO": {
      "plane_mode": "landmark", "plane_count": 3,
      "placements": [{"anchor": "label_connections",
                      "position": "fractions", "fractions": [0.2, 0.5, 0.8],
                      "fallback": {"anchor": "label_level", "level": 1,
                                   "path_selection": "longest", "position": "fractions",
                                   "fractions": [0.2, 0.5, 0.8]}}]
    },
    "MPA": {
      "plane_mode": "landmark", "plane_count": 1,
      "placements": [{"anchor": "label_connections", "position": "fractions", "fractions": [0.5]}]
    }
  }
}
```

This enables only AAO and MPA. A bare name-to-rule object is also accepted. `label_connections` retains every real arm of the target label, including two LRA paths at the same DAO junction. Detection is shared across labels and requires contiguous positive-label samples on the physical graph; it never bridges background or joins groups by proximity. `near` begins at the connection-side target-label boundary and moves into that label. `fractions` distributes positions over the selected own-label interval. For an interval shorter than twice the offset, its midpoint is used with a QC flag. AAO has an explicit fallback for components with no observed connection; this example's MPA rule omits such components.

```json
{
  "label_rules": {
    "BA": {
      "plane_mode": "landmark", "plane_count": 1,
      "placements": [
        {"anchor": "label_level", "level": 2, "path_selection": "longest",
         "when_connected_to": ["LVA", "RVA"], "otherwise_level": 1,
         "position": "fractions", "fractions": [0.5]}
      ]
    }
  }
}
```

This BA rule uses L2 only when BA actually connects to both VA labels in the same physical component; having both labels elsewhere is insufficient. `label_level` reads stored levels (stored 0 = L1), while `label_longest` selects the longest continuous own-label interval regardless of level. Exact longest-length ties and missing levels are omitted with QC. A declared `fallback` applies only to components without a connection, never to a connected arm failing section checks. Explicit older named-anchor plans remain readable. See [Planes](../features/planes.md#default-label-measurements) for each label's defaults.

| Parameter | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `label_rules_file` | path string | `plane_plans/default.json` in shipped module; empty built-in fallback | `planes.json`; workspace field of the same name | Relative paths resolve against the module directory; the file supplies the complete list; do not combine with inline label_rules in the same module | `autoflow/config.py` |
| `plane_config` / `--plane-config` | path string | empty: use the module's list | API / CLI; GUI Plane config file | Standalone complete list; CLI/API relative paths resolve against the working directory | `autoflow/api.py; autoflow/cli.py` |
| `label_rules` / `plane_label_rules` | object | module's selected list; API null inherits it | JSON / API; GUI per-label table | Names resolve through labels.label_map, or use positive numeric IDs as string keys. Explicit API mappings are complete; omitted labels are disabled | `autoflow/plane_rules.py` |
| `<label>.plane_count` | int | selected plan's primary position count | external JSON / GUI count | 0 disables; positive count matches primary declared positions. Every resolved arm uses these positions; fallback can declare a different count | `autoflow/plane_rules.py` |
| `<label>.plane_mode` | enum | `landmark` in shipped plans | detailed JSON / GUI Mode | Execute connection/path placements; explicitly saved old global-layout plans remain compatible | `autoflow/algorithms/landmark_planes.py` |
| `<label>.placements` | non-empty array | label-specific shipped list | JSON / GUI structured Edit dialog | Declared connection/path and position specifications | `autoflow/plane_rules.py` |
| `placements[].anchor` | enum | required | same | label_connections, label_longest, label_level, label_terminal; legacy named anchors pulmonary_split, portal_confluence, pv_first_split, portal_trunk, vertebral_confluence, pca_split, left_carotid_split, right_carotid_split, sinus_confluence, aortic_origin, arch_route remain supported | `autoflow/algorithms/landmark_planes.py` |
| `placements[].position` | enum | required | same | near or fractions | same |
| `placements[].offset_mm` | nonnegative finite float | 5 | same | near-boundary offset inside the target label; configurable engineering distance | same |
| `placements[].fractions` | non-empty array of finite floats in (0,1) | required for fractions | same | Positions within the selected label interval or route | same |
| `placements[].level` | positive int | 1 | same | One-based label-local path level for label_level | same |
| `placements[].path_selection` | enum | unique for explicit legacy level plans; default level plans specify longest | JSON / GUI Paths at level | unique requires one path; longest chooses the longest target-label interval and omits exact ties; all measures each matching path. Applied independently per physical component. label_longest always uses longest interval | `autoflow/algorithms/landmark_planes.py` |
| `placements[].fallback` | placement object | explicitly declared in default connection plans | JSON / GUI No connection | One non-connection placement for each component lacking a connection; cannot nest fallbacks or refer to label_connections | same |
| `placements[].when_connected_to` | non-empty list[string] | BA: [LVA, RVA]; otherwise absent | JSON / GUI Level condition | Use the selected label_level only if all listed labels actually connect to the target in this component | same |
| `placements[].otherwise_level` | positive int | BA: 1; required with when_connected_to | same | Alternative one-based level when the connection condition fails; missing selected levels are omitted | same |
| `<label>.spacing_ratio` | float, (0,1] | 1/3 for count 2; 0.3 otherwise | detailed rule / GUI Spacing ratio or Parameters | Fractional spacing for fixed_step; positions outside the trimmed interval are omitted | `autoflow/algorithms/planes.py` |
| `<label>.spacing_mode` | enum | `fraction` | detailed JSON / GUI Parameters | `fraction` or `distance` for fixed_step | `autoflow/algorithms/planes.py` |
| `<label>.cross_section_distance` | positive float, mm | 5 | same | Physical spacing for distance, anchored_offset, or fixed_step with distance spacing | same |
| `<label>.anchor` | enum | `center` | same | start, center, end, or junction for fixed_step; fallback endpoint for anchored_offset | same |
| `<label>.direction` | enum | `both` | same | toward_start, toward_end, or both for fixed_step | same |
| `<label>.anchor_offset_mm` | nonnegative float, mm | 5 | same | Distance from the junction in anchored_offset, matching the existing global mode | same |
| `<label>.start_distance`, `<label>.end_distance` | nonnegative float, mm | 0, 0 | same | Trim the flow-oriented start and end of the selected root | same |
| `<label>.segmentation_filter` | bool or null | null, inherit global true | same | Enable branch-support filtering; label-contained plane centers remain required | `autoflow/core/pipeline.py` |
| `<label>.fov_boundary_filter` | bool or null | null, inherit global true | same | Enable FOV contact filtering for this label | `autoflow/core/pipeline.py` |
| `<label>.fov_boundary_max_fraction` | float [0,1] or null | null, inherit global 0.25 | same | Maximum outer-ROI perimeter fraction touching an actual image-box face | `autoflow/algorithms/metrics/fov.py` |
| `<label>.smoothing_window` | positive int or null | null, inherit global 15 | same | Plane geometry smoothing window, multiplied by the interpolation factor | `autoflow/algorithms/label_planes.py` |
| `<label>.smoothing_polyorder` | nonnegative int or null | null, inherit global 2 | same | Savitzky–Golay order for this label's plane geometry | same |
| `<label>.inter_time` | positive int or null | null, inherit global 10 | same | Path interpolation factor for this label's geometry | same |
| `<label>.main_path` | legacy enum | absent from landmark plans | explicitly saved old plans only | Discarded for landmark plans; never used as a fallback | `autoflow/plane_rules.py` |

The five regional files are `cardiopulmonary.json` (12 primary positions), `portal.json` (3), `renal.json` (2), `brain.json` (14) and `neck.json` (8). These are configuration counts, not fixed output totals: connected arms, per-component fallbacks and section QC determine actual counts. Brain includes ICA, PCA and ACA. Neck ICA/VA use connected offsets, or longest L1 at 33%/66% without connections. IJV always uses longest L1 at 1/3 and 2/3, without radius weighting. PV/SMV/SV and StrS/LTS/RTS/SSS use connected 15 mm offsets (midpoint below 30 mm), or longest-interval midpoint when unconnected. The combined default lists all 32 labels and disables only ARCH, HA and SMA; their paths still participate in topology.

GUI Load replaces the whole table. GUI file Save writes the displayed table in the same format used by CLI; the settings dialog's bottom Save applies it to the workspace. File edits outside a running GUI take effect after Load or restart. Older `path_index`, `segment_index`, `require_pv_connection` and `reason` fields are accepted on read but discarded; they cannot override automatic selection and are not saved again.

Global layout fields do not override declared positions. Support/FOV filters and smoothing/interpolation inherit shared values. GUI rows show readable summaries, including fallback and level conditions; Edit provides structured controls and automatic primary-count updates. JSON files require primary fractions and plane_count to agree. QC includes resolved intervals, search shifts, counts and omissions. Existing selected_main_paths fields store placement intervals. Connection provenance uses `autoflow.label_connection_placement.v1`, including the requested anchor, connection ID/kind, numeric connected labels, connection side and fallback flag. Fractions and selected levels are recorded when applicable.

Older explicit files remain readable. The following is a legacy layout, not the new default or an automatic fallback:

```json
{"label_rules": {
  "AAO": {"plane_count": 3, "plane_mode": "uniform", "start_distance": 10, "end_distance": 0},
  "RPA": {"plane_count": 1, "plane_mode": "fixed_step", "anchor": "center"}
}}
```


## Optional overrides and compatibility

These accepted controls may be absent from shipped JSON.

| Parameter | Type | Default | Where configured | Effect / units | Code owner |
| --- | --- | --- | --- | --- | --- |
| `wss.viscosity` | float | fluid.viscosity = 4.0 | configs/wss.json | Override shared WSS viscosity in mPa s | `autoflow/config.py; autoflow/algorithms/metrics/wss.py` |
| `tke.rho` | float | fluid.rho = 1060.0 | configs/tke.json | Override shared density in kg/m3 | same |
| `pressure_gradient.rho` | float | fluid.rho = 1060.0 | configs/pressure_gradient.json | Override pressure density in kg/m3 | same |
| `pressure_gradient.viscosity` | float | fluid.viscosity = 4.0 | configs/pressure_gradient.json | Override pressure viscosity in mPa s | same |
| `phase_unwrapping.enabled` | bool | false | configs/phase_unwrapping.json / API | Legacy enable state; method selection is the public opt-in action | `autoflow/config.py; autoflow/api.py` |
| `phase_unwrapping.method` | enum | none | same | none, gc3D, lap4D, nprs, pudip, gust; learned methods require optional dependencies | same |
| `planes.use_center_plane` | bool/null | null | configs/planes.json / API | Legacy centre-plane override; prefer fixed_step placement controls for new workflows | `autoflow/algorithms/planes.py` |

New configurations should use named metric modules rather than `derived.json`. Its historical `wss_*`, `pressure_gradient_*`, `pressure_method`, `vortex_*` and `tke_rho` fields are fallback aliases resolved by `autoflow/config.py:_build_derived_metrics_config`; an explicit modern field takes precedence.

## Rendering groups

Common materials live in `render_style.json`, colourbar layout in `colorbar.json`, and each metric has a `render` object in its own JSON. For example, edit `wss.json` to change both GUI and movie colours:

```json
{"render": {"cmap": "viridis", "opacity": 0.8, "clim": [0, 5], "show_scalar_bar": true}}
```

Relative pressure is nested under `pressure_gradient.json -> render.relative_pressure`. Vortex uses three independent groups under `vortex.json -> render`: `vorticity_magnitude`, `q_criterion`, and `swirling_strength`. The full default/type/effect/owner tables are in [Configuration parameters](parameters.md).

TKE has a `render.volume` object. Its `opacity_points` is a list of `[fraction, opacity]` pairs over the current colour range:

```json
{"render": {"volume": {"opacity_points": [[0, 0], [0.2, 0.015], [0.7, 0.25], [1, 0.65]]}}}
```

Fractions and opacity values must be between zero and one. Start at `[0,0]`, end at fraction `1`, and increase fractions strictly. Intermediate values are interpolated; the layer opacity multiplies the curve. `interpolation` chooses linear/nearest sampling, `blending` selects composite/maximum/minimum/average/additive, and `opacity_unit_distance_scale` is positive. These settings affect the display only. A zero-energy voxel remains transparent even if the colour limits omit zero.


## Live plane display

`planes.json` accepts `render.plane_size_mm` as a finite number from 1 to 200 mm and `render.selected_only` as a boolean. These settings change outlines and picking, not metric support. For example:

```json
{
  "render": {
    "plane_size_mm": 8.0,
    "selected_only": false,
    "default": {"plane_color": "yellow", "plane_opacity": 0.75},
    "groups": {}
  }
}
```

The workspace records live GUI changes in `render_settings.plane_render_cfg`. A null video `plane_size` inherits this value. The selected-plane filter is limited to the GUI.
