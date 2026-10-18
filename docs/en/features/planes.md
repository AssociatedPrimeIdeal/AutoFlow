# Feature: Planes

## Status

| Entry point | Status | Notes |
| --- | --- | --- |
| GUI | Supported | includes path-constrained generated planes, free manual planes, and direct 3D editing |
| CLI | Supported | default per-label measurements, plus global uniform/fixed-step layouts |
| Python API | Supported | through config and pipeline execution |

## What It Does
Plane generation creates cross-sectional analysis planes along vessel paths.

### Default label measurements

The default `plane_mode=by_label` detects observed cross-label connections once on the physical path multigraph. Each label then controls positions on its own connected arms, or explicitly selects a label-local level. Detection uses the actual labels and topology rather than named anatomical templates. The previous inflow/outflow root selector is not used by these plans. Missing, ambiguous or unsupported positions are reported in QC.

| Labels | Connected position or explicit path rule | When the component has no cross-label connection |
| --- | --- | --- |
| AAO | 20%, 50%, 80% of each selected connected own-label interval | longest L1 own-label interval, 20%, 50%, 80% |
| DAO | always the globally longest continuous own-label interval, 10%, 30%, 50%; connections are ignored | same rule |
| MPA | 20% of each selected connected interval | 20% of the longest L1 interval |
| RBCT, CCA, LBCT, RPA, LPA | 15 mm inside each connected interval | longest continuous own-label interval midpoint |
| ARCH, HA, SMA | disabled; their labels and paths remain available to connection detection | disabled |
| PV | 20%, 50%, 80% along each selected connected own-label interval, measured from its connection side | longest interval at the highest label-local level in the unconnected component, at 20%, 50%, 80% |
| SMV, SV | 15 mm from the connection-side label boundary into the label | longest interval at the highest label-local level in the unconnected component, 20 mm before its flow-oriented end |
| StrS, LTS, RTS | always longest interval at the highest label-local level over the whole label, 20 mm after its flow-oriented start | same rule |
| SSS | always longest L1 interval over the whole label, 20 mm before its flow-oriented end | same rule |
| BA | longest L2 interval midpoint if BA actually connects to both LVA and RVA in this component; otherwise longest L1 midpoint | same conditional level rule |
| LIJV, RIJV | always longest interval at the highest label-local level over the whole label, 1/3 and 2/3; no radius weighting | same rule |
| LICA, RICA, LVA, RVA | 5 mm inside each connected interval | longest L1 interval at 33% and 66% |
| LMCA, RMCA | 5 mm inside each connected interval | longest L1 interval midpoint |
| LRA, RRA | 25%, 50%, 75% of every selected connected own-label interval; two preserved renal arms both receive three planes | longest continuous own-label interval midpoint |
| LPCA, RPCA, ACA | 5 mm inside each connected interval | longest continuous own-label interval midpoint |

The combined default explicitly lists all 32 non-background labels; only ARCH, HA and SMA are disabled. All regional presets use the same per-label rules. The combined plan declares 41 primary positions, which is not a fixed output total: each real connected arm can produce positions, components resolve independently unless `scope=label` selects over the whole label, and a declared fallback can have a different count. Missing labels, unavailable levels and failed section checks reduce the count.

Observed connections require consecutive positive-label samples at an actual label interface. First interfaces on different incident paths can form a local junction region at their shared graph node. Prefixes of the junction's own label leading into foreign arms belong to that region, rather than creating extra parent arms. Every actual foreign arm remains eligible, including two LRA paths connecting to DAO at the same junction. Different groups/components are never joined by proximity, and background gaps are never bridged. Graph and segmentation geometry are unchanged.

`label_level` uses `path_info.hierarchy.label_local_levels`: stored 0 means displayed L1. Default path-property rules explicitly select the longest target-label interval at the requested level; exact ties remain ambiguous and missing levels are omitted. `label_longest` ignores levels. `label_terminal` selects the maximum available label-local level, followed by longest/unique/all selection there. It does not mean the spatially highest point. `scope=label` selects over physical components; the default `component` handles each separately. `when_connected_to` tests actual connections, not mere label presence. An explicit connection `fallback` applies only to unconnected components, never to connected arms failing section QC.

`position=after_start` measures forward from the selected interval's start along reliable average-flow direction; `position=before_end` measures back from its end. Both 20 mm offsets use the midpoint below 40 mm. StrS/LTS/RTS now use the start reference; SSS and the unconnected SMV/SV fallback retain their end reference. PV uses interval fractions for both connected and unconnected selections. Missing reliable direction is reported as `flow_direction_unavailable`. Available label/graph endpoints may reflect scan coverage or segmentation boundaries rather than anatomical vessel termination.

Smoothing only determines placement geometry. Known source-label samples moved outside their label revert to their unsmoothed position. Offsets begin at the target label boundary on the connection side. An interval shorter than twice its requested offset uses its midpoint, explicitly flagged: 15 mm requires at least 30 mm of label interval, and 5 mm requires at least 10 mm. Positions from both ends of the same interval are deduplicated when closer than half the minimum voxel spacing; positions on distinct paths remain separate.

Geometry checks require the center to belong to the target label and apply the existing branch-support and FOV filters. If a nominal section fails, generation searches inside the same selected interval at half the minimum voxel spacing, away from the anchor for near-boundary placement or by nearest displacement for fractions. Search never crosses the next physical fork or a label boundary. Failure remains a QC omission.

Change explicit positions and the matching `plane_count` together. Zero disables a plan. Rule details and JSON examples are in [Label plane rules](../user/parameter-schemas.md#label-plane-rules).

### Physician annotation in CVI

Give the physician one short anatomical instruction per enabled label; omit labels that are not measured. The table below is the project's manual annotation protocol for independent measurements in the corresponding anatomical regions. It does not ask the physician to reproduce AutoFlow graph levels, path choices, exact coordinates, or automatic fallbacks. AAO's upper/middle/lower descriptions are anatomical regions rather than exact replacements for its automatic 20/50/80% positions. For percentage instructions, identify the measured segment's start and end on the reference image.

Measure distances along the vessel and place sections perpendicular to its local direction. Record an unavailable target as not covered. The fixed distances below are project choices, not a manufacturer or universal clinical standard. RBCT and LBCT retain the dataset's label names; identify the corresponding arch branch on the reference image.

| Label | Anatomical instruction for the physician |
| --- | --- |
| AAO | One section each in the upper, middle, and lower ascending aorta. |
| MPA | One section at 20% of the selected main pulmonary artery trunk. |
| DAO | One section each at 10%, 30%, and 50% of the selected descending aorta trunk. |
| RBCT | One section 15 mm beyond this branch's origin from the aortic arch. |
| CCA | One section 15 mm beyond this labelled common carotid artery's arch origin. |
| LBCT | One section 15 mm beyond this branch's origin from the aortic arch. |
| RPA | One section 15 mm beyond the right pulmonary artery's origin from the MPA. |
| LPA | One section 15 mm beyond the left pulmonary artery's origin from the MPA. |
| LRA | Proximal, middle, and distal sections in each left renal artery; measure separate arteries separately. |
| RRA | Proximal, middle, and distal sections in each right renal artery; measure separate arteries separately. |
| PV | Three sections at 20%, 50%, and 80% of the selected portal vein segment, starting from its connection-side junction and ending at the next fork or label boundary. |
| SMV | One section in the SMV, 15 mm before it joins the portal vein. |
| SV | One section in the splenic vein, 15 mm before it joins the portal vein. |
| LICA | Brain: 5 mm before the left ICA's terminal bifurcation. Neck: one section each in the upper-middle and lower-middle trunk. |
| LVA | Brain: 5 mm before the left vertebral artery joins the basilar artery. Neck: one section each in the upper-middle and lower-middle trunk. |
| RVA | Brain: 5 mm before the right vertebral artery joins the basilar artery. Neck: one section each in the upper-middle and lower-middle trunk. |
| RICA | Brain: 5 mm before the right ICA's terminal bifurcation. Neck: one section each in the upper-middle and lower-middle trunk. |
| BA | One section in the middle of the basilar artery trunk. |
| LTS | One section 20 mm downstream of the left transverse sinus start, following flow. |
| SSS | One section in the superior sagittal sinus, 20 mm before the confluence of sinuses. |
| RTS | One section 20 mm downstream of the right transverse sinus start, following flow. |
| StrS | One section 20 mm downstream of the straight sinus start, following flow. |
| LPCA | One section 5 mm beyond the left posterior cerebral artery's origin. |
| RPCA | One section 5 mm beyond the right posterior cerebral artery's origin. |
| LMCA | One section 5 mm beyond the left middle cerebral artery's origin. |
| RMCA | One section 5 mm beyond the right middle cerebral artery's origin. |
| ACA | One section 5 mm beyond each anterior cerebral artery's origin, separately on the left and right. |
| LIJV | Two sections at one-third and two-thirds of the left internal jugular vein's downstream trunk. |
| RIJV | Two sections at one-third and two-thirds of the right internal jugular vein's downstream trunk. |

The batch's `annotation-guide.html` presents these instructions in Chinese with one row per enabled label (29 in the current default). `annotation-rules.json` retains those labels' automatic rules separately from `doctor_instruction`; a manual anatomical instruction is not an exact translation of an automatic selection rule. These files describe the current protocol; existing case results retain the plan saved with that case until planes are regenerated. The portal measurement region is consistent with published [confluence-based 4D flow measurements](https://pmc.ncbi.nlm.nih.gov/articles/PMC11436693/); that source does not prescribe this project's 15 mm offset. Circle's [official 4D Flow overview](https://www.circlecvi.com/cardiac-mr) describes centreline definition and flow assessment in multiple planes; use the installed version's manual for interface instructions.

### Shared external plan files

The shipped `configs/planes.json` references `plane_plans/default.json`. Use one complete file for a case; GUI and CLI read and write the same JSON format.

| Plan | File | Declared primary positions, before topology and fallback |
| --- | --- | --- |
| Cardiopulmonary | `configs/plane_plans/cardiopulmonary.json` | 12 |
| Portal | `configs/plane_plans/portal.json` | 5 |
| Renal (LRA/RRA) | `configs/plane_plans/renal.json` | 6 |
| Brain (ICA, PCA and ACA included) | `configs/plane_plans/brain.json` | 14 |
| Neck (ICA/VA/IJV) | `configs/plane_plans/neck.json` | 8 |

All planes are independent measurements. CVI selects planes independently in the corresponding anatomical regions; it does not need to reproduce AutoFlow's fractions, coordinates or normals.

In grouped multi-label workflows, every plane keeps the group name of the path it came from, so plane and pathline objects can stay grouped in the GUI while still showing short browser-visible names such as `plane 5`. Saved plane outputs now include both `planes.json` and `planes.h5`, and the H5 file stores one root group per plane with the same per-plane payload.

## When To Use It
- use it after graph and path generation
- use evenly-spaced mode with `plane_count=1` for one representative plane per path
- use evenly-spaced mode with `plane_count>1` for repeated analysis distributed across a path
- use fixed-step mode with an anchor, direction, and physical or fractional spacing
- use junction-spacing mode to step away from the nearest graph junction along its branch
- use GUI editing when a generated plane needs path-position or orientation correction
- use `Add Plane` when the measurement plane must be placed independently of every generated path

## Quick Use

### GUI
The structured editor's `Scope` chooses each component or the whole label. `Position` supports `After flow-oriented start`, `Before flow-oriented end`, connection-side offsets and fractions.
Right-click `Generate Planes` and keep `By label`. Choose a preset such as `Neck (ICA / VA / IJV)`. Each row shows its connection/path rule, positions, fallback and optional level condition; unused legacy columns are hidden. Click `Edit…` or double-click the summary to edit anchors, levels, path choices, offsets, percentage positions, enable state, filters and smoothing. `No connection` edits the fallback; `Level condition` configures required connected labels and the alternative level. Count follows the primary declared positions, while output counts depend on resolved arms and fallback. Cancel discards edits. File Save exports the plan; bottom Save applies it. Existing workspaces retain saved rules until a preset/file is loaded.

1. run `Generate Graph`, right-click `Generate Planes` to configure placement, then save and run it, or click `Add Plane` to create a free plane at the current ortho cursor
2. select a plane in the Browser, or right-click inside its visible 3-D outline or near an edge; the picker gives planes priority over underlying paths and the PC-MRA volume
3. click `Edit Plane`; `Add Plane` enters editing immediately. All three labelled handles, guide rings and mouse instructions appear together in the 3D view; there are no movement/rotation mode buttons
4. left-drag the labelled cyan `Move` handle; generated planes stay on their associated path, while manually added planes can move anywhere
5. left-drag the orange `Rotate V` or yellow `Rotate U` endpoint along its guide ring to rotate around the corresponding local axis. Editing handles stay in the foreground and have screen-sized picking targets. 3D clicks keep the edited plane selected; selecting a different Browser item finishes the current edit before navigating
6. selecting a plane automatically replaces the three vertically stacked standard views with `U × V`, `V × N`, and `U × N` views centered on `Plane.center`; clearing the selection restores axial, coronal, and sagittal views
7. inspect the `U × V` view, which is the selected plane itself. Its colored overlay is the active segmentation sampled with nearest-neighbour interpolation, and its cyan outline is the display-smoothed current-time ROI used for area and flow metrics. A saved manual ROI is shown with a dashed yellow outline
8. select `Through-plane Flow (cm/s)` in the main `Content` menu to display velocity projected onto the selected plane normal in all three plane-orthogonal views
9. use `Display → Overlays` for contour smoothing and `Display → View` for default three-view FOV (0.5 means a 2x view-only zoom); changes do not alter metric areas or crop data
10. select `Edit contour` and draw on `U × V`: a closed stroke creates a contour when none exists; with an existing boundary, a stroke crossing it twice replaces the shorter local boundary arc. Endpoints snap automatically, the stroke is smoothed, and outward/inward routes expand/shrink the selected region without an add/remove mode. Each frame has its own contour operation; invalid joins, multiple crossings, and self-intersections are rejected. Submitting an edit updates that frame's metrics, or the selected plane's time series when later empty frames depend on its contour. TKE, WSS, pressure, and other derived volumes are not recomputed. `Undo`, `Redo`, and `Reset frame` are available
11. scalar images use linear interpolation, labels use nearest-neighbour sampling, and each viewer gets an automatic robust window/level from its displayed slice. Right-click zooms back out
12. release a handle to inspect the updated ortho preview, then click `Finish Plane Edit` to save geometry and refresh an active pathline. Existing metrics are recalculated once on Finish; a new or unmeasured plane waits for `Calculate && Save Metrics`. Mouse movement and release do not compute metrics or write files
13. adjust a selected plane's `Opacity` with the Browser slider (or its context menu), then use `Export -> Export Plane Coordinates...` to export selected Browser planes, or all planes when none are selected
14. load a target case, use `File -> Import Plane Coordinates...`, then choose world, local, or relative-centerline mapping and Replace or Append

When `Calculate && Save Metrics` runs in the GUI, its progress dialog advances
after each plane completes and shows the current plane count.

The Browser control area below its tree groups `Display size` and `Only selected plane` with `Visibility` and `Opacity`. The `Display size` slider and millimetre spin box resize every ordinary plane wireframe and the selected-plane highlight immediately, with no plane regeneration or metric calculation. The default is 10 mm, adjustable from 1 to 200 mm. Right-click selection follows the displayed size. `Only selected plane` hides other plane outlines while preserving their Browser visibility flags; select a plane in the Browser to inspect it, or turn the option off to restore the visible outlines. With no selected plane, this option shows no ordinary plane outlines. Editing handles move with the displayed size. `Visibility: Planes` hides or shows all planes, including the selected-plane highlight, while retaining the selection; the switch is disabled only when no plane objects exist.

`configs/planes.json -> render.plane_size_mm` supplies the initial GUI size and the default CLI/GUI plane-video size. `render.selected_only` supplies the GUI filter default; video exports still include the full plane layout. The Python viewer API can change these display settings with `scene.set_plane_display_settings(size_mm=8.0, selected_only=True)` and select a plane with `scene.highlight_plane(uid)`. Video-specific `plane_video.default.plane_size` or group overrides take precedence when non-null. Display settings leave centres, normals, ROI contours, areas, flow, pressure, and saved scientific arrays unchanged.

`Generate Planes` replaces the complete previous plane layout, including manually added or imported planes. Old plane measurements, PWV, pathlines, selection highlights, and edit handles are cleared. Plane wireframes use a foreground renderer that clears its depth each frame while retaining the vessel image; PC-MRA, noise, and segmentation visibility changes cannot occlude their edges. Removing a plane also removes its foreground actor. Previously exported files remain on disk until a later export replaces them.

Explicit older plans using `fixed_step`, `uniform`, `anchored_offset`, `count`, or `distance` remain readable for compatibility. They are not used as a fallback for landmark plans. Global placement fields do not replace declared landmark positions.

Per-label smoothing/interpolation and support/FOV filters can override the shared settings; null means inherit. Label-contained centers are always required. The common displayed smoothed path retains shared smoothing; an individual label's plane geometry uses its own override. Global placement fields do not override the label's layout fields.

In global modes, with the filter enabled, each graph path is densified and sampled against the
majority-voted 3-D label volume.  Non-zero samples are grouped into contiguous
physical-length runs. Graph forks are established from nodes with degree
`>= 3`; flow is then sampled on the filtered path geometry to orient the path.
For a path marked `outgoing` at a fork, runs on the free end half are
considered; for an `incoming` path, runs on the free start half are considered.
The longest eligible run becomes `owner_label`, the path is clipped to that
run, and plane metrics are masked to the same numeric label. This is a
path/plane selection rule, not a skeleton edit, and it cannot correct a source
segmentation that labels a branch as its parent vessel.

After plane placement, generated planes are checked against the rasterized
branch volume. A plane with no cells for its requested branch is omitted from
the plane list and recorded in `plane_qc.json` with
`reason=no_branch_support`. This check uses geometric cell support, not a
numeric flow threshold, so genuinely low-flow planes are not discarded. Path
and graph topology are retained even when all planes for a path are omitted;
the path is reported with `path_status=no_valid_planes`.

Generated planes also use an independent FOV filter, enabled by default. A plane is omitted when the length of its outer ROI contour on the image box faces divided by its complete outer perimeter is **at least 25%**. This uses physical lengths on the temporal-union, label/branch-supported connected ROI, with only numerical coordinate tolerance. Interior vessel walls, label boundaries, isolated tangent points, and holes do not trigger this rule. The image bounds follow VTK cell geometry (`origin` to `origin + shape * spacing`).

After generation, a missing current-phase cross-section uses the most recent valid earlier ROI for both its cyan display contour and plane metrics. Filling continues until a valid contour returns; leading empty phases are not filled. Current-phase velocity is always used. Explicit empty frame edits stop propagation. See [Plane metrics](metrics.md) for provenance fields and derived-field validity.

GUI Generate Planes settings expose `Exclude planes at FOV boundary` and `Maximum FOV contact fraction`. CLI uses `--fov-boundary-max-fraction 0.25` or `--no-fov-boundary-filter`; Python uses `AutoFlowConfig(fov_boundary_filter=True, fov_boundary_max_fraction=0.25)`. Manual/imported planes and later edits are not automatically removed. QC records contact/perimeter lengths, fraction, faces, threshold and `reason=fov_boundary_contact`; an unavailable contour is kept and flagged for review.

### CLI

Default independent label measurements:

```bash
autoflow-run case.h5 --output-dir results/case --plane-config configs/plane_plans/brain.json
```

Select a shared file with `--plane-config`, or set `configs/planes.json -> label_rules_file` relative to that module directory. Inline `label_rules` completely replace the list. Missing labels receive no planes. Inspect `plane_qc.json` for resolved anchors and omissions. To apply a global layout to every path instead:

```bash
autoflow-run case.h5 --output-dir results/case --plane-mode fixed_step \
  --plane-anchor center --plane-direction both --plane-count 3 \
  --plane-spacing-mode fraction --plane-spacing-ratio 0.25
```

Cross-case transfer:

```bash
autoflow-run target.h5 --output-dir results/target \
  --import-planes results/source/plane_positions.json \
  --plane-import-mode path_relative \
  --export-planes results/target/transferred_planes.json
```

### Python API

```python
from autoflow import AutoFlowConfig, run_case
config = AutoFlowConfig.from_config_dir(
    output_dir="./results/case",
    plane_config="configs/plane_plans/brain.json",
)
summary = run_case("case.h5", config=config)
```

An inline complete rule list is supported: `AutoFlowConfig(plane_label_rules={"MPA": {"plane_mode": "landmark", "plane_count": 1, "placements": [{"anchor": "label_connections", "position": "fractions", "fractions": [0.2]}]}})`. It enables only MPA and measures its observed connected arms; add an explicit `fallback` to allow unconnected components.

## Inputs

| Input | Required | Meaning |
| --- | --- | --- |
| graph and paths | for generated planes | geometry used to place and constrain generated planes |
| loaded volume and ortho cursor | for manual planes | defines the initial free-plane position |
| resolution | yes | spacing-aware geometry operations |
| path hierarchy and related labels | for level/landmark plans | label-local levels or distinct connected label arms; missing metadata/anchors omit the affected positions |
| temporal segmentation and branch labels | for geometric validation | existing branch-supported section and FOV checks; flow is not an anatomical-anchor fallback |

## Parameters

See [planes parameters](../user/parameters.md#planes), [labels parameters](../user/parameters.md#labels), [CLI flags](../user/cli-parameters.md), and [API fields](../user/api-parameters.md) for complete type/default/unit/effect/owner tables. Dictionary controls are expanded in [Structured parameters](../user/parameter-schemas.md).

| Parameter or control | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| 3D plane handles | mouse gesture | left-drag | 3D view while editing; temporary | cyan center translates; orange/yellow endpoints rotate around their local axes; all handles appear together | `autoflow/ui/app.py`, `autoflow/ui/plane_handles.py` |

## Outputs

| Output file or object | Created when | Meaning |
| --- | --- | --- |
| `planes.json` | planes exist | serialized plane geometry, optional `roi_polygon_uv_mm`, `segmentation_label`, `placement_mode`, and attached summaries when metrics exist |
| `planes.h5` | planes exist | one root group per plane with geometry, optional manual ROI in `payload_json`, `placement_mode`, `label_name`, path metadata, and metrics |
| `plane_positions.json` | planes exist | v2 portable coordinate file containing world/local centers, optional plane-local ROI polygons, and relative path mapping hints |
| `plane_qc.json` | plane generation or metrics run | explicit rules, raw label intervals, resolved/omitted anchors, selected paths, counts and reasons, branch/FOV decisions; metrics retain this under `plane_layout` |
| plane scene objects | GUI or pipeline plane step succeeds | grouped plane objects with stable internal keys such as `plane_aorta_systemic_branches_5`, shown in the browser as `plane 5` |

`PlaneData.center` and the saved `center` field are local physical millimetres. `center_world` is `center + origin`. Sampling converts local centers to world space only when calling VTK, so a non-zero image origin changes placement in the world scene without changing the sampled cross-section.

Each generated plane's `placement` object records its generation-time anchor nodes, matched/boundary paths, interval, nominal/actual distance, search shift and short-interval flag. Observed-connection and conditional-level plans use schema `autoflow.label_connection_placement.v1`, adding `requested_anchor`, `connection_fallback`, `connection_id`, `connection_kind`, `connected_labels` (numeric IDs) and `near_end`; fraction placements also record `fraction`. QC `selection_source` distinguishes `observed_label_connections`, `label_path_properties` and legacy `anatomical_landmarks`. JSON, H5, coordinate exports and workspace snapshots preserve provenance. Manual edits or imports do not revalidate connections. For cross-case import, `world` preserves `center_world_mm`; `local` preserves `center_local_mm`; `path_relative` matches group plus path rank and recomputes the tangent normal. Manual planes have no centerline fraction.

## Limitations
- end-offset rules need reliable flow orientation; missing direction/levels are omitted with QC rather than choosing an arbitrary endpoint
- highest-level selection can choose a short distal branch; use graph/skeleton views to verify the selected interval
- by-label generation needs a matching `labels.label_map`; workspaces without a name map or numeric rules fall back to global fixed-step placement
- missing/filtered labels have `label_absent`; labels with no usable path have `no_label_path`; plans losing requested positions have `incomplete`, with explicit missing/ambiguous anchor or geometric reasons
- paths without a selected measurement have `path_status=not_selected`; their geometry is retained
- near-boundary placement is relative to the observed connection-side label boundary; graph levels must exist for level-based rules
- observed connections require actual contiguous label interfaces; background gaps use only the declared unconnected fallback, without inventing edges
- explicitly saved legacy named-confluence templates span at most two neighboring physical forks; larger templates are omitted
- default component thresholds can remove small disconnected vessels within one group, so regional plane totals are planned counts rather than a guarantee
- depends on graph and path quality
- distance-plane placement is only meaningful when path geometry is stable
- segmentation filtering cannot infer a vessel identity if the source segmentation labels the entire short branch as its parent vessel; inspect `plane_qc.json` for low owner-label confidence
- planes with no branch-supported cross-section are intentionally omitted; if every plane on a path is omitted, the graph path remains for topology but has no plane metrics
- FOV contact uses a temporal-union probe and does not infer missing vessel area outside the image; the 25% threshold is a perimeter ratio, not an area-loss ratio
- junction-spacing uses the closest graph fork attached to the path; paths without a fork use the configured fallback endpoint
- plane editing is GUI-only
- generated plane centers remain constrained to their associated path; use `Add Plane` for unrestricted placement
- changed geometry invalidates the plane's previous measurements; they are shown as pending during editing. Existing metrics are recomputed on `Finish Plane Edit` or deliberate Browser navigation, while new or unmeasured planes require `Calculate && Save Metrics`
- selected-plane zoom is display-only; it never changes segmentation or the metric ROI
- the selected-plane overlay is sampled from the active mask at the selected time frame with nearest-neighbour interpolation; the cyan metric-ROI contour may apply display smoothing, but the underlying calculation mask is unchanged
- AutoFlow world coordinates are canonical physical millimetres based on origin and spacing, not a general DICOM registration transform

## Where To Change Code

| Change you want | Edit here | Also check | Tests |
| --- | --- | --- | --- |
| default per-label rules and observed connections | `autoflow/plane_rules.py`, `autoflow/algorithms/label_connections.py`, `autoflow/algorithms/landmark_planes.py`, `autoflow/algorithms/label_planes.py` | `configs/plane_plans/`, `autoflow/core/models.py`, `autoflow/core/pipeline.py`, `autoflow/ui/action_settings.py`, `autoflow/ui/landmark_editor.py` | `tests/test_smoke_phantoms.py` |
| plane generation algorithm | `autoflow/algorithms/planes.py` | `autoflow/core/pipeline.py` | `tests/test_smoke_phantoms.py`, `tests/test_pressure_gradient_phantom.py` |
| FOV filtering and exclusion QC | `autoflow/algorithms/metrics/fov.py`, `autoflow/algorithms/metrics/sampling.py` | `autoflow/quality_exclusions.py`, `autoflow/ui/app.py` | `tests/test_smoke_phantoms.py` |
| plane serialization | `autoflow/plane_io.py` | `autoflow/core/pipeline.py` | `tests/test_pressure_gradient_phantom.py` |
| plane geometry editing | `autoflow/ui/app.py`, `autoflow/ui/plane_handles.py` | `autoflow/ui/viewer.py`, `autoflow/ui/remote_plotter.py`, `autoflow/ui/ortho_viewer.py`, `autoflow/algorithms/paths.py` | GUI manual verification plus smoke coverage |

## Tests

- `/home/renyuyang/miniconda3/envs/autoflow311/bin/python -m pytest tests/test_smoke_phantoms.py -q`
- `/home/renyuyang/miniconda3/envs/autoflow311/bin/python -m pytest tests/test_pressure_gradient_phantom.py -q`
- `test_manual_plane_phantom_handles_selection_lock_and_deferred_metrics` exercises real remote-Qt mouse drags on simultaneously available translation/rotation handles, successive rotations with mirrored display axes, selection locking, preservation of existing actors/pathlines, and deferred metric computation

## Common Problems

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| no planes are created | graph or paths are missing | run skeleton and graph first |
| plane positions look poor | path geometry is noisy | improve graph quality or adjust planes manually in the GUI |
| planes are grouped differently than expected | grouped label configuration does not match the label mask | review `configs/labels.json -> label_groups` and regenerate graph and planes |
