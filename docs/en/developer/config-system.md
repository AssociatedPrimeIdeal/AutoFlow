# Configuration system

AutoFlow keeps defaults in per-module JSON files. The complete key/type/default/effect/owner tables are maintained in [Configuration parameters](../user/parameters.md); nested groups and learned-backend options are in [Structured parameters](../user/parameter-schemas.md).

## Precedence

| Entry point | Resolution order |
| --- | --- |
| CLI | Built-ins in `autoflow/config.py` → selected `configs/*.json` → explicit flags |
| GUI | Built-ins → selected JSON bundle → interactive edits |
| Python `AutoFlowConfig()` | Dataclass class defaults → explicit constructor fields |
| Python `AutoFlowConfig.from_config_dir()` | Built-ins → selected JSON bundle → explicit keyword overrides |

`build_workspace(config)` loads the bundle selected by `config.config_dir`, then applies mapped public API fields. Numerical fields without a direct dataclass member, such as WSS smoothing or PG erosion, come from the module bundle. Do not assume the dataclass defaults equal the shipped JSON: for example direct API WSS `clim` is `[0,10]`, whereas the shipped bundle uses `[0,5]`.

## Ownership

`autoflow/config.py` defines defaults, deep merging, legacy compatibility and bundle-to-API/workspace conversion. `autoflow/api.py` defines public runtime overrides. `autoflow/core/models.py` owns workspace parameter objects. CLI parsing lives in `autoflow/cli.py`; GUI configuration and runtime edits live in `autoflow/ui/app.py`.

Metric-specific JSON density/viscosity keys can override shared `fluid.json` values. Otherwise WSS and pressure use shared viscosity, and pressure/TKE use shared density. `derived.json` is a compatibility placeholder; use the named metric modules for new settings.

## Rendering configuration layout

| File | Ownership |
| --- | --- |
| `ui.json` | Shared GUI/video background |
| `render_style.json` | Anti-aliasing, anatomical context, text contrast and surface materials |
| `colorbar.json` | One shared colourbar layout and master visibility switch |
| `wss.json`, `tke.json`, `streamlines.json` | Metric `render.cmap`, `render.opacity`, `render.clim`, `render.show_scalar_bar`; streamline line width and TKE volume controls |
| `pressure_gradient.json` | Pressure-gradient `render` settings and a nested `render.relative_pressure` group |
| `vortex.json` | `render.vorticity_magnitude`, `render.q_criterion`, `render.swirling_strength` |
| `labels.json`, `planes.json`, `video_exporting.json` | Anatomical-group colours, live plane appearance, and movie/camera controls |

Edit the selected JSON directory and restart the GUI or start a fresh CLI/API run to apply it. Use the current nested pressure schema rather than the former top-level display opacity keys or flat relative-pressure render keys. Shipped metric files do not repeat colourbar layout fields.

`autoflow/config.py` defines fallback defaults, validates display controls and merges the selected files. It stores a resolved `render_style_cfg` in `workspace.render_settings`. Public API overrides and interactive scene-object edits retain precedence. Isolated video jobs carry this resolved style snapshot, so they do not reread a different JSON directory while rendering. `autoflow/rendering/style.py` applies it to VTK; `autoflow/rendering/datasets.py` owns display geometry. Direct video renderers resolve colour limits as explicit argument, scene object, workspace setting, then automatic fallback. TKE volume opacity never fades its quantitative colourbar, and zero energy remains transparent.

## Label measurements and group cleanup

`planes.plane_mode=by_label` is the default. `autoflow/plane_rules.py` owns the shared 32-label connection/path plans and validation. `algorithms/label_planes.py` dispatches `landmark` rules to `algorithms/landmark_planes.py`; its physical multigraph caches observed connections from `algorithms/label_connections.py`. Detection uses consecutive positive-label interfaces and real shared nodes, retaining every foreign arm, including same-label arms, without background bridging or graph edits. Per-label rules control positions, unconnected-component fallback or conditional label levels. ARCH/HA/SMA are disabled. Positive counts match primary declared positions; actual counts depend on resolved arms and fallback. The old flow-root selector remains limited to explicit legacy layouts. `main_path` is absent from normalized connection/path plans. QC and `PlaneData.placement` preserve resolved paths, connection metadata, fallback flags, intervals, search shifts and omissions through snapshots and plane I/O. Standalone files, inline module label_rules and explicit API mappings replace the complete list; omitted labels mean zero. `PlaneGenerationParams.label_rules` persists the rule snapshot, so existing workspaces must load updated files explicitly. CLI --plane-config and GUI Load/Save share the validator and atomic writer. Explicit older named anchors and layouts remain compatible.

Component cleanup runs after group merging and morphology. Closing defaults to off, including the aorta override. The default `skeleton.cc_filter_mode=hybrid`, `min_cc_volume_mm3=50`, `cc_rel_min_ratio=0.10` keeps every component at least max(50 mm³, 10% of the largest). The same rule filters each temporal group frame without skeleton morphology. It does not filter individual labels first. `PipelineEngine.preprocess()` projects final component decisions onto original source support without filling source labels. Per-group overrides remain available. QC records the morphology-before-component order and separate removal counts.

## Updating parameters

`skeleton.graph_connectivity` defaults to 26 and accepts only 6/26. It is persisted in SkeletonParams, group overrides, API batch copies and workspace snapshots. Pipeline, joint cerebral analysis, Willis detection and PWV graph construction pass it explicitly to build_graph_from_points. Hierarchy JSON records the shared and per-group settings. It does not alter preprocessing's 6-neighbor component cleanup. `ui/landmark_editor.py` owns structured placement summaries, fallback editing and conditional levels. Explicit label-level path_selection supports unique/longest/all; default level plans use longest at the stated level, with exact ties omitted.

The unified GUI display dock uses the existing workspace rendering payload:
`render_settings.viewer_display`, with `images.<Content ID>` and `views.<name>`
overrides. `ui/display_settings.py` owns its defaults and controls;
`ui/viewer_display.py` applies them. Local Qt default styles are merged first,
then restored workspace values; per-view overrides take precedence over shared
slice defaults. This GUI state does not change CLI/API configuration or offline
video styles. Controls marked shared (layout, guide text, legend geometry and
contour sampling/edit settings) apply to the viewer as a whole.

Each Content Browser render stores its mode, axis visibility, normalized cut
positions and drag settings in `render_settings.content_renders.<data_key>`.
`render_settings.selected_layer_on_top` defaults to true and saves the Browser
foreground-selection preference. Selected object IDs remain runtime state.
Scene objects store opacity, colormap, scalar window and visibility. These are
workspace-specific settings; local slice-style defaults do not create 3D layers.

1. Add or change the built-in and shipped JSON default.
2. Update the workspace model and config mapping, plus a dataclass field / CLI flag only if it is public there.
3. Update `docs/en/developer/parameter-descriptions.json` with meaning, units and ownership.
4. Update the affected rows in the configuration, structured-parameter, CLI and API references. Keep type, default, configuration location, effect and code owner synchronized with the source.
5. Update the affected feature's behaviour and limitation text.
6. Compare the edited references with the live merged bundle, dataclass fields and argparse declarations, then run `mkdocs build --strict`.

Parameter references are maintained alongside code changes. Review every new or changed control against its declaration and effective shipped default; documentation-site builds check links and rendering, not numerical parameter completeness.


## GUI action settings

`autoflow/ui/action_settings.py` defines field ownership, conditional rows,
transactional drafts and publishing. Workspace parameter objects hold settings;
the main window has no hidden parameter widgets. `app.py` binds action execution
and right-click/Shift+F10 settings. Run All edits the same fields, grouped by
action with nested tabs where needed.

Publishing copies only parameter objects; segmentation image arrays are shared.
Restore defaults reads the active config bundle; input defaults use the current
case's applied calibration. Input changes use the normal transactional reload.

Workspace snapshots retain GUI bookkeeping in `input_state.metadata`:
`gui_input_parameters` stores saved calibration drafts,
`gui_phase_unwrap_masks` remembers per-method mask choices,
`gui_pending_settings` lists computations requiring reruns, and
`gui_pending_pathline_planes` identifies trajectories needing replacement.
These keys add no CLI flags or scientific input fields. A successful input load
starts fresh case metadata. Pathline cache replacement happens only after
successful integration; failed/cancelled reruns retain the old visible result.

Validate through the existing smoke/phantom suite in
`tests/test_smoke_phantoms.py`; do not introduce a separate GUI test file.
