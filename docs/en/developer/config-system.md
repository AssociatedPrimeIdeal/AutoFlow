# Config System

## Purpose
AutoFlow stores repo-level defaults in per-module JSON files and applies them to CLI, GUI, and Python workflows.

## Config Modules

| File | Main contents |
| --- | --- |
| `configs/batch.json` | output, skip behavior, multithreading, plane reuse |
| `configs/loader.json` | background phase correction, dual-venc loader ratios, and DICOM read settings |
| `configs/phase_unwrapping.json` | optional traditional phase-unwrapping method, mask, device, and algorithm parameters |
| `configs/skeleton.json` | skeleton cleanup and morphology defaults |
| `configs/labels.json` | label maps, label groups, browser colors, and per-group preprocessing overrides |
| `configs/planes.json` | plane generation defaults |
| `configs/streamlines.json` | live-streamline defaults and video render settings |
| `configs/pathlines.json` | GUI pathline launch, rendering, color, and temporal-cache defaults |
| `configs/derived.json` | legacy compatibility placeholder for older derived config keys |
| `configs/fluid.json` | shared fluid properties such as `rho` and `viscosity` |
| `configs/wss.json` | WSS compute defaults |
| `configs/tke.json` | TKE density defaults |
| `configs/pressure_gradient.json` | pressure-gradient, relative-pressure, and centerline-pressure defaults |
| `configs/pwv.json` | PWV groups, plane spacing, waveform selection, and plot styling |
| `configs/segmentation.json` | segmentation dock and segmentation-run defaults |
| `configs/colorbar.json` | shared GUI colorbar defaults |
| `configs/ui.json` | 3-D viewport display defaults such as background color |
| `configs/video_exporting.json` | shared video and camera defaults |

## Main Code

- `autoflow/config.py`
- `autoflow/api.py`
- `autoflow/ui/app.py`
- `autoflow/cli.py`

## Precedence

### CLI
1. built-in defaults in `autoflow/config.py`
2. overrides from `configs/*.json`
3. explicit CLI flags

### GUI
1. built-in defaults in `autoflow/config.py`
2. overrides from `configs/*.json`
3. interactive UI edits

### Python API
1. dataclass defaults in `AutoFlowConfig`
2. or config-bundle values when using `AutoFlowConfig.from_config_dir()`
3. explicit field overrides in user code

## Skeleton Cleanup Config

| Parameter | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `remove_small_cc` | bool | `True` | `configs/skeleton.json` | remove small connected components before grouped preprocessing | `autoflow/core/models.py` |
| `separate_special_label_contacts` | bool | `True` | `configs/skeleton.json`, GUI, CLI `--separate-special-label-contacts` | separate contacts only between the configured special labels | `autoflow/algorithms/preprocess.py` |
| `special_contact_labels` | list[str] | `["RBCT", "CCA", "LBCT"]` | `configs/skeleton.json` | label names whose pairwise contacts are separated | `autoflow/core/models.py` |
| `min_cc_volume_mm3` | float | `50.0` | `configs/skeleton.json` | component-volume threshold | `autoflow/core/models.py` |
| `cc_filter_mode` | string | `hybrid` | `configs/skeleton.json` | choose `absolute`, `relative`, `hybrid`, or `largest` component filtering | `autoflow/core/models.py` |
| `cc_rel_min_ratio` | float | `0.01` | `configs/skeleton.json` | relative threshold against the largest component for `relative` and `hybrid` filtering | `autoflow/core/models.py` |
| `do_closing` | bool | `True` | `configs/skeleton.json` | morphological closing before skeletonization | `autoflow/algorithms/preprocess.py` |
| `do_opening` | bool | `False` | `configs/skeleton.json` | morphological opening before skeletonization | `autoflow/algorithms/preprocess.py` |
| `gaussian_sigma` | float | `0.5` | `configs/skeleton.json` | smoothing strength | `autoflow/algorithms/preprocess.py` |
| `gaussian_enabled` | bool | `True` | `configs/skeleton.json` | enable or disable Gaussian smoothing | `autoflow/algorithms/preprocess.py` |
| `dilation_iters` | int | `0` | `configs/skeleton.json` | global dilation iterations | `autoflow/algorithms/preprocess.py` |
| `erosion_iters` | int | `0` | `configs/skeleton.json` | global erosion iterations | `autoflow/algorithms/preprocess.py` |
| `opening_iters` | int | `0` | `configs/skeleton.json` | global opening iterations | `autoflow/algorithms/preprocess.py` |
| `closing_iters` | int | `0` | `configs/skeleton.json` | global closing iterations | `autoflow/algorithms/preprocess.py` |

## Label Group Config

| Parameter | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `label_map` | object | built-in vessel defaults | `configs/labels.json` | maps symbolic vessel names to integer labels | `autoflow/config.py` |
| `label_groups` | object | built-in vessel groups | `configs/labels.json` | combines labels into named groups | `autoflow/core/models.py` |
| `label_groups.<group>.browser_color` | string | group-specific | `configs/labels.json` | color of the group title in the GUI browser | `autoflow/core/models.py` |
| `label_groups.<group>.skeleton_color` | string | group-specific | `configs/labels.json` | group skeleton render color | `autoflow/core/models.py` |
| `label_groups.<group>.graph_color` | string | group-specific | `configs/labels.json` | group graph render color | `autoflow/core/models.py` |
| `label_groups.<group>.path_color` | string | group-specific | `configs/labels.json` | group smooth-path render color | `autoflow/core/models.py` |
| `label_groups.<group>.plane_color` | string | group-specific | `configs/labels.json` | group plane render color | `autoflow/core/models.py` |
| `label_groups.<group>.preprocess` | object | `{}` | `configs/labels.json` | per-group preprocessing overrides before skeletonization | `autoflow/algorithms/preprocess.py` |
| `single_label_group_name` | string | `single_label` | `configs/labels.json` | fallback group name for binary or one-label inputs | `autoflow/core/models.py` |
| `single_label_browser_color` | string | `#d9480f` | `configs/labels.json` | fallback browser title color for single-group inputs | `autoflow/core/models.py` |
| `default_group_browser_color` | string | `#1c7ed6` | `configs/labels.json` | browser title color for groups without explicit color | `autoflow/core/models.py` |

## Background Phase Correction Config

| Parameter | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `background_phase_correction.enabled` | bool | `False` | `configs/loader.json` | enable correction during input loading | `autoflow/algorithms/data.py` |
| `background_phase_correction.method` | string | `wrls_arto` | `configs/loader.json`, GUI method dropdown | choose `msac` or `wrls_arto` | `autoflow/algorithms/phase_correction.py`, `autoflow/ui/app.py` |
| `background_phase_correction.corr_fit_order` | int | `3` | `configs/loader.json` | polynomial correction order | `autoflow/algorithms/phase_correction.py` |
| `background_phase_correction.threshold` | float | `0.1` | `configs/loader.json` | MSAC stationary-tissue threshold | `autoflow/algorithms/phase_correction.py` |
| `background_phase_correction.wrls_lambda` | float | `5.0` | `configs/loader.json` | WRLS L1 regularization strength | `autoflow/algorithms/phase_correction.py` |
| `background_phase_correction.wrls_magnitude_threshold` | float | `0.04` | `configs/loader.json` | per-slice reference-magnitude fraction | `autoflow/algorithms/phase_correction.py` |
| `background_phase_correction.wrls_mid_fov_fraction` | float | `0.5` | `configs/loader.json` | initialization in-plane FOV fraction | `autoflow/algorithms/phase_correction.py` |
| `background_phase_correction.wrls_mid_slice_fraction` | float | `0.65` | `configs/loader.json` | initialization through-plane fraction | `autoflow/algorithms/phase_correction.py` |
| `background_phase_correction.wrls_arto_iterations` | int | `2` | `configs/loader.json` | ARTO exclusion and refit count | `autoflow/algorithms/phase_correction.py` |
| `background_phase_correction.wrls_tau` | float | `3.0` | `configs/loader.json` | central-Gaussian inclusion width | `autoflow/algorithms/phase_correction.py` |
| `background_phase_correction.wrls_delta` | float | `2.0` | `configs/loader.json` | minimum side-Gaussian separation | `autoflow/algorithms/phase_correction.py` |
| `background_phase_correction.wrls_central_probability` | float | `0.5` | `configs/loader.json` | minimum central-Gaussian prior | `autoflow/algorithms/phase_correction.py` |
| `background_phase_correction.wrls_fista_iterations` | int | `5000` | `configs/loader.json` | maximum FISTA iterations per fit | `autoflow/algorithms/phase_correction.py` |
| `background_phase_correction.wrls_gmm_iterations` | int | `1000` | `configs/loader.json` | maximum GMM EM iterations per ARTO pass | `autoflow/algorithms/phase_correction.py` |
| `background_phase_correction.dual_venc_ratio1` | float | `0.0` | `configs/loader.json` | first dual-venc alias window ratio | `autoflow/algorithms/data.py` |
| `background_phase_correction.dual_venc_ratio2` | float | `0.0` | `configs/loader.json` | second dual-venc alias window ratio | `autoflow/algorithms/data.py` |
| `background_phase_correction.force_recompute` | bool | `False` | `configs/loader.json` | ignore a compatible H5 correction cache | `autoflow/algorithms/data.py` |

GPU selection is intentionally not part of the config contract. WRLS+ARTO probes PyTorch CUDA internally for the ARTO GMM stage and falls back to CPU after an unavailable or failed CUDA attempt.

## PWV Config

| Parameter | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `enabled` | bool | `False` | `configs/pwv.json` | enable or disable PWV computation in the PWV step | `autoflow/core/pipeline.py` |
| `groups` | list | `[]` | `configs/pwv.json` | each item defines one PWV label combination | `autoflow/core/models.py` |
| `groups[].name` | string | generated name | `configs/pwv.json` | display name in `pwv.json` and the GUI dock | `autoflow/core/models.py` |
| `groups[].labels` | list[int or symbol] | required per group | `configs/pwv.json` | labels merged into one PWV mask | `autoflow/core/models.py` |
| `plane_interval_mm` | float | `10.0` | `configs/pwv.json` | spacing between PWV planes along the longest endpoint-to-endpoint path in the PWV group graph | `autoflow/algorithms/pwv.py` |
| `start_distance` | float | `0.0` | `configs/pwv.json` | offset from the path start | `autoflow/algorithms/pwv.py` |
| `end_distance` | float | `0.0` | `configs/pwv.json` | offset from the path end | `autoflow/algorithms/pwv.py` |
| `smoothing_window` | int | `15` | `configs/pwv.json` | path smoothing window used during PWV plane generation | `autoflow/algorithms/pwv.py` |
| `smoothing_polyorder` | int | `2` | `configs/pwv.json` | path smoothing polynomial order | `autoflow/algorithms/pwv.py` |
| `inter_time` | int | `10` | `configs/pwv.json` | interpolation factor used by plane generation | `autoflow/algorithms/pwv.py` |
| `waveform_key` | string | `flowrate_mL_s` | `configs/pwv.json` | metric waveform used for foot-to-foot timing and as the default PWV waveform family | `autoflow/algorithms/pwv.py` |
| `transit_time_method` | string | `foot_to_foot` | `configs/pwv.json` | choose `foot_to_foot` or `cross_correlation` PWV timing | `autoflow/algorithms/pwv.py` |
| `foot_method` | string | `tangent` | `configs/pwv.json` | choose `tangent` or `threshold` foot detection for `foot_to_foot` | `autoflow/algorithms/pwv.py` |
| `foot_savgol_window` | int | `5` | `configs/pwv.json` | Savitzky-Golay window for waveform smoothing | `autoflow/algorithms/pwv.py` |
| `foot_threshold_percent` | float | `10.0` | `configs/pwv.json` | threshold level used when `foot_method=threshold` | `autoflow/algorithms/pwv.py` |
| `xcorr_window` | string | `full` | `configs/pwv.json` | choose `full` waveform or `upstroke` window for cross-correlation timing | `autoflow/algorithms/pwv.py` |
| `xcorr_interp_factor` | int | `10` | `configs/pwv.json` | cyclic waveform interpolation factor used before cross-correlation timing | `autoflow/algorithms/pwv.py` |
| `allow_cycle_wrap` | bool | `True` | `configs/pwv.json` | allow one-cycle wrap correction and cycle-aware foot detection across the frame boundary | `autoflow/algorithms/pwv.py` |
| `foot_savgol_polyorder` | int | `2` | `configs/pwv.json` | Savitzky-Golay polynomial order | `autoflow/algorithms/pwv.py` |
| `minimum_valid_planes` | int | `2` | `configs/pwv.json` | minimum usable planes required to fit PWV | `autoflow/algorithms/pwv.py` |
| `scene_visible` | bool | `True` | `configs/pwv.json` | initial visibility of the grouped `PWV planes` scene object | `autoflow/core/pipeline.py` |
| `scene_color` | string | `#ffd43b` | `configs/pwv.json` | color of the grouped PWV planes in the 3D view | `autoflow/core/pipeline.py` |
| `plot_color` | string | `#2b8a3e` | `configs/pwv.json` | scatter color in saved and GUI plots | `autoflow/algorithms/pwv.py`, `autoflow/ui/app.py` |
| `fit_color` | string | `#f08c00` | `configs/pwv.json` | fit-line color in saved and GUI plots | `autoflow/algorithms/pwv.py`, `autoflow/ui/app.py` |
| `plot_dpi` | int | `160` | `configs/pwv.json` | PNG plot resolution | `autoflow/algorithms/pwv.py` |

## Fluid And Derived Metric Config

| Parameter | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `rho` | float | `1060.0` | `configs/fluid.json` | shared fluid density used by TKE and relative-pressure calculations unless a metric override is supplied | `autoflow/algorithms/metrics.py` |
| `viscosity` | float | `4.0` | `configs/fluid.json` | shared fluid viscosity used by WSS and relative-pressure calculations unless a metric override is supplied | `autoflow/algorithms/metrics.py` |
| `smoothing_iteration` | int | `200` | `configs/wss.json` | surface smoothing iterations for WSS preparation | `autoflow/algorithms/metrics.py` |
| `inward_distance` | float or `auto` | `auto` | `configs/wss.json` | near-wall sample distance for WSS | `autoflow/algorithms/metrics.py` |
| `parabolic_fitting` | bool | `True` | `configs/wss.json` | enable parabolic fitting in WSS estimation | `autoflow/algorithms/metrics.py` |
| `no_slip_condition` | bool | `False` | `configs/wss.json` | enable no-slip boundary handling for WSS | `autoflow/algorithms/metrics.py` |
| `rho` | float | shared from `configs/fluid.json` | `configs/tke.json` | optional TKE-specific density override | `autoflow/algorithms/metrics.py` |
| `rho` | float | shared from `configs/fluid.json` | `configs/pressure_gradient.json` | optional pressure-analysis density override | `autoflow/algorithms/metrics.py` |
| `viscosity` | float | shared from `configs/fluid.json` | `configs/pressure_gradient.json` | optional pressure-analysis viscosity override | `autoflow/algorithms/metrics.py` |
| `method` | string | `least_squares` | `configs/pressure_gradient.json` | choose `least_squares` or `ppe`; large least-squares systems use a cached PyAMG preconditioner with a Jacobi fallback | `autoflow/algorithms/metrics.py` |
| `smoothing_sigma` | float | `0.0` | `configs/pressure_gradient.json` | optional smoothing before pressure-gradient estimation | `autoflow/algorithms/metrics.py` |
| `use_convective_acceleration` | bool | `True` | `configs/pressure_gradient.json` | include convective acceleration in pressure-gradient computation | `autoflow/algorithms/metrics.py` |
| `smoothing_sigma` | float | `0.0` | `configs/vortex.json` | optional spatial Gaussian smoothing before vorticity, Q-criterion, and swirling-strength derivatives; time is not smoothed | `autoflow/algorithms/metrics.py` |
| `support_erosion_iters` | int | `1` | `configs/vortex.json` | inward lumen-mask erosion defining valid vortex-derivative voxels; does not edit the segmentation | `autoflow/algorithms/metrics.py` |

## Rendering Config

| Parameter | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `window_size` | list[int, int] | `[1600, 1200]` | `configs/video_exporting.json` | output render size for plane, WSS, TKE, pressure-gradient, relative-pressure, and streamline videos | `autoflow/rendering/videos.py` |
| `plane_video.show_skeleton` | bool | `True` | `configs/video_exporting.json` | show or hide skeleton points in plane videos | `autoflow/rendering/videos.py` |
| `plane_video.skeleton_point_size` | float | `10.0` | `configs/video_exporting.json` | skeleton point size in plane videos | `autoflow/rendering/videos.py` |
| `plane_video.default.skeleton_color` | string | empty | `configs/video_exporting.json` | fallback skeleton color for plane videos | `autoflow/rendering/videos.py` |
| `plane_video.default.plane_size` | float or null | `null` | `configs/video_exporting.json` | fallback plane size; `null` uses the automatic size from the vessel surface extent | `autoflow/rendering/videos.py` |
| `plane_video.default.plane_color` | string | `yellow` | `configs/video_exporting.json` | fallback plane color for plane videos | `autoflow/rendering/videos.py` |
| `plane_video.default.plane_opacity` | float | `0.75` | `configs/video_exporting.json` | fallback plane opacity for plane videos | `autoflow/rendering/videos.py` |
| `plane_video.label.prefix` | string | `planeidx=` | `configs/video_exporting.json` | plane-video index label prefix before the plane number | `autoflow/rendering/videos.py` |
| `plane_video.label.font_size` | int | `28` | `configs/video_exporting.json` | plane-video index label font size | `autoflow/rendering/videos.py` |
| `plane_video.label.text_color` | string | `black` | `configs/video_exporting.json` | plane-video index label text color | `autoflow/rendering/videos.py` |
| `plane_video.label.shape_color` | string | `yellow` | `configs/video_exporting.json` | plane-video index label background color | `autoflow/rendering/videos.py` |
| `plane_video.label.shape_opacity` | float | `0.85` | `configs/video_exporting.json` | plane-video index label background opacity | `autoflow/rendering/videos.py` |
| `plane_video.groups.<group>.skeleton_color` | string | group fallback | `configs/video_exporting.json` | per-group skeleton color in plane videos | `autoflow/rendering/videos.py` |
| `plane_video.groups.<group>.plane_size` | float or null | `null` | `configs/video_exporting.json` | per-group plane size in plane videos | `autoflow/rendering/videos.py` |
| `plane_video.groups.<group>.plane_color` | string | group fallback | `configs/video_exporting.json` | per-group plane color in plane videos | `autoflow/rendering/videos.py` |
| `plane_video.groups.<group>.plane_opacity` | float | `0.75` | `configs/video_exporting.json` | per-group plane opacity in plane videos | `autoflow/rendering/videos.py` |
| `rotate_dynamic_video` | bool | `True` | `configs/video_exporting.json` | rotate dynamic WSS, TKE, pressure-gradient, relative-pressure, and streamline videos; set `False` for a fixed view | `autoflow/rendering/videos.py` |
| `add_plane_idx` | bool | `True` | `configs/video_exporting.json` | show or hide plane index labels in the plane video | `autoflow/rendering/videos.py` |
| `wss.render.clim` | list[float, float] | `[0.0, 10.0]` | `configs/wss.json` | WSS color range for GUI and offline videos | `autoflow/rendering/videos.py`, `autoflow/core/pipeline.py` |
| `wss.render.show_scalar_bar` | bool | `True` | `configs/wss.json` | show or hide the WSS colorbar in GUI and offline videos | `autoflow/rendering/videos.py`, `autoflow/core/pipeline.py` |
| `wss.render.bar_cfg` | object | built-in default | `configs/wss.json` | WSS scalar-bar placement and font settings for offline video | `autoflow/rendering/videos.py` |
| `tke.render.clim` | list[float, float] | `[0.0, 100.0]` | `configs/tke.json` | TKE color range for GUI and offline videos | `autoflow/rendering/videos.py`, `autoflow/core/pipeline.py` |
| `tke.render.show_scalar_bar` | bool | `True` | `configs/tke.json` | show or hide the TKE colorbar in GUI and offline videos | `autoflow/rendering/videos.py`, `autoflow/core/pipeline.py` |
| `tke.render.bar_cfg` | object | built-in default | `configs/tke.json` | TKE scalar-bar placement and font settings for offline video | `autoflow/rendering/videos.py` |
| `pressure_gradient.render.clim` | list[float, float] or `null` | `null` | `configs/pressure_gradient.json` | pressure-gradient color range for GUI and offline videos | `autoflow/rendering/videos.py`, `autoflow/core/pipeline.py` |
| `pressure_gradient.render.show_scalar_bar` | bool | `True` | `configs/pressure_gradient.json` | show or hide the pressure-gradient colorbar in GUI and offline videos | `autoflow/rendering/videos.py`, `autoflow/core/pipeline.py` |
| `pressure_gradient.render.bar_cfg` | object | built-in default | `configs/pressure_gradient.json` | pressure-gradient scalar-bar placement and font settings for offline video | `autoflow/rendering/videos.py` |
| `pressure_gradient.render.relative_pressure_clim` | list[float, float] or `null` | `null` | `configs/pressure_gradient.json` | relative-pressure color range for GUI and offline videos | `autoflow/rendering/videos.py`, `autoflow/core/pipeline.py` |
| `pressure_gradient.render.relative_pressure_show_scalar_bar` | bool | `True` | `configs/pressure_gradient.json` | show or hide the relative-pressure colorbar in GUI and offline videos | `autoflow/rendering/videos.py`, `autoflow/core/pipeline.py` |
| `pressure_gradient.render.relative_pressure_bar_cfg` | object | built-in default | `configs/pressure_gradient.json` | relative-pressure scalar-bar placement and font settings for offline video | `autoflow/rendering/videos.py` |
| `streamlines.render.clim` | list[float, float] or `null` | `null` | `configs/streamlines.json` | explicit streamline velocity range; `null` uses `0` to the P99 finite segmented velocity across all phases in GUI and video | `autoflow/algorithms/streamlines.py`, `autoflow/ui/viewer.py`, `autoflow/rendering/videos.py` |
| `streamlines.render.show_scalar_bar` | bool | `True` | `configs/streamlines.json` | show or hide the streamline colorbar in GUI and offline videos | `autoflow/rendering/videos.py`, `autoflow/core/pipeline.py` |
| `streamlines.render.bar_cfg` | object | built-in default | `configs/streamlines.json` | streamline scalar-bar placement and font settings for offline video | `autoflow/rendering/videos.py` |
| `pathlines.seed_ratio` | float | `0.2` | `configs/pathlines.json` | cross-section sampling ratio in pathline ratio mode | `autoflow/algorithms/streamlines.py` |
| `pathlines.min_seeds` | int | `50` | `configs/pathlines.json` | minimum pathline seed count in ratio mode | `autoflow/algorithms/streamlines.py` |
| `pathlines.seed_mode` | string | `fixed` | `configs/pathlines.json` | choose fixed or cross-section-ratio pathline seed selection | `autoflow/core/models.py` |
| `pathlines.seed_count` | int | `250` | `configs/pathlines.json` | fixed seed count or ratio-mode upper bound | `autoflow/algorithms/streamlines.py` |
| `pathlines.max_steps` | int | `200` | `configs/pathlines.json` | maximum VTK cardiac-frame updates per pathline | `autoflow/algorithms/streamlines.py`, `autoflow/ui/app.py` |
| `pathlines.terminal_speed` | float | `0.01` | `configs/pathlines.json` | VTK pathline stopping threshold in m/s | `autoflow/algorithms/streamlines.py` |
| `pathlines.rng_seed` | int | `0` | `configs/pathlines.json` | deterministic pathline seed selection | `autoflow/algorithms/streamlines.py` |
| `pathlines.tube_radius` | float | `0.25` | `configs/pathlines.json` | GUI pathline tube radius in mm | `autoflow/ui/viewer.py` |
| `pathlines.color` | string | `deepskyblue` | `configs/pathlines.json` | uniform pathline color | `autoflow/core/models.py` |
| `pathlines.color_mode` | string | `per_plane` | `configs/pathlines.json` | automatic uniform, per-plane, or per-group color mode | `autoflow/core/models.py` |
| `pathlines.temporal_cache_mb` | float | `512.0` | `configs/pathlines.json` | capped VTK all-frame cache for all-plane jobs | `autoflow/algorithms/streamlines.py` |
| `colorbar.show` | bool | `True` | `configs/colorbar.json` | show or hide the shared GUI colorbar slot | `autoflow/ui/app.py`, `autoflow/ui/viewer.py` |
| `colorbar.bar_cfg` | object | built-in default | `configs/colorbar.json` | shared GUI colorbar placement and font settings used by every live scalar layer | `autoflow/ui/app.py`, `autoflow/ui/viewer.py` |

## Runtime Notes

- 4D label masks are collapsed to 3D labels by majority vote along time before grouping.
- connected-component cleanup runs per label value before groups are merged.
- each grouped vessel mask now keeps every connected component that passes the active skeleton cleanup rule; the default `hybrid` rule uses `max(min_cc_volume_mm3, cc_rel_min_ratio * largest_component_volume_mm3)`.
- grouped scene objects keep stable internal data keys such as `smooth_path_<group>_<idx>`, `plane_<group>_<idx>`, and `pathline_<group>_<idx>`.
- GUI-visible names are intentionally shorter: `path 3`, `plane 5`, and `pathline 5`.
- PWV planes are exposed as one grouped scene object with data key `pwv_planes` and browser name `PWV planes`.

## Current Caveat

The CLI auto-segmentation public config fields are owned by `AutoFlowConfig`. The GUI auto-segmentation dialog is initialized from the segmentation config bundle. Both use an empty model path to select the bundled model, which is resolved from the installed package rather than the current working directory. Keep this distinction explicit when documenting or refactoring.
