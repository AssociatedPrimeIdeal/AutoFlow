# Config System

## Purpose
AutoFlow stores repo-level defaults in per-module JSON files and applies them to CLI, GUI, and Python workflows.

## Config Modules

| File | Main contents |
| --- | --- |
| `configs/batch.json` | output, skip behavior, multithreading, plane reuse |
| `configs/loader.json` | background phase correction, dual-venc loader ratios, and DICOM read settings |
| `configs/skeleton.json` | skeleton cleanup and morphology defaults |
| `configs/labels.json` | label maps, label groups, browser colors, and per-group preprocessing overrides |
| `configs/planes.json` | plane generation defaults |
| `configs/streamlines.json` | streamline and pathline defaults |
| `configs/derived.json` | WSS, TKE, and pressure-gradient defaults |
| `configs/pwv.json` | PWV groups, plane spacing, waveform selection, and plot styling |
| `configs/segmentation.json` | segmentation dock and segmentation-run defaults |
| `configs/rendering.json` | video and camera defaults |

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
| `min_cc_volume_mm3` | float | `50.0` | `configs/skeleton.json` | component-volume threshold | `autoflow/core/models.py` |
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

## PWV Config

| Parameter | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `enabled` | bool | `False` | `configs/pwv.json` | enable or disable PWV computation during plane metrics export | `autoflow/core/pipeline.py` |
| `groups` | list | `[]` | `configs/pwv.json` | each item defines one PWV label combination | `autoflow/core/models.py` |
| `groups[].name` | string | generated name | `configs/pwv.json` | display name in `pwv.json` and the GUI dock | `autoflow/core/models.py` |
| `groups[].labels` | list[int or symbol] | required per group | `configs/pwv.json` | labels merged into one PWV mask | `autoflow/core/models.py` |
| `plane_interval_mm` | float | `10.0` | `configs/pwv.json` | spacing between PWV planes along the longest path | `autoflow/algorithms/pwv.py` |
| `start_distance` | float | `0.0` | `configs/pwv.json` | offset from the path start | `autoflow/algorithms/pwv.py` |
| `end_distance` | float | `0.0` | `configs/pwv.json` | offset from the path end | `autoflow/algorithms/pwv.py` |
| `smoothing_window` | int | `15` | `configs/pwv.json` | path smoothing window used during PWV plane generation | `autoflow/algorithms/pwv.py` |
| `smoothing_polyorder` | int | `2` | `configs/pwv.json` | path smoothing polynomial order | `autoflow/algorithms/pwv.py` |
| `inter_time` | int | `10` | `configs/pwv.json` | interpolation factor used by plane generation | `autoflow/algorithms/pwv.py` |
| `waveform_key` | string | `flowrate_signed_mL_s` | `configs/pwv.json` | metric waveform used for foot detection | `autoflow/algorithms/pwv.py` |
| `foot_savgol_window` | int | `5` | `configs/pwv.json` | Savitzky-Golay window for waveform smoothing | `autoflow/algorithms/pwv.py` |
| `foot_savgol_polyorder` | int | `2` | `configs/pwv.json` | Savitzky-Golay polynomial order | `autoflow/algorithms/pwv.py` |
| `minimum_valid_planes` | int | `2` | `configs/pwv.json` | minimum usable planes required to fit PWV | `autoflow/algorithms/pwv.py` |
| `scene_visible` | bool | `True` | `configs/pwv.json` | initial visibility of the grouped `PWV planes` scene object | `autoflow/core/pipeline.py` |
| `scene_color` | string | `#ffd43b` | `configs/pwv.json` | color of the grouped PWV planes in the 3D view | `autoflow/core/pipeline.py` |
| `plot_color` | string | `#2b8a3e` | `configs/pwv.json` | scatter color in saved and GUI plots | `autoflow/algorithms/pwv.py`, `autoflow/ui/app.py` |
| `fit_color` | string | `#f08c00` | `configs/pwv.json` | fit-line color in saved and GUI plots | `autoflow/algorithms/pwv.py`, `autoflow/ui/app.py` |
| `plot_dpi` | int | `160` | `configs/pwv.json` | PNG plot resolution | `autoflow/algorithms/pwv.py` |

## Runtime Notes

- 4D label masks are collapsed to 3D labels by majority vote along time before grouping.
- connected-component cleanup runs per label value before groups are merged.
- each grouped vessel mask is reduced to its largest connected component before skeletonization.
- grouped scene objects keep stable internal data keys such as `smooth_path_<group>_<idx>`, `plane_<group>_<idx>`, and `pathline_<group>_<idx>`.
- GUI-visible names are intentionally shorter: `path 3`, `plane 5`, and `pathline 5`.
- PWV planes are exposed as one grouped scene object with data key `pwv_planes` and browser name `PWV planes`.

## Current Caveat

The CLI auto-segmentation public config fields are owned by `AutoFlowConfig`. The GUI auto-segmentation dialog is initialized from the segmentation config bundle. Keep this distinction explicit when documenting or refactoring.
