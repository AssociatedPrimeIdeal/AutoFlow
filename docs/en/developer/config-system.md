# Config System

## Purpose
AutoFlow stores repo-level defaults in per-module JSON files and applies them to CLI, GUI, and Python workflows.

## Config Modules

| File | Main contents |
| --- | --- |
| `configs/batch.json` | output, skip behavior, multithreading, plane reuse |
| `configs/loader.json` | background phase correction, dual-venc loader ratios, and DICOM read settings |
| `configs/skeleton.json` | skeleton preprocessing defaults, grouped label maps, grouped colors, and per-group preprocessing overrides |
| `configs/planes.json` | plane generation defaults |
| `configs/streamlines.json` | streamline and pathline defaults |
| `configs/derived.json` | WSS, TKE, and pressure-gradient defaults |
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

## Grouped Skeleton Config

`configs/skeleton.json` now owns both classic skeleton cleanup settings and grouped multi-label behavior.

| Parameter | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `label_map` | object | built-in vessel defaults | `configs/skeleton.json` | maps symbolic vessel names to integer labels | `autoflow/config.py` |
| `label_groups` | object | built-in vessel groups | `configs/skeleton.json` | combines labels into named groups | `autoflow/core/models.py` |
| `label_groups.<group>.browser_color` | string | group-specific | `configs/skeleton.json` | color of the group title in the GUI browser | `autoflow/core/models.py` |
| `label_groups.<group>.skeleton_color` | string | group-specific | `configs/skeleton.json` | group skeleton render color | `autoflow/core/models.py` |
| `label_groups.<group>.graph_color` | string | group-specific | `configs/skeleton.json` | group graph render color | `autoflow/core/models.py` |
| `label_groups.<group>.path_color` | string | group-specific | `configs/skeleton.json` | group smooth-path render color | `autoflow/core/models.py` |
| `label_groups.<group>.plane_color` | string | group-specific | `configs/skeleton.json` | group plane render color | `autoflow/core/models.py` |
| `label_groups.<group>.preprocess` | object | `{}` | `configs/skeleton.json` | per-group preprocessing overrides before skeletonization | `autoflow/algorithms/preprocess.py` |
| `single_label_group_name` | string | `single_label` | `configs/skeleton.json` | fallback group name for binary or one-label inputs | `autoflow/core/models.py` |
| `single_label_browser_color` | string | `#d9480f` | `configs/skeleton.json` | fallback browser title color for single-group inputs | `autoflow/core/models.py` |
| `default_group_browser_color` | string | `#1c7ed6` | `configs/skeleton.json` | browser title color for groups without explicit color | `autoflow/core/models.py` |

Runtime notes:

- 4D label masks are collapsed to 3D labels by majority vote along time before grouping.
- connected-component cleanup runs per label value before groups are merged.
- grouped scene objects use stable prefixes such as `segmask_group_<group>`, `skeleton_<group>`, `graph_<group>`, `smooth_path_<group>_<path_idx>`, `plane_<group>_<plane_idx>`, and `pathline_<group>_<plane_idx>`.
- browser grouping lives in `autoflow/ui/app.py`; object mesh lookup for grouped names lives in `autoflow/ui/viewer.py`.

## Current Caveat
The CLI auto-segmentation public config fields are owned by `AutoFlowConfig`. The GUI auto-segmentation dialog is initialized from the segmentation config bundle. Keep this distinction explicit when documenting or refactoring.
