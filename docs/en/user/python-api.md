# Python API

## Status
The public Python API is supported through `autoflow/__init__.py` and `autoflow/api.py`.

Public entry points:

- `AutoFlowConfig`
- `build_workspace()`
- `run_case()`
- `run_batch()`
- `launch_gui()`

## Quick Use

### Run one case

```python
from autoflow import AutoFlowConfig, run_case

config = AutoFlowConfig(output_dir="./results/demo")
summary = run_case("./data/demo_data.h5", config=config)
```

Default plane generation uses `plane_mode="by_label"` with shared observed connections and per-label positions. Select a shared file with `AutoFlowConfig(plane_config="configs/plane_plans/brain.json")`. Alternatively supply a complete inline mapping such as `plane_label_rules={"MPA": {"plane_mode": "landmark", "plane_count": 1, "placements": [{"anchor": "label_connections", "position": "fractions", "fractions": [0.5]}]}}`. It replaces the file and enables only MPA. The example measures each observed connected MPA arm; add an explicit `fallback` placement to allow unconnected components. Unsupported sections are omitted with QC reasons. `plane_count` matches primary declared positions or is zero; topology and fallback determine actual counts. Null plane_label_rules inherits the selected file/module. See [Label plane rules](parameter-schemas.md#label-plane-rules).

Default preprocessing merges group labels, leaves closing off, applies morphology, then retains components with volume at least `max(50 mm³, 0.10 × largest component volume)`. Temporal group frames use the same threshold without skeleton morphology. `cc_filter_mode="largest"` explicitly restores single-component selection; `remove_small_cc=False` disables the filter.

Skeleton graph edges use `graph_connectivity=26` by default. Set `AutoFlowConfig(graph_connectivity=6)` for face-only neighbors. This is independent of preprocessing component filtering; diagonal skeletons can fragment with 6. Regenerate graph and planes after changing it. For neck-only scans, select `plane_config="configs/plane_plans/neck.json"`; ICA/VA use observed connected offsets, or longest L1 at 33%/66% without connections.

After Generate Graph, `workspace.graph_hierarchy` exposes anatomical groups and connected `subgroups`, each graded independently, plus all parents/children, inlet/outlet ids, directed loops and label-relative depths. Path records carry `subgroup_id`, zero-based `subgroup_index`, and nullable `subgroup_level`; `group_level` remains an equal-valued compatibility alias. Native path records are also attached to `workspace.path_info[i]["hierarchy"]`. Joint anterior/posterior cerebral analysis appears under `graph_hierarchy["joint_systems"]` with its own connected subgroups, separate ids and geometry; it leaves ordinary paths/planes unchanged. Null grades represent unresolved direction/upstream topology or closed unrooted loops. See [Graph and paths](../features/graph-paths.md) for schema and usage.

Generated planes also use `fov_boundary_filter=True` and `fov_boundary_max_fraction=0.25`: contact length on the image box faces divided by the outer ROI perimeter, measured on temporal-union geometry. Set `fov_boundary_filter=False` to keep these candidates. `summary["quality_report"]["exclusions"]` contains complete plane decisions and label/phase preprocessing provenance; direct workspaces retain `segmentation_preprocess_qc`. See [QC](../features/quality-control.md) for loss-fraction denominators and GUI details.

Generated path ids follow group order, connected subgroup order, then ascending depth; unresolved paths follow their subgroup's ranked paths. Regenerate graphs and planes to apply this numbering to an older workspace. Plane sampling automatically carries missing ROI geometry forward from the latest valid earlier phase, with no backward/cyclic fill. Inspect `roi_source_frame_t` and `roi_carried_forward_t` in each metric. Velocity stays current-phase; raw volumes and derived-field validity are unchanged. Explicit empty ROI edits stop propagation.

The whole Correction group is opt-in:

```python
config = AutoFlowConfig(correction_all=True, autoseg=True)
summary = run_case("case.h5", config=config)
```

It runs background correction, PC-MRA noise masking, unwrapping and PC-MRA generation before segmentation; default unwrap is `lap4D` with no mask. Separate fields are `background_phase_correction`, `noise_removal`, `phase_unwrap_method` and `generate_pcmra`. Noise masking does not modify magnitude/velocity or segmentation features. See [Noise removal](../features/noise-removal.md) for its thresholds and reusable output.

Phase unwrapping alone is opt-in:

```python
config = AutoFlowConfig(
    output_dir="./results/demo",
    phase_unwrap_method="lap4D",
    phase_unwrap_device="auto",
)
```

Use `phase_unwrap_method="pudip"` or `"gust"` for the optional learned
backends after initializing their git submodules and installing
`pip install ".[pu]"`. Put their training options in
`configs/phase_unwrapping.json` under `backend_params`, or pass the same mapping as
`AutoFlowConfig(phase_unwrap_backend_params={...})`.

The returned summary includes `phase_unwrap` statistics and, when enabled, a `phase_unwrap_file` NPZ. Dual-VENC inputs report a skipped phase-unwrapping step.

### Run a batch

```python
from autoflow import AutoFlowConfig, run_batch

config = AutoFlowConfig(
    inputs=["./data/demo_data.h5"],
    output_dir="./results/demo",
    requested_metrics=["pwv", "wss"],
    requested_videos=["plane", "wss"],
)
results, last_case_out = run_batch(config)
```

`run_batch()` returns per-case `ok`, `error`, or `partial` statuses. `partial`
means calculations completed but a requested available video failed; inspect
`summary.video_errors`. `run_case()` returns that summary directly, while
fatal processing errors still raise. Missing optional TKE is skipped.
`total_time_sec` includes final outputs and videos; `compute_time_sec` preserves
the former compute-stage endpoint. See [Outputs](outputs.md#case-timing-and-partial-video-output).

Requested background correction runs during the first input decode.

### Build a workspace from config defaults

```python
from autoflow import AutoFlowConfig, build_workspace

config = AutoFlowConfig.from_config_dir("./configs")
workspace = build_workspace(config)
```

### Enable PWV through config files

```python
from autoflow import AutoFlowConfig, run_case

config = AutoFlowConfig.from_config_dir("./configs")
summary = run_case("case.h5", config=config)
```

PWV group definitions, timing method selection, and plotting defaults still come from `configs/pwv.json`, while `AutoFlowConfig.requested_metrics=["pwv"]` decides whether the batch run executes PWV.

For the other derived metrics, `AutoFlowConfig.from_config_dir()` now reads:

- `configs/fluid.json` for shared fluid properties such as `rho` and `viscosity`
- `configs/wss.json` for WSS compute parameters
- `configs/tke.json` for TKE density
- `configs/pressure_gradient.json` for pressure-gradient estimation, relative-pressure reconstruction, and centerline-pressure outputs
- `configs/vortex.json` for vorticity, Q-criterion, and swirling-strength smoothing and support erosion
- `configs/planes.json` for GUI plane styling, plus `configs/video_exporting.json` for plane-video styling
- `configs/render_style.json` for shared materials, anti-aliasing and text; `configs/colorbar.json` for shared colourbar layout
- Metric JSON `render` groups for colormap, opacity, scalar range and colourbar visibility; TKE also has `render.volume` controls
- `configs/pathlines.json` for GUI pathline launch, color, tube, and temporal-cache defaults
- `configs/video_exporting.json` for shared video controls such as `window_size`, camera behavior, and rotation

WSS now defaults to zero wall velocity (`no_slip_condition=True`) and computes signed shear vectors in Pa with analytic wall derivatives. `cal_wss_from_surf()` accepts an optional `support_grid` whose `wss_lumen` cell scalars identify valid sampling segments. Invalid surface values are NaN with `wss_valid=0`. Pressure support always requires a valid spatial and temporal stencil; `support_erosion_iters=0` only disables the additional erosion margin. See [WSS and pressure](../features/pressure.md) for units, reconstruction methods, and migration of old results.

### Launch the GUI from Python

```python
from autoflow import launch_gui

launch_gui(config_dir="./configs")
```

## Parameter reference

Every supported field is explained in [Python API parameters](api-parameters.md). For shipped JSON values, see [Configuration parameters](parameters.md).

## Config Directory Support

`AutoFlowConfig.from_config_dir()` loads the per-module JSON files from a config directory and converts them to public API defaults.

Main code:

- `autoflow/config.py`
- `autoflow/api.py:AutoFlowConfig.from_config_dir()`

## Current API Boundaries

- batch processing flows through `run_case()` and `run_batch()`
- interactive GUI editing is not exposed as a stable batch API
- GUI pathlines are interactive behavior, not a public batch-processing API
- offline videos are supported through `requested_videos` in `run_case()` and `run_batch()`
- PWV is available through the config bundle loaded by `AutoFlowConfig.from_config_dir()` and `build_workspace()`

## Function contracts

| Function | Parameters | Result / behaviour |
| --- | --- | --- |
| `build_workspace(config=None)` | Optional AutoFlowConfig | Creates configured Workspace without loading a case |
| `run_case(input_path, output_dir=None, config=None, workspace=None)` | File/InputCase; optional output path/config/template workspace | Summary dictionary; creates per-case outputs. The supplied workspace is deep-copied and is not the loaded/result workspace |
| `run_batch(config)` | AutoFlowConfig including inputs | (summaries, last_case_output_dir); converts DICOM directories, collects H5 groups and processes them |
| `launch_gui(config_dir=None)` | Optional module config directory | Starts the Qt GUI; see GUI workflow |

## Optional DICOM conversion

DICOM directories always use Dicom2H5 before H5 loading. Set `dicom_h5_dir` to choose the conversion destination directory. `run_batch()` processes all converted H5 groups; `run_case()` requires one group or an explicit H5 `InputCase`. Use `convert_dicom_input()` for conversion without analysis. Installation, signatures, output contracts and failure behavior are documented in [DICOM loading and conversion](../features/dicom-loading.md).

`phase_unwrap_mask="auto"` resolves to `none` for gc3D/lap4D/NPRS and `pcmra_std` for PUDIP/GUST. All methods accept an available `segmask`; learned methods additionally accept `pcmra_mean`. A masked rerun updates working velocity while retaining existing downstream artifacts. Use `PipelineEngine.run_step` on a workspace for interactive reruns; rerun downstream actions explicitly. Noise controls are `noise_removal_method`, `noise_magnitude_fraction` (default `0.05`, fraction of maximum temporal-mean magnitude) and `noise_velocity_std_max` (default `0.80`, upper temporal-speed SD fraction of maximum SD, not VENC). These AutoFlow defaults can be overridden manually; lower magnitude or higher SD fractions retain more voxels. See [Noise removal](../features/noise-removal.md) for equations, reports and compatibility notes, and [Scientific references](../references/index.md) for the literature basis.

## Low-level plane sampling

`autoflow.algorithms.augment_plane_metrics_with_derived()` accepts optional `use_multithread=False`, `max_workers=None` and `progress_callback=None` keywords. Automatic process sampling starts at 1920 plane-phases, uses up to four workers, and preserves per-frame contour edits and output order. See [Metrics](../features/metrics.md) for the parameter table. The standard pipeline selects this through its existing parallel-computation setting.
