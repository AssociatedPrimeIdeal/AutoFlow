# Noise Removal

## Status

Supported. GUI creates a PC-MRA display mask; CLI and Python API create the same mask as a reusable NPZ. Velocity denoising and calibrated uncertainty estimation are not implemented by this feature.

## What it does

Builds one static spatial mask from time-averaged magnitude and, optionally, the temporal standard deviation of speed. Let `M = mean_t(magnitude)`, `V = sqrt(vx**2 + vy**2 + vz**2)` and `S = std_t(V)` (population SD, `ddof=0`). Retain voxels with finite magnitude/velocity, `M > magnitude_fraction * max(M)` and, when temporal screening is enabled, `S <= velocity_std_max * max(S)`. The magnitude maximum uses positive finite temporal-mean magnitude; the SD maximum uses all spatial voxels with finite velocity in every frame, including low-magnitude background. Neither reference is P99 or VENC.

Mask polarity is explicit: `True` means retain for PC-MRA, and `False` means exclude. The red noise overlay shows only `~pcmra_render_mask`. Excluded voxels include air, zero padding and other low-signal background; red is not a diagnosis that every highlighted voxel contains measured velocity noise.

Default fractions are **0.05 (5%) for magnitude** and **0.80 (80%) for temporal SD**. These are AutoFlow settings chosen for less restrictive screening, not the papers' numerical recommendations. The magnitude/temporal-SD masking rationale and the original empirical ranges are recorded in [Scientific references](../references/index.md#noise-masking-and-pc-mra). AutoFlow uses temporal-mean magnitude to create a single spatial mask, and does not remove static tissue or modify measured velocity; this is not a complete reproduction of the papers' preprocessing chain.

## When to use it

Use when low-signal background or unstable background velocity obscures the 3D PC-MRA backdrop. Review the retained region before interpreting missing small vessels. This display region is neither a vessel segmentation nor a probability of velocity accuracy.

## Quick use

GUI: **Correction > Noise Removal**. Choose **Magnitude + temporal speed SD** (default) or **Magnitude only**. In the Noise Removal parameter group, **Magnitude fraction of maximum** defaults to `0.050` and **Temporal SD fraction of maximum** to `0.800`. The panel contains controls only, without an explanatory paragraph underneath. Lower the magnitude fraction or raise the SD fraction to retain more voxels. After editing, rerun **Noise Removal** or **Run All**; edits do not automatically recompute the mask. **Run All** runs Background Correction, Noise Removal, Unwrap Phase and Generate PC-MRA in that order. Completing Noise Removal, alone or in Run All, enables the red 2-D overlay for immediate review; the 3-D **Noise Region** stays hidden. The **Noise mask** checkbox shows whether the overlay is enabled. Unchecking it displays 0% opacity; moving its slider above zero enables it, and setting zero disables it. This layer is above segmentation so even opaque segmentation cannot hide rejected voxels. It also works on the selected plane's U/V/N views. The checkbox and slider are disabled until a valid mask exists. **Reset PC-MRA Noise Mask** removes the display filter, not the selected parameter values.

CLI:

```bash
autoflow-run case.h5 --noise-removal --noise-removal-method magnitude_temporal
# Manual override, keeping more signal:
autoflow-run case.h5 --noise-removal --noise-magnitude-fraction 0.03 --noise-velocity-std-max 0.90
# Whole Correction group, before automatic segmentation:
autoflow-run case.h5 --correction --autoseg --no-cache-write
```

Python API:

```python
from autoflow import AutoFlowConfig, run_case
summary = run_case("case.h5", config=AutoFlowConfig(noise_removal=True))
# Manual override:
config = AutoFlowConfig(noise_removal=True, noise_magnitude_fraction=0.03,
                       noise_velocity_std_max=0.90)
# Entire group: AutoFlowConfig(correction_all=True, autoseg=True)
```

For an existing workspace, call `PipelineEngine().run_step(ws, StepId.REMOVE_NOISE, print)` after loading. The standalone `pcmra_render_mask(mag, flow, venc, ...)` function returns `(mask, report)` without modifying its input arrays.

## Inputs

Magnitude: `XYZT` or `XYZ` (also accepts one shared time frame). Velocity: `XYZT3` in cm/s. The `venc` argument remains in the standalone function signature for call compatibility but does not normalize thresholds. No segmentation is required.

## Parameters

| Parameter / flag | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `enabled` / `--noise-removal` / API `noise_removal` | bool | false in batch/API | `configs/noise_removal.json`, CLI/API | Run the display-mask stage; GUI action runs explicitly | `autoflow/processing.py`, `autoflow/ui/app.py` |
| `method` / `--noise-removal-method` / API `noise_removal_method` | string | `magnitude_temporal` | JSON, CLI/API, GUI | Magnitude plus temporal SD, or `magnitude` alone | `autoflow/algorithms/noise_removal.py` |
| `magnitude_fraction` / `--noise-magnitude-fraction` / API `noise_magnitude_fraction` | float, 0–1 | 0.05 | JSON, CLI/API, GUI | Fraction of maximum temporal-mean magnitude; lower retains more; zero retains all positive finite magnitude | `autoflow/algorithms/noise_removal.py` |
| `velocity_std_max` / `--noise-velocity-std-max` / API `noise_velocity_std_max` | float, 0–1 | 0.80 | JSON, CLI/API, GUI | Upper temporal speed SD fraction of maximum SD; higher retains more; zero disables temporal screening | `autoflow/algorithms/noise_removal.py` |
| ortho `Noise mask` | bool | off initially; on after GUI Noise Removal | GUI ortho viewer | Show rejected voxels in red above segmentation, with transparent retained voxels | `autoflow/ui/app.py`, `autoflow/ui/ortho_viewer.py`, `autoflow/ui/slice_view.py` |
| noise overlay opacity | 0–100% slider | 35% when first enabled; 0% when off | GUI ortho viewer | Adjust only the red 2-D layer; positive opacity enables it and zero disables it | `autoflow/ui/ortho_viewer.py` |
| 3-D `Noise Region` visibility / opacity | bool / float | hidden / 15% | Browser | Binary red volume with maximum-intensity projection; opacity does not accumulate along the ray | `autoflow/core/pipeline.py`, `autoflow/ui/viewer.py` |

## Outputs

GUI: `Workspace.pcmra_render_mask` is a boolean `XYZ` retention region, saved/restored with the workspace. It filters the 3D PC-MRA layer and its display range. PC-MRA uses nearest-neighbor voxel-center samples; it does not average retained signal into rejected voxels. The ortho viewer can show the excluded part (`~pcmra_render_mask`) as a red overlay, but the underlying scalar slices remain unchanged. Raw magnitude, working velocity, nnUNet features, metrics and trajectories do not use it.

CLI/API: `pcmra_noise_mask.npz` contains `mask`, `resolution` and `origin`. `summary.json` contains `noise_removal` (parameters, actual magnitude threshold, `magnitude_reference_max`, `magnitude_statistic`, `temporal_std_threshold`, `temporal_std_reference_max`, `temporal_std_statistic`, temporal-screening status and retained fraction) and `noise_removal_file`. The SD threshold/reference are `null` when temporal screening is bypassed. No source-H5 noise-mask dataset is written.

## Limitations

One static mask is used for every cardiac phase. Fewer than two frames bypass temporal screening. Constant speed has zero SD and is retained even when all voxels are constant; direction changes at constant speed are not detected by this statistic. The method is heuristic; strong pulsatility or wrapped-speed jumps can exclude real flow. Choose `magnitude`, increase the SD fraction, or set it to zero when that happens. A noise mask computed before unwrapping remains unchanged until Noise Removal is explicitly rerun. Maximum-based reference values are sensitive to outliers. The 2007 empirical ranges come from thoracic-aorta studies; portal-vein and other low-flow data still require slice review and manual adjustment. No lower-SD static-tissue cutoff is applied.

Existing parameter names are retained, but their meanings change from the earlier Triangle/P99 and component-SD/VENC implementation. Saved custom fractions are not overwritten by new defaults; review them when reopening a workspace. In particular, a saved magnitude fraction of zero now means positive signal only, not automatic Triangle thresholding. Stored masks do not change until rerun. To use the current AutoFlow defaults in an older workspace, enter `0.05` and `0.80` and rerun Noise Removal.

## Where to change code

Algorithm: `autoflow/algorithms/noise_removal.py`. Config/state: `autoflow/case_types.py`, `autoflow/config.py`, `autoflow/core/models.py`. Execution: `autoflow/core/pipeline.py`, `autoflow/processing.py`, `autoflow/api.py`, `autoflow/cli.py`. GUI controls/rendering: `autoflow/ui/app.py`, `autoflow/ui/ortho_viewer.py`, `autoflow/ui/slice_view.py`, `autoflow/ui/viewer.py`.

## Tests

Keep automated coverage in `tests/test_smoke_phantoms.py` and `tests/test_pressure_gradient_phantom.py`. Smoke checks defaults across configuration and entry points, manual parameter persistence, maximum-based references instead of P99/VENC, input immutability, low-magnitude/unstable-background exclusion, steady-flow retention, constant/zero/nonfinite/single-frame signal cases and Correction order. Phantom checks verify red pixels above opaque segmentation in standard and selected-plane views, slider/checkbox synchronization, automatic review after Correction, refresh/toggle/reset behavior, and zero PC-MRA at excluded voxel centers. A rendered shallow/deep mask check verifies that both smart and fixed-point volume mappers keep the same red opacity instead of saturating with thickness. Manually inspect PC-MRA with and without the mask.

```bash
conda activate autoflow311
PYVISTA_OFF_SCREEN=true python -m pytest tests/test_smoke_phantoms.py tests/test_pressure_gradient_phantom.py -q
```

## Common problems

An empty display region can result from absent positive magnitude or strict thresholds. Reset the mask and inspect the original PC-MRA, then use magnitude-only screening or a less restrictive cutoff. This filter is not used as `segmask` or as a learned unwrapping weight source.

Noise Removal also adds a hidden red **Noise Region** binary volume under **Global → Noise** in the Browser. Only excluded samples (`~pcmra_render_mask`) have nonzero opacity; retained samples and the padded border are transparent. Maximum-intensity projection applies the selected opacity once per projected pixel, rather than accumulating hundreds of translucent red points into an opaque block. It can be toggled or given a different opacity and is removed on reset. The ortho viewer's **Noise mask** control is independent of this 3-D layer. PC-MRA calculation itself happens only in the fourth Correction action, **Generate PC-MRA**; loading and noise-mask calculation do not compute it.

Low-signal background can surround retained vessels in 3-D without sharing any voxels with them. Its projection can therefore tint most of the image, but no more than the selected opacity. Use the red slice overlay to judge spatial overlap rather than interpreting an enclosing background region as a reversed mask. If actual vessels are highlighted, review the magnitude threshold and temporal cutoff; the heuristic can reject real flow. Conversely, retained static tissue or coherent tissue motion can still produce peripheral PC-MRA signal: this feature is not a complete velocity denoiser or a vessel-only filter.
