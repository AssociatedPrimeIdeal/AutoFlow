# Turbulent kinetic energy

## Status

Partial in GUI, CLI and Python: requires source TKE or supported complex-derived sigma.

## What it does

Retains or computes optional turbulent kinetic energy density. Complex-derived velocity dispersion permits `TKE = 0.5*rho*(sigma_x²+sigma_y²+sigma_z²)` in J/m3. The loader keeps sigma optional and does not eagerly materialize TKE. Requested TKE is reused for volume export and plane summaries.

See [Scientific references: TKE](../references/index.md#turbulent-kinetic-energy) for intravoxel dispersion measurement; it is distinct from the cardiac-cycle temporal SD used by noise masking.

## When to use it

Use only when the acquisition or normalized input provides valid turbulence information. Mean velocity magnitude alone does not establish dispersion.

## Quick use

GUI: use **WSS / TKE / Pressure / Vortex** when TKE support is available; inspect TKE in the content selector.

CLI:

```bash
autoflow-run case.h5 --output-dir results/case --with tke
```

Python:

```python
from autoflow import AutoFlowConfig, run_case
config = AutoFlowConfig.from_config_dir('./configs', requested_metrics=['tke'])
summary = run_case('case.h5', config=config)
```

## Inputs

Optional `tke_array` or supported complex-derived sigma, segmentation for vessel/plane summaries, density in kg/m3 and spatial metadata.

## Parameters

GUI and videos read shared materials from `configs/render_style.json`, shared colourbar layout from `configs/colorbar.json`, and layer styles from the metric JSON `render` groups. See [Rendering configuration layout](../developer/config-system.md#rendering-configuration-layout) and the complete [parameter tables](../user/parameters.md). Restart the GUI or start a new run after editing the selected JSON files.


[All TKE parameters](../user/parameters.md#tke) and [shared density](../user/parameters.md#fluid). An explicit metric density overrides the shared value. Rendering controls do not affect energy. GUI and movies share `inferno`, a composite volume-opacity ramp and current-phase segmentation support. TKE is placed at original voxel centres; zero-energy/background voxels are completely transparent. A linear display interpolation and increasing opacity reveal interior energy rather than colouring only the outer wall. The scene opacity scales the ramp and defaults to 1.0. Explicit colour limits are honoured and fixed across phases. There is no extra white surface shell in TKE movies.

## Outputs

Optional TKE volume in workspace and `derived_metrics_pixelwise.npz`, TKE mean/peak/percentile plane values, and requested TKE videos. See [Outputs](../user/outputs.md).

## Limitations

Unavailable TKE is skipped cleanly by pipeline, GUI and videos. Older mesh-only workspaces display their stored peak field because time-resolved energy is unavailable. DICOM or normalized mag/flow inputs without TKE support still support geometry, WSS and streamlines. AutoFlow does not create fake TKE from velocity magnitude.

## Where to change code

`autoflow/algorithms/data/normalization.py`, `autoflow/algorithms/data/h5_loader.py` and `autoflow/algorithms/metrics/tke.py`; optional execution/caching: `autoflow/core/pipeline.py`; export: `autoflow/processing.py`, `autoflow/rendering/videos.py`; shared display geometry: `autoflow/rendering/datasets.py`; shared colour/opacity defaults: `autoflow/config.py`; VTK styling: `autoflow/rendering/style.py`.

## Tests

`tests/test_smoke_phantoms.py` covers optional loading, unavailable data, analytic sigma-to-energy units, and exclusion of nonfinite background. Density must be finite and positive and spatial dimensions must match segmentation. Invalid values outside segmentation become zero without changing valid energy.

```bash
conda activate autoflow311
PYVISTA_OFF_SCREEN=true python -m pytest tests/test_smoke_phantoms.py tests/test_pressure_gradient_phantom.py -q
```

## Common problems

Missing TKE: inspect source capabilities rather than enabling more output flags. Unsupported source TKE is expected to remain unavailable.
