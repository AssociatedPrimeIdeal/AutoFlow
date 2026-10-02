# Turbulent kinetic energy

## Status

Partial in GUI, CLI and Python: requires source TKE or supported complex-derived sigma.

## What it does

Retains or computes optional turbulent kinetic energy density. Complex-derived velocity dispersion permits `TKE = 0.5*rho*(sigma_x²+sigma_y²+sigma_z²)` in J/m3. The loader keeps sigma optional and does not eagerly materialize TKE. Requested TKE is reused for volume export and plane summaries.

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

[All TKE parameters](../user/parameters.md#tke) and [shared density](../user/parameters.md#fluid). An explicit metric density overrides the shared value. Rendering controls do not affect energy.

## Outputs

Optional TKE volume in workspace and `derived_metrics_pixelwise.npz`, TKE mean/peak/percentile plane values, and requested TKE videos. See [Outputs](../user/outputs.md).

## Limitations

Unavailable TKE is skipped cleanly by pipeline, GUI and videos. DICOM or normalized mag/flow inputs without TKE support still support geometry, WSS and streamlines. AutoFlow does not create fake TKE from velocity magnitude.

## Where to change code

`autoflow/algorithms/data.py` and `autoflow/algorithms/metrics.py`; optional execution/caching: `autoflow/core/pipeline.py`; export: `autoflow/processing.py`, `autoflow/rendering/videos.py`.

## Tests

`tests/test_smoke_phantoms.py` covers optional sigma/TKE loading, derived-family opt-in and unavailable-data handling.

```bash
conda activate autoflow311
PYVISTA_OFF_SCREEN=true python -m pytest tests/test_smoke_phantoms.py tests/test_pressure_gradient_phantom.py -q
```

## Common problems

Missing TKE: inspect source capabilities rather than enabling more output flags. Unsupported source TKE is expected to remain unavailable.
