# AutoFlow

AutoFlow is a 4D flow MRI toolkit for H5 and DICOM, with a desktop GUI, batch CLI and Python API. It supports segmentation and review, vessel geometry, cross-section metrics, WSS, relative pressure, PWV, vortex analysis and flow visualization. TKE remains optional and depends on the input.

![Demo](https://github.com/user-attachments/assets/e2c17a9e-6a47-4f85-ba0d-c35b622802b1)

## Install

```bash
git clone https://github.com/AssociatedPrimeIdeal/AutoFlow.git
cd AutoFlow
pip install -e ".[gui,test]"
```

For CLI-only use, install `pip install .`. DICOM directory input uses `git submodule update --init third_party/4DFlow_Dicom2H5` and `pip install -e ".[dicom]"`. H5-only loading does not require that extra. For learned phase unwrapping, initialize submodules and install `.[pu]`. The optional SpatioTemporal Labeler bridge uses `.[labeler]`; `.[all]` installs all optional dependency groups. Model and runtime details are in [Quickstart](docs/en/user/quickstart.md).

## Quickstart

```bash
autoflow-run ./data/demo_data.h5 --output-dir ./results/demo
autoflow-gui
```

Optional metrics and videos are requested explicitly:

```bash
autoflow-run case.h5 --output-dir ./results/case \
  --with pwv,wss,pg,vortex --video plane,wss,pg
```

```python
from autoflow import AutoFlowConfig, run_case

config = AutoFlowConfig.from_config_dir("./configs", requested_metrics=["wss", "pg"])
summary = run_case("case.h5", config=config)
```

The [worked DV H5 example](docs/en/user/demo-case.md) starts with the actual H5 keys and follows the GUI from loading through segmentation, geometry, hemodynamics, QC and video export. Each step includes the equivalent CLI command and expected output. [DICOM loading and conversion](docs/en/features/dicom-loading.md) covers both input routes.

The GUI starts with **Input & QC → Correction → Segmentation**. Correction provides Background Correction, Noise Removal, Unwrap Phase and Generate PC-MRA, plus Run All in that order (default `lap4D`, no mask). Background correction and unwrapping update working velocity; noise masking filters only the PC-MRA display. GUI Noise Removal enables a red orthogonal-slice mask for review; its separate 3D layer stays hidden by default and uses non-accumulating opacity. Existing downstream results remain until manually rerun. CLI/Python opt into the full group with `--correction` / `correction_all=True`. Geometry, requested metrics, PWV and exports follow segmentation. Normal correction and automatic segmentation can write reusable H5 caches; [Inputs](docs/en/user/inputs.md) explains controls and formats.

GUI task progress locks other actions until completion. Use × to request cancellation; animated activity dots, elapsed time and stage/frame counts remain visible until the task stops.

## Documentation

Detailed documentation is English-only and organized by function.

| Topic | Pages |
| --- | --- |
| Start and run | [Quickstart](docs/en/user/quickstart.md), [Worked example](docs/en/user/demo-case.md), [GUI](docs/en/user/gui.md), [CLI](docs/en/user/cli.md), [Python API](docs/en/user/python-api.md) |
| Data correction | [Background correction](docs/en/features/background-phase-correction.md), [Noise removal](docs/en/features/noise-removal.md), [Phase unwrapping](docs/en/features/phase-unwrapping.md) |
| Functional workflow | [Feature directory](docs/en/features/index.md) |
| DICOM input | [Dicom2H5 conversion and H5 loading](docs/en/features/dicom-loading.md) |
| Input and output contracts | [Inputs](docs/en/user/inputs.md), [Outputs](docs/en/user/outputs.md), [Troubleshooting](docs/en/user/troubleshooting.md) |
| Every parameter | [20 configuration modules](docs/en/user/parameters.md), [Structured schemas](docs/en/user/parameter-schemas.md), [CLI flags](docs/en/user/cli-parameters.md), [API fields](docs/en/user/api-parameters.md) |
| Scientific basis | [References by calculation, with implementation differences](docs/en/references/index.md) |
| Implementation | [Architecture](docs/en/developer/architecture.md), [Modules and owners](docs/en/developer/feature-to-code-map.md), [Configuration](docs/en/developer/config-system.md), [Change recipes](docs/en/developer/change-recipes.md) |
| Validation and speed | [Testing](docs/en/developer/testing.md), [Current performance](docs/en/developer/performance.md), [Segmentation measurements](docs/en/developer/segmentation-performance.md) |

Build and validate the documentation locally:

```bash
conda activate autoflow311
mkdocs build --strict
```

The retained automated suite is smoke and phantom coverage:

```bash
PYVISTA_OFF_SCREEN=true python -m pytest tests/test_smoke_phantoms.py tests/test_pressure_gradient_phantom.py -q
```

For cold-run measurement and controlled comparisons, see [Current performance](docs/en/developer/performance.md).

## License

See [LICENSE](LICENSE). Optional third-party projects retain their own licenses.
