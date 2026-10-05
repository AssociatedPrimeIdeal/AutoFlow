# AutoFlow

AutoFlow is a 4D flow MRI toolkit for H5 and DICOM data. It combines a desktop GUI, batch CLI, and Python API for segmentation review, vessel geometry, haemodynamic metrics, flow visualisation, and offline rendering.

The demo below was generated from a cold-start dual-VENC `DV_heart` case. The five views are synchronised subplots: GUI-style vessel groups and planes, WSS, pressure gradient, streamlines, and vortex swirling strength.

<video src="./docs/assets/video/dv-heart-flow-demo.mp4" controls width="100%"></video>

## Install

AutoFlow requires Python 3.9 or newer. An editable install is convenient during development:

```bash
git clone https://github.com/AssociatedPrimeIdeal/AutoFlow.git
cd AutoFlow
python -m pip install -e ".[gui,test]"
```

Choose extras for the workflows you need:

| Extra | Adds |
| --- | --- |
| *(none)* | H5 loading, CLI, and Python API |
| `gui` | Qt GUI and bundled nnUNet inference runtime |
| `dicom` | DICOM-to-H5 conversion through Dicom2H5 |
| `pu` | Optional PUDIP-Flow and GUST-Flow phase-unwrapping backends |
| `all` | All optional groups |

Examples:

```bash
python -m pip install -e .                 # CLI/API, H5 only
python -m pip install -e ".[gui,dicom]"   # GUI, auto-segmentation, and DICOM
```

Configuration is loaded from `configs/*.json`; pass `--config-dir` to use another directory. Shared rendering settings use `render_style.json` and `colorbar.json`; metric appearance uses each metric JSON `render` group. See [configuration parameters](docs/en/user/parameters.md). The `gui` extra includes the packaged nnUNet model metadata and checkpoint. DICOM conversion and learned phase unwrapping remain optional.

Usage, input/output contracts, configuration details, GUI workflows, and feature references are documented under [docs/](docs/index.md):

- [Quickstart](docs/en/user/quickstart.md)
- [CLI](docs/en/user/cli.md)
- [GUI](docs/en/user/gui.md)
- [Python API](docs/en/user/python-api.md)
- [Feature directory](docs/en/features/index.md)
- [Worked DV case](docs/en/user/demo-case.md)

The documentation is still under active construction; the pages linked above are the source of truth for current behavior.

See [LICENSE](LICENSE) for licensing information.
