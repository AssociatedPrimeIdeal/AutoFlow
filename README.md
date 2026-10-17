# AutoFlow

AutoFlow is a 4D flow MRI toolkit for H5 and DICOM data. It combines a desktop GUI, batch CLI, and Python API for segmentation review, vessel geometry, haemodynamic metrics, flow visualisation, and offline rendering.

The demo below was generated from a cold-start dual-VENC `DV_heart` case. The five views are synchronised subplots: GUI-style vessel groups and planes, WSS, pressure gradient, streamlines, and vortex swirling strength.

<video src="./docs/assets/video/dv-heart-flow-demo.mp4" controls width="100%"></video>

## Windows portable application

Download the full Windows x64 package from [GitHub Releases](https://github.com/AssociatedPrimeIdeal/AutoFlow/releases), extract the complete folder to a short ASCII/English path such as `D:\AutoFlow`, and double-click `AutoFlow.exe`. Click **Docs** beside **File**, or press **F1**, to open the bundled searchable offline documentation. The Windows package provides the full GUI, DICOM, nnUNet 3D/4D, Labeler, PUDIP/GUST, and video dependencies. Models live separately under `models/`; Python and Conda are not required. See [Windows portable usage](docs/en/user/windows-portable.md) for GPU requirements, large-download joining, and diagnostics. Pushes to `dev` automatically build and publish the rolling `dev-windows` prerelease after verification; `v*` tags publish versioned releases.

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

Default preprocessing merges labels into anatomical groups and filters components after morphology using `max(50 mm³, 10% of the largest component volume)`. Skeleton graph edges default to 26 face/edge/corner neighbors; choose face-only adjacency with `--graph-connectivity 6` or Generate Graph settings. Plane generation uses [shared observed connections with per-label positions](docs/en/features/planes.md#default-label-measurements), including all real same-label arms and explicit fallback plans for unconnected components. ARCH, HA and SMA have no default planes. GUI Edit provides positions, fallback and conditional-level controls. Regional presets include Neck. GUI/CLI share `configs/plane_plans/` files; existing workspaces keep saved rules until a preset/file is loaded. Unsupported positions are skipped with QC reasons.

In the GUI, left-click an action to run it and right-click (or press Shift+F10) to edit its settings. Method-specific parameters appear only when applicable. See [GUI action settings](docs/en/user/gui.md#action-settings).

Usage, input/output contracts, configuration details, GUI workflows, and feature references are documented under [docs/](docs/index.md):

- [Quickstart](docs/en/user/quickstart.md)
- [Windows portable application](docs/en/user/windows-portable.md)
- [Windows builds and automatic releases](docs/en/developer/windows-packaging.md)
- [CLI](docs/en/user/cli.md)
- [GUI](docs/en/user/gui.md)
- [Viewer display settings](docs/en/features/viewer-display.md)
- [Python API](docs/en/user/python-api.md)
- [Feature directory](docs/en/features/index.md)
- [Worked DV case](docs/en/user/demo-case.md)

The documentation is still under active construction; the pages linked above are the source of truth for current behavior.

See [LICENSE](LICENSE) for licensing information.
