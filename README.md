# AutoFlow

AutoFlow is a 4D flow MRI toolkit for H5 and DICOM data. It combines a desktop GUI, batch CLI, and Python API for segmentation review, vessel geometry, haemodynamic metrics, flow visualisation, and offline rendering.

The demo below was generated from a cold-start dual-VENC `DV_heart` case. The five views are synchronised subplots: GUI-style vessel groups and planes, WSS, pressure gradient, streamlines, and vortex swirling strength.

<video src="./docs/assets/video/dv-heart-flow-demo.mp4" controls width="100%"></video>

## Windows portable application

Download the full Windows x64 package from [GitHub Releases](https://github.com/AssociatedPrimeIdeal/AutoFlow/releases), extract the complete folder to a short ASCII/English path such as `D:\AutoFlow`, and double-click `AutoFlow.exe`. Open `docs/site/index.html` in a web browser to read the bundled searchable offline documentation. The Windows package provides the full GUI, DICOM, nnUNet 3D/4D, Labeler, PUDIP/GUST, and video dependencies. Models live separately under `models/`; Python and Conda are not required. See [Windows portable usage](docs/en/user/windows-portable.md) for GPU requirements, large-download joining, and diagnostics. Pushes to `dev` automatically build and publish the rolling `dev-windows` prerelease after verification; `v*` tags publish versioned releases.

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

The GUI starts in a compact **Workspace**: selecting a Browser row opens its details inline, with collapsible Appearance; **Measurements** and the bottom **Log** dock are available on demand. Action settings separate **Basic** and **Advanced** controls. **Review QC** and **Export** switch the full work area to their own pages. See the [GUI guide](docs/en/user/gui.md) for the layout.

Configuration is loaded from `configs/*.json`; pass `--config-dir` to use another directory. Shared rendering settings use `render_style.json` and `colorbar.json`; metric appearance uses each metric JSON `render` group. See [configuration parameters](docs/en/user/parameters.md). The `gui` extra includes the packaged nnUNet model metadata and checkpoint. DICOM conversion and learned phase unwrapping remain optional.

Default preprocessing merges labels into anatomical groups and filters components after morphology using `max(50 mm³, 10% of the largest component volume)`. Skeleton graph edges default to 26 face/edge/corner neighbors; choose face-only adjacency with `--graph-connectivity 6` or Generate Graph settings. Plane generation uses [one shared per-label measurement plan](docs/en/features/planes.md#default-label-measurements). ICA/VA use continuous supported main trunks at 33%/66%, without graph-level gates or regional presets. GUI Defaults restores `configs/plane_plans/default.json`; Load/Save supports custom plans. Unavailable positions appear as QC Errors with per-vessel reasons while valid planes are retained and the pipeline continues. ARCH, HA and SMA have no default planes.

In the GUI, left-click an action to run it and right-click (or press Shift+F10) to edit its settings. Method-specific parameters appear only when applicable. See [GUI action settings](docs/en/user/gui.md#action-settings).

Skeletons and graphs use a fixed 1-mm topology grid. Graph generation defaults to removing terminal arms with up to eight edges (`min_edge_points=9`); the shared GUI/CLI/API cutoff applies to all groups unless explicitly overridden. Regenerate Graph and Planes after changing it. Quantitative plane ROIs and WSS walls use native acquisition-grid segmentation for every source. Automatic segmentation uses nnUNet's native-grid prediction; other sources use their input labels. The model's 1-mm cache is retained for topology and display. Acquired velocity keeps its native samples. See the [native measurement validation and five custom-plan cases](docs/en/developer/native-measurement.md), and the [75-case interactive placement review](docs/en/developer/testset-plane-review.md). See [Segmentation](docs/en/features/segmentation.md). Centerlines and planes default to [radius-adaptive smoothing](docs/en/features/centerline-smoothing.md): a 2r half-window, 0.15r movement cap and lumen containment. The [24-input phantom suite](docs/en/features/analytic-phantoms.md) verifies each input against its own resolution-specific reference. VMR0028 remains external manual validation.

Default label planes follow continuous supported vessel trunks and local origin/confluence references, using the approximate [physician measurement instructions](docs/en/developer/skeleton-method-review-2026-10-29/annotation-guide.html). Small forks no longer reset an L1/L2 measurement segment. Ambiguous routes remain QC Errors. Existing saved plans can be replaced with GUI Defaults. See [Planes](docs/en/features/planes.md).

Each processed CLI/API case and GUI geometry/measurement task writes a single-case English `analysis_report.html`. Use **Open QC Report** for an in-app English HTML tab, or open it offline to inspect measurements and click green, orange, red or gray QC markers on actual segmentation/skeleton context. See [Analysis reports](docs/en/features/reports.md) for automatic output, manual export and status meanings.

Usage, input/output contracts, configuration details, GUI workflows, and feature references are documented under [docs/](docs/index.md):

- [Quickstart](docs/en/user/quickstart.md)
- [End-to-end workflow with GUI screenshots](docs/en/user/workflow.md)
- [Windows portable application](docs/en/user/windows-portable.md)
- [Windows builds and automatic releases](docs/en/developer/windows-packaging.md)
- [CLI](docs/en/user/cli.md)
- [GUI](docs/en/user/gui.md)
- [Analysis reports](docs/en/features/reports.md)
- [Viewer display settings](docs/en/features/viewer-display.md)
- [Python API](docs/en/user/python-api.md)
- [Feature directory](docs/en/features/index.md)
- [Worked DV case](docs/en/user/demo-case.md)
- [Kimimaro and retained-cycle decisions](docs/en/developer/kimimaro-cycle-diagnostics.md) (experimental comparison and QC provenance)
- [Aorta_003 Voronoi and geometry smoothing](docs/en/developer/aorta003-centerline-study.md) (single-case interactive comparison)
- [Lee and Kimimaro across all vessel groups](docs/en/developer/lee-kimimaro-all-groups.md) (two cases, actual all-CPU runs and short-spur comparison)
- [75-case Lee and Kimimaro manual review](docs/en/developer/skeleton-method-review.md) (five methods, independent comments and complete placement workflow)
- [Velocity and segmentation grid comparison](docs/en/developer/velocity-grid-comparison.md) (manual validation study)

The documentation is still under active construction; the pages linked above are the source of truth for current behavior.

AutoFlow-authored source code is licensed under the [MIT License](LICENSE). Third-party components retain their own licenses; the current PyMaxflow dependency is GPL-licensed, so combined distributions must also comply with its GPL requirements. See [licensing and third-party software](docs/en/user/licensing.md).
