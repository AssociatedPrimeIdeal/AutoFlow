<div align="center">

<img src="assets/autoflow-logo.png" alt="AutoFlow penguin" width="120" height="120">

<h1>AutoFlow</h1>

<p><strong>Automated 4D flow MRI analysis across anatomical regions.</strong></p>

<p>
  <a href="https://github.com/AssociatedPrimeIdeal/AutoFlow/releases"><strong>Windows downloads</strong></a>
  &nbsp;·&nbsp;
  <a href="docs/en/user/changelog.md">Changelog</a>
  &nbsp;·&nbsp;
  <a href="#quick-start">Quick start</a>
</p>

</div>

<p align="center">
  <a href="autoflow-workflow.mp4?raw=true">
    <img src="autoflow-workflow.gif" alt="AutoFlow workflow: correction, segmentation, plane metrics, hemodynamics, and quality control" width="880">
  </a>
</p>

AutoFlow brings correction, segmentation, plane metrics, hemodynamics, and quality control into one workflow.
It supports cardiopulmonary, cerebrovascular, carotid, portal venous, and renal artery analysis.
For dual-VENC input, the [reconstruction workflow](docs/en/features/dual-venc.md) keeps loading, HV unwrapping and DV synthesis as separate steps.
PC-MRA [noise removal](docs/en/features/noise-removal.md) defaults to the **Strong** display coherence filter. Right-click **Correction → Noise Removal** to select Medium or turn coherence off. Display filtering preserves acquired data, measurement ROI and quantitative results.

## Quick start

### Windows

Pushes to `main` or `dev` automatically publish a numbered [GitHub Release](https://github.com/AssociatedPrimeIdeal/AutoFlow/releases) after Linux/Windows regression tests and Windows portable verification pass. Each release includes a changelog since the preceding version. Download the complete ZIP or all archive parts with the supplied joiner, extract to a short ASCII path, and double-click `AutoFlow.exe`. See [Windows download instructions](docs/en/user/windows-portable.md). PRs and manual CI runs provide the `AutoFlow-windows-x64` Actions artifact without publishing a release. The package includes the current application code and locked Linux segmentation models, without bundled documentation.

### Python

With Python 3.11:

```bash
git clone --recurse-submodules https://github.com/AssociatedPrimeIdeal/AutoFlow.git
cd AutoFlow
pip install -e ".[all]"
autoflow-gui
```


In the GUI, full QC and its HTML report are generated explicitly from **Review QC →
Generate QC / Refresh QC**. Existing H5 masks open in Labeler at their native
resolution. See [GUI workflow](docs/en/user/gui.md) and
[segmentation](docs/en/features/segmentation.md).
