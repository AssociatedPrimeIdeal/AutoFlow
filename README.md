<div align="center">

<img src="assets/autoflow-logo.png" alt="AutoFlow penguin" width="120" height="120">

<h1>AutoFlow</h1>

<p><strong>Automated 4D flow MRI analysis across anatomical regions.</strong></p>

<p>
  <a href="https://github.com/AssociatedPrimeIdeal/AutoFlow/releases"><strong>Download for Windows</strong></a>
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

## Quick start

### Windows

[Download the Windows app](https://github.com/AssociatedPrimeIdeal/AutoFlow/releases) from the latest release.

### Python

With Python 3.11:

```bash
git clone --recurse-submodules https://github.com/AssociatedPrimeIdeal/AutoFlow.git
cd AutoFlow
pip install -e ".[all]"
autoflow-gui
```
