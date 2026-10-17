# Windows portable application

## Status

The portable build provides the full GUI, DICOM conversion, 3D and 4D nnUNet inference, segmentation review in SpatioTemporal Labeler, traditional and learned phase unwrapping, metrics, and video export. PUDIP-Flow and GUST-Flow retain their Experimental status. Builds are verified before publication; the supplied `build-check.json` records whether CUDA execution was tested on the build host.

## Download and launch

1. Download the Windows x64 release from [GitHub Releases](https://github.com/AssociatedPrimeIdeal/AutoFlow/releases). `dev-windows` is the rolling development prerelease; version tags have their own releases.
2. Extract the complete ZIP to a writable, short **ASCII/English path**, for example `D:\AutoFlow`. Keep every extracted file and folder together. GUST's CUDA compiler cannot resolve its bundled headers through a Chinese/non-ASCII installation path.
3. Double-click `AutoFlow.exe`. Click **Docs** beside **File**, or press **F1**, to open the matching offline documentation in your default browser. Navigation, search, images, and workflow diagrams work without internet access.

Python, Conda, Git, and a system-wide CUDA Toolkit are not required on the destination PC. The application includes its own Python, CUDA runtime libraries, CuPy compilation components, Qt/VTK libraries, and FFmpeg encoder. Use Windows 10/11 x64. GPU features require an NVIDIA GPU and a driver compatible with CUDA 13. GUST requires CUDA; other algorithms retain their existing CPU support. Large 4D cases may exceed a laptop GPU's memory even when the runtime is installed correctly.

GitHub limits individual release assets to 2 GiB. Builds exceeding this limit are published as `.zip.part001`, `.zip.part002`, etc. Download **every** part, `release-manifest.json`, and `Join-Portable.ps1` into the same folder, then run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\Join-Portable.ps1
```

The joiner verifies each part and the reconstructed ZIP. Extract the ZIP normally, then double-click the executable. A complete ZIP is also produced by local builds.

## Folder layout

```text
AutoFlow-<version>-windows-x64/
  AutoFlow.exe
  portable.json
  source-manifest.json
  build-check.json
  configs/
  models/
    manifest.json
    nnunet3d/             # Current Dataset7010 static model
    nnunet4d/             # Current Dataset7020 temporal model
    nnunet3d-bundled/     # Original bundled static model
  app/                   # AutoFlow code used by this build
  _runtime/              # Application-local Python and dependencies
  source/                # Application and submodule sources, including build recipes
  docs/en/user/
  docs/en/features/
  docs/site/index.html    # Searchable offline HTML documentation
```

Models are independent of the application runtime. The default automatic segmentation backend remains `nnUNet4D`; it resolves `models/nnunet4d` instead of a Linux NAS path. Static `nnUNet` resolves `models/nnunet3d`. The original bundled model remains available by selecting `models/nnunet3d-bundled` explicitly.

The model bundle contains inference-only checkpoints with unchanged network tensors and the original inference configuration/mirroring settings. It excludes optimizer state, training-case maps, logs, and training data. Both best and final checkpoints are included where available. The named Dataset7010/7020 resampler is included in the runtime; CPU-only hosts use its corresponding CPU interpolation path.

## GUI configuration

Open H5 or import a DICOM directory from the GUI, then use its processing controls for correction, segmentation, geometry, metrics, and video export. Segmentation review opens the bundled SpatioTemporal Labeler. Windows users launch all workflows through `AutoFlow.exe`.

| Parameter | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| Configuration JSON files | directory | Extracted `configs/` | Portable folder; GUI launch argument `--config-dir` can override | Supplies GUI processing defaults | `autoflow/config.py` |
| `AUTOFLOW_MODELS_DIR` | path | Extracted `models/` | Environment before launch | Relocates the three default model profiles together | `autoflow/runtime_paths.py` |
| `auto_model` | path or `auto` | `auto` | GUI segmentation settings | Overrides a particular model folder | `autoflow/algorithms/segmentation/models.py` |

To keep models on another disk:

```powershell
$env:AUTOFLOW_MODELS_DIR = 'E:\AutoFlowModels'
.\AutoFlow.exe
```

Keep `nnunet3d`, `nnunet4d`, `nnunet3d-bundled`, and `manifest.json` directly below that directory. Explicit model selections still take precedence over automatic profiles. A missing default model is an error; the application does not silently replace the temporal model with a static model.

GUI logs are written below `%LOCALAPPDATA%\AutoFlow\logs`; CuPy and Matplotlib caches live below `%LOCALAPPDATA%\AutoFlow\cache`. Configuration files can be edited in the portable folder; model and source files are not changed by launching the application.

## Common problems

| Symptom | Action |
| --- | --- |
| Double-click reports missing application files | Extract the complete archive; do not run the executable from inside a ZIP or move it alone |
| The 4D model is missing | Restore the complete `models/` folder or set `AUTOFLOW_MODELS_DIR` |
| CUDA is unavailable or GUST cannot start | Update the NVIDIA driver; see the developer guide for GPU diagnostics |
| GUST reports `cannot open source file "cupy/complex.cuh"` | Move the entire extracted folder to an ASCII/English path such as `D:\AutoFlow`; Chinese characters in the installation path are unsupported by the Windows CUDA compiler |
| CUDA out of memory | Select CPU for supported operations, free GPU memory, or use a GPU with sufficient memory; GUST requires CUDA |
| GUI cannot create an OpenGL context | Update the graphics driver and check the GUI locally; remote desktop graphics support depends on the session |
| Python starts but a DLL cannot load | Keep `_runtime/` intact and inspect the startup log and diagnostics |
| A release has several `.partNNN` files | Download all parts and run the supplied joiner before extracting |

See [Windows build and release maintenance](../developer/windows-packaging.md) for development and CI details.
