# Windows portable builds and releases

## Architecture

The full Windows distribution uses one small x64 C# launcher with an application-local CPython 3.11 runtime. `AutoFlow.exe` launches the complete GUI through `pythonw.exe` without a console window. The launcher requires the .NET Framework supplied with Windows 10/11. It forwards GUI arguments with Windows-compatible quoting and returns the child exit status. The distribution exposes only this GUI entry point; the internal Python interpreter remains necessary for segmentation, Labeler, rendering, multiprocessing, and build diagnostics.

The top-level Docs action and F1 shortcut open `docs/site/index.html` in the default browser. The offline site includes local search and Mermaid assets; no documentation generator or embedded browser engine is added to the application runtime. `portable.json` records `documentation_manifest_sha256`, and the documentation manifest hashes every generated file.

The Python runtime is an ordinary interpreter. Existing `sys.executable -m ...` video and Labeler launches, multiprocessing spawn workers, nnUNet's source/class discovery, `inspect.getsource`, and CuPy's NVRTC compilation retain their normal behavior. The portable `_pth` file restricts imports to the bundled runtime and application. `sitecustomize.py` initializes paths, caches, DLL directories, and the child interpreter. Regular pip installations retain their existing configuration and model defaults.

Bundled Microsoft C++ runtime directories are registered before Torch or Qt imports, so application processes do not depend on a preinstalled Visual C++ redistributable. Deep dependency license paths are flattened with an original-path index and corresponding metadata/RECORD updates, retaining all notices while avoiding Windows extraction path limits.

Windows NVRTC header lookup does not support a non-ASCII installation path. The distribution therefore requires an ASCII extraction path for GUST; CUDA execution is validated on that supported path. H5 input locations and model relocation are separate from the runtime installation path.

The package includes every runtime dependency and the current sources of all four optional submodules. It does not install project extras from their remote Git URLs, which could select older code than the working checkout. A local build includes uncommitted main-project and submodule changes. `portable.json` records commits, dirty state, and installed dependency versions; `source-manifest.json` hashes the included source snapshot.

## Local build

Use a Windows x64 CPython 3.11 build interpreter. The script creates an isolated build environment; it does not require packages to be installed in the user's ordinary Python environment. Allow sufficient disk space for the environment, staged runtime, model bundle, and compressed release. A short build path on a disk with tens of GB free is recommended.

```powershell
.\packaging\windows\build.ps1 `
  -PythonBin C:\Python311\python.exe `
  -EnvironmentDirectory D:\AutoFlowBuild\venv `
  -OutputDirectory D:\AutoFlowBuild\release `
  -Version dev-local
```

The model bundle is read from `build/windows-models/autoflow-models.zip` when present, or downloaded from the immutable release asset in `models.lock.json`. The build verifies its SHA-256 and all model file hashes. `-ModelsArchive` accepts another local copy of that same locked archive. Model changes require preparing and locking a new bundle.

For local builds with a compute server available, the verified Windows folder can be sent as a binary native TAR stream to `stream_archive.py --output /path/to/AutoFlow-<version>-windows-x64.zip` on that server. The script compresses without installing AutoFlow, verifies ZIP CRCs, and produces the same checksum/part manifest as the Windows archive builder. Keep the folder name equal to the ZIP basename. Use binary process streams rather than a text-converting shell pipeline. This is optional; GitHub-hosted builds remain independent of local servers.

| Parameter | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `-PythonBin` | executable | `python` | Build command | Creates the isolated Python 3.11 build environment | `packaging/windows/build.ps1` |
| `-EnvironmentDirectory` | directory | Temporary `autoflow-windows-build` | Build command | Stores the build environment | `packaging/windows/build.ps1` |
| `-OutputDirectory` | directory | Repository `release/` | Build command | Stores staged sources, models, folder and ZIP | `packaging/windows/build.ps1` |
| `-ModelsArchive` | ZIP path | Local cache, then locked download | Build command | Selects a local copy of the locked models | `packaging/windows/build.ps1` |
| `-DocumentationDirectory` | directory | Generated in a separate build environment | Build command | Uses a prebuilt offline site with a verified manifest; generators remain outside the runtime | `packaging/windows/build.ps1`, `packaging/docs/build_offline.py` |
| `-Version` | string | `dev-<HEAD>` | Build command or CI | Names the package and records its version | `packaging/windows/build.ps1` |
| `-SkipArchive` | switch | Off | Build command | Leaves the verified folder without creating a ZIP | `packaging/windows/build.ps1` |
| `-SkipDependencyInstall` | switch | Off | Build command | Reuses an already prepared build environment; optional local packages are still rebuilt from current sources | `packaging/windows/build.ps1` |
| Model asset tag/hash | string | `models.lock.json` | Repository | Makes server-independent model acquisition reproducible | `packaging/windows/models.lock.json` |
| `stream_archive.py --output` | ZIP path | Required | Optional server command | Compresses an incoming staged-folder TAR and verifies the release ZIP | `packaging/windows/stream_archive.py` |

## Offline documentation

The Windows builder creates a separate `build-docs-env`, installs only `packaging/docs/requirements.txt` there, and builds the current checkout with `mkdocs.offline.yml`. CI follows the same path. To prepare HTML on a compute server and reuse it in a local build:

```bash
conda activate autoflow311
python -m pip install -r packaging/docs/requirements.txt
python packaging/docs/build_offline.py
```

Pass the resulting `build/docs-offline` folder through `-DocumentationDirectory` when building on Windows. This avoids installing documentation generators in either the application or its build runtime environment.

The offline configuration inherits navigation and content from `mkdocs.yml`, uses explicit `.html` links, and enables Material's built-in offline search plugin. Search's iframe-worker shim and Mermaid's classic-script Tiny build are acquired from pinned npm archives, checked against `assets.lock.json`, cached under `build/docs-assets`, and included with their licenses. Classic scripts work from `file://` without module-fetch restrictions. Fonts, CSS, images, search, and diagrams need no remote subresources. HTTP 404 templates are omitted because local file navigation does not use HTTP error pages.

The builder checks local page links, HTML/CSS resources, and search-index presence, then writes `offline-manifest.json` with source, asset, and output hashes. Assembly verifies those output hashes before copying the site. The source snapshot includes Markdown, source images, MkDocs configuration, and build recipes for reproducing the pages. Online documentation deployment remains independent of this offline build.

## Model preparation

Prepare models on a server using `autoflow311`, with explicit trusted checkpoint directories:

```bash
conda activate autoflow311
python packaging/windows/prepare_models.py \
  --model-3d /path/to/Dataset7010/TrainerConfig \
  --model-4d /path/to/Dataset7020/TrainerConfig \
  --model-bundled autoflow/segmodel/nnUNetTrainerPartBalanced__nnUNetPlans__3d_fullres_iso1mm \
  --output build/windows-models/autoflow-models.zip
```

The exporter copies only inference-required dataset/plans fields and checkpoint fields. It preserves network tensors, trainer name, selected configuration, and inference mirroring axes. It verifies every saved tensor against the original. Training case maps, optimizer state, and training inputs are excluded. Each profile must contain `fold_all/checkpoint_final.pth`; best checkpoints and other available folds are retained. The current checkpoints use the upstream `nnUNetTrainer`; the exporter fails on an unbundled custom trainer.

Publish the ZIP as a new model-asset release, update the tag/URL/SHA-256 in `models.lock.json`, and rebuild. Do not replace an existing locked model asset. The compatibility resampler is in `packaging/windows/gpu_resampling.py` and is installed in the portable nnUNet package, without editing the server environment or original model plans.

## GitHub CI and automatic publication

`.github/workflows/windows-portable.yml` runs on:

- Pushes to `dev`: verify the retained regression suite, build the complete Windows package, run distribution diagnostics, upload an Actions artifact, then update the rolling `dev-windows` prerelease.
- Pushes of `v*` tags: run the same checks and publish under that tag. Tags containing a hyphen are prereleases.
- Manual dispatch: use the selected ref and optionally disable publication to retain only an Actions artifact.

The Windows workflow calls the existing reusable Linux CI for smoke/phantom coverage. CI obtains the legacy fixture model from the same locked model bundle. The Windows build uses `windows-2022`, Python 3.11, and the checkout's pinned submodule revisions. A separate publication job receives `contents: write`; builds and tests have read access. No custom release token, NAS credentials, self-hosted GPU runner, or server filesystem access is required.

Uncommitted local changes cannot be read by GitHub-hosted runners. The local build uses them immediately; automatic CI uses the exact source commit pushed to `dev` or tagged. Submodule edits must be committed and their parent gitlinks updated before CI can reproduce them.

The regular ZIP is always retained by local builds. If it exceeds GitHub's 2 GiB asset limit, publication selects verified parts below the limit plus a joiner and manifest. Every part and the joined ZIP have SHA-256 verification. `dev-windows` is deliberately a moving prerelease tag; versioned release tags remain fixed.

Consecutive pushes to the same ref cancel a superseded in-progress build. The rolling prerelease is updated only by the latest successful verified run.

## Verification and code ownership

Keep automated regression coverage in `tests/test_smoke_phantoms.py` and `tests/test_pressure_gradient_phantom.py`. Portable model/config precedence is covered in the existing smoke file. Distribution diagnostics are a manual/build tool, not an additional pytest suite.

The builder runs the private `app/portable_check.py` through the bundled interpreter to verify offline documentation/search, imports, both active model profiles, named resamplers, a spawned numerical worker, GUI/Labeler child entry points, and MP4 encoding/decoding. When CUDA is available it also executes CuPy, checks Torch/CuPy tensor exchange, and compiles both GUST kernel groups. For developer qualification on a real Windows GPU workstation, run the following from the extracted folder; it also verifies Qt/VTK rendering:

```powershell
.\_runtime\python.exe .\app\portable_check.py --gui --require-gpu --output .\workstation-check.json
```

These are internal maintenance diagnostics. Normal Windows use is double-clicking `AutoFlow.exe`.

GitHub-hosted Windows runners normally lack a CUDA GPU. A passing build confirms that the full CUDA runtime is present and importable; it does not establish real GPU execution on that runner. Inspect `build-check.json` and perform a GPU workstation check for release qualification. Model inference initialization confirms architecture/weight compatibility; it is not an accuracy evaluation on clinical data.

| Change | Owner |
| --- | --- |
| GUI launching, argument quoting, child exit status | `packaging/windows/Launcher.cs`, `portable_entry.py` |
| Python import/DLL/cache paths | `packaging/windows/sitecustomize.py`, `assemble.py` |
| Portable configuration/model precedence | `autoflow/runtime_paths.py`, `autoflow/config.py`, `autoflow/algorithms/segmentation/models.py` |
| Runtime dependency versions | `packaging/windows/requirements.txt`, `constraints.txt`, PyTorch install command in `build.ps1` |
| Inference-only models | `prepare_models.py`, `models.lock.json`, `gpu_resampling.py` |
| Build diagnostics | `portable_check.py` |
| Docs/F1 action and offline resolution | `autoflow/ui/app.py`, `autoflow/runtime_paths.py` |
| Offline HTML, search, assets, and manifests | `mkdocs.offline.yml`, `packaging/docs/`, `packaging/windows/assemble.py` |
| CI and publication | `.github/workflows/ci.yml`, `.github/workflows/windows-portable.yml` |
