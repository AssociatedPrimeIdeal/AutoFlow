# Windows portable builds

## Architecture

The full Windows distribution uses one small x64 C# launcher with an application-local CPython 3.11 runtime. `AutoFlow.exe` launches the complete GUI through `pythonw.exe` without a console window. The launcher requires the .NET Framework supplied with Windows 10/11. It forwards GUI arguments with Windows-compatible quoting and returns the child exit status. The distribution exposes only this GUI entry point; the internal Python interpreter remains necessary for segmentation, Labeler, rendering, multiprocessing, and build diagnostics.

The portable package excludes documentation and documentation generators. The GUI has no Docs action or F1 documentation shortcut. QC uses an English HTML page in the main work area. The PySide6 distribution includes QtWebEngine and its helper/resources for QC. `portable.json` records `documentation_bundled=false`.

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

The lock may list independently hashed archive parts. `packaging/windows/download_models.py` fetches them, verifies each part and the reconstructed ZIP, and caches only the complete archive after success. CI uses GitHub CLI authentication; ordinary local builds can download the public assets without GitHub CLI. Both regression fixtures and the portable builder use this same downloader.

For local builds with a compute server available, the verified Windows folder can be sent as a binary native TAR stream to `stream_archive.py --output /path/to/AutoFlow-<version>-windows-x64.zip` on that server. The script compresses without installing AutoFlow, verifies ZIP CRCs, and produces the same checksum/part manifest as the Windows archive builder. Keep the folder name equal to the ZIP basename. Use binary process streams rather than a text-converting shell pipeline. This is optional; GitHub-hosted builds remain independent of local servers.

| Parameter | Type | Default | Where configured | Effect | Code owner |
| --- | --- | --- | --- | --- | --- |
| `-PythonBin` | executable | `python` | Build command | Creates the isolated Python 3.11 build environment | `packaging/windows/build.ps1` |
| `-EnvironmentDirectory` | directory | Temporary `autoflow-windows-build` | Build command | Stores the build environment | `packaging/windows/build.ps1` |
| `-OutputDirectory` | directory | Repository `release/` | Build command | Stores staged sources, models, folder and ZIP | `packaging/windows/build.ps1` |
| `-ModelsArchive` | ZIP path | Local cache, then locked download | Build command | Selects a local copy of the locked models | `packaging/windows/build.ps1` |
| `-Version` | string | `dev-<HEAD>` | Build command or CI | Names the package and records its version | `packaging/windows/build.ps1` |
| `-SkipArchive` | switch | Off | Build command | Leaves the verified folder without creating a ZIP | `packaging/windows/build.ps1` |
| `-SkipDependencyInstall` | switch | Off | Build command | Reuses an already prepared build environment; optional local packages are still rebuilt from current sources | `packaging/windows/build.ps1` |
| Model asset tag/hash | string | `models.lock.json` | Repository | Makes server-independent model acquisition reproducible | `packaging/windows/models.lock.json` |
| `stream_archive.py --output` | ZIP path | Required | Optional server command | Compresses an incoming staged-folder TAR and verifies the release ZIP | `packaging/windows/stream_archive.py` |

## Model preparation

Prepare models on a server using `autoflow311`, with explicit trusted checkpoint directories:

```bash
conda activate autoflow311
python packaging/windows/prepare_models.py --from-current-defaults \
  --output /tmp/autoflow-models/autoflow-models.zip
```

The exporter resolves the current checkout's Linux model defaults, copies only inference-required dataset/plans and checkpoint fields, and preserves network tensors, trainer name, selected configuration, and inference mirroring axes. It verifies every saved tensor against the original and records source hashes. Training case maps, optimizer state, and training inputs are excluded. Static profiles require `fold_all/checkpoint_final.pth`; 4D requires `fold_all/checkpoint_best.pth` and may have no final checkpoint. Available other checkpoints/folds are retained. The current checkpoints use the upstream `nnUNetTrainer`; the exporter fails on an unbundled custom trainer. Explicit paths remain available through the three `--model-*` flags instead of `--from-current-defaults`.

The immutable model assets in `v2.0.0` match the Linux Dataset7010 static model, the 37-channel/30-class `nnunet4d_labels30_20261102_af40f6a7f3d4` model and the original bundled static model. The previous `windows-models-20261106` release was renamed; model bytes and hashes are unchanged. Every exported model produced exactly equal CPU logits to its source on a fixed random input. This verifies representation/inference parity, not clinical accuracy or bitwise identity across operating systems and GPU kernels.

Attach a changed model ZIP to a new numbered release, update the tag/URL/SHA-256 in `models.lock.json`, and rebuild. Keep locked model assets available and do not replace their contents. Model resources share ordinary version tags with releases instead of creating separate date-based tags. The compatibility resampler is in `packaging/windows/gpu_resampling.py` and is installed in the portable nnUNet package, without editing the server environment or original model plans.

## GitHub CI and manual builds

Pushes and pull requests to `main` or `dev` run the Linux/Windows regression matrix. When both pass, CI calls the reusable `windows-portable.yml` workflow to build and verify the Windows package. Successful push builds then publish a GitHub Release automatically. PRs, manual CI runs, and other reusable callers build without publishing by default. To start the complete verification flow manually, open **Actions → CI → Run workflow** and select the ref. The verified package is also available as the `AutoFlow-windows-x64` Actions artifact for 14 days. `AutoFlow-windows-diagnostics` preserves `build-check.json` when available, including failed verification.

`packaging/windows/release.py` selects `vMAJOR.MINOR.PATCH` before packaging. `project.version` in `pyproject.toml` is the minimum version; if it is not greater than the highest numbered tag or release, the next patch number is selected. For example, after the `v2.0.0` source/model baseline, the next push publishes `v2.0.1`, followed by `v2.0.2`. Set a higher `project.version` to start a new minor or major version. Package names and `portable.json` use the selected release version. Publication does not commit version bumps back to the branch.

Push builds share a concurrency group across both branches so version selection, packaging, and publication cannot race. Retries resume their commit's draft or keep its already published release. The publisher checks archive-part order, sizes, each SHA-256 and the joined archive SHA-256, uploads to a draft, and verifies GitHub asset sizes and hashes before publishing it. Interrupted uploads stay drafts. The build job has read-only repository access; only the final publisher has `contents: write`, delegated by the CI caller. The built-in GitHub token is sufficient; no release PAT is required.

Release notes list non-merge commits since the preceding published numbered version, with commit links and a full comparison link. This also covers direct pushes without pull requests. Notes include download instructions and the source commit. The user-facing entry point is [Changelog](../user/changelog.md).

Linux runs the full retained suite, including SSH-renderer phantoms. Windows runs the numerical, batch, output and cross-platform Qt checks; phantoms that explicitly force the Linux SSH renderer are platform-scoped because that renderer is outside the Windows launcher path. The portable build separately verifies the native Windows `QtInteractor` GUI with `portable_check.py --gui`. CI obtains the legacy fixture model from the same locked model bundle. The Windows build uses `windows-2022`, Python 3.11, the checkout's pinned submodule revisions and read-only repository access. The model archive is cached by lock-file hash; dependency downloads are not retained on the build runner to limit disk use.

The local build can include uncommitted changes. GitHub-hosted builds use the selected commit and its pinned submodules; commit submodule edits and update their gitlinks before building on GitHub.

Reusable CI installs the `test` extra with the portable dependency constraints,
including an FFmpeg executable for the MP4 phantoms. Before regression tests and portable GUI verification,
the pinned PyVista headless-display action prepares Linux Qt/OpenGL libraries
and Windows offscreen Mesa. Tests explicitly use EGL on Linux and OSMesa on
Windows; an empty Windows backend would override the action and fall back to
native WGL on a runner without a usable OpenGL context. See
[Headless CI rendering](testing.md#headless-ci-rendering) for maintenance details.

The local archive build retains the complete ZIP. The Actions artifact includes the ZIP or, for large builds, the verified parts, joiner and manifest. Every part and the joined ZIP have SHA-256 verification. A newer CI run for the same ref cancels the in-progress run.

## Verification and code ownership

See [Windows CI GUI verification](windows-ci-headless.md) for offscreen interactor handling and the rendering fields in `build-check.json`.

Keep automated regression coverage in `tests/test_smoke_phantoms.py` and `tests/test_pressure_gradient_phantom.py`. Portable model/config precedence is covered in the existing smoke file. Distribution diagnostics are a manual/build tool, not an additional pytest suite.

The builder runs the private `app/portable_check.py` through the bundled interpreter to verify imports, both active model profiles and their labels/default checkpoints, named resamplers, a spawned numerical worker, GUI/Labeler child entry points, and MP4 encoding/decoding. When CUDA is available it also executes CuPy, checks Torch/CuPy tensor exchange, and compiles both GUST kernel groups. For developer qualification on a real Windows GPU workstation, run the following from the extracted folder; it also verifies Qt/VTK rendering:

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
| Inference-only models and locked downloads | `prepare_models.py`, `models.lock.json`, `download_models.py`, `gpu_resampling.py` |
| Build diagnostics | `portable_check.py` |
| CI, version selection, release notes and publication | `.github/workflows/ci.yml`, `.github/workflows/windows-portable.yml`, `packaging/windows/release.py` |
