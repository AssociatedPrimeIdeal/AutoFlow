# Troubleshooting

## GUI Does Not Start

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `GUI dependencies are not installed` | GUI extras missing | run `pip install -e ".[gui]"` |
| import error for `PySide6`, `pyqtgraph`, `pyvistaqt`, `vtk`, or `matplotlib` | GUI stack missing | install GUI extras in the same environment |
| `Cannot connect to DISPLAY=...` | the inherited X11 display or SSH-forwarding session is unavailable | use one of the local displays listed by the error, or reconnect with `ssh -Y` and verify `xdpyinfo` succeeds |
| SSH launch previously stopped at a forwarded `DISPLAY`, or reports `Could not create shader object`, `Could not find a decent config`, or `Failed to initialize OpenGL functions` | forwarded X11 can show Qt widgets but cannot provide a reliable embedded OpenGL context for VTK on this host | install the current build and reconnect with `ssh -Y`; AutoFlow detects `localhost:N.0` and uses interactive EGL off-screen rendering automatically |
| the SSH 3D panel is black and the terminal repeatedly reports `BufferError: memoryview: underlying buffer is not C-contiguous` | an older SSH renderer exposed the RGB channels of a PyVista RGBA screenshot as a non-contiguous view | reinstall the current editable build; the renderer now packs screenshots into a contiguous RGB buffer before creating the Qt image |
| the SSH 3D view feels less responsive than a local display | each rendered RGB frame must be transferred through X11 | reduce the window size while rotating, or use a local `DISPLAY=:1` session for full native frame rate; data processing and numerical results are unchanged |
| the 3D panel is white except for axes or a colorbar after loading data | an older build reset the camera before the first data actor existed, leaving the actor outside the view | reinstall the current build; the camera now fits the first visible actor after an empty scene is populated |
| `inotify_add_watch(...) failed: (No space left on device)` while the disks still have free space | the per-user Linux inotify watch quota is exhausted, commonly by several VS Code Remote file watchers | close unused remote VS Code sessions or ask an administrator to raise `fs.inotify.max_user_watches`; this is a file-watcher quota, not an AutoFlow data or disk-space error |
| `qt.svg.draw: The requested buffer size is too big` at GUI startup | a scalable system-theme icon was requested at an invalid effective SSH display size | reinstall the current build, which eagerly rasterizes standard controls to fixed 16 px icons |
| startup or data loading reports `Cannot set range [nan, nan]`, ends in `GraphicsView.paintEvent`, or shows VTK shader errors followed by a segmentation fault | an older PySide6 slice-view build shadowed a Qt virtual method or included decoration items in automatic image bounds | reinstall the current editable build with `python -m pip install -e ".[gui]"` |

## Segmentation-Dependent Steps Are Skipped

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| skeleton, graph, planes, or metrics say they were skipped | no segmentation is active | load, import, threshold, or auto-generate segmentation first |
| CLI run skips segmentation-dependent steps | input has no segmentation and `--autoseg` was not used | rerun with `--autoseg` or provide segmentation |

## Auto Segmentation Fails

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `auto segmentation requires loaded mag and flow data` | input is missing normalized magnitude or flow | load a supported case with `mag` and `flow` |
| `bundled default model is missing` | the install predates wheel model packaging or was built without the required local model files | reinstall the current package, or set `--autoseg-model` or the GUI model field explicitly |
| `unsupported auto segmentation backend` | backend is not `nnUNet` | use `nnUNet` |
| nnUNet subprocess fails | missing `nnUNetv2_predict_from_modelfolder`, bad model folder, or invalid checkpoint | verify nnUNet install, model folder, and checkpoint name |
| nnUNet fails with `numpy.dtype size changed` or another binary-compatibility error | the active GUI environment contains an incompatible nnUNet dependency build | repair the nnUNet dependencies in that environment or launch AutoFlow from a compatible environment; the GUI failure dialog shows the complete traceback and AutoFlow does not switch environments automatically |
| bundled auto-seg model is not the expected one | an empty model setting selects the model shipped inside the installed `autoflow` package | override with `--autoseg-model` or change the GUI auto model path if you need another trainer |
| packaged Windows auto segmentation cannot find the checkpoint | the exe was built before `fold_all/checkpoint_final.pth` was placed in the expected model directory | restore the final checkpoint and rerun `packaging\windows\build.bat`; the build script fails early when required model files are missing |
| standalone exe takes time to open | the single-file package is extracting Qt, VTK, PyTorch, nnUNet, and the model to a temporary directory | wait for extraction or build the faster-starting `onedir` form with `powershell -File packaging\windows\build.ps1 -Mode onedir` |

## TKE Is Missing

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| TKE view or TKE video is unavailable | input does not include TKE or complex source data, or `tke` was not requested | this is expected for many mag/flow-only inputs |

## DICOM Import Looks Wrong

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| wrong axis or velocity direction | DICOM parameter override is wrong | inspect and correct resolution, venc, spatial order, venc order, and RR in the DICOM confirmation dialog |
| very slow load | low worker count | try `--dicom-read-workers` for CLI or tune loader defaults |

## Video Export Warnings

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `PyVistaFutureWarning` about the default `extract_surface` algorithm during WSS or video rendering | an older call relied on PyVista's changing default surface algorithm | use the updated build, which explicitly selects `dataset_surface` when the installed PyVista supports it and falls back compatibly on older versions |
| repeated `vtkEGLRenderWindow ... Unable to eglMakeCurrent: 12290` lines during GUI export | off-screen export tried to create a second local render context while a display-backed GUI VTK context was already active | rerun with the updated build; if the host still prefers display-backed export, launch `autoflow-gui` with `AUTOFLOW_OFFSCREEN_MODE=display` |
| `eglMakeCurrent: 12290` followed by `Timers cannot be started from another thread` after generating pathlines | pathline completion updated the VTK scene from its worker thread | use the current build, which queues pathline scene updates onto the GUI thread |
| `TiffWriter.write() got an unexpected keyword argument \`fps\`` during MP4 export | ImageIO selected a still-image plugin instead of FFmpeg for an `.mp4` output | use the updated build, which forces MP4 export through the FFmpeg writer path |

## Plane Metrics Or Derived Metrics Look Incomplete

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| no planes generated | graph or segmentation is missing | generate segmentation, skeleton, and graph first |
| relative-pressure map looks trimmed | support mask excludes boundary voxels from the reconstruction support | expected behavior; check `pressure_gradient_support_mask` |
| expected PWV, WSS, TKE, pressure-gradient, or relative-pressure outputs are missing | the stage was not requested from CLI or API | rerun with `--with ...` or `requested_metrics=[...]` |

## Where To Inspect Code

- loader issues: `autoflow/algorithms/data.py`, `autoflow/algorithms/dicom.py`
- segmentation issues: `autoflow/algorithms/segmentation.py`, `autoflow/ui/app.py`
- pipeline behavior: `autoflow/core/pipeline.py`, `autoflow/processing.py`
- rendering issues: `autoflow/rendering/videos.py`
