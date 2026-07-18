# Troubleshooting

## GUI Does Not Start

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `GUI dependencies are not installed` | GUI extras missing | run `pip install -e ".[gui]"` |
| import error for `PyQt5`, `pyvistaqt`, `vtk`, or `matplotlib` | GUI stack missing | install GUI extras in the same environment |

## Segmentation-Dependent Steps Are Skipped

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| skeleton, graph, planes, or metrics say they were skipped | no segmentation is active | load, import, threshold, or auto-generate segmentation first |
| CLI run skips segmentation-dependent steps | input has no segmentation and `--autoseg` was not used | rerun with `--autoseg` or provide segmentation |

## Auto Segmentation Fails

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `auto segmentation requires loaded mag and flow data` | input is missing normalized magnitude or flow | load a supported case with `mag` and `flow` |
| `nnUNet model folder not configured` | explicit model folder missing and bundled default not found | set `--autoseg-model` or configure the GUI model path |
| `unsupported auto segmentation backend` | backend is not `nnUNet` | use `nnUNet` |
| nnUNet subprocess fails | missing `nnUNetv2_predict_from_modelfolder`, bad model folder, or invalid checkpoint | verify nnUNet install, model folder, and checkpoint name |
| bundled auto-seg model is not the expected one | AutoFlow defaults to `autoflow/segmodel/nnUNetTrainerPartBalanced__nnUNetPlans__3d_fullres_iso1mm` | override with `--autoseg-model` or change the GUI auto model path if you need another trainer |

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
| `extract_surface()` got an unexpected keyword argument `algorithm` during plane, WSS, pressure-gradient, relative-pressure, or streamline video export | the installed `PyVista` release does not support the newer `extract_surface(algorithm=...)` keyword | use the updated build, which falls back to the older `extract_surface()` call automatically |
| repeated `vtkEGLRenderWindow ... Unable to eglMakeCurrent: 12290` lines during GUI export | off-screen export tried to create a second local render context while a display-backed GUI VTK context was already active | rerun with the updated build; if the host still prefers display-backed export, launch `autoflow-gui` with `AUTOFLOW_OFFSCREEN_MODE=display` |
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
