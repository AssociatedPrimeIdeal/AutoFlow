# Feature-To-Code Map

| Feature | Primary files to edit | Also check | Related tests |
| --- | --- | --- | --- |
| CLI flags, defaults, help text | `autoflow/cli.py` | `autoflow/api.py`, `autoflow/config.py`, `configs/*.json`, `pyproject.toml` | `tests/test_smoke_phantoms.py`, `tests/test_pressure_gradient_phantom.py` |
| Public Python API | `autoflow/api.py`, `autoflow/__init__.py` | `autoflow/config.py`, `autoflow/processing.py` | `tests/test_smoke_phantoms.py`, `tests/test_pressure_gradient_phantom.py` |
| Input collection and batch organization | `autoflow/processing.py`, `autoflow/algorithms/dicom.py` | `autoflow/api.py` | `tests/test_smoke_phantoms.py`, `tests/test_pressure_gradient_phantom.py` |
| H5 loading | `autoflow/algorithms/data.py` | `autoflow/core/pipeline.py`, `autoflow/case_types.py` | `tests/test_smoke_phantoms.py`, `tests/test_pressure_gradient_phantom.py` |
| DICOM scan and load | `autoflow/algorithms/dicom.py` | `autoflow/ui/dicom_confirm.py`, `autoflow/core/pipeline.py` | manual verification plus smoke and phantom regression only |
| Background phase correction | `autoflow/algorithms/phase_correction.py` | `autoflow/algorithms/data.py`, `autoflow/algorithms/dicom.py`, `autoflow/cli.py`, `autoflow/ui/app.py` | `tests/test_smoke_phantoms.py`, `tests/test_pressure_gradient_phantom.py` |
| Segmentation file IO and threshold segmentation | `autoflow/algorithms/segmentation.py` | `autoflow/ui/segmentation.py`, `autoflow/ui/app.py` | `tests/test_smoke_phantoms.py` |
| nnUNet auto segmentation | `autoflow/algorithms/segmentation.py` | `autoflow/processing.py`, `autoflow/ui/app.py`, `autoflow/config.py` | manual verification plus smoke and phantom regression only |
| Segmentation state and provenance | `autoflow/core/models.py`, `autoflow/ui/app.py` | `autoflow/ui/segmentation.py` | `tests/test_smoke_phantoms.py` |
| Skeleton | `autoflow/algorithms/preprocess.py`, `autoflow/algorithms/skeleton.py` | `autoflow/core/pipeline.py` | `tests/test_smoke_phantoms.py` |
| Graph, branch, and path logic | `autoflow/algorithms/graph.py`, `autoflow/algorithms/branch.py`, `autoflow/algorithms/paths.py` | `autoflow/core/pipeline.py`, `autoflow/ui/app.py`, `autoflow/ui/viewer.py` | `tests/test_smoke_phantoms.py` |
| Plane generation and save logic | `autoflow/algorithms/planes.py`, `autoflow/plane_io.py` | `autoflow/core/pipeline.py`, `autoflow/ui/app.py` | `tests/test_smoke_phantoms.py`, `tests/test_pressure_gradient_phantom.py` |
| Plane metrics and QC | `autoflow/algorithms/metrics.py` | `autoflow/core/pipeline.py`, `autoflow/reporting.py`, `autoflow/ui/app.py`, `autoflow/ui/ortho_viewer.py` | `tests/test_smoke_phantoms.py`, `tests/test_pressure_gradient_phantom.py` |
| WSS, TKE, pressure gradient | `autoflow/algorithms/metrics.py` | `autoflow/core/pipeline.py`, `autoflow/ui/viewer.py`, `autoflow/ui/ortho_viewer.py`, `autoflow/rendering/videos.py` | `tests/test_smoke_phantoms.py`, `tests/test_pressure_gradient_phantom.py` |
| Streamlines and pathlines | `autoflow/algorithms/streamlines.py` | `autoflow/core/pipeline.py`, `autoflow/ui/app.py`, `autoflow/rendering/videos.py` | manual verification plus smoke and phantom regression only |
| GUI main window, menus, and workflow | `autoflow/ui/app.py` | `autoflow/config.py`, `autoflow/ui/launcher.py`, `autoflow/ui/segmentation.py`, `autoflow/ui/viewer.py` | `tests/test_smoke_phantoms.py` |
| Ortho viewer and 2D interaction | `autoflow/ui/ortho_viewer.py` | `autoflow/ui/app.py` | `tests/test_smoke_phantoms.py`, `tests/test_pressure_gradient_phantom.py` |
| Offline video rendering | `autoflow/rendering/videos.py` | `autoflow/processing.py`, `autoflow/api.py` | `tests/test_smoke_phantoms.py` |
| Reporting output | `autoflow/reporting.py`, `autoflow/processing.py` | `autoflow/core/pipeline.py` | `tests/test_pressure_gradient_phantom.py` |
