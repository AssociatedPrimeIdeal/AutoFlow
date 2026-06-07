# 功能到代码映射

| 功能 | 主文件 | 还要看 | 相关测试 |
| --- | --- | --- | --- |
| CLI 参数、默认值、help 文案 | `autoflow/cli.py` | `autoflow/api.py`, `autoflow/config.py`, `configs/*.json` | `tests/test_smoke_phantoms.py`, `tests/test_pressure_gradient_phantom.py` |
| 公开 Python API | `autoflow/api.py`, `autoflow/__init__.py` | `autoflow/config.py`, `autoflow/processing.py` | `tests/test_smoke_phantoms.py`, `tests/test_pressure_gradient_phantom.py` |
| H5 loader | `autoflow/algorithms/data.py` | `autoflow/core/pipeline.py`, `autoflow/case_types.py` | `tests/test_smoke_phantoms.py`, `tests/test_pressure_gradient_phantom.py` |
| DICOM 扫描和载入 | `autoflow/algorithms/dicom.py` | `autoflow/ui/dicom_confirm.py`, `autoflow/core/pipeline.py` | 仅保留 smoke/phantom 回归，其他靠手工验证 |
| 分割和阈值分割 | `autoflow/algorithms/segmentation.py` | `autoflow/ui/segmentation.py`, `autoflow/ui/app.py` | `tests/test_smoke_phantoms.py` |
| nnUNet 自动分割 | `autoflow/algorithms/segmentation.py` | `autoflow/processing.py`, `autoflow/ui/app.py` | 仅保留 smoke/phantom 回归，其他靠手工验证 |
| 骨架 | `autoflow/algorithms/preprocess.py`, `autoflow/algorithms/skeleton.py` | `autoflow/core/pipeline.py` | `tests/test_smoke_phantoms.py` |
| 图、branch、path | `autoflow/algorithms/graph.py`, `autoflow/algorithms/branch.py`, `autoflow/algorithms/paths.py` | `autoflow/core/pipeline.py`, `autoflow/ui/app.py` | `tests/test_smoke_phantoms.py` |
| 平面和保存 | `autoflow/algorithms/planes.py`, `autoflow/plane_io.py` | `autoflow/core/pipeline.py`, `autoflow/ui/app.py` | `tests/test_smoke_phantoms.py`, `tests/test_pressure_gradient_phantom.py` |
| 平面指标和 QC | `autoflow/algorithms/metrics.py` | `autoflow/core/pipeline.py`, `autoflow/reporting.py`, `autoflow/ui/ortho_viewer.py` | `tests/test_smoke_phantoms.py`, `tests/test_pressure_gradient_phantom.py` |
| WSS、TKE、压力梯度 | `autoflow/algorithms/metrics.py` | `autoflow/core/pipeline.py`, `autoflow/rendering/videos.py` | `tests/test_smoke_phantoms.py`, `tests/test_pressure_gradient_phantom.py` |
| 流线和路径线 | `autoflow/algorithms/streamlines.py` | `autoflow/core/pipeline.py`, `autoflow/ui/app.py`, `autoflow/rendering/videos.py` | 仅保留 smoke/phantom 回归，其他靠手工验证 |
