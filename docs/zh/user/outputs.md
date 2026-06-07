# 输出

## 常见输出文件

| 输出 | 何时生成 | 含义 | 主要代码 |
| --- | --- | --- | --- |
| `planes.json` | 有平面时 | 平面几何和摘要 | `autoflow/core/pipeline.py` |
| `plane_positions.json` | 有平面时 | 可复用的平面位置 | `autoflow/plane_io.py` |
| `plane_metrics.json` | 计算平面指标后 | 时间分辨平面指标 | `autoflow/core/pipeline.py` |
| `plane_qc.json` | 计算平面指标后 | fork/path 的 QC | `autoflow/core/pipeline.py` |
| `plane_metrics_pixelwise.h5` | 计算平面指标后 | 按平面保存的像素级派生采样 | `autoflow/core/pipeline.py` |
| `derived_metrics_pixelwise.npz` | 跑派生指标后 | 全体积 WSS、压力梯度和可选 TKE | `autoflow/processing.py` |
| `summary.json` | 每个病例都会有 | 单病例摘要 | `autoflow/processing.py` |
| `batch_report.json` | 批处理后 | 批量摘要 | `autoflow/api.py`, `autoflow/processing.py` |
| `time_summary.txt` | 批处理后 | 耗时摘要 | `autoflow/api.py`, `autoflow/processing.py` |

## 视频输出

| 输出 | 何时生成 | 含义 |
| --- | --- | --- |
| `planes_rotate.mp4` | `--plane-video` | 旋转平面总览 |
| `streamlines_video.mp4` 或 `streamlines_rotate.mp4` | `--streamlines-video` | 时间分辨流线视频 |
| `wss_video.mp4` 或 `wss_rotate.mp4` | `--wss-video` | WSS 视频 |
| `tke_video.mp4` 或 `tke_rotate.mp4` | `--tke-video` 且有 TKE | TKE 视频 |

## 分割 sidecar

| 输出 | 何时生成 | 含义 |
| --- | --- | --- |
| `*_threshold_segmentation.h5` | 阈值分割成功后 | 保存阈值分割结果 |
| `*_auto_segmentation.h5` | 自动分割成功后 | 保存自动分割结果 |
| 用户指定的 H5、NPY 或 NPZ | `Save Active Segmentation...` | 手动保存当前活动分割 |

## 说明

- `derived_metrics_pixelwise.npz` 保存全体积派生数组
- `plane_metrics_pixelwise.h5` 保存按平面的 slice-cell 级采样
- 对没有 TKE 的输入，TKE 相关输出会自然缺失
- `planes.json` 在有 path 元信息时会写出 `path_info`；如果平面来自 grouped multi-label 工作流，`path_info` 里可以带 `group_name`
- GUI 场景对象不是导出文件，但 grouped 工作流会使用稳定前缀命名，例如 `segmask_group_<group>`、`skeleton_<group>`、`graph_<group>`、`smooth_path_<group>_<path_idx>`、`plane_<group>_<plane_idx>`、`pathline_<group>_<plane_idx>`
- `plane_positions.json` 只保存几何信息，不保存 Browser 标题颜色或 GUI 专用分组样式
