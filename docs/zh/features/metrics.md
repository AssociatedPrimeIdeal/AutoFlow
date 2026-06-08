# 功能：平面指标

## 状态

| 入口 | 状态 | 说明 |
| --- | --- | --- |
| GUI | Supported | 对应 `Calculate && Save Metrics` |
| CLI | Supported | 默认批处理的一部分 |
| Python API | Supported | 通过 `run_case()` / `run_batch()` |

## 功能说明
平面指标会对每个平面计算时间分辨的截面指标，并写入 JSON 和 HDF5 输出。

如果 `configs/pwv.json -> enabled` 为真，平面指标步骤还会额外计算每个 PWV group 的 PWV，并写出 PWV 输出。

## 何时使用
- 平面生成之后使用
- 需要流量、面积、平均速度、峰值速度、净流量等截面指标时使用
- 没有分割时不能用

## 快速使用

### GUI
1. 先生成平面
2. 点击 `Calculate && Save Metrics`
3. 在 selection 和 ortho viewer 里查看结果

### CLI

```bash
autoflow-run case.h5 --output-dir results/case
```

### Python API
使用标准 `run_case()` 或 `run_batch()`；除非显式设置 `skip_plane_metrics=True`，否则会计算平面指标。

## 输入

| 输入 | 是否必需 | 含义 |
| --- | --- | --- |
| flow | 是 | 时间分辨速度场 |
| segmentation | 是 | 平面采样有效区域 |
| planes | 是 | 分析平面 |

## 参数

| 参数 | 类型 | 默认值 | 设置位置 | 作用 |
| --- | --- | --- | --- | --- |
| `skip_plane_metrics` | bool | `False` | batch 配置或 CLI/API | 跳过整个平面指标步骤 |
| `use_multithread` | bool | `True` | `configs/batch.json` | 平面指标多线程计算 |

## 输出

| 输出 | 何时生成 | 含义 |
| --- | --- | --- |
| `plane_metrics.json` | 平面指标成功后 | 时间分辨平面指标 |
| `plane_qc.json` | 平面指标成功后 | fork 和 path 的 QC |
| `plane_metrics_pixelwise.h5` | 平面指标成功后 | 按平面保存的像素级派生采样 |
| `planes.json` 中附带的摘要 | 平面指标成功后 | 平面几何和摘要一起保存 |
| `pwv.json` | 启用 PWV 时 | 配置好的 label group 的 PWV 结果 |
| `pwv_<group>.png` | PWV 绘图成功后 | PWV 图 |

## 限制
- 必须有分割
- 强依赖平面位置质量
- 平面摘要里是否有派生统计，取决于派生指标是否可用

## 改代码时看哪里

| 想改什么 | 主文件 | 还要看 | 测试 |
| --- | --- | --- | --- |
| 平面指标计算 | `autoflow/algorithms/metrics.py` | `autoflow/core/pipeline.py` | `tests/test_smoke_phantoms.py`, `tests/test_pressure_gradient_phantom.py` |
| 平面指标保存格式 | `autoflow/core/pipeline.py`, `autoflow/plane_io.py` | `autoflow/reporting.py` | `tests/test_pressure_gradient_phantom.py` |
| GUI 中的平面指标刷新和 PWV dock | `autoflow/ui/app.py`, `autoflow/ui/ortho_viewer.py` | `autoflow/core/pipeline.py`, `autoflow/algorithms/pwv.py` | GUI 手工验证 |

## 测试

- `~/miniconda3/envs/ryy/bin/python -m pytest tests/test_smoke_phantoms.py -q`
- `~/miniconda3/envs/ryy/bin/python -m pytest tests/test_pressure_gradient_phantom.py -q`

## 常见问题

| 现象 | 常见原因 | 处理 |
| --- | --- | --- |
| 指标步骤被跳过 | 没有分割或没有 flow | 检查输入和分割 |
| 拖拽平面后指标没更新 | 平面拖拽还没完成 | 结束拖拽，等待 GUI 重算并持久化 |
