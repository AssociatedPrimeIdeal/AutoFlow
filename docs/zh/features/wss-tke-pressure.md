# 功能：WSS、TKE、压力梯度

## 状态

| 入口 | 状态 | 说明 |
| --- | --- | --- |
| GUI | Supported | 支持派生体数据和场景刷新 |
| CLI | Supported | 支持派生导出和可选视频 |
| Python API | Supported | 通过批处理 API |
| 仅 TKE 这一项 | Partial | 只有输入本身支持时才有 |

## 功能说明
这部分从速度场和分割中计算壁面剪切应力、压力梯度，以及可选的 TKE 结果。

## 何时使用
- 有分割和 flow 之后使用
- 需要 WSS 或压力梯度图时使用
- 输入自带 TKE 或可由 complex 源支持时才考虑 TKE
- 不要假设所有病例都有 TKE

## 快速使用

### GUI
1. 先准备好分割
2. 如需平面摘要，先运行 `Calculate && Save Metrics`
3. 点击 `WSS / TKE / Pressure Gradient`
4. 在 browser 或 ortho viewer 里查看结果

### CLI

```bash
autoflow-run case.h5 --output-dir results/case
```

### Python API
使用 `run_case()` 或 `run_batch()`，保持 `skip_derived=False` 即可。

## 输入

| 输入 | 是否必需 | 含义 |
| --- | --- | --- |
| flow | 是 | 速度场 |
| segmentation | 是 | 有效派生计算区域 |
| `sigma` 或 `tke_array` | 可选 | TKE 来源 |
| resolution 和 origin | 是 | 空间标定 |

## 参数

| 参数 | 类型 | 默认值 | 设置位置 | 作用 |
| --- | --- | --- | --- | --- |
| `smoothing_iteration` | int | `200` | `configs/derived.json` | WSS 平滑处理 |
| `viscosity` | float | `4.0` | `configs/derived.json` | WSS 黏度参数 |
| `inward_distance` | float 或 `auto` | `auto` | `configs/derived.json` | 壁内采样距离 |
| `parabolic_fitting` | bool | `True` | `configs/derived.json` | WSS 拟合方式 |
| `no_slip_condition` | bool | `False` | `configs/derived.json` | 是否启用 no-slip |
| `rho` | float | `1060.0` | `configs/derived.json` | 压力梯度使用的密度 |
| `pressure_gradient_smoothing_sigma` | float | `0.0` | `configs/derived.json` | 压力梯度平滑 |
| `pressure_gradient_use_convective_acceleration` | bool | `True` | `configs/derived.json` | 是否包含 convective acceleration |
| `skip_derived` | bool | `False` | batch 配置或 CLI/API | 跳过整个派生步骤 |

## 输出

| 输出 | 何时生成 | 含义 |
| --- | --- | --- |
| `derived_metrics_pixelwise.npz` | 派生导出成功后 | 全体积 WSS、压力梯度、可选 TKE |
| workspace 中的 WSS surface | 派生步骤成功后 | WSS 场景数据 |
| workspace 中的 pressure-gradient 数组 | 派生步骤成功后 | 压力梯度和 support mask |
| workspace 中的 TKE 数组 | 有 TKE 时 | 可选 TKE 输出 |
| 平面摘要中的派生统计 | 同时存在平面指标和派生指标时 | 每个平面的派生摘要 |

## 限制
- 必须有分割
- TKE 必须保持可选
- 只有 `mag + flow` 的 DICOM 输入不能伪造 TKE
- 压力梯度边界体素会被 support mask 排除

## 改代码时看哪里

| 想改什么 | 主文件 | 还要看 | 测试 |
| --- | --- | --- | --- |
| WSS 计算 | `autoflow/algorithms/metrics.py` | `autoflow/ui/ortho_viewer.py` | `tests/test_smoke_phantoms.py` |
| TKE 处理 | `autoflow/algorithms/data.py`, `autoflow/algorithms/metrics.py` | `autoflow/core/pipeline.py` | `tests/test_smoke_phantoms.py` |
| 压力梯度计算 | `autoflow/algorithms/metrics.py` | `autoflow/ui/ortho_viewer.py` | `tests/test_pressure_gradient_phantom.py` |
| 派生导出串联 | `autoflow/core/pipeline.py`, `autoflow/processing.py` | `autoflow/rendering/videos.py` | `tests/test_pressure_gradient_phantom.py` |

## 测试

- `~/miniconda3/envs/ryy/bin/python -m pytest tests/test_smoke_phantoms.py -q`
- `~/miniconda3/envs/ryy/bin/python -m pytest tests/test_pressure_gradient_phantom.py -q`

## 常见问题

| 现象 | 常见原因 | 处理 |
| --- | --- | --- |
| 没有 TKE 输出 | 输入本身不支持 TKE | 这是预期行为 |
| 派生步骤被跳过 | 没有分割 | 先准备分割 |
| 压力梯度边界被裁掉 | support mask 排除了边界体素 | 这是当前设计行为 |
