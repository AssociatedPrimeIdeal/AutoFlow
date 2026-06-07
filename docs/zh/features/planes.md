# 功能：平面

## 状态

| 入口 | 状态 | 说明 |
| --- | --- | --- |
| GUI | Supported | 支持拖拽调整平面 |
| CLI | Supported | 支持中心平面和按距离布面 |
| Python API | Supported | 支持通过配置控制 |

## 功能说明
平面生成功能会沿 vessel path 布置横截面分析平面。
在 grouped multi-label 工作流里，每个平面都会保留它所属 path 的 group 名称，这样 GUI 里的平面和路径线都能继续按 group 管理。

## 何时使用
- 图和路径生成完成后使用
- 每条路径只需要一个代表性平面时用中心平面模式
- 需要沿路径连续分析时用按距离模式
- 平面自动位置不理想时，用 GUI 拖拽修正

## 快速使用

### GUI
1. 运行 `Generate Graph`
2. 点击 `Generate Planes`
3. 在 Browser 或 3D 视图里选中带 group 的平面
4. 如需修正，拖拽中心球或法向球

### CLI

```bash
autoflow-run case.h5 --output-dir results/case --plane-by-distance --cross-section-dist 15
```

### Python API

```python
from autoflow import AutoFlowConfig, run_case
config = AutoFlowConfig(
    output_dir="./results/case",
    use_center_plane=False,
    cross_section_dist=15.0,
)
summary = run_case("case.h5", config=config)
```

## 输入

| 输入 | 是否必需 | 含义 |
| --- | --- | --- |
| graph 和 paths | 是 | 用于布置平面的几何基础 |
| resolution | 是 | 空间尺度信息 |

## 参数

| 参数 | 类型 | 默认值 | 设置位置 | 作用 |
| --- | --- | --- | --- | --- |
| `use_center_plane` | bool | `True` | `configs/planes.json` 或 CLI/API | 每条 path 一个中心平面 |
| `cross_section_distance` / `cross_section_dist` | float mm | `5.0` | `configs/planes.json` 或 CLI/API | 距离模式平面间距 |
| `start_distance` / `start_dist` | float mm | `5.0` | `configs/planes.json` 或 CLI/API | 起始偏移 |
| `end_distance` / `end_dist` | float mm | `0.0` | `configs/planes.json` 或 CLI/API | 结束偏移 |
| `smoothing_window` | int | `15` | `configs/planes.json` | path 平滑窗口 |
| `smoothing_polyorder` | int | `2` | `configs/planes.json` | 平滑多项式阶数 |
| `inter_time` | int | `10` | `configs/planes.json` | 平面代码使用的插值时间设置 |

## 输出

| 输出 | 何时生成 | 含义 |
| --- | --- | --- |
| `planes.json` | 有平面后 | 平面几何序列化结果 |
| `plane_positions.json` | 有平面后 | 可复用平面位置文件 |
| GUI 场景中的平面对象 | GUI 或 pipeline 成功后 | 如 `plane_aorta_systemic_branches_5` 这样的分组平面对象 |

## 限制
- 依赖 graph 和 path 质量
- GUI 才支持平面拖拽编辑

## 改代码时看哪里

| 想改什么 | 主文件 | 还要看 | 测试 |
| --- | --- | --- | --- |
| 平面生成算法 | `autoflow/algorithms/planes.py` | `autoflow/core/pipeline.py` | `tests/test_smoke_phantoms.py`, `tests/test_pressure_gradient_phantom.py` |
| 平面序列化 | `autoflow/plane_io.py` | `autoflow/core/pipeline.py` | `tests/test_pressure_gradient_phantom.py` |
| 平面拖拽编辑 | `autoflow/ui/app.py` | `autoflow/ui/ortho_viewer.py` | GUI smoke 覆盖 |

## 测试

- `~/miniconda3/envs/ryy/bin/python -m pytest tests/test_smoke_phantoms.py -q`
- `~/miniconda3/envs/ryy/bin/python -m pytest tests/test_pressure_gradient_phantom.py -q`

## 常见问题

| 现象 | 常见原因 | 处理 |
| --- | --- | --- |
| 没有生成平面 | graph 或 path 不存在 | 先生成 skeleton 和 graph |
| 平面位置不合理 | path 几何噪声较大 | 优化 graph，或在 GUI 里手工调整 |
| 平面的分组不符合预期 | 当前 `label_groups` 配置和 label mask 不匹配 | 检查 `configs/skeleton.json -> label_groups`，然后重新生成 graph 和 planes |
