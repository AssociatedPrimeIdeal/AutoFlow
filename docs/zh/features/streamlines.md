# 功能：流线与路径线

## 状态

| 入口 | 状态 | 说明 |
| --- | --- | --- |
| GUI 流线 | Supported | 支持实时流线 |
| GUI 路径线 | Supported | 支持从平面发射的时间分辨路径线 |
| CLI 流线视频 | Partial | 只支持离线流线视频 |
| CLI 路径线 | Not implemented | 当前没有公共批量路径线导出 |
| Python API | Partial | 支持离线流线视频 |

## 功能说明
流线用于看瞬时流动轨迹，路径线用于看 GUI 中从平面发射的时间分辨粒子轨迹。在 grouped multi-label 工作流里，路径线内部会继承平面的 group 名称，而 Browser 里显示给用户的名字会更短，例如 `pathline 5`。

## 何时使用
- 想看瞬时流场趋势时用流线
- 想看平面发射的时间分辨轨迹时用 GUI 路径线
- 想要离线视频时用 CLI 或 Python API

## 快速使用

### GUI
1. 载入带分割的病例
2. 点击 `Generate Streamlines`
3. 选中平面后点击 `Pathlines`，系统会为每个平面生成一条带 group 前缀的路径线对象
4. 在左侧 Browser 里可以整组控制路径线显隐，也可以右键某一条路径线单独改颜色

### CLI

```bash
autoflow-run case.h5 --output-dir results/case --streamlines-video
```

### Python API

```python
from autoflow import AutoFlowConfig, run_case
config = AutoFlowConfig(
    output_dir="./results/case",
    make_streamlines_video=True,
)
summary = run_case("case.h5", config=config)
```

## 输入

| 输入 | 是否必需 | 含义 |
| --- | --- | --- |
| flow | 是 | 速度场 |
| segmentation | 是 | seed mask 来源 |
| planes | 路径线时必需 | 发射平面 |
| RR | 路径线时必需 | 时间步缩放 |

## 参数

| 参数 | 类型 | 默认值 | 设置位置 | 作用 | 代码归属 |
| --- | --- | --- | --- | --- | --- |
| `seed_ratio` | float | `0.02` | `configs/streamlines.json` | seed 密度 | `autoflow/core/models.py` |
| `max_steps` | int | `2000` | `configs/streamlines.json` | 积分步数 | `autoflow/core/models.py` |
| `min_seeds` | int | `50` | `configs/streamlines.json` | 最少 seed 数 | `autoflow/core/models.py` |
| `terminal_speed` | float | `0.01` | `configs/streamlines.json` | 停止阈值 | `autoflow/core/models.py` |
| `rng_seed` | int | `0` | `configs/streamlines.json` | 随机种子 | `autoflow/core/models.py` |
| `tube_radius` | float | `0.05` | `configs/streamlines.json` | 渲染管半径 | `autoflow/core/models.py` |
| `pathline_color` | string | `deepskyblue` | `configs/streamlines.json` 或 GUI | 新生成路径线的默认颜色；生成后每条路径线还可以在 Browser 中单独改色 | `autoflow/ui/app.py` |

## 输出

| 输出 | 何时生成 | 含义 |
| --- | --- | --- |
| 实时流线场景对象 | GUI 流线步骤成功后 | 动态流线 |
| 实时路径线场景对象 | GUI 路径线步骤成功后 | 按平面生成的分组路径线，每条都有独立显隐和颜色 |
| `streamlines_video.mp4` 或 `streamlines_rotate.mp4` | 流线视频导出时 | 离线流线视频 |

## 限制
- 必须有分割
- 当前没有公共批量路径线导出
- 路径线目前主要是 GUI 工作流

## 改代码时看哪里

| 想改什么 | 主文件 | 还要看 | 测试 |
| --- | --- | --- | --- |
| 流线生成 | `autoflow/algorithms/streamlines.py` | `autoflow/core/pipeline.py` | `tests/test_smoke_phantoms.py` |
| 路径线行为 | `autoflow/algorithms/streamlines.py`, `autoflow/ui/app.py` | `autoflow/ui/viewer.py` | 仅保留 smoke/phantom 回归，其他靠手工验证 |
| 流线视频导出 | `autoflow/rendering/videos.py` | `autoflow/processing.py` | `tests/test_smoke_phantoms.py` |

## 测试

- `~/miniconda3/envs/ryy/bin/python -m pytest tests/test_smoke_phantoms.py -q`
- GUI 路径线行为超出这份保留回归时，改动后主要靠手工验证

## 常见问题

| 现象 | 常见原因 | 处理 |
| --- | --- | --- |
| 流线被跳过 | 没有分割或没有 flow | 检查输入和分割 |
| 路径线被跳过 | 没有平面 | 先生成平面 |
| Browser 里路径线分组不符合预期 | `label_groups` 配置不符合当前 label mask | 检查 `configs/labels.json` 里的分组并重新生成平面和路径线 |
