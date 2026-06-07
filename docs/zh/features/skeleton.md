# 功能：骨架

## 状态

| 入口 | 状态 | 说明 |
| --- | --- | --- |
| GUI | Supported | 单 group 情况下支持交互式骨架编辑 |
| CLI | Supported | 支持批处理生成 |
| Python API | Supported | 通过 pipeline 执行 |

## 功能说明
骨架生成把血管分割收缩成中心线式结构，作为图生成的输入。
对于 grouped multi-label 分割，AutoFlow 会先把 4D label 沿时间做多数决压成 3D，再按 label 清理小连通域，按配置 group 合并，做每组预处理，最后分别提取每个 group 的骨架。

## 何时使用
- 已经有可用分割后使用
- 图和平面之前使用
- 没有分割时不能使用

## 快速使用

### GUI
1. 先准备好分割
2. 点击 `Generate Skeleton`
3. 在 Browser 里检查 `skeleton_aorta_systemic_branches` 这类分组骨架对象
4. 如需修正，并且当前只有一个 group，点击 `Edit Skeleton`

### CLI

```bash
autoflow-run case.h5 --output-dir results/case
```

### Python API

```python
from autoflow import AutoFlowConfig, run_case
summary = run_case("case.h5", config=AutoFlowConfig(output_dir="./results/case"))
```

## 输入

| 输入 | 是否必需 | 含义 |
| --- | --- | --- |
| segmentation | 是 | 用于骨架化的 mask |
| resolution | 是 | 体素间距，用于体积相关预处理 |

## 参数

| 参数 | 类型 | 默认值 | 设置位置 | 作用 | 代码归属 |
| --- | --- | --- | --- | --- | --- |
| `remove_small_cc` | bool | `True` | `configs/skeleton.json` | 去掉小连通域 | `autoflow/core/models.py` |
| `min_cc_volume_mm3` | float | `50.0` | `configs/skeleton.json` | 连通域体积阈值 | `autoflow/core/models.py` |
| `do_closing` | bool | `True` | `configs/skeleton.json` | 骨架化前闭运算 | `autoflow/algorithms/preprocess.py` |
| `do_opening` | bool | `False` | `configs/skeleton.json` | 骨架化前开运算 | `autoflow/algorithms/preprocess.py` |
| `gaussian_sigma` | float | `0.5` | `configs/skeleton.json` | 平滑强度 | `autoflow/algorithms/preprocess.py` |
| `gaussian_enabled` | bool | `True` | `configs/skeleton.json` | 是否启用高斯平滑 | `autoflow/algorithms/preprocess.py` |
| `dilation_iters` | int | `0` | `configs/skeleton.json` | 全局膨胀次数 | `autoflow/algorithms/preprocess.py` |
| `erosion_iters` | int | `0` | `configs/skeleton.json` | 全局腐蚀次数 | `autoflow/algorithms/preprocess.py` |
| `opening_iters` | int | `0` | `configs/skeleton.json` | 全局开运算次数 | `autoflow/algorithms/preprocess.py` |
| `closing_iters` | int | `0` | `configs/skeleton.json` | 全局闭运算次数 | `autoflow/algorithms/preprocess.py` |
| `label_map` | mapping | 内置血管默认值 | `configs/skeleton.json` | 把 label 名称映射到整数 label 值 | `autoflow/config.py` |
| `label_groups` | mapping | 内置血管 group | `configs/skeleton.json` | 把多个 label 合并成命名 group，并定义颜色与预处理覆盖 | `autoflow/core/models.py` |
| `single_label_group_name` | string | `single_label` | `configs/skeleton.json` | binary 或单 label 输入的兜底 group 名称 | `autoflow/core/models.py` |
| `single_label_browser_color` | string | `#d9480f` | `configs/skeleton.json` | 单 group 兜底时 Browser 标题颜色 | `autoflow/core/models.py` |
| `default_group_browser_color` | string | `#1c7ed6` | `configs/skeleton.json` | group 未单独配色时的默认 Browser 颜色 | `autoflow/core/models.py` |

## 输出

| 输出 | 何时生成 | 含义 |
| --- | --- | --- |
| workspace 中的 skeleton points | 骨架步骤成功后 | 骨架结果 |
| GUI 场景中的骨架对象 | GUI 步骤成功后 | 如 `skeleton_aorta_systemic_branches` 这样的分组骨架 |

## 限制
- 必须先有分割
- 结果强依赖分割质量
- 当前没有单独对外公开的骨架文件导出格式
- 只有当前恰好一个 segmentation group 时才能交互式编辑骨架

## 改代码时看哪里

| 想改什么 | 主文件 | 还要看 | 测试 |
| --- | --- | --- | --- |
| 骨架前预处理 | `autoflow/algorithms/preprocess.py` | `autoflow/core/models.py` | `tests/test_smoke_phantoms.py` |
| 骨架提取 | `autoflow/algorithms/skeleton.py` | `autoflow/core/pipeline.py` | `tests/test_smoke_phantoms.py` |
| 交互式骨架编辑 | `autoflow/ui/app.py`, `autoflow/ui/editors.py` | `autoflow/core/pipeline.py` | GUI smoke 覆盖 |

## 测试

- `~/miniconda3/envs/ryy/bin/python -m pytest tests/test_smoke_phantoms.py -q`

## 常见问题

| 现象 | 常见原因 | 处理 |
| --- | --- | --- |
| 骨架步骤被跳过 | 没有分割 | 先准备分割 |
| 骨架分支很多很乱 | 分割噪声大 | 提高清理阈值或改进分割 |
| `Edit Skeleton` 不可用 | 当前有多个 segmentation group | 改用单 group 病例，或调整 group 配置 |
