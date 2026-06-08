# 功能：骨架

## 状态

| 入口 | 状态 | 说明 |
| --- | --- | --- |
| GUI | Supported | 交互式骨架编辑仍然只适用于单 group 情况 |
| CLI | Supported | 仅支持批处理生成 |
| Python API | Supported | 通过 pipeline 执行 |

## 功能说明
骨架生成把血管分割收缩成中心线式结构，作为图生成的输入。

对于 grouped multi-label 分割，AutoFlow 现在会：

1. 先把 4D label 沿时间做多数决压成 3D
2. 按 label 清理小连通域（如果启用）
3. 按 `configs/labels.json` 里的 group 配置合并 label
4. 对每个 group mask 只保留最大连通域
5. 做每组预处理
6. 分别提取每个 group 的骨架

## 何时使用
- 已经有可用分割后使用
- 图和平面之前使用
- 需要按血管 group 分开建树时使用

## 快速使用

### GUI
1. 先准备好分割
2. 点击 `Generate Skeleton`
3. 在 Browser 里检查分组骨架对象
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
| segmentation | 是 | 用于提取骨架的 binary 或 label mask |
| resolution | 是 | 体素间距，用于体积相关清理和预处理 |
| `configs/labels.json` | grouped label 工作流时需要 | label map、label group、颜色和每组预处理 |

## 参数

| 参数 | 类型 | 默认值 | 设置位置 | 作用 | 代码归属 |
| --- | --- | --- | --- | --- | --- |
| `remove_small_cc` | bool | `True` | `configs/skeleton.json` | 在 grouped 预处理前去掉小连通域 | `autoflow/core/models.py` |
| `min_cc_volume_mm3` | float | `50.0` | `configs/skeleton.json` | 连通域体积阈值 | `autoflow/core/models.py` |
| `do_closing` | bool | `True` | `configs/skeleton.json` | 全局闭运算 | `autoflow/algorithms/preprocess.py` |
| `do_opening` | bool | `False` | `configs/skeleton.json` | 全局开运算 | `autoflow/algorithms/preprocess.py` |
| `gaussian_sigma` | float | `0.5` | `configs/skeleton.json` | 全局平滑强度 | `autoflow/algorithms/preprocess.py` |
| `gaussian_enabled` | bool | `True` | `configs/skeleton.json` | 是否启用全局高斯平滑 | `autoflow/algorithms/preprocess.py` |
| `label_map` | mapping | 内置血管默认值 | `configs/labels.json` | 把符号名映射到整数 label 值 | `autoflow/config.py` |
| `label_groups` | mapping | 内置血管 group | `configs/labels.json` | 合并 label，并定义颜色与预处理覆盖 | `autoflow/core/models.py` |
| `label_groups.<group>.preprocess` | mapping | `{}` | `configs/labels.json` | 骨架化前的每组预处理覆盖 | `autoflow/algorithms/preprocess.py` |
| `single_label_group_name` | string | `single_label` | `configs/labels.json` | binary 或单 label 输入的兜底 group 名称 | `autoflow/core/models.py` |

## 输出

| 输出 | 何时生成 | 含义 |
| --- | --- | --- |
| workspace 中的 skeleton points | 骨架步骤成功后 | 中心线式骨架结果 |
| GUI 场景中的骨架对象 | GUI 步骤成功后 | 如 `skeleton_aorta_systemic_branches` 这样的分组骨架 |

## 限制
- 必须先有分割
- 结果强依赖分割质量
- 交互式骨架编辑只适用于当前恰好一个 segmentation group 的情况

## 改代码时看哪里

| 想改什么 | 主文件 | 还要看 | 测试 |
| --- | --- | --- | --- |
| 骨架前预处理 | `autoflow/algorithms/preprocess.py` | `autoflow/core/models.py`, `autoflow/config.py` | `tests/test_smoke_phantoms.py` |
| grouped 骨架 pipeline | `autoflow/core/pipeline.py` | `autoflow/algorithms/skeleton.py` | `tests/test_smoke_phantoms.py` |
| 交互式骨架编辑 | `autoflow/ui/app.py`, `autoflow/ui/editors.py` | `autoflow/core/pipeline.py` | GUI 手工验证 |

## 测试
- `~/miniconda3/envs/ryy/bin/python -m pytest tests/test_smoke_phantoms.py -q`

## 常见问题

| 现象 | 常见原因 | 处理 |
| --- | --- | --- |
| 骨架步骤被跳过 | 没有分割 | 先准备分割 |
| 骨架分支很多很乱 | 分割噪声大 | 提高清理阈值或改进分割 |
| 某个 group 的血管消失了 | 合并后只剩很小的断开成分，不是最大连通域 | 检查分割，或调整 `configs/labels.json` 的分组 |
| `Edit Skeleton` 不可用 | 当前有多个 segmentation group | 改用单 group 病例，或简化 `configs/labels.json` |
