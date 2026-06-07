# 功能：图与路径

## 状态

| 入口 | 状态 | 说明 |
| --- | --- | --- |
| GUI | Supported | 单 group 情况下支持交互式图编辑 |
| CLI | Supported | 属于默认批处理顺序 |
| Python API | Supported | 通过 pipeline 执行 |

## 功能说明
图与路径生成会把骨架变成节点、边、branch、fork 和 path，供平面和后续指标使用。
当分割被拆成多个 group 时，图生成会先按 group 分别处理，再把节点和 path 编号合并到同一个 workspace 里，同时保留每条 path 的 `group_name`。

## 何时使用
- 骨架成功后使用
- 平面生成前使用
- 自动结果需要修正时，在 GUI 中编辑图

## 快速使用

### GUI
1. 先运行 `Generate Skeleton`
2. 点击 `Generate Graph`
3. 在 Browser 和 3D 视图里检查分组 graph、fork、path 对象
4. 如有需要，并且当前只有一个 group，点击 `Edit Graph`

### CLI

```bash
autoflow-run case.h5 --output-dir results/case
```

### Python API
使用标准 `run_case()` 或 `run_batch()` 即可，图生成属于正常步骤顺序。

## 输入

| 输入 | 是否必需 | 含义 |
| --- | --- | --- |
| skeleton | 是 | 图构建的直接输入 |
| segmentation | 间接必需 | 需要先有分割才能生成骨架 |

## 参数

| 参数 | 类型 | 默认值 | 设置位置 | 作用 |
| --- | --- | --- | --- | --- |
| graph step trigger | step | 固定 | GUI 步骤或批处理顺序 | 生成节点、边、branch、path |
| edit mode | GUI mode | 关闭 | GUI | 拖拽节点、切换边、删除节点或边 |

## 输出

| 输出 | 何时生成 | 含义 |
| --- | --- | --- |
| workspace 中的 graph 数据 | 图步骤成功后 | 节点和边 |
| workspace 中的 branches 和 paths | 图步骤成功后 | 拓扑结构 |
| GUI 场景中的图对象 | GUI 图步骤成功后 | 如 `graph_aorta_systemic_branches`、`smooth_path_aorta_systemic_branches_3` 这样的分组对象 |

## 限制
- 强依赖骨架质量
- 当前没有单独成套的 CLI 图调参入口
- 只有当前恰好一个 segmentation group 时才能交互式编辑图

## 改代码时看哪里

| 想改什么 | 主文件 | 还要看 | 测试 |
| --- | --- | --- | --- |
| 图构建逻辑 | `autoflow/algorithms/graph.py` | `autoflow/core/pipeline.py` | `tests/test_smoke_phantoms.py` |
| branch 或 path 逻辑 | `autoflow/algorithms/branch.py`, `autoflow/algorithms/paths.py` | `autoflow/core/pipeline.py` | `tests/test_smoke_phantoms.py` |
| 交互式图编辑 | `autoflow/ui/app.py`, `autoflow/ui/editors.py` | `autoflow/ui/viewer.py` | GUI smoke 覆盖 |

## 测试

- `~/miniconda3/envs/ryy/bin/python -m pytest tests/test_smoke_phantoms.py -q`

## 常见问题

| 现象 | 常见原因 | 处理 |
| --- | --- | --- |
| 图步骤被跳过 | 没有骨架 | 先运行 `Generate Skeleton` |
| 路径拓扑不合理 | 骨架噪声较大 | 优化分割或在 GUI 手工编辑图 |
| `Edit Graph` 不可用 | 当前有多个 segmentation group | 改用单 group 病例，或调整 group 配置 |
