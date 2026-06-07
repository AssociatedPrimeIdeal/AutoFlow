# GUI 指南

## 状态
`autoflow-gui` 支持交互式载入、分割检查、分组血管处理、步骤执行和可视化。

主要入口：

- `autoflow/ui/launcher.py`
- `autoflow/ui/app.py`

## 启动

```bash
autoflow-gui
```

或者：

```bash
autoflow-gui --config-dir ./configs
```

## 主界面布局

主窗口包含：

- 菜单栏
- 左侧 Browser
- 中央 3D 视图
- Steps 区域
- 参数面板
- 右侧 ortho viewer
- 底部 Timeline
- 底部 Selection
- 底部 Log
- 右侧 Segmentation dock

左侧 Browser 支持按分割 group 分组。一个多 label 分割如果被拆成多个血管 group，Browser 会给每个 group 建一个顶层条目，方便整组显隐。

## File 菜单

| 菜单项 | 作用 |
| --- | --- |
| `Open H5` | 打开 H5/HDF5 |
| `Import DICOM Directory` | 扫描 DICOM 目录并选择病例 |
| `Clear Workspace` | 清空当前工作区并恢复 UI 默认值 |
| `Exit` | 退出 GUI |

## 标准流程

1. 启动 GUI
2. 用 `Open H5` 或 `Import DICOM Directory` 载入病例
3. DICOM 情况下确认或修改 `Resolution XYZ`、`VENC XYZ`、`Spatial Order`、`VENC Order`、`RR`
4. 检查 `Input / Background Correction` 参数；如果载入的是 legacy `Nv=7` H5，还要确认 dual-venc ratio 参数
5. 如果没有分割，用 segmentation 菜单或 dock 选择分割来源
6. 如果分割是 label mask，AutoFlow 会先把 4D label 沿时间做多数决压成 3D，再按 label 清理连通域、按配置归组、做每组预处理，然后继续生成骨架、图和平面
7. 单独运行步骤，或点击 `Run All`
8. 在 Browser、3D 视图、ortho viewer、selection 里检查各个 group 的结果
9. 如果需要单独修改某条路径线颜色，在左侧 Browser 里右键该路径线
10. 如需复用分割，保存当前活动分割

## 步骤按钮

| 步骤 | 作用 | 说明 |
| --- | --- | --- |
| `Generate Skeleton` | 生成骨架 | binary mask 当成一个 group；label mask 按配置 group 处理 |
| `Generate Graph` | 生成图、branch、path | 依赖骨架，并保留 group 名称 |
| `Generate Planes` | 生成平面 | 依赖图，并保留 path/group 归属 |
| `Calculate && Save Metrics` | 计算并保存平面指标 | 依赖分割和流场 |
| `WSS / TKE / Pressure Gradient` | 计算派生体数据并刷新场景 | TKE 仍然可选 |
| `Generate Streamlines` | 生成实时流线 | 使用合并后的整体分割 |
| `Pathlines` | 从平面发射时间分辨路径线 | 会继承平面的 group 名称 |
| `Edit Skeleton` | 交互式改骨架 | 仅在当前只有一个 segmentation group 时可用 |
| `Edit Graph` | 交互式改图 | 仅在当前只有一个 segmentation group 时可用 |
| `Run All` | 跑固定顺序 | 不会自动创建分割 |

当前 `Run All` 顺序：

1. `Generate Skeleton`
2. `Generate Graph`
3. `Generate Planes`
4. `Calculate && Save Metrics`
5. `WSS / TKE / Pressure Gradient`

## 参数面板

| 面板 | 主要作用 |
| --- | --- |
| `Input / Background Correction` | loader 和 DICOM 设置 |
| `Generate Skeleton Parameters` | 清理和形态学参数；label map、label group、group 颜色来自 `configs/skeleton.json` |
| `Generate Planes Parameters` | 中心平面或按距离布面 |
| `Streamline Parameters` | seed 密度、步数、终止阈值、颜色 |
| `WSS Parameters` | WSS 相关参数 |
| `Flow / TKE / Pressure Gradient Parameters` | 派生指标参数 |

## 分割工作流

GUI 分割系统支持：

- 原始分割
- 导入分割
- 阈值分割
- `nnUNet` 自动分割
- segmentation dock 中的手工编辑

关键行为：

- `Run All` 不会自动触发分割
- 依赖分割的步骤运行前，必须已经有活动分割
- 自动分割通过 `Configure Segmentation...` 触发
- GUI 自动分割会弹出模态进度框，并在推理和 sidecar 保存完成前锁定当前病例状态
- 阈值分割和自动分割成功后都会写 sidecar H5

grouped label-mask 行为：

- binary mask 和 single-label mask 会被当成一个 group
- 4D label mask 会在后续血管步骤前，先沿时间维度做多数决得到 3D label
- 小连通域清理是按每个 label 值分别做的，再按 group 合并
- label 归组、Browser 颜色、每组预处理来自 `configs/skeleton.json`
- 每个 group 可以单独覆盖高斯平滑、膨胀、腐蚀、开运算、闭运算参数

## 选择、Browser 和 Timeline

- Browser 会为每个 segmentation group 建一个顶层条目，没有 group 的对象放在 `Global`
- 勾选或取消一个 group 顶层条目，会同时控制这个 group 下所有对象的显隐
- group 标题颜色来自 `label_groups.<group>.browser_color`；single-label 或未匹配 group 时使用 fallback 颜色
- group 对象命名采用前缀区分，例如 `segmask_group_aorta_systemic_branches`、`skeleton_aorta_systemic_branches`、`graph_aorta_systemic_branches`、`smooth_path_aorta_systemic_branches_3`、`plane_aorta_systemic_branches_5`、`pathline_aorta_systemic_branches_5`
- 右键单条路径线可以只改这一条路径线的颜色
- 选中平面会更新 selection 和 ortho viewer
- 选中 path 会显示 path 级信息
- Timeline 控制时间分辨对象和 ortho 切片

## Ortho Viewer

ortho viewer 支持：

- flow 分量
- magnitude
- PC-MRA
- speed
- WSS
- TKE
- pressure-gradient 分量和模长
- 编辑时的 segmentation paint
- jump-to-plane-center

## 何时使用 GUI

适合 GUI 的情况：

- 需要交互式检查或修正分割
- 需要在多 label 工作流里整组控制可视化
- 需要在单 group 情况下编辑 skeleton 或 graph
- 需要拖拽平面并立刻重算指标
- 需要时间维导航

适合 CLI 的情况：

- 无人值守批处理
- 多病例重复运行
- 离线视频导出
