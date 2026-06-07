# 配置系统

## 目的
AutoFlow 使用模块级 JSON 文件保存仓库默认值，并把这些默认值应用到 CLI、GUI、Python 三条路径。

## 配置模块

| 文件 | 主要内容 |
| --- | --- |
| `configs/batch.json` | 输出、跳过策略、多线程、plane reuse |
| `configs/loader.json` | 背景相位校正、dual-venc ratio、DICOM 读取设置 |
| `configs/skeleton.json` | 骨架预处理默认值、多 label label map、group 颜色、每组预处理覆盖 |
| `configs/planes.json` | 平面生成默认值 |
| `configs/streamlines.json` | 流线和路径线默认值 |
| `configs/derived.json` | WSS、TKE、压力梯度默认值 |
| `configs/segmentation.json` | segmentation dock 和自动分割对话框默认值 |
| `configs/rendering.json` | 视频和相机默认值 |

## 关键代码

- `autoflow/config.py`
- `autoflow/api.py`
- `autoflow/ui/app.py`
- `autoflow/cli.py`

## 优先级

### CLI
1. `autoflow/config.py` 里的内置默认值
2. `configs/*.json` 的覆盖值
3. 明确传入的 CLI 参数

### GUI
1. `autoflow/config.py` 里的内置默认值
2. `configs/*.json` 的覆盖值
3. GUI 交互时修改的值

### Python API
1. `AutoFlowConfig` dataclass 默认值
2. `AutoFlowConfig.from_config_dir()` 读取的 config bundle 值
3. 用户代码里的显式字段覆盖

## Skeleton 分组配置

`configs/skeleton.json` 现在同时管理传统骨架清理参数和 grouped multi-label 行为。

| 参数 | 类型 | 默认值 | 配置位置 | 作用 | 代码归属 |
| --- | --- | --- | --- | --- | --- |
| `label_map` | object | 内置血管默认值 | `configs/skeleton.json` | 把血管名称映射到整数 label | `autoflow/config.py` |
| `label_groups` | object | 内置血管 group | `configs/skeleton.json` | 把多个 label 合并成命名 group | `autoflow/core/models.py` |
| `label_groups.<group>.browser_color` | string | 各组单独默认值 | `configs/skeleton.json` | GUI Browser 顶层 group 标题颜色 | `autoflow/core/models.py` |
| `label_groups.<group>.skeleton_color` | string | 各组单独默认值 | `configs/skeleton.json` | group 骨架渲染颜色 | `autoflow/core/models.py` |
| `label_groups.<group>.graph_color` | string | 各组单独默认值 | `configs/skeleton.json` | group 图渲染颜色 | `autoflow/core/models.py` |
| `label_groups.<group>.path_color` | string | 各组单独默认值 | `configs/skeleton.json` | group 平滑路径颜色 | `autoflow/core/models.py` |
| `label_groups.<group>.plane_color` | string | 各组单独默认值 | `configs/skeleton.json` | group 平面颜色 | `autoflow/core/models.py` |
| `label_groups.<group>.preprocess` | object | `{}` | `configs/skeleton.json` | 骨架化前的每组预处理覆盖 | `autoflow/algorithms/preprocess.py` |
| `single_label_group_name` | string | `single_label` | `configs/skeleton.json` | binary 或单 label 输入的兜底 group 名称 | `autoflow/core/models.py` |
| `single_label_browser_color` | string | `#d9480f` | `configs/skeleton.json` | 单 group 兜底时的 Browser 标题颜色 | `autoflow/core/models.py` |
| `default_group_browser_color` | string | `#1c7ed6` | `configs/skeleton.json` | group 未单独配色时的 Browser 标题颜色 | `autoflow/core/models.py` |

运行时说明：

- 4D label mask 会在分组前先沿时间维度做多数决，压成 3D label。
- 小连通域清理是按每个 label 值分别做的，再合并 group。
- grouped 场景对象使用稳定前缀，例如 `segmask_group_<group>`、`skeleton_<group>`、`graph_<group>`、`smooth_path_<group>_<path_idx>`、`plane_<group>_<plane_idx>`、`pathline_<group>_<plane_idx>`。
- Browser 分组逻辑在 `autoflow/ui/app.py`，分组对象的数据集解析在 `autoflow/ui/viewer.py`。

## 当前注意点
CLI 自动分割相关公共字段目前由 `AutoFlowConfig` 管，GUI 自动分割对话框初始值来自 segmentation config bundle。写文档和重构时必须明确区分这两条路径。
