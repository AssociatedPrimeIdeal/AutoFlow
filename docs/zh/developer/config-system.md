# 配置系统

## 目的
AutoFlow 使用模块级 JSON 文件保存仓库默认值，并把这些默认值应用到 CLI、GUI、Python 三条路径。

## 配置模块

| 文件 | 主要内容 |
| --- | --- |
| `configs/batch.json` | 输出、跳过策略、多线程、plane reuse |
| `configs/loader.json` | 背景相位校正、dual-venc ratio、DICOM 读取设置 |
| `configs/skeleton.json` | 骨架清理和形态学默认值 |
| `configs/labels.json` | label map、label group、Browser 颜色、每组预处理覆盖 |
| `configs/planes.json` | 平面生成默认值 |
| `configs/streamlines.json` | 流线和路径线默认值 |
| `configs/derived.json` | WSS、TKE、压力梯度默认值 |
| `configs/pwv.json` | PWV group、平面间距、波形选择和绘图样式 |
| `configs/segmentation.json` | segmentation dock 和分割运行默认值 |
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

## Skeleton 清理配置

| 参数 | 类型 | 默认值 | 配置位置 | 作用 | 代码归属 |
| --- | --- | --- | --- | --- | --- |
| `remove_small_cc` | bool | `True` | `configs/skeleton.json` | 在 grouped 预处理前去掉小连通域 | `autoflow/core/models.py` |
| `min_cc_volume_mm3` | float | `50.0` | `configs/skeleton.json` | 连通域体积阈值 | `autoflow/core/models.py` |
| `do_closing` | bool | `True` | `configs/skeleton.json` | 骨架化前闭运算 | `autoflow/algorithms/preprocess.py` |
| `do_opening` | bool | `False` | `configs/skeleton.json` | 骨架化前开运算 | `autoflow/algorithms/preprocess.py` |
| `gaussian_sigma` | float | `0.5` | `configs/skeleton.json` | 平滑强度 | `autoflow/algorithms/preprocess.py` |
| `gaussian_enabled` | bool | `True` | `configs/skeleton.json` | 是否启用高斯平滑 | `autoflow/algorithms/preprocess.py` |
| `dilation_iters` | int | `0` | `configs/skeleton.json` | 全局膨胀次数 | `autoflow/algorithms/preprocess.py` |
| `erosion_iters` | int | `0` | `configs/skeleton.json` | 全局腐蚀次数 | `autoflow/algorithms/preprocess.py` |
| `opening_iters` | int | `0` | `configs/skeleton.json` | 全局开运算次数 | `autoflow/algorithms/preprocess.py` |
| `closing_iters` | int | `0` | `configs/skeleton.json` | 全局闭运算次数 | `autoflow/algorithms/preprocess.py` |

## Label 分组配置

| 参数 | 类型 | 默认值 | 配置位置 | 作用 | 代码归属 |
| --- | --- | --- | --- | --- | --- |
| `label_map` | object | 内置血管默认值 | `configs/labels.json` | 把血管名称映射到整数 label | `autoflow/config.py` |
| `label_groups` | object | 内置血管 group | `configs/labels.json` | 把多个 label 合并成命名 group | `autoflow/core/models.py` |
| `label_groups.<group>.browser_color` | string | 各组单独默认值 | `configs/labels.json` | GUI Browser 顶层 group 标题颜色 | `autoflow/core/models.py` |
| `label_groups.<group>.skeleton_color` | string | 各组单独默认值 | `configs/labels.json` | group 骨架渲染颜色 | `autoflow/core/models.py` |
| `label_groups.<group>.graph_color` | string | 各组单独默认值 | `configs/labels.json` | group 图渲染颜色 | `autoflow/core/models.py` |
| `label_groups.<group>.path_color` | string | 各组单独默认值 | `configs/labels.json` | group 平滑路径颜色 | `autoflow/core/models.py` |
| `label_groups.<group>.plane_color` | string | 各组单独默认值 | `configs/labels.json` | group 平面颜色 | `autoflow/core/models.py` |
| `label_groups.<group>.preprocess` | object | `{}` | `configs/labels.json` | 骨架化前的每组预处理覆盖 | `autoflow/algorithms/preprocess.py` |
| `single_label_group_name` | string | `single_label` | `configs/labels.json` | binary 或单 label 输入的兜底 group 名称 | `autoflow/core/models.py` |
| `single_label_browser_color` | string | `#d9480f` | `configs/labels.json` | 单 group 兜底时的 Browser 标题颜色 | `autoflow/core/models.py` |
| `default_group_browser_color` | string | `#1c7ed6` | `configs/labels.json` | group 未单独配色时的 Browser 标题颜色 | `autoflow/core/models.py` |

## PWV 配置

| 参数 | 类型 | 默认值 | 配置位置 | 作用 | 代码归属 |
| --- | --- | --- | --- | --- | --- |
| `enabled` | bool | `False` | `configs/pwv.json` | 是否在平面指标步骤中计算 PWV | `autoflow/core/pipeline.py` |
| `groups` | list | `[]` | `configs/pwv.json` | 每个元素定义一个 PWV label 组合 | `autoflow/core/models.py` |
| `groups[].name` | string | 自动生成 | `configs/pwv.json` | 在 `pwv.json` 和 GUI dock 中显示的名字 | `autoflow/core/models.py` |
| `groups[].labels` | list[int 或符号名] | 每组必填 | `configs/pwv.json` | 合并成一个 PWV mask 的 label 列表 | `autoflow/core/models.py` |
| `plane_interval_mm` | float | `10.0` | `configs/pwv.json` | 沿最长路径布置 PWV 平面的间距 | `autoflow/algorithms/pwv.py` |
| `start_distance` | float | `0.0` | `configs/pwv.json` | 起点偏移 | `autoflow/algorithms/pwv.py` |
| `end_distance` | float | `0.0` | `configs/pwv.json` | 终点偏移 | `autoflow/algorithms/pwv.py` |
| `smoothing_window` | int | `15` | `configs/pwv.json` | PWV 平面生成时使用的路径平滑窗口 | `autoflow/algorithms/pwv.py` |
| `smoothing_polyorder` | int | `2` | `configs/pwv.json` | 路径平滑多项式阶数 | `autoflow/algorithms/pwv.py` |
| `inter_time` | int | `10` | `configs/pwv.json` | 平面生成使用的插值倍数 | `autoflow/algorithms/pwv.py` |
| `waveform_key` | string | `flowrate_signed_mL_s` | `configs/pwv.json` | 用于 foot detection 的波形字段 | `autoflow/algorithms/pwv.py` |
| `foot_savgol_window` | int | `5` | `configs/pwv.json` | 波形平滑的 Savitzky-Golay 窗口 | `autoflow/algorithms/pwv.py` |
| `foot_savgol_polyorder` | int | `2` | `configs/pwv.json` | 波形平滑的 Savitzky-Golay 阶数 | `autoflow/algorithms/pwv.py` |
| `minimum_valid_planes` | int | `2` | `configs/pwv.json` | 拟合 PWV 所需的最少有效平面数 | `autoflow/algorithms/pwv.py` |
| `scene_visible` | bool | `True` | `configs/pwv.json` | 分组 `PWV planes` 场景对象的初始显隐 | `autoflow/core/pipeline.py` |
| `scene_color` | string | `#ffd43b` | `configs/pwv.json` | 3D 视图中 PWV 平面的颜色 | `autoflow/core/pipeline.py` |
| `plot_color` | string | `#2b8a3e` | `configs/pwv.json` | 保存图和 GUI 图里的散点颜色 | `autoflow/algorithms/pwv.py`, `autoflow/ui/app.py` |
| `fit_color` | string | `#f08c00` | `configs/pwv.json` | 保存图和 GUI 图里的拟合线颜色 | `autoflow/algorithms/pwv.py`, `autoflow/ui/app.py` |
| `plot_dpi` | int | `160` | `configs/pwv.json` | PNG 分辨率 | `autoflow/algorithms/pwv.py` |

## 运行时说明

- 4D label mask 会在分组前先沿时间维度做多数决，压成 3D label。
- 小连通域清理是按每个 label 值分别做的，再合并 group。
- 每个 group 的血管 mask 在骨架化前还会再保留最大连通域。
- grouped 场景对象保留稳定的内部 data key，例如 `smooth_path_<group>_<idx>`、`plane_<group>_<idx>`、`pathline_<group>_<idx>`。
- GUI 里显示给用户的名字会更短，例如 `path 3`、`plane 5`、`pathline 5`。
- PWV 平面在场景里只暴露为一个 data key=`pwv_planes` 的分组对象，Browser 里显示为 `PWV planes`。

## 当前注意点

CLI 自动分割相关公共字段目前由 `AutoFlowConfig` 管，GUI 自动分割对话框初始值来自 segmentation config bundle。写文档和重构时必须明确区分这两条路径。
