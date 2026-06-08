# 功能：PWV

## 状态

| 入口 | 状态 | 说明 |
| --- | --- | --- |
| GUI | Supported | 运行平面指标后会在 `PWV` dock 中显示图 |
| CLI | Supported | 由 `configs/pwv.json` 配置驱动 |
| Python API | Supported | 由 `AutoFlowConfig.from_config_dir()` 读取配置驱动 |

## 功能说明
PWV 会针对一个或多个配置好的 label group 计算脉搏波传导速度。

对每个 PWV group，AutoFlow 会：

1. 把配置里的多个 label 合并成一个 mask
2. 只保留最大连通域
3. 对这个 grouped mask 做骨架化
4. 建图并找到最长路径
5. 按配置的毫米间距沿这条路径布置平面
6. 对这些 PWV 平面计算平面指标
7. 在每个平面上提取波形 foot time
8. 拟合 time-to-foot 对 slice position
9. 输出 PWV 并保存图

## 何时使用
- 需要沿某条血管树做 config 驱动的 PWV 测量时使用
- 一个 PWV 血管段由多个 segmentation label 表示时使用
- 已经有分割和流场后使用

## 快速使用

### GUI
1. 在 `configs/pwv.json` 里启用 PWV group
2. 载入带分割和流场的病例
3. 运行 `Calculate && Save Metrics`
4. 查看 `PWV` dock 和 Browser 里的 `PWV planes` 分组对象

### CLI

```bash
autoflow-run case.h5 --config-dir ./configs --output-dir results/case
```

### Python API

```python
from autoflow import AutoFlowConfig, run_case

config = AutoFlowConfig.from_config_dir("./configs")
summary = run_case("case.h5", config=config)
```

## 输入

| 输入 | 是否必需 | 含义 |
| --- | --- | --- |
| flow | 是 | 时间分辨速度场 |
| segmentation | 是 | 用来定义 PWV group 的 label mask |
| `configs/pwv.json` | 是 | PWV group 和测量参数 |
| `configs/labels.json` | 使用符号 label 名时需要 | 解析 `AAO`、`PV` 这类 label 名 |

## 参数

| 参数 | 类型 | 默认值 | 设置位置 | 作用 | 代码归属 |
| --- | --- | --- | --- | --- | --- |
| `enabled` | bool | `False` | `configs/pwv.json` | 是否启用 PWV 计算 | `autoflow/core/pipeline.py` |
| `groups` | list | `[]` | `configs/pwv.json` | 每个元素定义一个 PWV label 组合 | `autoflow/core/models.py` |
| `groups[].name` | string | 自动生成 | `configs/pwv.json` | 在结果和 GUI 中显示的名字 | `autoflow/core/models.py` |
| `groups[].labels` | list[int 或符号名] | 每组必填 | `configs/pwv.json` | 合并成一个 PWV mask 的 label 列表 | `autoflow/core/models.py` |
| `plane_interval_mm` | float | `10.0` | `configs/pwv.json` | PWV 平面间距 | `autoflow/algorithms/pwv.py` |
| `start_distance` | float | `0.0` | `configs/pwv.json` | 起点偏移 | `autoflow/algorithms/pwv.py` |
| `end_distance` | float | `0.0` | `configs/pwv.json` | 终点偏移 | `autoflow/algorithms/pwv.py` |
| `waveform_key` | string | `flowrate_signed_mL_s` | `configs/pwv.json` | 用于 foot detection 的波形字段 | `autoflow/algorithms/pwv.py` |
| `foot_savgol_window` | int | `5` | `configs/pwv.json` | 波形平滑窗口 | `autoflow/algorithms/pwv.py` |
| `foot_savgol_polyorder` | int | `2` | `configs/pwv.json` | 波形平滑多项式阶数 | `autoflow/algorithms/pwv.py` |
| `minimum_valid_planes` | int | `2` | `configs/pwv.json` | 拟合 PWV 所需最少有效平面数 | `autoflow/algorithms/pwv.py` |

## 输出

| 输出 | 何时生成 | 含义 |
| --- | --- | --- |
| `pwv.json` | 启用 PWV 且运行平面指标后 | 每个 PWV group 一个结果块 |
| `pwv_<group>.png` | 绘图成功后 | 每组的拟合图 |
| `summary.json -> pwv_results` | 写 summary 时 | PWV 结果摘要 |
| `PWV planes` 场景对象 | GUI 或 pipeline 中 PWV 成功后 | Browser 里控制全部 PWV 平面的单个分组对象 |

## 限制
- PWV 强依赖分割质量、图质量和最长路径启发式
- 当前没有单独的 CLI flag，完全由配置文件驱动
- 如果 foot detection 后有效平面太少，该 group 会被跳过

## 改代码时看哪里

| 想改什么 | 主文件 | 还要看 | 测试 |
| --- | --- | --- | --- |
| PWV 算法 | `autoflow/algorithms/pwv.py` | `autoflow/core/pipeline.py` | 手工验证加保留 smoke/phantom 套件 |
| PWV 场景对象注册 | `autoflow/core/pipeline.py` | `autoflow/ui/viewer.py` | GUI 手工验证 |
| PWV dock 和 plot | `autoflow/ui/app.py` | `autoflow/algorithms/pwv.py` | GUI 手工验证 |

## 测试
- `~/miniconda3/envs/ryy/bin/python -m pytest tests/test_smoke_phantoms.py tests/test_pressure_gradient_phantom.py -q`

## 常见问题

| 现象 | 常见原因 | 处理 |
| --- | --- | --- |
| 没有写出 PWV 结果 | `configs/pwv.json -> enabled` 为假，或没有配置 group | 启用 PWV 并定义 group |
| 某个 PWV group 被跳过 | group mask 为空，或有效平面数不够 | 检查分割 label 和波形质量 |
| Browser 里看不到每个 PWV plane 单独一项 | 这是预期行为 | 使用单个 `PWV planes` 项统一控制显隐 |
