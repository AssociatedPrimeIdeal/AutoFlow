# 功能：分割

## 状态

| 入口 | 状态 | 说明 |
| --- | --- | --- |
| GUI | Supported | 支持原始、导入、阈值、自动、手工编辑 |
| CLI | Supported | `--autoseg` 只会在输入没有分割时触发 |
| Python API | Partial | 可通过 `AutoFlowConfig` 驱动，但不是独立高层编辑 API |

## 功能说明
分割提供血管腔体 mask，是骨架、图、平面、平面指标、WSS、流线、路径线的前提。

## 何时使用
- 输入没有可用血管 mask 时使用
- 已有外部分割时优先用导入分割
- 需要快速生成粗分割时用阈值分割
- 有 nnUNet 模型且希望直接落地可复用 sidecar 时用自动分割

## 快速使用

### GUI
1. 载入病例
2. 打开 `Segmentation > Configure Segmentation...`
3. 选择 `input`、`threshold` 或 `auto`
4. 应用配置；如果选了 `auto`，等待自动分割进度框完成
5. 如需修正，在 segmentation dock 编辑后点击 `Apply`

### CLI

```bash
autoflow-run case.h5 --output-dir results/case --autoseg
```

### Python API

```python
from autoflow import AutoFlowConfig, run_case

config = AutoFlowConfig(
    output_dir="./results/case",
    autoseg=True,
    autoseg_model="/path/to/model",
)
summary = run_case("case.h5", config=config)
```

## 输入

| 输入 | 是否必需 | 含义 |
| --- | --- | --- |
| `mag` | 阈值或自动分割时必需 | 分割生成输入 |
| `flow` | 自动分割时必需 | nnUNet 输入通道来源 |
| 内嵌分割 | 可选 | 原始分割来源 |
| 外部分割文件 | 可选 | 导入分割来源 |

## 参数

| 参数 | 类型 | 默认值 | 设置位置 | 作用 |
| --- | --- | --- | --- | --- |
| `mode` | string | `input` | GUI 分割配置 | 选择 `input`、`threshold`、`auto` |
| `input_source` | string | `original` | GUI 分割配置 | 选择原始或导入分割 |
| `threshold_scalar` | string | `pcmra` | `configs/segmentation.json` | 阈值分割使用的标量 |
| `threshold_keep_largest_cc` | bool | `True` | `configs/segmentation.json` | 只保留最大连通域 |
| `threshold_min_component_volume_mm3` | float | `0.0` | `configs/segmentation.json` | 去掉过小连通域 |
| `threshold_closing` | bool | `True` | `configs/segmentation.json` | 闭运算 |
| `threshold_opening` | bool | `False` | `configs/segmentation.json` | 开运算 |
| `auto_backend` / `--autoseg-backend` | string | `nnUNet` | GUI 或 CLI/API | 自动分割后端 |
| `auto_model` / `--autoseg-model` | path | GUI 默认是仓库内模型；CLI/API 默认先为空再解析 | GUI 或 CLI/API | 模型目录 |
| `auto_checkpoint` / `--autoseg-checkpoint` | string | `checkpoint_final.pth` | GUI 或 CLI/API | checkpoint 名称 |
| `auto_device` / `--autoseg-device` | string | `auto` | GUI 或 CLI/API | 设备选择 |
| `auto_label_map` / `--autoseg-label-map` | JSON | 空 | GUI 或 CLI/API | 标签重映射 |
| `edit_all_timepoints` | bool | `True` | segmentation dock | 是否按 3D-all-frames 编辑 |

## 输出

| 输出 | 何时生成 | 含义 |
| --- | --- | --- |
| 工作区中的活动分割来源 | 每次分割来源切换后 | 后续步骤使用的当前分割 |
| `*_threshold_segmentation.h5` | 阈值分割成功后 | 阈值分割 sidecar |
| `*_auto_segmentation.h5` | 自动分割成功后 | GUI 和 CLI 都会保存的自动分割 sidecar |
| 用户选择的 H5、NPY、NPZ | 手工保存时 | 导出当前活动分割 |

## 限制
- 当前实现的自动分割后端是 `nnUNet`
- 自动分割要求已经载入 `mag` 和 `flow`
- GUI 自动分割会放到后台线程执行，并持续显示进度框直到推理和 sidecar 保存完成
- CLI 自动分割只会在输入没有分割时触发
- GUI 的 `Run All` 不会自动开始分割

## 改代码时看哪里

| 想改什么 | 主文件 | 还要看 | 测试 |
| --- | --- | --- | --- |
| 新增分割来源 | `autoflow/core/models.py`, `autoflow/ui/app.py` | `autoflow/ui/segmentation.py`, `autoflow/algorithms/segmentation.py` | `tests/test_smoke_phantoms.py` |
| 改阈值分割 | `autoflow/algorithms/segmentation.py` | `autoflow/ui/segmentation.py` | `tests/test_smoke_phantoms.py` |
| 改 nnUNet 自动分割 | `autoflow/algorithms/segmentation.py` | `autoflow/processing.py`, `autoflow/ui/app.py` | 仅保留 smoke/phantom 回归，其他靠手工验证 |

## 测试

- `~/miniconda3/envs/ryy/bin/python -m pytest tests/test_smoke_phantoms.py -q`
- `~/miniconda3/envs/ryy/bin/python -m pytest tests/test_pressure_gradient_phantom.py -q`
- 分割专项行为超出这两份回归时，改动后主要靠手工验证
