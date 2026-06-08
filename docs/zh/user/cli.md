# CLI 指南

## 状态
`autoflow-run` 是主批处理入口，当前是支持状态。

核心入口文件：

- `autoflow/cli.py`
- `autoflow/api.py`
- `autoflow/processing.py`
- `autoflow/core/pipeline.py`

## 按任务使用

### 跑一个 H5

```bash
autoflow-run ./data/demo_data.h5 --output-dir ./results/demo
```

### 跑一个 DICOM 根目录

```bash
autoflow-run /path/to/dicom_root --output-dir ./results/dicom_batch
```

### 复用已有平面

```bash
autoflow-run ./data/demo_data.h5 \
  --output-dir ./results/demo \
  --reuse-planes ./results/old_case/plane_positions.json
```

### 输入没有分割时自动分割

```bash
autoflow-run ./data/demo_data.h5 \
  --output-dir ./results/demo \
  --autoseg
```

### 按距离布置平面

```bash
autoflow-run ./data/demo_data.h5 \
  --output-dir ./results/demo \
  --plane-by-distance \
  --cross-section-dist 15
```

### 导出离线视频

```bash
autoflow-run ./data/demo_data.h5 \
  --output-dir ./results/demo \
  --plane-video \
  --wss-video \
  --streamlines-video
```

## 当前执行顺序

`process_single()` 当前顺序是：

1. `load_data`
2. 如果启用了 `--autoseg` 且输入没有分割，则先跑 CLI 自动分割
3. `Generate Skeleton`
4. `Generate Graph`
5. `Generate Planes`
6. 平面指标
7. 派生指标导出
8. 已启用的视频导出

行为说明：

- 没有分割时，依赖分割的步骤会被跳过
- 启用 `--autoseg` 且输入无分割时，会在骨架和图之前先补自动分割
- CLI 自动分割会打印 backend/model/device、阶段进度，以及推理和 sidecar 保存耗时
- 没有 TKE 时，不会伪造 TKE；能算的 WSS 和压力梯度仍会继续
- 如果 `configs/pwv.json -> enabled` 为真，平面指标阶段还会额外计算 PWV，并写出 `pwv.json` 和每组 PNG 图

## 参数表

### 输入和输出

| CLI 参数 | 类型 | 默认值 | 配置位置 | 作用 | 代码位置 |
| --- | --- | --- | --- | --- | --- |
| `inputs` | 路径列表 | 必填 | 命令行 | 要处理的文件或目录 | `autoflow/cli.py`, `autoflow/processing.py` |
| `--output-dir` | 路径 | `./results` | `AutoFlowConfig.output_dir` | 输出根目录 | `autoflow/api.py` |
| `--config-dir` | 路径 | 有仓库 `configs/` 时优先用它 | 命令行 | 载入模块级 JSON 默认值 | `autoflow/config.py` |
| `--reuse-planes` | 路径 | 空 | `configs/batch.json` 或命令行 | 复用保存过的平面位置 | `autoflow/plane_io.py` |

### 批处理和跳过行为

| CLI 参数 | 类型 | 默认值 | 配置位置 | 作用 | 代码位置 |
| --- | --- | --- | --- | --- | --- |
| `--skip-derived` | bool | `False` | `configs/batch.json` | 跳过 WSS、TKE、压力梯度导出 | `autoflow/processing.py` |
| `--skip-plane-metrics` | bool | `False` | `configs/batch.json` | 跳过平面指标导出 | `autoflow/processing.py` |
| `--single-thread` | bool | 默认多线程 | `configs/batch.json` | 关闭平面指标多线程 | `autoflow/core/pipeline.py` |

### 载入和背景相位校正

| CLI 参数 | 类型 | 默认值 | 配置位置 | 作用 | 代码位置 |
| --- | --- | --- | --- | --- | --- |
| `--bgc` | bool | `False` | `configs/loader.json` | 开启背景相位校正 | `autoflow/algorithms/phase_correction.py` |
| `--bgc-fit-order` | int | `3` | `configs/loader.json` | 背景相位拟合阶数 | `autoflow/algorithms/phase_correction.py` |
| `--bgc-threshold` | float | `0.1` | `configs/loader.json` | 背景相位掩膜阈值 | `autoflow/algorithms/phase_correction.py` |
| `--dual-venc-ratio1` | float | `0.0` | `configs/loader.json` | legacy `Nv=7` H5 的第一段 dual-venc 去混叠阈值比率 | `autoflow/algorithms/data.py` |
| `--dual-venc-ratio2` | float | `0.0` | `configs/loader.json` | legacy `Nv=7` H5 的第二段 dual-venc 去混叠阈值比率 | `autoflow/algorithms/data.py` |
| `--dicom-read-workers` | int | `1` | `configs/loader.json` | DICOM 读取线程数；`0` 表示自动 | `autoflow/algorithms/dicom.py` |

### 平面生成

| CLI 参数 | 类型 | 默认值 | 配置位置 | 作用 | 代码位置 |
| --- | --- | --- | --- | --- | --- |
| `--plane-by-distance` | bool | 默认中心平面模式开启 | `configs/planes.json` | 切换到按距离均匀布面 | `autoflow/algorithms/planes.py` |
| `--cross-section-dist` | float mm | `5.0` | `configs/planes.json` | 距离模式平面间距 | `autoflow/algorithms/planes.py` |
| `--start-dist` | float mm | `5.0` | `configs/planes.json` | 起点偏移 | `autoflow/algorithms/planes.py` |
| `--end-dist` | float mm | `0.0` | `configs/planes.json` | 终点偏移 | `autoflow/algorithms/planes.py` |

### PWV

当前 PWV 由配置驱动，而不是单独的 CLI flag。

- 在 `configs/pwv.json` 里启用
- 在 `configs/pwv.json -> groups` 里定义一个或多个 PWV group
- 批处理运行时会在平面指标阶段自动计算 PWV

### 自动分割

| CLI 参数 | 类型 | 默认值 | 配置位置 | 作用 | 代码位置 |
| --- | --- | --- | --- | --- | --- |
| `--autoseg` | bool | `False` | 命令行 | 只有输入无分割时才自动分割 | `autoflow/processing.py` |
| `--autoseg-backend` | string | `nnUNet` | `AutoFlowConfig` | 自动分割后端 | `autoflow/algorithms/segmentation.py` |
| `--autoseg-model` | 路径 | 空字符串，之后尝试解析为仓库内默认模型 | `AutoFlowConfig` | nnUNet 模型目录 | `autoflow/algorithms/segmentation.py` |
| `--autoseg-checkpoint` | string | `checkpoint_final.pth` | `AutoFlowConfig` | checkpoint 名称 | `autoflow/algorithms/segmentation.py` |
| `--autoseg-device` | string | `auto` | `AutoFlowConfig` | `auto`、`cpu`、`cuda` | `autoflow/algorithms/segmentation.py` |
| `--autoseg-label-map` | JSON 字符串 | 空 | `AutoFlowConfig` | 预测标签重映射 | `autoflow/algorithms/segmentation.py` |

注：

- 当前 CLI 的自动分割默认值来自 `AutoFlowConfig`
- GUI 的自动分割初始值来自 `configs/segmentation.json`
