# 快速开始

## 目标
这页给的是 AutoFlow 最短可用路径。

## 安装

完整本地安装，包含 GUI 和测试：

```bash
pip install -e ".[gui,test]"
```

如果只需要 CLI：

```bash
pip install .
```

仓库里已知可用的测试环境：

```bash
~/miniconda3/envs/ryy/bin/python -m pytest tests/test_smoke_phantoms.py tests/test_pressure_gradient_phantom.py -q
```

## 用 CLI 跑一个 H5

```bash
autoflow-run ./data/demo_data.h5 --output-dir ./results/demo
```

当前标准批处理顺序：

1. 载入数据
2. 生成骨架
3. 生成图
4. 生成平面
5. 计算平面指标
6. 计算派生指标
7. 导出已启用的视频

典型输出目录 `./results/demo/<case_name>/` 下会有：

- `planes.json`
- `plane_positions.json`
- `plane_metrics.json`
- `plane_qc.json`
- `summary.json`

## 用 CLI 跑一个 DICOM 根目录

```bash
autoflow-run /path/to/dicom_root --output-dir ./results/dicom_batch
```

## 打开 GUI

```bash
autoflow-gui
```

典型 GUI 流程：

1. `File > Open H5` 或 `File > Import DICOM Directory`
2. 在 3D 视图和 ortho viewer 里检查数据
3. 如有需要，选择或生成分割
4. 点击 `Run All`
5. 查看平面、指标和派生体数据

## 从 Python 调用

```python
from autoflow import AutoFlowConfig, run_batch

config = AutoFlowConfig(
    inputs=["./data/demo_data.h5"],
    output_dir="./results/demo",
)
results, last_case_out = run_batch(config)
```

## 常见起步命令

### 按距离生成平面

```bash
autoflow-run ./data/demo_data.h5 \
  --output-dir ./results/demo \
  --plane-by-distance \
  --cross-section-dist 15 \
  --start-dist 5 \
  --end-dist 0
```

### 输入没有分割时启用自动分割

```bash
autoflow-run ./data/demo_data.h5 \
  --output-dir ./results/demo \
  --autoseg
```

### 导出视频

```bash
autoflow-run ./data/demo_data.h5 \
  --output-dir ./results/demo \
  --plane-video \
  --wss-video \
  --streamlines-video
```
