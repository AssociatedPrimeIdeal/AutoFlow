# 首次运行：从零开始

## 你要完成什么

这页假设你没有使用过 4D Flow MRI 软件。目标不是立刻得到“漂亮的图”，而是先建立一个可解释的最小闭环：安装 → 打开数据 → 确认几何和单位 → 准备分割 → 生成中心线和截面 → 审阅输出。

## 1. 安装

在项目根目录创建或激活一个 Python 环境。GUI、自动分割和测试建议使用完整安装：

```bash
pip install -e ".[gui,test]"
```

只做 CLI 批处理时可以使用：

```bash
pip install -e "."
```

如果需要外部 SpatioTemporal Labeler：

```bash
git submodule update --init --recursive
pip install -e ".[gui,labeler]"
```

安装后应能看到两个命令：

```bash
autoflow-run --help
autoflow-gui --help
```

## 2. 先区分三个概念

| 概念 | 医学用户可以怎样理解 | AutoFlow 中的作用 |
| --- | --- | --- |
| `mag` / magnitude | 解剖结构和信号强度图 | 检查覆盖范围、构造 PC-MRA、辅助分割 |
| `flow` / velocity | 每个体素的三方向速度信息 | 流量、速度、WSS、压力、流线等分析 |
| `seg` / segmentation | 你希望纳入分析的血管区域 | 限定骨架、路径、截面和指标的空间范围 |

`corr` 是背景相位校正缓存，不是“另一张分割图”。它用于估计并去除静止组织带来的速度偏置。没有它并不意味着数据不能打开，但冷启动分析通常应该明确运行校正并记录参数。

## 3. 选择入口

=== "第一次使用：GUI"

    ```bash
    autoflow-gui --config-dir ./configs
    ```

    推荐顺序：

    1. `File > Open H5` 或 `File > Import DICOM Directory`。
    2. 在 `Input & QC` 检查空间方向、分辨率、VENC、心动周期和时间帧数。
    3. 若没有 `corr`，在校正步骤中选择是否运行背景相位校正。
    4. 打开 `Segmentation` 工作区，导入、阈值生成、自动生成或手动修订掩膜。
    5. 运行 `Skeleton`、`Graph/Paths` 和 `Planes`，确认中心线没有穿出血管。
    6. 在 `Hemodynamics` 选择需要的指标，最后查看 `Review & Export`。

=== "批处理：CLI"

    最小命令：

    ```bash
    autoflow-run /path/to/case.h5 --output-dir ./results/case
    ```

    冷启动、显式要求背景相位校正和自动分割：

    ```bash
    autoflow-run /path/to/case.h5 \
      --output-dir ./results/case \
      --bgc \
      --autoseg \
      --with pwv,wss,pg,vortex \
      --video plane,wss,pg
    ```

    > **建议**：第一次先不要加所有派生指标。先确认 segmentation、skeleton 和 planes，避免把几何问题与压力/WSS 等计算问题混在一起。

=== "研究脚本：Python API"

    ```python
    from autoflow import AutoFlowConfig, run_batch

    config = AutoFlowConfig(
        inputs=["/path/to/case.h5"],
        output_dir="./results/case",
        background_phase_correction=True,
        autoseg=True,
    )
    results, last_case_output = run_batch(config)
    ```

## 4. 第一次检查什么

不要只看最终数值，至少检查：

- **方向**：左右、前后、头脚方向是否和你的阅片习惯一致；显示方向和数据坐标是两个概念。
- **分辨率**：`Resolution` 是否以 mm 表示，是否被正确读取；压力和 WSS 对空间尺度很敏感。
- **VENC**：三个速度编码方向是否合理；VENC 错误会直接影响速度量级。
- **分割**：分割是否覆盖目标血管，是否包含不想分析的分支或断裂。
- **时间**：心动周期 `RR` 和 phase 数是否真实；时间间隔会影响加速度、PWV 和视频。
- **缓存**：首次冷启动时，输出日志和 `summary.json` 应能说明是否运行了校正和分割。

## 5. 下一步

- 输入不确定：看[输入格式](inputs.md)。
- 没有分割：看[分割与修订](../features/segmentation.md)。
- 想复现示例：看[冷启动示例](demo-case.md)。
- 想知道文件怎么解释：看[输出文件](outputs.md)。
