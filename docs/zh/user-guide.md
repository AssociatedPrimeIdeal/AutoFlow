# 用户指南

## 先建立一个简单认识

AutoFlow 不是单纯的 DICOM 浏览器，而是一套从 4D Flow MRI 输入到可复核结果的工作流：

| 入口 | 适合谁 | 在这里做什么 |
| --- | --- | --- |
| GUI | 第一次使用、人工检查和修订 | 看图、选分割、编辑对象、拖动时间轴、导出结果 |
| CLI | 批处理、重复实验和留痕 | 指定输入、选择步骤、生成结构化输出目录 |
| Python API | 研究脚本和二次开发 | 组织配置、调用批处理、读取结果 |

三者使用同一套概念：输入 → 分割 → 中心线/图 → 截面 → 指标 → 审阅/导出。

## 安装与启动

```bash
pip install -e ".[gui]"
autoflow-gui --config-dir ./configs
```

只跑批处理：

```bash
pip install -e "."
autoflow-run case.h5 --output-dir ./results/case
```

## GUI 与 CLI 对照

| 阶段 | GUI 怎么用 | CLI 怎么用 | 主要结果 |
| --- | --- | --- | --- |
| 加载 | `File > Open H5` 或 `Import DICOM Directory` | 直接写输入路径 | 统一后的 `mag`、`flow`、几何和元数据 |
| 背景校正 | 没有可复用校正结果时在流程中选择 | `--bgc`；可加 `--bgc-method msac` | 校正后的速度、报告和缓存 |
| 相位解缠 | `Phase Unwrapping` 步骤 | `--phase-unwrap-method gc3D/lap4D/nprs` | 开启时的解缠速度场 |
| 分割 | `Segmentation` 工作区：导入、阈值、自动分割、编辑 | `--autoseg`；导入方式也可由 GUI/API 提供 | 后续几何使用的活动掩膜 |
| 中心线 | 依次运行 `Skeleton`、`Graph/Paths` | 批处理默认顺序执行 | 骨架、图、分支和路径 |
| 截面 | 设置数量、锚点、间距和方向 | `--plane-mode`、`--plane-count`、`--plane-spacing-*` | `planes.json`、`planes.h5` 和位置文件 |
| 基础指标 | `Hemodynamics` 中运行 plane metrics | 默认运行，可用 `--skip-plane-metrics` 跳过 | 每个截面的时序流量/速度 |
| 可选指标 | 在 `Hemodynamics` 选择 PWV/WSS/压力/涡旋 | `--with pwv,wss,tke,pg,vortex` | JSON、NPZ、H5、PNG |
| 动态查看 | 3D Browser、正交视图、时间轴、流线/路径线 | `--video plane,wss,tke,pg,streamlines` | MP4 和帧图 |
| 质量控制 | `Review & Export` | 读取 `summary.json`、QC JSON | 质量报告和平面 QC |

## 功能怎么用

### 输入与 QC

先打开病例，不要一开始就调算法参数。确认空间尺寸、phase 数、体素间距、VENC、空间轴标签和活动分割。输入可能是 complex image，也可能已经是 normalized `mag + flow`，详见[输入与输出](input-output.md)。

### 分割

内嵌分割只有在确实是你要分析的目标血管时才直接使用。否则可以导入外部分割、从 PC-MRA/magnitude 阈值生成、运行 nnUNet，或在 GUI 中编辑。自动分割是起点，不是无需检查的金标准。

### 骨架、图和路径

骨架把血管区域变成中心线；图描述拓扑；路径是截面、路径线和路径指标真正使用的对象。GUI 中先检查这些对象，再开始血流动力学计算。

### 截面与指标

截面沿路径放置，可以按距离或路径比例间隔。尽量避开分叉、掩膜末端和明显狭窄边界。先比较邻近截面和 `plane_qc.json`，再解释单个流量或速度数值。

### 血流动力学

PWV 使用配置好的组和波形；WSS 需要壁面和速度梯度；TKE 只在输入提供所需信息时计算，不能从速度大小伪造。压力模块可生成压力梯度和相对压力；涡旋模块提供 vorticity、Q-criterion 和 swirling strength。

### 流线、路径线和 PC-MRA

流线展示瞬时场，路径线跟踪粒子随时间的运动。GUI 适合交互式找种子点和调相机，CLI 适合重复导出视频。PC-MRA 是结构背景，不能替代速度、分割和 QC 检查。

## 导出视频

CLI 视频默认关闭，需要显式请求：

```bash
autoflow-run case.h5 \
  --output-dir ./results/case \
  --with wss,pg \
  --video plane,wss,pg,streamlines \
  --fps 12 \
  --camera-view right
```

GUI 对应 `Export > Export Videos...`，可以选择视频类型和渲染参数。首页展示了一个基于 Aorta 示例数据生成的动态 review 预览。

## Python API

```python
from autoflow import AutoFlowConfig, run_batch

config = AutoFlowConfig(
    inputs=["case.h5"],
    output_dir="./results/case",
    requested_metrics=["wss", "pg"],
    requested_videos=["plane", "wss"],
)
results, case_output = run_batch(config)
```

算法选择、配置归属和代码修改请看[开发者架构](../en/developer/architecture.md)与[功能到代码映射](../en/developer/feature-to-code-map.md)。
