# 输入与输出

## 软件最少需要什么

AutoFlow 会把输入统一成 `mag`、`flow`、`resolution`、`origin`、`venc` 和 `rr`。分割、背景校正、sigma 和 TKE 都是可选能力。

| 输入类型 | 必需数据 | 常见形状 | 说明 |
| --- | --- | --- | --- |
| legacy complex H5 | `img_complex` 或 complex `img`、`Resolution`、`VENC`、`RR` | `(X,Y,Z,T,4)` | 通道 0 是 magnitude，通道 1–3 是速度编码 |
| normalized H5 | `mag`、`flow`、`Resolution`、`Origin`、`VENC`、`RR` | `mag=(X,Y,Z,T)`、`flow=(X,Y,Z,T,3)` | 上游已解码时推荐 |
| real `img` | `img` 和元数据 | `(X,Y,Z,T,4)` 或 channel-first | 按 magnitude + 三个 flow 分量解释 |
| dual-VENC complex H5 | 7 通道 complex `img_complex` 和 `VENCOrder` | `(X,Y,Z,T,7)` | 两组三方向编码会被配对重建 |
| DICOM 目录 | 可读的速度/幅度序列和 DICOM 几何 | 由厂商决定 | 扫描目录并自动解析病例 |

加载器会对常见字段名不区分大小写，也能从 H5 的嵌套 group 中识别病例；同一文件中多个可识别 group 可以生成多个 case。

## complex image 怎么解释

```text
img_complex[..., 0]  -> magnitude 参考
img_complex[..., 1]  -> 速度编码分量 1
img_complex[..., 2]  -> 速度编码分量 2
img_complex[..., 3]  -> 速度编码分量 3
```

复数通道本身还不是最终的带符号速度场。AutoFlow 使用 `VENC` 和 `VENCOrder` 统一速度分量，并生成标准三分量 `flow`。

## normalized `mag + flow`

```text
mag   : (X, Y, Z, T)
flow  : (X, Y, Z, T, 3)
```

`flow` 的最后一维是三个速度分量。物理单位必须和上游转换过程及元数据一致，不能根据 `.h5` 扩展名猜单位。

## 元数据

| 字段 | 医学/工程含义 | 为什么重要 |
| --- | --- | --- |
| `Resolution` | 三个体素间距，通常为 mm | 把 voxel 换算成物理距离，影响面积和梯度 |
| `Origin` | 体数据原点 | 将数组坐标映射到世界坐标 |
| `SpatialOrder` | `FH`、`RL`、`AP` 等标签 | 统一空间轴和方向符号 |
| `VENC` | 一个或三个速度编码上限 | 决定速度标尺和 dual-VENC 配对 |
| `VENCOrder` | 速度编码分量标签 | 把通道映射到标准分量 |
| `RR` | R-R 间期，常见单位为 ms | 计算 phase 时间间隔 |

三元数据可以是一维数组、单行或单列；字段名大小写不敏感，下划线和连字符也能被容忍。

## 可选字段与能力

- `segmask`、`segmentation` 或 `seg`：存在时可作为内嵌活动分割。
- `corr`，或 dual-VENC 的 `corr_low` / `corr_high`：可复用的背景相位校正。
- complex 输入可能提供 TKE 路径所需的 sigma 信息。
- 只有 `mag + flow` 也能运行骨架、图、截面、截面指标、WSS 和流线。
- 如果输入没有 TKE 所需信息，AutoFlow 会跳过 TKE，不会从速度大小伪造 TKE。

## 输出文件

| 文件 | 内容 | 用途 |
| --- | --- | --- |
| `summary.json` | 阶段耗时和请求标记 | 确认批处理执行了什么 |
| `quality_report.json` | 输入、分割、拓扑、平面、流量一致性和 PWV 检查 | 解释数值前先审阅 |
| `planes.json` / `planes.h5` | 截面几何和序列化对象 | 保存、重载和共享截面 |
| `plane_positions.json` | 可移植的截面坐标 | 在另一轮运行中复用 |
| `plane_metrics.json` | phase 级和汇总截面指标 | 表格和绘图 |
| `plane_qc.json` | 每个截面的 QC | 发现可疑截面 |
| `pwv.json`、PNG | PWV 组、波形和图 | PWV 分析 |
| NPZ/H5 | WSS、压力、涡旋和其它体数据 | 下游数值分析 |
| MP4/PNG | 动态或静态渲染 | 审阅、汇报和演示 |

所有结果写入 `--output-dir`。包含多个数据 group 的 H5 会为每个识别到的 case 创建独立目录。
