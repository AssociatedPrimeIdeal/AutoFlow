# 输入

## 支持的输入类型

| 输入类型 | 支持情况 | 常见入口 | 说明 |
| --- | --- | --- | --- |
| 传统 complex H5 | 支持 | CLI、GUI、Python API | 可能自带分割和 complex 导出的 sigma |
| 归一化 `mag + flow` H5 | 支持 | CLI、GUI、Python API | 当前推荐的标准化路径 |
| 直接 DICOM 目录 | 支持 | CLI、GUI、Python API | 先扫描可导入病例 |

## legacy complex H5 变体

- 经典 legacy complex H5 使用 `img_complex[..., 0]` 作为 magnitude 参考，`img_complex[..., 1:4]` 作为三个速度编码通道
- 当 `img_complex.shape[-1] == 7` 时，AutoFlow 支持 legacy dual-venc H5，要求文件按 `VENCOrder` 存储 1 个参考通道加 6 个速度编码通道
- 对 `Nv=7`，AutoFlow 会把 `1:4` 解释为 low-venc 三通道，把 `4:7` 解释为 high-venc 三通道，分别做背景相位校正，然后执行 dual-venc 去混叠重建，最后把重建结果作为最终 `flow`
- `Nv=7` 的最终 loader `venc` 使用 high-venc 三元组
- dual-venc 阈值来自 `configs/loader.json -> background_phase_correction -> dual_venc_ratio1` 和 `dual_venc_ratio2`；如果两者都保持 `0.0`，AutoFlow 会根据 high/low venc 比值自动推导

## Loader 归一化结果

AutoFlow 会把 loader 输出统一成 `LoadedCase`。

### 必需字段

| 字段 | 含义 |
| --- | --- |
| `mag` | 幅值体数据 |
| `flow` | 速度场 |
| `resolution` | 体素间距 |
| `origin` | 世界坐标原点 |
| `venc` | 速度编码 |
| `rr` | RR 间期 |

### 可选字段

| 字段 | 含义 |
| --- | --- |
| `segmentation` | 内嵌或外部载入的分割；可以是 binary、3D label 或 4D label mask |
| `tke_array` | 可选 TKE 数组 |
| `sigma` | 可选 complex 导出 sigma |
| `metadata` | 源数据元信息 |
| `source_format` | 输入格式标记 |
| `source_group` | H5 顶层 group 名称 |
| `capabilities` | 明确的下游能力标记 |

### `capabilities` 字段

| 字段 | 含义 |
| --- | --- |
| `has_segmentation` | 当前病例有可用分割 |
| `has_tke` | 当前病例有 TKE |
| `has_complex_source` | 当前病例保留了 complex 来源信息 |
| `supports_wss` | 有分割时可以计算 WSS |
| `supports_plane_metrics` | 有平面和分割时可以计算平面指标 |

## DICOM 说明

- 直接 DICOM 读取逻辑在 `autoflow/algorithms/dicom.py`
- Siemens 有符号 phase velocity 通道会在 direct load 时恢复成物理速度
- `IN`、`TH` 之类文本方向标签会参与速度方向识别
- 只有 `mag + flow` 的 DICOM 输入不会伪造 TKE

## 各类输入的分割预期

| 输入类型 | 内嵌分割 | 外部分割 | 自动分割 |
| --- | --- | --- | --- |
| 传统 complex H5 | 有则支持 | 支持 | 支持 |
| 归一化 H5 | 有则支持 | 支持 | 支持 |
| 直接 DICOM | 通常没有 | 支持 | 支持 |

## grouped multi-label 分割行为

- 后续血管步骤支持 binary mask、3D label mask 和 4D label mask
- 4D label mask 会在骨架、图、平面、路径线步骤前，先沿时间维度做多数决得到 3D label
- 连通域清理是按每个 label 值分别做的，再构建 group
- label 值通过 `configs/labels.json -> label_map` 和 `label_groups` 合并成 group
- 每个 group 的血管 mask 在骨架化前会再保留最大连通域
- 每个 group 可以通过 `configs/labels.json -> label_groups.<group>.preprocess` 单独覆盖高斯平滑、膨胀、腐蚀、开运算、闭运算
- 如果前景里最终只有一个 label，AutoFlow 会自动退化成单 group 流程，所以 single-label 分割不需要单独适配

## 代码参考

- H5 读取：`autoflow/algorithms/data.py`
- DICOM 扫描和载入：`autoflow/algorithms/dicom.py`
- pipeline 载入步骤：`autoflow/core/pipeline.py`
- 类型定义：`autoflow/case_types.py`
