<div class="af-hero af-hero--split" markdown>

<div markdown>

# AutoFlow

## A 4D Flow MRI workbench for review, analysis and reproducible export

从医学影像检查到血流动力学结果，AutoFlow 把数据检查、分割、中心线、截面、指标和视频导出放进同一条可审阅工作流。

[中文文档](zh/user-guide.md){ .md-button .md-button--primary }
[English docs](en/user-guide.md){ .md-button }

</div>

<div class="af-hero-stat" markdown>

**GUI**

Inspect and correct

**CLI**

Batch and reproduce

**Python**

Integrate and extend

</div>

</div>

## Exported video preview

这是用当前仓库的 Aorta 示例数据生成的动态 review 素材：它展示了 magnitude、分割包络、速度大小和 phase 播放。实际 CLI 视频还可以导出截面、WSS、TKE、压力梯度和流线等专用视图。

<video class="af-video" controls autoplay muted loop playsinline poster="assets/images/demo/flow-segmentation.png">
  <source src="assets/video/aorta-flow-demo.mp4" type="video/mp4">
  <img src="assets/video/aorta-flow-demo.gif" alt="AutoFlow Aorta dynamic review preview">
</video>

<p class="af-note">The source H5 is not shipped with the repository. The preview is a derived documentation asset, not a clinical result.</p>

## What AutoFlow covers

<div class="af-grid" markdown>

<div class="af-card" markdown>
**Load**

Legacy complex H5, normalized `mag + flow` H5, and direct DICOM directory input.
</div>

<div class="af-card" markdown>
**Prepare**

Axis/VENC normalization, optional background phase correction, optional phase unwrapping, and multiple segmentation sources.
</div>

<div class="af-card" markdown>
**Analyze**

Skeleton, graph, branch/path, planes, flow metrics, PWV, WSS, TKE, pressure and vortex quantities.
</div>

<div class="af-card" markdown>
**Review & export**

3D/ortho views, streamlines, pathlines, PC-MRA rendering, quality reports, JSON/H5/NPZ/PNG/video.
</div>

</div>

## Choose a path

| Need | Start here |
| --- | --- |
| 我是第一次使用 / I am new to 4D Flow | [中文用户指南](zh/user-guide.md) · [English User Guide](en/user-guide.md) |
| 我想照着真实病例跑一遍 / I want a complete example | [中文示例](zh/example.md) · [English example](en/example.md) |
| 我不确定 H5 里应该放什么 / I need the input contract | [中文输入输出](zh/input-output.md) · [English Input & Output](en/input-output.md) |
| 我想改算法或扩展模块 / I want to modify code | [Developer architecture](en/developer/architecture.md) · [Feature-to-code map](en/developer/feature-to-code-map.md) |

## Positioning

如果你熟悉 CVI 等软件：AutoFlow 的 GUI 负责“看”和“改”，CLI 负责“批量跑”和“留痕”，Python API 负责“嵌入研究脚本”。它不是 DICOM viewer 的替代品，也不是临床诊断工具；最终结果仍需要按照扫描协议、分割质量和研究方案进行人工复核。
