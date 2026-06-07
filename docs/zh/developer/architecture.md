# 架构

## 目的
这页用于让维护者快速定位主要职责在哪一层。

## 主要层次

| 层 | 目录 | 职责 |
| --- | --- | --- |
| 公开入口 | `autoflow/` | CLI、GUI launcher、公开 Python API |
| pipeline 编排 | `autoflow/core` | workspace 状态和步骤编排 |
| 算法层 | `autoflow/algorithms` | loader、预处理、图、平面、指标、分割、流线 |
| GUI | `autoflow/ui` | Qt 主窗口、dock、viewer、dialog、editor |
| 渲染 | `autoflow/rendering` | 离线视频 |
| 测试 | `tests` | 行为覆盖 |
| 配置 | `configs`, `autoflow/config.py` | 模块级 JSON 默认值 |
