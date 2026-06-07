# 功能：离线视频

## 状态

| 入口 | 状态 | 说明 |
| --- | --- | --- |
| CLI | Supported | 主视频导出入口 |
| Python API | Supported | 和 CLI 共享同一渲染路径 |
| GUI | Partial | GUI 更偏交互查看，不是主视频导出路径 |

## 功能说明
离线视频会导出 planes、WSS、TKE、streamlines 的 MP4 文件。

## 何时使用
- 做报告、展示、回顾时使用
- 需要稳定可复现导出时优先用 CLI 或 Python API
- 上游数据还没准备好时不要先跑视频

## 快速使用

### CLI

```bash
autoflow-run case.h5 \
  --output-dir results/case \
  --plane-video \
  --wss-video \
  --streamlines-video
```

### Python API

```python
from autoflow import AutoFlowConfig, run_case
config = AutoFlowConfig(
    output_dir="./results/case",
    make_plane_video=True,
    make_wss_video=True,
)
summary = run_case("case.h5", config=config)
```

### GUI
先在 GUI 中检查视角、内容和结果，再用 CLI 或 Python API 导出视频。

## 输入

| 输入 | 是否必需 | 含义 |
| --- | --- | --- |
| planes | plane video 时必需 | 平面对象 |
| derived metrics | WSS 或 TKE 视频时必需 | 派生数据 |
| flow 和 segmentation | streamlines video 时必需 | 流线源数据 |

## 参数

| 参数 | 类型 | 默认值 | 设置位置 | 作用 |
| --- | --- | --- | --- | --- |
| `make_plane_video` / `--plane-video` | bool | `False` | `configs/rendering.json` | 导出平面视频 |
| `make_wss_video` / `--wss-video` | bool | `False` | `configs/rendering.json` | 导出 WSS 视频 |
| `make_streamlines_video` / `--streamlines-video` | bool | `False` | `configs/rendering.json` | 导出流线视频 |
| `make_tke_video` / `--tke-video` | bool | `False` | `configs/rendering.json` | 导出 TKE 视频 |
| `fps` / `--fps` | int | `12` | `configs/rendering.json` | 帧率 |
| `plane_rotation_frames` / `--plane-rotation-frames` | int | `180` | `configs/rendering.json` | 平面旋转帧数 |
| `camera_view` / `--camera-view` | string | `right` | `configs/rendering.json` | 视角预设 |
| `camera_distance_scale` / `--camera-distance-scale` | float | `1.5` | `configs/rendering.json` | 相机距离缩放 |
| `rotate_dynamic_video` | bool | `True` | `configs/rendering.json` | 动态视频是否旋转 |
| `dynamic_rotation_frames` | int | `180` | `configs/rendering.json` | 动态旋转帧数 |
| `dynamic_rotation_elevation_deg` | float | `10.0` | `configs/rendering.json` | 动态旋转抬升角 |
| `dynamic_time_repeat` | int | `3` | `configs/rendering.json` | 每个时间点重复次数 |
| `add_plane_idx` | bool | `False` | `configs/rendering.json` | 标注平面编号 |
| `add_path_idx` | bool | `False` | `configs/rendering.json` | 标注 path 编号 |

## 输出

| 输出 | 何时生成 | 含义 |
| --- | --- | --- |
| `planes_rotate.mp4` | 启用 plane video 时 | 平面旋转总览 |
| `wss_video.mp4` 或 `wss_rotate.mp4` | 启用 WSS video 时 | WSS 视频 |
| `streamlines_video.mp4` 或 `streamlines_rotate.mp4` | 启用 streamlines video 时 | 流线视频 |
| `tke_video.mp4` 或 `tke_rotate.mp4` | 启用 TKE video 且有 TKE 时 | TKE 视频 |

## 限制
- 上游数据没准备好时视频不会成功
- 没有 TKE 时不会有 TKE 视频
- 当前没有单独的批量路径线视频导出特性

## 改代码时看哪里

| 想改什么 | 主文件 | 还要看 | 测试 |
| --- | --- | --- | --- |
| 视频渲染逻辑 | `autoflow/rendering/videos.py` | `autoflow/processing.py` | `tests/test_smoke_phantoms.py` |
| CLI 或 API 视频参数 | `autoflow/cli.py`, `autoflow/api.py` | `autoflow/config.py` | `tests/test_smoke_phantoms.py` |

## 测试

- `~/miniconda3/envs/ryy/bin/python -m pytest tests/test_smoke_phantoms.py -q`

## 常见问题

| 现象 | 常见原因 | 处理 |
| --- | --- | --- |
| 没生成视频 | 对应 `make_*_video` 没开 | 打开对应开关 |
| 没有 TKE 视频 | 没有 TKE 数据 | 这是预期行为 |
