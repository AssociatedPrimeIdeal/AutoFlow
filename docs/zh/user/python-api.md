# Python API

## 状态
公开 Python API 入口在 `autoflow/__init__.py` 和 `autoflow/api.py`。

公开对象：

- `AutoFlowConfig`
- `build_workspace()`
- `run_case()`
- `run_batch()`
- `launch_gui()`

## 快速用法

### 跑单个病例

```python
from autoflow import AutoFlowConfig, run_case

config = AutoFlowConfig(output_dir="./results/demo")
summary = run_case("./data/demo_data.h5", config=config)
```

### 跑批处理

```python
from autoflow import AutoFlowConfig, run_batch

config = AutoFlowConfig(
    inputs=["./data/demo_data.h5"],
    output_dir="./results/demo",
)
results, last_case_out = run_batch(config)
```

### 从配置目录构建默认值

```python
from autoflow import AutoFlowConfig
config = AutoFlowConfig.from_config_dir("./configs")
```

### Dual-VENC `Nv=7` 配置

```python
from autoflow import AutoFlowConfig, run_case

config = AutoFlowConfig(
    output_dir="./results/demo",
    background_phase_correction=False,
    dual_venc_ratio1=0.0,
    dual_venc_ratio2=0.0,
)
summary = run_case("./data/demo_data.h5", config=config)
```

当 `dual_venc_ratio1` 和 `dual_venc_ratio2` 都为 `0.0` 时，loader 会根据 high/low venc 比值自动推导 dual-venc 阈值。
