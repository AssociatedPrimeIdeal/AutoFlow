# 测试

## 推荐环境

```bash
~/miniconda3/envs/ryy/bin/python -m pytest tests/test_smoke_phantoms.py tests/test_pressure_gradient_phantom.py -q
```

## 保留的自动化测试

| 范围 | 主要测试 |
| --- | --- |
| smoke 回归 | `tests/test_smoke_phantoms.py` |
| 压力梯度 phantom 回归 | `tests/test_pressure_gradient_phantom.py` |

## 期望
- 自动化测试只保留 smoke 和 phantom 回归
- 不要求每次代码改动都新增专项 pytest 文件
- 对超出保留回归范围的行为改动，主要依靠手工验证和同步文档更新
