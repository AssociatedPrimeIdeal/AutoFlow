# 变更配方

## 新增一个 CLI 参数
1. 改 `autoflow/cli.py`
2. 如果要成为公共 API，改 `autoflow/api.py` 里的 `AutoFlowConfig`
3. 如果要有配置默认值，改 `autoflow/config.py` 和对应 `configs/<module>.json`
4. 如果需要进入 workspace，改 `build_workspace()`
5. 更新中英文文档
6. 如果 quickstart、顶层导航或文档清单变了，也要更新 `README.md`
7. 按需运行保留的 smoke 和 phantom 回归测试

## 新增一个 GUI 参数
1. 改对应 `configs/*.json`
2. 改 `autoflow/config.py`
3. 如需进 workspace，改 `autoflow/core/models.py`
4. 改 `autoflow/ui/app.py` 的 UI 构建和同步逻辑
5. 更新 GUI 文档和相关功能页
6. 按需运行保留的 smoke 和 phantom 回归测试

## 新增一个输出文件
1. 在 `autoflow/core/pipeline.py`、`autoflow/processing.py` 或 `autoflow/rendering/videos.py` 里写出文件
2. 如果摘要依赖它，补 `autoflow/reporting.py`
3. 更新中英文输出文档
4. 如果 quickstart 可见输出变了，也要更新 `README.md`
5. 按需运行保留的 smoke 和 phantom 回归测试
