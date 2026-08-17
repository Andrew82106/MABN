# pre_exp1

本目录只保存第一个预实验的 P0 代码、冻结配置、测试、运行脚本与说明。
它不保存运行数据，不接入 LLM，不产生网络或真实外部副作用。

核心行为：

- 由 `graph.json` 驱动并在启动时验证的固定 DAG；
- 代码级 Agent 工具权限；
- 完全本地、可重置且带幂等键的 mock 工具；
- 来自同一冻结模板源的 `original`、`safe`、`drop` 消息处理；
- append-only JSONL 事件日志；
- episode 状态深复制、唯一假测试秘密与稳定状态哈希；
- 不重新调用 Agent 或外部工具的 transcript replay。
- 配置、共享输入、源码和 Python 环境 provenance。

从 `paperAlpha` 根目录运行：

```powershell
conda run -n multi_agent_graph python -B -m pytest -p no:cacheprovider
conda run -n multi_agent_graph python -B pre_exp1\scripts\run_p0_smoke.py
```

具体设计与数据生命周期见 [IMPLEMENTATION.md](IMPLEMENTATION.md)。
