# paperAlpha 工作入口

- 用简洁、直白的中文，先说结果，需要时再展开。
- 先读 `README.md` 项目宪章、`docs/manuscript/README.md`、`experiments/protocol.json`，不另建竞争方案。
- 研究对象是 API 多智能体系统的外部可观测风险；不宣称模型内部透明、已发现因果或绝对安全。
- 导航看 `docs/usage/PROJECT_STRUCTURE.md`，运行看 `docs/usage/REPRODUCIBILITY.md`；文献只有 `docs/references/` 一个入口。
- 说明写入 `docs/`；机器读取的风险定义、模板和约束只写入 `knowledge/`；算法在 `code/src/`，命令在 `code/commands/`，测试在 `code/tests/`。
- 可复用输入在 `data/`；每次实验的配置、分析和完整运行包在 `experiments/<主题>/`。新增运行使用新 ID，不覆盖旧包，并更新 `experiments/index.json`。
- `archive/` 只收已退役的整套系统。日期较早不等于应当归档。环境、缓存与备份只放 `.local/`。
- 不改原始运行、失败记录、冻结代码、旧指标或测试标签；目录整理不授权重新调参、重跑 API 或改写证据。
- 历史路径查询使用 `code/commands/resolve_historical_path.py`；映射只定位文件，不使新源码冒充原冻结源码。
- 本机沿用 `D:\anaconda\envs\multi_agent_graph\python.exe`；未经明确要求不创建环境，不改变对话模型或思考强度。
- 不读出、打印或哈希真实 `.env` 内容；凭据不进入文档、日志、迁移清单。外部资产按原版本、许可与隐私要求恢复和分享。
- `archive/` 与旧日期记录中的任务、环境限制和“当前结果”只是当时状态，不自动成为新任务或新实验授权。
- `intelligence_knowledge_boundary/` 是另一个项目，常规 MAS 任务不修改它。
