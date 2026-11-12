# 外部资产

位置相对于 `paperAlpha/`；资产存在不表示已完成原方法复现。方法角色以 [baselines.json](../../experiments/baselines.json) 和[原生 MAS 审计](../references/mas_safety_2026-09-11/journal_alignment/native_mas_baseline_audit_20261005.md)为准。

| 位置 | 内容 |
| --- | --- |
| `data/raw/external/MAST/` | MAST/MAD 数据发布 |
| `data/raw/external/ATBench/` | ATBench 数据发布 |
| `data/raw/external/Who_and_When/` | 失败定位数据 |
| `code/external/A2ASecBench/`、`AgentLeak/` | 含代码的基准资产，整仓保留 |
| `code/external/HarnessAudit/` | API MAS 采集与基准实现 |
| `code/external/BlindGuard/`、`G-safeguard/` | 图监测作者实现 |
| `code/external/QuadSentinel/`、`ALTEDA/` | MAS 安全作者实现 |
| `code/external/AgentMonitor/`、`AgentMonitor_audit_snapshot/` | 两个不同版本的监测参考副本，保留各自元数据 |
| `code/external/TAMAS/`、`MultiAgentHarmDataset/` | 静态案例与运行器等资料，不自动构成 runtime 金标准 |
| `code/external/runtime_dependencies/` | 原采集运行依赖 |

本次未更新上游或重新下载数据。准确版本、来源和 hash 以各实验原始 manifest 为准；父路径变化不改变数据版本。

第三方代码和纯数据默认不提交 Git。按各自许可恢复与分发；不将未审查的外部仓库、模型缓存或凭据直接加入发布包。
