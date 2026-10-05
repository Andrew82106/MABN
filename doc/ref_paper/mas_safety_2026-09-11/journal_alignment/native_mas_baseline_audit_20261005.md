# 原生 MAS baseline 可复现性审计（2026-10-05）

本审计区分“同目标、同信息边界的主表 baseline”和“只能作为补充参照的方法”。不能因为代码公开，就把方法直接写成公平复现。

| 方法 | 公开实现状态 | 输入与输出 | 本项目结论 |
|---|---|---|---|
| AgentMonitor | 官方代码可运行 | 每个 Agent 的输入/输出统计、token/长度/次数、图统计；外部 LLM judge + XGBoost 预测任务表现 | 可做 MAS 统计聚合控制；没有原生安全风险概率/标签，不能直接作为同目标主 baseline |
| G-Safeguard / BlindGuard | BlindGuard 仓库含 checkpoint 与数据生成脚本；G-Safeguard 主仓库不含 checkpoint | 固定拓扑、384-d 文本 embedding、边特征、攻击者索引和问答场景；运行时需重新生成对话 | 需按本文 schema 重训/适配；预训练 checkpoint 不能直接迁移到 API-only raw trace |
| QuadSentinel | 代码公开，但当前无 checkpoint/基准数据 | policy、消息、工具 action、sender/recipient；输出 allow/deny | 可做策略守卫/干预补充对照；目标不是未来风险概率，不能直接拼主表 |
| ALTEDA | 代码与日志公开 | 同步 application、network、host 日志，且依赖作者 red-team traces | 观测比 API-only 更丰富，只能标作 richer-observation 上界，不能与本文作同信息公平比较 |

因此，当前主表只保留同一公开字段、同一标签定义、同一 family holdout 下的 proxy：`local_only_learned`、`graph_features_learned`、`dynamic_taint_path`、`trust_reputation_risk` 和 reliability-weighted Bayesian baseline。外部方法若要进入主表，必须补齐：同一原始轨迹、同一风险谓词、同一训练/测试划分和公开可重放的输入转换；否则放入补充实验并明确适配边界。

## 对论文的写法

不要写“我们复现了 AgentMonitor/G-Safeguard/QuadSentinel/ALTEDA”。应写成：

> We audited publicly available native MAS monitors and separated exact same-information baselines from adapted or richer-observation references. The main table uses protocol-compatible controls; native methods are reported only when their original input, target, and evaluation boundary can be reproduced.

该结论不否定这些工作的价值，而是说明它们解决的是不同问题：任务表现预测、策略守卫、拓扑异常或多源主机侧检测。它们可以支持 related work 和补充实验，但不能替代本文的 API-only 系统级风险监测对照。
