# 基线映射

| 方法 | 作用 |
|---|---|
| `single_agent_local` | 只看单 Agent 局部证据，检验系统级关系是否有增益 |
| `flat_event_logistic` | 扁平事件统计，不保留连通关系 |
| `runtime_graph_logistic` | API-only 的 ALTEDA 风格运行时图投影，不是其私有实现复现 |
| `knowledge_template_only` | 只用授权/工作流偏离模板，检验知识编译本身 |
| `two_layer_bn_runtime` | 去掉规范工作流融合的 BN/运行时消融 |
| `workflow_only` | 只使用规范 DAG 与执行图差异 |
| `ours_hierarchical_bn_runtime` | 局部 BN + 运行时传播 + 规范 DAG + 知识模板的双线融合 |
| `taint_path_rule` | 透明污点可达性规则 |
| `llm_judge_prefix` | 同一可见前缀、固定 token 预算的 LLM-as-Judge 对照；尚未纳入主表 |
| `policy_guard_proxy` | 规范/策略守卫式确定性检查；尚未纳入主表 |
| `temporal_graph_classifier` | 时序图分类器或锁定 ALTEDA 作者实现；尚未纳入主表 |
| `QuadSentinel_native_guard` | 作者实现重放；已在 HarnessAudit 派生 cohort 完成，但目标是 current-policy action detection，不能与本文 future/system-risk 指标直接拼表 |
| `BlindGuard_author_path` | 作者适配器和源码存在，但 `independent_mas_v3` 没有文本消息，且本机缺少 torch-geometric 依赖；未宣称复现 |
| `graph_features_learned` | 同一公开字段上的学习型图统计/连通性控制；已在严格五折 family holdout 中运行 |
| `local_only_learned` | 同一公开字段上的局部 Agent 池化控制；已在严格五折 family holdout 中运行 |
| `dynamic_taint_path` | 透明污点可达性规则控制；已在严格五折 family holdout 中运行，不是外部论文复现 |

`per_agent_max` 在当前公开释放中几乎恒为 0.5，因此全预测为正，只保留为 sanity control；主竞争表使用 `local_only_learned`、`graph_features_learned` 和 `dynamic_taint_path`，避免把退化规则误写成强基线。

文献边界：ALTEDA（IPM, DOI [10.1016/j.ipm.2026.104768](https://doi.org/10.1016/j.ipm.2026.104768)）提供运行时图、早期告警和归因参照；SEL（ESWA, DOI [10.1016/j.eswa.2026.133781](https://doi.org/10.1016/j.eswa.2026.133781)）提供语义意图与确定性授权执行的双线参照；Cracks in Collaboration（TDSC, DOI [10.1109/TDSC.2026.3670889](https://doi.org/10.1109/TDSC.2026.3670889)）提供拓扑/攻击传播参照；ESWA 的 ATC-Bayes（DOI [10.1016/j.eswa.2026.132241](https://doi.org/10.1016/j.eswa.2026.132241)）提供多角色 Agent、规则护栏与 reliability-weighted Bayesian fusion 的实验范式；LLM-Agent-UMF（Information Fusion, DOI [10.1016/j.inffus.2025.103865](https://doi.org/10.1016/j.inffus.2025.103865)）、PTFusion（DOI [10.1016/j.inffus.2025.103731](https://doi.org/10.1016/j.inffus.2025.103731)）和 PenExpert（ESWA, DOI [10.1016/j.eswa.2026.133284](https://doi.org/10.1016/j.eswa.2026.133284)）是系统建模/知识编译参照，不把任务完成率直接混入检测 F1。

## 期刊近邻与公平对照边界

| 期刊近邻 | 论文提供的可借鉴点 | 当前可比任务 | 为什么不能直接拼进主表 |
|---|---|---|---|
| ESWA — SEL | 语义意图识别 + 授权执行的双线设计 | 公共意图/权限证据下的当前风险检测 | 其任务和标签定义不是本文 MAS 风险标签；只能做同信息适配后对照 |
| Information Processing & Management — ALTEDA | 运行时事件图、早期告警、归因 | API-only 动态执行图与动作前预警 | 原作者实现、数据和攻击标签边界不同；当前仅有风格适配器 |
| IEEE TDSC — Cracks in Collaboration | 多种协作拓扑与攻击传播 | 拓扑族留出、跨 Agent 关系消融 | 论文重点是协作脆弱性/攻击成功，不是概率风险监测器 |
| Information Fusion — LLM-Agent-UMF / PTFusion | Agent 模块化、动态知识图与多源融合 | 规范知识线 + 运行时证据线 | 没有与本文相同的公开风险标签和 API 观察边界 |
| ESWA — PenExpert | ATT&CK 知识库与任务状态编译 | 知识模板可审计性、知识增量消融 | 其目标是渗透测试任务辅助，不等于 MAS 系统风险概率 |
| ESWA — ATC-Bayes | 多角色 Agent + 规则护栏 + reliability-weighted Bayesian fusion；多后端、重复留出、低阳性率和延迟报告 | 多后端风险监测、规则下限、概率校准、FPR/召回和成本 | 原论文标签与我们的 API MAS 风险谓词不同；必须在同一输入/标签/分组上重跑，不能直接搬其 recall/FPR |

主表/补表现在包含同一输入、同一标签、同一 family split 下的学习型图控制和透明规则控制；详细数值见 `independent_mas_learned_baselines_v1` 与 `independent_mas_mas_baselines_v1`。这些仍是协议兼容的 proxy，不是已发表方法的复现。投稿前还必须补齐同可见字段、同 API 预算的 `llm_judge_prefix`、`policy_guard_proxy` 和至少一个可运行的外部 MAS 监测实现；若复现失败，报告失败原因，不以风格适配器冒充原方法。上述期刊论文作为方法与实验设计参照，不能用论文摘要中的任务成功率替代公平的风险监测基线；完整来源和核验等级见 `doc/ref_paper/mas_safety_2026-09-11/journal_alignment/catalog.json`。

原生 MAS 方法的输入/目标边界审计见 `doc/ref_paper/mas_safety_2026-09-11/journal_alignment/native_mas_baseline_audit_20261005.md`。其中 AgentMonitor 可作为统计聚合控制，BlindGuard/G-Safeguard 需按本文 schema 重训，QuadSentinel 是策略守卫目标，ALTEDA 是 richer-observation 上界；均不得在未满足同信息、同标签、同划分前写成公平复现。
