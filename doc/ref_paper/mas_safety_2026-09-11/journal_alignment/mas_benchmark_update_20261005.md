# MAS 基准补充审计（2026-10-05）

本文件只补充滚动调研中的基准资源，不把它们冒充中科院一区/二区期刊论文，也不把攻击成功率直接当作本文监测器的检测指标。

## 1. TAMAS：应更新为 ACL 2026 主会论文

TAMAS 已在本目录第 55 项和本地 PDF `pdf/055_*.pdf` 中归档，但原记录仍停留在 2025 Workshop 预印本状态。应以 ACL Anthology 的最终记录为准：

- [ACL Anthology 正式页面](https://aclanthology.org/2026.acl-long.1442/)；DOI `10.18653/v1/2026.acl-long.1442`，ACL 2026 Long Papers。
- [官方代码与数据仓库](https://github.com/microsoft/TAMAS)。
- 最终摘要可核验：5 个场景、300 个 adversarial instances、100 个 harmless tasks、6 类攻击、211 个工具、10 个 backbone，以及 AutoGen/CrewAI 的 3 种交互配置；提出 Effective Robustness Score（ERS）。

它是目前最适合做 MAS 安全外部评测的公开基准之一，但它评估的是系统鲁棒性和安全—效用权衡，不是一个现成的在线风险概率监测器。本文若使用它，应固定可见前缀、从轨迹中剥离攻击标签和最终评分字段，再单独报告迁移结果；不能把 TAMAS 的 ASR/ERS 与本文 AUROC/F1 直接横比。

## 2. DUMA-Bench：新增相邻基准，不作为原生 MAS baseline

[DUMA-Bench 官方仓库](https://github.com/ai-security-lab-itmo/duma-benchmark)与[论文预印本](https://arxiv.org/abs/2609.24662)（2026-09-21）提出 dual-control 评测：Agent 与用户模拟器都能通过消息和工具调用改变共享环境。仓库列出 8 个安全域，覆盖 RAG poisoning、cross-agent manipulation、unsafe output、数据过度披露、资源滥用、钓鱼、身份伪造和 tool shadowing，并同时支持 solo 与 dual-control 条件。

DUMA-Bench 的价值是提供“主动用户 + 共享环境”的系统级压力测试；它不是纯 MAS 基准，主体仍是单个 Agent 与用户的双控制交互。因此可作为本文的相邻迁移集或环境效应对照，不应声称它已验证多 Agent 协作监测，也不应与 TAMAS、MAST 的标签直接合并。当前仅记录官方代码和预印本状态，未把其结果写入主表。

## 3. 对当前实验的决策

1. TAMAS：保留为首要 MAS 外部迁移基准，并更新索引中的发表状态；不替换本文独立人工安全确认集。
2. DUMA-Bench：先做接口与标签边界审计；若能在不读 evaluator 字段的条件下取得完整轨迹，再作为 dual-control 外部诊断，否则只保留为文献与候选数据源。
3. 两者都不是中科院分区证据。目标期刊仍按 `journal_alignment/catalog.json` 的官方分区核验规则处理。
