# 当前目标期刊对照门槛（2026-10-04）

这份表只服务于最终投稿实验，不把期刊论文的任务指标直接当成本文的 baseline。

## 两个最有用的参照

| 参照 | 它实际证明了什么 | 对本文的硬启发 |
|---|---|---|
| Sadak, *Expert Systems with Applications*, DOI `10.1016/j.eswa.2026.132241` | 四个专门角色、显式安全规则、可靠性加权 Bayesian fusion；894 条专家标注真实 ATC 交换（52 个阳性）+400 条压力集；两个后端；严格留出、低阳性率、召回/FPR、延迟 | 不能只给一套合成数据和一个模型；必须报告严格隔离、低阳性率、固定 FPR、后端迁移、延迟/成本和组件增益 |
| Yang et al., *IEEE TDSC*, DOI `10.1109/TDSC.2026.3670889` | 中央式、水平链、自由通信三种协作拓扑；直接拼接/总结/投票三种通信方式；用攻击实验刻画协作面 | 必须按拓扑和通信方式分层评估，不能把 MAS 当成一个无结构分类表；要报告攻击传播、漏报和拓扑留出 |

## 当前候选相对状态

| 门槛 | 当前证据 | 判定 |
|---|---|---|
| 双线方法本身可审计 | workflow/knowledge BN + runtime BN；模型变量、因子、路径和未知证据策略均有冻结说明 | 已有 |
| 强基线与同信息条件 | runtime logistic、graph/local learned controls、topology/taint controls；family-disjoint 5-fold OOF | 已有，但仍需外部复现 |
| 严格 MAS 结构评测 | 186 topology families；A2ASecBench leave-one-attack-family-out；LANYUN 80-case transfer | 部分已有，外部标签仍非独立人工标注 |
| 多后端/跨 API 迁移 | 本地网关与 LANYUN `glm-5.3-flash` 均已跑通；LANYUN 80 案例 64 完成 | 有迁移证据，样本标签仍来自同一模拟器 |
| 低阳性率与固定 FPR | 主集 family-cluster CI、5% FPR 诊断 | 已有；动作前固定 FPR 仍出现分布偏移 |
| API 真实成本 | LANYUN 请求数、token、p50/p95 传输；监测器 CPU p50/p95 | 部分已有；语义抽取端到端成本/内存仍缺 |
| 真实外部效度 | AgentLeak、QuadSentinel、A2ASecBench 边界/迁移实验；新增 MAST/MAD 1,642 条轨迹迁移审计 | MAST 标签是公开 annotation pipeline，人工子集仅 19 条且 18 阳性；仍缺与本文安全目标一致的独立人工标注或授权队列 |
| 可复现性 | 公开字段、冻结协议、manifest、失败记录、目标脚本 | 已有；结果文件按仓库规则保留在本地结果目录 |

## 结论

当前结果已经达到“完整开发候选”的严谨度，但还不能宣称满足 ESWA/TDSC 的最终投稿门槛。真正决定是否可投的不是再堆一个弱 baseline，而是补齐独立 MAS 外部队列，并把语义解析/API/监测器的端到端成本拆开报告。若外部队列无法获得，论文应明确定位为公开轨迹与受控 API 迁移研究，不能写成真实部署性能证明。

## 核验来源

- ESWA 论文出版社页：[A multi-agent LLM framework with Bayesian fusion and safety guardrails](https://www.sciencedirect.com/science/article/pii/S0957417426011541)
- TDSC 作者公开复现实验仓库：[Cracks-in-Agent-Collaboration](https://github.com/S1mpleyang/Cracks-in-Agent-Collaboration)
- ESWA 期刊范围：[Expert Systems with Applications](https://www.sciencedirect.com/journal/expert-systems-with-applications)
- IEEE 可复现性说明：[IEEE Research Reproducibility](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/research-reproducibility/)
