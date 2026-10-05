# MAS 风险监测：目标期刊与近邻论文对照（2026-10-05）

这份表只做两件事：确定投稿实验要对齐的标准，和区分“可以直接复现的监测基线”与“只能借鉴实验设计的论文”。论文数值只有在当前文献库已经从出版社摘要、公开代码或公开数据核验过时才写入；没有核验的地方明确标注。

## 期刊候选

| 期刊 | 与本文的匹配点 | 当前分区证据 | 结论 |
|---|---|---|---|
| **Expert Systems with Applications (ESWA)** | 官方范围包含 risk assessment、multi-agent systems、knowledge management 和 monitoring；最适合“双线风险监测 + 知识融合”主线 | 机构页面/二级目录报告中科院计算机大类一区；当前官方分区条目尚未核验 | 首选候选。必须把双线写成可审计的方法，并用严格留出、强基线和成本报告证明不是概念拼接 |
| **Information Fusion** | 若核心贡献是真正的多源证据、不确定性和关系结构融合，则高度匹配 | 2025 机构页面明确引用中科院表称一区 Top；当前官方条目尚未核验 | 有条件候选。只有“信息融合”本身是方法贡献时才适合，不能只把两个模型串联 |
| **IEEE TDSC** | 多 Agent 威胁模型、攻击传播、系统级漏报/误报和可复现性标准高度相关 | IEEE 官方标题列表可核验 SCIE；中科院二区来自机构/二级页面，官方分区条目尚未核验 | 安全方向候选/严谨度参照。需要攻击传播、拓扑留出和独立安全标签，不可只报普通分类分数 |

分区结论仅表示“候选”，不等于已完成学校采用版本的中科院一区/二区认证；投稿前必须用学校可访问的官方分区库复核年份和大小类。

## 最接近的论文对照

| 论文（期刊） | 研究对象与标签 | 对照/指标 | 本文已满足 | 仍需补齐 |
|---|---|---|---|---|
| [Sadak, Bayesian fusion and safety guardrails](https://doi.org/10.1016/j.eswa.2026.132241)（ESWA） | ATC-pilot 通信错误；4 个专门角色。894 条专家标注真实交换（52 条错误，阳性率 5.8%）+400 条压力集 | 显式规则 + reliability-weighted Bayesian fusion；Gemini 2.5 Flash/Llama-4 Maverick；严格留出、5-fold OOF、召回/FPR、端到端延迟。摘要报告平均召回 90.6%、FPR 2.4%–4.6%、约 1.6 s | 本文已有双线 BN、family-disjoint 5-fold、固定 FPR、校准、解释路径、两种 API 后端及 API/监测器成本拆分 | 独立人工/授权 MAS 标签；更低且真实的阳性率；多后端的完整端到端语义抽取成本；不能把 ATC 标签当作本文 baseline |
| [Chahine, Separating intent from execution](https://doi.org/10.1016/j.eswa.2026.133781)（ESWA） | MAS 中“意图—执行”安全边界；23 项实验、16 类攻击向量（不是 23 条轨迹） | 完整 SEL、LLM-only、pre-parsed LLM、function-calling/structured-output、确定性规则；正确性、注入抵抗、延迟、模糊凭证解释 | 本文已区分规范线与运行线，并有规则/逻辑回归/图特征等同信息控制 | 把“规范知识阻止了什么、运行证据补充了什么”做成预注册消融；在攻击注入和权限/信息流场景报告攻击成功、漏报与拦截位置 |
| [Yang et al., Cracks in Collaboration](https://doi.org/10.1109/TDSC.2026.3670889)（IEEE TDSC） | 多 LLM 协作；centralized、horizontal-chain、free-communication 三种拓扑；Decision Poisoning、Indirect EchoLeak、Information Collision | 比较直接拼接、总结、投票三种通信；公开仓库提供运行入口。论文完整样本、baseline、CI 和消融尚未核验，不能抄其未核验数值 | 本文已开始统一 chain/fork/join/review 队列，并保留失败记录、monitor/evaluator 隔离 | 主实验必须按拓扑/通信方式分层；报告风险传播、系统级漏报、拓扑留出与攻击预算，而不是只合并成一个 F1 |
| [Rabieinejad et al., ALTEDA](https://doi.org/10.1016/j.ipm.2026.104768)（Information Processing & Management） | 多 Agent 应用/网络/主机日志图；800 条红队轨迹（400 benign + 400 compromised，12 个攻击变体，公开 metadata） | 图级威胁分类 + SHAP 归因；摘要称约 40% 轨迹进度可发出高可信早报。具体完整 baseline 表未核验 | 本文有动态运行证据线、白盒变量/路径和前缀告警诊断 | ALTEDA 依赖主机/网络日志；若本文只允许 API 可见事件，必须做同信息投影，或把 ALTEDA 明确标为 richer-observation 上界，不能直接公平混比 |
| [Ben Hassouna et al., LLM-Agent-UMF](https://doi.org/10.1016/j.inffus.2025.103865)（Information Fusion） | 将 Agent 拆为 planning、memory、profile、action、security，并区分 active/passive core-agent | 架构/组件建模参照，不是风险监测器；未提供可直接复用的风险标签与监测 baseline | 可指导第一层工作流 DAG 的节点语义与边类型 | 需要把每个模板节点绑定到公开事件证据、版本和权限条件；不能把架构图当作检测性能证据 |
| [Wang et al., PTFusion](https://doi.org/10.1016/j.inffus.2025.103731)（Information Fusion） | Master/Recon/Attack 多 Agent、MCP 工具调用、动态知识图，用于 Web 渗透测试 | 目标是任务完成与知识融合，不是风险概率监测；不把完成率与本文 F1 混比 | 可借鉴知识版本、实体绑定、工具证据来源的记录方式 | 做“无知识/静态知识/版本化知识”消融，并冻结知识更新边界，避免知识库在测试集泄漏 |

## 对本文实验的硬约束

1. **对象必须是系统级 MAS 风险**：至少覆盖多种无环拓扑、跨 Agent 信息/权限传播和攻击或违规行为；不能只把独立 Agent 轨迹拼接。
2. **标签必须独立于监测特征**：monitor 只能看公开工作流、消息、工具调用、参数和结果；违规标签、隐藏状态、评测器字段只能由 evaluator 使用。最好由两名标注者独立标注首个违规事件、类别和证据，并报告一致性。
3. **基线必须同信息量**：规则下限、runtime-only、workflow-only、扁平 GLM/树模型、图/日志模型和 reliability-weighted Bayesian control 都要在相同划分上跑；ALTEDA 这类更丰富日志只能另列上界。
4. **主指标不能只有 AUROC**：AUPRC、Brier/ECE、固定 FPR 下召回、最早报警提前量、误报率、拓扑留出、延迟、token/API 成本和失败覆盖都要报告，并给 family/episode 聚类区间。
5. **双线贡献要可证伪**：报告 workflow-only、runtime-only、无关系结构、无知识模板、无融合头和完整模型；如果双线没有稳定增益，就不能声称融合有效。

## 已执行的 MAS 对照

同一 `independent_mas_v3`、同一五折 scenario-family 留出上的 MAS 专用代理结果为：

| 对照 | AUROC | AUPRC | 解释 |
|---|---:|---:|---|
| 独立 Agent 分数平均 | 0.480 | 0.324 | 不建模跨 Agent 关系 |
| topology-only | 0.545 | 0.358 | 只看可达性，不看 provenance |
| no-taint contribution | 0.573 | 0.382 | 图贡献分数，但不区分数据来源 |
| dynamic taint path | 0.597 | 0.405 | 规则式跨 Agent 传播 |
| **本文双线 BN** | **0.820** | **0.675** | 工作流规范线 + 运行时证据线 + OOF 融合 |

这些代理不是对论文代码的逐行复现，而是严格同信息量、同数据划分的协议兼容对照；
因此可以支持“系统级关系和 provenance 同时建模”的实验论点，但不能写成已复现 ALTEDA、
CogTrust 或 HalluProp 的完整实现。

## 当前判定

现有结果已经满足“开发级、可审计、与 ESWA/TDSC 实验范式对齐”的大部分要求；两组 LANYUN 冻结队列已经补齐 160 个四拓扑、双模型的策略意图集成样本，但标签仍是手工 evaluator 目标。决定性缺口仍是：独立双人/许可 MAS 安全确认与一致性、按拓扑/通信方式分层的攻击传播评测，以及部署级内存和训练成本。当前不得把策略意图队列或受控模拟器迁移结果写成通用 API Agent 安全性能结论。

## 证据入口

- 现有论文与期刊记录：[`catalog.json`](./catalog.json)
- 当前投稿门槛：[`current_target_gate_20261004.md`](./current_target_gate_20261004.md)
- ESWA/TDSC 要求映射：[`submission_requirements_20261002.md`](./submission_requirements_20261002.md)
- 统一实验结果：[`FINAL_RESULTS.md`](../../../../paperAlpha/submission/FINAL_RESULTS.md)
