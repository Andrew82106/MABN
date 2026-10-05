# 目标期刊与近邻论文核查报告（2026-10-05）

本报告是对投稿目标和近期近邻工作的独立核查，不修改本文模型、数据或主结果。只把出版社/作者公开材料和可复查的 Crossref 元数据作为论文事实依据；搜索结果和二级分区页面只用于发现线索。

## 先给结论

1. **最匹配的投稿叙事仍是 ESWA**：风险评估、多智能体系统、知识/规则融合和应用监测都在期刊官方范围内。Sadak 的 ESWA 论文是目前最接近的“多 Agent + Bayesian fusion + 安全约束”实验参照。
2. **Information Fusion 是有条件的更高目标**：只有把两条证据线、关系传播和知识版本化写成真正的信息融合方法，才有充分匹配；简单并联两个分类器不够。
3. **IEEE TDSC 是安全严谨度参照**：Yang 等工作说明必须按协作拓扑、消息处理方式和攻击传播分层评估，但它本身是攻击研究，不是可直接替换的风险监测 baseline。
4. **Information Processing & Management（IPM）可作为新增候选**：ALTEDA 与本文的 API 可见性边界不同，但其“运行日志图 + 威胁检测 + 归因 + 部分轨迹早报”对实验设计很有价值。它是否符合学校采用的中科院一区/二区版本，仍需官方分区库确认。
5. **中科院分区尚未完成官方条目认证**：现有机构网页和二级数据库可以支持“候选”，不能写成“已确认一区/二区”。Clarivate 的 SCIE 收录或 JCR/SJR 分区也不能替代中科院分区。

## 期刊候选

| 期刊 | 可核验匹配 | 中科院分区状态 | 当前用途 |
|---|---|---|---|
| Expert Systems with Applications（ESWA） | 官方期刊范围包含 risk assessment、multi-agent systems、knowledge management；与双线风险监测最贴合 | 机构页/二级页称计算机大类一区或一区 Top；尚未从中科院官方登录条目核验年份、版本和大小类 | **首选投稿候选**；对照 Sadak、Chahine 的实验完整性 |
| Information Fusion | 官方范围覆盖多源/多过程/多层融合、态势感知、不完备信息下学习和信息安全 | 机构页明确引用 2025 表称一区 Top；官方分区条目尚未取得 | **条件候选**；要求把融合机制本身做成贡献 |
| IEEE Transactions on Dependable and Secure Computing（TDSC） | 官方出版物/作者仓库可核验其系统可靠性与安全方向；适合 MAS 威胁模型、攻击传播和可复现性 | 机构页称中科院二区；IEEE 资料可证 SCIE，但不能证中科院二区 | **安全参照/候选**；需独立攻击标签和拓扑留出 |
| Information Processing & Management（IPM） | 官方期刊页定位信息处理、检索、可信/安全信息系统；ALTEDA 已发表运行日志图威胁检测论文 | 二级/机构信息称 2025 计算机科学大类一区 Top；官方分区条目尚未取得 | **可选候选**；仅在强调可观测日志、风险归因和早报时合适 |

官方分区入口和所需凭据见 [`catalog.json`](./catalog.json) 的 `official_partition_verification`。在拿到学校实际采用版本的官方记录前，正文只能写“目标候选”，不能写“中科院一区/二区已认证”。

## 可直接核验的期刊近邻论文

### 1. Sadak et al., ESWA：Bayesian fusion + safety guardrails

- 论文：[A multi-agent LLM framework with Bayesian fusion and safety guardrails for ATC-pilot communication error detection](https://doi.org/10.1016/j.eswa.2026.132241)
- Crossref/出版社元数据：ESWA，vol. 321，article 132241，2026-07 发布记录。
- 出版社摘要/预览可核验：四个专门角色处理 syntax、semantics、context、risk；显式安全规则与 reliability-weighted Bayesian fusion；894 条专家标注真实交换（52 条错误）和 400 条压力样本；报告 Gemini 2.5 Flash、Llama-4 Maverick、严格留出/5-fold OOF、召回/FPR 和约 1.6 s 端到端延迟。
- **与本文的关系**：它是实验范式参照，不是同任务 baseline。ATC 通信错误标签不能直接当作 API MAS 授权/信息流风险标签。
- **本文必须对齐**：低阳性率、固定 FPR 下召回、后端迁移、延迟/Token/API 成本、规则与融合的组件消融。

### 2. Chahine et al., ESWA：intent–execution defense in depth

- 论文：[Separating intent from execution: A defense-in-depth security architecture for LLM-based multi-agent systems](https://doi.org/10.1016/j.eswa.2026.133781)
- Crossref 元数据：ESWA，vol. 332，article 133781，标记 2027-01 卷期；不能把 DOI 的 2026 前缀或记录创建日当成 2026 首次上线日期。
- 出版社摘要/预览可核验：真实 Ed25519 密码操作原型与实时 API；23 项实验、16 类攻击向量；比较完整 SEL、LLM-only、pre-parsed LLM、function-calling/structured-output 和确定性规则。
- **与本文的关系**：不是通用风险概率监测器，而是授权和执行边界架构。它最值得借鉴的是“规范约束阻止什么、运行语义补充什么”的双线消融。
- **本文必须对齐**：预注册规范线/运行线消融，并报告攻击成功、漏报、拦截位置和模糊语义输入的代价。

### 3. Yang et al., IEEE TDSC：协作拓扑与攻击面

- 论文：[Cracks in Collaboration: Threat Models and Attacks on Multi-LLM Collaborative Systems](https://doi.org/10.1109/TDSC.2026.3670889)
- 作者公开仓库：[Cracks-in-Agent-Collaboration](https://github.com/S1mpleyang/Cracks-in-Agent-Collaboration)
- Crossref/仓库引用可核验：TDSC，vol. 23，issue 3，pp. 7191–7207，2026；三类拓扑 centralized、horizontal chain、joint free communication；三类消息处理 direct、summary、vote；攻击类别 Decision Poisoning、Indirect EchoLeak、Information Collision。
- **未核验**：完整论文样本量、重复次数、置信区间、完整 baseline/消融表。仓库文件名不能替代论文分母。
- **本文必须对齐**：按拓扑/通信方式分层报告风险传播、系统漏报、拓扑留出和攻击预算，不能只给一个合并 F1。

### 4. Rabieinejad et al., IPM：ALTEDA 日志图威胁检测

- 论文：[Beyond the prompt: Log-based threat detection and attribution for multi-Agent LLMs](https://doi.org/10.1016/j.ipm.2026.104768)
- 出版社摘要和 Crossref 可核验：Information Processing & Management，vol. 63，issue 6，article 104768，2026-09 发布记录；800 条红队轨迹、31.4% attack success rate；同步应用/网络/主机日志构成带属性有向多重图；图级威胁分类、SHAP 归因和部分轨迹早报。
- **输入边界**：ALTEDA 使用主机、系统和网络日志；本文声明 API 可见文本/消息/工具调用边界。因此只能作为 richer-observation 上界，或先做同信息投影后比较。
- **本文必须对齐**：归因到 agent/交互边、部分轨迹早报、按任务/拓扑划分、同信息输入控制，并独立记录端到端资源开销。

### 5. Ben Hassouna et al., Information Fusion：LLM-Agent-UMF

- 论文：[LLM-Agent-UMF: LLM-based Agent Unified Modeling Framework for Seamless Design of Multi Active/Passive Core-Agent Architectures](https://doi.org/10.1016/j.inffus.2025.103865)
- 出版社/作者公开摘要可核验：Information Fusion，vol. 127，article 103865，2026-03 卷期记录；把 core-agent 拆成 planning、memory、profile、action、security，并区分 active/passive core-agent；评估 13 个既有 agents 和 5 个混合架构。
- **与本文的关系**：架构语义参照，不是风险检测 baseline。可帮助固定第一层工作流节点/边的词汇，但不能把架构分类分数混入风险检测结果。

### 6. Wang et al., Information Fusion：PTFusion

- 论文：[PTFusion: LLM-driven context-aware knowledge fusion for web penetration testing](https://doi.org/10.1016/j.inffus.2025.103731)
- 出版社/元数据可核验：Information Fusion，vol. 127，article 103731，2026-03 卷期记录；Master/Recon/Attack 半去中心化多 Agent、MCP 工具调用和动态知识图。
- **与本文的关系**：知识版本、实体绑定、工具证据来源的设计参照；目标是渗透测试任务完成，不是通用风险概率监测器。
- **本文必须对齐**：无知识/静态知识/版本化知识消融，冻结测试集知识版本，防止知识库泄漏。

## 最新 MAS 监测近邻（目前不是 SCI 期刊论文）

这两篇不能满足“SCI 一区/二区期刊文章”要求，但必须纳入前沿对照，否则容易误判研究空白。

### HalluProp（arXiv:2607.26836）

论文：[Before Agents Speak: Pre-hoc Failure Risk Inference in Multi-Agent Systems](https://arxiv.org/abs/2607.26836)

它在 Agent 交互前估计单 Agent intrinsic risk，再用语义影响和通信拓扑建模传播，最后用可微 Noisy-OR 得到系统级风险。论文报告 MMLU/MATH/MedQA、多种 MAS 规模/拓扑、平均 AUROC 约 84.6%，并与 LLM Debate、Inspector、SelfCheckGPT、GUARDIAN 比较；去掉 topology、inter-agent propagation 或 intrinsic-risk 模块会明显下降。

**对本文的直接影响**：我们的双线模型不能只说明“有 DAG”；必须证明系统层风险不是各 Agent 风险简单平均，并做 topology/propagation 去除消融。HalluProp 只处理 hallucination pre-hoc 风险，不能替代本文的授权、工具和信息流风险标签。

### Bayesian Uncertainty Propagation（arXiv:2607.00972）

论文：[Bayesian Uncertainty Propagation for Agentic RAG Pipelines](https://arxiv.org/abs/2607.00972)

它从 planner、evaluator、generator 的语义分歧和自评信号出发，用 BN 传播多阶段不确定性；在 StrategyQA/HotpotQA、GPT-3.5-Turbo/GPT-4.1-Nano 上报告 AUROC、AUARC、ECE、Brier。它说明“阶段级不确定性→系统级风险”的 BN 方向已有公开先例，但任务是 Agentic RAG 问答，不能直接视作 API MAS 安全 benchmark。

**对本文的直接影响**：需要报告节点级风险、系统级风险和校准，而不只报告最终分类；同时把 token/logprob 或 evaluator 自评信号与本文可观测事件线的差异写清楚。

## 可投实验的最小对齐表

| 对齐项 | 本文应报告 |
|---|---|
| 系统对象 | 至少 chain、fork、join、review 四种无环拓扑；消息/权限跨 Agent 传播 |
| 监督独立性 | monitor 只看公开前缀；违规标签、隐藏状态、evaluator 字段不可进入特征 |
| 方法消融 | workflow-only、runtime-only、无关系结构、无知识模板、规则下限、扁平同信息模型、完整双线 |
| 风险指标 | AUROC、AUPRC、Brier/ECE、固定 FPR 召回、首个违规事件、提前量、归因准确性 |
| 泛化 | family/topology holdout、至少两个 API/model 条件；按拓扑和攻击类型分层 |
| 工程指标 | 语义抽取失败率、p50/p95 延迟、token/API 成本、监测器 CPU/RSS、失败请求是否保留 |
| 解释性 | 风险节点、关系路径、证据 ID、删除/替换证据后的风险变化 |

## 当前投稿判定

现有论文与期刊核查已经足以确定实验对齐方向，但不能证明当前结果已经达到最终可投状态。主结果仍需独立 MAS 安全确认队列、双人盲标/一致性、固定低误报点的早报评测，以及完整端到端语义成本/内存。任何期刊的中科院一区/二区身份，在取得学校采用版本的官方分区条目之前都只能写成“候选”。

## 来源入口

- [Clarivate Master Journal List 说明](https://mjl.clarivate.com/scope/scope_ssci/)
- [中科院分区官方入口](https://www.fenqubiao.com/)
- [ESWA 官方范围](https://shop.elsevier.com/journals/expert-systems-with-applications/0957-4174)
- [Information Fusion 官方范围](https://shop.elsevier.com/journals/information-fusion/1566-2535)
- [IPM 官方期刊页](https://www.sciencedirectelsevier.com/journal/Information-Processing-%26-Management)
- [Yang et al. 作者复现实验仓库](https://github.com/S1mpleyang/Cracks-in-Agent-Collaboration)
