# API Agent 运行时风险评估：Baseline 与数据集调研

更新日期：2026-09-13  
后续补核（2026-09-20）：新增原生MAS近邻QuadSentinel与SafeFlow。QuadSentinel已锁定作者代码并完成明确配置的API接入检查，详见[当前动作与未来风险分离](../../experiments/harnessaudit/notes/MAS原生对照接入_当前动作与未来风险分离_20260920.md)；下文保留早期候选清单，不表示所有方法均可在同一目标直接比较或已经复现。

同日实跑补充：[真实MAS采购对照](../../experiments/harnessaudit/notes/真实MAS采购对照_系统关系必要但BN尚无独占优势_20260920.md)已完成8个真实API开发流程的当前批量提案比较。BN、完整规则、LLM裁判与QuadSentinel作者适配配置均判对；删除跨Agent预算汇总漏2例。它是一次明确配置的离线前缀重放，不是论文完整复现、独立泛化成绩或BN学习优势。

研究对象：通过 API 运行的单 Agent / 多 Agent 系统；监测器可见请求、消息、工具调用、回执、权限/批准、记忆、状态和外部效果。  
本文件回答两个问题：**拿什么和我们的风险评估器比较？拿什么数据评测？**

## 一、先给结论

我们的算法不应与“会不会完成任务”的 Agent 直接比较，而应与**监测器/评估器**比较：它能否在运行过程中识别风险、定位风险机制、估计概率、提前预警，并且不把正常行为大量误报为危险。

先纠正一个筛选层级：前一版把很多“可迁移到 MAS 的单 Agent 监测器”和“原生 MAS 监测器”放在了一张表里。它们不能承担同样的比较义务。真正的 MAS baseline 必须至少观察 Agent 间消息/通信图、角色或委派关系、共享状态，或系统级协作结果。

重新筛后，原生 MAS 监测/防护候选包括 SentinelAgent、Node Contribution Backpropagation 和 AgentMonitor（MAS 版）。它们应进入核心比较；AgentDoG、GuardAgent、ShieldAgent、AgentTrust 等降为迁移下界或专项对照。

最接近的现成工作有三类：

1. **MAS 图/协作监测型**：SentinelAgent 在动态执行图的节点、边和路径上检测异常；Node Contribution Backpropagation 在通信 DAG 上回溯 Agent 对最终决策的贡献。这两类才是系统级核心近邻。[SentinelAgent](https://arxiv.org/abs/2505.24201)、[Node Contribution Backpropagation](https://arxiv.org/abs/2510.19420)
2. **MAS 级统计监测型**：AgentMonitor 逐 Agent 采集输入输出，用统计模型预测系统表现并做安全修正，但不是细粒度风险机制模型。[AgentMonitor](https://arxiv.org/abs/2408.14972)
3. **轨迹诊断型**：AgentDoG 直接从完整轨迹诊断风险来源、失败方式和危害，是单 Agent/轨迹层的强迁移基线。
4. **知识/策略守卫型**：GuardAgent、ShieldAgent 把自然语言策略转成可执行检查，最接近“知识进入监测机制”。
5. **运行时拦截型**：AgentTrust 在工具执行前给出 allow/warn/block/review，最接近真实 API 工具边界。

此外必须加入 LLM-as-Judge、静态规则、传统监督模型、朴素跨 Agent 聚合，以及 BayesTrace 原版作为结构消融。AgentMonitor 虽然直接面向 MAS，但它主要预测任务表现，安全修正是附加能力，不能把它当成与我们完全同目标的方法。[8]

数据集应分两层：

- **主安全评测集**：R-Judge、Agent-SafetyBench、AgentDojo、ATBench、MSB/MCP-SafetyBench、A2ASecBench。
- **正常与外部状态对照集**：τ-bench、ToolSandbox、AppWorld，必要时加 WebArena/BrowserGym。

不能只用攻击数据。没有正常轨迹，就无法测误报；没有工具回执和外部状态，就无法区分“模型说要做”和“动作真的产生了效果”。

## 二、Baseline 候选

| 编号 | 方法/基线 | 类型 | 主要输入 | 输出 | MAS 适配 | 对我们最有用的比较 |
|---|---|---|---|---|---|---|
| B1 | 无监测上限（Oracle/独立验收） | 参考上限 | 由环境状态、工具回执和人工/程序化安全谓词直接判定；不作为可部署算法。 | 安全事件/轨迹标签 | 可有MAS | 衡量监测器距离真实外部效果还有多远。 |
| B2 | 静态规则与策略门 | 规则/策略 | 工具白名单、参数 schema、权限、预算、目标域和时序约束。 | 允许/阻断/升级 | 可包在MAS外围 | 最低成本、可审计，适合检测硬约束违反。 |
| B3 | 单步文本/工具调用分类器 | 监督分类 | 只看当前请求、消息、工具说明或一次调用；可用 TF-IDF/embedding+LogReg、XGBoost、MLP。 | 单步风险标签 | 可包在MAS外围 | 成本低，测量“局部检测”能力。 |
| B4 | 全轨迹 LLM-as-Judge | LLM 评审 | 把完整交互记录、工具调用和回执交给外部 LLM，要求输出 safe/unsafe、机制和理由。 | 轨迹级判定/解释 | 可处理MAS日志 | 最直接的黑盒基线；R-Judge就是这一评测形式。 |
| B5 | GuardAgent | 知识到可执行检查 | 外部 guard agent 将自然语言安全要求转成 guardrail 代码，检查目标 Agent 行为。 | 违反/通过与阻断 | 非原生MAS，可包裹 | 与“文献知识→可执行检查”最接近的外部守卫基线。 |
| B6 | ShieldAgent | 策略规则电路 | 从策略文档抽取可验证规则，形成 action-based probabilistic rule circuits，再对轨迹检查。 | 违规判定/屏蔽计划 | 主要单Agent | 知识约束和可验证规则的强基线。 |
| B7 | AgentDoG | 轨迹诊断守卫模型 | 对轨迹做上下文监测，并按 risk source、failure mode、consequence 诊断根因。 | 安全标签、根因和解释 | 非原生MAS，但可输入MAS轨迹 | 当前最直接的轨迹级诊断竞争者；有开放模型和 ATBench。 |
| B8 | AgentTrust | 工具调用运行时拦截 | 在工具执行前输出 allow/warn/block/review，结合规则、链式风险和 LLM Judge。 | 动作级处置 | 工具链/MCP；非MAS专属 | 和 API Agent 的实际执行边界高度相关。 |
| B9 | AgentMonitor（MAS） | MAS统计预测 | 在 Agent 层采集输入/输出统计，用 XGBoost 预测任务表现，并做实时安全修正。 | 任务表现预测/安全修正 | 原生MAS | 少数直接面向 MAS 的现成基线，可比较系统级聚合。 |
| B10 | AgentMonitor（安全测试） | 上下文运行监视器 | 审计 Agent 行动和上下文，遇到越界行为停止测试并记录可疑行为。 | 停止/排序/人工复核 | 单Agent，可置于MAS外层 | 适合运行时安全边界基线。 |
| B11 | AgentGuard | 工作流安全扫描 | 用编排器发现不安全工具工作流，执行验证后生成约束。 | 不安全工作流/约束 | 主要单Agent | 适合比较“先审查计划再执行”的路线。 |
| B12 | ToolEmu 安全评估器 | 模拟轨迹+LLM评审 | 在仿真工具环境中生成候选轨迹，再评估风险严重度和任务有用性。 | 风险/有用性评分 | 单Agent工具环境 | 低成本、结果级和轨迹级对照。 |
| B13 | 记忆/信息流专项守卫 | 专项策略守卫 | 对记忆写入、检索、晋升和跨 Agent 复用，或 source-to-sink 流程设门。 | 记忆/流向违规 | MAS专项 | 可检验我们的子模板是否覆盖已有专项防御。 |
| B14 | 朴素分数聚合 | 结构消融 | 每个 Agent 独立输出风险分数，再取 max/mean/Noisy-OR/加权和。 | 系统风险分数 | 原生MAS | 是我们必须做的关键消融：检验“显式跨节点关系”是否有价值。 |
| B15 | 序列/图模型基线 | 统计/深度序列 | 用 HMM/Markov、Transformer、GNN/Temporal-GNN 对事件序列或工作流图分类。 | 事件/轨迹风险概率 | 可支持MAS图 | 检验 BN 的结构先验与可解释性是否带来增益。 |

### 2.1 哪些应列为论文主表

### 2.1A 原生 MAS 方法（核心比较对象）

| 方法 | 观察对象 | 主要输出 | 与我们方法的关系 |
|---|---|---|---|
| SentinelAgent | 动态 Agent 交互图的节点、边、路径 | 异常、协同风险、根因与干预 | 图结构和监督 Agent 的直接近邻；不是贝叶斯概率传播 |
| Node Contribution Backpropagation | 带方向的 MAS 通信 DAG 和最终决策 | 每个 Agent 对最终结果的贡献，定位/隔离可疑 Agent | 最接近“跨节点责任归因”；重点是贡献回溯，不是风险知识模板 |
| AgentMonitor（MAS） | 每个 Agent 的输入/输出统计 | 系统任务表现预测与安全修正 | 可作 MAS 级统计基线；粒度较粗，安全风险不是主目标 |

SentinelAgent 明确将 Agent 交互表示为动态执行图，并在节点、边和路径层面做语义异常检测；其案例包含邮件助手和 Magentic-One。[SentinelAgent](https://arxiv.org/abs/2505.24201) Node Contribution Backpropagation 则把通信表示为有向无环图，通过反向传播计算每个 Agent 对最终决策的贡献。[Node Contribution Backpropagation](https://arxiv.org/abs/2510.19420)

建议第一版主表至少包含：

- B1 Oracle/独立外部验收：不是算法，而是真实结果上限；
- B2 静态规则/策略门；
- B4 全轨迹 LLM-as-Judge；
- SentinelAgent；
- Node Contribution Backpropagation；
- B9 AgentMonitor（MAS）；
- B5 GuardAgent / B6 ShieldAgent；
- B7 AgentDoG；
- B8 AgentTrust；
- B14 朴素分数聚合；
- B15 序列或图模型；
- 我们的方法。

其中 MAS 原生方法和迁移方法的任务定义不同；B4、B5、B6、B7、B8 中有些是检测，有些是阻断，有些是解释。论文中必须分别报告“检测性能”和“干预后任务性能”，不能用一个总分把它们混在一起。

### 2.2 我们最关键的消融

B14 不是陪跑基线，而是最重要的科学对照：

- 每个 Agent 独立评分，再取 max；
- 取 mean；
- Noisy-OR；
- 按 Agent 角色加权；
- 只用文本，不用权限/状态/工具回执；
- 去掉跨节点关系；
- 去掉文献约束，只学参数。

如果我们的分层图有效，它应当在“局部都不明显、组合后才危险”的样本上明显优于这些聚合器。

## 三、数据集与环境候选

| 编号 | 数据集/环境 | 主要用途 | MAS | 可见证据 | 是否建议进入第一轮 |
|---|---|---|---|---|---|
| D1 | R-Judge | 安全风险判断 | 569 条多轮 Agent 交互记录；27 场景、5 类应用、10 类风险；安全标签和风险描述。 | 否；多轮记录但非协作系统 | 完整交互记录 | 最适合先做离线全轨迹 Judge/监测基线。 |
| D2 | Agent-SafetyBench | 广泛 Agent 安全 | 349 个交互环境、2,000 测试案例；8 类风险、10 类失败方式。 | 否；主要单Agent | 交互环境与安全测试 | 主安全集之一，覆盖面广。 |
| D3 | AgentDojo | 工具调用与间接提示注入 | 97 个现实任务、629 个安全测试；邮箱、银行、旅行等动态工具环境。 | 否 | 工具调用、环境状态、任务安全属性 | 主安全集之一，能测提示/工具链防御和外部效果。 |
| D4 | ATBench / AgentDoG | 长轨迹安全诊断 | 最新公开版约 1,000 条轨迹，503 safe/497 unsafe；平均约 9 轮；按 risk source、failure mode、harm 诊断。 | 否；可把轨迹拼成MAS输入 | 完整轨迹、工具池、延迟触发与人工审计 | 最适合比较细粒度根因诊断和解释。 |
| D5 | MCP Security Bench (MSB) | MCP 工具链安全 | 12 类 MCP 攻击；10 个领域、405 个工具、2,000 个攻击实例；覆盖规划、调用和响应，运行真实良性/恶意工具。 | 否；多服务器但非多Agent | 工具名称/描述/参数/响应、调用链 | 最接近 API 工具运行时的主安全集之一。 |
| D6 | MCP-SafetyBench | 真实 MCP 多步安全 | 5 个领域、20 类 MCP 攻击；多轮任务和跨服务器协作。 | 否；有跨服务器协调，非原生MAS | 真实 MCP 服务器、跨服务轨迹 | 补充 MSB 的攻击面和跨服务场景。 |
| D7 | A2ASecBench | Agent-to-Agent 协议安全 | A2A 协议攻击场景、fixtures、运行 harness 和评分。 | 是；直接 A2A/MAS | Agent Card、任务、资源引用、产物与生命周期 | 评测跨 Agent 身份、资源引用和任务状态。 |
| D8 | MT-AgentRisk | 多轮工具风险 | 将单轮危险任务转成多轮攻击序列，评估多轮工具使用风险，并提供 ToolShield 对照。 | 否 | 多轮请求、工具调用和风险结果 | 测试风险随多轮上下文累积的能力。 |
| D9 | AgentLeak | MAS 内部隐私泄漏 | 面向输出、Agent 间消息、记忆等内部通道的泄漏评测。 | 是 | 多 Agent 消息/记忆/输出及隐私结果 | 隐私子任务的直接数据集。 |
| D10 | τ-bench | 正常政策与外部状态 | 零售/航空多轮用户模拟、领域 API 工具和政策；以对话末数据库状态和 pass^k 评估。 | 否 | 用户、Agent、API 调用、政策、最终数据库状态 | 最适合正常合规轨迹、授权/参数/重复副作用对照。 |
| D11 | ToolSandbox | 状态依赖工具使用 | 有状态工具执行、工具间隐式状态依赖、用户模拟和任意轨迹的中间/最终里程碑评估。 | 否 | 工具调用、状态和中间里程碑 | 适合测节点状态传递、信息不足和外部状态变化。 |
| D12 | AppWorld | 多应用 API 正常环境 | 9 个日常应用、457 个 API、模拟人物和程序化状态/执行评测。 | 否；可构建多个 Agent 角色 | API 调用、程序状态、最终任务验收 | 适合长流程、跨应用和外部效果正常对照。 |
| D13 | AgentHarm | 有害任务结果 | 110 个恶意用户任务、11 类危害；多步工具 Agent 的有害完成评估。 | 否 | 任务、工具调用、危害结果 | 补充结果级 harmfulness。 |
| D14 | Monitoring Sequentially Contextual Harm | 组合意图监测 | 50 个 Agent 任务、8 类危害；将有害目标拆成看似正常的子任务，测试外部监测器。 | 否 | 有序子任务和整体意图 | 直接测试“局部正常、整体危险”的监测难点。 |
| D15 | WebArena / BrowserGym | 正常网页操作轨迹 | 多站点或统一浏览器 Agent 环境，提供长流程执行和状态评估。 | 否 | 网页操作、页面状态、轨迹 | 可作正常长链和权限边界对照。 |
| D16 | AgentBench / OSWorld | 通用能力与副作用对照 | 多环境 Agent 能力评测；OSWorld 提供桌面/web 任务和执行后状态。 | 否 | 多轮动作、环境状态 | 只用于正常行为/失败和外部状态对照。 |

### 3.1 推荐的最小组合

#### 针对双层图的主测试集选择

如果第一版只能选一个主测试集，我会优先核验 **HarnessAudit-Bench**：它直接记录完整执行轨迹，覆盖单 Agent 和多 Agent harness，并把资源访问、上下文泄漏、Agent 间传递等系统级安全约束放进运行过程里，最贴合“工作流 DAG + 动态执行图”的两层设计。论文已报告 210 个任务、8 个领域；正式冻结前仍需确认公开 release、许可证和复现实验脚本。[HarnessAudit](https://arxiv.org/abs/2605.14271)

但不建议只用一个数据集。最低可行的测试套件是：

1. **HarnessAudit-Bench：主系统级安全集**，检验整图风险、跨 Agent 信息流和运行时预警。
2. **A2ASecBench：通信/协议专项集**，检验委派、Agent Card、任务生命周期和跨 Agent 消息风险。[A2ASecBench](https://github.com/SaFo-Lab/A2ASecBench)
3. **τ-bench（或 AWS Multi-Agent Collaboration Scenario Benchmark）：正常任务对照集**，测任务成功率、授权遵循和误报率；它不是攻击集，但能证明监测器不会把正常协作都判成危险。[τ-bench](https://arxiv.org/abs/2406.12045)

如果论文强调“提前报警”，再加入 **AgentForesight 的 AFTraj-2K**；如果强调自然发生的协作失效，再加入 **MAST-Data**。前者适合评估最早报警步和责任定位，后者适合评估 MAS 失败模式泛化，但两者都不能单独代表完整安全集。[AgentForesight](https://arxiv.org/abs/2605.08715)、[MAST-Data](https://arxiv.org/abs/2503.13657)

若 HarnessAudit 的数据暂时拿不到，替代方案是 **MAST-Data + A2ASecBench + 我们自己生成一小批带工作流约束的场景**，但这时自建部分只能作为补充，不能冒充已有标准基准。

若算力和工程时间有限，第一轮不要全做。建议：

1. **R-Judge**：先验证离线轨迹风险判断和解释。
2. **AgentDojo**：验证工具调用、外部内容注入和真实任务状态。
3. **ATBench**：验证长轨迹和细粒度根因诊断。
4. **τ-bench**：补充正常政策遵循、授权、参数和状态对照。
5. **A2ASecBench**：若 release 可复现，作为 MAS/A2A 专项测试。
6. **MSB 或 MCP-SafetyBench 二选一**：若主系统采用 MCP，再加入；否则不应让 MCP 专属数据主导结论。

这一组覆盖：正常/危险、短链/长链、单 Agent/MAS、文本/工具/状态、攻击/政策违反/外部结果。

### 3.2 数据集选择的硬标准

一个数据集只有满足下列大部分条件，才适合做我们的主评测集：

- 有完整运行轨迹，而非只有 prompt 和 final answer；
- 有工具调用、参数和工具回执；
- 有程序化外部状态或独立结果验证；
- 有安全标签、风险机制标签，或可可靠补标；
- 同时包含正常与危险/失败对照；
- 能按任务模板或场景拆分，防止同一攻击模板泄漏到测试集；
- API 模型可以运行，或已有完整可复现轨迹；
- 许可证、版本和数据来源明确。

### 3.3 不要混用的三类数据

- **安全数据集**：能直接测危险检测，如 R-Judge、AgentDojo。
- **能力/合规环境**：能测正常工作和政策遵循，如 τ-bench、ToolSandbox、AppWorld。
- **攻击专项数据**：只覆盖某个机制，如 AgentLeak、MSB、MAPLE-Guard。

第三类很有价值，但不能拿它们单独宣称“整体 Agent 安全”。

## 四、统一评测协议

所有 baseline 必须看到同一组外部字段：

- 用户请求与任务目标；
- Agent 间消息；
- 工具名称、参数、返回值和调用顺序；
- 记忆读写；
- 权限、批准和策略版本；
- 时间、重试、并发和预算；
- 独立执行回执或环境状态（如果该数据集提供）。

禁止某个 baseline 额外读取我们的内部变量；也不能让我们的 BN 使用其他方法看不到的信息。

### 4.1 指标

| 目标 | 指标 |
|---|---|
| 风险识别 | precision、recall、F1、PR-AUC |
| 机制诊断 | 按风险机制/工作流节点的 macro-F1 |
| 早期预警 | time-to-detection、intervention lead time |
| 概率质量 | Brier score、ECE、可靠性图 |
| 正常任务影响 | safe-trajectory FPR、误阻断率、任务完成率 |
| 轨迹与事件 | 事件级召回 + 整轨迹 unsafe 召回 |
| 解释 | 根因与人工标签一致性、证据覆盖率 |
| 运行代价 | API 次数、延迟、token、每轨迹监测成本 |

安全率和任务成功率必须同时报告。近期生命周期评测也显示，“任务完成”并不能证明运行安全。[29]

### 4.2 公平划分

- 按任务模板/场景划分 train/dev/test，不按随机轨迹切分；
- 固定 Agent 框架、模型、工具集、预算、随机种子；
- 同一风险机制至少跨多个任务和多个工作流；
- 公开规则、提示词、模型版本和评测脚本；
- 对攻击数据报告攻击成功，不把它解释成现实发生率。

## 五、对我们算法定位的直接影响

这些 baseline 反向确定了我们的算法边界：

- 我们不是比谁更会完成任务，而是比谁更能**识别和解释运行风险**；
- 我们不能只输出一个总风险分数，至少还要输出风险机制、涉及节点和证据；
- BN 的价值应体现在概率校准、跨节点组合和不完整证据下的风险更新；
- “知识驱动”必须通过固定的风险模板、成立条件和证据来源体现，而不是只在引言里描述；
- MAS 的增益必须通过“朴素独立评分 vs 显式跨节点建模”实验展示；
- 如果只有文本日志，必须明确哪些标签只能判为未知；如果有执行状态，才可评估实际外部危害。

当前最合理的第一轮论文比较框架是：

> **SentinelAgent + Node Contribution Backpropagation + AgentMonitor（MAS）+ 规则门 + 全轨迹 LLM Judge + AgentDoG + 朴素跨 Agent 聚合 + 序列/图模型 + BayesTrace 原版 + 我们的方法**

## 七、近期补充候选（先登记，不进入第一轮主表）

以下工作与“监测器是否会被刻意规避”或“真正的 MAS 轨迹”很相关，但公开版本、标签质量或复现条件仍需单独核验：

- **Monitor Red Teaming（MRT）**：专门测试 Agent 隐瞒恶意副任务时监测器的可靠性，包含 SHADE-Arena 和 CUA-SHADE-Arena；适合作为监测器对抗鲁棒性压力测试，而不是普通安全分类集。[论文](https://arxiv.org/abs/2508.19461)
- **MAD / MAST-Data**：公开多 MAS 轨迹和 MAST 失败模式标签，适合训练或测试“从完整协作轨迹识别失败”；但主要标签是一般可靠性失败，不应直接改名为安全危害。[数据](https://huggingface.co/datasets/mcemri/MAST-Data)、[论文](https://arxiv.org/abs/2503.13657)
- **AFTraj-2K**：面向多 Agent 轨迹的安全/不安全与最早决定性错误步骤标注，适合研究提前预警和责任归因；公开数据的来源、版本和许可证需先核对。[数据](https://huggingface.co/datasets/ZBox008003/AFTraj)
- **SafeArena**：250 条安全任务与 250 条有害网页 Agent 任务，覆盖错误信息、违法活动、骚扰、网络犯罪和社会偏见；适合作为网页 Agent 结果级补充，不是 MAS 集。[论文](https://arxiv.org/abs/2503.04957)
- **ST-WebAgentBench**：222 个企业网页任务，每个任务带安全/信任策略，使用 Completion Under Policy 和 Risk Ratio；适合授权与政策维度的外部对照。[论文](https://arxiv.org/abs/2410.06703)
- **LlamaFirewall**：工程化的 PromptGuard、Agent Alignment Checks 和 CodeShield 组件，可作为工具化安全守卫对照；组件目标不同，不能当作统一风险概率模型。[项目](https://github.com/meta-llama/PurpleLlama)

这些候选不改变第一轮主组合；纳入论文主表前，要锁定具体 release、运行依赖、许可证和标签生成方式。

## 八、来源与证据范围

下列链接优先指向论文、会议页面、作者仓库或官方项目。近期预印本、作者仓库和专项数据使用前仍需锁定版本并核对许可证。

[^1]: Yuan et al.。*R-Judge: Benchmarking Safety Risk Awareness for LLM Agents*，2024。[原始来源](https://arxiv.org/abs/2401.10019)
[^2]: Zhang et al.。*Agent-SafetyBench: Evaluating the Safety of LLM Agents*，2024。[原始来源](https://arxiv.org/abs/2412.14470)
[^3]: Debenedetti et al.。*AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents*，NeurIPS 2024。[原始来源](https://proceedings.neurips.cc/paper_files/paper/2024/hash/97091a5177d8dc64b1da8bf3e1f6fb54-Abstract-Datasets_and_Benchmarks_Track.html)
[^4]: Liu et al.。*AgentDoG: A Diagnostic Guardrail Framework for AI Agent Safety and Security*，2026。[原始来源](https://arxiv.org/abs/2601.18491)
[^5]: Xiang et al.。*GuardAgent: Safeguard LLM Agents by a Guard Agent via Knowledge-Enabled Reasoning*，ICML 2025。[原始来源](https://arxiv.org/abs/2406.09187)
[^6]: Chen, Kang & Li。*ShieldAgent: Shielding Agents via Verifiable Safety Policy Reasoning*，2025。[原始来源](https://arxiv.org/abs/2503.22738)
[^7]: Yang。*AgentTrust: Runtime Safety Evaluation and Interception for AI Agent Tool Use*，2026。[原始来源](https://arxiv.org/abs/2605.04785)
[^8]: Chan et al.。*AgentMonitor: A Plug-and-Play Framework for Predictive and Secure Multi-Agent Systems*，2024。[原始来源](https://arxiv.org/abs/2408.14972)
[^9]: Naihin et al.。*Testing Language Model Agents Safely in the Wild*，2023。[原始来源](https://arxiv.org/abs/2311.10538)
[^10]: Chen & Cong。*AgentGuard: Repurposing Agentic Orchestrator for Safety Evaluation of Tool Orchestration*，2025。[原始来源](https://arxiv.org/abs/2502.09809)
[^11]: Ruan et al.。*ToolEmu: Identifying the Risks of LM Agents with an LM-Emulated Sandbox*，ICLR 2024。[原始来源](https://github.com/ryoungj/toolemu)
[^12]: Li et al.。*Unsafer in Many Turns: Benchmarking and Defending Multi-Turn Safety Risks in Tool-Using Agents*，2026。[原始来源](https://arxiv.org/abs/2602.13379)
[^13]: Zhang et al.。*MCP Security Bench (MSB): Benchmarking Attacks Against Model Context Protocol in LLM Agents*，ICLR 2026。[原始来源](https://arxiv.org/abs/2510.15994)
[^14]: Zong et al.。*MCP-SafetyBench: A Benchmark for Safety Evaluation of Large Language Models with Real-World MCP Servers*，2026。[原始来源](https://arxiv.org/abs/2512.15163)
[^15]: Lu et al.。*ToolSandbox: A Stateful, Conversational, Interactive Evaluation Benchmark for LLM Tool Use Capabilities*，NAACL 2025。[原始来源](https://arxiv.org/abs/2408.04682)
[^16]: Yao et al.。*τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains*，2024。[原始来源](https://arxiv.org/abs/2406.12045)
[^17]: Chen et al.。*Monitoring LLM Agents for Sequentially Contextual Harm*，ICLR Workshop 2025。[原始来源](https://mlanthology.org/iclrw/2025/yuehhan2025iclrw-monitoring/)
[^18]: Andriushchenko et al.。*AgentHarm: A Benchmark for Measuring Harmfulness of LLM Agents*，2024。[原始来源](https://arxiv.org/abs/2410.09024)
[^19]: Liu et al.。*AgentBench: Evaluating LLMs as Agents*，ICLR 2024。[原始来源](https://github.com/THUDM/AgentBench)
[^20]: Stony Brook NLP。*AppWorld*，2024。[原始来源](https://github.com/StonyBrookNLP/appworld)
[^21]: SaFo Lab。*A2ASecBench*，2026。[原始来源](https://github.com/SaFo-Lab/A2ASecBench)
[^22]: El Yagoubi et al.。*AgentLeak: A Benchmark for Internal-Channel Privacy Leakage in Multi-Agent LLM Systems*，2026。[原始来源](https://arxiv.org/html/2602.11510v3)
[^23]: Xiong et al.。*MAPLE-Guard: Memory-Aware Link Enforcement Against Memory-Link Poisoning in Multi-Agent Systems*，2026。[原始来源](https://arxiv.org/html/2608.00426v1)
[^24]: Dai et al.。*SafeFlow: Semantic Information-Flow Control for Blocking Malicious Propagation in Multi-Agent Systems*，2026。[原始来源](https://arxiv.org/html/2607.25255v1)
[^25]: MITRE。*MITRE ATLAS*，动态知识库。[原始来源](https://atlas.mitre.org/)
[^26]: MCP Project。*Security Best Practices*，draft。[原始来源](https://modelcontextprotocol.io/docs/draft/tutorials/security/security_best_practices)
[^27]: WebArena Project。*WebArena*，2024。[原始来源](https://github.com/web-arena-x/webarena)
[^28]: ServiceNow。*BrowserGym*，2024。[原始来源](https://github.com/ServiceNow/BrowserGym)
[^29]: Zhou et al.。*AgentS4D: Benchmarking Runtime Risks across the Execution Lifecycle of LLM-Based Workspace Agents*，2026。[原始来源](https://arxiv.org/abs/2607.27294)
[^30]: Kale et al.。*Reliable Weak-to-Strong Monitoring of LLM Agents*，2025。[原始来源](https://arxiv.org/abs/2508.19461)
[^31]: Cemri et al.。*Why Do Multi-Agent LLM Systems Fail?*，2025。[原始来源](https://arxiv.org/abs/2503.13657)
[^32]: ZBox。*AFTraj-2K*，数据集页面。[原始来源](https://huggingface.co/datasets/ZBox008003/AFTraj)
[^33]: Tur et al.。*SafeArena: Evaluating the Safety of Autonomous Web Agents*，2025。[原始来源](https://arxiv.org/abs/2503.04957)
[^34]: Levy et al.。*ST-WebAgentBench: A Benchmark for Evaluating Safety and Trustworthiness in Web Agents*，ICLR 2026。[原始来源](https://arxiv.org/abs/2410.06703)
[^35]: Meta Llama。*LlamaFirewall*，项目仓库。[原始来源](https://github.com/meta-llama/PurpleLlama)
