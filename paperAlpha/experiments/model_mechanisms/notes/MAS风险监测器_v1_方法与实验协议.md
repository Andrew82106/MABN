# MAS 风险监测器 v1：方法与实验协议

## 研究问题

在只能观察 API 请求、Agent 间消息、委派、工具调用、权限和外部回执的条件下，如何识别“单个 Agent 看起来正常、但跨 Agent 组合后形成风险”的运行轨迹，并给出可审计的风险证据链？

## 方法定义

### 规范图 W：工作流先验（不是风险传播层）

系统的角色、职责和允许的工作流构成规范 DAG。DAG 不是从标签反推的因果图，也不是模型自由生成的图；它来自系统配置、任务规范和风险知识库。每个工作流节点绑定一个局部风险模板，变量包括不可信输入、授权状态、记忆/状态变化、工具副作用和结果审核等。

### Layer 1：局部风险模板 BN

每个规范节点绑定一个角色/能力模板。模板只把公开运行时事实映射为局部风险证据；episode 级标签不复制成节点真值。当前原型用 episode 等权的 Laplace CPT + noisy-OR，后续升级为模板化 Beta-CPT，并在 episode likelihood 上直接训练。这样可以保留节点可读性，同时避免“大图自动获得更大监督权重”。

### Layer 2：动态执行图 G_t

运行时只把显式消息、委派、任务父子关系作为边；这些边描述实际观察到的行为，不宣称是因果边。W 只提供 expected/unexpected、缺失前置条件和能力偏离等规范特征，不能替代 G_t 中不存在的事件。节点和边携带来源、数据类别、权限、工具和回执等证据。系统级特征包括跨 Agent 信息流、权限传播、共享状态变化、循环/深度和资源累积。

### 风险计算

局部模板使用带 Laplace 平滑的条件概率和 noisy-OR 组合。动态层使用可审计的边传播/门控，最终输出系统风险概率、最早预警时间、涉及 Agent、触发证据和解释路径。参数只能在训练折估计；测试集不参与结构或阈值选择。

对于较深工作流，raw product 会随边数产生非语义的数值衰减；升级版对候选 source-to-sink 路径使用边门控的 geometric mean，并报告观测覆盖率。长度归一化只能消除尺度偏差，不能恢复 API 日志中根本没有的边；缺失边时应降低置信度并单独报告可观测性上限。

在部分可观测场景，监测器同时计算：

- lower bound：只把明确标为不可信的边纳入传播；
- upper bound：把 unknown/redacted 边视为“可能携带风险”；
- abstain：上下界跨过报警阈值时，不输出虚假的确定判断，而是请求额外审计或人工复核。

### 端到端两层实现（Unified v2）

第一层对每个 Agent 的局部证据向量 `x_i` 计算
`q_i = sigmoid(b_i + w^T x_i)`。`x_i` 只包含可命名的运行时事实，例如不可信输入、特权工具调用、清洗事件、异常边、向外发送的不可信数据和收到的不可信数据。这样每个局部风险都有明确的变量来源。

第二层对显式消息/委派边 `e=(i,j)` 计算语义门控
`g_e = sigmoid(a_0 + a_1·untrusted + a_2·unexpected + a_3·confidence)`，并用 noisy-OR 在动态图上迭代传播：
`s_j <- 1 - (1-s_j) Π_e(1-s_i g_e)`。最终系统风险为
`p = sigmoid(c_0 + c_path·max_sink(s) + c_local·max_i(q_i) + c_hard·I(path) + c_dist/(1+d))`，其中 `I(path)` 是已观测不可信边的显式路径证据，`d` 是最短已观测路径长度。硬路径和距离项与软传播分开保留，便于消融和解释。

所有参数只在训练折用 BCE + L2 学习；标签字段、`label_evidence` 和终态违规字段不进入输入。输出同时保留 `q_i`、`s_i`、高分边/路径和风险概率，因而“局部为什么危险”和“风险如何跨 Agent 传播”可以分开审计。这个传播是证据聚合，不是因果发现。

对 `unknown/redacted` 边同时计算 lower（未知按不可用）和 upper（未知按可能携带不可信数据）两种风险；两者跨越报警阈值时 abstain。每次正式实验还需运行字段篡改审计，确认修改标签相关字段不会改变模型分数。

## 评测协议

1. 训练/测试按任务、case 或攻击族分组，禁止同一轨迹变体跨集合。
   对合成数据还必须按完整拓扑/模板族留出；随机 episode 切分只能作为开发诊断。
2. 终态违规事件不能同时作为输入特征和标签依据；若标签由同一事件流派生，结果只能标为 adapter sanity check。
3. 必须同时报告 precision、recall、F1、AUROC/AUPRC、Brier/ECE、误报率和提前预警时间。
4. MAS baseline 分开报告：逐 Agent 聚合、通信图统计、节点贡献回溯、全轨迹评审和规则策略门；没有源码时只能称 protocol-compatible proxy。
5. 合成数据只验证机制，不替代真实人工标注轨迹。真实结论需要独立安全谓词、正常对照和外部效果回执。

## 当前证据

- A2ASecBench：完成 900 episode 的 API/A2A 事件适配和重放；标签与事件定义存在耦合，因此不作为无泄漏 SOTA 证据。
- compositional_mas_v1：2,000 个受控 episode，标签由不可信入口到特权工具的结构可达性生成，输入不含终态风险事件。动态图模型 F1=1.000，flat logistic=0.608，single-agent BN=0.558；这是组合机制 sanity check。
- UnifiedLocalBNGraphMonitor-v2：在同一基准上将局部 BN 与可学习动态图传播端到端联合；完整轨迹 F1=1.000，但 25%/50%/75% 前缀 F1=0.000/0.173/0.813，显示提前预警仍是独立难点。
- independent_mas_v2：标签来自隐藏的未授权外部副作用，观测 provenance 带随机误报/漏报，工具执行前的 permission mismatch 是额外的带噪授权证据；在修正 episode 等权 CPT、移除事件级终态字段后，3,000 条公开轨迹上 TwoLayer v2 F1=0.693、AUROC=0.904、AUPRC=0.774、Brier=0.098，动态规则 AUROC=0.801。该基准仍为合成数据，但比“标签等于可达性规则”的旧基准更适合检验学习监测器。
- independent_mas_v2 的公开轨迹与评测标签物理分离，运行时监测器只读 `traces_public.jsonl`；`labels.jsonl` 仅在评测阶段按 episode_id 合并。
- independent_mas_v2 的拓扑族留出子集（300 条、训练 262/测试 38）F1=0.500、AUROC=0.738，作为更严格的泛化参考。
- 统一公平 group split（409/102 拓扑族）上，graph-features learned F1=0.775/AUROC=0.930；修正后的 TwoLayer BN+graph v2 F1=0.663/AUROC=0.881、AUPRC=0.722。当前模型的优势主要在结构化解释和观测区间，检测分数仍需继续提升。
- 不可辨识性控制将标签随机置换后，graph-features learned AUROC=0.468、AUPRC=0.205，接近随机水平；该控制必须保留在正式评测中。
- Fusion v3 将局部 BN+消息传播、全图上下文统计和观测覆盖率送入训练折内的线性融合头；修正后原始 group split 上 F1=0.753、AUROC=0.933、AUPRC=0.744。组级 OOF 融合后 F1=0.752、AUROC=0.922、AUPRC=0.682、ECE=0.060；正式结果优先采用 OOF，不应被表述为全面超过 graph baseline。
- 固定按 family 排序留出的 split 主要集中在 n=8，不能单独代表跨拓扑泛化。随机 3 次 group holdout（runtime-only）上，graph-features learned F1=0.778±0.027、AUROC=0.918±0.039；TwoLayer BN+graph v2 F1=0.721±0.056、AUROC=0.891±0.043。屏蔽机制特异的 `permission_mismatch` 后，TwoLayer F1=0.711±0.045、AUROC=0.798±0.047；该 parity 设置应作为更公平的主参考。
- `HierarchicalNormativeRuntimeMonitor-v3` 将规范 `workflow_dag` 与运行时消息/委派图分开。由于 independent_mas_v2 中每条规范边都被完整记录为运行时边，规范偏离特征没有真实变异；因此该结果（修正后 F1=0.762、AUROC=0.940）只能作为 workflow-known 原型，不能证明规范 DAG 本身带来增益。
- `independent_mas_v3` 包含 4,000 条轨迹、5 类隐藏机制，规范边会缺失且运行时会出现意外边；事件时间遵循前向链。TwoLayer 的随机划分 F1=0.413、AUROC=0.717，表明当前模型尚未覆盖更一般的风险机制；workflow-known 融合在 n=8 为主的留出上 F1=0.650、AUROC=0.791，仍需随机 group-CV 与 OOF 融合验证。
- v3 机制分解中，`taint_path`/`collusion` AUROC 约 0.75，`privilege_only` 约 0.61，`unexpected_edge` 约 0.55。后两类被列为下一轮模板扩充优先级：能力/授权不匹配与规范外边偏离，不能用污点传播特征替代。
- Conformal v1 在拓扑族隔离校准/测试上以 alpha=0.10 输出预测集合，边际覆盖率=0.890、平均集合大小=1.125；它把“未知”显式保留下来，但保证依赖交换性，不能替代真实安全保证。

旧 compositional/stress 结果经过审计后只能标为 rule-consistency/oracle mechanism check：如果标签构造直接使用了输入中的 `data_class` 或路径，不能把满分解释为检测性能。
- MAS-specific baseline v1：Agent max/mean、拓扑、边数和无污点贡献回溯均低于显式污点路径模型；这些是文献方法的协议兼容代理，不是原论文源码复现。

## 不应作出的声明

- 不声称发现了真实因果图。
- 不声称 API 文本监测恢复了模型内部白盒机制。
- 不把合成基准上的 1.000 写成真实部署性能或 SOTA。
- 不把“模型说要做”当作“外部动作已经发生”。

## 下一步

将动态传播从硬规则升级为训练折内学习的、受拓扑和风险知识约束的边门控；加入独立人工标注或外部状态结果，验证校准、提前预警和跨工作流泛化。
