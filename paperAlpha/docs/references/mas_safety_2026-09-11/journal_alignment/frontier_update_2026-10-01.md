# 目标期刊近邻更新（2026-10-01）

本页只记录出版社页面可核验的论文事实；中科院分区仍以官方分区库为准，不能用 JCR 或搜索结果替代。

## 最接近的系统级参照

**A multi-agent LLM framework with Bayesian fusion and safety guardrails for ATC-pilot communication error detection**（Expert Systems with Applications, DOI: [10.1016/j.eswa.2026.132241](https://doi.org/10.1016/j.eswa.2026.132241)）。论文使用四个专门角色（句法、语义、上下文、风险），再用规则和可靠性加权的 Bayesian fusion 汇总；出版社摘要报告 894 条专家标注真实交流、400 条合成压力样本，重复严格留出下约 90.6% recall，误报率约 2.4%–4.6%，平均延迟约 1.6 秒。它是我们最应该对照的“多Agent + 贝叶斯融合 + 安全约束”参照，但任务是通信错误检测，不是通用 API-MAS 运行时风险预警，因此只能复用实验设计和指标，不能直接混合 F1。

## 需要纳入最终对照设计的近邻

**Beyond the prompt: Log-based threat detection and attribution for multi-Agent LLMs**（Information Processing & Management, DOI: [10.1016/j.ipm.2026.104768](https://doi.org/10.1016/j.ipm.2026.104768)）把每次运行转换为带属性的有向多图，用图级检测器做攻击识别、前缀早检和 agent/交互归因。它要求我们增加 prefix 曲线、跨 agent 关系消融和逐事件解释，而不能只报告整条轨迹的 AUROC。

**Separating intent from execution: A defense-in-depth security architecture for LLM-based multi-agent systems**（Expert Systems with Applications, DOI: [10.1016/j.eswa.2026.133781](https://doi.org/10.1016/j.eswa.2026.133781)）用多个 API、多个架构和多种攻击测试“语义意图”和“确定性执行门”分离的价值。它支持本项目保留双线结构：规范/工作流线表达意图与授权，运行/消息线验证实际执行；最终实验要分别报告两线及其融合。

**Enhancing robustness of LLM-driven multi-agent systems through randomized smoothing**（Chinese Journal of Aeronautics, DOI: [10.1016/j.cja.2025.103779](https://doi.org/10.1016/j.cja.2025.103779)）代表黑盒 MAS 防御的鲁棒性路线：不查看模型权重，而是在输入扰动下给决策提供统计保证。它提示我们增加通信扰动、agent dropout、消息缺失和异步执行压力集，并报告性能下降。

## 目标期刊对实验的共同要求信号

Information Fusion 的近期综述 **Security of LLM-based agents regarding attacks, defenses, and applications**（DOI: [10.1016/j.inffus.2025.103941](https://doi.org/10.1016/j.inffus.2025.103941)）把攻击、防御和安全应用放在统一评价标准下，并明确指出 agent 的多步执行会扩大攻击面。对我们最重要的启示是：必须同时交代威胁覆盖、检测指标、适应性攻击/跨拓扑泛化、解释性和部署成本。

Safety Science 的综述 **From hallucinations to hazards: benchmarking LLMs for hazard analysis in safety-critical systems**（DOI: [10.1016/j.ssci.2025.107056](https://doi.org/10.1016/j.ssci.2025.107056)）特别强调重复运行的一致性、因果/危险分析质量和不确定性处理。它支持我们保留多次分组重复、校准、低误报预算召回和最早报警时间，而不能只报一次随机划分的准确率。

## 对本项目实验表的直接影响

1. 主比较要有：纯规则、单Agent/扁平模型、只用执行线、只用规范线、去掉关系结构、匹配 GLM/树模型，以及完整双线模型。
2. 主指标至少包括：AUROC、AUPRC、Brier/ECE、固定误报率下的危险召回、首次报警提前量、延迟和调用成本。
3. API 实验必须把“采集链路审计通过”和“检测性能通过”分开；没有独立的动作前真值，就不能把真实 API 运行直接写成准确率结果。
4. 当前项目的 API 确认批次只证明采集和审计链路可运行，不能替代独立 MAS 风险测试集。
5. 最终投稿包应把四个拓扑（chain/fork/join/review）、至少两种 API/model 条件、通信方式和攻击/正常比例固定在同一协议中；确认集标签必须由独立标注者给出，不能把 simulator policy-intent 当作安全金标准。
