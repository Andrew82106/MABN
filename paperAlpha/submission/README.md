# 投稿实验执行入口

本目录只维护一套投稿主方案。目标是面向 API 多智能体系统的风险监测，保留规范知识与工作流、实际执行证据两条线，并让监测器的事件含义和概率计算可核查。被监测的大模型本身仍是黑盒。

用户明确要求中科院一区或二区，不能用 JCR 或 SJR 代替。期刊分区年份、来源可信度以及未核验项保留在文献目录中。实验规模与统计方法依据我们的研究主张设计，不把某篇论文的样本数说成期刊统一要求。

目前优先核对 ESWA 的主题匹配；具体分区仍待学校采用年份的官方条目确认。Computers & Security 的当前官方介绍明确排除 AI/ML 及 LLM 安全相关稿件，已从投稿候选中移出；其论文只保留为方法参照，不能因已有相关文章就默认现在可投。

## 当前工作和最终交付

当前主开发结果为 `results/independent_mas_journal_v1/`。4,000 条自建合成轨迹按 186 个 topology family 做五折 StratifiedGroupKFold；融合头使用内层 OOF 分数，避免把基础模型的训练内预测当作无偏证据。两线融合得到 F1=0.668、AUROC=0.820、AUPRC=0.675、Brier=0.160；同信息运行时逻辑回归为 F1=0.653、AUROC=0.805。该结果说明规范 DAG 与运行时证据互补，但仍是合成基准开发证据，不能宣称全面胜出或替代真实独立 MAS 测试。字段公平性复核、聚类区间和解释路径反事实审计已补齐，仍不等于独立外部确认。

- [执行协议](protocol.json)定义研究对象、两条线、预测任务、数据与统计边界，以及 E1 至 E7 验收要求。
- [修正版五折主评测](../results/independent_mas_journal_v1/REPORT.md)给出 family-disjoint OOF 主表、固定 5% FPR 结果、机制分解与聚类 bootstrap；[动作前前缀评测](../results/independent_mas_prefix_v1/REPORT.md)单独报告早期预警检查。
- [首次报警/提前量审计](../results/submission/final_eval/prefix_alarm_v1/REPORT.md)报告固定误报率下的动作前及时召回、误报和提前事件数。
- [不确定性与覆盖率审计](../results/submission/final_eval/uncertainty_coverage_v1/REPORT.md)报告未知证据区间、拒识率和 coverage-risk 曲线。
- [在线监测缩放审计](../results/submission/final_eval/online_scaling_v1/REPORT.md)报告不同事件宽度下的 monitor-side p50/p95、内存和说明边界。
- [MODEL_SPEC](../results/independent_mas_journal_v1/MODEL_SPEC.md)固定两层 BN、融合、未知证据和白盒边界。
- [结果清单与 SHA-256](../results/independent_mas_journal_v1/MANIFEST.json)固定运行命令和当前候选证据版本。
- [观测缺失分层](../results/independent_mas_journal_v1/strata.json)和 [CPU 推理延迟/内存](../results/independent_mas_journal_v1/latency.json)是补充诊断，不把 API 生成和网络耗时伪装成监测器耗时。
- [基线清单](baselines.json)记录作者实现、任务边界、数据可见性、许可证和适配差异，不把消融当论文基线。
- [跨 Agent 约束](relational_contracts.json)记录对象绑定、累计额度和执行顺序的原始依据；它们是明确标注的扩展，尚未当作原基准标签使用。
- [期刊与相关工作目录](../../doc/ref_paper/mas_safety_2026-09-11/journal_alignment/catalog.json)区分正式期刊参照、直接可比算法和仅有摘要的材料。
- [投稿实验要求固定表](../../doc/ref_paper/mas_safety_2026-09-11/journal_alignment/submission_requirements_20261002.md)把 ESWA/Elsevier/TDSC 官方要求映射到当前实验验收项。
- [当前主评测结果](../results/submission/final_eval/RESULTS.md)包含最终候选模型与所有声明基线的同划分统计；[基线映射](../results/submission/final_eval/BASELINE_MAPPING.md)说明哪些是可直接比较的算法、哪些只是文献参照。
- [最终实验门槛审计](../results/submission/final_eval/FINAL_GATE.md)是当前投稿候选的单页结论；历史运行只作为可复现原始证据保留。
- [学习型 MAS 基线](../results/independent_mas_learned_baselines_v1/REPORT.md)和 [透明结构/污点规则控制](../results/independent_mas_mas_baselines_v1/REPORT.md)已按同一五折 topology-family holdout 运行；它们是协议兼容的 proxy/control，不冒充外部论文复现。
- [字段公平性复核](../results/independent_mas_journal_parity_v1/REPORT.md)去掉运行时 `permission_mismatch` 后重跑同一协议；双线融合 AUROC/AUPRC=0.811/0.661，说明主结果不是由该单字段单独造成的。
- [全方法 family-cluster bootstrap](../results/submission/final_eval/all_methods_ci_v1/REPORT.md)为主模型、学习型基线和规则控制统一给出 95% 区间；其中 F1@0.5 仅是补充诊断，主表仍使用 fold-local threshold。
- [解释路径反事实审计](../results/submission/final_eval/explanation_counterfactual_v1/REPORT.md)在 4,000 条 OOF 轨迹上检查最高风险路径是否真的影响分数；这是可核查性/faithfulness 诊断，不是因果归因。
- [独立基准发布校验](../results/submission/final_eval/release_validator_v1.json)确认 trace/label ID 精确对应、无重复、无额外标签和无禁用公开字段。
- [QuadSentinel 外部基线重放](../results/submission/development/quadsentinel_replication_20261001/REPORT.md)已完成 12 个 HarnessAudit 派生 episode；15 个 covered action 的 precision/recall 均为 1.000，但整体覆盖率只有 5.19%，目标与本文 future-risk 任务不同，因此只作外部边界/控制，不并入主表。
- 真实 API 阶段前瞻开发评测见 [开发报告](../results/submission/development/api_stage_forecast_20261001_v6/REPORT.md)，可检查模型参数见 [MODEL_SPEC](../results/submission/development/api_stage_forecast_20261001_v6/MODEL_SPEC.json)，三任务同任务重跑见 [复现报告](../results/submission/development/api_stage_forecast_replication_20261001_v6/REPORT.md)。两者都不是最终独立测试集。
- 扩展集之外的三任务跨任务检查见 [transfer3 报告](../results/submission/development/api_stage_forecast_transfer3_20261001_v6/REPORT.md)；该检查仍是事后开发证据，不是预先冻结的最终确认集。
- 当前真实 API 驱动的 MAS 运行与四攻击族对照见 [A2ASecBench API 确认队列](../results/submission/development/a2asecbench_api_final_20261002/REPORT.md)；80 个 episode 已统一适配、做两层融合和前缀预警评测，但标签仍来自参考运行器，不能冒充独立最终测试。
- A2ASecBench 的严格预注册角色审计见 [strict role audit](../results/submission/development/a2asecbench_api_final_20261002/strict_role_audit_v1/REPORT.md)：排除 `observed_violation` 和 outcome metrics 后，80 episode 的两线 AUROC/AUPRC=0.788/0.835；仍属于 benchmark-design 标签，不是人工独立确认集。
- [外部 API 迁移检查](../results/submission/development/external_api_mas_v1/REPORT.md)使用八个已完成的三智能体真实 API 轨迹，四个独立参考策略阳性；hierarchical fusion 的 AUROC=0.750、AUPRC=0.800。样本很小且同一网关，只作为迁移证据，不作为期刊主结果。
- [平衡 API MAS 迁移检查](../results/submission/development/balanced_api_mas_transfer_v1/REPORT.md)使用 8 个预先固定的双智能体场景（4 安全/4 违规），采集审计确认 8/8 完成且无标签泄漏；hierarchical fusion 的 AUROC=0.750、AUPRC=0.750。样本仍很小，只用于外部迁移与校准诊断。
- 扩展平衡 API cohort 的冻结迁移评测见 [v2 报告](../results/submission/development/balanced_api_mas_transfer_v2/REPORT.md)：12 个预设场景中 11 个完整、1 个失败记录保留；11 个完整案例中 5 个参考策略阳性，hierarchical fusion AUROC=0.700、AUPRC=0.673。它是容量/失败覆盖补充，不是第二端点或最终确认集。
- [本地生成模型迁移诊断](../results/submission/development/model_transfer_local_v1/REPORT.md)用 `gpt-5.5/low` 重跑固定 8 案例，8/8 完成、46/46 请求成功；两线融合 AUROC/AUPRC=0.750/0.750。由于固定模拟器产生相同公开投影，这只是模型迁移审计，不是第二端点确认。
- [Lanyun 第二端点迁移诊断](../results/submission/development/balanced_api_mas_transfer_lanyun_v1/REPORT.md)使用 `glm-5.3-flash`，12 个预注册案例中 10 个完成、2 个阶段失败记录保留；两线模型排序 AUROC/AUPRC=1.000/1.000。该结果首次证明第二生成端点链路可运行，但 cohort 仍小、标签来自同一模拟器，不能替代独立确认集。
- [Lanyun 扩展第二端点诊断](../results/submission/development/balanced_api_mas_transfer_lanyun_r14/REPORT.md)预注册 40 个案例，26 个完成、14 个失败原样保留；9 个完成案例为参考策略阳性。冻结 runtime BN AUROC/AUPRC=0.944/0.927，双线融合=0.889/0.927；196 次请求的传输 p50/p95=9.688/36.774 秒。样本仍来自同一模拟器，不能替代独立确认集。
- [Lanyun 前执行策略风险扩展](../results/submission/development/lanyun_knowledge_attempt_r14/REPORT.md)在同一 40 个可观测前缀上单独预测“请求是否越权/使用已撤销授权”，把拒绝执行的 scope conflict 也保留为风险样本；公开知识规则 F1/AUROC/AUPRC=0.865/0.867/0.843。该规则只读公开授权、撤销和消息绑定证据，是目标特定的知识线，不是通用安全结果。
- [Lanyun 扩大迁移诊断](../results/submission/development/balanced_api_mas_transfer_lanyun_r15/REPORT.md)预注册 80 个案例，64 个完成、16 个失败原样保留；24 个完成案例为参考策略阳性。冻结 runtime BN AUROC/AUPRC=0.872/0.827，双线融合=0.854/0.824；416 次请求、459,902 tokens，传输 p50/p95=7.429/28.555 秒。它仍是同一手工模拟器和事后策略标签，不能替代独立确认集。
- [Lanyun 扩大前执行策略风险扩展](../results/submission/development/lanyun_knowledge_attempt_r15/REPORT.md)在 80 个可观测前缀上预测越权/撤销意图；公开知识规则 F1/AUROC/AUPRC/Brier=0.947/0.950/0.950/0.055。该结果是目标特定的知识线，不是通用 MAS 安全结果。
- [知识编译的绑定一致性扩展](../results/submission/development/binding_consistency_extension_v2/REPORT.md)只读取公开的工作流、授权、消息和动作事件，用固定可审计规则补充对象/权限绑定检查；扩展平衡 cohort 的 11 个完整案例 F1/AUROC/AUPRC=0.750/0.800/0.782，旧 branched cohort 为 0.889/0.969/0.950。另有 1 个 API 失败案例原样保留、未计入指标。它是非主结果诊断扩展，不改变冻结的两线 BN 主表，也不替代第二端点和更大独立确认集。
- [API 成本与失败覆盖摘要](../results/submission/development/api_cost_report_v1/balanced_r8.json)记录上述 cohort 的请求数、token、延迟和失败运行；只作开发审计，不把价格或 API 传输时间伪装成监测器推理成本。
- 独立基准的前缀/动作前检查见 [prefix 报告](../results/independent_mas_prefix_v1/REPORT.md)；`pre_action` AUROC=0.735，但它仍是开发性合成检查，字段可用时刻还需在真实 API MAS 上确认。

`results/submission/final_eval/RESULTS.md` 现在只作为当前候选的单一指针，集中链接修正版主表；第二端点链路已验证，但在更大独立真实标签确认集、完整在线成本/延迟和 E1-E7 验收补齐前，它不代表最终投稿结论。不会用代码测试通过、几条满分案例或拼接各版本最好分数代替最终结果。

本轮收尾新增两项可复查对照：同信息的衰减式 agent reputation（CogTrust 代理）和带 JSON mode/一次重试的语义抽取成本审计。它们分别写入主结果表和 E7 记录；语义抽取仍是独立的 abstain 模块，不进入风险指标。

另已冻结通信方式分层队列：`scripts/collect_frozen_communication_queue_v1.py` 覆盖四种无环拓扑与 direct/summary/vote 三种通信契约，作为下一轮 API 条件实验入口；未执行前不计入主结果。

## 保留证据而不保留多个活跃方案

原始调用、失败记录及冻结源码属于实验依据，不能为目录简洁而销毁。它们原位保存；主 README 只指向本执行入口。旧代码先退出活跃导入和默认命令，再按依赖清单做可恢复归档，避免破坏正在运行的采集或历史复验。
