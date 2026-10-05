# Submission Packet（截至 2026-10-05）

## 一句话状态

这是可复现的 **development / external-diagnostic package**，尚未达到“独立人工确认、可直接投稿”的门槛。所有数字均来自 `FINAL_RESULTS.md`、`protocol.json` 及最新独立确认/MAST 审计；不得把条件迁移结果写成独立安全金标准。

## 方法

本文方法是 **two-line knowledge-compiled conditional BN factorization（Knowledge-compiled relational event risk monitor）**：

- 规范/工作流线：角色、能力、权限、工作流边和危险汇点；
- 运行时线：消息、工具调用、权限不匹配、异常边、冲突、验证与终止事件；
- 两线先分别评分，再以训练折内 OOF 分数融合。监测器自身变量、关系、概率因子和证据路径可检查，但不读取模型权重或隐藏状态；“白盒”不等于被保护 LLM 白盒化。
- 这里的 BN 指显式 DAG、条件风险因子与 noisy-OR/融合组合；它不是因果发现，也不声称被保护 LLM 内部实际采用同一计算图。

## 主结果（`independent_mas_v3`）

4,000 条公开 MAS 轨迹、1,345 条正例、186 个拓扑族；按拓扑族隔离的五折 StratifiedGroupKFold，阈值仅在训练折确定。

| 方法 | F1 | AUROC | AUPRC | Brier |
|---|---:|---:|---:|---:|
| runtime logistic | 0.653 | 0.805 | 0.653 | 0.167 |
| graph-features logistic | 0.578 | 0.724 | 0.527 | 0.194 |
| local-only logistic | 0.564 | 0.688 | 0.485 | 0.204 |
| AgentMonitor-style statistics + logistic（目标适配） | 0.581 | 0.712 | 0.503 | 0.197 |
| reliability-weighted Bayesian fusion | 0.604 | 0.763 | 0.596 | 0.459 |
| **two-line BN fusion** | **0.668** | **0.820** | **0.675** | **0.160** |

主模型族聚类 95% CI：AUROC [0.804, 0.835]、AUPRC [0.646, 0.707]。相对 reliability-weighted fusion 的成对 bootstrap 差值：ΔAUROC 0.056 [0.045, 0.067]，ΔAUPRC 0.079 [0.061, 0.098]；2,000/2,000 次重采样为正。相对最强同信息 trust/reputation proxy 的 ΔAUROC/ΔAUPRC 为 0.090 [0.077, 0.103] / 0.176 [0.153, 0.200]；相对 graph-features logistic 为 0.096 [0.084, 0.108] / 0.147 [0.127, 0.167]。

按训练折负类 5% 固定误报点，主模型测试 FPR 0.052、召回 0.303；ECE 0.019、Brier 0.160。该低误报点召回仍不高，因此不能把结果写成稳定的在线预警保证。

## MAS 基线

同一数据、同一五折的结构对照：per-agent mean（AUROC 0.480）、topology-only（0.545）、no-taint contribution（0.573）、dynamic taint path（0.597）、AgentMonitor-style statistics（0.712）、trust/reputation risk（0.730）；本文方法 0.820。AgentMonitor-style 使用原论文公开的统计特征族，但目标改为本文风险标签，属于同信息适配而非原方法复现。上述为同信息代理基线。ALTEDA 使用主机/网络日志，属于 richer-observation 上界参考，不可直接混比。

AgentMonitor-style 适配基线只使用其公开的逐 Agent 活动统计与工作流图统计，得到 AUROC 0.712、AUPRC 0.503、Brier 0.197、F1 0.581；这是同信息适配，不是原论文安全目标复现。完整记录见 `results/submission/development/agentmonitor_style_baseline_20261005/REPORT.md`。另有同信息强基线审计显示，平面 Logistic 的 AUROC 与本文双线 BN 接近，因此本文不宣称全面击败所有同信息模型；见 `results/submission/final_eval/MATCHED_INFORMATION_AUDIT.md`。

另有一个严格同信息的 zero-shot LLM judge 对照：LANYUN `glm-5.3-flash` 对 4,000 条公开轨迹最终全部返回有效预测；judge 的 AUROC/AUPRC/F1/Brier 为 0.771/0.578/0.571/0.317，本文双层 BN 为 0.820/0.675/0.634/0.160。它是全量辅助 baseline，不是已发表方法复现，也不支持普遍优越性结论；完整协议和重试账本见 `results/submission/final_eval/LLM_JUDGE_BASELINE.md`。

## LANYUN 完整通信队列

冻结队列为 `qwen3.6-flash`，120 episodes（chain/fork/join/review × direct/summary/vote × 每格 10），330/330 请求完成；输入/输出 token 61,629/259,326，传输延迟 p50/p95 为 6.078/9.718 秒，策略意图 evaluator 准确率 1.000。接入运行时语义子图后：AUROC 0.992、AUPRC 0.987、F1 0.968、Brier 0.039；拓扑 AUROC：chain .980、fork .993、join 1.000、review 1.000；通信方式：direct .994、summary 1.000、vote .981。

这些标签是手工 policy-intent evaluator，不是独立人工安全金标准；因此只能作为 API 条件迁移/通信鲁棒性证据。结构/tool-sink 旧投影在文本队列上为 AUROC/AUPRC 0.500/0.500，语义子图修复的是可观测边界，不是普适安全准确率。

同一 LANYUN authority-payment 队列中，前 10 个 episode 与语义抽取账通过 episode ID 对齐：串行 API+语义服务时间估计 p50/p95 为 47.386/83.429 秒；同一 monitor projection 的本地语义匹配+图特征+BN 推理 p50/p95 为 0.691/0.996 ms。该测量覆盖 10/80 episode，明确排除训练和并发调度，不能外推为完整部署延迟。

## 外部数据集与审计边界

- **MAST/MAD full**：1,642 条轨迹按 MAS×benchmark 分组；修正版适配器剥离末尾 `Evaluation` 标签块后，结构线 AUROC/AUPRC **0.295/0.675**，加入固定语义知识线后 **0.468/0.763**，Brier **0.189**。这是公开标注的二次迁移审计，不是新的人类金标准；旧版 0.681/0.852 仅保留为历史诊断，不作为有效外部结果。
- **A2ASecBench**：严格角色审计 AUROC/AUPRC 0.788/0.835；标签是 benchmark 角色定义，不是人工安全裁决。
- **AgentLeak**：4,258 条轨迹、937 个 request group；独立留组 late fusion AUROC/AUPRC 0.561/0.353，显示真实泄露流迁移较弱。
- **ATBench**：1,000 条人审过的通用 Agent 轨迹（497 unsafe/503 safe）作为非 MAS 外部迁移；排除所有标签与风险解释字段后，加入四类知识层风险模式，双线 AUROC/AUPRC **0.608/0.582**、Brier **0.257**、F1@0.5 **0.173**。这仍是弱迁移边界，不能宣称普适安全检测。
- **Who-and-When**：126 条 Algorithm-Generated failure-localization 样本全部为失败样本，只作外部响应边界审计；监测器分数平均变化 **+0.039**，上升率 **5.6%**，首报警率 **19.8%**，不作为二分类安全结果。
- **低阳性率压力**：将主集 OOF 结果重加权到 5%/10% 阳性率时，两线 AUROC 保持 **0.820**，AUPRC 分别为 **0.201/0.336**；这是 prevalence-shift 诊断，不是新测试集。
- A2ASecBench API、HarnessAudit、本地/LANYUN authority-payment 队列均为外部/迁移诊断，不能拼接冒充独立确认集。MAST 人工标注子集仅 19 条，也不足以替代新确认队列。

## 期刊对照

首选主题候选为 **Expert Systems with Applications**（风险评估、MAS、知识管理方向匹配）；这只是主题匹配，不代表录用或 CAS 资格。MAS 威胁/拓扑参考 IEEE TDSC *Cracks in Collaboration*；MAST 使用其官方仓库。不得用 JCR/SJR 代替学校采用版本的 CAS 官方核验；`Computers & Security` 当前官方范围被协议列为不适合作为本轮目标。逐项对照见 `doc/ref_paper/mas_safety_2026-09-11/journal_alignment/target_comparison_20261005.md`，最新近邻刷新见 `doc/ref_paper/mas_safety_2026-09-11/journal_alignment/near_neighbor_refresh_20261005.md`。

## 不可宣称

不可宣称：因果发现、绝对安全、自动理解任意 MAS、对所有指标或所有模型/数据集的普遍优越性、独立人工确认优越性、稳定在线早期预警、部署级端到端成本/内存上限，以及“白盒 LLM”。前缀预警固定开发点动作前正例召回仅 0.155；动作前负例误报率 0.067（全轨迹任意时刻诊断值为 0.200），因此仍不能宣称稳定在线早期预警。语义抽取 v2 最终 abstain 28.6%，尚未达部署门槛（≤10%）。

## 提交前硬门槛

1. 冻结至少 160 个目标完成 episode：四拓扑 × 两个独立 API/model 条件 × 每格 20（10 benign/10 violating）；失败、拒答、超时、截断全部保留。
2. 两名标注者独立只看监测器可见材料与执行证据，报告 agreement、Cohen’s κ，争议按预注册规则裁决；确认标签不得来自 monitor、hidden attack ID 或同一 oracle。
3. 在最后冻结模型/阈值后才打开确认标签，完成独立 fit/calibration/confirmation；报告 AUROC、AUPRC、group-weighted Brier、固定 FPR 召回、lead time、误报率、弃权覆盖率。当前已准备 120 条 Qwen 加 32 条完整 GLM、覆盖两个 API/model 条件的去标识双盲标注包，但尚未完成两名独立标注者的标注与仲裁。
4. 补齐部署级 API/语义抽取成本、训练成本、端到端内存、监测 p50/p95 与失败率，并完成固定低误报点的独立在线预警确认。
5. 完成学校采用年份/版本/类别的 CAS 官方记录核验；在上述事项完成前，投稿状态只能写 development package / external diagnostic。

补充公平性边界：同信息 Logistic/HistGB 审计已经完成开发集比较，但尚未在模型冻结后的新确认集上完成配对显著性检验；因此主模型的论文卖点应优先放在双线可审计结构、知识增量与校准/解释路径，而不是“全面性能领先”。
