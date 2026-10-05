# PaperAlpha：双线可审计 MAS 风险监测

这是当前唯一的投稿入口。研究对象是 API 多智能体系统；受保护的大模型仍是黑盒，
“白盒”只指监测器自己的变量、边、概率、未知证据和解释路径可核查。

## 当前结论

主模型把两条证据线融合：

1. 规范工作流线：角色、权限、预期 DAG 和风险模板；
2. 运行时关系线：消息、委派、工具调用、异常边和证据传播。

在 4,000 条、186 个 topology family 的合成 MAS 基准上，五折 family-disjoint OOF 的两线融合为
F1 **0.667**、AUROC **0.820**、AUPRC **0.675**、Brier **0.160**。这些是可复现的开发结果，
不是独立人工确认后的普适安全结论。

## 唯一复现入口

- [主协议](protocol.json)
- [主结果汇报](FINAL_RESULTS.md)
- [单页投稿包](SUBMISSION_PACKET.md)
- [主基准复现脚本](reproduce_main.ps1)
- [模型规格](../results/independent_mas_journal_v1/MODEL_SPEC.md)
- [主评测报告](../results/independent_mas_journal_v1/REPORT.md)
- [发布包清单](PACKAGE_MANIFEST.json)：包含主脚本及其动态导入的公平基线、前缀评测依赖
- [结果门槛审计](../results/submission/final_eval/FINAL_GATE.md)
- [同信息基线映射](../results/submission/final_eval/BASELINE_MAPPING.md)
- [同信息 LLM judge 对照](../results/submission/final_eval/LLM_JUDGE_BASELINE.md)
- [同信息强基线审计](../results/submission/final_eval/MATCHED_INFORMATION_AUDIT.md)
- [按拓扑族校准审计](../results/submission/final_eval/GROUP_CALIBRATION_AUDIT.md)
- [原生 MAS baseline 可复现性审计](../../doc/ref_paper/mas_safety_2026-09-11/journal_alignment/native_mas_baseline_audit_20261005.md)
- [基线目录](baselines.json)
- [跨 Agent 约束目录](relational_contracts.json)

运行 `reproduce_main.ps1` 会按固定种子生成公开轨迹并运行五折评测；原始生成数据和结果留在
本地 `results/`，不提交 API key 或外部数据。

## 外部边界证据

- [MAST/MAD 修正版审计](../results/submission/development/mast_two_layer_v2_20261005/REPORT.md)：剥离末尾 `Evaluation` 标签块后，作为二次迁移边界。
- [TAMAS 官方数据边界审计](../../doc/ref_paper/mas_safety_2026-09-11/journal_alignment/tamas_official_data_audit_20261005.md)：官方包只有静态攻击案例，不能直接当作 runtime 金标准。
- [ATBench 外部迁移](../results/submission/development/atbench_external_20261005_v2/REPORT.md)：通用单 Agent 迁移边界，不进入 MAS 主表。
- [Who-and-When 失败定位审计](../results/external_who_when_audit_v1/REPORT.md)：只有失败样本，只作响应边界。
- [低阳性率压力诊断](../results/submission/development/prevalence_shift_20261005/REPORT.md)：把主集 OOF 结果重加权到 5%/10% 阳性率，作为部署压力检查，不替代独立测试集。
- [配对族级 bootstrap](../results/submission/development/paired_bootstrap_20261005/REPORT.md)：补充与 trust/reputation、graph 和 local 基线的成对差异区间。
- [端到端成本闭合](../results/submission/development/e2e_cost_closure_20261005/REPORT.md)：分开记录 API、语义抽取和图/BN 监测成本，不伪造单一 episode 延迟。
- [同 episode 成本审计](../results/submission/development/e2e_episode_cost_20261005/REPORT.md)：在共享的 10 个 LANYUN episode 上对齐 API 与语义抽取账，并测量同一 monitor projection 的本地 BN 路径。
- [独立确认集审计](INDEPENDENT_CONFIRMATION_AUDIT_20261005.md)：当前仍缺双盲人工 MAS 确认集。

## 投稿对标

首选目标是 ESWA；Information Fusion 只有在把规范、运行时和不完整证据形式化成真正的多源融合算法后才适合。安全拓扑和威胁模型参考 TDSC。

- [期刊投稿缺口审计](../../doc/ref_paper/mas_safety_2026-09-11/journal_alignment/journal_submission_gap_audit_20261005.md)
- [近邻论文刷新](../../doc/ref_paper/mas_safety_2026-09-11/journal_alignment/near_neighbor_refresh_20261005.md)
- [投稿要求对照](../../doc/ref_paper/mas_safety_2026-09-11/journal_alignment/submission_requirements_20261002.md)

在独立人工确认、原生 MAS baseline、端到端 API 成本和低误报早期预警补齐前，主包应称为
**development / external-diagnostic package**，不能写成“已完成独立安全优越性验证”。

独立确认的准备工作已扩展到 120 条 Qwen API 轨迹，并另备 32 条完整 GLM 轨迹；两包均去除了 episode/scenario/request 编号。它们仍是空白双盲标注材料，在两名独立标注者完成标注、仲裁前，不能写成独立安全金标准。
