# 验证记录

- 数据：4,000 条公开轨迹，1,345 个正例，186 个 topology family。
- 切分：5 个固定种子，每次按 family 做 80/20 留出；主模型融合在外层训练组内做 3 折组级折外拟合，阈值和概率校准使用独立组。
- 禁止输入：标签、hidden effect、机制名和 family 不进入特征；公开轨迹通过 forbidden-field 检查。`permission_mismatch` 属于当前工具事件字段，只支持整段轨迹/当前动作审计，不支持动作前预警结论。
- 主方法：episode-balanced Laplace local BN、运行时消息传播、规范工作流上下文和知识模板融合。
- 离线特征提取：4,000 条约 0.35 秒，约 0.087 ms/trace，峰值约 0.96 MB。
- 回归：shared-credit 相关测试 49 passed。
- `metrics.json` 经过 JSON 解析复核；机制分解中 benign 无正例，AUPRC 记为 n/a。

同信息审计见 `comparison_audit.json`：匹配的 78 维公开特征逻辑回归在 AUROC、Brier 和 ECE 上略好，本方法在 F1/AUPRC 上略高；不能声称全面领先。数据来自本仓库合成生成器，不是外部公开基准。粗粒度 family 是生成器摘要分组，不自动等于独立任务群。49 项测试只验证 shared-credit 采集与适配，不覆盖整个预测模型。旧 0.087 ms/trace 为特征提取时延，不能当成当前完整监测器时延；完整严格评测包含重复拟合，尚需补充在线 p50/p95 监测时延。

协议中的真实 API 独立确认、独立校准、跨生成模型迁移、解释忠实度和完整成本门槛仍未全部完成。目录名不构成投稿就绪证据。

真实 API 复核队列 `results/submission/development/quadsentinel_confirmation_20261001` 已固定 12 个案例；本轮 5 个完整、1 个因本地网关 `InternalServerError` 部分完成、其余尚未运行。不得把未覆盖动作当作安全预测。

新复制队列 `results/submission/development/quadsentinel_replication_20261001` 的 12 个报告均已生成，但本地网关在第一案后连续失败；仅 15/289 个动作被处理，274 个动作未覆盖。它验证了失败保留与审计链路，不提供可投稿的 API 性能证据。

规范 DAG 诊断 `results/submission/development/dag_api_v1` 已完成 8 个跨领域任务：直接提示策略节点 F1=.924、边 F1=.528、DAG 合法率 1.0；分步策略节点 F1=.863、边 F1=.308、DAG 合法率 .875。参考图来自任务规范，不是因果真值，因此只作开发诊断。

当前重跑的 12 任务 DAG 诊断 `results/harnessaudit_dag_v1` 完成 36/36 请求、35/36 可解析；direct/decompose/edge_audit 边 F1=.561/.552/.539，节点 F1 均为 1.0，三种策略合法率均为 1.0。解析失败响应已保留，未替换。

`api_recovery_pilot_20261001` 未进入评测：本机缺少 `agents` SDK，且默认文本解码无法读取 HarnessAudit 上游 YAML；没有把失败转换成负例或安全样本。

修复后，隔离运行时完成 `api_recovery_pilot_55_20261001` 的单任务 `off-t6`：14 个请求和唯一响应，独立 `audit_submission_harnessaudit.py` 返回 `status=passed`。这是接口与证据链验证，不是独立测试集，也不支持性能或泛化结论。

`api_recovery_mini_20261001` 的独立审计也通过：`ec-t3` 完成，`fin-t1` 保留为 `UserError` 失败；共 22 次请求、22 个唯一响应。失败轨迹未被改写为安全或负例。

阶段前瞻开发评测已补齐：`api_stage_forecast_20261001_v5` 使用 16 个完成任务、103 个阶段窗口，按任务留一；分层 BN AUROC=.874、AUPRC=.698、Brier=.127，OOF 融合 AUROC=.892、AUPRC=.748、Brier=.121，同信息 HistGB Brier=.112。跨任务迁移（扩展集训练，`ec-t3`/`fin-t1` 测试）分层 BN AUROC=.778、Brier=.152，flat logistic Brier=.136；均为开发证据。`MODEL_SPEC.json` 保存变量、特征和拟合参数，`api_stage_latency_20261001` 保存 103 个窗口的在线 p50/p95 延迟。

v6 候选结构在同一输入边界上加入运行时执行图的六个白盒统计量，并重新生成留一、重跑、三任务跨任务和延迟报告：`api_stage_forecast_20261001_v6`、`api_stage_forecast_replication_20261001_v6`、`api_stage_forecast_transfer3_20261001_v6`、`api_stage_latency_20261001_v6`。v6 的 16 任务留一分层 BN 为 AUROC=.875、AUPRC=.703、Brier=.127；三任务跨任务为 AUROC=.963、AUPRC=.953、Brier=.124。指标仍不足以替代预先冻结的独立确认集，且 HistGB 在部分设置更强。
