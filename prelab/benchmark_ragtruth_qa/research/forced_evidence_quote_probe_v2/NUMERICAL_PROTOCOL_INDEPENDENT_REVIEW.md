# Forced-Evidence Quote Probe V2 数值协议独立复审

**结论：PASS，无阻断项。**

独立审阅任务 `/root/numeric_protocol_v2/bf16_tolerance_review` 对冻结协议进行了只读复审。审阅对象为 `research/forced_evidence_quote_probe_v2/NUMERICAL_PROTOCOL.md`，SHA-256：

```text
ad0a2198ba69e75c78852eab103e8a829b6f9fb21389f322d5879f016e989df1
```

## 检查结果

| 检查项 | 结论 | 核验内容 |
|---|---|---|
| 数值容差冻结 | PASS | cache/replay 的 logprob、entropy、signed margin 分别固定为纯绝对容差 `0.0625 / 0.0625 / 0.5`，`rtol=0`，边界相等通过；容差只由 V1 首条 label-free 诊断按统一二进制 guard-band 规则确定。 |
| near-tie 规则 | PASS | 仅在固定 8 条 smoke 上检查。argmax 不同时，只允许 cached margin `<=0.5` 且 replay 对 cached token 的 signed margin `>=-0.5`；高置信冲突立即失败。 |
| smoke/full 边界 | PASS | 连续漂移与高置信 argmax 冲突是 smoke-only 硬门；全量只记录漂移与 mismatch，不据此筛行、补行、重试或调阈值。 |
| 权威信号来源 | PASS | cached greedy 唯一决定生成 token，并权威提供生成时 logprob、entropy、margin；full no-cache replay 只权威提供 hidden64/attention；replay stats 仅作 QA，不进入特征。 |
| 同路径 repeat | PASS | 最短和最长 claim 重复检查 cached token IDs 完全一致，cached 三种统计 `atol=1e-6`；replay hidden64/attention `atol=1e-6`。该门与跨路径的较宽容差没有冲突。 |
| 全量结构硬门 | PASS | 全量仍强制 finite、shape、ID/position/offset、hash、模型栈和原子写入一致；replay target IDs 必须精确等于 cached 生成序列。 |
| 防泄漏与版本隔离 | PASS | V2 使用新的 runner/version/result/review 链；V1 不能原地放宽或混入正式 cache；feature freeze 与独立审查前不得打开 fit gold，calibration/official test 保持封存。 |

## 实现注记

以下两点不构成协议阻断，但 V2 runner 的静态审查必须核实：

1. repeat 中的“attention 原始数组”应覆盖 `p3_ratio_token_mean`、`p3_ratio_token_min`、`p3_ratio_token_max`、`p3_passage_mass_mean`、`p3_claim_mass_mean`，并同时比较 `p3_attention_summary256`。
2. “不得因 parse-invalid 补零”指 invalid 状态本身不能成为补零理由；V1 已冻结的 `T=0` token/hidden/attention 三块置零规则继续保留。

## 审阅完整性声明

- 未修改数值协议、V1 协议、runner 或 baseline。
- 未读取 fit gold、calibration、official test 或任何标签和成绩。
- 未初始化 CUDA，未启用 GPU，未加载预训练模型。
- 未运行特征提取、训练或评分。
- 本 PASS 只批准数值协议进入 V2 runner 实现与独立静态审查；它本身不批准 GPU smoke 或 full extraction。

