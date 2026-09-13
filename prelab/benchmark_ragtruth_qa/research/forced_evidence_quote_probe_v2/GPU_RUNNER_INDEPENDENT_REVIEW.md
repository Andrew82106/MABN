# Forced-Evidence Quote Probe V2 GPU runner 独立静态复审

**结论：PASS，无静态阻断项。** 冻结 runner 可以进入固定 8 条 GPU smoke。`full_extract_allowed=true` 只表示完整提取的静态代码路径通过；完整提取仍被独立 `GPU_SMOKE_INDEPENDENT_REVIEW.json` 运行时门关闭。

## 冻结对象

| 对象 | SHA-256 |
|---|---|
| V2 GPU runner | `eb4a51353ba4edaff2cff877f576ff206f1fca1d6bd820928fb0db4acebfcca2` |
| CPU selfcheck | `a6ef47c093b11f1665e340b703bbd3275c37bcac07025fe9ae74eed6381c86cf` |
| V1 scientific PROTOCOL | `78e5558b61592238b0979c7c60aea23c79b8c7d3181e88113d3972b4d056f7aa` |
| V1 PLAN | `6d439ee257dde65d2ef2c9974cd3dd716f4f0d06890cfb7df9c8af6da562680c` |
| V2 NUMERICAL_PROTOCOL | `ad0a2198ba69e75c78852eab103e8a829b6f9fb21389f322d5879f016e989df1` |
| sole label-free input | `ae6bf145a0e48ff310cdde5f54457eec7e8986c9b012d52cc463bb9e357e2777` |

## 核验结果

- V2 结果、记录、失败回执和审查文件全部写入新的 `forced_evidence_quote_probe_v2` 路径；V1 的 protocol、plan、preparation、manifest 和 label-free input 只读复用。运行签名同时绑定 runner、V1 科学协议、V2 数值协议、输入、PCA、辅助代码和模型资产。
- 唯一数据输入仍是 3,776 行、256 个回答的 `label_free_inputs.jsonl`。V1 helper 强制 14 字段精确 allowlist，并递归拒绝 gold、label、risk、quality、原回答和已有分数。runner 没有其他 gold/cal/test 数据读取路径。
- V1/V2 AST 函数级差分中有 42 个顶层函数源码逐字相同。prompt/region、BM25、P1/P2、生成、logit statistics、token offset、hidden/PCA、attention、relation、模型加载等 18 个科学核心函数全部不变。P1/P2/P3 仍为 21/549/549 维，拼接仍为 `21+11+5+256+256`。
- P3 的生成 IDs、停止位置和 token logprob/entropy/margin 只来自 KV-cached greedy 路径；分类器的 11 维 token statistics 明确索引 cached arrays。full no-cache replay 只权威提供 hidden64/attention；replay logits 只保存为 QA，不进入 `p3_features`。
- 固定 smoke 索引为 `[2826, 994, 1533, 1695, 1299, 1143, 3352, 3752]`，prompt 长度固定为 `[255, 354, 385, 435, 508, 568, 634, 799]`。三种连续量以 float64 做纯绝对差，`rtol=0`，边界包含，`atol=0.0625/0.0625/0.5`。
- smoke argmax mismatch 只在 cached margin `<=0.5` 且 replay 对 cached token 的 signed margin `>=-0.5` 时作为 near-tie 放行；其他 mismatch 为高置信冲突并停止。第 2826 条标为 tolerance anchor，其余 7 条标为 validation smoke。
- 最短和最长 claim 各重复一次。cached token IDs、位置、offset、停止和解析严格相等；cached 三种统计以及 replay hidden64、五组 attention 原始汇总和 `attention_summary256` 均以 `rtol=0, atol=1e-6` 比较。
- formal 路径调用 `qa_mode="formal_record_only"`。超容差、near-tie 失败或 replay argmax mismatch 只进入每 claim QA 摘要，不抛错、不筛行、不补行、不重试。每 claim 保存 mismatch count/rate 及三种差的 mean/P50/P95/P99/max；完成时从所有原子记录重新计算并按全部生成 token 汇总。
- 全量仍对 finite、shape、cached feature provenance、teacher-forcing target IDs、位置、offset、hash 和原子记录一致性 fail closed。parse-invalid 保留实际内容，只有冻结的 `T=0` 规则把 token/hidden/attention 三块置零。
- 原子 NPZ、metadata、commit 顺序与 row/runtime hash 绑定仍有效。若 3,776 个记录均存在而 completion 缺失，`pending` 为空，代码跳过 tokenizer、PCA、GPU lease、模型加载和 CUDA 清理，直接做 CPU record audit 后补 completion。CPU selfcheck 用一条合成记录验证了相同 audit；真实 3,776 记录分支尚未运行。
- `gpu-smoke` 在模型加载前要求本审查文件精确绑定 runner、两份 V1 科学文件、V2 数值协议和 CPU selfcheck。`extract` 还要求独立 smoke PASS，并绑定当前 runner、V2 数值协议和 `GPU_SMOKE.json` 哈希。
- 模型固定 local-only、`trust_remote_code=false`、NF4 double quant、BF16 compute/`lm_head`、SDPA、TF32 off、batch 1。模型加载前逐文件验证本地资产；GPU 使用复用已审计的 Windows WDDM 独占门。
- CPU selfcheck JSON 可解析，运行签名重算一致，所有直接依赖哈希均重算通过；runner 也通过 `py_compile`。

## 剩余运行风险

- V2 尚未运行 GPU smoke，因此 8 条实际数值门、同路径 repeat、峰值显存和吞吐仍未知。
- 8 GB GPU 对最长 959-token 路径的余量只能由 smoke 证实。
- 无 GPU completion 重建在代码上成立，并做过单记录 CPU 合成测试；尚无 3,776 个真实 V2 记录可做端到端演练。
- 本审查未验证 quote exact rate 或任何 AP/F1；这些必须等特征冻结及独立审计后由 evaluator 处理。

## 审阅完整性

本复审未修改 runner、protocol、plan、baseline 或数据；未启用 GPU、未初始化 CUDA、未加载预训练模型；未读取任何 label、fit gold、calibration 或 official test；未运行特征提取、训练或评分。

