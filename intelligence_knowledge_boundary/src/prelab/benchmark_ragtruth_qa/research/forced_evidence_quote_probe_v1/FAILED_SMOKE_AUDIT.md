# Forced-evidence quote probe V1：失败 smoke 独立审计

**结论：FAIL-CLOSED，V1 必须停止。** 两次 GPU smoke 都在固定的第一条样本
`record_index=2826`（`16215__atomic_000`）失败；没有成功的 `GPU_SMOKE.json`，没有正式
claim 记录、完整提取回执、特征冻结或真实 gold 评分。当前能看到的指标只有 CPU
evaluator 的合成自测值，不能作为实验成绩。

本审计仅读取协议、runner、无标签输入、失败记录与无标签诊断；没有启动 GPU，也没有
打开 fit gold、calibration 或 official test。

## 两次失败

| 次序 | 时间 | runner SHA256 | 失败点 | 性质 |
|---|---|---|---|---|
| 1 | 2026-09-12 21:12:54 +08:00 | `7989c1c9f5e8c835df8ccf4797a242b50138513e6a9c255374d386787c574139` | `decode(ids)` 后重新 `encode(text)`，断言 ID 必须逐个相同 | 实现中的 tokenizer 逆映射假设错误 |
| 2 | 2026-09-12 21:38:31 +08:00 | `a142d00ef1601aff6af203cfaf60644a11c37f108c8c4043f641b51f8e01e0cb` | cached generation 与 full replay 的 `selected_logprob` 超过冻结容差 | 预注册 smoke oracle 实质失败 |

第一次失败后，offset 实现改成按冻结 SentencePiece decoder 的状态机追踪原生成 ID，并完成
CPU 与静态复核。第二次失败来自这个修复后、与当前 canonical runner 相同 SHA256 的版本；
因此它不是旧代码留下的无关失败。

## 独立复算：第一次 offset 失败

诊断样本生成 29 个 token，解码为 149 个字符。原生成 ID 与把最终字符串单独重新编码后的
ID 只有第 0 位不同：

```text
原生成：8003   "Position"
重编码：20627  "▁Position"
其余：  28/28 完全相同
```

独立按 runner 的 canonical JSON digest 复算：

- 原生成 ID digest：`a1371f02ee8ba0749b77814ba79ed790f3741a8ba12887bbcacb3d62d5f655d7`
- 重编码 ID digest：`6ec3310d2c1fb5ef2b786a8e1021c56aa25a0b982405f6bdf0dbfef5195d83d3`
- 最终文本 SHA256：`436bd6a65b15a5f358f68d64355463a57635ef5f7e7bbe279f2813b2fc4ad0fd`

这说明文本内容没有变；变化来自 SentencePiece 对“接在 prompt 后生成的首词”与“把最终
文本当成新句重新编码”采用不同的词边界表示。`encode(decode(ids)) == ids` 不是合法断言。
逐前缀最长公共前缀复算出的字符区间从 `(0,8)`、`(8,11)`、`(11,14)` 开始，且在这个
样本上与重编码器返回的 29 个字符区间完全相同；失败只发生在 ID 身份断言。

## 独立复算：第二次 cache/replay 失败

29/29 个 replay argmax token ID 与 cached generation 完全一致，argmax mismatch 为 **0**。
这表示离散生成轨迹一致，但连续值信号没有同时通过协议冻结的
`np.allclose(rtol=0.005, atol=0.005)`：

| 信号 | 最大绝对差 | 均值绝对差 | P95 绝对差 | 未通过 token 数 | 未通过位置 |
|---|---:|---:|---:|---:|---|
| selected logprob | 0.0239888 | 0.00100350 | 0.00180692 | 1/29 | `[2]` |
| vocabulary entropy | 0.0277591 | 0.00179515 | 0.00846168 | 4/29 | `[0,2,18,27]` |
| signed margin | 0.250000 | 0.0775862 | 0.250000 | 14/29 | `[1,2,5,6,8,10,11,14,17,18,20,25,26,27]` |

selected logprob 的最大差发生在位置 2：cached 为 `-0.2254964560`，replay 为
`-0.2015076727`；该位置是引用正文中的 `▁is`，所以失败不能归因于闭合标签。只看 26 个
正文 token，selected logprob、entropy、signed margin 仍分别有 1、3、12 个位置超容差。

最合理的技术解释是：NF4/BF16 下逐 token KV-cache 路径和整串 no-cache replay 的运算形状
及舍入次序不同。它们没有改变 argmax，却足以改变 logit 派生的概率、熵和 margin。这个
解释是基于诊断结果的推断；诊断能够直接证明的是“ID 一致、连续信号超容差”。

## 正式产物核对

以下预期路径全部不存在：

- `results/forced_evidence_quote_probe_v1/GPU_SMOKE.json`
- `research/forced_evidence_quote_probe_v1/GPU_SMOKE_INDEPENDENT_REVIEW.json`
- `results/forced_evidence_quote_probe_v1/gpu_claim_records/`
- `results/forced_evidence_quote_probe_v1/GPU_EXTRACTION_COMPLETE.json`
- `results/forced_evidence_quote_probe_v1/frozen_features/`
- `results/forced_evidence_quote_probe_v1/frozen_features/FEATURE_FREEZE.json`
- `results/forced_evidence_quote_probe_v1/frozen_features/FEATURE_MANIFEST.json`
- `results/forced_evidence_quote_probe_v1/frozen_features/claim_features.npz`
- `results/forced_evidence_quote_probe_v1/EVALUATION.json`

V1 结果树中也没有 `.npz`、`.npy`、`.pkl`、`.parquet`、`.csv` 或 `.tsv` 正式特征文件。
smoke 在内存中构造首条样本的中间数组后抛错；`gpu-smoke` 路径本身不调用正式记录的
`save_record`。因此没有可供训练或评分的 V1 GPU 特征。

`CPU_EVALUATOR_SELFTEST.json` 明确标为
`passed_synthetic_CPU_only_no_real_scoring`，同时记录 `real_feature_freeze_present=false`、
`real_evaluation_run=false`、`fit_gold_opened=false`。其中 P0–P3 数值只是合成数据上的代码
自测，不能引用为本实验结果。

## 未受影响的数据与 baseline

- 无标签输入仍为 3,776 claims、256 answers、256 groups；14 个字段中没有冻结的 forbidden
  label/score 字段。文件 SHA256 仍为
  `ae6bf145a0e48ff310cdde5f54457eec7e8986c9b012d52cc463bb9e357e2777`。
- 协议与计划 SHA256 仍分别为
  `78e5558b61592238b0979c7c60aea23c79b8c7d3181e88113d3972b4d056f7aa` 和
  `6d439ee257dde65d2ef2c9974cd3dd716f4f0d06890cfb7df9c8af6da562680c`。
- P0 的 Lookback raw matrix/token index、公共 4-BPE window/answer index，以及已完成的
  Lookback、LUMINA、GHOST baseline 结果目录，其最近修改时间均早于第一次 smoke；runner
  的写入目标只在本 V1 结果目录和临时 GPU lease，不包含 baseline 路径。
- 两份无标签 GPU 诊断都记录 `gold_read=false`、`calibration_or_test_read=false`、
  `scoring_run=false`。没有 feature freeze，真实 evaluator 的 fail-closed 门无法开启。

所以这些失败只否定 **V1 当前执行契约可继续运行**，不改变冻结数据集、baseline 模型结构
或既有 baseline 分数，也没有产生可以与 baseline 比较的新分数。

## 停止依据

冻结协议第 85、91 节与 `PLAN.json` 的 `smoke_oracle.failure_action` 都规定：任一 ID 不同、
非有限值或超容差就停止；不得在本 V1 中放宽容差、换精度、换数值来源、换 prompt 或换
smoke 样本。第二次失败直接命中该条件。按最严格解释，第一次 smoke 失败本身也已触发
停止条款；即使把第一次视为不涉及标签的实现 bug 修复，第二次仍使 V1 无法继续。

若要改成“只要求 argmax 一致”、统一从 replay 或 cache 取三类统计、改变精度，或重新设定
数值容差，应建立 V2，在再次运行 GPU 及打开任何 gold 前冻结新契约。不能在 V1 上根据这条
已见 smoke 的误差回填新阈值。
