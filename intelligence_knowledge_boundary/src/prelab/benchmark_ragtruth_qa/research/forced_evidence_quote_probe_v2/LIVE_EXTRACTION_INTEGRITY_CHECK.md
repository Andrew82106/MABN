# V2 Live Extraction Integrity Check

> **INTERIM DESCRIPTION ONLY.** 这是一次 label-free 的中途完整性描述。它不筛行、不重试、不评分、不定阈值，也不作接受/拒绝决策。

## 固定快照

本报告只覆盖稳定前缀 `0000000–0000258`：259 条 claim、259 个 commit、259 个 metadata、259 个 NPZ，共 777 个文件、11,339,704 bytes。后续生成记录不纳入本快照。

- 序号缺口：0
- 缺件：0
- pending：0
- 未知文件：0
- GPU 启动：否
- 源记录修改：否
- gold/calibration/test/label 数据读取：否

快照 manifest SHA-256：`c75674c2f100db090efa9702a598646568b102b311e85cbed33bfd5c1a0e158a`

算法：按文件名排序，每行写入 `filename<TAB>byte_length<TAB>file_sha256<LF>`，再对 UTF-8 拼接结果做 SHA-256。

## 完整性结果

结构性阻断：**0**。

- 259/259 `complete=true`
- commit→metadata、commit→NPZ、metadata→NPZ 哈希全部一致
- 文件序号与 commit/metadata 身份字段全部一致
- 259 条 NPZ schema 一致；所有序列长度与 metadata 一致
- replay drift 与 argmax mismatch 从 NPZ 重算后，259/259 与 metadata 一致
- metadata 非有限数：0
- NPZ 非有限数：0（11,655 个数组，3,210,747 个数值）
- cache `all_values_finite=true`：259/259
- label-free guard 状态干净：259/259

## Parse、原文命中与长度

| 指标 | 结果 |
|---|---:|
| P2 parse_valid | 259/259 |
| P2 source_exact | 259/259 |
| P3 parse_valid | 259/259 |
| P3 source_exact | 100/259 |
| P2/P3 stop_reason | close：259/259 |
| P3 generated tokens（min / p50 / p95 / max / mean） | 6 / 22 / 42.2 / 70 / 22.278 |
| P3 content tokens（min / p50 / p95 / max / mean） | 3 / 19 / 39.2 / 67 / 19.278 |
| P2 content tokens（min / p50 / p95 / max / mean） | 3 / 24 / 90 / 111 / 31.992 |

P3 `source_exact=false` 有 159 条。索引与逐 claim 值保存在 JSON；这里只做描述，不据此筛行。

## Cache replay

teacher-forcing target equality 为 259/259；high-confidence conflict 为 0。7 条 claim 各有 1 个 argmax mismatch：`20@2; 71@1; 111@2; 144@6; 184@10; 205@0; 215@19`。

36 条 claim 带有未强制的 drift gate 标记：`11[selected_logprob,vocab_entropy]; 20[selected_logprob,vocab_entropy,signed_margin]; 25[selected_logprob]; 46[signed_margin]; 52[selected_logprob]; 60[selected_logprob]; 61[vocab_entropy]; 71[selected_logprob]; 74[selected_logprob,vocab_entropy]; 90[vocab_entropy]; 91[selected_logprob]; 97[vocab_entropy]; 102[selected_logprob,vocab_entropy]; 107[selected_logprob,vocab_entropy]; 134[selected_logprob]; 135[selected_logprob]; 143[vocab_entropy]; 144[selected_logprob]; 159[vocab_entropy]; 176[selected_logprob]; 181[signed_margin]; 183[signed_margin]; 184[selected_logprob]; 189[selected_logprob]; 197[selected_logprob,vocab_entropy]; 203[selected_logprob,vocab_entropy]; 206[selected_logprob,vocab_entropy]; 207[signed_margin]; 210[selected_logprob,vocab_entropy]; 212[vocab_entropy,signed_margin]; 213[vocab_entropy]; 214[signed_margin]; 215[selected_logprob]; 234[vocab_entropy]; 240[vocab_entropy]; 249[selected_logprob]`。所有 metric 的 `smoke_gate_enforced=false`，且 259/259 `used_for_filtering_or_retry=false`；这些标记不构成本报告的筛选或决策。

| Drift 指标 | 每 claim max p50 | p95 | p99 | 全局 max（claim@position） |
|---|---:|---:|---:|---|
| selected_logprob | 0.005859 | 0.088598 | 0.149908 | 0.1971350908（144@6） |
| vocab_entropy | 0.014644 | 0.064573 | 0.093971 | 0.1340579987（203@6） |
| signed_margin | 0.25 | 0.44375 | 0.6775 | 0.75（46@6, 181@11, 214@45） |

逐 claim 的 commit/metadata/NPZ SHA-256、parse/source/stop、长度、mismatch、drift、非有限值与重算一致性均在同名 JSON 中。

