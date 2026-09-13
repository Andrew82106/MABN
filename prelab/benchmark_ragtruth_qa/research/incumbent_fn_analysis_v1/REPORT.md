# Incumbent FN/FP relation diagnosis

只读 CPU 事后分析 fit634/cal159；未拟合模型或阈值，未打开 test，未修改 baseline。

当前 cal 窗口 TP/FP/FN=3839/1300/2145，F1=0.690281。

## FN 的互斥错误簇

| Gold类型 | FN窗 | entail<.50 | .50-.75 | >=.75 | NLI315中位数 | Cross中位数 |
|---|---:|---:|---:|---:|---:|---:|
| Evident Conflict | 802 | 410 | 117 | 275 | 0.161 | 0.061 |
| Subtle Conflict | 27 | 2 | 14 | 11 | 0.455 | 0.072 |
| Evident Baseless Info | 1052 | 798 | 124 | 130 | 0.441 | 0.125 |
| Subtle Baseless Info | 264 | 227 | 15 | 22 | 0.461 | 0.342 |

### Evident Conflict 中的可观测关系代理

| 代理 | 全部802个FN | 其中275个高entail FN |
|---|---:|---:|
| subject_low_coverage | 172 | 14 |
| entity_incomplete | 286 | 71 |
| predicate_absent | 372 | 58 |
| negation_mismatch | 191 | 3 |
| comparator_reversal | 0 | 0 |
| comparator_missing | 64 | 0 |
| quantity_different | 186 | 12 |
| quantity_missing | 32 | 0 |
| condition_mismatch | 80 | 0 |
| cited_source_gap_015 | 46 | 28 |
| hard_relation_or_source | 370 | 40 |

## FP 的来源

| FP簇 | 窗口 | NLI315中位数 | Cross中位数 |
|---|---:|---:|---:|
| inside_gold_positive_microclaim | 210 | 0.867 | 0.810 |
| outside_positive_microclaim_in_risky_answer | 970 | 0.838 | 0.665 |
| clean_answer | 120 | 0.776 | 0.175 |

## 可迁移信号检查

AUROC是在 incumbent 当前负判中分 FN/TN；fit 与 cal 同向才值得继续。

| 信号 | fit | cal |
|---|---:|---:|
| atomic_NLI315_readout | 0.674 | 0.776 |
| crossencoder | 0.594 | 0.772 |
| raw_NLI_risk | 0.528 | 0.647 |
| one_minus_global_entailment | 0.591 | 0.699 |
| pair_contradiction | 0.584 | 0.646 |
| cited_gap | 0.548 | 0.529 |
| entity_incomplete | 0.503 | 0.571 |
| quantity_different | 0.520 | 0.522 |
| hard_relation_or_source | 0.563 | 0.581 |

## 高entailment难例

- fit：167/3960 为错误（4.2%），其中冲突 72 条。
- cal：37/939 为错误（3.9%），其中冲突 19 条。
- NLI315 在该簇的 claim AUROC：fit 0.736 / cal 0.695；Cross：fit 0.656 / cal 0.651。
- 现有表面关系 flag 在 fit/cal 不稳定；尤其否定、比较和条件 mismatch 在 cal 高entailment正例中几乎不触发。它们不能直接当规则。

## 两个下一候选

1. **证据对齐的高entailment关系专头**：每个证据句保留 E/N/C 与主体、实体、谓词、数量、否定、来源的同一对交互，再做聚合；高entailment样本走单独的类平衡冲突头。最低成本先在冻结特征上做 fit-group OOF 小头，阈值只取 fit。
2. **同答安全微主张对比头**：在同一个风险回答内，把错误主张与安全主张配对，减掉共享的整答风险，只用完全 OOF 的主张证据特征学习排序。最低成本先用 NLI315、Cross 和证据对齐特征做 fit-only 分组逻辑回归。

## 可达增益（gold oracle，仅算术）

- 完美补回全部冲突 FN：+829 TP，F1=0.781。
- 只补高entailment冲突：+286 TP，F1=0.723。
- 完美删掉主张边界内 FP：-210 FP，F1=0.704。

这些是开发集诊断，不是新模型成绩。
