# 按拓扑族的校准审计

主表的 Brier/ECE 是 episode-micro 汇总；由于 186 个 topology family 的规模从 singleton 到 246 条不等，另计算每个 family 等权后的 Brier。数值来自冻结的 `independent_mas_journal_v1/predictions_oof.jsonl`，没有重新拟合模型。

| 方法 | episode-micro Brier | family-macro Brier |
|---|---:|---:|
| runtime logistic | 0.16714 | 0.15985 |
| two-line BN fusion | **0.16035** | **0.14609** |

2,000 次 family bootstrap 的 95% 区间为：runtime logistic micro [0.1609, 0.1728]、macro [0.1461, 0.1731]；two-line BN micro [0.1536, 0.1668]、macro [0.1333, 0.1601]。这项结果补充说明主模型的概率误差优势不是由少数大族完全造成的；它仍不能替代独立人工确认集。
