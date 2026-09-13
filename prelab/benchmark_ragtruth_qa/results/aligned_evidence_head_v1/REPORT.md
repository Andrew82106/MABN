# Aligned evidence head v1

每个槽位只来自一个真实证据 pair；E/N/C 与 BM25、关系和来源特征不再跨句拼接。4 个 LR/HGB 候选和全部门槛只由 source-group fit OOF 决定；cal 只严格评测一次。

| 选中候选 | fit OOF窗口F1 | fit OOF整答F1 | strict cal窗口F1 | strict cal整答F1 |
|---|---:|---:|---:|---:|
| conflict_only__hgb | 0.816444 | 0.902208 | 0.687479 | 0.806630 |

strict cal 相对同一 fit 门槛 incumbent：新增 TP 35、新增 FP 40，增量精度 0.466667。由于 v1 只做 add gate，原 TP 保留 3984/3984，去除 FP 0。

同一 fit 门槛 base strict cal 为 0.685891/0.806630；历史 cal 选型 incumbent 为 0.690281/0.891089。

未计算 cal-F1Opt；未用 GPU、未打开 official test、未改正式 baseline。
