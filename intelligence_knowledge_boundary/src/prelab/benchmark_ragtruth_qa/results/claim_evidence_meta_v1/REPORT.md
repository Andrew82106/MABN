# Claim evidence meta v1

六个候选只按 fit 的 source-group OOF 双层 F1 选择；calibration 在选择冻结后才计分。正式基线未改。

| fit-only候选 | OOF窗口F1 | OOF整答F1 |
|---|---:|---:|
| lr_C0.01 | 0.604148 | 0.772932 |
| lr_C0.1 | 0.600291 | 0.775330 |
| hist_leaf7 | 0.601453 | 0.755224 |
| hist_leaf15 | 0.592644 | 0.756522 |
| extra_depth8 | 0.591424 | 0.759388 |
| extra_depth12 | 0.591678 | 0.764163 |

选中 `lr_C0.01`。

| 划分 | 窗口F1 | 整答F1 |
|---|---:|---:|
| fit OOF | 0.604148 | 0.772932 |
| calibration | 0.654597 | 0.836957 |

句段分数投影回原4-BPE窗口；测试仍封存。
