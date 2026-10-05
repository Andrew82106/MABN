# MAST external transfer: TwoLayerBNGraphMonitor-v2

This is a development-only external transfer audit. Labels are the released MAST annotations.

| condition | AUROC | AUPRC | F1 | Brier |
|---|---:|---:|---:|---:|
| structural-only (text disabled) | 0.295 | 0.675 | 0.859 | 0.197 |
| semantic runtime extension | 0.468 | 0.763 | 0.859 | 0.189 |

Delta (semantic - structural): AUROC 0.173; AUPRC 0.088; Brier -0.009.

The projection uses public trajectory text and message headers only; released annotations are joined after feature construction. This audit is not an independent human-gold superiority claim.

