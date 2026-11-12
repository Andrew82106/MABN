# Target-domain weighted expanded semantic window v2

Provenance gate passed: native fit634 and calibration159 are both released `llama-2-7b-chat` answers. The expanded-only 3046 answers come from five other generators. The gate is bound to the development manifest and its exact exporter; it did not open calibration answer/span rows.

The audited 26-column `whitebox_geometry` design and C=0.1 were fixed. Native training mass is 1; total auxiliary mass is alpha, split equally across five generators. Within each generator, source groups, answers, and windows are hierarchically equalized before separate binary window-class balance. Every fold/candidate is finally normalized to the native training-window mass.

| auxiliary alpha | native OOF window F1 | window AP | native OOF answer F1 | answer AP |
|---:|---:|---:|---:|---:|
| 0 | 0.610815 | 0.588625 | 0.742857 | 0.795561 |
| 0.25 | 0.610380 | 0.585365 | 0.742857 | 0.794785 |
| 0.5 | 0.609463 | 0.581206 | 0.746702 | 0.795173 |
| 1 | 0.607101 | 0.573449 | 0.749669 | 0.797093 |

Selected alpha: **0**, using only held native OOF scores. Alpha=0 reproduced native-only v2 OOF within one float64 ULP, with identical threshold decisions and confusion counts (max absolute score error 1.11e-16).

| evaluation | window F1 | answer F1 |
|---|---:|---:|
| selected native OOF / native-F1Opt | 0.610815 | 0.742857 |
| selected all-domain OOF / frozen native thresholds | 0.591586 | 0.620626 |
| calibration strict / frozen native OOF thresholds | 0.653995 | 0.854545 |
| calibration F1Opt diagnostic only | 0.660492 | 0.870813 |
| native-only v2 strict | 0.653995 | 0.854545 |
| historical cal-selected incumbent | 0.690281 | 0.891089 |

Strict deltas versus native-only v2 are +0.000000 window and +0.000000 answer F1. Versus the historical cal-selected incumbent they are -0.036286 and -0.036544.

Generator identity has zero inference columns. It is used only to construct training sample weights; every scaler and classifier consumes the same 26 numeric whitebox+geometry columns. Calibration was evaluated once after alpha, the full-fit model, and both thresholds were frozen. Cal-F1Opt is diagnostic only. Formal baseline hashes stayed unchanged and the official test remained unopened.
