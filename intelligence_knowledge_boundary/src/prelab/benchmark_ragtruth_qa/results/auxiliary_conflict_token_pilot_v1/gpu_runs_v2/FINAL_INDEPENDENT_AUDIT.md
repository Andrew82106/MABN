# Auxiliary conflict v2 final independent audit

## Result

All 132,213 window scores and 747 answer-max scores were independently rebuilt from 99,000 frozen token predictions. Candidate/control reconstruction error is exactly 0. Both checkpoint and prediction hashes match their arm `complete.json`; each arm reports and arithmetically contains 395 + 27 = 422 optimizer updates.

| score | window F1-opt | window AP | window AUROC | answer F1-opt | answer AP | answer AUROC |
|---|---:|---:|---:|---:|---:|---:|
| unchanged v4 | 0.622716 | 0.639050 | 0.928287 | 0.696471 | 0.754043 | 0.855386 |
| candidate conflict only | 0.418202 | 0.353284 | 0.857214 | 0.612174 | 0.569747 | 0.774649 |
| candidate max(v4, conflict) | 0.619440 | 0.626571 | 0.916751 | 0.693333 | 0.753381 | 0.857241 |
| control conflict only | 0.365864 | 0.239169 | 0.823991 | 0.572662 | 0.463683 | 0.710874 |
| control max(v4, conflict) | 0.505012 | 0.356756 | 0.891936 | 0.590210 | 0.472537 | 0.726566 |

At the fixed v4 budget of 11,580 windows:

| score | TP | FP | EC recall | SC recall | conflict recall |
|---|---:|---:|---:|---:|---:|
| unchanged v4 | 7,259 | 4,321 | 11.638% | 33.846% | 13.092% |
| candidate max(v4, conflict) | 7,163 | 4,417 | 12.284% | 33.846% | 13.696% |
| control max(v4, conflict) | 4,671 | 6,909 | 29.526% | 29.231% | 29.507% |

The top-k cutoff has 31 tied candidate windows (13 selected) and 26 tied control windows (1 selected). The frozen stable original-window order resolves them. Neither boundary tie contains an EC/SC window, so conflict recall and the failed gate are invariant to tie order.

The pilot gate **fails**. Candidate conflict-only AP is 0.018597, versus control 0.024321; candidate combined overall AP differs from v4 by -0.012480.

## Diagnosis

The candidate's auxiliary stage is not well aligned with held QA: its final QA mean loss remains 0.708465, versus 0.037236 for control after repeated QA training. Only 27 final-QA updates follow the auxiliary pass. Because fusion is `max(v4, conflict)`, an imprecise conflict head can only promote extra windows; it cannot lower existing v4 false positives.

The control's repeated QA exposure raises conflict recall, but its held-fold scores are broad: its F1-opt rule selects 22,582 windows and produces 13,917 false positives, versus v4's 11,580 selected and 4,321 false positives. High conflict recall therefore reflects low specificity rather than a clean conflict boundary.

This audit did not read or trust `summary.json`, import/call the runner, use GPU, or modify a baseline. Preparation and both arm logs report zero calibration/test rows. Their hashes and the runner's static input boundary support that claim; no OS-level historical file-access trace exists.
