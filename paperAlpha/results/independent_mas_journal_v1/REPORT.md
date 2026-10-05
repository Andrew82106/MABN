# Independent MAS journal evaluation v1

Five-fold StratifiedGroupKFold by scenario_family; no family is shared between train and test in a fold.
The public trace contains the normative workflow DAG and runtime events; labels are used only after feature construction.

- n=4000, positives=1345, groups=186

| method | F1 | precision | recall | AUROC | AUPRC | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| workflow_bn | 0.516 | 0.351 | 0.978 | 0.517 | 0.340 | 0.639 | 0.645 |
| runtime_bn | 0.626 | 0.493 | 0.859 | 0.716 | 0.545 | 0.196 | 0.113 |
| workflow_logistic | 0.518 | 0.357 | 0.945 | 0.578 | 0.389 | 0.219 | 0.009 |
| runtime_logistic | 0.653 | 0.562 | 0.778 | 0.805 | 0.653 | 0.167 | 0.019 |
| hierarchical_fusion | 0.668 | 0.598 | 0.756 | 0.820 | 0.675 | 0.160 | 0.019 |

## Fixed 5% FPR (fold-local)

| method | empirical FPR | recall |
|---|---:|---:|
| workflow_bn | 0.743 | 0.799 |
| runtime_bn | 0.051 | 0.258 |
| workflow_logistic | 0.055 | 0.075 |
| runtime_logistic | 0.051 | 0.292 |
| hierarchical_fusion | 0.052 | 0.303 |

## Interpretation
- `workflow_bn` is the normative DAG line; `runtime_bn` is the observable message/delegation line.
- `hierarchical_fusion` is train-fitted only and exposes its two line scores plus missingness/unknown-evidence terms.
- Confidence intervals and mechanism breakdown are in `metrics.json`; the benchmark is synthetic and requires confirmation on an independently generated/API cohort before a journal claim.
