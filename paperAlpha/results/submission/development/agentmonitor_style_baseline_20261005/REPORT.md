# AgentMonitor-style MAS risk baseline

Strict five-fold family-disjoint evaluation. The original AgentMonitor target is team performance; this run adapts its observable per-agent statistics to the declared risk target.

- n=4000, positives=1345, groups=186, feature_count=60

| method | F1 | precision | recall | AUROC | AUPRC | Brier |
|---|---:|---:|---:|---:|---:|---:|
| AgentMonitor-style statistics + logistic | 0.581 | 0.444 | 0.842 | 0.712 | 0.503 | 0.197 |

This is a protocol-compatible adaptation, not a claim that the original AgentMonitor directly predicts this risk label.
Fixed threshold 0.5: FPR=0.136, recall=0.317.
