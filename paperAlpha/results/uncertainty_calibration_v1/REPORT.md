# Uncertainty and Calibration Audit v1

本报告独立评估风险概率的可靠性，不把 F1 当作概率质量。所有模型参数、阈值和温度均只在训练折拟合。

## Across-seed summary (mean ± std)

| variant | F1 | AUROC | AUPRC | Brier ↓ | ECE-10 ↓ | log loss ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Raw | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.002 ± 0.000 | 0.042 ± 0.001 | 0.043 ± 0.001 |
| Temperature-scaled | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |

## MAS baseline calibration comparison

| baseline | F1 | AUROC | AUPRC | Brier ↓ | ECE-10 ↓ |
|---|---:|---:|---:|---:|---:|
| per_agent_max | 0.558 ± 0.000 | 0.500 ± 0.000 | 0.387 ± 0.000 | 0.250 ± 0.000 | 0.113 ± 0.000 |
| per_agent_mean | 0.558 ± 0.000 | 0.546 ± 0.014 | 0.468 ± 0.009 | 0.241 ± 0.002 | 0.116 ± 0.009 |
| topology_only | 0.656 ± 0.005 | 0.669 ± 0.007 | 0.488 ± 0.005 | 0.406 ± 0.008 | 0.406 ± 0.008 |
| edge_count | 0.558 ± 0.000 | 0.501 ± 0.020 | 0.390 ± 0.016 | 0.282 ± 0.008 | 0.173 ± 0.008 |
| no_taint_contribution | 0.656 ± 0.005 | 0.767 ± 0.015 | 0.622 ± 0.020 | 0.198 ± 0.004 | 0.128 ± 0.003 |
| dynamic_taint_path | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |

## Interpretation

温度缩放只校准置信度，不改变排序证据；若 Brier/ECE 改善而 F1 基本不变，说明改进来自概率可靠性而非检测能力。
bootstrap 区间和跨随机种子方差应同时报告。该数据是受控合成机制验证集，不能据此宣称真实部署校准或 SOTA。

## Artifacts
- `metrics.json` contains every seed, bootstrap interval and reliability bin.
- Raw predictions are regenerated from the canonical model and input; no terminal risk field is used.
