# ATBench external transfer audit

冻结模型在 independent_mas_v3 上拟合；ATBench 仅在特征构造完成后用于计算指标。ATBench 是单代理/通用 Agent 迁移集，不是 MAS 主结果。

数据来源：AgentDoG 公开 ATBench 测试划分（仓库内本地副本 `paperAlpha/data/external/ATBench/test.json`）；来源：[AI45Lab/AgentDoG](https://github.com/AI45Lab/AgentDoG)。

## Overall

| method | F1 | precision | recall | AUROC | AUPRC | Brier |
|---|---:|---:|---:|---:|---:|---:|
| two-layer (ours) | 0.043 | 0.917 | 0.022 | 0.593 | 0.554 | 0.264 |
| layer1_bn | 0.000 | 0.000 | 0.000 | 0.586 | 0.545 | 0.254 |
| topology_only | 0.652 | 0.559 | 0.781 | 0.586 | 0.545 | 0.415 |
| edge_count | 0.000 | 0.000 | 0.000 | 0.500 | 0.497 | 0.497 |
| single_agent | 0.652 | 0.559 | 0.781 | 0.586 | 0.545 | 0.415 |
| dynamic_rule | 0.652 | 0.559 | 0.781 | 0.586 | 0.545 | 0.415 |
| flat_logistic | 0.000 | 0.000 | 0.000 | 0.447 | 0.530 | 0.274 |

## Audit

- n=1000; positive=497; negative=503; parse failures=0.
- 1953 unique tool names; mean public text length=9404.2 characters.
- Excluded from features: label, risk_source, failure_mode, reason, real_world_harm.

## Boundaries

- This is external transfer evidence only; it must not be merged into the MAS headline table.
- Tool-name normalization is a fixed prior, and benchmark labels are not independent human deployment labels.
