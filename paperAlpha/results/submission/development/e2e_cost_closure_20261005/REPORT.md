# API-to-monitor cost closure

This report joins the existing transport, semantic extraction and monitor
ledgers without pretending that they came from the same episode. It closes the
cost accounting component-wise and keeps failure/abstention visible.

| component | cohort | key result |
|---|---|---|
| API transport | LANYUN R15, 80 planned runs | 416 requests, 64/80 runs complete, 16 failed; 459,902 tokens; latency p50/p95 7.43/28.55 s |
| semantic extraction | 21 public replies, bounded retry | 32 requests, 31 complete, 15 parsed, 6 abstentions (28.6%); latency p50/p95 10.62/16.39 s; 23,013 tokens |
| graph + BN monitor | 1,000 local latency cases | CPU inference p50/p95 1.01/1.71 ms; peak tracemalloc 0.047 MB |

The monitor latency excludes API generation, network transfer, semantic parsing
and training. Failed API calls remain failures; invalid extraction becomes an
explicit abstention. This is a cost/observability closure, not a new accuracy
result and not a single end-to-end episode latency claim. Full machine-readable
values are in `REPORT.json`.
