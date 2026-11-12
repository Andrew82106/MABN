# Independent large-combination audit

All 12 new models and 12 frozen controls replay exactly. All thresholds, counts, answer maxima and single-candidate selection checks pass. No refitting or GPU was used.

| Peer | Combination | Window F1 | Answer F1 | Matched control window F1 | Matched control answer F1 |
|---|---|---:|---:|---:|---:|
| lookback | large_lr | 0.672237 | 0.860000 | 0.647496 | 0.867580 |
| lookback | large_tree | 0.670858 | 0.858537 | 0.651954 | 0.878505 |
| harp_claim | large_lr | 0.673836 | 0.857143 | 0.679660 | 0.868687 |
| harp_claim | large_tree | 0.670858 | 0.858537 | 0.677223 | 0.852632 |
| semantic_claim | large_lr | 0.674838 | 0.859813 | 0.661283 | 0.858639 |
| semantic_claim | large_tree | 0.670858 | 0.858537 | 0.660850 | 0.861702 |

Each row uses one candidate for both metrics. Standalone large and the same peer fixed-weight controls are retained in the JSON report. Upstream cost differs; fit scores are in-sample and calibration has been repeatedly inspected. This is not an independent test or a guarantee of future superiority.
