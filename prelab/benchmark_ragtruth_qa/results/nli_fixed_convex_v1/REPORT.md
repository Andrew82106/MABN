# nli independent fixed convex comparison

| Original base | Target weight | Window F1 | Answer F1 |
|---|---:|---:|---:|
| lookback__old_lr | 0.2 | 0.665229 | 0.872549 |
| lookback__old_tree | 0.2 | 0.670381 | 0.874419 |
| harp_claim__old_lr | 0.2 | 0.685063 | 0.873096 |
| harp_claim__old_tree | 0.2 | 0.683599 | 0.876289 |
| semantic_claim__old_lr | 0.2 | 0.680312 | 0.864078 |
| semantic_claim__old_tree | 0.2 | 0.682716 | 0.870466 |

All36 candidates and12 exact endpoints are retained. The six bases are the original old controls; no large-combination output is fed into this run. Both metrics in each row belong to one candidate. Repeated calibration, not independent confirmation.
