# fava independent fixed convex comparison

| Original base | Target weight | Window F1 | Answer F1 |
|---|---:|---:|---:|
| lookback__old_lr | 0.4 | 0.672805 | 0.871795 |
| lookback__old_tree | 0.4 | 0.671722 | 0.871795 |
| harp_claim__old_lr | 0.2 | 0.687696 | 0.870466 |
| harp_claim__old_tree | 0.2 | 0.687364 | 0.865979 |
| semantic_claim__old_lr | 0.4 | 0.683675 | 0.858586 |
| semantic_claim__old_tree | 0.6 | 0.680976 | 0.859903 |

All36 candidates and12 exact endpoints are retained. The six bases are the original old controls; no large-combination output is fed into this run. Both metrics in each row belong to one candidate. Repeated calibration, not independent confirmation.
