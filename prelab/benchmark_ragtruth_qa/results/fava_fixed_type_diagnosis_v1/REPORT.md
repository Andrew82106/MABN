# Fixed thresholds: error type recall

| Method | Window F1 | Answer F1 | Evident conflict detected /997 | Clean-answer FP windows |
|---|---:|---:|---:|---:|
| fava_standalone | 0.661103 | 0.833333 | 164 | 340 |
| lookback__old_lr__fava_weight0.4 | 0.672805 | 0.871795 | 257 | 250 |
| lookback__old_tree__fava_weight0.4 | 0.671722 | 0.871795 | 227 | 214 |
| harp_claim__old_lr__fava_weight0.2 | 0.687696 | 0.870466 | 202 | 145 |
| harp_claim__old_tree__fava_weight0.2 | 0.687364 | 0.865979 | 240 | 139 |
| semantic_claim__old_lr__fava_weight0.4 | 0.683675 | 0.858586 | 200 | 154 |
| semantic_claim__old_tree__fava_weight0.6 | 0.680976 | 0.859903 | 182 | 171 |

All thresholds were already selected before this diagnosis. No new models or scores were fitted. Overlapping type counts are descriptive only; original labels and denominators remain unchanged.
