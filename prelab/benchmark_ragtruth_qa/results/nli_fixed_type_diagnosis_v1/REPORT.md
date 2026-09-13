# Fixed thresholds: error type recall

| Method | Window F1 | Answer F1 | Evident conflict detected /997 | Clean-answer FP windows |
|---|---:|---:|---:|---:|
| nli_standalone | 0.638658 | 0.847291 | 178 | 219 |
| lookback__old_lr__nli_weight0.2 | 0.665229 | 0.872549 | 274 | 234 |
| lookback__old_tree__nli_weight0.2 | 0.670381 | 0.874419 | 234 | 177 |
| harp_claim__old_lr__nli_weight0.2 | 0.685063 | 0.873096 | 266 | 162 |
| harp_claim__old_tree__nli_weight0.2 | 0.683599 | 0.876289 | 236 | 128 |
| semantic_claim__old_lr__nli_weight0.2 | 0.680312 | 0.864078 | 280 | 148 |
| semantic_claim__old_tree__nli_weight0.2 | 0.682716 | 0.870466 | 275 | 144 |

All thresholds were already selected before this diagnosis. No new models or scores were fitted. Overlapping type counts are descriptive only; original labels and denominators remain unchanged.
