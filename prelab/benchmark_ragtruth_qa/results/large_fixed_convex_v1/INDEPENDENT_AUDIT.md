# Independent fixed-convex audit

All 36 saved formulas, answer maxima, thresholds, counts and choices pass; all 12 endpoints replay exactly. No refitting or GPU.

The unique cross-family winner under the existing selection key is **semantic_claim__old_tree__large_weight0.4**: window F1 **0.690281399**, answer F1 **0.891089109**. Both come from this one candidate.

| Fixed base | Selected large weight | Window F1 | Answer F1 |
|---|---:|---:|---:|
| lookback__old_lr | 0.4 | 0.679293 | 0.873096 |
| lookback__old_tree | 0.6 | 0.677619 | 0.879227 |
| harp_claim__old_lr | 0.4 | 0.689809 | 0.883495 |
| harp_claim__old_tree | 0.4 | 0.685120 | 0.890995 |
| semantic_claim__old_lr | 0.4 | 0.690159 | 0.878788 |
| semantic_claim__old_tree | 0.4 | 0.690281 | 0.891089 |

The JSON retains all 36×76 candidate/baseline differences and the complete baseline roster. Any unavailable schemas are listed explicitly. This is repeatedly inspected calibration, not a new test; the result does not guarantee future nondegradation.
