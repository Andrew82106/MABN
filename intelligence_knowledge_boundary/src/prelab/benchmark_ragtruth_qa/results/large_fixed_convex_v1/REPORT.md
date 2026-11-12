# Frozen controls + fixed large probabilities

| Base | Large weight | Window F1 | Answer F1 |
|---|---:|---:|---:|
| lookback__old_lr | 0.4 | 0.679293 | 0.873096 |
| lookback__old_tree | 0.6 | 0.677619 | 0.879227 |
| harp_claim__old_lr | 0.4 | 0.689809 | 0.883495 |
| harp_claim__old_tree | 0.4 | 0.685120 | 0.890995 |
| semantic_claim__old_lr | 0.4 | 0.690159 | 0.878788 |
| semantic_claim__old_tree | 0.4 | 0.690281 | 0.891089 |

Every row uses one candidate for both metrics. All 36 candidates and the exact old-control/standalone-large endpoints are retained. No model was refitted. This is repeatedly developed calibration; selecting alpha0 cannot guarantee future nondegradation.
