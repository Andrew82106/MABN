# Direct type heads: independent audit

Passed. All 12 saved heads reproduce all 696,220 window probabilities exactly. Both-head max, 3,839 answer maxima, calibration thresholds, fit/cal counts and the shared-C choices match. No fitting, GPU or official test access.

| Input | Selected C | Window F1 | Answer F1 | Window delta vs selected binary | Answer delta |
|---|---:|---:|---:|---:|---:|
| hidden64 | 1e-05 | 0.610489566 | 0.844444444 | -0.005131770 | +0.001804851 |
| hidden64_risk | 0.0001 | 0.618730726 | 0.849557522 | -0.001273930 | +0.006420267 |

Both input variants lose localization F1 against their selected binary controls. This is an observed result under the fixed labels, shared binary weights and max aggregation, not evidence that type supervision cannot work. The two learned boundaries also change capacity relative to the deterministic duplicated control.
