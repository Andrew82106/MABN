# v3 fit-only scoring independent audit

## Conclusion

The saved score is reproduced exactly. Feature matrix and all token/window/answer OOF scores have max absolute error **0.0**; Shapley efficiency error is **7.11e-15**. Gold labels, 4-BPE stride-one windows, answer max pooling, five GroupKFold splits, weights, weighted scaler, LR C=0.1, thresholds, and metrics all match. `full_nll` is correctly defined as `-log p(token | full evidence)`; no sign or scoring implementation error was found.

## Why AUROC is near random

- The frozen signals are weak for this local-error target. Raw `full_nll` has token/window AUROC **0.590/0.598**, but answer-max AUROC falls to **0.350**. Risk-oriented `-(full_minus_empty)` reaches **0.628** per window but only **0.509** per answer.
- Fold offsets materially lower pooled OOF ranking. For B/C, mean within-fold window AUROC is **0.567/0.565**, while pooled AUROC is **0.517/0.510**. Held-fold window prevalence ranges from **4.42%** to **11.67%**.
- The weighting sequence balances classes and then re-equalizes groups, leaving positive loss mass at only **20.79%-22.42%**. This explains low probability levels and contributes to fold offsets; it does not by itself explain weak ranking.
- B has an exact duplicate: `full_minus_empty = empty_nll - full_nll`. Across all 29 features, only 20 standardized singular values exceed `1e-4`; nine directions are algebraically redundant up to float32 noise. Citation is also sparse: **2,950/39,252** lexical tokens are cited, with risk prevalence **7.25%** versus **8.51%** when uncited.
- Answer max pooling selects the single highest-risk window, often an unusual but correct token. At the fit-OOF F1 threshold, A/B/C mark **256/245/219** of 256 answers positive. The answer F1 therefore mainly comes from very high recall, not precise ranking.
- There are 46,481 raw tokens, but only **256 independent groups and 79 positive answers**. Local-error tokens cluster within answers, so token count overstates independent evidence.

This audit used only the fixed 256 fit responses and expanded-fit gold. It did not open calibration/test data, use GPU, create a candidate, or modify/score a paper baseline.
