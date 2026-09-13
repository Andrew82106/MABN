# R32 independent audit

20 new and10 reused models replay exactly. All390 calibration entries, selections,9526-window/598-answer counts and fixed-C differences pass. No refitting or GPU.

| Input | Method | Window F1 | Answer F1 | Window Δ vs fixed C | Answer Δ vs fixed C |
|---|---|---:|---:|---:|---:|
| base769 | semantic_probe | 0.477186 | 0.572944 | -0.016542 | -0.012130 |
| base769 | lookback_plus_semantic | 0.633563 | 0.695122 | -0.003937 | +0.001623 |
| base769 | r26_plus_semantic | 0.723978 | 0.812287 | +0.001939 | +0.008233 |
| large1025 | semantic_probe | 0.608167 | 0.666667 | -0.002369 | -0.032421 |
| large1025 | lookback_plus_semantic | 0.683636 | 0.742475 | +0.000860 | +0.003201 |
| large1025 | r26_plus_semantic | 0.730906 | 0.832215 | -0.000801 | +0.002783 |

All602 answers remain in candidate geometry;117 reviewed safe refusals retain their negative answer labels for evaluation, and4 unresolved answers are excluded only from scored answers. The LR training still uses eligible window labels only; unknown/refusal windows were not zero-filled. This is repeatedly developed assistant-labelled R16 data, not public human-gold QA.
