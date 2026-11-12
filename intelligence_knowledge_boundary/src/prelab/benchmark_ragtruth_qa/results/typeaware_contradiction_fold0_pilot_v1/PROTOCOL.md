# Type-aware contradiction head: frozen fold-0 pilot protocol

This diagnostic uses only the held portion of fit fold 0. It reads exactly the
first 34,919 fit microclaims and first 3,680 fit answers, then stops. Calibration
and test are not read.

## Frozen score candidates

1. `v4_risk`: the original expanded-v4 fold-0 risk score.
2. `type_risk`: `P(neutral) + P(contradiction)` from the single type-aware pilot.
3. `type_contradiction`: `P(contradiction)` from that pilot.
4. `max_v4_contradiction`: `max(v4_risk, type_contradiction)`.
5. `convex_75v4_25c`: `0.75*v4_risk + 0.25*type_contradiction`.
6. `convex_50v4_50c`: `0.50*v4_risk + 0.50*type_contradiction`.

The list and weights are frozen before aggregate metrics are computed. No grid,
new model, calibration, or threshold transfer is allowed.

## Frozen evaluation

- Data/group split: the existing source-connected held fold 0.
- Localization unit: the existing eligible 4-raw-BPE, stride-1 windows.
- Window score: maximum score of the frozen microclaims owning that window.
- Gold: the existing factual-error window label. Evident Conflict (EC) and
  Subtle Conflict (SC) masks are reconstructed from the same official spans.
- Primary ranking metric: overall window average precision (AP).
- Diagnostic F1: each score's best held-fold threshold, using the common stable
  threshold rule. This is an optimistic pilot diagnostic, not a final estimate.
- Matched-budget diagnostic: every score predicts exactly as many windows as
  `v4_risk` predicts at its held-fold F1-optimal threshold. Report overall FP and
  EC/SC recall at that same alert count.

## Frozen continuation rule

Run the remaining folds only if at least one contradiction-augmented candidate
meets either condition:

- overall route: AP and F1 each improve by at least 0.005 over `v4_risk`; or
- targeted route: at the matched alert budget, combined EC/SC recall improves
  by at least 0.02, overall FP does not increase, and AP falls by no more than
  0.002.

No published baseline is involved or modified in this pilot.
