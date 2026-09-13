# Target-domain weighted evidence-union v2 protocol

This is a development-only, independently audited experiment. It does not
modify or replace any formal baseline and never opens official-test data.

## Frozen feature and model

- Each eligible example is the existing exact four-lexical-BPE window.
- The first 26 inputs are the audited semantic-window-v2
  `whitebox_geometry` columns from the combined 3,680-answer fit artifact.
- The next 44 inputs are the frozen evidence-union-NLI-v2 claim aggregates.
  Fit-native (9,055 claims) and fit-expanded (25,864 claims) are concatenated
  only after exact response, claim, microclaim, hypothesis, and feature-name
  identity checks. Each window receives the arithmetic mean of its four
  lexical BPE owners' claim vectors. This exactly implements the v3
  token-ownership weighted claim mean, including repeated claim owners.
- The fixed structure is the v3-selected `backbone_union`: exactly 70 numeric
  inputs, with no attribution add-on, conflict head, or claim-risk max
  projection.
- The fixed estimator is weighted `StandardScaler` followed by liblinear L2
  logistic regression with `C=0.001`, seed 20260913, and sklearn 1.6.1.
- Generator, source-group, answer, response, claim, and window identities are
  excluded from the prediction interface. Generator identity is used only to
  construct training weights and to report domain diagnostics.

## Frozen split, weighting, and selection

- Use the exact five source-connected native-634 GroupKFold splits. In each
  fold, every fit answer from a held native source group is withheld across
  all six generators. Every other generator is eligible for training.
- Search only total auxiliary mass `alpha` in `{0, 0.25, 0.5, 1}`. Native
  mass is 1; auxiliary mass is alpha and is split equally among the five
  auxiliary generators (`alpha/5` each).
- Within every generator, first equalize source-group mass, then answer mass
  within source group, then eligible-window mass within answer. Separately
  balance the two window classes inside that generator for the logistic loss.
  Finally normalize both scaler and loss weights to the number of native
  training windows in that fold (or 168,123 for full fit), preserving the
  fixed C's effective scale.
- At `alpha=0`, every auxiliary row has exact zero weight. Native held OOF
  scores must reproduce the frozen native-only v3 `backbone_union`, C=0.001
  control within one float64 ULP, with identical thresholds and confusion
  counts.
- For each alpha, fit all five folds and score every held source group. Choose
  both window and answer thresholds only on held native target-domain OOF
  scores. Select alpha by descending: minimum of native window/answer F1,
  window F1, answer F1, window AP, answer AP; break remaining ties toward the
  smaller alpha. All-domain OOF is diagnostic and cannot affect selection.
- After alpha and thresholds are frozen, fit once on all eligible fit data.

## Frozen evaluation and audit

- Calibration contains 159 answers from the same native
  `llama-2-7b-chat` target generator, proven from the hash-bound development
  manifest and exporter. It is evaluated exactly once after full fit.
- Strict calibration uses the frozen native-OOF thresholds. A calibration
  F1-optimal threshold is reported only as a post-hoc diagnostic and cannot
  change any model, feature, alpha, or deployed threshold.
- Report window and whole-answer F1, precision, recall, AP, AUROC, and
  confusion counts for native OOF, selected all-domain OOF, strict
  calibration, and calibration F1Opt. Compare with native-only v3,
  semantic-window-v2, formal baselines, and the historical calibration-selected
  incumbent (0.690281 window / 0.891089 answer F1).
- Audit exact 4-BPE mapping, all claim identities, feature coverage and
  finiteness, source-connected leakage, weighting masses, inference columns,
  calibration single use, official-test non-use, and unchanged formal baseline
  hashes. An independent verifier must replay all 20 fold models from frozen
  inputs without reopening calibration labels.
- Candidate alpha values, model structure, C, folds, weighting, selection,
  mapping, and metrics are frozen here. They will not be expanded or weakened
  after results are observed.

## Evidence-union extraction state interpretation

`manifest_fit_expanded.json` and `missing_requests_fit_expanded.jsonl` freeze
the 158,929 requests that originally required inference. They are immutable
inputs to extraction, not a live missing-output counter. Readiness requires
`extraction_fit_expanded.json` status `complete_frozen_probabilities`, all 39
declared chunks present with exact hashes, `diagnostics_fit_expanded.json`
covering 3,046 answers and 25,864 claims, and an exact hash-bound
`claim_aggregates_fit_expanded.npz`. If any check fails, stop before fitting.
