# Semantic window v3 protocol — frozen before fit

## Question

Test whether the frozen 44-column `evidence_union_nli_v2` claim interface repairs
the evidence-coverage weakness while keeping semantic-window v2's direct native
4-BPE window supervision.

This is a native-634 development experiment.  It is not a replacement for a
same-budget run over all 3,680 fit answers and is not an independent test.

## Frozen inputs and features

- The backbone is exactly semantic-window v2's 12 whitebox plus 14 geometry
  window columns.  Its formulas and ordering cannot change in v3.
- All 44 frozen evidence-union claim statistics are used.  They are pooled into
  each native 4-BPE window by lexical-token ownership weighted mean.
- The optional attribution block is exactly v2's 36 predeclared band statistics.
- Fit whitebox inputs must pass the existing source-group OOF reconstruction for
  both Lookback and large.  Calibration uses their corresponding full-fit values.
- No labels enter feature construction.  No conflict head, conflict-max,
  add-gate, claim-risk max projection, learned feature selection, or 1,394-column
  dump is allowed.  Evidence maxima already present inside the frozen 44-column
  descriptor remain ordinary claim attributes; they are never used to broadcast
  a claim detector score across windows.

## Preregistered grid

Only these six candidates exist:

| Variant | Fixed blocks | Width |
|---|---|---:|
| `backbone_union` | whitebox12 + geometry14 + union44 | 70 |
| `backbone_union_attribution` | whitebox12 + geometry14 + union44 + attribution36 | 106 |

Each variant uses `C ∈ {0.001, 0.01, 0.1}` with weighted StandardScaler and L2
logistic regression (`liblinear`).  There is no backbone-only selectable control;
the already frozen v2 result is the external control.

## Fit-only selection

Use five-fold `GroupKFold` over source-connected `group_id`.  Each fit window is
held out exactly once and no source-connected group crosses a fold boundary.
Within every training fold, give groups equal base mass, answers equal mass inside
their group, and windows equal mass inside their answer; then apply one binary
class-balance step.

For every candidate, select separate window and answer thresholds on fit OOF by
F1, then precision, then the higher threshold.  Select one candidate by this
fixed lexicographic key:

1. larger `min(window F1, answer F1)`;
2. larger window F1;
3. larger answer F1;
4. larger window AP;
5. larger answer AP;
6. fewer dimensions;
7. smaller C;
8. earlier variant in the table.

Only the selected variant and C are refit on all fit windows.  Model, columns,
C, both thresholds, source hashes, and runner hashes are frozen before opening
calibration labels.

## Calibration and references

Calibration is evaluated once per fresh output directory after the fit freeze.
Strict metrics use fit-OOF thresholds.  Separate calibration-F1-opt thresholds
are diagnostic only and cannot alter the frozen method.

Report both levels against:

- semantic-window v2 strict: `0.6539950722 / 0.8545454545`;
- semantic-window v2 calibration F1-opt: `0.6604922391 / 0.8708133971`;
- formal Lookback calibration F1-opt: `0.6008821240 / 0.8454545455`;
- historical incumbent calibration-selected: `0.6902813989 / 0.8910891089`.

Formal baseline files are read-only.  The runner has no official-test loader and
uses CPU only.
