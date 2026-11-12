# Human-correction pair pilot: stop report

## Decision

Stop this route before repair-v2 training. The RAGTruth `Original:` note is not
a reliable corrected-claim gold label. Training on mechanically constructed
pairs would mainly risk learning source-copy, length, schema, and explanation
style cues rather than factual conflict.

## Evidence

The independent feasibility audit read only
`auxiliary_human_v1/candidate_fit.jsonl` (9,678 fit rows, 4,381 released EC/SC
spans) and a fixed 100-item manual-rule review. It read no calibration or test
file, loaded no model, ran no training/GPU, and did not inspect or change a
published baseline.

- Strict `Original:` parsing succeeds for 4,159/4,381 spans, but parse success
  does not imply a usable corrected claim.
- In 100 fixed task×conflict-stratified reviews, only 14 are safe direct pairs;
  62 need human rewriting and 24 must be rejected. Only 19/100 are even
  grammatically safe as raw replacements.
- The population-weighted descriptive direct-pair rate is 2.73%; this is not a
  confidence interval because the review sample is deterministic.
- A frozen mechanical prefilter yields 272 candidate spans / 263 unique pairs.
  Requiring the answer to contain no other released error leaves 150 / 146.
  These are only a human-review queue, not training gold.
- The surface proxy has precision 0.387 and recall 0.857 against the 100-item
  direct-pair decisions: it admits too many unsafe pairs.
- Full-corpus shortcut signals are common: 2,660 corrections contain
  explanatory/schema style, 2,144 copy the good side exactly from evidence while
  the bad side does not, and 650 have extreme length mismatch.

## Artifact status

Attempt 1 remains byte-for-byte preserved and is rejected before GPU. Its 786
triples and arrays must not be used for training. Repair v2 generated no triples,
arrays, runner, checkpoint, or evaluation. Its protocol is retained as an
audit-only record of the abandoned candidate design.

The independent evidence is in:

- `research/human_original_pair_feasibility_v1/RESULTS.json`
- `research/human_original_pair_feasibility_v1/REPORT.md`
- `research/human_original_pair_feasibility_v1/MANUAL_REVIEW.jsonl`
- `research/human_original_pair_feasibility_v1/SHA256.json`

## What would be needed to resume

Resume only after humans rewrite and verify each candidate as a grammatical,
source-supported minimal claim pair, with group isolation and shortcut-balanced
controls. The present annotations alone do not meet that bar.
