# Preparation repair history

All repairs occurred before any new model inference, training, or gold-label access. Failed directories are preserved beside the final run.

1. `atomic_microclaim_nli_v1_failed_prepare_01`: preparation required every atomic record to own a lexical BPE. The inherited atomic audit intentionally contains seven punctuation-only records (` ``` ` or `":`), so the assertion stopped at response 12633. The first diagnosis also added explicit multi-ownership for a BPE that straddles two microclaim character spans; this is deterministic and label-blind.
2. `atomic_microclaim_nli_v1_failed_prepare_02`: the multi-ownership repair did not address the actual punctuation-only records, so the same invariant failed. The protocol was then refrozen to retain all seven records in an audit list and exclude them from factual NLI/scoring, matching the established v1 nonlexical policy.
3. `atomic_microclaim_nli_v1_failed_prepare_03`: after filtering those seven records, a validation assertion still required the original sparse `microclaim_index` to equal the new dense scored `claim_id`. The assertion was corrected; claim text, spans, retrieval, and scoring semantics did not change.
4. `atomic_microclaim_nli_v1_refrozen_after_cpu_review_04`: CPU preparation/check passed, then cache review found repeated identical premise/hypothesis texts could differ by at most 5.9e-6 because their old FP32 padding batches differed. The execution rule was refrozen: reuse only when all old occurrences are bit-identical; recompute ambiguous repeats. Inputs, labels, model checkpoint, and feature definition did not change. Full CPU preparation/check was then rerun.

The final frozen preparation contains 11,329 original atomic records, seven audited nonlexical records, 11,322 scored microclaims, and 98,854 NLI pairs. Formal baselines and official test artifacts were untouched.
