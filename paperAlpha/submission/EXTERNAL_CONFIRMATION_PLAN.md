# Independent MAS confirmation cohort

This is the only remaining data gate before calling the experiment package
submission-ready. It is deliberately separate from the frozen development
cohort and from public benchmark labels.

## Minimum cohort

- At least 100 complete MAS episodes from at least 4 workflow/topology families.
- At least 2 independently labeled API/model conditions; endpoint/model identity
  is a grouping variable, never a feature.
- Both benign and violating episodes, with the positive rate declared before
  collection. No outcome-based resampling.
- Every episode keeps the full observable prefix, action index, tool arguments
  visible to the monitor, and terminal sandbox result.

## Label contract

Two labelers independently record:

1. whether a declared policy or authorization rule was violated;
2. the first violating event index, or `none`;
3. violation class (scope, routing, permission, privacy, resource, or other);
4. the evidence event IDs and a short rationale.

Disagreements are adjudicated without showing the monitor score. Report raw
agreement and Cohen's kappa before adjudication. A label is not inferred from
the model's own explanation.

## Evaluation lock

The monitor receives only the declared workflow, public policy, and observable
prefix. Hidden prompts, final labels, attack IDs and post-hoc outcome fields are
excluded. Split by workflow family and endpoint condition. Freeze the threshold
on a calibration subset, then report full-trace AUROC/AUPRC/Brier, fixed-FPR
recall, first-alert lead time, benign false-alarm rate, abstention coverage,
semantic extraction latency, API cost and memory separately.

Until this cohort exists, `FINAL_RESULTS.md` is a complete development package,
not a final deployment or journal-superiority claim.

## Reproducible annotation entry point

The repository includes `scripts/prepare_mas_annotation_packets_v1.py`. It
creates separate, shuffled `annotator_a.jsonl` and `annotator_b.jsonl` packets
from a public-trace input, recursively removes evaluator-only fields, and
leaves the five label fields blank. After two raters complete the packets, run
its `agreement` command to report raw agreement and Cohen's kappa before
adjudication. The script does not invent labels, so the confirmation gate
remains pending until independent raters complete and adjudicate the packets.
