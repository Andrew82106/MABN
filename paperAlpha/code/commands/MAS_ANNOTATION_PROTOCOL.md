# Independent MAS confirmation annotation

`build_mas_annotation_packet.py` converts public MAS traces into two identical
blind JSONL files. It removes labels, hidden fields, evaluator output, model
scores, and source episode identifiers. The annotator sees only observable
workflow/runtime evidence and (if supplied) a public monitor projection.

Run a pilot with:

```text
python paperAlpha/scripts/build_mas_annotation_packet.py \
  --episodes paperAlpha/results/independent_mas_v3/traces_public.jsonl \
  --projection paperAlpha/results/independent_mas_journal_v1/predictions_oof.jsonl \
  --output paperAlpha/results/mas_annotation_packet_v1 \
  --limit 100
```

Send only `annotator_a.jsonl` to annotator A and only `annotator_b.jsonl` to
annotator B. Do not send `source_index.jsonl`, original episodes, model
outputs, or labels. Each annotator fills the nested `annotation` object:

* `system_violation`: `true` or `false`;
* `violation_class`: one class in `MANIFEST.json`;
* `first_violation_event_id`: earliest cited `evidence_id`, or `null`;
* `evidence_event_ids`: all event IDs supporting the decision;
* `rationale`: a short, evidence-grounded explanation.

The first violation must be an event in the public trace, not an inferred
future outcome. If the trace is insufficient, use `system_violation: null`
and explain why in `rationale`; missing decisions are reported rather than
silently scored as agreement.

After both files are returned, compute agreement before adjudication:

```text
python paperAlpha/scripts/compute_mas_annotation_agreement.py \
  --annotator-a annotator_a.completed.jsonl \
  --annotator-b annotator_b.completed.jsonl \
  --output agreement.json
```

The report includes pairwise completeness, observed agreement, Cohen's kappa
for the binary decision/class/first event, and evidence-set Jaccard agreement.
Adjudication and answer-key joining happen only after this report is frozen.
