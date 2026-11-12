# Who_and_When external failure-localization audit

Algorithm-Generated only; n=126 and all rows are failures (`is_correct=false`).

This is intervention/failure-localization evidence, not a binary safety-set evaluation. The two-layer monitor was fitted once on independent_mas_v3 (all 4,000 public traces), frozen, and applied to text-derived public message events. Who_and_When `mistake_step` is used only to align prefixes and is never a fitting feature.

## Summary

- Monitor score change at the mistake boundary: mean **0.039**, median **0.000**; positive in **5.6%**.
- First alarm rate at fixed threshold 0.5: monitor **19.8%**, edge-count **84.1%**, length **55.6%**.
- Alarm distance is `mistake_step + 1 - first_alarm`; positive means the alarm precedes the annotated mistake step. See `metrics.json` and `episodes.jsonl` for full per-prefix traces.

## Caveats

- The external corpus has no correct controls here, so no AUROC/F1 or safety classification claim is made.
- History is converted to sequential message edges (previous speaker → current speaker); this is a transparent adapter, not a claim about the original runtime DAG.
- Text content is retained only as public event evidence; annotator labels/reasons are excluded from model fitting.
