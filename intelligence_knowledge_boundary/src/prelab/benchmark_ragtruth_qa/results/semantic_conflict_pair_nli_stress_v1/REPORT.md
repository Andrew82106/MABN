# Frozen ModernBERT pair stress test v1

These are fit-only synthetic silver sensitivity results, not human-gold factual
accuracy and not held-out RAGTruth evaluation. The checkpoint is unchanged and
no model was trained.

| type | pairs | C pair acc | mean ΔC | C endpoint AUROC | risk pair acc | mean Δrisk | risk endpoint AUROC | hard E→C pair acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 4109 | 0.9852 | 0.7138 | 0.9858 | 0.9642 | 0.6895 | 0.9715 | 0.7396 |
| entity | 12 | 1.0000 | 0.8190 | 1.0000 | 1.0000 | 0.8546 | 1.0000 | 0.9167 |
| number | 291 | 0.9897 | 0.7353 | 0.9869 | 0.9897 | 0.7359 | 0.9747 | 0.7801 |
| negation | 1944 | 1.0000 | 0.9601 | 0.9998 | 1.0000 | 0.9391 | 0.9993 | 0.9799 |
| temporal | 348 | 0.9971 | 0.7460 | 0.9861 | 0.9885 | 0.7602 | 0.9719 | 0.7730 |
| attribution | 1514 | 0.9624 | 0.3851 | 0.9339 | 0.9075 | 0.3425 | 0.8653 | 0.4141 |

Pair accuracy uses the strict within-pair comparison `corrupted > supported`;
ties count as incorrect and are available in `summary.json`. Risk is
`max(C, 1-E)`, which equals `1-E` for normalized E/N/C probabilities. Endpoint
AUROC instead pools the two endpoint roles and labels corrupted endpoints 1.

The separate 92-row research-agent QC contains 84
clear silver, 5 ambiguous, and
3 reject. Its strata are descriptive only;
ambiguous/reject rows were not removed from the primary 4,109-pair result.
