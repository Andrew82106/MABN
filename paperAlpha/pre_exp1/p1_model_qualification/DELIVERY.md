# P1 local-model qualification delivery

The mandated offline gates, historical-run checks, and both authorized local
three-episode qualification batches are complete. This record is an evidence
index for independent acceptance; it is not a scientific conclusion or model
freeze.

```text
P1 scientific gate: NOT_STARTED
```

## Evidence

- Offline qualification tests: `32 passed`.
- P1 Phase A regression: `224 passed`.
- P0 regression: `57 passed`.
- qwen3:8b: `P1-BENIGN-QUAL-QWEN3-20260731T121840716022Z`;
  validation and replay passed; 3 episodes / 27 model calls.
- ministral-3:8b: `P1-BENIGN-QUAL-MINISTRAL3-20260731T121957481982Z`;
  validation and replay passed; 3 episodes / 27 model calls.
- Inventory: Ollama `0.24.0`; both exact local tags present; no pull or
  mutation command attempted.
- Summary: `paperAlpha/data/pre_exp1/p1_model_qualification/reports/model_qualification_summary.md`.

All qualification manifests record `eligible_for_scientific_analysis=false`
and `scientific_gate=NOT_STARTED`. Historical P0 raw-event SHA-256 values
were rechecked and remain unchanged. Main-Agent independent acceptance is
still required.
