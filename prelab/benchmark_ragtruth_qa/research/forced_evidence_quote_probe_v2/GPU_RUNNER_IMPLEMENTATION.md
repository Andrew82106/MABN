# Forced-Evidence Quote Probe V2 GPU runner implementation

Status: CPU implementation selfcheck passed; GPU has not been initialized or
used.  Independent static review is still required before `gpu-smoke`.

## Frozen inputs and lineage

- Runner: `src/run_forced_evidence_quote_probe_v2_gpu.py`
- Runner SHA-256: `eb4a51353ba4edaff2cff877f576ff206f1fca1d6bd820928fb0db4acebfcca2`
- V1 scientific protocol SHA-256: `78e5558b61592238b0979c7c60aea23c79b8c7d3181e88113d3972b4d056f7aa`
- V1 plan SHA-256: `6d439ee257dde65d2ef2c9974cd3dd716f4f0d06890cfb7df9c8af6da562680c`
- V2 numerical protocol SHA-256: `ad0a2198ba69e75c78852eab103e8a829b6f9fb21389f322d5879f016e989df1`
- Sole label-free input SHA-256: `ae6bf145a0e48ff310cdde5f54457eec7e8986c9b012d52cc463bb9e357e2777`
- CPU selfcheck SHA-256: `a6ef47c093b11f1665e340b703bbd3275c37bcac07025fe9ae74eed6381c86cf`

V1 protocol, plan, preparation, manifest, and label-free input are read-only.
All V2 records, receipts, failures, and reviews use separate V2 directories.

## Implemented numerical behavior

- KV-cached greedy decoding is authoritative for generated IDs, stop position,
  selected log probability, full-vocabulary entropy, and signed margin.
- Full no-cache replay is authoritative only for hidden64 and attention.
  Replay logits are retained solely for descriptive numerical QA and never
  enter the 549-dimensional feature vector.
- The fixed eight-claim smoke applies `rtol=0` and absolute tolerances
  `0.0625 / 0.0625 / 0.5`.  An argmax mismatch passes only when the cached
  generated-token margin is at most `0.5` and the replay margin for that same
  cached token is at least `-0.5`.
- Formal extraction records each claim's mismatch count/rate and absolute-drift
  max/mean/P50/P95/P99.  These values do not filter, retry, or fail a row.
  The completion receipt recomputes every per-claim summary and produces the
  same statistics over all generated tokens.
- The shortest and longest smoke claims are repeated.  Cached IDs and structural
  metadata must match exactly; cached statistics and replay hidden/attention
  must match with `rtol=0, atol=1e-6`.
- Parse-invalid output still uses its actually generated content.  Only a true
  `T=0` produces zero token, hidden, and attention blocks, as in V1.

## Resume and review gates

Atomic claim records remain individually hash-bound.  If all 3,776 records are
present after an interruption but the completion marker is absent, `extract`
rebuilds the whole-batch audit and completion receipt without loading a model or
initializing CUDA.  A partial batch resumes only the missing atomic records.

`gpu-smoke` requires an independent runner review bound to the runner, CPU
selfcheck, V1 protocol/plan, and V2 numerical protocol hashes.  Full extraction
also requires a separately hash-bound independent smoke review.

## CPU checks run

The CPU selfcheck verified all 3,776 label-free rows and the fixed smoke indices,
mechanical quote digest, prompt/token geometry, SentencePiece and byte-fallback
offset provenance, malformed-tag handling, P1/P2/P3 widths, Q/K hook equivalence
to eager attention, atomic record round-trip, complete-marker reconstruction,
and WDDM-safe GPU exclusion.  Synthetic numerical cases verify inclusive smoke
boundaries, allowed near-tie flips, smoke failure for a high-confidence conflict,
and record-only handling of the same conflict during formal extraction.

