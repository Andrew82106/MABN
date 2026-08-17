# P1v2 Q2-B repair final derived audit delivery

- Audit ID: `P1V2Q2B-REPAIR-AUDIT-20260802T044126194259Z`
- Artifact identity: `artifact_kind=derived_audit`, `is_derived_audit=true`, `is_new_run=false`.
- Derived from frozen original run: `P1V2Q2B-REMOTE-20260802T033140463464Z`.
- Source execution mode: `remote_staged_qualification`.
- Historical pre-rework audits retained unchanged: `P1V2Q2B-REPAIR-AUDIT-20260802T035603081713Z` and `P1V2Q2B-REPAIR-AUDIT-20260802T041527196020Z`.
- Interpreter: `D:\anaconda\envs\multi_agent_graph\python.exe`; Python `3.11.15`.
- Offline test command: `conda run --no-capture-output -n multi_agent_graph python -B -m unittest discover -s paperAlpha/pre_exp1/p1v2_qualification_q2b_repair/tests -v`.
- Offline test result: 15/15 passed.
- Public validation → replay → validation before final delivery binding: all passed with `errors: []`.
- Deadline evidence: start metadata, completion, and end metadata execute through the same pure in-memory deadline guard; start/completion/end timeout and zero/negative-budget blocking are covered by tests.
- Failure-evidence proof: failed `tool_calls` completion retains only recomputable request hash and safe provider identity; raw response, reasoning, tool arguments, error body, credentials, and Authorization are not persisted.
- Metadata identity oracle: validator and replay independently derive the exact model-list hash, target presence, and count from frozen `configs/repair_provider_profile.json`; coordinated transcript+ledger rechain tampering with a legal alternate two-model list is rejected by both.
- Delivery isolation: this local `reports/DELIVERY.md` is the only delivery report hashed by this audit manifest. A root-level current index, if present, is not an audit artifact.
- Provenance: original Q2-B per-file SHA-256 pre/post maps are stored in this audit and are equal.
- Network calls: 0. `.env` reads: 0. Live completion calls: 0.
- Original Q2-B code/data: unchanged. P1/P2: not started. Caches and temporary copies: cleaned.
- Repair decision: `passed` for the offline repair evidence chain only; this is not a model qualification conclusion.
