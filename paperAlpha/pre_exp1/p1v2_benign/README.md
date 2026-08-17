# P1v2-A offline readiness admission

This namespace implements only a deterministic, offline admission check for a
future P1v2.  It is not a P1 run, P2 artifact, causal analysis, model choice,
or authorization for a real model call.

The only writer is the `dry` command: it creates a new, non-overwritable
`P1V2-READINESS-DRY-*` artifact.  `validate` and `replay` read an existing
artifact, resolve no fixture, and write nothing.  The manifest SHA-256 map
covers events, outcomes, validation, replay, and report; the manifest itself
is deliberately excluded from that map to avoid a self-referential hash.

```powershell
conda run --no-capture-output -n multi_agent_graph python -B pre_exp1\p1v2_benign\scripts\capture_baseline.py --phase before --label rework1
# The writer requires a new unused ID; do not reuse the documented current artifact.
conda run --no-capture-output -n multi_agent_graph python -B pre_exp1\p1v2_benign\scripts\run_readiness_dry.py --run-id P1V2-READINESS-DRY-20260801T000002000000Z
conda run --no-capture-output -n multi_agent_graph python -B pre_exp1\p1v2_benign\scripts\validate_readiness_dry.py --run-id P1V2-READINESS-DRY-20260801T000002000000Z
conda run --no-capture-output -n multi_agent_graph python -B pre_exp1\p1v2_benign\scripts\replay_readiness_dry.py --run-id P1V2-READINESS-DRY-20260801T000002000000Z
conda run --no-capture-output -n multi_agent_graph python -B pre_exp1\p1v2_benign\scripts\capture_baseline.py --phase after --label rework1
conda run --no-capture-output -n multi_agent_graph python -B pre_exp1\p1v2_benign\scripts\compare_baselines.py --label rework1
```

All artifacts are confined to `data/pre_exp1/p1v2_benign/`.  The implementation
uses only the Python standard library and a deterministic fixture source.

The documented current artifact is `P1V2-READINESS-DRY-20260801T000002000000Z`.
Both `...000000000000Z` and `...000001000000Z` are retained unchanged as
historical preliminary/pre-rework evidence. Neither is documented as current
or passing under the reworked implementation provenance.
