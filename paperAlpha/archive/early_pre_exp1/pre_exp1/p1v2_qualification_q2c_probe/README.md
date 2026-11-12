# P1v2 Q2-C Codex response-behavior probe

This directory contains two deliberately separated execution paths:

- `offline_fake`: a deterministic fake-transport qualification run used to
  verify evidence, deadlines, redaction, validation, and replay mechanics;
- `live_probe`: a guarded, one-time runner for the fixed local endpoint and
  fixed target model.  Its transport is tested here only through injected fake
  HTTP connections.

The normal validation and replay paths never read credentials and never make a
network request.  The live branch is not selected by default: it is reachable
only through `scripts/run_q2c_probe.py --allow-live-q2c`, reserves its
one-time guard before reading the three named credential variables, and has no
retry loop.  The offline implementation and unit-test task does not execute
that branch.

The separately authorized live execution has since completed exactly once:
`P1V2Q2C-PROBE-20260802T061803504801Z` made one metadata request and four
fixed, benign completion requests.  Its safe summary records `observed`, with
all four profiles classified as `content_match`; the public
validation → replay → validation sequence passed.  This is only a narrow
gateway response-behavior observation, not a model qualification, P1/P2
unlock, or research result.  The one-time Q2-C live authorization is consumed.
See the [run-local DELIVERY](../../data/pre_exp1/p1v2_qualification_q2c_probe/runs/P1V2Q2C-PROBE-20260802T061803504801Z/reports/DELIVERY.md).

Run the offline checks from the repository root with the fixed project Python:

```powershell
conda run --no-capture-output -n multi_agent_graph python -B -m unittest discover -s paperAlpha/pre_exp1/p1v2_qualification_q2c_probe/tests -v
conda run --no-capture-output -n multi_agent_graph python -B paperAlpha/pre_exp1/p1v2_qualification_q2c_probe/scripts/run_fake_q2c_probe.py
conda run --no-capture-output -n multi_agent_graph python -B paperAlpha/pre_exp1/p1v2_qualification_q2c_probe/scripts/validate_q2c_probe.py --run-id <run-id>
conda run --no-capture-output -n multi_agent_graph python -B paperAlpha/pre_exp1/p1v2_qualification_q2c_probe/scripts/replay_q2c_probe.py --run-id <run-id>
```

Artifacts retain only hashes, lengths, outcome classes, and bounded provider
identity fields.  They never persist request text, response text, tool
arguments, reasoning text, error bodies, or credentials.  Offline and live
artifacts use distinct schema identities and distinct canonical `DELIVERY.md`
claims; validator and replay rebuild those claims field by field.
