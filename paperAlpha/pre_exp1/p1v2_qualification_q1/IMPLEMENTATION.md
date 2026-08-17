# Q1 implementation contract

`provenance/source_contracts.json` freezes the only seven Q0 files that Q1 may
read for provenance comparison. Their byte-identical copies live in this Q1
tree; Q1 imports no Q0 Python package. Preflight checks the frozen protocol,
task cards, schema, prompt, sink rule, Q0 hashes, model tag fingerprint, and
Ollama version before it can issue a chat request.

The runner is sequential and has no retry path. It writes an authorization
record before its first chat request, records exactly one terminal state per
started request, never repairs output, suppresses raw reasoning if a thinking
field or `<think>` marker appears, and permits only valid publisher reports to
the four-field public sink. Screen failures block Confirmation but do not stop
remaining Screen cards. Environment, transport, timing, identity, provenance,
ledger, validation, and replay faults have priority over model-output faults:
their single batch decision is always `rework`.

The public validator recomputes hashes, task coverage, payloads, state-machine
rules, provenance, sink contents, and the decision from the ledger. The public
replay validates and replays ledger order without networking or model calls.
