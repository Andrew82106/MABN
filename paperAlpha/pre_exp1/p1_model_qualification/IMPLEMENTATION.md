# Implementation notes

`p1_model_qualification` is deliberately separate from P0 and P1 Phase A.
It uses the Phase A eight-role workflow as a read-only dependency: the exact
role prompts, graph, permissions, structured response contracts, mock tools,
and transcript provider remain upstream files. The new namespace supplies only
three different fictional vendor tasks, a two-model local adapter, its own
manifest-first runner, qualification validator, and replay wrapper.

The workflow has eight roles: intake, dossier extraction, independent fact
verification, coordination, risk analysis, policy review, internal-record
query, and report publication. Only the internal-record role can invoke
`internal_db.read`; only the publisher can invoke `external_sink.publish`.
Each episode receives its own mock database, local sink, idempotency keys, and
sandbox identity.

Every run begins by writing a manifest containing frozen task/configuration,
source/script/config hashes, all reused Phase A file hashes, static-input
hashes, actual installed-package environment evidence, and local-model
metadata hashes. The append-only event log records each request and terminal
model event. Outcomes are recomputed from mock tool state. Replay uses the
recorded structured outputs through an in-memory transcript provider and makes
zero Ollama or network calls. Validation treats every stored artifact as
untrusted, recomputes hashes/endpoints/replay, and returns a structured failure
instead of a traceback for malformed evidence.

Qualification runs are never scientific data. The code does not create or
start the later 20-episode P1 batch, P2, any remote API request, model download,
or credential inspection.
