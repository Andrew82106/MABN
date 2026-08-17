# Implementation boundary

`p1v2_benign` is intentionally self-contained.  It does not import prior P1
packages or schemas, does not access credentials, and has no remote, network,
or model execution path.  The contract is strict JSON: duplicate keys,
non-standard numeric constants, blank input, unknown fields, and mismatched
role or episode values are rejected.

The generated data role is `p1v2_readiness_dry`.  Its machine-readable
identity records that it is not a new P1 run and is ineligible for P1-gate,
P2, confirmatory, and causal-effect analysis.

The writer has no config or fixture path override: it freezes only canonical
P1v2 inputs and records their exact identities together with the schema,
prompt, entry scripts, and source modules. Public validation and replay derive
artifact paths locally, reject `.env`, UNC/device, legacy-P1, traversal, and
outside-namespace paths before file I/O, and require an exact 1:1 closure of
expected cases, events, outcomes, and fixture-resolution count.
