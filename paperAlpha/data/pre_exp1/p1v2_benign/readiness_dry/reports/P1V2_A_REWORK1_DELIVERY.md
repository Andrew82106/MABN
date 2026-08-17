# P1v2-A offline admission — first rework delivery

## Current artifact and historical status

- Current reworked dry-run: `P1V2-READINESS-DRY-20260801T000002000000Z`.
- Historical artifacts retained unchanged: `...000000000000Z` (preliminary)
  and `...000001000000Z` (pre-rework). Neither is current evidence under this
  implementation.
- This delivery is only an offline output-contract admission artifact. It is
  not a new P1 run, a P2 artifact, a model selection, or a P1/P2 Go decision.

## Rework controls

- The sole writer reads only canonical config and fixture files and records
  exact config, fixture, schema, prompt, entry-script, and source-module
  identities in `execution_inputs`.
- Validation and replay derive all artifact paths from the run ID; they never
  open paths declared by an untrusted manifest. `.env`, UNC/device, traversal,
  legacy-P1, and non-P1v2 roots are rejected before read/create operations.
- Strict parsing rejects duplicate keys, nonstandard constants, and nonfinite
  JSON numbers. Malformed manifest/event/outcome nested values fail closed in
  both APIs and the CLI.
- The actual schema uses exclusive allow/reject branches. Allow requires only
  `public_report`; reject requires only a nonblank `rejection_reason`.
- `deterministic_fixture_resolutions`, expected cases, events, and outcomes
  must have the same count and ordered case-ID closure.

## Recorded execution evidence

- Cases / fixture resolutions / replayed cases: `12 / 12 / 12`.
- Public CLI validation: `passed=true`, zero fixture resolution, model,
  provider, and network calls.
- Public CLI replay: `passed=true`, zero fixture resolution, model, provider,
  and network calls.
- Manifest artifact hashes:
  - events: `bbac77a7238ea3abae70fc01d65e581d67d4dcf6b477f7aa030a8e1d5e213058`
  - outcomes: `a0ccbe315bdcdd48d75383d068efbfbe5f81807e674cd109a1ffe23f99323d8b`
  - validation: `147440fad22298a36ffd111b43b2bf04e0c71c6d4036d2b89cce20281b8cd9fb`
  - replay: `f5185993f5f2e370a265fb75a123f4c31a8a2e8c51e99121d14bc58e54eade02`
  - report: `64e8693c78412a179c89afc0d928104243e37b8a5da59a4e21b7a5be85c806d2`

## Verification completed

- `pre_exp1/p1v2_benign/tests`: 59 passed.
- `pre_exp1/p1_benign/tests`: 224 passed.
- `pre_exp1/p1_model_qualification/tests`: 32 passed.
- `pre_exp1/tests`: 57 passed.
- Rework protection/dependency comparison: passed. Protected-tree SHA-256 was
  `3f5a36c4d547875586bd19329582d74e8fc2e489fc50765d0e05ffe48da6432b`
  before and after. Dependency inventory was unchanged (10 distributions;
  SHA-256 `8e16bcaafc1b4ef6bb44342be3621b4b08c2a7b45f7455674c883fe19e345047`).
