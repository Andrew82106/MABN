# Human-correction pair feasibility audit: repair v2 (stopped)

## Status

This proposed training protocol was withdrawn before data generation or model
training. It is retained only to record the candidate design and the stopping
decision. It must not be treated as a trainable dataset specification.

## Frozen scope

The only dataset is the already-quarantined
`auxiliary_human_v1/candidate_fit.jsonl`. Only released Evident/Subtle Conflict
annotations are candidates. Baseless annotations are used solely to reject a
microclaim that still contains another error. No QA, calibration, or official
test file is opened. Prior preparations, runners, and published baselines are
not modified.

All rows derived from a material `group_id` use the same deterministic fold:
`int(SHA256("20261024|" + group_id)[:16], 16) mod 5`. Fold 0 is an internal
diagnostic; folds 1--4 are training.

## Candidate transform that was not approved for training

For every one of the 4,381 human conflict spans:

1. Parse literal line-leading `Original:` and `AIGC:`/`Generated:`/
   `Generative:` fields. Empty, absent, or structured key-value corrections are
   excluded. No model completes or rewrites metadata.
2. The full normalized `Original:` word sequence must occur contiguously in one
   exact source sentence. Distinct matching source sentences are ambiguous and
   excluded. This gives source lexical coverage 1.0, above the fixed 0.8
   priority threshold.
3. The full normalized generated sequence must occur contiguously and uniquely
   inside one label-blind atomic microclaim, with every matched word inside the
   target EC/SC character span. The microclaim may touch no other human error
   span at an alphanumeric character.
4. Align generated and original sequences with deterministic `SequenceMatcher`
   over casefolded Porter stems. Require at least one unchanged anchor token.
   Every non-anchor opcode must be `replace` with the same number of generated
   and original tokens. Total replaced tokens may not exceed
   `max(2, ceil(0.25 * generated_words))`.
5. For each accepted replacement block, use the actual token offsets in the bad
   microclaim and exact source sentence. Replace the block plus its adjacent
   inter-token punctuation with the exact source block. Apply blocks right to
   left. Reject capitalization damage, unchanged output, or any word-count
   change.

This transform was considered as an audit rule, but it was not accepted as a
gold-label generator. Independent review found that surface rules still admit
style and source-copy shortcuts and cannot certify grammatical or semantic
equivalence. No `repair_v2/triples.jsonl` or training arrays were generated.

## Proposed triples and supervision (not executed)

Each retained triple uses one premise—exact question plus exact source evidence
sentence—and three claim-level hypotheses:

- unchanged bad microclaim: conflict;
- same microclaim after the traced factual-slot edit: non-conflict;
- exact source evidence sentence: non-conflict.

No token label exists. The verifier produces one scalar per complete claim.

## Proposed verifier and loss (not executed)

- Local ModernBERT NLI backbone, explicit SDPA and
  `reference_compile=False`.
- A separate scalar conflict head on its mean-pooled NLI representation,
  initialized as `C - mean(E,N)`; backbone and head train jointly.
- Classification loss:
  `(BCE(bad,1) + BCE(corrected,0) + 0.5*BCE(source,0)) / 2.5`.
- Pair ranking:
  `(softplus(-(bad-corrected)) + softplus(-(bad-source))) / 2`.
- Total is classification plus ranking. Material groups have equal total weight.
- One pass over folds 1--4; AdamW 3e-6, weight decay 0.01, clip 1, no scheduler,
  seed 20261024; FP32 state, BF16 CUDA forward; at most eight triples and 4096
  padded tokens per microbatch, accumulation four. No grid, early stopping, or
  checkpoint selection.

Fold 0 reports pair AP/F1 and `bad>corrected` / `bad>source` ranking accuracy
before and after training. Later QA use assigns each claim score to its
overlapping existing 4-BPE windows; it does not revive tokenwise supervision or
tokenwise maximum.

## Stop rule and observed result

The design stops if automatic replacement cannot supply high-precision gold
pairs without human rewriting. The independent audit reviewed 100 fixed,
task×conflict-stratified examples: 14 were direct pairs, 62 required rewriting,
and 24 were rejected. Its automatic high-precision queue contains 272 spans / 263
unique pairs, falling to 150 / 146 when answers with any other released label
are excluded; every item still requires human verification. This stop rule was
therefore met. See `../STOP_REPORT.md`.
