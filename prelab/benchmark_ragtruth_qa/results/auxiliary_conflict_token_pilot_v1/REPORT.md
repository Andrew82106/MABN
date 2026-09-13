# Auxiliary exact-span conflict pilot: CPU review

## Status

The fold-0 pilot is fully specified and CPU-audited. No checkpoint was loaded,
no CUDA operation ran, and no calibration or test row was read. GPU execution
remains gated on review.

## Audited data

| Item | Count |
|---|---:|
| Released human Evident/Subtle Conflict spans | 4,381 |
| Evident / Subtle Conflict spans | 4,252 / 129 |
| Auxiliary atomic microclaims before selection | 110,788 |
| Clean conflict / matched safe auxiliary microclaims | 4,064 / 4,064 |
| Auxiliary material groups selected | 1,205 |
| QA folds 1--4 conflict / matched safe microclaims | 341 / 339 |
| QA training / held material groups | 142 / 122 |
| QA fold-0 held answers / microclaims | 747 / 6,983 |
| QA fold-0 eligible 4-BPE windows | 132,213 |
| Positive / EC / SC held windows | 11,734 / 928 / 65 |

All 4,381 released conflict spans align to their unchanged source answer and an
atomic microclaim. Of them, 4,058 occur in a selected clean-conflict
microclaim. The other 323 remain in the trace file but are excluded because the
same microclaim also touches baseless annotation; they are not relabelled or
expanded into token targets.

The full QA fit prefix has 615 material groups and every group belongs to one
fold. The selected QA training and fold-0 held group intersection is zero. The
upstream auxiliary audit also reports zero retained material links to QA fit
after quarantining 62 sources.

## Exact supervision audit

The independent audit reconstructed all 3,479,473 stored token targets from the
original answer text, human span offsets, and hypothesis token offsets. Positive
weight is assigned only to the fraction of a hypothesis token's alphanumeric
characters inside an EC/SC span. Tokens touching EBI/SBI, premise tokens,
special tokens, and punctuation-only tokens have zero loss weight.

The reconstruction found 30,896 positive tokens, four soft boundary tokens,
and 7,507 ignored tokens that touch baseless spans. It also found 3,513 conflict
microclaims containing both exact positive tokens and exact local safe tokens.
This directly rules out sentence-level or microclaim-level positive-label
broadcasting.

The audit independently rebuilt the held answer order, lexical window
eligibility, risk labels, EC/SC labels, and exact token-to-window mapping. The
unchanged expanded-v4 window score reproduces AP `0.6390503697171015`.

## Frozen GPU comparison

Both arms start from the same expanded-v4 fold-0 ModernBERT NLI checkpoint and
use the same tokenwise classifier, exact-span loss, optimizer, seed, final QA
pass, and held evaluation.

- Candidate: one pass over 8,128 auxiliary examples, then one QA training pass.
- Control: cycle the QA training batches for the same 395 pre-stage optimizer
  updates, then run the same final QA pass.
- Readout: token conflict probability is mapped to the existing 4-BPE windows;
  the only combination with unchanged v4 is the parameter-free maximum.

The control processes 13.5% more padded tokens in its pre-stage because its QA
batches are shorter, so equal update count is conservative for the candidate.
No threshold or hyperparameter is selected from fold 0.

The pilot passes only if the candidate beats the control on conflict AP, raises
combined EC+SC recall by at least 0.02 at the unchanged v4 alert budget without
adding false positives, and keeps overall AP within 0.002 of v4.

## Runtime estimate

The existing v4 throughput gives 9.6 minutes for both training arms and both
held inferences. A conservative allowance is 17.7 minutes including model
load/save. Peak GPU memory is unknown until the separately gated smoke run.

## Review artifacts

- `PROTOCOL.md`: frozen data, model, loss, control, evaluation, and gate.
- `preparation_complete.json`: counts, isolation, source hashes, and scope.
- `INDEPENDENT_AUDIT.json`: independent reconstruction results.
- `CPU_CHECK.json`: two-example CPU gradient-path check.
- `GPU_PLAN.json`: deterministic batch/update and runtime plan.
