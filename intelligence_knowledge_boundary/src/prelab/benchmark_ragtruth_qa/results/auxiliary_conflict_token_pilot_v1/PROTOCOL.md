# Auxiliary human-conflict token pilot v1

## Question

Can 4,381 released human Evident/Subtle Conflict spans teach a dedicated
evidence-contradiction signal that recovers high-overlap QA conflicts missed by
expanded-v4, without turning baseless material into conflict?

## Frozen data scope

- Auxiliary stage: all allowed `auxiliary_human_v1/candidate_fit.jsonl` records.
  Its already-audited 1,558 material groups remain indivisible. Quarantined
  sources are absent from this file.
- QA stage: expanded-v4 fit only. For the pilot, held source-connected fold 0 is
  evaluation; folds 1--4 are training.
- Calibration and official test are absent from every preparation and training
  command.

## Frozen label-preserving transform

1. Reproduce the existing label-blind microclaim transform: NLTK English Punkt
   parent spans, the existing 96-BPE long-parent boundary rule, then the exact
   `atomic_relation_audit_r32_v1.atomic_split` rule.
2. Pair each exact microclaim text with the task/question and deterministic
   claim-only BM25 top two sentences from its original material. Retrieval never
   reads labels, scores, model identity, calibration, or test.
3. Tokenize `(premise, exact_microclaim)` with the local
   `tasksource/ModernBERT-base-nli` tokenizer without truncation.
4. Supervise only hypothesis tokens. A token's soft conflict target is the
   fraction of its alphanumeric source characters inside an unchanged EC/SC
   span. Tokens touching EBI/SBI are ignored. Alphanumeric characters outside
   every released error span are exact non-conflict negatives. Premise, special,
   punctuation-only, and baseless-overlapping tokens receive zero loss weight.

No sentence, microclaim, or original Llama BPE inherits a positive label merely
because another character in its sentence is annotated. The released character
span is the sole positive supervision.

## Frozen example selection

- Retain every microclaim with at least one conflict-supervised character and no
  baseless-supervised character.
- Within each material group, retain at most the same number of clean-safe
  microclaims. Rank safe candidates by distance from that group's conflict
  median word count, then top-two evidence query coverage, then SHA256. This is
  deterministic and fixed before outcome evaluation.
- Groups with conflict but no safe candidate retain their conflict examples;
  loss weighting handles any remaining imbalance.
- Apply the same rule independently to auxiliary fit and QA folds 1--4.

## Frozen model and loss

- Initialization: the completed expanded-v4 fold-0 ModernBERT checkpoint.
- Architecture: reuse its ModernBERT encoder, NLI prediction head, and three-way
  classifier at every hypothesis token. Conflict logit is
  `C - logsumexp(E,N)`.
- Loss: soft binary cross entropy on the exact supervised hypothesis tokens.
  Base weight gives equal material-group, answer, selected-microclaim, then
  supervised-character mass. E/C token target mass is balanced inside the
  current training stage and group mass is restored.
- Optimizer: fresh AdamW, learning rate 3e-6, weight decay 0.01, clip norm 1,
  no scheduler; seed 20261023; FP32 parameters/optimizer, BF16 CUDA forward,
  gradient checkpointing; padded-token budget 1536, at most 8 examples,
  accumulate 4 microbatches.
- Candidate stream: one auxiliary pass, followed by one QA folds-1--4 pass;
  optimizer state continues across the boundary.
- Same-budget control: identical initialization, architecture, loss, optimizer,
  and final QA pass. It replaces the auxiliary pass with deterministic cycling
  of QA folds-1--4 batches for exactly the candidate auxiliary optimizer-update
  count. There is one candidate and one control; no grid or epoch selection.

## Frozen fold-0 evaluation

- Map hypothesis-token conflict probabilities back through exact alphanumeric
  character overlap to the existing raw answer BPEs. Each existing eligible
  4-BPE stride-1 window takes the maximum mapped conflict score.
- Keep expanded-v4 unchanged and combine by the parameter-free score
  `max(v4_window_risk, conflict_window_probability)`.
- Report overall window AP and held-fold F1-opt diagnostic, EC/SC AP and recall,
  and FP at the frozen v4 fold-0 alert budget. Answers use maximum window score.
- Report the high-overlap subset fixed as selected-evidence union query coverage
  at least 0.5. It is diagnostic only.

The pilot is useful only if the candidate beats the same-budget control on
combined EC/SC AP and, at the v4 alert budget, raises EC+SC recall by at least
0.02 without increasing overall FP, while combined overall AP is no more than
0.002 below unchanged v4.

## Stage gate

`CPU prepare -> CPU independent audit -> human/root review -> optional GPU
smoke -> candidate/control GPU run`. This artifact stops before GPU smoke.
Published baselines, their structures, and their scores are untouched.
