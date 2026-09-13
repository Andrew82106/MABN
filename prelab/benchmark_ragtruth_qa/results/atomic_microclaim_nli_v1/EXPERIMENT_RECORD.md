# Atomic microclaim retrieved-NLI v1 complete record

## Scope and method

This development experiment uses only RAGTruth QA fit634 + calibration159. It keeps the unchanged 4-BPE stride-one labels, refines frozen answer claims into 11,329 deterministic atomic records, audits seven punctuation-only records, and scores 11,322 factual microclaims. Each microclaim retrieves top-2 sentences independently from passages 1/2/3. Frozen ModernBERT-base-nli scores each sentence and the in-source two-sentence concatenation. A 315-column evidence/relation representation includes E/N/C, BM25/coverage, citation source, subject/predicate, negation, comparison, quantity/unit, temporal, and condition compatibility; an optional 12-column clean group-OOF white-box summary is selected only through fit OOF.

Fit uses source-group 5-fold OOF predictions. The selected family and window/answer thresholds are determined only by fit OOF. Calibration is reporting only. Formal baselines and official test files remain untouched. The 3,680-answer expansion requires a separate version and frozen manifest.

## Results

| Result | fit OOF window F1 | strict cal window F1 | fit OOF answer F1 | strict cal answer F1 | cal-F1Opt window | cal-F1Opt answer |
|---|---:|---:|---:|---:|---:|---:|
| selected `fusion__hist_leaf7` | 0.607222 | 0.632379 | 0.758527 | 0.861386 | 0.642598 | 0.861386 |
| raw NLI risk | 0.316248 | 0.361109 | 0.694414 | 0.769841 | 0.373020 | 0.779528 |
| incumbent reference | — | 0.690281 | — | 0.891089 | — | — |

There are 98,854 NLI pairs: 37,785 bit-identical old pairs reused and 61,069 newly inferred. The independent final audit recomputed all strict/common metrics and verified every frozen artifact and source hash.

## Error review

Compared with the incumbent on calibration, the new method uniquely recovers 450 true-positive windows but uniquely adds 1,524 false-positive windows. Evident Conflict recall rises from 195/997 (19.56%) to 257/997 (25.78%); Evident Baseless Info rises from 3034/4086 to 3123/4086; Subtle Baseless Info rises from 548/814 to 600/814. This confirms complementary recall, but its precision is insufficient as a standalone replacement.

Gold microclaim labels projected through the exact same geometry yield calibration window F1 0.889351 and answer F1 1.0. The localization geometry therefore is not the limiting factor. The remaining problem is distinguishing genuine relation/source conflict from harmless lexical mismatch. A fit-only high-specificity router or relation-specific residual head should consume the complementary signal instead of broadcasting its full risk score.

## Preparation repairs

All four repairs happened before new NLI inference, training, or gold access. `failed_prepare_01` and `_02` exposed seven punctuation-only inherited records; the protocol was refrozen to audit rather than score them. `_03` fixed a stale dense-ID assertion after filtering without changing inputs. `refrozen_after_cpu_review_04` made old-cache reuse stricter after detecting at most 5.9e-6 FP32 padding-batch variation: only bit-identical repeated pairs are reused and all ambiguous repeats are recomputed. Full CPU preparation/check was rerun after every semantic or execution-rule change.
