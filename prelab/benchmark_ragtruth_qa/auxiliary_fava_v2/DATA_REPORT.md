# FAVA synthetic auxiliary candidates, v2

Data staging and independent CPU audit completed: **7,482 answers, 7,408 material groups, 19,729 unchanged synthetic spans**. Largest group: 7. The fixed starting set was 7,483 exactly aligned factual-only examples; no model scores or QA calibration results selected these rows.

The sole v2 correction excludes empty/whitespace-only material from exact source matching and shared-reference links. All five original Reference sections remain in the input, including empty ones. The fixed candidates contain exactly one empty section: raw 11,525, Reference 1; no other empty/whitespace-only sections were found. Only that answer was restored. The original 7,409 candidate components and all group IDs are unchanged, because its empty block had no other FAVA member. All common 7,481 answers, evidence, labels and other fields are exact v1 copies. All v1 manifest artifacts retain their original hashes.

Raw 3,705 remains quarantined because its third Reference shares 69 distinct consecutive-20-word hashes with blocked RAGTruth material. The independent blocked-only scan confirms that none of the retained references matches blocked material under the fixed nonempty exact/20-word rule. The original answer text, reference boundaries and all 19,729 span offsets/texts passed the independent audit.

The question is empty. The stored released prompt is the actual FAVA checking-prompt prefix, not an original user question or a native generator trace. Corrections and edited projections are not model inputs. Unmarked positions are noisy silver negatives under synthetic annotation, not human-verified true statements.

Source isolation uses the 989 QA identities and a conservative complement of confirmed non-QA training identities, then propagates prior material links. This is not complete recovery of every original official split. Shared-reference groups can connect distractors and do not prove event/entity independence; paraphrase overlap remains outside the matching rule.

This data stage performed no tokenization, model loading, GPU computation or training, and read no official-test answers/labels. Root owns the separate `tokenization_v2` stage and its normalization-aware mapping. The original human QA training/evaluation files remain unchanged.
