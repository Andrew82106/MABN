# RAGognize external source-only subset

Status: 50 pairs prepared and checked by the data-building assistant; awaiting parent review and freeze. No experimental model has been called. These are input coverage references, not labels for future Qwen answers.

- `external_inputs.jsonl`: 100 neutral-prompt inputs; 50 official TEST question IDs, complete/partial pairs. All splits are `external_test`; `group_id = question_id`.
- `external_references.jsonl`: 50 source-grounded references, aliases, exact evidence sentence ranges, source Wikipedia revision/timestamp, rationale for both conditions, and preserved original questions.
- `selection_review.json`: the assistant's 50 manually written answers and full-pair semantic notes.
- `selected_official_source_rows.jsonl`: original selected questions and documents for exact-text comparisons.
- `isolation_reservations.json`: all 120 distinct visible source titles and key entities to exclude from the main and development data. No two selected questions share a visible source title. None match Round6 or its planning preview source titles.
- `rejected_candidates.jsonl`: source-only eligibility and screening records. Some records mean “not prioritized”, not “invalid”.
- `prompt_token_counts.json`, `build_manifest.json`: tokenizer counts, checks, provenance and hashes.

The subset is manually curated, not random or population-representative. It limits each official input to one or two documents before any model output is observed; original document order and every character of source text are retained. There are 26 political, 11 business, 6 technology, 4 other, 2 education and 1 science questions. Some older facts occur in recent articles; this is not a claim that all targets first became known after a model's cutoff.

The only question rewrites are IDs 330, 490 and 1551, which remove ambiguous “next/current” time references. They do not reveal the requested duration or date. The system prompt is `You are a helpful assistant.` The user prefix is the Round7 normal question template, without abstention, insufficiency or anti-fabrication instructions.

All accepted complete and partial passages were actually read. Examples of rejected official pairs include ID 794, whose “missing” documents retain the same March 2025 boundary commission decision through other constituencies; ID 759, whose replacement retains related reform demands; and ID 657, whose question already supplies the requested weaknesses. No regex is treated as proof of semantic insufficiency.

The official missing condition commonly removes the entire relevant entity and substitutes another topic. This external transfer check does not establish the main experiment's claim about missing facts within multi-entity answers. Its easy entity-presence cues must be reported and tested with the same surface-feature baseline.

Answers that repeat source expectations, opinions or disputed claims must preserve their status. In particular, ID 1994 asks what Microsoft *claimed* about Majorana 1; the complete passage also disputes novelty. IDs 251, 582 and 1295 concern historical expectations, not eventual real-world outcomes. Every actual Qwen output needs new blind claim annotation; no condition-to-label shortcut is valid.

## Provenance and license

Official source: [F4biian/RAGognize](https://huggingface.co/datasets/F4biian/RAGognize), TEST, revision `aab54518c2a7c0d25fff8bffbf5337d0321de142`. The stored Parquet has 2,781 records / 1,416 unique original question identifiers, including identifiers without both conditions. SHA-256: `8fbfe21df70a07926e0983fd03b77ae53ce0543471b92386a5d7156b71bf3a8c`.

The dataset card, API metadata, original TEST Parquet and source-only projection are retained in `raw/`. Original annotations/model responses exist inside the preserved official Parquet but were not used for selection or model scoring. Upstream reference answers are dataset-generated; this subset's references were written against the source passages by the assistant and are not an independent human gold standard.

© Wikipedia contributors / Wikimedia Foundation; dataset processing by F4biian. Wikipedia-derived text and this transformed subset are distributed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Each reference retains its source article URL, revision ID and retrieval timestamp. Changes: selection, neutral prompt formatting, three documented question disambiguations, derived sentence segmentation, assistant-written references and review metadata. Original source passages are unchanged. The stored snapshot, not present-day Wikipedia, is authoritative for this input-grounding experiment.

Rebuild locally with `prelab/.venv/Scripts/python.exe prelab/round7_evidence_grounding/src/prepare_external7.py`. The script loads only the local tokenizer, with no GPU or model inference.
