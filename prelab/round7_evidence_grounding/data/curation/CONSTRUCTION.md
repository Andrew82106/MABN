# Round7 data construction record

The main task is a new controlled QA task derived from HotpotQA comparison entity pairs and original source sentences. It is not a score on the unmodified HotpotQA benchmark. Original questions and gold answers are preserved as provenance. When the attribute changes, `derived_attribute_changed` is recorded; new values and comparisons are established directly from sources, independently of the old gold.

Only complete original sentence records are selected as short search-result snippets. The identical extraction rule applies to complete and partial inputs. The full original articles remain in raw downloads for auditing; unshown sentences are never passed to Qwen. A partial input substitutes one attribute-bearing snippet with a reviewed natural source snippet, preferentially retaining the same subject's general background. No facts are edited or invented.

Every main input has four search results and three neutral questions. The comparison question does not presuppose equality, inequality, a unique winner or a particular shared location. The prompt contains no condition label, answer, missing-evidence marker or refusal hint. References and coverage are evaluation-side records only.

Curation includes reading both key snippets, both backgrounds and the replacement; checking source values, reference comparison, question premises, indirect evidence, namesake disambiguation and source artifacts. Automatic scans help find duplicates and candidate dates, but are not described as semantic review. Independent assistant reviews are distinguished from human annotation.

The local Qwen tokenizer checks each removed/replacement pair: absolute length difference must not exceed max(20 tokens,20% of removed length). The total chat input must not exceed3072 tokens. Passage order uses seed20260910 plus the SHA256-derived question identifier. Both conditions use the same order. Source sentence indices and original text are saved.

Development20 is entirely separate from formal200. Source titles, subjects and displayed texts are isolated across groups; cross-question mentions receive semantic review because a word such as Chestnut in Chestnut Hill is a different referent from the plant genus. Old Round6 and planning-preview groups are excluded. Main120/validation40/test40 are assigned before generation with seed20260910 (train/heldout) and20260911 (validation/test), stratified by temporal versus non-temporal attribute and deletion side.

The external50 RAGognize questions use official test data and are held out. Their frozen100 inputs are concatenated unchanged with the400 main inputs. External scores never choose curation, model settings, detector settings or thresholds. All source curation precedes consulting model outputs; generated facts still require separate output annotation.

The source topic distribution will be reported as observed. The main data are a controlled encyclopedia-derived test of evidence use, not a collection of operational intelligence cases. The external recent-source set tests transfer to a different source distribution.
