# Human-correction minimal-pair conflict verifier v1

## Question

Can RAGTruth's human `Original:` correction notes support a claim-level
contradiction verifier without the style and domain shortcuts seen in the prior
tokenwise auxiliary transfer?

## Frozen scope and isolation

- The only data file is the already-quarantined
  `auxiliary_human_v1/candidate_fit.jsonl`.
- Only released human Evident Conflict and Subtle Conflict annotations are
  considered. Baseless labels never become negative or positive examples.
- Every answer, correction pair, and derived row retains the existing material
  `group_id`. A SHA256 assignment places an entire group in one of five folds;
  fold 0 is the internal held diagnostic and folds 1--4 are training.
- Calibration, QA fit, and official test files are absent from preparation and
  training code. Published baselines and prior runners are read-only and are
  never output destinations.

## Frozen high-precision transform

For each released EC/SC span:

1. Parse an exact line-leading `Original:` field and an exact line-leading
   `AIGC:`, `Generated:`, or `Generative:` field from its unchanged human
   metadata. No language model repairs or completes either field.
2. Split the original material into exact source sentences. The Unicode word
   sequence in `Original:` must occur contiguously in a source sentence. Its
   set-lexical source coverage is therefore 1.0 and exceeds the prespecified
   0.8 priority threshold. If the phrase occurs in distinct source sentences,
   exclude it as ambiguous; identical duplicate sentences use the earliest.
3. Reproduce the existing label-blind atomic microclaim transform on the model
   answer. The metadata's generated word sequence must occur contiguously and
   unambiguously inside one microclaim, and every matched word must lie inside
   the target human EC/SC character span.
4. The selected microclaim may touch no other released error span at an
   alphanumeric character. This ensures replacing the target does not silently
   leave a second annotated error in the negative example.
5. Replace only the matched generated word range with the exact-cased source
   phrase located by `Original:`. Outer microclaim context is byte-for-byte
   unchanged. Keep the pair only when it changes lexical content, the source
   phrase has at least two words, and the corrected microclaim's word-count
   change is at most `max(2, ceil(0.25 * bad_words))`. It must retain an
   unchanged context word, unless generated and corrected ranges have equal
   word count.

Every excluded span remains in `span_audit.jsonl` with machine-readable failure
reasons. No whole sentence or microclaim is converted into token labels.

## Frozen triples and labels

Each retained record contains one premise and three claim-level hypotheses:

- premise: exact question plus the exact matched source evidence sentence;
- bad: the unchanged atomic microclaim, target conflict;
- corrected: the same microclaim after the single traced replacement, target
  non-conflict;
- source: the exact evidence sentence itself, target non-conflict.

The corrected and bad hypotheses share the same premise and almost all surface
form. The source hypothesis anchors ordinary entailment. These are claim-level
labels only; token labels are never created.

## Frozen model and objective

- Backbone: local ModernBERT NLI model with explicit SDPA and
  `reference_compile=False`.
- Head: a separate scalar conflict head on the existing mean-pooled NLI
  representation. Initialize it from `C - mean(E,N)` weights; then train the
  backbone and this head jointly.
- Per triple classification loss:
  `(BCE(bad,1) + BCE(corrected,0) + 0.5*BCE(source,0)) / 2.5`.
- Fixed pair-ranking loss:
  `(softplus(-(bad-corrected)) + softplus(-(bad-source))) / 2`.
- Total loss is classification plus ranking. Material groups receive equal
  total weight and triples inside a group divide that weight equally.
- One training pass, AdamW 3e-6, weight decay 0.01, clip norm 1, no scheduler,
  seed 20261024, FP32 parameters/optimizer and BF16 CUDA forward. Batch at most
  eight triples under a 4096 padded-token budget and accumulate four
  microbatches. There is no search, early stopping, or epoch selection.

## Frozen diagnostics and later deployment

Internal fold 0 reports pair AP/F1 and the proportions satisfying
`bad > corrected` and `bad > source`, both before and after the single training
pass. Held labels do not select a threshold or checkpoint.

Later QA deployment scores each existing atomic claim once with its question
and retrieved evidence. Its scalar claim score is copied to the existing
eligible 4-BPE windows that overlap that claim; overlapping claims use maximum.
No tokenwise training or tokenwise maximum is reintroduced. That QA mapping is
outside this CPU data-preparation stage.
