# Llama2 QA replay selfcheck

Passed on the first and last frozen development plans: fit `16023` (306 answer tokens) and calibration `15975` (239). No test/withheld content or annotation values were used.

The verified checkpoint is `NousResearch/Llama-2-7b-chat-hf`, revision `351844e75ed0bcbbe3f10671b3c808d2b83894ee`. The RTX 3070 run used NF4 with double quantization, uint8 weight storage, BF16 computation/embedding/output head, SDPA, no TF32 and four CPU threads. The observed linear module was `Linear4bit`; the architecture was 32 layers × 32 heads, 32 KV heads, hidden width 4096. Asset, tokenizer, template/token plans, installed-library and runner hashes are frozen in `data/replay_selfcheck/signature.json`.

Both samples passed exact repeated extraction of every base array. Independent official-library attention rows at beginning/middle/end, causal-LM forward NLL for the first 16 answer tokens, and a final-norm hook all matched the extractor with **maximum absolute difference 0**. The oracle captures the official FP32 softmax before its BF16 cast, matching the declared feature precision. The cast itself changes the ratio by up to 0.001023; this is recorded separately and is not treated as extractor error. Each sample also passed an exact repeated hidden-only no-context replay for future interface feasibility; full-dataset delta extraction is not part of the current base run.

The first reconstructed token overlaps the wrapper separator and answer: raw offset `[-1,5]`, evaluation offset `[0,5]` in both samples. It is retained. The full 793-plan CPU check already established no lost non-whitespace answer characters and preserved original/no-context answer IDs and clipped offsets. The released prompt, including its original refusal instruction, remains intact inside the explicit `<s>[INST] ... [/INST] ` wrapper.

Model load took 8.65 seconds, complete verification plus selfchecks 29.43 seconds; peak allocated GPU memory during base extraction was 3.741 GiB. Two-sample estimates suggest 9–18 minutes of base feature work for 793 answers / 213,159 tokens; uncompressed base arrays require 4,374,875,316 bytes. This interval is a planning range, not a confidence interval; final measured costs will be in `data/feature_manifest.json`.

**Scope:** this verifies a disclosed quantized, teacher-forced reconstruction of published text. It cannot recover the original unpublished generation tokenization, precision, decoding state or historical trace. Causal post-token hidden/LB features are distinct from pre-token NLL; no future-token smoothing is used.

Machine-readable evidence: `data/replay_selfcheck/manifest.json`, the two per-answer JSON/NPZ files and their independent-oracle NPZ files. The authorized 793-row base runner is `src/run_feature_qa_all.py`; it freezes its own signature, checkpoints verified rows and sets `complete=true` only after full read-back audit.
