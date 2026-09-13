# Generic ModernBERT-large QA control

Preparation, the real longest-input resource step, and all six training epochs completed. Session45966 (actual Python PID24180) exited0; all epochs0..6 were saved. Official QA test content remains sealed.

## Completed development result

The unchanged calibration rule selects epoch3. These are repeatedly used development/calibration results, not an independent final test.

| Same QA-only budget | Selected epoch |4BPE window F1| Answer F1|
|---|---:|---:|---:|
| Generic ModernBERT-base v2 |3|0.6417962|0.8294931|
| Generic ModernBERT-large v1 |3|0.6727819|0.8640777|

The descriptive increases are3.10 and3.46 percentage points. Localization remains below0.75. The generic model size and `foreach=False` implementation differ together; this is not an isolated optimizer or precision ablation, and no statistical or independent-test improvement is claimed.

| Epoch | Window F1 | Answer F1 |
|---|---:|---:|
|0 diagnostic only|0.2482646|0.7722008|
|1|0.6438858|0.8405797|
|2|0.6651956|0.8623853|
|3 selected|0.6727819|0.8640777|
|4|0.6522471|0.8504673|
|5|0.6589295|0.8465116|
|6|0.6687074|0.8469388|

The full run took5213.65seconds (86.9minutes, including all diagnostic/per-epoch inference and saving); base-v2 took2557.69seconds (42.6minutes). Large peak CUDA allocation over the complete run was7212389888bytes (7.21GB), versus base3116843008bytes (3.12GB). Training updated all parameters with FP32 gradients and Adam state, BF16 forward and checkpoint recomputation, and no precision fallback. `full_finetune/complete.json` SHA256: `ec4af1968ba7ab25c07563ada6757db084c58232a7630d0f0b01357ef1340290`.

- Generic checkpoint: `answerdotai/ModernBERT-large`, revision `45bb4654a4d5aaff24dd11d4781fa46d39bf8c13`. Download session91020 exited0. Official safetensors1583544840bytes, SHA256 `44510fec5d3a81a1877f225637b869495f18e55f6f23a09abb9be0acc030295f`; source LFS hash matched. No task-finetuned checkpoint.
- Runner: `../../src/run_full_context_encoder_large.py`; directly reuses frozen base-v2 geometry, loss, epochs and evaluation. Only training differences are the larger checkpoint/configuration and `AdamW(foreach=False)`.
- CPU prepare92734 and cpu-test12207 exited0. All3839 input records are byte-identical to base-v2, all loss arrays and six epoch orders exact. Same3680fit/159cal, seed20261005, six epochs, microbatch1/accumulation8, lr1e-5, wd.01, same calibration selection and original4BPE evaluation. Independent static review: `../../research/modernbert_large_resource_check/LARGE_RUNNER_REVIEW.json`.
- GPU resource session62807 exited0 and released the GPU. Longest979-token input15303 used only synthetic zero targets, not human gold. Full backward, clipping and optimizer step succeeded. The smoke-updated model was deleted; any later training reloads generic initial weights.

| Actual resource check | Result |
|---|---:|
| Trainable classifier parameters |395833346|
| Peak allocated bytes |6769905152 (6.77GB)|
| Peak reserved bytes |7235174400 (7.24GB)|
| RTX3070 total bytes |8589410304|
| Training-step seconds |0.5739701|
| Base-v2 same-step seconds |0.2836039|
| Repeated inference max difference |0|

Parameters, accumulated gradients and floating Adam states are FP32; forward and non-reentrant layer recomputation use BF16. Character mapping and loss are FP32. No quantization, precision fallback or budget change was used. A single resource step establishes feasibility of that step, not full-training peak or throughput.

The diagnostic untrained FP32-vs-BF16 maximum mapped-logit difference was2.1671393. It is recorded descriptively; the frozen repeatability gate compares the same BF16 path (difference0), not two different precisions. Neither numerical equivalence nor improved localization is claimed.

`GPU_SELFCHECK.json`, `GPU_RESOURCE_DETAIL.json`, `CPU_SELFCHECK.json` and `preparation_complete.json` retain the exact checks and hashes. Legacy `V1_REUSE_AGREEMENT`/`v1_reference` names refer to base-v2 here, as disclosed by `protocol.json`.
