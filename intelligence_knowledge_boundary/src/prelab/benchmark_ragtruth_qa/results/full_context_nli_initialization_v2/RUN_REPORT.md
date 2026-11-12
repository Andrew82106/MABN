# Completed NLI encoder initialization comparison

Session97326 (actual Python PID22116) exited0 after the complete fixed six-epoch schedule. All epochs0..6 retain model/optimizer/RNG state, predictions and metrics; epoch0 was excluded from selection. Original preparation/STATUS files remain historical records and were not overwritten. Official QA test content remains sealed.

| QA-only matched schedule | Selected epoch |4BPE window F1| Answer F1|
|---|---:|---:|---:|
| Generic ModernBERT-base v2 |3|0.6417962|0.8294931|
| NLI encoder initialization v2 |5|0.6386584|0.8472906|

The selected NLI model is0.31percentage points lower for localization and1.78points higher for answer classification. It does not improve localization over the generic base control. These are repeatedly used calibration results; neither an independent-test improvement nor upstream-data independence is claimed.

| Epoch | Window F1 | Answer F1 |
|---|---:|---:|
|0 diagnostic|0.2486968|0.7795276|
|1|0.6094164|0.8341709|
|2|0.6072742|0.8235294|
|3|0.6155333|0.8416290|
|4|0.6165942|0.8407080|
|5 selected|0.6386584|0.8472906|
|6|0.6335011|0.8545455|

Only the134 encoder tensors come from the pinned NLI initialization. The generic prediction transform and identically seeded binary classifier are preserved; all parameters train. All3839 inputs, loss weights and six orders are byte-identical to base-v2. FP32 parameters/gradients/Adam state, BF16 forward and checkpoint recomputation, original optimizer, learning rate, batch and selection rule remain unchanged.

The real979-token zero-target GPU smoke passed with repeated difference0 and allocated peak3108717056bytes. Its untrained FP32/BF16 logit difference0.2658389 is descriptive, not an equality claim. Full training plus per-epoch evaluation and saving took2440.48seconds (40.7minutes), with allocated peak3116843008bytes (3.12GB).

`full_finetune/complete.json` SHA256: `d51a552f878f7bb9077ab2ce3b7ec6f26effe104ff8568a300a5f7c8971f8514`. `INDEPENDENT_ENTRY_CHECK.json` records the bounded prelaunch CPU/static check; `RUN_SUMMARY.json` gives exact counts, metrics and hashes. This is a completed run record, not a new post-training numerical audit.
