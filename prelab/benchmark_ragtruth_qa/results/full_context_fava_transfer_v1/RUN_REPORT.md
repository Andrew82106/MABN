# Completed FAVA auxiliary-to-QA transfer

Original session79363 (actual Python PID63224) exited0 after all7482 auxiliary answers once, followed by all3680 QA fit answers for three complete epochs. The optimizer remained continuous across phases:936 auxiliary updates plus3×460 QA updates,2316 total. No early stopping, precision change, truncation or sample removal occurred. Original preparation/STATUS artifacts remain unchanged historical records. Official QA test content remains sealed.

The fixed calibration rule selects QA epoch2.

| QA epoch |4BPE window F1| Answer F1|
|---|---:|---:|
|1|0.6552066|0.8585366|
|2 selected|0.6611029|0.8333333|
|3|0.6423686|0.8516746|

For reference, QA-only generic base-v2 with its full six-epoch selection reached0.6417962/0.8294931. The descriptive differences are+1.93 and+0.38 percentage points. These are repeatedly exposed development/calibration results, not an independent final test or an isolated equal-compute improvement. Localization remains below0.75.

The run used5853211 logical input tokens in the auxiliary phase and11160661 across all training forwards; QA evaluation used5572614 logical input tokens. Total time was1822.37seconds (30.4minutes), including QA evaluation and saving. Logical counts exclude checkpoint recomputation; wall time includes it. The auxiliary phase alone took659.52seconds. QA-only six epochs and FAVA auxiliary1→QA3 differ in additional supervision, answer lengths, update schedule and evaluation cost.

The real1481-token synthetic-zero-target smoke passed with repeated difference0, finite nonzero gradients, FP32 parameters/gradients/Adam states, and BF16 forward/checkpoint recomputation. Its allocated peak was3111667712bytes and complete step took0.310857seconds. The auxiliary training allocated peak was3116731904bytes. No quantization or fallback was introduced.

FAVA adds synthetic silver factual-error supervision; unmarked auxiliary text is not independently verified as true. The original five Reference sections and corrupted answer are used without correction markup; its empty question is an adaptation. Original QA input, labels, weights, first three orders and4BPE/answer-max evaluation remain frozen.

`transfer/complete.json` SHA256: `c418e510ff25a1643afb2b915c36cea6d625e39410eaeae559cb2bfc7d2df352`. `RUN_SUMMARY.json` records the exact completed phases, metrics, token counts and exit status. This is a run record; root separately owns the36 fixed-budget probability-combination results and overall experiment status.

The subsequently completed fixed-budget comparison (`../fava_fixed_convex_v1/REPORT.md`) retained all36 candidates and12 endpoint replays. Its strongest window result was original HARP LR plus FAVA weight0.2:0.6876955 window F1 and0.8704663 answer F1. This remained below the existing large semantic combination0.690281/0.891089; FAVA did not replace that control. These are calibration comparisons, not final-test improvements.
