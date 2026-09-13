# GPU failure history

## V1 smoke attempt — retained failure

- Date: 2026-09-12 (Asia/Shanghai)
- Entrypoint: `src/run_auxiliary_conflict_token_pilot_v1.py gpu-smoke`
- Result: process exit code 1 during ModernBERT initialization, before a
  successful smoke result could be committed.
- Failure mechanism: the loaded ModernBERT configuration entered its
  `reference_compile` path and attempted to use Triton instead of the intended
  SDPA implementation.
- Preserved state: the v1 runner is unchanged; no v1 `GPU_SMOKE.json`, arm
  checkpoint, prediction file, or summary exists.

The console traceback was not written to an artifact by the v1 runner, so this
history records the observed exit and failure mechanism without inventing a
traceback.

## V2 recovery boundary

V2 changes only runtime backend selection and output isolation:

1. `from_pretrained` receives `attn_implementation="sdpa"` and
   `reference_compile=False`, followed by assertions on both configuration
   values.
2. Every v2 CPU/GPU artifact is written below `gpu_runs_v2`.
3. Finalization additionally reports the already-prespecified answer score as
   the maximum of its window scores.

Data, labels, initialization checkpoint, model parameters, tokenwise loss,
seed, batches, optimizer, training streams, token/window readout, and pilot gate
remain unchanged.
