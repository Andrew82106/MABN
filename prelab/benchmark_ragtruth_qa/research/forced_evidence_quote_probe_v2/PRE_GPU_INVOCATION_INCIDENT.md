# V2 pre-GPU invocation incident

On 2026-09-12, the first `gpu-smoke` command was invoked with the base
`D:\anaconda\python.exe` interpreter instead of the frozen CA environment.
It stopped inside `check_cpu_ready()` while reading package metadata because
that interpreter has no `bitsandbytes` installation.

- Failure receipt:
  `results/forced_evidence_quote_probe_v2/GPU_RUNNER_FAILURE_gpu-smoke_1789222299176746400.json`
- Exception: `PackageNotFoundError('bitsandbytes')`
- Runner SHA-256: `eb4a51353ba4edaff2cff877f576ff206f1fca1d6bd820928fb0db4acebfcca2`
- `CUDA_initialized`: `false`
- `GPU_SMOKE.json`: absent
- Smoke claims executed: `0 / 8`
- Model loaded, features extracted, labels opened, or scoring run: no

This was a rejected preflight invocation, not a failed numerical smoke gate.
The frozen environment recorded by the CPU selfcheck is Python 3.11.15 with
PyTorch 2.5.1+cu121, Transformers 5.9.0, bitsandbytes 0.49.2, NumPy 2.4.6,
and scikit-learn 1.8.0.  The matching interpreter is
`D:\anaconda\envs\CA\python.exe`; its `describe` stage reproduced the frozen
runner and input identity without initializing CUDA.  The failure receipt is
retained and no protocol, runner, review, sample, or tolerance was changed.
