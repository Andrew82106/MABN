# Frozen pair NLI stress v1 runbook

The CPU preparation is complete before GPU admission. Run these commands from
`prelab/benchmark_ragtruth_qa` only after the shared GPU is free:

```powershell
& '..\.venv\Scripts\python.exe' -X utf8 src/run_semantic_conflict_pair_nli_stress_v1.py gpu-smoke
& '..\.venv\Scripts\python.exe' -X utf8 src/run_semantic_conflict_pair_nli_stress_v1.py extract
& '..\.venv\Scripts\python.exe' -X utf8 src/run_semantic_conflict_pair_nli_stress_v1.py score
& '..\.venv\Scripts\python.exe' -X utf8 src/run_semantic_conflict_pair_nli_stress_v1.py status
```

`gpu-smoke` does not chain into extraction. The score stage is CPU-only and
reports the full strict silver set before the small qualitative-QC strata.
Silver endpoint roles are synthetic and are not human gold.
