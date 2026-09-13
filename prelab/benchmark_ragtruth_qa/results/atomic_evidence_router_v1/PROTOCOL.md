# Atomic Evidence Router v1

This is a CPU-only development candidate. It pools the existing HARP/NLL/GHOST/LUMINA token caches into the frozen atomic microclaims, then waits for the 315-dimensional raw relation/evidence output from `atomic_microclaim_nli_v1`. Missing upstream output produces `WAIT_UPSTREAM`; no surrogate feature or score is fabricated.

Fit uses five source-group folds. The fixed K=4 router and K=1 ablation share architecture, loss, epochs and preprocessing. The NLL tail cutoff and robust scalers are fitted inside each outer fold. Four-BPE windows and answer scores are max projections from microclaim risk. Calibration is read only after the frozen fit-OOF gate passes; it never selects structure, parameters, epoch or threshold. Official test and formal baseline code are outside this runner.

Exact commands from repository root:

```powershell
prelab\.venv\Scripts\python.exe prelab\benchmark_ragtruth_qa\src\run_atomic_evidence_router_v1.py status
prelab\.venv\Scripts\python.exe prelab\benchmark_ragtruth_qa\src\run_atomic_evidence_router_v1.py train
prelab\.venv\Scripts\python.exe prelab\benchmark_ragtruth_qa\src\run_atomic_evidence_router_v1.py verify
```
