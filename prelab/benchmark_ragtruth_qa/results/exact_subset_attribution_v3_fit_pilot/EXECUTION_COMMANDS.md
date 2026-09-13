# Reviewed execution sequence

Run from `D:\Projects\Multi_Agent_Graph_Analysis`.  Each command must exit 0
and its output must be reviewed before starting the next command.

```powershell
$py = 'prelab/.venv/Scripts/python.exe'
$runner = 'prelab/benchmark_ragtruth_qa/src/run_exact_subset_attribution_v3_fit_pilot.py'
& $py $runner gpu-smoke
```

After reviewing the new `GPU_SMOKE.json`:

```powershell
$py = 'prelab/.venv/Scripts/python.exe'
$runner = 'prelab/benchmark_ragtruth_qa/src/run_exact_subset_attribution_v3_fit_pilot.py'
& $py $runner extract
```

After all 256 caches and `extraction_complete.json` are frozen and reviewed:

```powershell
$py = 'prelab/.venv/Scripts/python.exe'
$runner = 'prelab/benchmark_ragtruth_qa/src/run_exact_subset_attribution_v3_fit_pilot.py'
& $py $runner score
```

`score` is CPU-only.  It is the first stage allowed to open the three expanded
fit gold files.  It never reads calibration or official-test files.
