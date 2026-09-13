# Reviewed execution commands

Run from `D:\Projects\Multi_Agent_Graph_Analysis` only after the current GPU
owner has exited.  Do not combine the two commands; require the first process to
exit 0 and review `GPU_SMOKE.json` before full extraction.

```powershell
$py = 'prelab/.venv/Scripts/python.exe'
$runner = 'prelab/benchmark_ragtruth_qa/src/run_exact_subset_attribution_v2.py'
& $py $runner gpu-smoke
```

After smoke exits 0 and its new artifact is reviewed:

```powershell
$py = 'prelab/.venv/Scripts/python.exe'
$runner = 'prelab/benchmark_ragtruth_qa/src/run_exact_subset_attribution_v2.py'
& $py $runner extract
```

The gate itself refuses a project lock conflict, numeric foreign GPU memory,
pure-compute or ambiguous process entries, unreviewed GUI executables, and less
than 6 GiB free memory.  Both commands write only
`results/exact_subset_attribution_v2`.
