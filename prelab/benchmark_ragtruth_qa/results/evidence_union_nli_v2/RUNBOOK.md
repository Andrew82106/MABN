# Evidence-union NLI v2 runbook

CPU work available now:

```powershell
& 'D:/Projects/Multi_Agent_Graph_Analysis/prelab/.venv/Scripts/python.exe' prelab/benchmark_ragtruth_qa/src/run_evidence_union_nli_v2.py prepare --cohort fit_native
& 'D:/Projects/Multi_Agent_Graph_Analysis/prelab/.venv/Scripts/python.exe' prelab/benchmark_ragtruth_qa/src/run_evidence_union_nli_v2.py prepare --cohort calibration
& 'D:/Projects/Multi_Agent_Graph_Analysis/prelab/.venv/Scripts/python.exe' prelab/benchmark_ragtruth_qa/src/run_evidence_union_nli_v2.py score --cohort fit_native
& 'D:/Projects/Multi_Agent_Graph_Analysis/prelab/.venv/Scripts/python.exe' prelab/benchmark_ragtruth_qa/src/run_evidence_union_nli_v2.py score --cohort calibration
```

After the separate 3,046-answer attribution extractor has written a complete
`semantic_source_attribution_expanded_fit_v1/feature_manifest.json`:

```powershell
& 'D:/Projects/Multi_Agent_Graph_Analysis/prelab/.venv/Scripts/python.exe' prelab/benchmark_ragtruth_qa/src/run_evidence_union_nli_v2.py prepare --cohort fit_expanded
& 'D:/Projects/Multi_Agent_Graph_Analysis/prelab/.venv/Scripts/python.exe' prelab/benchmark_ragtruth_qa/src/run_evidence_union_nli_v2.py gpu-smoke --cohort fit_expanded
& 'D:/Projects/Multi_Agent_Graph_Analysis/prelab/.venv/Scripts/python.exe' prelab/benchmark_ragtruth_qa/src/run_evidence_union_nli_v2.py extract --cohort fit_expanded
& 'D:/Projects/Multi_Agent_Graph_Analysis/prelab/.venv/Scripts/python.exe' prelab/benchmark_ragtruth_qa/src/run_evidence_union_nli_v2.py score --cohort fit_expanded
```

`extract` is resumable in immutable 4,096-request chunks.  Use
`--limit-chunks 1` for a bounded extraction invocation.  Neither `score` nor
any other command trains the final detector.
