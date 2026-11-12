# Forced-evidence quote probe CPU evaluator self-test

Status: **PASS (synthetic CPU only; no real scoring)**

- Evaluator: `src/evaluate_forced_evidence_quote_probe_v1.py`
- Evaluator SHA256: `5f879d0714f7d0566659f24dabe2f0308db52fd1944d3e380bc624f65788549e`
- Protocol SHA256: `78e5558b61592238b0979c7c60aea23c79b8c7d3181e88113d3972b4d056f7aa`
- Plan SHA256: `6d439ee257dde65d2ef2c9974cd3dd716f4f0d06890cfb7df9c8af6da562680c`
- Runtime: scikit-learn 1.6.1
- `py_compile`: PASS
- Synthetic P0/P1/P2/P3 nested CV: PASS (20 inner + 5 outer fits per condition)
- Fold-local answer weights, weighted scaler, class balance, fixed C, threshold rule, 4-BPE mapping, answer max, pooled AP/F1, pairwise fold deltas, and advance gate: PASS
- Missing frozen feature bundle: correctly blocked before fit gold (`fit_gold_open_count=0`)
- Real fit scoring: not run
- Calibration/test: not opened
- GPU: not used
- Existing baseline: not modified

Machine-readable evidence is in `CPU_EVALUATOR_SELFTEST.json` (SHA256 `812078f9ad705db3692fcc624b7936b0a32b410703c130be1a33f46c8971a167`).
