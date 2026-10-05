$ErrorActionPreference = 'Stop'

# Rebuild the public synthetic MAS benchmark and run the single canonical OOF
# evaluator. Generated data/results stay under paperAlpha/results (ignored by
# Git); the tracked submission docs contain the frozen headline table.
$root = Split-Path -Parent $PSScriptRoot
python "$root/scripts/generate_independent_mas_benchmark_v3.py" `
  --n 4000 --seed 20260917 --out "$root/results/independent_mas_v3"
python "$root/scripts/evaluate_independent_mas_journal_v1.py" `
  --input "$root/results/independent_mas_v3/traces_public.jsonl" `
  --labels "$root/results/independent_mas_v3/labels.jsonl" `
  --out "$root/results/independent_mas_journal_v1" `
  --folds 5 --seed 20261002
