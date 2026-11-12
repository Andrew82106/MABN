param(
    [Parameter(Mandatory=$true)][string]$RunId,
    [string]$Python = 'D:\anaconda\envs\multi_agent_graph\python.exe'
)
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
if ($RunId -notmatch '^[a-zA-Z0-9][a-zA-Z0-9_-]*$') { throw 'Use a simple unique run ID' }
$out = Join-Path $root ('experiments\synthetic_mas\runs\'+$RunId)
if (Test-Path -LiteralPath $out) { throw "Run already exists; refusing to overwrite: $out" }
& $Python -B (Join-Path $PSScriptRoot 'generate_independent_mas_benchmark_v3.py') --n 4000 --seed 20260917 --out (Join-Path $out 'data')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python -B (Join-Path $PSScriptRoot 'evaluate_independent_mas_journal_v1.py') --input (Join-Path $out 'data\traces_public.jsonl') --labels (Join-Path $out 'data\labels.jsonl') --out (Join-Path $out 'evaluation') --folds 5 --seed 20261002
exit $LASTEXITCODE
