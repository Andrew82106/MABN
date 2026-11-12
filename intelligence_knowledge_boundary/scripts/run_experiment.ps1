param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$ExperimentScript,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ScriptArguments
)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$repositoryRoot = Split-Path -Parent $projectRoot
$sourceRoot = Join-Path $projectRoot 'src\prelab'
$scriptPath = [IO.Path]::GetFullPath((Join-Path $sourceRoot $ExperimentScript))
if (-not $scriptPath.StartsWith($sourceRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'The experiment script must be inside this project source directory.'
}
$pythonPath = Join-Path $projectRoot 'artifacts\environments\prelab\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) { throw "Python environment missing: $pythonPath" }
if (-not (Test-Path -LiteralPath $scriptPath -PathType Leaf)) { throw "Experiment script missing: $scriptPath" }
. (Join-Path $PSScriptRoot 'with_legacy_paths.ps1')
$execution = @{exitCode=1}
Invoke-IKBLegacyPaths -ProjectRoot $projectRoot -Action {
    Push-Location -LiteralPath $repositoryRoot
    try {
        & $pythonPath $scriptPath @ScriptArguments
        $execution.exitCode = $LASTEXITCODE
    } finally {
        Pop-Location
    }
}
exit $execution.exitCode
