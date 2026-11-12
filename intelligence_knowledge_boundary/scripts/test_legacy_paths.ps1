param([switch]$CheckActualManifest)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'with_legacy_paths.ps1')
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$checkRoot = Join-Path $projectRoot ('artifacts/migration/root_cleanup_20261014/checks/' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $checkRoot | Out-Null

function New-LayoutFixture([string]$Name) {
    $root = Join-Path $checkRoot "$Name/intelligence_knowledge_boundary"
    New-Item -ItemType Directory -Path (Join-Path $root 'configs') -Force | Out-Null
    $aliases = @(
        @{alias='prelab';target='intelligence_knowledge_boundary/src/prelab'},
        @{alias='research';target='intelligence_knowledge_boundary/reports/research'},
        @{alias='tmp';target='intelligence_knowledge_boundary/artifacts/legacy_tmp'}
    )
    foreach ($entry in $aliases) {
        $target = Join-Path (Split-Path -Parent $root) $entry.target
        New-Item -ItemType Directory -Path $target -Force | Out-Null
        New-Item -ItemType File -Path (Join-Path $target 'preserved.sentinel') | Out-Null
    }
    @{legacy_runtime_aliases=$aliases} | ConvertTo-Json -Depth 4 |
        Set-Content -LiteralPath (Join-Path $root 'configs/workspace_layout.json') -Encoding utf8
    return $root
}
function Assert-NoRuntimeAliases([string]$Root) {
    $parent = Split-Path -Parent $Root
    foreach ($name in @('prelab','research','tmp')) {
        if (Get-Item -LiteralPath (Join-Path $parent $name) -Force -ErrorAction SilentlyContinue) {
            throw "Runtime alias was not removed: $name"
        }
    }
    foreach ($relative in @('src/prelab','reports/research','artifacts/legacy_tmp')) {
        if (-not (Test-Path -LiteralPath (Join-Path $Root "$relative/preserved.sentinel") -PathType Leaf)) {
            throw 'A real target file was removed.'
        }
    }
}

$normal = New-LayoutFixture 'normal'
Invoke-IKBLegacyPaths -ProjectRoot $normal -Action {
    foreach ($name in @('prelab','research','tmp')) {
        if (-not (Test-Path -LiteralPath (Join-Path (Split-Path -Parent $normal) "$name/preserved.sentinel"))) {
            throw "Legacy path not available: $name"
        }
    }
}
Assert-NoRuntimeAliases $normal

$failure = New-LayoutFixture 'failure'
$caught = $false
try { Invoke-IKBLegacyPaths -ProjectRoot $failure -Action { throw 'expected fixture failure' } }
catch { if ($_.Exception.Message -notmatch 'expected fixture failure') { throw }; $caught=$true }
if (-not $caught) { throw 'Action failure did not propagate.' }
Assert-NoRuntimeAliases $failure

$borrowed = New-LayoutFixture 'borrowed'
$borrowedLink = Join-Path (Split-Path -Parent $borrowed) 'prelab'
New-Item -ItemType Junction -Path $borrowedLink -Target (Join-Path $borrowed 'src/prelab') | Out-Null
Invoke-IKBLegacyPaths -ProjectRoot $borrowed -Action { }
if ((Get-Item -LiteralPath $borrowedLink -Force).LinkType -ne 'Junction') { throw 'Borrowed link was removed.' }
Remove-Item -LiteralPath $borrowedLink -Force
Assert-NoRuntimeAliases $borrowed

$conflict = New-LayoutFixture 'conflict'
$conflictPath = Join-Path (Split-Path -Parent $conflict) 'tmp'
New-Item -ItemType Directory -Path $conflictPath | Out-Null
$caught = $false
try { Invoke-IKBLegacyPaths -ProjectRoot $conflict -Action { throw 'Action must not run after conflict.' } }
catch { if ($_.Exception.Message -notmatch 'nonmatching path') { throw }; $caught=$true }
if (-not $caught -or (Get-Item -LiteralPath $conflictPath).LinkType) { throw 'Conflict was not preserved.' }
foreach ($name in @('prelab','research')) {
    if (Test-Path -LiteralPath (Join-Path (Split-Path -Parent $conflict) $name)) { throw 'Conflict left a partial alias set.' }
}
Write-Output 'PASS: normal exit, exception cleanup, borrowed-link preservation, conflict refusal; target files preserved.'

if ($CheckActualManifest) {
    $manifestPath = Join-Path $projectRoot 'data/benchmark_ragtruth_qa/gold_manifest.json'
    $before = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash
    $state = @{checked=0}
    Invoke-IKBLegacyPaths -ProjectRoot $projectRoot -Action {
        $gold = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
        foreach ($property in $gold.source_paths.psobject.Properties) {
            if (-not (Test-Path -LiteralPath $property.Value -PathType Leaf)) { throw "Frozen source unavailable: $($property.Name)" }
            $state.checked++
        }
    }
    if ((Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash -ne $before) { throw 'Frozen manifest changed.' }
    foreach ($name in @('prelab','research','tmp')) {
        if (Get-Item -LiteralPath (Join-Path (Split-Path -Parent $projectRoot) $name) -Force -ErrorAction SilentlyContinue) {
            throw 'Actual repository root still has a runtime alias.'
        }
    }
    Write-Output "PASS: $($state.checked) frozen source paths resolve inside the legacy wrapper; manifest unchanged and root cleaned."
}
