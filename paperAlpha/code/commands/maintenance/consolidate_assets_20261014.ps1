# One-time, local asset relocation. Default is a read-only plan.
# Existing experiment results, credentials and unrelated projects are not targets.
param([switch]$Apply)

$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$workspaceRoot = Split-Path -Parent $projectRoot
$receiptRoot = Join-Path $projectRoot 'results/organization_20261014'
$plan = @(
    @{source='doc/ref_paper/mas_safety_2026-09-11'; destination='paperAlpha/literature/mas_safety_2026-09-11'; operation='move'},
    @{source='doc/ref_paper/my_work'; destination='paperAlpha/literature/my_work'; operation='move'},
    @{source='doc/长期'; destination='paperAlpha/archive/project_docs/长期'; operation='move'},
    @{source='doc/临时'; destination='paperAlpha/archive/project_docs/临时'; operation='move'},
    @{source='shared_lab'; destination='paperAlpha/archive/shared_lab'; operation='move'},
    @{source='tmp/AgentMonitor'; destination='paperAlpha/data/external/AgentMonitor'; operation='move'},
    @{source='tmp/agentmonitor-audit'; destination='paperAlpha/data/external/AgentMonitor_audit_snapshot'; operation='move'},
    @{source='tmp/TAMAS'; destination='paperAlpha/data/external/TAMAS'; operation='move'},
    @{source='tmp/MultiAgentHarmDataset'; destination='paperAlpha/data/external/MultiAgentHarmDataset'; operation='move'},
    @{source='doc/Doc工作指南.md'; destination='paperAlpha/archive/project_docs/Doc工作指南.md'; operation='copy'},
    @{source='doc/Python环境说明.md'; destination='paperAlpha/archive/project_docs/Python环境说明.md'; operation='copy'}
)

function Assert-ContainedPath([string]$Path, [string]$Root) {
    $absolute = [IO.Path]::GetFullPath($Path)
    $boundary = $Root.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    if (-not $absolute.StartsWith($boundary, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path escapes the intended directory: $absolute"
    }
    # Also reject a junction in any existing parent, including the source itself.
    $cursor = $absolute
    while ($cursor -and $cursor -ne $Root) {
        if (Test-Path -LiteralPath $cursor) {
            if ((Get-Item -LiteralPath $cursor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "Refusing to operate through a link: $cursor"
            }
        }
        $cursor = Split-Path -Parent $cursor
    }
    return $absolute
}

$inventory = @()
foreach ($item in $plan) {
    $source = Assert-ContainedPath (Join-Path $workspaceRoot $item.source) $workspaceRoot
    $destination = Assert-ContainedPath (Join-Path $workspaceRoot $item.destination) $projectRoot
    if (-not (Test-Path -LiteralPath $source)) { throw "Missing source: $source" }
    if (Test-Path -LiteralPath $destination) { throw "Destination already exists: $destination" }
    $entry = Get-Item -LiteralPath $source -Force
    $descendants = if ($entry.PSIsContainer) { @(Get-ChildItem -LiteralPath $source -Recurse -Force) } else { @($entry) }
    if (@($descendants | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count) {
        throw "Linked descendant in source: $source"
    }
    $files = @($descendants | Where-Object { -not $_.PSIsContainer })
    if (@($files | Where-Object { $_.Name -match '(^|\.)env($|\.)|\.(pem|key|p12|pfx)$' }).Count) {
        throw "Credential-like file found; review the target without reading or hashing it: $source"
    }
    foreach ($file in $files) {
        $relative = if ($entry.PSIsContainer) { [IO.Path]::GetRelativePath($source, $file.FullName) } else { '' }
        $targetFile = if ($relative) { Join-Path $destination $relative } else { $destination }
        $inventory += [pscustomobject]@{
            source = [IO.Path]::GetRelativePath($workspaceRoot, $file.FullName).Replace('\','/')
            destination = [IO.Path]::GetRelativePath($workspaceRoot, $targetFile).Replace('\','/')
            bytes = $file.Length
            sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }
    [pscustomobject]@{operation=$item.operation; source=$source; destination=$destination; files=$files.Count}
}
if (-not $Apply) { return }

$receiptRoot = Assert-ContainedPath $receiptRoot $projectRoot
if (Test-Path -LiteralPath $receiptRoot) { throw "Refusing to overwrite an existing receipt: $receiptRoot" }
New-Item -ItemType Directory -Path $receiptRoot | Out-Null
$before = [ordered]@{
    schema='paperalpha-asset-relocation-v1'
    date='2026-10-14'
    scope='Byte-preserving move/copy before separately reviewed navigation edits; not an experiment.'
    plan=$plan
    files=$inventory
}
# Generated audit artifacts, never credentials or hand-written source files.
$before | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $receiptRoot 'BEFORE.json') -Encoding utf8
foreach ($item in $plan) {
    $source = Assert-ContainedPath (Join-Path $workspaceRoot $item.source) $workspaceRoot
    $destination = Assert-ContainedPath (Join-Path $workspaceRoot $item.destination) $projectRoot
    if (Test-Path -LiteralPath $destination) { throw "Destination appeared during relocation: $destination" }
    New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
    if ($item.operation -eq 'move') {
        Move-Item -LiteralPath $source -Destination $destination
    } else {
        Copy-Item -LiteralPath $source -Destination $destination
    }
}
foreach ($file in $inventory) {
    $destination = Join-Path $workspaceRoot $file.destination
    $actual = Get-Item -LiteralPath $destination -Force
    $hash = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual.Length -ne $file.bytes -or $hash -ne $file.sha256) { throw "Relocation verification failed: $destination" }
}
$after = [ordered]@{
    schema='paperalpha-asset-relocation-verification-v1'
    verified_at_utc=[DateTime]::UtcNow.ToString('o')
    files_verified=$inventory.Count
    total_bytes=($inventory | Measure-Object bytes -Sum).Sum
    all_sha256_equal=$true
    note='Verified immediately after relocation. Subsequent documentation link edits are intentional.'
}
$after | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $receiptRoot 'AFTER.json') -Encoding utf8
$after
