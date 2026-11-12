# One-time cleanup authorized on 2026-10-14. Without -Apply, validate and show the plan.
param([switch]$Apply)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$workspace = Split-Path -Parent $projectRoot
$receipt = Join-Path $projectRoot 'artifacts/migration/root_cleanup_20261014'
$expected = @('atomic_evidence_router_v1_precodefix_20260912','hf_meta_llama2_tokenizer_audit',
    'literature_archive','pdfs','uhead_official','venvs','ghost_acl2026.txt','ikb_category_plan.json',
    'ikb_files_before.json','ikb_git_status_before.bin','ikb_inventory_before.json','ikb_move_plan.json',
    'ikb_moves_performed.json','indomain_pair_preview.jsonl','redeep_iclr2025.txt','SIRG_ACL2026_Findings.txt')
$expectedPdfs = @('bayestrace_read_da26bd589cd94d97b919efb8a9b9efba','literature_explanation',
    'ragognizer_2604.15945v1','CORTEX_2026_preprint.txt','First_Hallucination_Tokens_v4.txt','RAGognizer_2026_preprint.txt')

function Safe-Path([string]$Relative) {
    $absolute = [IO.Path]::GetFullPath((Join-Path $workspace $Relative))
    if (-not $absolute.StartsWith($workspace + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path escapes the repository: $absolute"
    }
    $cursor = $absolute
    while ($cursor -ne $workspace) {
        if ((Test-Path -LiteralPath $cursor) -and
            ((Get-Item -LiteralPath $cursor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            throw "Linked ancestor is not a move target: $cursor"
        }
        $cursor = Split-Path -Parent $cursor
    }
    return $absolute
}
function Remove-EmptyDirectory([string]$Relative) {
    $path = Safe-Path $Relative
    if (@(Get-ChildItem -LiteralPath $path -Force).Count) { throw "Directory is not empty: $path" }
    Remove-Item -LiteralPath $path -Force
}

$tmp = Safe-Path 'tmp'
if (Compare-Object @(Get-ChildItem -LiteralPath $tmp -Force | ForEach-Object Name) $expected) {
    throw 'Unexpected tmp entry. Recheck ownership before moving anything.'
}
if (Compare-Object @(Get-ChildItem -LiteralPath (Join-Path $tmp 'pdfs') -Force | ForEach-Object Name) $expectedPdfs) {
    throw 'Unexpected PDF reading-cache entry.'
}
$sources = @($expected | Where-Object { $_ -ne 'pdfs' } | ForEach-Object { "tmp/$_" })
$sources += @($expectedPdfs | ForEach-Object { "tmp/pdfs/$_" })
$masSources = @('tmp/literature_archive','tmp/pdfs/bayestrace_read_da26bd589cd94d97b919efb8a9b9efba','tmp/pdfs/literature_explanation')
$plan = @($sources | ForEach-Object {
    $owner = if ($_ -in $masSources) { 'paperAlpha/archive/local_scratch_20261014' } else { 'intelligence_knowledge_boundary/artifacts/legacy_tmp' }
    [pscustomobject]@{source=$_; destination="$owner/$($_.Substring(4))"}
})
$inventory = [Collections.Generic.List[object]]::new()
foreach ($entry in $plan) {
    $source = Safe-Path $entry.source
    $destination = Safe-Path $entry.destination
    if (Test-Path -LiteralPath $destination) { throw "Destination exists: $destination" }
    $item = Get-Item -LiteralPath $source -Force
    $children = if ($item.PSIsContainer) { @(Get-ChildItem -LiteralPath $source -Recurse -Force) } else { @($item) }
    if (@($children | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count) {
        throw "Linked descendant in $source"
    }
    foreach ($file in @($children | Where-Object { -not $_.PSIsContainer })) {
        if ($file.Name -match '^\.env($|\.)|^id_(rsa|ed25519)$') { throw 'Credential-like file requires a separate, non-content review.' }
        $suffix = if ($item.PSIsContainer) { [IO.Path]::GetRelativePath($source, $file.FullName) } else { '' }
        $target = if ($suffix) { Join-Path $destination $suffix } else { $destination }
        $inventory.Add([pscustomobject]@{
            source=[IO.Path]::GetRelativePath($workspace,$file.FullName).Replace('\','/')
            destination=[IO.Path]::GetRelativePath($workspace,$target).Replace('\','/')
            bytes=$file.Length
            sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        })
    }
}
$retiredLinks = @(
    @{alias='prelab';target='intelligence_knowledge_boundary/src/prelab'},
    @{alias='research/ragognizer_geometry_debug';target='intelligence_knowledge_boundary/reports/research/ragognizer_geometry_debug'},
    @{alias='research/recent_span_rag_datasets';target='intelligence_knowledge_boundary/reports/research/recent_span_rag_datasets'}
)
foreach ($link in $retiredLinks) {
    $path = [IO.Path]::GetFullPath((Join-Path $workspace $link.alias))
    $target = Safe-Path $link.target
    $item = Get-Item -LiteralPath $path -Force
    if ($item.LinkType -ne 'Junction' -or [IO.Path]::GetFullPath([string]@($item.Target)[0]) -ne $target) {
        throw "Unexpected link: $path"
    }
}
$outputItems = @(Get-ChildItem -LiteralPath (Safe-Path 'outputs') -Recurse -Force)
if (@($outputItems | Where-Object { -not $_.PSIsContainer -or ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) }).Count) {
    throw 'outputs is not an empty ordinary directory tree.'
}
$plan | Format-Table -AutoSize
if (-not $Apply) { return }
if (Test-Path -LiteralPath $receipt) { throw 'Refusing to overwrite the cleanup receipt.' }
New-Item -ItemType Directory -Path $receipt | Out-Null
foreach ($original in @('README.md','PAPER_HANDOFF.md','intelligence_knowledge_boundary/README.md',
    'intelligence_knowledge_boundary/configs/workspace_layout.json','intelligence_knowledge_boundary/scripts/run_experiment.ps1')) {
    $backup = Join-Path $receipt "originals/$original"
    New-Item -ItemType Directory -Path (Split-Path -Parent $backup) -Force | Out-Null
    Copy-Item -LiteralPath (Safe-Path $original) -Destination $backup
}
@{schema='root-cleanup-inventory-v1';plan=$plan;files=$inventory;retired_links=$retiredLinks} |
    ConvertTo-Json -Depth 7 | Set-Content -LiteralPath (Join-Path $receipt 'BEFORE.json') -Encoding utf8
foreach ($entry in $plan) {
    $source = Safe-Path $entry.source
    $destination = Safe-Path $entry.destination
    if (Test-Path -LiteralPath $destination) { throw "Destination appeared: $destination" }
    New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
    Move-Item -LiteralPath $source -Destination $destination
}
foreach ($file in $inventory) {
    $path = Safe-Path $file.destination
    if ((Get-Item -LiteralPath $path).Length -ne $file.bytes -or
        (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $file.sha256) {
        throw "Moved file changed: $path"
    }
}
Remove-EmptyDirectory 'tmp/pdfs'
Remove-EmptyDirectory 'tmp'
foreach ($link in $retiredLinks) {
    $path = [IO.Path]::GetFullPath((Join-Path $workspace $link.alias))
    $target = Safe-Path $link.target
    $item = Get-Item -LiteralPath $path -Force
    if ($item.LinkType -ne 'Junction' -or [IO.Path]::GetFullPath([string]@($item.Target)[0]) -ne $target) {
        throw "Link changed during cleanup: $path"
    }
    # No -Recurse: remove only the verified junction, never its contents.
    Remove-Item -LiteralPath $path -Force
    if (-not (Test-Path -LiteralPath $target -PathType Container)) { throw 'Junction target disappeared.' }
}
Remove-EmptyDirectory 'research'
foreach ($item in @($outputItems | Sort-Object { $_.FullName.Length } -Descending)) {
    Remove-EmptyDirectory ([IO.Path]::GetRelativePath($workspace, $item.FullName))
}
Remove-EmptyDirectory 'outputs'
$after = @{schema='root-cleanup-verification-v1';files_verified=$inventory.Count;all_sha256_equal=$true;
    removed_empty_directories=@('outputs','outputs/literature-review-2026-09-11','research','tmp','tmp/pdfs');
    retired_links=$retiredLinks;verified_at_utc=[DateTime]::UtcNow.ToString('o')}
$after | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $receipt 'AFTER.json') -Encoding utf8
$after | ConvertTo-Json -Depth 5
