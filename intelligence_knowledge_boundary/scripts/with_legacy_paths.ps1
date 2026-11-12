# Preserve frozen absolute paths without keeping permanent aliases at repo root.
# The caller must wait for its child process; do not start detached jobs in Action.
function Invoke-IKBLegacyPaths {
    param(
        [Parameter(Mandatory=$true)][string]$ProjectRoot,
        [Parameter(Mandatory=$true)][scriptblock]$Action
    )
    $ErrorActionPreference = 'Stop'
    $ProjectRoot = [IO.Path]::GetFullPath($ProjectRoot)
    $repositoryRoot = Split-Path -Parent $ProjectRoot
    if ((Split-Path -Leaf $ProjectRoot) -ne 'intelligence_knowledge_boundary') {
        throw 'Expected the intelligence_knowledge_boundary project directory.'
    }
    $layout = Get-Content -LiteralPath (Join-Path $ProjectRoot 'configs/workspace_layout.json') -Raw | ConvertFrom-Json
    $aliases = @($layout.legacy_runtime_aliases)
    if ($aliases.Count -ne 3 -or
        (Compare-Object @($aliases | ForEach-Object alias | Sort-Object) @('prelab','research','tmp'))) {
        throw 'Unexpected legacy runtime alias set.'
    }
    $digest = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData(
        [Text.Encoding]::UTF8.GetBytes($repositoryRoot.ToLowerInvariant())))
    $mutex = [Threading.Mutex]::new($false, "Local\IKBLegacyPaths_$($digest.Substring(0,24))")
    $acquired = $false
    $owned = [Collections.Generic.List[object]]::new()
    try {
        try { $acquired = $mutex.WaitOne(0) }
        catch [Threading.AbandonedMutexException] { $acquired = $true }
        if (-not $acquired) { throw 'Another legacy-path run is active in this repository. Retry after it exits.' }
        $resolved = @($aliases | ForEach-Object {
            $aliasPath = [IO.Path]::GetFullPath((Join-Path $repositoryRoot $_.alias))
            $targetPath = [IO.Path]::GetFullPath((Join-Path $repositoryRoot $_.target))
            if (-not $aliasPath.StartsWith($repositoryRoot + '\', [StringComparison]::OrdinalIgnoreCase) -or
                -not $targetPath.StartsWith($ProjectRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
                throw 'A runtime alias escapes its declared project.'
            }
            if (-not (Test-Path -LiteralPath $targetPath -PathType Container)) { throw "Missing legacy target: $targetPath" }
            $existing = Get-Item -LiteralPath $aliasPath -Force -ErrorAction SilentlyContinue
            if ($existing -and ($existing.LinkType -ne 'Junction' -or
                [IO.Path]::GetFullPath([string]@($existing.Target)[0]) -ne $targetPath)) {
                throw "Will not replace an existing nonmatching path: $aliasPath"
            }
            [pscustomobject]@{alias=$aliasPath;target=$targetPath;exists=[bool]$existing}
        })
        foreach ($entry in $resolved) {
            if (-not $entry.exists) {
                New-Item -ItemType Junction -Path $entry.alias -Target $entry.target | Out-Null
                $owned.Add($entry)
            }
        }
        & $Action
    } finally {
        try {
            for ($index=$owned.Count-1; $index -ge 0; $index--) {
                $entry = $owned[$index]
                $actual = Get-Item -LiteralPath $entry.alias -Force -ErrorAction SilentlyContinue
                if (-not $actual) { continue }
                if ($actual.LinkType -eq 'Junction' -and
                    [IO.Path]::GetFullPath([string]@($actual.Target)[0]) -eq $entry.target) {
                    # Remove this invocation's junction only; never recurse into its target.
                    Remove-Item -LiteralPath $entry.alias -Force
                } else {
                    Write-Warning "Runtime alias was replaced; left untouched: $($entry.alias)"
                }
            }
        } finally {
            if ($acquired) { $mutex.ReleaseMutex() }
            $mutex.Dispose()
        }
    }
}
