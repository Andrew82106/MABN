param([switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$repositoryRoot = Split-Path -Parent $projectRoot
$manifestPath = Join-Path $projectRoot 'configs\workspace_layout.json'
$layout = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$checked = 0
$missing = [Collections.Generic.List[string]]::new()
foreach ($entry in $layout.directory_aliases) {
    $aliasPath = [IO.Path]::GetFullPath((Join-Path $repositoryRoot $entry.alias))
    $targetPath = [IO.Path]::GetFullPath((Join-Path $repositoryRoot $entry.target))
    foreach ($path in @($aliasPath, $targetPath)) {
        if (-not $path.StartsWith($repositoryRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
            throw "Layout entry is outside this repository: $path"
        }
    }
    if (Test-Path -LiteralPath $aliasPath) {
        $item = Get-Item -LiteralPath $aliasPath -Force
        if ($item.LinkType -ne 'Junction') { throw "Expected a directory junction: $aliasPath" }
        $actual = [IO.Path]::GetFullPath([string]@($item.Target)[0])
        if (-not $actual.Equals($targetPath, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Unexpected junction target: $aliasPath -> $actual"
        }
        $checked++
        continue
    }
    if (-not (Test-Path -LiteralPath $targetPath -PathType Container)) {
        $missing.Add($entry.target)
        continue
    }
    if ($CheckOnly) { $missing.Add($entry.alias); continue }
    $null = New-Item -ItemType Directory -Path (Split-Path -Parent $aliasPath) -Force
    $null = New-Item -ItemType Junction -Path $aliasPath -Target $targetPath
    $checked++
}
Write-Output "Valid directory links: $checked / $($layout.directory_aliases.Count)"
if ($missing.Count) {
    $missing | ForEach-Object { Write-Output "Not available locally: $_" }
    if ($CheckOnly) { exit 1 }
}
