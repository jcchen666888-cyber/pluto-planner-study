param(
    [string]$EnvironmentName = "pluto-study",
    [switch]$SkipEnvironment
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$depsRoot = Join-Path $repoRoot "_deps"
$plutoCommit = "b9964b649c660f1f4a971d614c66f5992e24c18a"
$nuplanCommit = "ce3c323af01c0d7ec5672f7832ef53f9c679aab0"

$driveName = [System.IO.Path]::GetPathRoot($repoRoot).Substring(0, 1)
$freeGiB = (Get-PSDrive -Name $driveName).Free / 1GB
if ($freeGiB -lt 50) {
    throw "50 GB guard failed: only $([math]::Round($freeGiB, 2)) GiB free."
}

New-Item -ItemType Directory -Force -Path $depsRoot | Out-Null

function Get-PinnedRepository([string]$Url, [string]$Path, [string]$Commit) {
    if (-not (Test-Path -LiteralPath (Join-Path $Path ".git"))) {
        git clone $Url $Path
    }
    git -C $Path fetch --depth 1 origin $Commit
    git -C $Path checkout --detach $Commit
    $actual = git -C $Path rev-parse HEAD
    if ($actual -ne $Commit) { throw "Commit mismatch at ${Path}: $actual" }
}

Get-PinnedRepository "https://github.com/jchengai/pluto.git" (Join-Path $depsRoot "pluto-upstream") $plutoCommit
Get-PinnedRepository "https://github.com/motional/nuplan-devkit.git" (Join-Path $depsRoot "nuplan-devkit") $nuplanCommit

if (-not $SkipEnvironment) {
    $knownEnvironments = conda env list --json | ConvertFrom-Json
    $exists = $knownEnvironments.envs | Where-Object { (Split-Path $_ -Leaf) -eq $EnvironmentName }
    if (-not $exists) {
        conda env create -n $EnvironmentName -f (Join-Path $repoRoot "environment.yml")
    }
    conda run -n $EnvironmentName python -m pip install --no-deps -e (Join-Path $depsRoot "nuplan-devkit")
}

Write-Host "Pinned source and environment are ready."
