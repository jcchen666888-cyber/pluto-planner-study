param([string]$EnvironmentName = "pluto-study")

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$downloadRoot = Join-Path $repoRoot "downloads"
$dataRoot = Join-Path $repoRoot "data"
$checkpointRoot = Join-Path $repoRoot "checkpoints"

$driveName = [System.IO.Path]::GetPathRoot($repoRoot).Substring(0, 1)
$freeGiB = (Get-PSDrive -Name $driveName).Free / 1GB
if ($freeGiB -lt 50) {
    throw "50 GB guard failed: only $([math]::Round($freeGiB, 2)) GiB free."
}

New-Item -ItemType Directory -Force -Path $downloadRoot, $dataRoot, $checkpointRoot | Out-Null
$mapsZip = Join-Path $downloadRoot "nuplan-maps-v1.0.zip"
$miniZip = Join-Path $downloadRoot "nuplan-v1.1_mini.zip"

curl.exe -L -C - --fail --retry 5 -o $mapsZip "https://d1qinkmu0ju04f.cloudfront.net/public/nuplan-v1.1/nuplan-maps-v1.0.zip"
curl.exe -L -C - --fail --retry 5 -o $miniZip "https://d1qinkmu0ju04f.cloudfront.net/public/nuplan-v1.1/nuplan-v1.1_mini.zip"

$expected = @{}
$expected[$mapsZip] = "D0310009FA9E8DD88014038336538ACA678842C009FBF03FAE76ED28F702FFC6"
$expected[$miniZip] = "A3FE40AFD81CC634884F8D0B7EA3604F2E617E365D5C258C61CFDD833C8D987B"
foreach ($path in $expected.Keys) {
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash
    if ($actual -ne $expected[$path]) { throw "SHA-256 mismatch: $path" }
}

tar -xf $mapsZip -C $dataRoot
tar -xf $miniZip -C $dataRoot

$mirrorDownload = Join-Path $downloadRoot "pluto_1M_aux_cil.ckpt"
conda run -n $EnvironmentName python -m gdown "1rI_we4zkk3Jk6wr7IzF-nMBnRMJsuOGT" -O $mirrorDownload
$checkpointHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $mirrorDownload).Hash
if ($checkpointHash -ne "CE60D7C854D10B310C4E1705099D799A6EA008BAA4288E668626F4A2F22943E1") {
    throw "Checkpoint SHA-256 mismatch."
}
Copy-Item -LiteralPath $mirrorDownload -Destination (Join-Path $checkpointRoot "pluto_1M_aux_cil.ckpt") -Force

$splitParent = Join-Path $dataRoot "nuplan-v1.1\splits"
$miniLink = Join-Path $splitParent "mini"
$miniTarget = Join-Path $dataRoot "data\cache\mini"
New-Item -ItemType Directory -Force -Path $splitParent | Out-Null
if (-not (Test-Path -LiteralPath $miniLink)) {
    New-Item -ItemType Junction -Path $miniLink -Target $miniTarget | Out-Null
}

Write-Host "Artifacts verified and prepared under $repoRoot"
