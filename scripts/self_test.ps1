param([string]$EnvironmentName = "pluto-study")

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
conda run -n $EnvironmentName python (Join-Path $repoRoot "demo\self_test.py") --json (Join-Path $repoRoot "demo\outputs\self_test.json")
if ($LASTEXITCODE -ne 0) { throw "NATTEN/CUDA self-test failed" }
conda run -n $EnvironmentName python (Join-Path $repoRoot "demo\validate_checkpoint.py") --json (Join-Path $repoRoot "demo\outputs\checkpoint_validation.json")
if ($LASTEXITCODE -ne 0) { throw "Checkpoint validation failed" }
conda run -n $EnvironmentName python (Join-Path $repoRoot "demo\validate_artifacts.py") --json (Join-Path $repoRoot "demo\outputs\artifact_validation.json")
if ($LASTEXITCODE -ne 0) { throw "Data/map validation failed" }
conda run -n $EnvironmentName python (Join-Path $repoRoot "demo\repo_audit.py")
if ($LASTEXITCODE -ne 0) { throw "Publish-scope audit failed" }
