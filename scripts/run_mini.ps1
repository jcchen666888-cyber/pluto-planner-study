param(
    [string]$EnvironmentName = "pluto-study",
    [string]$RunName = (Get-Date -Format "yyyyMMdd-HHmmss"),
    [switch]$Render
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$upstream = Join-Path $repoRoot "_deps\pluto-upstream"
$nuplan = Join-Path $repoRoot "_deps\nuplan-devkit"
$checkpoint = Join-Path $repoRoot "checkpoints\pluto_1M_aux_cil.ckpt"
$output = Join-Path $repoRoot "demo\outputs\closed_loop_nonreactive_agents\$RunName"
$videos = Join-Path $repoRoot "demo\outputs\videos"

foreach ($required in @($upstream, $nuplan, $checkpoint, (Join-Path $repoRoot "data\maps"))) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Missing artifact: $required" }
}

$env:NUPLAN_DATA_ROOT = Join-Path $repoRoot "data"
$env:NUPLAN_MAPS_ROOT = Join-Path $repoRoot "data\maps"
$env:NUPLAN_EXP_ROOT = Join-Path $repoRoot "demo\outputs"
$env:NUPLAN_MAP_VERSION = "nuplan-maps-v1.0"
$env:PYTHONPATH = "$(Join-Path $repoRoot 'compat');$upstream;$nuplan"
$env:PYTHONWARNINGS = "ignore"

$renderValue = if ($Render) { "true" } else { "false" }
$arguments = @(
    (Join-Path $upstream "run_simulation.py"),
    "+simulation=closed_loop_nonreactive_agents",
    "planner=pluto_planner",
    "scenario_builder=nuplan_mini",
    "scenario_filter=mini_demo_scenario",
    "worker=sequential",
    "verbose=false",
    "experiment_uid=$RunName",
    "planner.pluto_planner.render=$renderValue",
    "planner.pluto_planner.planner_ckpt=$checkpoint",
    "output_dir=$output"
)
if ($Render) { $arguments += "+planner.pluto_planner.save_dir=$videos" }

Push-Location $upstream
try {
    conda run -n $EnvironmentName python @arguments
}
finally {
    Pop-Location
}

conda run -n $EnvironmentName python (Join-Path $repoRoot "demo\summarize_run.py") $output --output (Join-Path $repoRoot "demo\outputs\mini_closed_loop_result.json")
