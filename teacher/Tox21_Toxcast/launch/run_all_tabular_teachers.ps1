param(
    [string]$Python = "C:\Users\85290\anaconda3\envs\graph_dgl\python.exe",
    [string]$ProjectRoot = (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))),
    [switch]$DryRun
)

$LaunchRoot = Join-Path $ProjectRoot "teacher\Tox21_Toxcast\launch"

& (Join-Path $ProjectRoot "tools\build_tox21_toxcast_tabular_features.ps1") -Python $Python -ProjectRoot $ProjectRoot
if ($LASTEXITCODE -ne 0) { throw "Feature generation failed." }

$MlpArgs = @("-Python", $Python, "-ProjectRoot", $ProjectRoot)
if ($DryRun) { $MlpArgs += "-DryRun" }
& (Join-Path $LaunchRoot "run_mlp_teachers_from_hpo.ps1") @MlpArgs
if ($LASTEXITCODE -ne 0) { throw "MLP teacher launch failed." }

$MlArgs = @("-Python", $Python, "-ProjectRoot", $ProjectRoot)
if ($DryRun) { $MlArgs += "-DryRun" }
& (Join-Path $LaunchRoot "run_rf_svm_teachers.ps1") @MlArgs
if ($LASTEXITCODE -ne 0) { throw "RF/SVM teacher launch failed." }
