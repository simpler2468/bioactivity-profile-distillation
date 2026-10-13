param(
    [string]$Python = "C:\Users\85290\anaconda3\envs\graph_dgl\python.exe",
    [string]$ProjectRoot = (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))),
    [string]$Dataset = "Tox21_Toxcast",
    [string]$SelectedProfileName = "feature=taskwise_selected__model=Teacher__split=train_oof_auc__seed=mixed",
    [string]$ScaledProfileName = "feature=taskwise_selected_platt__model=Teacher__split=train_oof_auc__seed=mixed"
)

$OutputRoot = Join-Path $ProjectRoot "teacher"
$ProfilesRoot = Join-Path $OutputRoot "$Dataset\profiles"
$MetricsRoot = Join-Path $OutputRoot "$Dataset\metrics"
$AllMetricsCsv = Join-Path $MetricsRoot "model_feature_summary.csv"
$EvaluateScript = Join-Path $ProjectRoot "teacher\evaluate_teacher_profiles.py"
$SelectScript = Join-Path $ProjectRoot "teacher\select_teacher_profile.py"
$CalibrateScript = Join-Path $ProjectRoot "teacher\calibrate_teacher_profile.py"
$ReportScript = Join-Path $ProjectRoot "teacher\report_teacher_performance.py"

function Invoke-Python {
    param([string[]]$CommandArgs)
    & $Python @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $Python $($CommandArgs -join ' ')"
    }
}

Invoke-Python @(
    $EvaluateScript,
    "--profiles_root", $ProfilesRoot,
    "--output_root", $MetricsRoot
)

Invoke-Python @(
    $SelectScript,
    "--dataset", $Dataset,
    "--profiles_root", $ProfilesRoot,
    "--metrics_csv", $AllMetricsCsv,
    "--output_root", $OutputRoot,
    "--metric", "roc_auc",
    "--selection_split", "train_oof",
    "--mlp_seed", "ensemble5",
    "--ml_seeds", "2026",
    "--mlp_splits", "contain_valid_ecfp4_similarity",
    "--ml_splits", "contain_valid_ecfp4_similarity_no_valid",
    "--profile_name", $SelectedProfileName
)

$SelectedProfileDir = Join-Path $ProfilesRoot $SelectedProfileName
$ScaledProfileDir = Join-Path $ProfilesRoot $ScaledProfileName
$ScaledMetricsDir = Join-Path $MetricsRoot $ScaledProfileName

Invoke-Python @(
    $CalibrateScript,
    "--input_profile_dir", $SelectedProfileDir,
    "--output_profile_dir", $ScaledProfileDir,
    "--metrics_dir", $ScaledMetricsDir,
    "--copy_selection"
)

Invoke-Python @(
    $ReportScript,
    "--metrics_root", $MetricsRoot,
    "--output_csv", (Join-Path $MetricsRoot "teacher_profile_test_auc_summary.csv")
)
