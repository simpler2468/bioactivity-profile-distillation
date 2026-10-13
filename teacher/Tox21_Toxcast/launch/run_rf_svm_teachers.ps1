param(
    [string]$Python = "C:\Users\85290\anaconda3\envs\graph_dgl\python.exe",
    [string]$ProjectRoot = (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))),
    [string[]]$Features = @("Avalon", "ECFP4", "descriptor", "Avalon_descriptor", "ECFP4_descriptor"),
    [string[]]$Models = @("RF", "SVM"),
    [int]$Seed = 2026,
    [int[]]$Folds = @(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    [int]$RfNumTrees = 500,
    [int]$NJobs = -1,
    [switch]$DryRun
)

$Dataset = "Tox21_Toxcast"
$SplitId = "contain_valid_ecfp4_similarity_no_valid"
$OutputRoot = Join-Path $ProjectRoot "teacher"
$FeatureRoot = Join-Path $ProjectRoot "data\fp_descriptor_data\merge_data\Tox21_Toxcast\contain_valid"
$FoldRoot = Join-Path $ProjectRoot "data\raw_data\merge_data\Tox21_Toxcast\contain_valid\training_cluster_CV"
$RunScript = Join-Path $ProjectRoot "teacher\run_tabular_teacher.py"
$BuildScript = Join-Path $ProjectRoot "teacher\build_teacher_profile.py"
$EvalScript = Join-Path $ProjectRoot "teacher\evaluate_teacher_profiles.py"
$ProfilesRoot = Join-Path $OutputRoot "$Dataset\profiles"
$MetricsRoot = Join-Path $OutputRoot "$Dataset\metrics"

function Invoke-Or-Print {
    param([string[]]$CommandArgs)
    $CommandText = "$Python " + ($CommandArgs -join " ")
    if ($DryRun) {
        Write-Host $CommandText
    } else {
        & $Python @CommandArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed with exit code $LASTEXITCODE`: $CommandText"
        }
    }
}

foreach ($Model in $Models) {
    foreach ($Feature in $Features) {
        $FeatureNpz = Join-Path $FeatureRoot "data_by_ecfp4_similarity_$Feature.npz"
        foreach ($Fold in $Folds) {
            $FoldCsv = Join-Path $FoldRoot "training_cluster_CV_no_valid$Fold.csv"
            $Args = @(
                $RunScript,
                "--dataset", $Dataset,
                "--feature", $Feature,
                "--model", $Model,
                "--split_id", $SplitId,
                "--fold", "$Fold",
                "--seed", "$Seed",
                "--feature_npz", $FeatureNpz,
                "--fold_csv", $FoldCsv,
                "--output_root", $OutputRoot
            )
            if ($Model -eq "RF") {
                $Args += @("--rf_n_estimators", "$RfNumTrees", "--n_jobs", "$NJobs")
            }
            Invoke-Or-Print $Args
        }
        Invoke-Or-Print @(
            $BuildScript,
            "--dataset", $Dataset,
            "--feature", $Feature,
            "--model", $Model,
            "--split_id", $SplitId,
            "--feature_npz", $FeatureNpz,
            "--output_root", $OutputRoot,
            "--folds", ($Folds -join ","),
            "--seeds", "$Seed",
            "--ensemble_name", "single"
        )
    }
}

Invoke-Or-Print @(
    $EvalScript,
    "--profiles_root", $ProfilesRoot,
    "--output_root", $MetricsRoot
)
