param(
    [string]$Python = "C:\Users\85290\anaconda3\envs\graph_dgl\python.exe",
    [string]$ProjectRoot = (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))),
    [string[]]$Features = @("Avalon", "ECFP4", "descriptor", "Avalon_descriptor", "ECFP4_descriptor"),
    [string]$Seeds = "2026,2027,2028,2029,2030",
    [string]$Folds = "0,1,2,3,4,5,6,7,8,9",
    [string]$Device = "cuda",
    [switch]$Force,
    [switch]$DryRun
)

$Dataset = "Tox21_Toxcast"
$SplitId = "contain_valid_ecfp4_similarity"
$FeatureRoot = Join-Path $ProjectRoot "data\fp_descriptor_data\merge_data\Tox21_Toxcast\contain_valid"
$FoldRoot = Join-Path $ProjectRoot "data\raw_data\merge_data\Tox21_Toxcast\contain_valid\training_cluster_CV"
$Runner = Join-Path $ProjectRoot "teacher\run_mlp_teacher_from_hpo.py"
$OutputRoot = Join-Path $ProjectRoot "teacher"

$ConfigByFeature = @{
    "Avalon" = "hpo_t_mlp_av"
    "ECFP4" = "hpo_t_mlp_ec4"
    "descriptor" = "hpo_t_mlp_desc"
    "Avalon_descriptor" = "hpo_t_mlp_avd"
    "ECFP4_descriptor" = "hpo_t_mlp_e4d"
}

foreach ($Feature in $Features) {
    if (-not $ConfigByFeature.ContainsKey($Feature)) {
        throw "Unknown feature: $Feature"
    }
    $FeatureNpz = Join-Path $FeatureRoot "data_by_ecfp4_similarity_$Feature.npz"
    $BestConfig = Join-Path $ProjectRoot "hpo\Tox21_Toxcast\$($ConfigByFeature[$Feature])\best_config.json"
    $Args = @(
        $Runner,
        "--dataset", $Dataset,
        "--feature", $Feature,
        "--feature_npz", $FeatureNpz,
        "--fold_root", $FoldRoot,
        "--best_config", $BestConfig,
        "--output_root", $OutputRoot,
        "--split_id", $SplitId,
        "--python", $Python,
        "--seeds", $Seeds,
        "--folds", $Folds,
        "--epochs", "300",
        "--patience", "40",
        "--eval_batch_size", "128",
        "--device", $Device
    )
    if ($Force) { $Args += "--force" }
    if ($DryRun) { $Args += "--dry_run" }
    & $Python @Args
    if ($LASTEXITCODE -ne 0) { throw "MLP teacher failed for $Feature" }
}
