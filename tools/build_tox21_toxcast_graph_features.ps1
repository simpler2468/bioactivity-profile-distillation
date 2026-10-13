param(
    [string]$Python = "C:\Users\85290\anaconda3\envs\graph_dgl\python.exe",
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [switch]$IncludeSubgraphs
)

$InputCsv = Join-Path $ProjectRoot "data\raw_data\merge_data\Tox21_Toxcast\contain_valid\data_by_ecfp4_similarity.csv"
$OutputDir = Join-Path $ProjectRoot "data\graph_data\merge_data\Tox21_Toxcast\contain_valid\data_by_ecfp4_similarity"
$Script = Join-Path $ProjectRoot "dataset\graph_features.py"

$args = @($Script, "--input_csv", $InputCsv, "--output_dir", $OutputDir)
if ($IncludeSubgraphs) {
    $args += "--include_subgraphs"
}

& $Python @args
if ($LASTEXITCODE -ne 0) {
    throw "Graph feature generation failed."
}
