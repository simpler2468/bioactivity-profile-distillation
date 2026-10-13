param(
    [string]$Python = "C:\Users\85290\anaconda3\envs\graph_dgl\python.exe",
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$InputCsv = Join-Path $ProjectRoot "data\raw_data\merge_data\Tox21_Toxcast\contain_valid\data_by_ecfp4_similarity.csv"
$Script = Join-Path $ProjectRoot "dataset\molecular_features.py"

& $Python $Script --input_csv $InputCsv
if ($LASTEXITCODE -ne 0) {
    throw "Tabular feature generation failed."
}
