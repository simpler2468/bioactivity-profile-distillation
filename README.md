# Bioactivity Profile Distillation

This repository provides the companion code for the manuscript
*Dissecting Multitask Learning Gains in In Vitro Bioactivity Prediction Using
Teacher-Derived Bioactivity Profiles*. It contains the code and data-processing
workflow used for the analyses described in the manuscript. For additional
background, methodological details, and interpretation of the results, please
refer to the manuscript.

## Environment

The validated environment uses Conda for Python and the base scientific
packages, and pip for PyTorch, RDKit, HPO, and Excel support. The commands
below reproduce the Windows CUDA 11.8 environment used for development.

```powershell
conda create -n bioactivity_profile python=3.8.0 -y
conda activate bioactivity_profile
```

Install the base scientific and graph-learning packages:

```powershell
conda install `
  -c dglteam/label/cu118 `
  -c conda-forge `
  -c defaults `
  dgl=1.1.2.cu118 `
  numpy=1.24.3 `
  pandas=2.0.3 `
  scipy=1.9.3 `
  scikit-learn=1.3.2 `
  joblib=1.4.2 `
  -y
```

Install the CUDA 11.8 build of PyTorch:

```powershell
python -m pip install `
  torch==2.1.0+cu118 `
  --index-url https://download.pytorch.org/whl/cu118
```

Install the remaining Python packages:

```powershell
python -m pip install `
  rdkit-pypi==2022.9.5 `
  optuna==4.5.0 `
  openpyxl==3.1.5
```

The pinned versions are:

| Package | Version |
|---|---:|
| Python | 3.8.0 |
| numpy | 1.24.3 |
| pandas | 2.0.3 |
| rdkit-pypi | 2022.9.5 |
| scipy | 1.9.3 |
| scikit-learn | 1.3.2 |
| joblib | 1.4.2 |
| torch | 2.1.0+cu118 |
| dgl | 1.1.2.cu118 |
| optuna | 4.5.0 |
| openpyxl | 3.1.5 |

Verify the installation with:

```powershell
python -c "import numpy, pandas, rdkit, torch, dgl, sklearn, scipy, optuna, openpyxl, joblib; print('Environment OK')"
```

## Data

The `data` directory contains the raw source files, processed release tables,
task definitions, and audit statistics used in this study. The main source
collections under `data/raw_data` are:

1. **Tox21 Challenge.** This collection originates from the official Tox21
   Challenge data. It includes the original challenge test records used for
   benchmark evaluation and a curated analysis version in which duplicate
   structures are resolved. The curated view is used for downstream analyses,
   including PCA and related profile characterization. The collection contains
   12 endpoint/assay tasks.
2. **ToxCast auxiliary assays.** This collection was downloaded from
   invitroDB version 4.3 and provides 81 auxiliary endpoint/assay tasks.
3. **TOXRIC.** This collection was obtained from the ToxACoL publication and
   retains the original five-fold cross-validation organization used for the
   TOXRIC analyses. The five source files are named
   `TOXRIC_canonical_in_vivo_0.csv` through
   `TOXRIC_canonical_in_vivo_4.csv`.
4. **Merged Tox21--ToxCast data.** The Tox21 and ToxCast collections are
   integrated into a 93-endpoint profile while preserving benchmark alignment
   with the standalone Tox21 dataset. The merged release provides the
   8,557/952/722 training, validation, and test split, together with the
   corresponding no-validation view and cluster-based cross-validation files.

The 10-fold cross-validation dataset is derived from the 8,557-compound
training group of the validation-containing merged dataset. Only training
compounds are assigned to folds, and complete ECFP4 clusters are kept intact
so that a cluster is never split across a fold's training and validation
parts. The fixed 952-compound validation group and 722-compound test group are
not used to define the fold assignment. For each fold, both a validation view
and a no-validation view are provided; in the latter, the held-out fold is
relabelled as training for workflows that do not use a separate validation
group.

For additional information about the source data, please consult the relevant
database documentation, source publications, and the manuscript associated
with this repository.

## Feature Generation

Feature generation is performed from a processed merged CSV containing the
`SMILES`, endpoint-label, and `group` columns. The default commands below use
the validation-containing release table:

```text
data/raw_data/merge_data/Tox21_Toxcast/contain_valid/data_by_ecfp4_similarity.csv
```

### Tabular molecular features

The tabular feature generator is implemented in
`dataset/molecular_features.py`. For each valid molecule it produces:

| Feature | Details | Output |
|---|---|---|
| ECFP4 | Morgan fingerprint with radius 2 and 1,024 bits | `*_ECFP4.npz` |
| Avalon | RDKit Avalon fingerprint with 1,024 bits | `*_Avalon.npz` |
| RDKit descriptors | All descriptors exposed by `rdkit.Chem.Descriptors` | `*_descriptor.npz` |
| Avalon + descriptors | Concatenation of Avalon and normalized descriptors | `*_Avalon_descriptor.npz` |
| ECFP4 + descriptors | Concatenation of ECFP4 and normalized descriptors | `*_ECFP4_descriptor.npz` |

Descriptor normalization is fitted from the rows labelled `training` only.
Each descriptor is divided by its training-set 95th percentile of absolute
values, non-finite values are replaced by zero, and the normalized values are
clipped to `[-10, 10]`. This prevents validation and test records from
influencing descriptor scaling. Each NPZ file stores the feature matrix,
labels, group labels, SMILES, endpoint names, feature names, failed-record
information, and descriptor scales when applicable.

Generate the tabular features with:

```powershell
powershell -ExecutionPolicy Bypass -File tools\build_tox21_toxcast_tabular_features.ps1 `
  -Python "C:\Users\85290\anaconda3\envs\graph_dgl\python.exe" `
  -ProjectRoot (Get-Location)
```

The output is written under:

```text
data/fp_descriptor_data/merge_data/Tox21_Toxcast/contain_valid/
```

The underlying generator can also be called directly without writing files:

```powershell
python dataset\molecular_features.py `
  --input_csv data\raw_data\merge_data\Tox21_Toxcast\contain_valid\data_by_ecfp4_similarity.csv `
  --dry_run
```

### Molecular graph features

The graph feature generator is implemented in `dataset/graph_features.py` and
uses DGL and RDKit. It creates one full molecular graph for each successfully
parsed molecule. Chemical bonds are represented as bidirectional graph edges.
Each atom stores an 8-dimensional feature vector containing normalized atomic
number, degree, chirality, formal charge, hybridization, total hydrogen count,
implicit valence, and covalent radius. Each bond stores a 2-dimensional vector
containing bond type and bond direction.

The standard output includes `graphs.bin`, `metadata.csv`,
`failed_records.csv`, `task_names.json`, and `graph_schema.json`. Row indices,
SMILES, labels, and group labels are retained so graph samples remain aligned
with the tabular feature files.

Generate the molecular graphs with:

```powershell
powershell -ExecutionPolicy Bypass -File tools\build_tox21_toxcast_graph_features.ps1 `
  -Python "C:\Users\85290\anaconda3\envs\graph_dgl\python.exe" `
  -ProjectRoot (Get-Location)
```

The output is written under:

```text
data/graph_data/merge_data/Tox21_Toxcast/contain_valid/data_by_ecfp4_similarity/
```

## Teacher Models

Teacher models convert each molecular representation into a 93-dimensional
bioactivity profile. The teacher workflow is kept separate from student-model
training and consists of hyperparameter optimization, repeated training,
profile assembly, task-wise teacher selection, and calibration.

### HPO method

HPO is implemented in `hpo/optimize_teacher_mlp.py` and uses Optuna's
TPE sampler with `seed=2026`. The default search contains:

| Parameter | Search values |
|---|---|
| `num_hidden_layers` | 2, 3, 4 |
| `shrink_ratio` | 0.5, 0.65, 0.8 |
| `min_hidden_dim` | 64, 128, 186, 256 |
| `dropout` | 0.1, 0.2, 0.3, 0.4, 0.5 |
| `lr` | 1e-4, 3e-4, 5e-4, 1e-3, 3e-3 |
| `weight_decay` | 1e-6, 1e-5, 1e-4, 1e-3 |
| `batch_size` | 128, 256, 512 |
| `use_pos_weight` | enabled or disabled |
| `hard_reduction` | `observation` or `task` |

The activation, normalization, and gradient clipping settings are fixed to
LeakyReLU, layer normalization, and a clipping norm of 1.0. The default HPO
budget is 30 trials. Each trial trains one MLP on the training group of
`training_cluster_CV_0.csv`. When that fold file contains a `valid` group,
the validation ROC-AUC is used for early stopping. The Optuna objective is the
mean ROC-AUC over the 12 Tox21 benchmark tasks on the fold-held-out `oof`
predictions; the official test group is never used for HPO fitting or model
selection.

Run HPO for the descriptor representation with:

```powershell
python hpo\optimize_teacher_mlp.py `
  --dataset Tox21_Toxcast `
  --feature descriptor `
  --feature_npz data\fp_descriptor_data\merge_data\Tox21_Toxcast\contain_valid\data_by_ecfp4_similarity_descriptor.npz `
  --fold_csv data\raw_data\merge_data\Tox21_Toxcast\contain_valid\training_cluster_CV\training_cluster_CV_0.csv `
  --output_dir hpo\Tox21_Toxcast\teacher_MLP_descriptor `
  --n_trials 30 `
  --seed 2026 `
  --fold 0 `
  --epochs 300 `
  --patience 40 `
  --eval_batch_size 128 `
  --device cuda
```

The same script can be run for `Avalon`, `ECFP4`, `Avalon_descriptor`, and
`ECFP4_descriptor` by changing `--feature`, the NPZ path, and the output
directory. Each HPO run writes an Optuna SQLite study, `trials.csv`, the
selected `best_config.json`, and a copy of the best trial outputs.

### Teacher training

The low-level trainer is `teacher/run_tabular_teacher.py`. MLP teachers use
the validation-containing ten-fold files and are trained for every requested
fold and seed. A run uses the fold's `training` group for fitting, the fold's
`valid` group for early stopping when present, and the fold's held-out group
for OOF prediction. It also writes predictions for the fixed global
validation and test groups. MLP raw scores are logits.

Random-forest and SVM teachers are trained task-wise with the same trainer.
Their raw scores are, respectively, the clipped positive-class probability
logit and the SVM decision function. The no-validation fold files are used for
these models because they do not perform neural-network early stopping.

Run all MLP teacher seeds and folds from the saved HPO configurations with:

```powershell
powershell -ExecutionPolicy Bypass -File teacher\Tox21_Toxcast\launch\run_mlp_teachers_from_hpo.ps1 `
  -Python "C:\Users\85290\anaconda3\envs\graph_dgl\python.exe" `
  -ProjectRoot (Get-Location) `
  -Device cuda
```

The default MLP seeds are `2026,2027,2028,2029,2030` and the default folds are
`0` through `9`. Use `-DryRun` to print the commands without training.

Run the task-wise RF and SVM teachers with:

```powershell
powershell -ExecutionPolicy Bypass -File teacher\Tox21_Toxcast\launch\run_rf_svm_teachers.ps1 `
  -Python "C:\Users\85290\anaconda3\envs\graph_dgl\python.exe" `
  -ProjectRoot (Get-Location)
```

The combined tabular-teacher launcher is:

```powershell
powershell -ExecutionPolicy Bypass -File teacher\Tox21_Toxcast\launch\run_all_tabular_teachers.ps1 `
  -Python "C:\Users\85290\anaconda3\envs\graph_dgl\python.exe" `
  -ProjectRoot (Get-Location)
```

### Profile assembly and evaluation

`teacher/build_teacher_profile.py` gathers the fold-level predictions for a
given feature/model pair. For the MLP teachers, the five seeds are first
assembled into an `ensemble5` profile by an element-wise nan-mean over the
seed-level profiles. Each profile retains `train_oof`, `valid`, and `test`
raw scores, labels, row indices, and task names. Profile performance is
evaluated by `teacher/evaluate_teacher_profiles.py`, which writes per-task and
summary ROC-AUC, R2, and RMSE metrics.

For example, to assemble and evaluate one Avalon MLP profile manually:

```powershell
python teacher\build_teacher_profile.py `
  --dataset Tox21_Toxcast `
  --feature Avalon `
  --model MLP `
  --split_id contain_valid_ecfp4_similarity `
  --feature_npz data\fp_descriptor_data\merge_data\Tox21_Toxcast\contain_valid\data_by_ecfp4_similarity_Avalon.npz `
  --output_root teacher `
  --folds 0,1,2,3,4,5,6,7,8,9 `
  --seeds 2026,2027,2028,2029,2030 `
  --ensemble_name ensemble5

python teacher\evaluate_teacher_profiles.py `
  --profiles_root teacher\Tox21_Toxcast\profiles `
  --output_root teacher\Tox21_Toxcast\metrics
```

### Teacher selection

Teacher selection is implemented in `teacher/select_teacher_profile.py`.
Selection is performed separately for each endpoint, rather than choosing one
single representation for all 93 tasks. Candidate profiles are ranked by
`train_oof` ROC-AUC. The standard selection compares the five-seed MLP
ensembles with the single-seed RF/SVM profiles, using the validation-containing
split for MLP and the no-validation split for RF/SVM. The selected profile
records the selected feature/model and source profile for every task in
`task_teacher_selection.csv`.

The complete selection-plus-scaling workflow is:

```powershell
powershell -ExecutionPolicy Bypass -File teacher\Tox21_Toxcast\launch\build_scaled_teacher_profile.ps1 `
  -Python "C:\Users\85290\anaconda3\envs\graph_dgl\python.exe" `
  -ProjectRoot (Get-Location)
```

The two stages can also be run separately. First evaluate all available
profiles and select the best teacher for each endpoint:

```powershell
python teacher\evaluate_teacher_profiles.py `
  --profiles_root teacher\Tox21_Toxcast\profiles `
  --output_root teacher\Tox21_Toxcast\metrics

python teacher\select_teacher_profile.py `
  --dataset Tox21_Toxcast `
  --profiles_root teacher\Tox21_Toxcast\profiles `
  --metrics_csv teacher\Tox21_Toxcast\metrics\model_feature_summary.csv `
  --output_root teacher `
  --metric roc_auc `
  --selection_split train_oof `
  --mlp_seed ensemble5 `
  --ml_seeds 2026 `
  --mlp_splits contain_valid_ecfp4_similarity `
  --ml_splits contain_valid_ecfp4_similarity_no_valid `
  --profile_name "feature=taskwise_selected__model=Teacher__split=train_oof_auc__seed=mixed"
```

Then apply task-wise Platt scaling to the selected profile:

```powershell
python teacher\calibrate_teacher_profile.py `
  --input_profile_dir "teacher\Tox21_Toxcast\profiles\feature=taskwise_selected__model=Teacher__split=train_oof_auc__seed=mixed" `
  --output_profile_dir "teacher\Tox21_Toxcast\profiles\feature=taskwise_selected_platt__model=Teacher__split=train_oof_auc__seed=mixed" `
  --metrics_dir "teacher\Tox21_Toxcast\metrics\feature=taskwise_selected_platt__model=Teacher__split=train_oof_auc__seed=mixed" `
  --copy_selection
```

The scaling command fits one pair of parameters `(a, b)` per endpoint from
`train_oof_raw_score` and `train_oof_y`, then transforms all three splits:
`train_oof`, `valid`, and `test`. It writes the calibrated profile,
`platt_parameters.csv`, per-task calibration metrics, and the copied
`task_teacher_selection.csv`.

### Teacher scaling

Scaling is implemented in `teacher/calibrate_teacher_profile.py`. After
task-wise selection, a separate Platt calibration is fitted for each endpoint
using only its `train_oof` raw scores and labels. The fitted transformation is

```text
calibrated_logit = a * raw_score + b
```

The same task-specific parameters are then applied to the validation and test
scores; neither split is used to fit the calibration. The calibrated profile
stores Platt-scaled logits in `*_raw_score`, probabilities in `*_prob`, and
the parameters and calibration diagnostics in `platt_parameters.csv` and the
corresponding metrics directory. The launcher above writes the standard
selected profile as
`feature=taskwise_selected__model=Teacher__split=train_oof_auc__seed=mixed`
and the calibrated version as
`feature=taskwise_selected_platt__model=Teacher__split=train_oof_auc__seed=mixed`.

## TOXRIC Main External Analysis

The main TOXRIC analysis tests whether the learned in vitro profile transfers
to external acute in vivo toxicity endpoints. The primary comparison contains
three feature settings:

1. **Avalon-only:** a 1,024-bit Avalon molecular fingerprint.
2. **In vitro profile-only:** the 93-dimensional calibrated-logit teacher
   profile.
3. **Avalon + in vitro profile:** concatenation of the Avalon fingerprint and
   the 93-dimensional profile.

The TOXRIC files retain the original five folds from the ToxACoL source. The
59 in vivo endpoints are used only as downstream training/evaluation targets;
they are not used to generate the in vitro profile. The profile is generated
from the selected Tox21/ToxCast teachers and their endpoint-specific Platt
parameters. For compounds overlapping the teacher data, the corresponding
leakage-safe `train_oof`, `valid`, or `test` profile is used. For non-overlap
compounds, predictions from the ten teacher folds and five MLP seeds are
averaged and then Platt-scaled.

### Feature construction

For each TOXRIC fold:

1. Parse the fold's SMILES and generate Avalon fingerprints with the same RDKit
   feature builder used elsewhere in the project.
2. Generate the 93-dimensional in vitro profile from the calibrated teacher
   profile. The stored feature values are calibrated logits; the associated
   probabilities are retained as a separate representation.
3. Concatenate Avalon and the in vitro profile after verifying identical
   SMILES, labels, groups, and row order. The primary analysis uses the raw
   calibrated logits. Probability and training-group task-wise z-score
   versions are optional sensitivity representations.

The validated external-analysis implementation uses the following commands
for one fold; repeat them for fold IDs `0` through `4`:

```powershell
python dataset\molecular_features.py `
  --input_csv data\raw_data\TOXRIC\TOXRIC_canonical_in_vivo_0.csv `
  --output_dir data\fp_descriptor_data\TOXRIC

python dataset\build_in_vitro_feature.py `
  --raw_csv data\raw_data\TOXRIC\TOXRIC_canonical_in_vivo_0.csv `
  --feature_root data\fp_descriptor_data\TOXRIC `
  --output_dir data\in_vitro_feature\TOXRIC `
  --safe_profile_dir "teacher\Tox21_Toxcast\profiles\feature=taskwise_selected_platt__model=Teacher__split=train_oof_auc__seed=mixed" `
  --teacher_root teacher `
  --teacher_dataset Tox21_Toxcast `
  --teacher_split_id contain_valid_ecfp4_similarity_hpo `
  --seeds 2026,2027,2028,2029,2030 `
  --folds 0,1,2,3,4,5,6,7,8,9 `
  --device cuda

python dataset\build_toxacol_combined_features.py `
  --avalon_path data\fp_descriptor_data\TOXRIC\TOXRIC_canonical_in_vivo_0_Avalon.npz `
  --in_vitro_path data\in_vitro_feature\TOXRIC\TOXRIC_canonical_in_vivo_0_in_vitro_feature.npz `
  --output_path data\combined_feature\TOXRIC\TOXRIC_canonical_in_vivo_0_Avalon_in_vitro_raw.npz `
  --feature_name Avalon_in_vitro_raw `
  --mode raw `
  --train_group training
```

Use `--mode prob` for the probability representation or `--mode zscore` for
the task-wise z-score representation. The z-score parameters are estimated
from the current fold's training group only.

The TOXRIC-specific feature-builder and regression-driver commands above
correspond to the validated external-analysis module. The minimal release
currently retains the raw TOXRIC five-fold files and the general Avalon and
teacher-profile code; the TOXRIC-specific external regression driver should be
transferred alongside these commands before running this section from a clean
clone.

### Downstream evaluation

For each outer fold and seed, train the same ToxACoL regression model on the
training group and evaluate it on the fold's test group. The three primary
runs use the following feature names and split namespaces:

```powershell
python student\run_toxacol_student.py `
  --dataset TOXRIC `
  --fold_id 0 `
  --feature_name Avalon `
  --split_id canonical_in_vivo `
  --seeds original `
  --epochs 120 `
  --monitor mean_r2 `
  --final_model best `
  --device cuda

python student\run_toxacol_student.py `
  --dataset TOXRIC `
  --fold_id 0 `
  --feature_name in_vitro_feature `
  --split_id canonical_in_vivo_in_vitro_feature `
  --seeds original `
  --epochs 120 `
  --monitor mean_r2 `
  --final_model best `
  --device cuda

python student\run_toxacol_student.py `
  --dataset TOXRIC `
  --fold_id 0 `
  --feature_name Avalon_in_vitro_raw `
  --feature_cache data\combined_feature\TOXRIC\TOXRIC_canonical_in_vivo_0_Avalon_in_vitro_raw.npz `
  --split_id canonical_in_vivo_Avalon_in_vitro_raw `
  --seeds original `
  --epochs 120 `
  --monitor mean_r2 `
  --final_model best `
  --device cuda
```

Binary toxicity outcomes are evaluated with endpoint-level ROC-AUC after the
specified training-set thresholding. Continuous outcomes are evaluated with
endpoint-level R2 and RMSE. Metrics are calculated per endpoint and then
macro-averaged across the 59 TOXRIC endpoints. The same compounds and endpoint
labels are used for the paired feature comparisons.

Summarize the three feature settings and calculate per-endpoint deltas against
Avalon-only with:

```powershell
python analysis\compare_toxric_in_vitro_features.py `
  --project_root (Get-Location) `
  --dataset TOXRIC `
  --baseline Avalon `
  --output_dir analysis\toxric_comparison
```

This writes `toxric_in_vitro_delta_vs_avalon_per_task.csv` and
`toxric_in_vitro_delta_vs_avalon_summary.csv`. The summary reports the number
of improved/tied/worse endpoints, mean and median metric changes, and weighted
changes based on the number of valid observations. The main interpretation is
therefore a paired comparison of Avalon-only, profile-only, and combined
features, rather than a comparison of unrelated compound sets.

## Student Models

Student models are trained on the fixed merged split containing 8,557 training
compounds, 952 validation compounds, and 722 test compounds. The validation
group is used for early stopping and HPO. The test group is evaluated only
after the configuration and seed ensemble have been fixed.

The repository supports two student families:

1. **Tabular MLP.** `student/run_mlp_student.py` trains an MLP from an ECFP4,
   Avalon, descriptor, or concatenated feature NPZ file.
2. **Graph GIN/eGIN.** `student/run_graph_student.py` trains the graph model
   from the DGL graph dataset. The command-line model name is `GIN`.

Both families support masked hard-label BCE. The soft-label variant adds a
teacher-profile loss to the hard BCE objective. For classification endpoints,
the soft term is an MSE between student logits and teacher raw scores; the
default soft regression weight is zero. Teacher targets are loaded only for
training rows from `train_oof_raw_score`, and validation/test performance is
always calculated from the hard labels.

### BCE baseline HPO

MLP and graph BCE HPO are implemented in
`hpo/optimize_student_mlp.py` and `hpo/optimize_graph_student.py`. Both use an
Optuna TPE sampler with `seed=2026`, a default budget of 30 trials, and
maximize the validation mean ROC-AUC. For a 93-task student, the default
objective is the 12 Tox21 benchmark tasks; use `--objective_tasks all` to
optimize the mean across all 93 tasks.

The tabular MLP search space is:

| Parameter | Search values |
|---|---|
| `num_hidden_layers` | 2, 3, 4 |
| `shrink_ratio` | 0.5, 0.65, 0.8 |
| `min_hidden_dim` | 64, 128, 186, 256 |
| `dropout` | 0.1, 0.2, 0.3, 0.4, 0.5 |
| `lr` | 1e-4, 3e-4, 5e-4, 1e-3, 3e-3 |
| `weight_decay` | 1e-6, 1e-5, 1e-4, 1e-3 |
| `batch_size` | 128, 256, 512 |
| `use_pos_weight` | enabled or disabled |
| `hard_reduction` | `observation` or `task` |

The graph GIN search space is:

| Parameter | Search values |
|---|---|
| `hidden_dim` | 128, 256 |
| `num_layers` | 3, 4, 5 |
| `dropout` | 0.1, 0.2, 0.3, 0.4 |
| `norm_type` | `batch_norm` or `layer_norm` |
| `readout` | `max`, `mean`, or `attention` |
| `lr` | 1e-4, 3e-4, 5e-4, 1e-3 |
| `weight_decay` | 1e-6, 1e-5, 1e-4, 1e-3 |
| `batch_size` | 32 or 64 |
| `use_pos_weight` | enabled or disabled |
| `hard_reduction` | `observation` or `task` |
| `mlp_hidden_dims` | `128`, `256`, or `256,128` |

Run MLP BCE HPO, for example, with Avalon features:

```powershell
python hpo\optimize_student_mlp.py `
  --dataset Tox21_Toxcast `
  --feature Avalon `
  --tasks all_93 `
  --feature_npz data\fp_descriptor_data\merge_data\Tox21_Toxcast\contain_valid\data_by_ecfp4_similarity_Avalon.npz `
  --output_dir hpo\Tox21_Toxcast\student_MLP_Avalon_all93 `
  --objective_tasks tox21_12 `
  --n_trials 30 `
  --seed 2026 `
  --epochs 300 `
  --patience 40 `
  --eval_batch_size 128 `
  --device cuda
```

Run graph GIN BCE HPO with:

```powershell
python hpo\optimize_graph_student.py `
  --dataset Tox21_Toxcast `
  --model GIN `
  --tasks all_93 `
  --graph_dir data\graph_data\merge_data\Tox21_Toxcast\contain_valid\data_by_ecfp4_similarity `
  --output_dir hpo\Tox21_Toxcast\student_GIN_all93 `
  --objective_tasks tox21_12 `
  --n_trials 30 `
  --seed 2026 `
  --epochs 300 `
  --patience 40 `
  --eval_batch_size 128 `
  --num_workers 0 `
  --device cuda
```

Each HPO directory contains the Optuna SQLite study, `trials.csv`,
`best_config.json`, and a copy of the best trial. The `best_config.json` is the
input for the repeated-seed training commands below.

### BCE baseline training

Train a five-seed MLP ensemble from the selected BCE configuration with:

```powershell
python student\run_mlp_student_from_hpo.py `
  --dataset Tox21_Toxcast `
  --feature Avalon `
  --tasks all_93 `
  --feature_npz data\fp_descriptor_data\merge_data\Tox21_Toxcast\contain_valid\data_by_ecfp4_similarity_Avalon.npz `
  --best_config hpo\Tox21_Toxcast\student_MLP_Avalon_all93\best_config.json `
  --output_root student `
  --split_id contain_valid_mlp_bce `
  --loss_name mask_bce `
  --seeds 2026,2027,2028,2029,2030 `
  --epochs 300 `
  --patience 40 `
  --device cuda
```

Train the graph GIN BCE ensemble with:

```powershell
python student\run_graph_student_from_hpo.py `
  --dataset Tox21_Toxcast `
  --model GIN `
  --tasks all_93 `
  --graph_dir data\graph_data\merge_data\Tox21_Toxcast\contain_valid\data_by_ecfp4_similarity `
  --best_config hpo\Tox21_Toxcast\student_GIN_all93\best_config.json `
  --output_root student `
  --split_id contain_valid_graph_bce `
  --loss_name mask_bce `
  --seeds 2026,2027,2028,2029,2030 `
  --epochs 300 `
  --patience 40 `
  --device cuda
```

The runners skip completed seed runs unless `--force` is supplied. They also
write a seed-summary CSV under `student/Tox21_Toxcast/metrics/`.

### Soft-label HPO and training

Soft-label HPO is implemented in
`hpo/optimize_student_soft_weights.py`. It keeps the selected BCE architecture
fixed and searches only the classification soft-loss weight using a grid:

```text
0.0003, 0.001, 0.003, 0.01, 0.03, 0.1
```

The default soft configuration is `soft_cls_weight` from this grid,
`soft_reg_weight=0`, `aux_reduction=observation`, and
`aux_target_clip=0`. The default objective is validation mean ROC-AUC on the
12 Tox21 tasks. The teacher profile should normally be the calibrated,
task-wise selected profile generated above.

Run soft-label HPO for the graph student with:

```powershell
python hpo\optimize_student_soft_weights.py `
  --dataset Tox21_Toxcast `
  --student_type graph `
  --model GIN `
  --tasks all_93 `
  --graph_dir data\graph_data\merge_data\Tox21_Toxcast\contain_valid\data_by_ecfp4_similarity `
  --base_best_config hpo\Tox21_Toxcast\student_GIN_all93\best_config.json `
  --teacher_profile_dir "teacher\Tox21_Toxcast\profiles\feature=taskwise_selected_platt__model=Teacher__split=train_oof_auc__seed=mixed" `
  --output_dir hpo\Tox21_Toxcast\student_GIN_soft_full `
  --objective_tasks tox21_12 `
  --soft_profile_mode full `
  --seed 2026 `
  --epochs 300 `
  --patience 40 `
  --device cuda
```

Train the five-seed graph soft-label ensemble from the soft HPO result:

```powershell
python student\run_graph_student_from_hpo.py `
  --dataset Tox21_Toxcast `
  --model GIN `
  --tasks all_93 `
  --graph_dir data\graph_data\merge_data\Tox21_Toxcast\contain_valid\data_by_ecfp4_similarity `
  --best_config hpo\Tox21_Toxcast\student_GIN_soft_full\best_config.json `
  --output_root student `
  --split_id contain_valid_graph_soft_full `
  --loss_name mask_bce_soft `
  --teacher_profile_dir "teacher\Tox21_Toxcast\profiles\feature=taskwise_selected_platt__model=Teacher__split=train_oof_auc__seed=mixed" `
  --soft_profile_mode full `
  --seeds 2026,2027,2028,2029,2030 `
  --epochs 300 `
  --patience 40 `
  --device cuda
```

The same HPO and runner scripts support `student_type mlp` by replacing the
graph arguments with `--feature` and `--feature_npz`, and using the MLP runner.

### Student ablations

The three ablation families use the same fixed split, seed list, and hard-label
evaluation protocol as the main student runs.

**Task-count ablation.** First create nested task sets containing 12, 32, 52,
72, and 93 tasks. The 12 Tox21 tasks are retained in every set, and additional
ToxCast tasks are added in a fixed seeded order:

```powershell
python tools\build_task_count_ablation_sets.py `
  --graph_dir data\graph_data\merge_data\Tox21_Toxcast\contain_valid\data_by_ecfp4_similarity `
  --seed 2026
```

Run one graph BCE task-count group, for example the 32-task group:

```powershell
python tools\run_task_count_ablation_one.py `
  --best_config hpo\Tox21_Toxcast\student_GIN_all93\best_config.json `
  --tasks tox21_12_plus_20 `
  --task_file data\task_sets\Tox21_Toxcast\task_count_ablation_seed2026\tox21_12_plus_20.txt `
  --graph_dir data\graph_data\merge_data\Tox21_Toxcast\contain_valid\data_by_ecfp4_similarity `
  --output_root student `
  --seeds 2026,2027,2028,2029,2030 `
  --device cuda
```

Repeat with `tox21_12`, `tox21_12_plus_40`, `tox21_12_plus_60`, and `all_93`.

**Profile-content ablation.** The wrapper runs `full`, `target_only`, and
`non_target` soft-profile modes in separate output namespaces:

```powershell
python tools\run_profile_content_ablation.py `
  --runner graph `
  --python "C:\Users\85290\anaconda3\envs\graph_dgl\python.exe" `
  --modes full,target_only,non_target `
  -- `
  --model GIN `
  --tasks all_93 `
  --graph_dir data\graph_data\merge_data\Tox21_Toxcast\contain_valid\data_by_ecfp4_similarity `
  --best_config hpo\Tox21_Toxcast\student_GIN_soft_full\best_config.json `
  --teacher_profile_dir "teacher\Tox21_Toxcast\profiles\feature=taskwise_selected_platt__model=Teacher__split=train_oof_auc__seed=mixed" `
  --loss_name mask_bce_soft `
  --seeds 2026,2027,2028,2029,2030 `
  --device cuda
```

**Profile-reconstruction ablation.** This wrapper substitutes PCA-reconstructed
teacher profiles for the full selected profile. The supported component sets
are `pc1`, `pc1_5`, `pc1_10`, and `pc2_10`:

```powershell
python tools\run_profile_reconstruction_ablation.py `
  --runner graph `
  --profile_root teacher\Tox21_Toxcast\profiles `
  --python "C:\Users\85290\anaconda3\envs\graph_dgl\python.exe" `
  --components pc1,pc1_5,pc1_10,pc2_10 `
  -- `
  --model GIN `
  --tasks all_93 `
  --graph_dir data\graph_data\merge_data\Tox21_Toxcast\contain_valid\data_by_ecfp4_similarity `
  --best_config hpo\Tox21_Toxcast\student_GIN_soft_full\best_config.json `
  --loss_name mask_bce_soft `
  --seeds 2026,2027,2028,2029,2030 `
  --device cuda
```

All ablation wrappers accept `--dry_run` where implemented, allowing the
expanded commands and output namespaces to be checked before training.

## Result Aggregation

`tools/aggregate_results_to_xlsx.py` scans the currently available files under
`teacher/`, `student/`, `hpo/`, and `analysis/`. It collects metric and summary
CSV files, HPO and run-summary JSON files, and a file inventory into one
inspection workbook. The scanner is intentionally tolerant: it reports the
results that exist and does not require every model family to have been run.

Run it from the project root with:

```powershell
python tools\aggregate_results_to_xlsx.py `
  --project_root (Get-Location) `
  --output analysis\result_summaries\model_results_summary.xlsx
```

The workbook contains:

| Sheet | Contents |
|---|---|
| `metric_rows` | Rows collected from metric, summary, and HPO trial CSV files |
| `summaries` | Flattened fields from summary JSON, `best_config.json`, and run metadata |
| `file_inventory` | Relative path, suffix, and file size for generated result files |

The command does not start training or modify existing result files; it only
writes the requested workbook.

## Reproducible Execution Order

The recommended order for a complete run is the following. All commands are
run from the repository root. The commands below use the fixed random seeds
and split definitions recorded by the processing scripts.

The repository already contains the processed merged datasets whose
`training/validation/test` partition matches the manuscript:
`8,557 / 952 / 722` compounds. These release tables are provided without
generated molecular feature files. The main validation-containing table is
`data/raw_data/merge_data/Tox21_Toxcast/contain_valid/data_by_ecfp4_similarity.csv`,
and the corresponding no-validation table is
`data/raw_data/merge_data/Tox21_Toxcast/data.csv`. Therefore, users reproducing the reported
experiments can start directly from **Step 3, Generate molecular features**.
Steps 1 and 2 are only needed when independently rebuilding the release tables
and the ten-fold cluster cross-validation files from the raw inputs.

### 1. Build the merged 93-endpoint release

Starting from the raw Tox21 train/test files and the raw ToxCast auxiliary
table, generate the two release views:

```powershell
python tools\build_release_93_dataset.py `
  --project_root (Get-Location) `
  --train_ratio 0.9 `
  --random_seed 42
```

This writes:

```text
data/raw_data/merge_data/Tox21_Toxcast/data.csv
data/raw_data/merge_data/Tox21_Toxcast/contain_valid/data_by_ecfp4_similarity.csv
```

The script checks that the validation-containing view has exactly 8,557
training, 952 validation, and 722 test compounds.

### 2. Generate training cluster cross-validation files

Use only the 8,557-compound training group to create ten cluster-preserving
folds:

```powershell
python tools\build_training_cluster_cv.py `
  --project_root (Get-Location) `
  --n_folds 10 `
  --random_seed 42
```

Both validation-containing and no-validation fold views are written under
`data/raw_data/merge_data/Tox21_Toxcast/contain_valid/training_cluster_CV/`.

### 3. Generate molecular features

Generate the tabular features and full molecular graphs:

```powershell
powershell -ExecutionPolicy Bypass -File tools\build_tox21_toxcast_tabular_features.ps1 `
  -Python "C:\Users\85290\anaconda3\envs\graph_dgl\python.exe" `
  -ProjectRoot (Get-Location)

powershell -ExecutionPolicy Bypass -File tools\build_tox21_toxcast_graph_features.ps1 `
  -Python "C:\Users\85290\anaconda3\envs\graph_dgl\python.exe" `
  -ProjectRoot (Get-Location)
```

### 4. Train and prepare teacher profiles

For each selected tabular representation, first run the MLP HPO. Then run
the five MLP seeds over ten folds, the task-wise RF/SVM teachers, and the
teacher selection plus Platt scaling workflow:

```powershell
python hpo\optimize_teacher_mlp.py `
  --dataset Tox21_Toxcast `
  --feature descriptor `
  --feature_npz data\fp_descriptor_data\merge_data\Tox21_Toxcast\contain_valid\data_by_ecfp4_similarity_descriptor.npz `
  --fold_csv data\raw_data\merge_data\Tox21_Toxcast\contain_valid\training_cluster_CV\training_cluster_CV_0.csv `
  --output_dir hpo\Tox21_Toxcast\teacher_MLP_descriptor `
  --n_trials 30 `
  --seed 2026 `
  --device cuda

powershell -ExecutionPolicy Bypass -File teacher\Tox21_Toxcast\launch\run_mlp_teachers_from_hpo.ps1 `
  -Python "C:\Users\85290\anaconda3\envs\graph_dgl\python.exe" `
  -ProjectRoot (Get-Location) `
  -Device cuda

powershell -ExecutionPolicy Bypass -File teacher\Tox21_Toxcast\launch\run_rf_svm_teachers.ps1 `
  -Python "C:\Users\85290\anaconda3\envs\graph_dgl\python.exe" `
  -ProjectRoot (Get-Location)

powershell -ExecutionPolicy Bypass -File teacher\Tox21_Toxcast\launch\build_scaled_teacher_profile.ps1 `
  -Python "C:\Users\85290\anaconda3\envs\graph_dgl\python.exe" `
  -ProjectRoot (Get-Location)
```

In a complete run, repeat the teacher HPO step for the representations used by
the selection configuration: `Avalon`, `ECFP4`, `descriptor`,
`Avalon_descriptor`, and `ECFP4_descriptor`.

### 5. HPO and train BCE students

Run the MLP or graph BCE HPO, then pass its `best_config.json` to the repeated
seed runner. The main graph-student example is:

```powershell
python hpo\optimize_graph_student.py `
  --dataset Tox21_Toxcast `
  --model GIN `
  --tasks all_93 `
  --graph_dir data\graph_data\merge_data\Tox21_Toxcast\contain_valid\data_by_ecfp4_similarity `
  --output_dir hpo\Tox21_Toxcast\student_GIN_all93 `
  --objective_tasks tox21_12 `
  --n_trials 30 `
  --seed 2026 `
  --device cuda

python student\run_graph_student_from_hpo.py `
  --dataset Tox21_Toxcast `
  --model GIN `
  --tasks all_93 `
  --graph_dir data\graph_data\merge_data\Tox21_Toxcast\contain_valid\data_by_ecfp4_similarity `
  --best_config hpo\Tox21_Toxcast\student_GIN_all93\best_config.json `
  --output_root student `
  --split_id contain_valid_graph_bce `
  --loss_name mask_bce `
  --seeds 2026,2027,2028,2029,2030 `
  --device cuda
```

### 6. HPO and train soft-label students

Use the calibrated selected teacher profile and the BCE `best_config.json` as
inputs to soft-weight HPO. Then use the resulting soft HPO configuration for
the five-seed student ensemble:

```powershell
python hpo\optimize_student_soft_weights.py `
  --dataset Tox21_Toxcast `
  --student_type graph `
  --model GIN `
  --tasks all_93 `
  --graph_dir data\graph_data\merge_data\Tox21_Toxcast\contain_valid\data_by_ecfp4_similarity `
  --base_best_config hpo\Tox21_Toxcast\student_GIN_all93\best_config.json `
  --teacher_profile_dir "teacher\Tox21_Toxcast\profiles\feature=taskwise_selected_platt__model=Teacher__split=train_oof_auc__seed=mixed" `
  --output_dir hpo\Tox21_Toxcast\student_GIN_soft_full `
  --objective_tasks tox21_12 `
  --soft_profile_mode full `
  --device cuda

python student\run_graph_student_from_hpo.py `
  --dataset Tox21_Toxcast `
  --model GIN `
  --tasks all_93 `
  --graph_dir data\graph_data\merge_data\Tox21_Toxcast\contain_valid\data_by_ecfp4_similarity `
  --best_config hpo\Tox21_Toxcast\student_GIN_soft_full\best_config.json `
  --output_root student `
  --split_id contain_valid_graph_soft_full `
  --loss_name mask_bce_soft `
  --teacher_profile_dir "teacher\Tox21_Toxcast\profiles\feature=taskwise_selected_platt__model=Teacher__split=train_oof_auc__seed=mixed" `
  --soft_profile_mode full `
  --seeds 2026,2027,2028,2029,2030 `
  --device cuda
```

### 7. Run the ablations

Generate the nested task sets and run the task-count, profile-content, and
profile-reconstruction ablations using the commands in
[Student ablations](#student-ablations). Keep their output split IDs separate
from the main BCE and soft-label runs.

### 8. Aggregate available results

After any subset of the experiments has finished, generate the inspection
workbook:

```powershell
python tools\aggregate_results_to_xlsx.py `
  --project_root (Get-Location) `
  --output analysis\result_summaries\model_results_summary.xlsx
```

Because the aggregator is incremental and tolerant of missing outputs, it can
be run after HPO, after baseline training, or again after all ablations have
finished. The workbook reflects whatever result files are present at the time
of the command.
