from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import TabularMLP
from teacher.map_old_profiles_to_new_project import fit_platt_one_task, per_task_metrics, metric_summary, save_json, sigmoid

FEATURE_SUFFIX = {
    "av": "Avalon",
    "ec4": "ECFP4",
    "desc": "descriptor",
    "avd": "Avalon_descriptor",
    "e4d": "ECFP4_descriptor",
}
FEATURES = ["av", "ec4", "desc", "avd", "e4d"]
SEEDS = [26, 27, 28, 29, 30]
FOLDS = list(range(10))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def predict_mlp(model: torch.nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    loader = DataLoader(TensorDataset(torch.tensor(x, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            preds.append(model(xb.to(device)).detach().cpu().numpy())
    return np.vstack(preds).astype(np.float32)


def build_model_from_config(config: dict, input_dim: int, output_dim: int) -> TabularMLP:
    hidden_dims = config.get("hidden_dims")
    if isinstance(hidden_dims, str) and hidden_dims:
        hidden_dims = [int(x) for x in hidden_dims.split(",")]
    elif not hidden_dims:
        hidden_dims = None
    return TabularMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        num_hidden_layers=config.get("num_hidden_layers", 3),
        shrink_ratio=config.get("shrink_ratio", 0.5),
        hidden_dims=hidden_dims,
        min_hidden_dim=config.get("min_hidden_dim"),
        dropout=config.get("dropout", 0.2),
        activation=config.get("activation", "leaky_relu"),
        norm_type=config.get("norm_type", "layer_norm"),
        batch_norm=config.get("batch_norm", False),
        negative_slope=config.get("negative_slope", 0.01),
    )


def find_run_dir(old_project: Path, feature: str, fold: int, seed: int) -> Path:
    root = old_project / "teacher" / "Tox21_Toxcast" / "runs"
    candidates = [
        root / f"f={feature}__m=mlp__sp=cve__fd={fold}__s={seed}",
        root / f"f={feature}__m=mlp__sp=contain_valid_ecfp4_similarity__fd={fold}__s={seed}",
    ]
    for p in candidates:
        if (p / "best.pt").exists() and (p / "run_config.json").exists():
            return p
    raise FileNotFoundError(f"Missing MLP checkpoint for feature={feature}, fold={fold}, seed={seed}")


def ensemble_mlp_predictions(old_project: Path, new_project: Path, feature: str, row_indices: np.ndarray, device: torch.device) -> np.ndarray:
    suffix = FEATURE_SUFFIX[feature]
    feature_npz = new_project / "data" / "fp_descriptor_data" / "merge_data" / "Tox21_Toxcast" / "contain_valid" / f"data_by_ecfp4_similarity_{suffix}.npz"
    data = np.load(feature_npz, allow_pickle=True)
    x = data["X"][row_indices]
    output_dim = int(data["y"].shape[1])
    preds = []
    for fold in FOLDS:
        for seed in SEEDS:
            run_dir = find_run_dir(old_project, feature, fold, seed)
            config = load_json(run_dir / "run_config.json")
            model = build_model_from_config(config, input_dim=x.shape[1], output_dim=output_dim).to(device)
            checkpoint = torch.load(run_dir / "best.pt", map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            preds.append(predict_mlp(model, x, batch_size=128, device=device))
    return np.mean(np.stack(preds, axis=0), axis=0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Impute missing mapped teacher profile rows with saved MLP ensemble predictions.")
    parser.add_argument("--old_project", default=r"E:\z_data\tox21_final2")
    parser.add_argument("--new_project", default=r"E:\z_data\tox21_clean")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    old_project = Path(args.old_project)
    new_project = Path(args.new_project)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    base = new_project / "teacher" / "Tox21_Toxcast"
    audit_dir = base / "mapping_audit"
    selected_dir = base / "profiles" / "f=taskwise_selected_mapped__m=tea__sp=oofauc__s=mixed"
    platt_dir = base / "profiles" / "f=taskwise_selected_platt_mapped__m=tea__sp=oofauc__s=mixed"
    selected_metric_dir = base / "metrics" / "f=taskwise_selected_mapped__m=tea__sp=oofauc__s=mixed"
    platt_metric_dir = base / "metrics" / "f=taskwise_selected_platt_mapped__m=tea__sp=oofauc__s=mixed"

    missing = pd.read_csv(audit_dir / "canonical_mapping_missing_rows.csv").drop_duplicates(["row_index"]).copy()
    if missing.empty:
        print("No missing mapped rows to impute.")
        return
    row_indices = missing["row_index"].astype(int).to_numpy()

    selected_npz = np.load(selected_dir / "profile_arrays.npz", allow_pickle=True)
    platt_npz = np.load(platt_dir / "profile_arrays.npz", allow_pickle=True)
    selected_arrays = {k: selected_npz[k] for k in selected_npz.files}
    platt_arrays = {k: platt_npz[k] for k in platt_npz.files}
    task_names = [str(x) for x in selected_arrays["task_names"].tolist()]

    selection = pd.read_csv(selected_dir / "task_teacher_selection.csv")
    candidate_auc = pd.read_csv(base / "metrics" / "mapped_candidate_train_oof_auc_by_task.csv")
    mlp_candidate_auc = candidate_auc[candidate_auc["model"] == "mlp"].copy()

    test_row_index = selected_arrays["test_row_index"].astype(int)
    test_pos_by_row = {int(r): i for i, r in enumerate(test_row_index)}

    mlp_preds: Dict[str, np.ndarray] = {}
    for feature in FEATURES:
        print(f"Predict missing rows with MLP ensemble: feature={feature}, rows={len(row_indices)}")
        mlp_preds[feature] = ensemble_mlp_predictions(old_project, new_project, feature, row_indices, device)

    fallback_platt = {}
    for feature in FEATURES:
        candidate_name = f"f={feature}__m=mlp__sp=mapped_cve__s=ensemble5"
        z = np.load(base / "profiles" / candidate_name / "profile_arrays.npz", allow_pickle=True)
        train_score = z["train_oof_raw_score"]
        train_y = z["train_oof_y"]
        for task_idx, task_name in enumerate(task_names):
            fit = fit_platt_one_task(train_score[:, task_idx], train_y[:, task_idx])
            fallback_platt[(feature, task_idx)] = fit

    impute_rows = []
    for miss_pos, row in enumerate(missing.itertuples(index=False)):
        row_index = int(row.row_index)
        test_pos = test_pos_by_row[row_index]
        for task_idx, task_name in enumerate(task_names):
            chosen = selection.loc[selection["task_index"] == task_idx].iloc[0]
            selected_model = str(chosen["selected_model"])
            selected_feature = str(chosen["selected_feature"])
            if selected_model == "mlp" and selected_feature in FEATURES:
                fallback_feature = selected_feature
                fallback_reason = "selected_teacher_is_mlp"
            else:
                sub = mlp_candidate_auc[mlp_candidate_auc["task_index"] == task_idx]
                sub = sub[np.isfinite(sub["train_oof_roc_auc"])]
                if sub.empty:
                    fallback_feature = "desc"
                    fallback_reason = "selected_teacher_non_mlp_default_desc"
                else:
                    best = sub.sort_values("train_oof_roc_auc", ascending=False).iloc[0]
                    fallback_feature = str(best["feature"])
                    fallback_reason = "selected_teacher_non_mlp_best_mlp_oof_auc"
            raw_value = float(mlp_preds[fallback_feature][miss_pos, task_idx])
            fit = fallback_platt[(fallback_feature, task_idx)]
            calibrated = float(fit["a"] * raw_value + fit["b"])
            selected_arrays["test_raw_score"][test_pos, task_idx] = raw_value
            platt_arrays["test_raw_score"][test_pos, task_idx] = calibrated
            if "test_prob" in platt_arrays:
                platt_arrays["test_prob"][test_pos, task_idx] = float(sigmoid(np.asarray([calibrated]))[0])
            impute_rows.append({
                "row_index": row_index,
                "test_position": int(test_pos),
                "SMILES": row.SMILES,
                "task_index": task_idx,
                "task_name": task_name,
                "selected_model": selected_model,
                "selected_feature": selected_feature,
                "fallback_feature": fallback_feature,
                "fallback_reason": fallback_reason,
                "raw_logit": raw_value,
                "platt_a": fit["a"],
                "platt_b": fit["b"],
                "platt_logit": calibrated,
            })

    np.savez_compressed(selected_dir / "profile_arrays.npz", **selected_arrays)
    np.save(selected_dir / "test_raw_score.npy", selected_arrays["test_raw_score"])
    np.savez_compressed(platt_dir / "profile_arrays.npz", **platt_arrays)
    np.save(platt_dir / "test_raw_score.npy", platt_arrays["test_raw_score"])
    np.save(platt_dir / "test_y.npy", platt_arrays["test_y"])

    impute_df = pd.DataFrame(impute_rows)
    impute_df.to_csv(audit_dir / "mlp_imputed_missing_rows.csv", index=False)

    selected_meta = load_json(selected_dir / "profile_metadata.json")
    platt_meta = load_json(platt_dir / "profile_metadata.json")
    impute_meta = {
        "method": "MLP ensemble fallback for rows not mapped from old profiles",
        "n_rows": int(len(row_indices)),
        "row_indices": [int(x) for x in row_indices],
        "features": FEATURES,
        "folds": FOLDS,
        "seeds": SEEDS,
        "audit_file": str(audit_dir / "mlp_imputed_missing_rows.csv"),
    }
    selected_meta["missing_row_imputation"] = impute_meta
    platt_meta["missing_row_imputation"] = impute_meta
    save_json(selected_dir / "profile_metadata.json", selected_meta)
    save_json(platt_dir / "profile_metadata.json", platt_meta)

    selected_test_metrics = per_task_metrics(selected_arrays["test_raw_score"], selected_arrays["test_y"], task_names)
    platt_test_metrics = per_task_metrics(platt_arrays["test_raw_score"], platt_arrays["test_y"], task_names)
    selected_metric_dir.mkdir(parents=True, exist_ok=True)
    platt_metric_dir.mkdir(parents=True, exist_ok=True)
    selected_test_metrics.to_csv(selected_metric_dir / "test_per_task_metrics.csv", index=False)
    platt_test_metrics.to_csv(platt_metric_dir / "test_per_task_metrics.csv", index=False)

    for metric_dir, arrays in [(selected_metric_dir, selected_arrays), (platt_metric_dir, platt_arrays)]:
        summaries = {}
        for split_name, score_key, y_key in [
            ("train_oof", "train_oof_raw_score", "train_oof_y"),
            ("valid", "valid_raw_score", "valid_y"),
            ("test", "test_raw_score", "test_y"),
        ]:
            metrics = per_task_metrics(arrays[score_key], arrays[y_key], task_names)
            metrics.to_csv(metric_dir / f"{split_name}_per_task_metrics.csv", index=False)
            summaries[split_name] = metric_summary(metrics, task_names)
        summaries["missing_row_imputation"] = impute_meta
        save_json(metric_dir / "profile_metric_summary.json", summaries)

    print(f"Imputed missing rows: {len(row_indices)}")
    print(f"Audit: {audit_dir / 'mlp_imputed_missing_rows.csv'}")
    print("Platt test summary:")
    print(json.dumps(metric_summary(platt_test_metrics, task_names), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
