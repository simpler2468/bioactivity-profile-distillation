import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teacher.common import save_json
from experiment_naming import parse_kv_name
from training.metrics import compute_multitask_metrics, summarize_metrics


def parse_profile_name(name):
    items = {}
    for part in name.split("__"):
        if "=" in part:
            key, value = part.split("=", 1)
            items[key] = value
    compact_items = parse_kv_name(name)
    items.update(compact_items)
    return items


def profile_key(row):
    return (str(row["feature"]), str(row["model"]), str(row["split"]), str(row["seed"]))


def load_profile_index(profiles_root):
    profiles = {}
    for profile_dir in Path(profiles_root).iterdir():
        if not profile_dir.is_dir():
            continue
        metadata = parse_profile_name(profile_dir.name)
        required = ["feature", "model", "split", "seed"]
        if all(key in metadata for key in required):
            profiles[(metadata["feature"], metadata["model"], metadata["split"], metadata["seed"])] = profile_dir
    return profiles


def select_candidates(metrics_df, metric, selection_split, mlp_seed, ml_seeds, mlp_splits=None, ml_splits=None):
    df = metrics_df[metrics_df["prediction_split"] == selection_split].copy()
    df = df[df["task_type"].fillna("classification") == "classification"]
    df = df[np.isfinite(df[metric])]

    ml_models = {"RF", "SVM", "rf", "svm"}
    mlp_models = {"MLP", "mlp"}

    def seed_variants(values):
        out = set()
        for value in values:
            text = str(value)
            out.add(text)
            if text.isdigit() and len(text) == 4 and text.startswith("20"):
                out.add(text[-2:])
        return out

    is_mlp = df["model"].astype(str).isin(mlp_models) & df["seed"].astype(str).isin(seed_variants([mlp_seed]))
    is_ml = df["model"].astype(str).isin(ml_models) & df["seed"].astype(str).isin(seed_variants(ml_seeds))
    if mlp_splits:
        is_mlp = is_mlp & df["split"].astype(str).isin(set(mlp_splits))
    if ml_splits:
        is_ml = is_ml & df["split"].astype(str).isin(set(ml_splits))
    df = df[is_mlp | is_ml].copy()

    if df.empty:
        raise ValueError("No candidate teacher metrics matched the selection filters.")

    df = df.sort_values(["task_index", metric], ascending=[True, False])
    selected = df.groupby("task_index", as_index=False).head(1).copy()
    selected = selected.sort_values("task_index").reset_index(drop=True)
    return selected


def assert_same_axis(base, current, keys, profile_dir):
    for key in keys:
        if np.issubdtype(base[key].dtype, np.floating) or np.issubdtype(current[key].dtype, np.floating):
            is_equal = np.allclose(base[key], current[key], equal_nan=True)
        else:
            is_equal = np.array_equal(base[key], current[key])
        if not is_equal:
            raise ValueError(f"{key} mismatch in {profile_dir}")


def build_selected_profile(selected, profiles_root, output_dir):
    profile_dirs = load_profile_index(profiles_root)
    loaded = {}
    base = None
    task_names = None

    for _, row in selected.iterrows():
        key = profile_key(row)
        if key not in profile_dirs:
            raise FileNotFoundError(f"Profile not found for {key}")
        if key not in loaded:
            arrays = np.load(profile_dirs[key] / "profile_arrays.npz", allow_pickle=True)
            loaded[key] = {name: arrays[name] for name in arrays.files}
            if base is None:
                base = loaded[key]
                task_names = base["task_names"].astype(str)
            else:
                assert_same_axis(
                    base,
                    loaded[key],
                    [
                        "task_names",
                        "train_oof_y",
                        "train_oof_row_index",
                        "valid_y",
                        "valid_row_index",
                        "test_y",
                        "test_row_index",
                    ],
                    profile_dirs[key],
                )

    if base is None:
        raise ValueError("No selected profiles were loaded.")

    payload = {
        "train_oof_raw_score": np.full_like(base["train_oof_raw_score"], np.nan, dtype=np.float32),
        "valid_raw_score": np.full_like(base["valid_raw_score"], np.nan, dtype=np.float32),
        "test_raw_score": np.full_like(base["test_raw_score"], np.nan, dtype=np.float32),
        "train_oof_y": base["train_oof_y"],
        "train_oof_row_index": base["train_oof_row_index"],
        "train_oof_smiles": base["train_oof_smiles"],
        "valid_y": base["valid_y"],
        "valid_row_index": base["valid_row_index"],
        "valid_smiles": base["valid_smiles"],
        "test_y": base["test_y"],
        "test_row_index": base["test_row_index"],
        "test_smiles": base["test_smiles"],
        "task_names": base["task_names"],
    }

    for _, row in selected.iterrows():
        task_index = int(row["task_index"])
        source = loaded[profile_key(row)]
        for split in ["train_oof", "valid", "test"]:
            payload[f"{split}_raw_score"][:, task_index] = source[f"{split}_raw_score"][:, task_index]

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "profile_arrays.npz", **payload)
    for key, value in payload.items():
        if key.endswith("_raw_score") or key.endswith("_y") or key.endswith("_row_index"):
            np.save(output_dir / f"{key}.npy", value)

    selected_out = selected.copy()
    selected_out["source_profile_dir"] = selected_out.apply(
        lambda row: str(profile_dirs[profile_key(row)]), axis=1
    )
    selected_out.to_csv(output_dir / "task_teacher_selection.csv", index=False)
    return payload, selected_out, task_names


def evaluate_payload(payload, task_names):
    rows = []
    summaries = {}
    for split, pred_key, y_key in [
        ("train_oof", "train_oof_raw_score", "train_oof_y"),
        ("valid", "valid_raw_score", "valid_y"),
        ("test", "test_raw_score", "test_y"),
    ]:
        metrics_df = compute_multitask_metrics(payload[pred_key], payload[y_key], task_types=None, task_names=task_names)
        metrics_df["prediction_split"] = split
        rows.append(metrics_df)
        summaries[split] = summarize_metrics(metrics_df)
    return pd.concat(rows, ignore_index=True), summaries


def parse_args():
    parser = argparse.ArgumentParser(description="Build a task-wise selected teacher profile by train OOF metric.")
    parser.add_argument("--dataset", default="Tox21_Toxcast")
    parser.add_argument("--profiles_root", required=True)
    parser.add_argument("--metrics_csv", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--metric", default="roc_auc")
    parser.add_argument("--selection_split", default="train_oof")
    parser.add_argument("--mlp_seed", default="ensemble5")
    parser.add_argument("--ml_seeds", default="2026")
    parser.add_argument("--mlp_splits", default=None, help="Optional comma-separated allowed MLP split ids.")
    parser.add_argument("--ml_splits", default=None, help="Optional comma-separated allowed RF/SVM split ids.")
    parser.add_argument("--profile_name", default="feature=taskwise_selected__model=Teacher__split=train_oof_auc__seed=mixed")
    return parser.parse_args()


def main():
    args = parse_args()
    metrics_df = pd.read_csv(args.metrics_csv)
    ml_seeds = [seed.strip() for seed in str(args.ml_seeds).split(",") if seed.strip()]
    mlp_splits = [split.strip() for split in str(args.mlp_splits).split(",") if split.strip()] if args.mlp_splits else None
    ml_splits = [split.strip() for split in str(args.ml_splits).split(",") if split.strip()] if args.ml_splits else None

    selected = select_candidates(metrics_df, args.metric, args.selection_split, args.mlp_seed, ml_seeds, mlp_splits, ml_splits)
    output_dir = Path(args.output_root) / args.dataset / "profiles" / args.profile_name
    payload, selected_out, task_names = build_selected_profile(selected, args.profiles_root, output_dir)
    metrics, summaries = evaluate_payload(payload, task_names)

    metrics_dir = Path(args.output_root) / args.dataset / "metrics" / args.profile_name
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for split, split_df in metrics.groupby("prediction_split"):
        split_df.to_csv(metrics_dir / f"{split}_per_task_metrics.csv", index=False)
    metrics.to_csv(metrics_dir / "all_per_task_metrics.csv", index=False)
    save_json(metrics_dir / "profile_metric_summary.json", summaries)
    save_json(
        output_dir / "profile_metadata.json",
        {
            "dataset": args.dataset,
            "selection_metric": args.metric,
            "selection_split": args.selection_split,
            "mlp_seed": args.mlp_seed,
            "ml_seeds": ml_seeds,
            "mlp_splits": mlp_splits,
            "ml_splits": ml_splits,
            "num_tasks": int(len(selected_out)),
            "metrics_dir": str(metrics_dir),
        },
    )
    print(f"Saved selected profile: {output_dir}")
    print(f"Saved selected metrics: {metrics_dir}")


if __name__ == "__main__":
    main()
