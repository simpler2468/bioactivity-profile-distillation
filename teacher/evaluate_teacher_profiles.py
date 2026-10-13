import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teacher.common import save_json, write_metric_table
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


def evaluate_profile(profile_dir, output_dir=None):
    profile_dir = Path(profile_dir)
    arrays_path = profile_dir / "profile_arrays.npz"
    if not arrays_path.exists():
        raise FileNotFoundError(arrays_path)
    arrays = np.load(arrays_path, allow_pickle=True)
    task_names = arrays["task_names"].astype(str).tolist()
    metadata = parse_profile_name(profile_dir.name)

    rows = []
    summaries = {}
    split_specs = [
        ("train_oof", "train_oof_raw_score", "train_oof_y"),
        ("valid", "valid_raw_score", "valid_y"),
        ("test", "test_raw_score", "test_y"),
    ]
    output_dir = Path(output_dir) if output_dir is not None else profile_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, pred_key, y_key in split_specs:
        pred = arrays[pred_key]
        y = arrays[y_key]
        if len(pred) == 0:
            continue
        metrics_df = compute_multitask_metrics(pred, y, task_types=None, task_names=task_names)
        summary = summarize_metrics(metrics_df)
        summary.update(metadata)
        summary["prediction_split"] = split_name
        summaries[split_name] = summary
        write_metric_table(
            output_dir / f"{split_name}_per_task_metrics.csv",
            metrics_df,
            {**metadata, "prediction_split": split_name},
        )
        enriched = metrics_df.copy()
        for key, value in metadata.items():
            enriched[key] = value
        enriched["prediction_split"] = split_name
        rows.append(enriched)

    summary_path = output_dir / "profile_metric_summary.json"
    save_json(summary_path, summaries)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(), summaries


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate teacher profile per-task metrics.")
    parser.add_argument("--profile_dir", default=None)
    parser.add_argument("--profiles_root", default=None)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--dataset", default="Tox21_Toxcast")
    return parser.parse_args()


def main():
    args = parse_args()
    if bool(args.profile_dir) == bool(args.profiles_root):
        raise ValueError("Provide exactly one of --profile_dir or --profiles_root.")

    if args.profile_dir:
        metrics_df, summaries = evaluate_profile(args.profile_dir, output_dir=args.output_root)
        print(f"Evaluated {args.profile_dir}: rows={len(metrics_df)}")
        return

    profiles_root = Path(args.profiles_root)
    output_root = Path(args.output_root) if args.output_root else profiles_root.parent / "metrics"
    output_root.mkdir(parents=True, exist_ok=True)
    all_rows = []
    summary_rows = []
    for profile_dir in sorted(path for path in profiles_root.iterdir() if path.is_dir()):
        profile_output = output_root / profile_dir.name
        metrics_df, summaries = evaluate_profile(profile_dir, output_dir=profile_output)
        if not metrics_df.empty:
            all_rows.append(metrics_df)
        for split_name, summary in summaries.items():
            row = dict(summary)
            row["prediction_split"] = split_name
            row["profile_dir"] = str(profile_dir)
            summary_rows.append(row)
        print(f"Evaluated {profile_dir.name}: rows={len(metrics_df)}")

    if all_rows:
        all_metrics = pd.concat(all_rows, ignore_index=True)
        all_metrics.to_csv(output_root / "all_profile_per_task_metrics.csv", index=False)

        for split_name, split_df in all_metrics.groupby("prediction_split"):
            split_df.to_csv(output_root / f"{split_name}_auc_by_task.csv", index=False)
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(output_root / "model_feature_summary.csv", index=False)
    print(f"Saved metrics under: {output_root}")


if __name__ == "__main__":
    main()
