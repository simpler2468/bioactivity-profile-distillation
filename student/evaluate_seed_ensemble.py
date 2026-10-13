import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teacher.common import save_json, write_metric_table
from experiment_naming import parse_kv_name, short_kv_id
from training.metrics import compute_multitask_metrics, summarize_metrics


TOX21_12_TASKS = [
    "NR-AhR",
    "NR-AR",
    "NR-AR-LBD",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
]


def parse_run_name(name):
    return parse_kv_name(name)


def ensemble_id(feature, model, tasks, loss, split, seed_name):
    return short_kv_id(feature=feature, model=model, tasks=tasks, loss=loss, split=split, seed=seed_name)


def load_prediction(path):
    data = np.load(path, allow_pickle=True)
    return {
        "split_name": str(np.asarray(data["split_name"]).item()),
        "row_index": data["row_index"].astype(np.int64),
        "smiles": data["smiles"].astype(str),
        "y_true": data["y_true"].astype(np.float32),
        "logits": data["logits"].astype(np.float32),
        "task_names": data["task_names"].astype(str).tolist(),
        "selected_task_indices": data["selected_task_indices"].astype(np.int64),
    }


def save_ensemble_prediction(path, base, logits):
    prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        split_name=np.asarray(base["split_name"], dtype=object),
        row_index=base["row_index"],
        smiles=np.asarray(base["smiles"], dtype=object),
        y_true=base["y_true"],
        logits=logits.astype(np.float32),
        prob=prob.astype(np.float32),
        task_names=np.asarray(base["task_names"], dtype=object),
        selected_task_indices=base["selected_task_indices"],
    )


def find_groups(runs_root, pattern=None, models=None):
    model_set = None if models is None else {str(model) for model in models}
    groups = {}
    for run_dir in sorted(Path(runs_root).iterdir()):
        if not run_dir.is_dir():
            continue
        meta = parse_run_name(run_dir.name)
        required = ["feature", "model", "tasks", "loss", "split", "seed"]
        if not all(key in meta for key in required):
            continue
        if model_set is not None and meta["model"] not in model_set:
            continue
        if pattern and not re.search(pattern, run_dir.name):
            continue
        key = (meta["feature"], meta["model"], meta["tasks"], meta["loss"], meta["split"])
        groups.setdefault(key, {})[str(meta["seed"])] = run_dir
    return groups


def assert_same_prediction_axis(base, current, run_dir):
    for key in ["row_index", "y_true", "task_names", "selected_task_indices"]:
        a = base[key]
        b = current[key]
        if np.issubdtype(np.asarray(a).dtype, np.floating):
            ok = np.allclose(a, b, equal_nan=True)
        else:
            ok = np.array_equal(a, b)
        if not ok:
            raise ValueError(f"{key} mismatch in {run_dir}")


def evaluate_group(dataset, group_key, seed_dirs, output_root, seeds):
    feature, model, tasks, loss, split = group_key
    seed_name = f"ensemble{len(seeds)}"
    eid = ensemble_id(feature, model, tasks, loss, split, seed_name)
    out_dir = Path(output_root) / dataset / "ensembles" / eid
    metrics_dir = Path(output_root) / dataset / "metrics" / eid
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}
    all_metric_rows = []
    for split_name in ["train", "valid", "test"]:
        loaded = []
        base = None
        for seed in seeds:
            if str(seed) not in seed_dirs:
                raise FileNotFoundError(f"Missing seed={seed} for {group_key}")
            run_dir = seed_dirs[str(seed)]
            pred_path = run_dir / f"{split_name}_predictions.npz"
            if not pred_path.exists():
                raise FileNotFoundError(pred_path)
            pred = load_prediction(pred_path)
            if base is None:
                base = pred
            else:
                assert_same_prediction_axis(base, pred, run_dir)
            loaded.append(pred["logits"])

        logits = np.nanmean(np.stack(loaded, axis=0), axis=0).astype(np.float32)
        save_ensemble_prediction(out_dir / f"{split_name}_predictions.npz", base, logits)
        metrics_df = compute_multitask_metrics(logits, base["y_true"], task_types=None, task_names=base["task_names"])
        metadata = {
            "dataset": dataset,
            "feature": feature,
            "model": model,
            "tasks": tasks,
            "loss": loss,
            "split": split,
            "seed": seed_name,
            "member_seeds": ",".join(str(seed) for seed in seeds),
            "prediction_split": split_name,
        }
        write_metric_table(metrics_dir / f"{split_name}_per_task_metrics.csv", metrics_df, metadata)
        enriched = metrics_df.copy()
        for key, value in metadata.items():
            enriched[key] = value
        all_metric_rows.append(enriched)
        summaries[split_name] = summarize_metrics(metrics_df)

    all_metrics = pd.concat(all_metric_rows, ignore_index=True)
    all_metrics.to_csv(metrics_dir / "all_per_task_metrics.csv", index=False)
    save_json(metrics_dir / "profile_metric_summary.json", summaries)
    save_json(
        out_dir / "ensemble_metadata.json",
        {
            "ensemble_id": eid,
            "feature": feature,
            "model": model,
            "tasks": tasks,
            "loss": loss,
            "split": split,
            "member_seeds": seeds,
            "logit_aggregation": "mean over member logits",
            "metrics_dir": str(metrics_dir),
        },
    )
    return all_metrics


def summarize_all(all_rows, output_path):
    rows = []
    for (feature, model, tasks, loss, prediction_split), df in all_rows.groupby(
        ["feature", "model", "tasks", "loss", "prediction_split"]
    ):
        rows.append(
            {
                "feature": feature,
                "model": model,
                "tasks": tasks,
                "loss": loss,
                "prediction_split": prediction_split,
                "mean_auc": float(pd.to_numeric(df["roc_auc"], errors="coerce").mean()),
                "tox21_12_auc": float(
                    pd.to_numeric(df[df["task_name"].isin(TOX21_12_TASKS)]["roc_auc"], errors="coerce").mean()
                ),
                "mean_ap": float(pd.to_numeric(df["average_precision"], errors="coerce").mean()),
                "valid_auc_tasks": int(pd.to_numeric(df["roc_auc"], errors="coerce").notna().sum()),
            }
        )
    summary = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Average seed logits first, then evaluate student ensembles.")
    parser.add_argument("--dataset", default="Tox21_Toxcast")
    parser.add_argument("--runs_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--seeds", default="2026,2027,2028,2029,2030")
    parser.add_argument("--pattern", default=None, help="Optional regex filter on run directory names.")
    parser.add_argument("--models", default=None, help="Optional comma-separated model filter, e.g. MLP,GIN.")
    parser.add_argument("--summary_name", default="student_seed_ensemble_summary.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = [seed.strip() for seed in str(args.seeds).split(",") if seed.strip()]
    models = None
    if args.models:
        models = [item.strip() for item in str(args.models).split(",") if item.strip()]
    groups = find_groups(args.runs_root, pattern=args.pattern, models=models)
    all_rows = []
    for key, seed_dirs in sorted(groups.items()):
        if not all(seed in seed_dirs for seed in seeds):
            continue
        metrics = evaluate_group(args.dataset, key, seed_dirs, args.output_root, seeds)
        all_rows.append(metrics)
        print(f"Evaluated seed ensemble: {key}")
    if not all_rows:
        raise ValueError("No complete seed groups found.")
    all_metrics = pd.concat(all_rows, ignore_index=True)
    summary_path = Path(args.output_root) / args.dataset / "metrics" / args.summary_name
    summary = summarize_all(all_metrics, summary_path)
    print(f"Saved ensemble summary: {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
