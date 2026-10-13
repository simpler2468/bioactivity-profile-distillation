import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.graph_dataset import GraphDataset, graph_collate_fn
from experiment_naming import parse_kv_name, short_kv_id
from models import PlainGIN, TabularMLP
from student.run_mlp_student import TOX21_12_TASKS
from teacher.common import load_feature_npz, save_json, write_metric_table
from training.metrics import compute_multitask_metrics, summarize_metrics

FEATURE_FILES = {
    "av": ("Avalon", "data/fp_descriptor_data/merge_data/Tox21_Toxcast/contain_valid/data_by_ecfp4_similarity_Avalon.npz"),
    "ec4": ("ECFP4", "data/fp_descriptor_data/merge_data/Tox21_Toxcast/contain_valid/data_by_ecfp4_similarity_ECFP4.npz"),
    "desc": ("descriptor", "data/fp_descriptor_data/merge_data/Tox21_Toxcast/contain_valid/data_by_ecfp4_similarity_descriptor.npz"),
    "avd": ("Avalon_descriptor", "data/fp_descriptor_data/merge_data/Tox21_Toxcast/contain_valid/data_by_ecfp4_similarity_Avalon_descriptor.npz"),
    "e4d": ("ECFP4_descriptor", "data/fp_descriptor_data/merge_data/Tox21_Toxcast/contain_valid/data_by_ecfp4_similarity_ECFP4_descriptor.npz"),
}


def parse_int_tuple(value):
    if value is None or value == "":
        return None
    return tuple(int(x) for x in str(value).split(",") if str(x).strip())


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


def build_mlp(config, input_dim, output_dim):
    return TabularMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        num_hidden_layers=config.get("num_hidden_layers", 3),
        shrink_ratio=config.get("shrink_ratio", 0.5),
        hidden_dims=[int(x) for x in str(config.get("hidden_dims", "")).split(",") if x] if config.get("hidden_dims") else None,
        min_hidden_dim=config.get("min_hidden_dim"),
        dropout=config.get("dropout", 0.2),
        activation=config.get("activation", "leaky_relu"),
        norm_type=config.get("norm_type", "layer_norm"),
        batch_norm=config.get("batch_norm", False),
        negative_slope=config.get("negative_slope", 0.01),
    )


def build_gin(config, output_dim):
    return PlainGIN(
        output_dim=output_dim,
        hidden_dim=config.get("hidden_dim", 256),
        num_layers=config.get("num_layers", 3),
        dropout=config.get("dropout", 0.2),
        norm_type=config.get("norm_type", "batch_norm"),
        readout=config.get("readout", "mean"),
        mlp_hidden_dims=parse_int_tuple(config.get("mlp_hidden_dims")),
    )


def save_predictions(path, split_name, row_index, smiles, y_true, logits, task_names, task_indices):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        split_name=np.asarray(split_name, dtype=object),
        row_index=np.asarray(row_index, dtype=np.int64),
        smiles=np.asarray(smiles, dtype=object),
        y_true=np.asarray(y_true, dtype=np.float32),
        logits=np.asarray(logits, dtype=np.float32),
        prob=sigmoid(np.asarray(logits, dtype=np.float32)).astype(np.float32),
        task_names=np.asarray(task_names, dtype=object),
        selected_task_indices=np.asarray(task_indices, dtype=np.int64),
    )


def evaluate_mlp_run(run_dir, output_root, device):
    out_run_dir = output_root / "runs" / run_dir.name
    out_metric_dir = output_root / "metrics" / run_dir.name
    if (out_run_dir / "test_predictions.npz").exists() and (out_metric_dir / "test_per_task_metrics.csv").exists():
        print(f"Skip existing current eval {run_dir.name}", flush=True)
        return run_dir.name
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    meta = parse_kv_name(run_dir.name)
    feature_name, feature_file = FEATURE_FILES[meta["feature"]]
    data = load_feature_npz(PROJECT_ROOT / feature_file)
    task_indices = np.asarray(config["selected_task_indices"], dtype=np.int64)
    task_names = list(config["task_names"])
    model = build_mlp(config, data["X"].shape[1], len(task_indices)).to(device)
    load_checkpoint(model, run_dir / "best.pt", device)
    out_run_dir = output_root / "runs" / run_dir.name
    out_metric_dir = output_root / "metrics" / run_dir.name
    summaries = {}
    for group, split_name in [("training", "train"), ("valid", "valid"), ("test", "test")]:
        idx = np.where(data["groups"] == group)[0].astype(np.int64)
        loader = DataLoader(TensorDataset(torch.tensor(data["X"][idx], dtype=torch.float32)), batch_size=512, shuffle=False)
        logits = []
        with torch.no_grad():
            for (xb,) in loader:
                logits.append(model(xb.to(device)).detach().cpu().numpy())
        logits = np.vstack(logits).astype(np.float32)
        y = data["y"][idx][:, task_indices]
        metrics = compute_multitask_metrics(logits, y, task_types=None, task_names=task_names)
        summary = summarize_metrics(metrics)
        summaries[split_name] = summary
        save_predictions(out_run_dir / f"{split_name}_predictions.npz", split_name, idx, data["smiles"][idx], y, logits, task_names, task_indices)
        write_metric_table(out_metric_dir / f"{split_name}_per_task_metrics.csv", metrics, {"dataset":"Tox21_Toxcast", "source_run": run_dir.name, "prediction_split": split_name, "current_test_eval": True})
    save_json(out_metric_dir / "profile_metric_summary.json", summaries)
    return run_dir.name


def move_graph_batch(batch, device):
    moved = {}
    for k, v in batch.items():
        if hasattr(v, "to") and k.endswith("graph"):
            moved[k] = v.to(device) if v is not None else None
        elif torch.is_tensor(v):
            moved[k] = v.to(device)
        else:
            moved[k] = v
    return moved


def evaluate_gin_run(run_dir, output_root, graph_dir, device):
    out_run_dir = output_root / "runs" / run_dir.name
    out_metric_dir = output_root / "metrics" / run_dir.name
    if (out_run_dir / "test_predictions.npz").exists() and (out_metric_dir / "test_per_task_metrics.csv").exists():
        print(f"Skip existing current eval {run_dir.name}", flush=True)
        return run_dir.name
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    task_indices = np.asarray(config["selected_task_indices"], dtype=np.int64)
    task_names = list(config["task_names"])
    model = build_gin(config, len(task_indices)).to(device)
    load_checkpoint(model, run_dir / "best.pt", device)
    out_run_dir = output_root / "runs" / run_dir.name
    out_metric_dir = output_root / "metrics" / run_dir.name
    summaries = {}
    for group, split_name in [("training", "train"), ("valid", "valid"), ("test", "test")]:
        ds = GraphDataset(graph_dir, split=group, load_brics=False, load_murcko=False, load_fg=False)
        loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0, collate_fn=graph_collate_fn)
        logits_list, y_list, row_index_list, smiles_list = [], [], [], []
        with torch.no_grad():
            for batch in loader:
                batch = move_graph_batch(batch, device)
                logits = model(batch["graph"])
                logits_list.append(logits.detach().cpu().numpy())
                y_list.append(batch["y"][:, task_indices].detach().cpu().numpy())
                row_index_list.append(batch["row_index"].detach().cpu().numpy())
                smiles_list.extend(batch["smiles"])
        logits = np.vstack(logits_list).astype(np.float32)
        y = np.vstack(y_list).astype(np.float32)
        row_index = np.concatenate(row_index_list).astype(np.int64)
        metrics = compute_multitask_metrics(logits, y, task_types=None, task_names=task_names)
        summary = summarize_metrics(metrics)
        summaries[split_name] = summary
        save_predictions(out_run_dir / f"{split_name}_predictions.npz", split_name, row_index, smiles_list, y, logits, task_names, task_indices)
        write_metric_table(out_metric_dir / f"{split_name}_per_task_metrics.csv", metrics, {"dataset":"Tox21_Toxcast", "source_run": run_dir.name, "prediction_split": split_name, "current_test_eval": True})
    save_json(out_metric_dir / "profile_metric_summary.json", summaries)
    return run_dir.name


def evaluate_ensembles(output_root, dataset="Tox21_Toxcast"):
    runs_root = output_root / "runs"
    ens_root = output_root / "ensembles"
    metrics_root = output_root / "metrics"
    groups = {}
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        meta = parse_kv_name(run_dir.name)
        required = ["feature", "model", "tasks", "loss", "split", "seed"]
        if not all(k in meta for k in required):
            continue
        key = (meta["feature"], meta["model"], meta["tasks"], meta["loss"], meta["split"])
        groups.setdefault(key, {})[meta["seed"]] = run_dir
    seeds = ["26", "27", "28", "29", "30"]
    summary_rows = []
    seed_stat_rows = []
    for key, seed_dirs in sorted(groups.items()):
        if not all(seed in seed_dirs for seed in seeds):
            continue
        feature, model, tasks, loss, split = key
        eid = short_kv_id(feature=feature, model=model, tasks=tasks, loss=loss, split=split, seed="ensemble5")
        ens_dir = ens_root / eid
        metric_dir = metrics_root / eid
        ens_dir.mkdir(parents=True, exist_ok=True)
        metric_dir.mkdir(parents=True, exist_ok=True)
        split_summaries = {}
        for split_name in ["train", "valid", "test"]:
            loaded = []
            base = None
            for seed in seeds:
                pred = np.load(seed_dirs[seed] / f"{split_name}_predictions.npz", allow_pickle=True)
                item = {k: pred[k] for k in pred.files}
                if base is None:
                    base = item
                else:
                    for axis in ["row_index", "y_true", "task_names", "selected_task_indices"]:
                        if axis == "y_true":
                            ok = np.allclose(base[axis], item[axis], equal_nan=True)
                        else:
                            ok = np.array_equal(base[axis], item[axis])
                        if not ok:
                            raise ValueError(f"{axis} mismatch in {seed_dirs[seed]}")
                loaded.append(item["logits"].astype(np.float32))
            logits = np.mean(np.stack(loaded, axis=0), axis=0).astype(np.float32)
            task_names = base["task_names"].astype(str).tolist()
            metrics = compute_multitask_metrics(logits, base["y_true"].astype(np.float32), task_types=None, task_names=task_names)
            split_summaries[split_name] = summarize_metrics(metrics)
            save_predictions(ens_dir / f"{split_name}_predictions.npz", split_name, base["row_index"], base["smiles"], base["y_true"], logits, task_names, base["selected_task_indices"])
            write_metric_table(metric_dir / f"{split_name}_per_task_metrics.csv", metrics, {"dataset":dataset, "feature":feature, "model":model, "tasks":tasks, "loss":loss, "split":split, "seed":"ensemble5", "prediction_split":split_name, "current_test_eval": True})
        save_json(metric_dir / "profile_metric_summary.json", split_summaries)
        save_json(ens_dir / "ensemble_metadata.json", {"ensemble_id": eid, "member_seeds": seeds, "logit_aggregation":"mean over member logits", "current_test_eval": True})
        # Summary from test ensemble.
        test_metrics = pd.read_csv(metric_dir / "test_per_task_metrics.csv")
        summary_rows.append({
            "feature": feature,
            "model": model,
            "tasks": tasks,
            "loss": loss,
            "split": split,
            "test_tox21_12_ensemble_auc": float(pd.to_numeric(test_metrics[test_metrics["task_name"].isin(TOX21_12_TASKS)]["roc_auc"], errors="coerce").mean()),
            "test_all_available_ensemble_auc": float(pd.to_numeric(test_metrics["roc_auc"], errors="coerce").mean()),
        })
        vals = []
        for seed in seeds:
            m = pd.read_csv(metrics_root / seed_dirs[seed].name / "test_per_task_metrics.csv")
            vals.append(float(pd.to_numeric(m[m["task_name"].isin(TOX21_12_TASKS)]["roc_auc"], errors="coerce").mean()))
        vals = np.asarray(vals, dtype=float)
        med = float(np.median(vals))
        seed_stat_rows.append({
            "feature_code": feature,
            "model_code": model,
            "tasks_code": tasks,
            "loss_code": loss,
            "split_code": split,
            "seed_values": ";".join(f"{x:.6f}" for x in vals),
            "n_seeds": len(vals),
            "seed_mean_test_tox21_12_auc": float(vals.mean()),
            "seed_std_test_tox21_12_auc": float(vals.std(ddof=1)),
            "seed_median_test_tox21_12_auc": med,
            "seed_mad_test_tox21_12_auc": float(np.median(np.abs(vals - med))),
            "ensemble_test_tox21_12_auc": summary_rows[-1]["test_tox21_12_ensemble_auc"],
            "ensemble_test_all_available_auc": summary_rows[-1]["test_all_available_ensemble_auc"],
        })
    pd.DataFrame(summary_rows).to_csv(metrics_root / "student_current_test_summary.csv", index=False)
    pd.DataFrame(seed_stat_rows).to_csv(metrics_root / "student_current_test_seed_stats.csv", index=False)
    bce_rows = [row for row in summary_rows if row.get("loss") == "bce"]
    bce_seed_rows = [row for row in seed_stat_rows if row.get("loss_code") == "bce"]
    if bce_rows:
        pd.DataFrame(bce_rows).to_csv(metrics_root / "bce_baseline_current_test_summary.csv", index=False)
    if bce_seed_rows:
        pd.DataFrame(bce_seed_rows).to_csv(metrics_root / "bce_baseline_current_test_seed_stats.csv", index=False)
    soft_rows = [row for row in summary_rows if row.get("loss") == "soft"]
    soft_seed_rows = [row for row in seed_stat_rows if row.get("loss_code") == "soft"]
    if soft_rows:
        pd.DataFrame(soft_rows).to_csv(metrics_root / "soft_current_test_summary.csv", index=False)
    if soft_seed_rows:
        pd.DataFrame(soft_seed_rows).to_csv(metrics_root / "soft_current_test_seed_stats.csv", index=False)
    return pd.DataFrame(seed_stat_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_root", default="student/Tox21_Toxcast/runs")
    parser.add_argument("--output_root", default="student/Tox21_Toxcast/current_eval")
    parser.add_argument("--graph_dir", default="data/graph_data/merge_data/Tox21_Toxcast/contain_valid/data_by_ecfp4_similarity")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    runs_root = PROJECT_ROOT / args.runs_root
    output_root = PROJECT_ROOT / args.output_root
    graph_dir = PROJECT_ROOT / args.graph_dir
    output_root.mkdir(parents=True, exist_ok=True)
    selected = []
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        meta = parse_kv_name(run_dir.name)
        if meta.get("loss") == "bce" and meta.get("split") not in {"cve", "cvg", "tcab"}:
            continue
        if meta.get("loss") == "soft" and not str(meta.get("split", "")).startswith("ax"):
            continue
        if meta.get("loss") not in {"bce", "soft"}:
            continue
        if str(meta.get("seed", "")).startswith("ensemble"):
            continue
        if meta.get("model") == "mlp":
            print(f"Evaluating MLP {run_dir.name}", flush=True)
            selected.append(evaluate_mlp_run(run_dir, output_root, device))
        elif meta.get("model") == "gin":
            print(f"Evaluating eGIN {run_dir.name}", flush=True)
            selected.append(evaluate_gin_run(run_dir, output_root, graph_dir, device))
    df = evaluate_ensembles(output_root)
    print(df.sort_values(["model_code", "feature_code", "tasks_code"]).to_string(index=False))


if __name__ == "__main__":
    main()

