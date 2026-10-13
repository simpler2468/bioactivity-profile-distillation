import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.graph_dataset import GraphDataset, graph_collate_fn
from models import PlainGIN
from student.distillation_targets import DistillationTargetProvider, validate_distillation_args
from student.run_mlp_student import TOX21_12_TASKS, select_tasks, select_tasks_from_file, student_run_id
from teacher.common import save_json, write_metric_table
from training.losses import MultiTaskLossComputer, compute_pos_weight_from_labels
from training.metrics import compute_multitask_metrics, summarize_metrics


def set_seed(seed):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_epoch_progress(epoch, epochs, train_loss, valid_auc=None, best_auc=None, elapsed=None):
    parts = [f"Epoch {epoch:03d}/{epochs:03d}", f"train_loss={train_loss:.4f}"]
    if valid_auc is not None and np.isfinite(valid_auc):
        parts.append(f"valid_auc={valid_auc:.4f}")
    if best_auc is not None and np.isfinite(best_auc):
        parts.append(f"best_auc={best_auc:.4f}")
    if elapsed is not None:
        parts.append(f"elapsed={elapsed:.1f}s")
    return " | ".join(parts)


def parse_int_tuple(value):
    if isinstance(value, (list, tuple)):
        return tuple(int(x) for x in value)
    return tuple(int(x) for x in str(value).split(",") if str(x).strip())


def move_graph_batch(batch, device):
    moved = {}
    for key, value in batch.items():
        if hasattr(value, "to") and key.endswith("graph"):
            moved[key] = value.to(device) if value is not None else None
        elif torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def build_model(args, output_dim):
    model_name = str(args.model)
    if model_name == "GIN":
        return PlainGIN(
            output_dim=output_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            norm_type=args.norm_type,
            readout=args.readout,
            mlp_hidden_dims=parse_int_tuple(args.mlp_hidden_dims) if args.mlp_hidden_dims else None,
        )
    raise ValueError(f"Unsupported graph student model: {model_name}")


def forward_model(model, batch, model_name):
    if model_name == "GIN":
        return model(batch["graph"])
    raise ValueError(f"Unsupported graph student model: {model_name}")


def make_loader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=graph_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


def collect_labels(dataset, task_indices):
    values = []
    for idx in dataset.indices:
        values.append(dataset.y[idx, task_indices].detach().cpu().numpy())
    return np.asarray(values, dtype=np.float32)


def predict(model, dataset, task_indices, batch_size, device, num_workers, model_name):
    loader = make_loader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    logits_list = []
    y_list = []
    row_index_list = []
    smiles_list = []
    with torch.no_grad():
        for batch in loader:
            batch = move_graph_batch(batch, device)
            logits = forward_model(model, batch, model_name)
            logits_list.append(logits.detach().cpu().numpy())
            y_list.append(batch["y"][:, task_indices].detach().cpu().numpy())
            row_index_list.append(batch["row_index"].detach().cpu().numpy())
            smiles_list.extend(batch["smiles"])
    return {
        "logits": np.vstack(logits_list).astype(np.float32),
        "y": np.vstack(y_list).astype(np.float32),
        "row_index": np.concatenate(row_index_list).astype(np.int64),
        "smiles": np.asarray(smiles_list, dtype=object),
    }


def evaluate_split(model, dataset, task_indices, task_names, args, device):
    pred = predict(
        model,
        dataset,
        task_indices,
        args.eval_batch_size,
        device,
        args.num_workers,
        args.model,
    )
    metrics_df = compute_multitask_metrics(pred["logits"], pred["y"], task_types=None, task_names=task_names)
    return pred, metrics_df, summarize_metrics(metrics_df)


def monitor_from_metrics(metrics_df, summary, monitor):
    if monitor == "tox21_12_mean_roc_auc":
        values = pd.to_numeric(
            metrics_df[metrics_df["task_name"].isin(TOX21_12_TASKS)]["roc_auc"],
            errors="coerce",
        )
        return float(values.mean())
    return summary.get(monitor, np.nan)


def save_predictions(path, split_name, pred, task_names, task_indices):
    path.parent.mkdir(parents=True, exist_ok=True)
    logits = np.asarray(pred["logits"], dtype=np.float32)
    prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    np.savez_compressed(
        path,
        split_name=np.asarray(split_name, dtype=object),
        row_index=np.asarray(pred["row_index"], dtype=np.int64),
        smiles=np.asarray(pred["smiles"], dtype=object),
        y_true=np.asarray(pred["y"], dtype=np.float32),
        logits=logits,
        prob=prob.astype(np.float32),
        task_names=np.asarray(task_names, dtype=object),
        selected_task_indices=np.asarray(task_indices, dtype=np.int64),
    )


def train_once(datasets, task_indices, task_names, args, run_dir):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(args, output_dim=len(task_indices)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_y_np = collect_labels(datasets["training"], task_indices)
    pos_weight = compute_pos_weight_from_labels(train_y_np) if args.use_pos_weight else None
    loss_fn = MultiTaskLossComputer(
        task_types=None,
        soft_cls_weight=args.soft_cls_weight,
        soft_reg_weight=args.soft_reg_weight,
        hard_reduction=args.hard_reduction,
        aux_reduction=args.aux_reduction,
        pos_weight=pos_weight,
    )
    targets = DistillationTargetProvider(
        teacher_profile_dir=args.teacher_profile_dir,
        task_indices=task_indices,
        target_task_indices=args.target_task_indices,
        soft_profile_mode=args.soft_profile_mode,
        target_clip=args.aux_target_clip,
    )
    train_loader = make_loader(
        datasets["training"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    best_metric = None
    best_state = None
    bad_epochs = 0
    history = []
    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0
        loss_sums = {}
        for batch in train_loader:
            batch = move_graph_batch(batch, device)
            yb = batch["y"][:, task_indices]
            optimizer.zero_grad(set_to_none=True)
            logits = forward_model(model, batch, args.model)
            aux_targets = targets.batch(batch["row_index"], device)
            loss, loss_items = loss_fn.compute(logits, yb, **aux_targets)
            loss.backward()
            if args.gradient_clip_norm and args.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            optimizer.step()
            batch_size = int(yb.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size
            total_seen += batch_size
            for key, value in loss_items.items():
                loss_sums[key] = loss_sums.get(key, 0.0) + float(value.detach().cpu()) * batch_size

        _, valid_metrics_df, valid_summary = evaluate_split(model, datasets["valid"], task_indices, task_names, args, device)
        monitor = monitor_from_metrics(valid_metrics_df, valid_summary, args.monitor)
        row = {"epoch": epoch, "train_loss": total_loss / max(total_seen, 1)}
        for key, value in sorted(loss_sums.items()):
            row[f"train_{key}"] = value / max(total_seen, 1)
        row.update({f"valid_{key}": value for key, value in valid_summary.items()})
        history.append(row)

        if best_metric is None or (np.isfinite(monitor) and monitor > best_metric):
            best_metric = float(monitor)
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        print(
            format_epoch_progress(
                epoch=epoch,
                epochs=args.epochs,
                train_loss=row["train_loss"],
                monitor=args.monitor,
                monitor_value=monitor,
                best_value=best_metric if best_metric is not None else monitor,
                elapsed=time.perf_counter() - start_time,
            ),
            flush=True,
        )
        if args.patience is not None and bad_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}; best_epoch={epoch - bad_epochs}.", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"model_state_dict": model.state_dict(), "history": history}, run_dir / "best.pt")
    pd.DataFrame(history).to_csv(run_dir / "epoch_metrics.csv", index=False)
    return model, device


def save_split_outputs(datasets, model, device, task_indices, task_names, args, run_dir, metrics_dir):
    summaries = {}
    for split_name in ["training", "valid", "test"]:
        out_split = "train" if split_name == "training" else split_name
        pred, metrics_df, summary = evaluate_split(
            model,
            datasets[split_name],
            task_indices,
            task_names,
            args,
            device,
        )
        summaries[out_split] = summary
        save_predictions(run_dir / f"{out_split}_predictions.npz", out_split, pred, task_names, task_indices)
        metadata = {
            "dataset": args.dataset,
            "feature": args.feature,
            "model": args.model,
            "tasks": args.tasks,
            "loss": args.loss_name,
            "split": args.split_id,
            "seed": args.seed,
            "prediction_split": out_split,
        }
        write_metric_table(run_dir / f"{out_split}_metrics.csv", metrics_df, metadata)
        write_metric_table(metrics_dir / f"{out_split}_per_task_metrics.csv", metrics_df, metadata)
        save_json(run_dir / f"{out_split}_summary.json", summary)
    save_json(metrics_dir / "profile_metric_summary.json", summaries)
    return summaries


def parse_args():
    parser = argparse.ArgumentParser(description="Train one fixed-split graph student run.")
    def add_bool_optional(name, default=True):
        dest = name.replace("-", "_")
        parser.add_argument(f"--{name}", dest=dest, action="store_true")
        parser.add_argument(f"--no-{name}", dest=dest, action="store_false")
        parser.set_defaults(**{dest: default})

    parser.add_argument("--dataset", default="Tox21_Toxcast")
    parser.add_argument("--model", choices=["GIN"], required=True)
    parser.add_argument("--feature", default="graph")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--task_file", default=None, help="Optional newline-delimited task names for custom task-set ablations.")
    parser.add_argument("--graph_dir", required=True)
    parser.add_argument("--output_root", default="student")
    parser.add_argument("--split_id", default="contain_valid_graph")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--loss_name", default="mask_bce")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--norm_type", default="batch_norm")
    parser.add_argument("--readout", default="max")
    parser.add_argument("--mlp_hidden_dims", default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--monitor", default="mean_roc_auc")
    parser.add_argument("--use_pos_weight", action="store_true")
    parser.add_argument("--hard_reduction", default="observation", choices=["observation", "task", "task_balanced", "task_mean"])
    parser.add_argument("--aux_reduction", default="observation", choices=["observation", "task", "task_balanced", "task_mean"])
    parser.add_argument("--teacher_profile_dir", default=None)
    parser.add_argument(
        "--soft_profile_mode",
        default="full",
        choices=["full", "target_only", "non_target"],
        help="Teacher-profile dimensions used by soft loss; hard BCE task set is unchanged.",
    )
    parser.add_argument("--soft_cls_weight", type=float, default=0.0)
    parser.add_argument("--soft_reg_weight", type=float, default=0.0)
    parser.add_argument("--aux_target_clip", type=float, default=0.0)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    validate_distillation_args(args)
    set_seed(args.seed)
    datasets = {
        "training": GraphDataset(args.graph_dir, split="training", load_brics=False, load_murcko=False, load_fg=False),
        "valid": GraphDataset(args.graph_dir, split="valid", load_brics=False, load_murcko=False, load_fg=False),
        "test": GraphDataset(args.graph_dir, split="test", load_brics=False, load_murcko=False, load_fg=False),
    }
    task_names_all = datasets["training"].task_names
    if task_names_all is None:
        raise ValueError("Graph dataset is missing task_names.json.")
    if args.task_file:
        task_indices, task_names = select_tasks_from_file(task_names_all, args.task_file)
    else:
        task_indices, task_names = select_tasks(task_names_all, args.tasks)
    args.target_task_indices = [task_names_all.index(task) for task in TOX21_12_TASKS if task in task_names_all]
    output_root = Path(args.output_root)
    rid = student_run_id(args.feature, args.model, args.tasks, args.loss_name, args.split_id, args.seed)
    run_dir = output_root / args.dataset / "runs" / rid
    metrics_dir = output_root / args.dataset / "metrics" / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        run_dir / "run_config.json",
        {
            **vars(args),
            "run_id": rid,
            "task_names": task_names,
            "selected_task_indices": task_indices,
        },
    )
    model, device = train_once(datasets, task_indices, task_names, args, run_dir)
    save_split_outputs(datasets, model, device, task_indices, task_names, args, run_dir, metrics_dir)
    print(f"Saved graph student run: {run_dir}")


if __name__ == "__main__":
    main()
