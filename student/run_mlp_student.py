import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import TabularMLP
from experiment_naming import student_run_id as compact_student_run_id
from student.distillation_targets import DistillationTargetProvider, validate_distillation_args
from teacher.common import load_feature_npz, save_json, write_metric_table
from training.losses import MultiTaskLossComputer, compute_pos_weight_from_labels
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


def sanitize_id(value):
    return str(value).replace(" ", "-").replace("/", "-").replace("\\", "-")


def student_run_id(feature, model, tasks, loss, split, seed):
    return compact_student_run_id(feature, model, tasks, loss, split, seed)


def select_tasks(task_names, task_set):
    task_set = str(task_set)
    if task_set == "all_93":
        return list(range(len(task_names))), list(task_names)
    if task_set == "tox21_12":
        missing = [task for task in TOX21_12_TASKS if task not in task_names]
        if missing:
            raise ValueError(f"Missing Tox21 12 tasks: {missing}")
        indices = [task_names.index(task) for task in TOX21_12_TASKS]
        return indices, list(TOX21_12_TASKS)
    raise ValueError(f"Unsupported task_set: {task_set}")


def select_tasks_from_file(task_names, task_file):
    path = Path(task_file)
    tasks = [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.strip().startswith("#")]
    missing = [task for task in tasks if task not in task_names]
    if missing:
        raise ValueError(f"{path} contains tasks missing from dataset: {missing}")
    indices = [task_names.index(task) for task in tasks]
    return indices, tasks


def predict(model, x, batch_size, device):
    model.eval()
    preds = []
    loader = DataLoader(TensorDataset(torch.tensor(x, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            preds.append(model(xb.to(device)).detach().cpu().numpy())
    return np.vstack(preds).astype(np.float32)


def evaluate_split(model, data, indices, task_indices, task_names, batch_size, device):
    logits = predict(model, data["X"][indices], batch_size, device)
    y = data["y"][indices][:, task_indices]
    metrics_df = compute_multitask_metrics(logits, y, task_types=None, task_names=task_names)
    return logits, y, metrics_df, summarize_metrics(metrics_df)


def save_predictions(path, split_name, row_index, smiles, y_true, logits, task_names, task_indices):
    path.parent.mkdir(parents=True, exist_ok=True)
    prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    np.savez_compressed(
        path,
        split_name=np.asarray(split_name, dtype=object),
        row_index=np.asarray(row_index, dtype=np.int64),
        smiles=np.asarray(smiles, dtype=object),
        y_true=np.asarray(y_true, dtype=np.float32),
        logits=np.asarray(logits, dtype=np.float32),
        prob=np.asarray(prob, dtype=np.float32),
        task_names=np.asarray(task_names, dtype=object),
        selected_task_indices=np.asarray(task_indices, dtype=np.int64),
    )


def train_once(data, task_indices, task_names, args, run_dir):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = TabularMLP(
        input_dim=data["X"].shape[1],
        output_dim=len(task_indices),
        num_hidden_layers=args.num_hidden_layers,
        shrink_ratio=args.shrink_ratio,
        hidden_dims=[int(x) for x in args.hidden_dims.split(",")] if args.hidden_dims else None,
        min_hidden_dim=args.min_hidden_dim,
        dropout=args.dropout,
        activation=args.activation,
        norm_type=args.norm_type,
        batch_norm=args.batch_norm,
        negative_slope=args.negative_slope,
    ).to(device)

    train_idx = np.where(data["groups"] == "training")[0].astype(np.int64)
    valid_idx = np.where(data["groups"] == "valid")[0].astype(np.int64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_y_np = data["y"][train_idx][:, task_indices]
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

    train_x = torch.tensor(data["X"][train_idx], dtype=torch.float32)
    train_y = torch.tensor(train_y_np, dtype=torch.float32)
    train_row_index = torch.tensor(train_idx, dtype=torch.long)
    train_loader = DataLoader(TensorDataset(train_x, train_y, train_row_index), batch_size=args.batch_size, shuffle=True)

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
        for xb, yb, row_idx in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            aux_targets = targets.batch(row_idx, device)
            loss, loss_items = loss_fn.compute(logits, yb, **aux_targets)
            loss.backward()
            if args.gradient_clip_norm and args.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(xb)
            total_seen += len(xb)
            for key, value in loss_items.items():
                loss_sums[key] = loss_sums.get(key, 0.0) + float(value.detach().cpu()) * len(xb)

        _, _, valid_metrics, valid_summary = evaluate_split(
            model, data, valid_idx, task_indices, task_names, args.eval_batch_size, device
        )
        monitor = valid_summary.get(args.monitor, np.nan)
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


def save_split_outputs(data, model, device, task_indices, task_names, args, run_dir, metrics_dir):
    split_map = {
        "train": np.where(data["groups"] == "training")[0].astype(np.int64),
        "valid": np.where(data["groups"] == "valid")[0].astype(np.int64),
        "test": np.where(data["groups"] == "test")[0].astype(np.int64),
    }
    summaries = {}
    for split_name, indices in split_map.items():
        logits, y_true, metrics_df, summary = evaluate_split(
            model, data, indices, task_indices, task_names, args.eval_batch_size, device
        )
        summaries[split_name] = summary
        save_predictions(
            run_dir / f"{split_name}_predictions.npz",
            split_name,
            indices,
            data["smiles"][indices],
            y_true,
            logits,
            task_names,
            task_indices,
        )
        metadata = {
            "dataset": args.dataset,
            "feature": args.feature,
            "model": "MLP",
            "tasks": args.tasks,
            "loss": args.loss_name,
            "split": args.split_id,
            "seed": args.seed,
            "prediction_split": split_name,
        }
        write_metric_table(run_dir / f"{split_name}_metrics.csv", metrics_df, metadata)
        write_metric_table(metrics_dir / f"{split_name}_per_task_metrics.csv", metrics_df, metadata)
        save_json(run_dir / f"{split_name}_summary.json", summary)
    save_json(metrics_dir / "profile_metric_summary.json", summaries)
    return summaries


def parse_args():
    parser = argparse.ArgumentParser(description="Train one fixed-split student MLP run.")
    parser.add_argument("--dataset", default="Tox21_Toxcast")
    parser.add_argument("--feature", required=True)
    parser.add_argument("--tasks", choices=["tox21_12", "all_93"], required=True)
    parser.add_argument("--feature_npz", required=True)
    parser.add_argument("--output_root", default="student")
    parser.add_argument("--split_id", default="contain_valid_ecfp4_similarity")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--loss_name", default="mask_bce")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--hidden_dims", default=None)
    parser.add_argument("--num_hidden_layers", type=int, default=3)
    parser.add_argument("--shrink_ratio", type=float, default=0.5)
    parser.add_argument("--min_hidden_dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--activation", default="leaky_relu")
    parser.add_argument("--norm_type", default="layer_norm")
    parser.add_argument("--batch_norm", action="store_true")
    parser.add_argument("--negative_slope", type=float, default=0.01)
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
    parser.add_argument("--gradient_clip_norm", type=float, default=0.0)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    validate_distillation_args(args)
    set_seed(args.seed)
    data = load_feature_npz(args.feature_npz)
    task_indices, task_names = select_tasks(data["task_names"], args.tasks)
    args.target_task_indices = [data["task_names"].index(task) for task in TOX21_12_TASKS if task in data["task_names"]]
    output_root = Path(args.output_root)
    rid = student_run_id(args.feature, "MLP", args.tasks, args.loss_name, args.split_id, args.seed)
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
    model, device = train_once(data, task_indices, task_names, args, run_dir)
    save_split_outputs(data, model, device, task_indices, task_names, args, run_dir, metrics_dir)
    print(f"Saved student run: {run_dir}")


if __name__ == "__main__":
    main()
