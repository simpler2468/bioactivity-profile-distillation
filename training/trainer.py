import argparse
import json
import random
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
from dataset.tabular_dataset import TabularFeatureDataset, tabular_collate_fn
from models import PlainGIN, TabularMLP
from training.losses import MultiTaskLossComputer, compute_pos_weight_from_labels
from training.metrics import PredictionCollector


def load_task_types(task_metadata_path, task_names, name_column="task_name", type_column="task_type"):
    if task_metadata_path is None:
        return None
    metadata = pd.read_csv(task_metadata_path)
    if name_column not in metadata.columns or type_column not in metadata.columns:
        raise ValueError(
            f"task metadata must contain columns {name_column!r} and {type_column!r}."
        )
    mapping = dict(zip(metadata[name_column].astype(str), metadata[type_column].astype(str)))
    missing = [task for task in task_names if task not in mapping]
    if missing:
        raise ValueError(f"task metadata is missing {len(missing)} task(s), first: {missing[:5]}")
    return [mapping[task] for task in task_names]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def model_forward(model, batch, modality):
    if modality == "tabular":
        return model(batch["x"])
    if modality == "graph":
        return model(
            batch["graph"],
            brics_graph=batch.get("brics_graph"),
            brics_counts=batch.get("brics_counts"),
            murcko_graph=batch.get("murcko_graph"),
            murcko_counts=batch.get("murcko_counts"),
            fg_graph=batch.get("fg_graph"),
            fg_counts=batch.get("fg_counts"),
        )
    raise ValueError(f"Unsupported modality: {modality}")


def make_optimizer(model, config):
    config = dict(config or {})
    name = str(config.get("name", "adamw")).lower()
    lr = float(config.get("lr", 1e-3))
    weight_decay = float(config.get("weight_decay", 1e-4))
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(config.get("momentum", 0.9)),
            weight_decay=weight_decay,
            nesterov=bool(config.get("nesterov", False)),
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def make_scheduler(optimizer, config):
    config = dict(config or {})
    name = str(config.get("name", "none")).lower()
    if name in {"none", "null", "false"}:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config.get("t_max", config.get("epochs", 50))),
            eta_min=float(config.get("eta_min", 0.0)),
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(config.get("step_size", 20)),
            gamma=float(config.get("gamma", 0.5)),
        )
    raise ValueError(f"Unsupported scheduler: {name}")


def score_from_summary(summary, monitor):
    value = summary.get(monitor)
    if value is None:
        return np.nan
    return float(value)


def metric_label(monitor):
    name = str(monitor)
    if name == "mean_roc_auc":
        return "auc"
    if name.startswith("mean_"):
        return name[len("mean_") :]
    return name


def format_epoch_progress(epoch, epochs, train_loss, monitor, monitor_value, best_value, elapsed):
    label = metric_label(monitor)
    return (
        f"Epoch {int(epoch):03d}/{int(epochs):03d} | "
        f"train_loss={float(train_loss):.4f} | "
        f"valid_{label}={float(monitor_value):.4f} | "
        f"best_{label}={float(best_value):.4f} | "
        f"elapsed={float(elapsed):.1f}s"
    )


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        valid_loader=None,
        test_loader=None,
        loss_computer=None,
        task_types=None,
        task_names=None,
        optimizer=None,
        scheduler=None,
        device=None,
        output_dir="runs/debug",
        monitor="mean_roc_auc",
        monitor_mode="max",
        patience=None,
        modality="tabular",
    ):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.loss_computer = loss_computer or MultiTaskLossComputer(task_types=task_types)
        self.task_types = task_types
        self.task_names = task_names
        self.optimizer = optimizer or make_optimizer(model, {})
        self.scheduler = scheduler
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.patience = patience
        self.modality = modality

        if self.monitor_mode not in {"max", "min"}:
            raise ValueError("monitor_mode must be 'max' or 'min'.")
        self.model.to(self.device)

    def _is_better(self, value, best):
        if np.isnan(value):
            return False
        if best is None:
            return True
        return value > best if self.monitor_mode == "max" else value < best

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_seen = 0
        for batch in self.train_loader:
            batch = move_batch_to_device(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            pred = model_forward(self.model, batch, self.modality)
            loss, _ = self.loss_computer.compute(pred, batch["y"])
            loss.backward()
            self.optimizer.step()
            batch_size = int(batch["y"].shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size
            total_seen += batch_size
        return {"loss": total_loss / max(total_seen, 1)}

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0.0
        total_seen = 0
        collector = PredictionCollector()
        for batch in loader:
            batch = move_batch_to_device(batch, self.device)
            pred = model_forward(self.model, batch, self.modality)
            loss, _ = self.loss_computer.compute(pred, batch["y"])
            batch_size = int(batch["y"].shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size
            total_seen += batch_size
            collector.update(pred, batch["y"])

        metrics_df, summary = collector.compute(task_types=self.task_types, task_names=self.task_names)
        summary["loss"] = total_loss / max(total_seen, 1)
        return summary, metrics_df

    def save_checkpoint(self, path, epoch, summary):
        torch.save(
            {
                "epoch": int(epoch),
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "summary": dict(summary),
                "task_names": self.task_names,
                "task_types": self.task_types,
            },
            path,
        )

    def fit(self, epochs):
        history = []
        best_value = None
        best_epoch = None
        bad_epochs = 0
        start_time = time.perf_counter()

        for epoch in range(1, int(epochs) + 1):
            train_summary = self.train_epoch()
            if self.scheduler is not None:
                self.scheduler.step()

            row = {"epoch": epoch, "train_loss": train_summary["loss"]}
            valid_metrics_df = None
            if self.valid_loader is not None:
                valid_summary, valid_metrics_df = self.evaluate(self.valid_loader)
                row.update({f"valid_{key}": value for key, value in valid_summary.items()})
                monitor_value = score_from_summary(valid_summary, self.monitor)
            else:
                monitor_value = -train_summary["loss"] if self.monitor_mode == "max" else train_summary["loss"]

            is_best = self._is_better(monitor_value, best_value)
            if is_best:
                best_value = monitor_value
                best_epoch = epoch
                bad_epochs = 0
                self.save_checkpoint(self.output_dir / "best.pt", epoch, row)
                if valid_metrics_df is not None:
                    valid_metrics_df.to_csv(self.output_dir / "valid_per_task_metrics_best.csv", index=False)
            else:
                bad_epochs += 1

            self.save_checkpoint(self.output_dir / "last.pt", epoch, row)
            row["is_best"] = bool(is_best)
            history.append(row)
            pd.DataFrame(history).to_csv(self.output_dir / "epoch_metrics.csv", index=False)

            print(
                format_epoch_progress(
                    epoch=epoch,
                    epochs=epochs,
                    train_loss=row["train_loss"],
                    monitor=self.monitor,
                    monitor_value=monitor_value,
                    best_value=best_value if best_value is not None else monitor_value,
                    elapsed=time.perf_counter() - start_time,
                ),
                flush=True,
            )

            if self.patience is not None and bad_epochs >= int(self.patience):
                print(f"Early stopping at epoch {epoch}; best_epoch={best_epoch}.", flush=True)
                break

        if self.test_loader is not None:
            test_summary, test_metrics_df = self.evaluate(self.test_loader)
            test_metrics_df.to_csv(self.output_dir / "test_per_task_metrics.csv", index=False)
            (self.output_dir / "test_summary.json").write_text(
                json.dumps(test_summary, indent=2, allow_nan=True),
                encoding="utf-8",
            )

        run_summary = {
            "best_epoch": best_epoch,
            "best_monitor": best_value,
            "monitor": self.monitor,
            "monitor_mode": self.monitor_mode,
        }
        (self.output_dir / "run_summary.json").write_text(
            json.dumps(run_summary, indent=2, allow_nan=True),
            encoding="utf-8",
        )
        return pd.DataFrame(history)


def build_tabular_run(args):
    train_ds = TabularFeatureDataset(args.data_path, split=args.train_split)
    valid_ds = TabularFeatureDataset(args.data_path, split=args.valid_split) if args.valid_split else None
    test_ds = TabularFeatureDataset(args.data_path, split=args.test_split) if args.test_split else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=tabular_collate_fn)
    valid_loader = (
        DataLoader(valid_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=tabular_collate_fn)
        if valid_ds is not None and len(valid_ds) > 0
        else None
    )
    test_loader = (
        DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=tabular_collate_fn)
        if test_ds is not None and len(test_ds) > 0
        else None
    )

    model = TabularMLP(
        input_dim=train_ds.input_dim,
        output_dim=train_ds.output_dim,
        hidden_dims=[int(x) for x in args.hidden_dims.split(",")] if args.hidden_dims else None,
        dropout=args.dropout,
    )
    return model, train_loader, valid_loader, test_loader, train_ds.task_names


def build_graph_run(args):
    load_brics = args.use_brics_view
    load_murcko = args.use_murcko_view
    load_fg = args.use_fg_view
    train_ds = GraphDataset(
        args.data_path,
        split=args.train_split,
        load_brics=load_brics,
        load_murcko=load_murcko,
        load_fg=load_fg,
    )
    valid_ds = (
        GraphDataset(args.data_path, split=args.valid_split, load_brics=load_brics, load_murcko=load_murcko, load_fg=load_fg)
        if args.valid_split
        else None
    )
    test_ds = (
        GraphDataset(args.data_path, split=args.test_split, load_brics=load_brics, load_murcko=load_murcko, load_fg=load_fg)
        if args.test_split
        else None
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=graph_collate_fn)
    valid_loader = (
        DataLoader(valid_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=graph_collate_fn)
        if valid_ds is not None and len(valid_ds) > 0
        else None
    )
    test_loader = (
        DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=graph_collate_fn)
        if test_ds is not None and len(test_ds) > 0
        else None
    )

    model = PlainGIN(
        output_dim=len(train_ds.task_names),
        hidden_dim=args.graph_hidden_dim,
        num_layers=len([x for x in args.full_depths.split(",") if x]),
        dropout=args.dropout,
        norm_type=getattr(args, "norm_type", "batch_norm"),
        readout=getattr(args, "readout", "mean"),
    )
    return model, train_loader, valid_loader, test_loader, train_ds.task_names


def parse_args():
    parser = argparse.ArgumentParser(description="Train tabular MLP baseline.")

    def add_bool_arg(name, default=True):
        dest = name.replace("-", "_")
        parser.add_argument(f"--{name}", dest=dest, action="store_true")
        parser.add_argument(f"--no_{name}", dest=dest, action="store_false")
        parser.set_defaults(**{dest: default})

    parser.add_argument("--modality", choices=["tabular", "graph"], required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--train_split", default="training")
    parser.add_argument("--valid_split", default="valid")
    parser.add_argument("--test_split", default="test")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default=None)
    parser.add_argument("--monitor", default="mean_roc_auc")
    parser.add_argument("--monitor_mode", choices=["max", "min"], default="max")
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=["adamw", "adam", "sgd"], default="adamw")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden_dims", default=None, help="Comma-separated MLP hidden dims, e.g. 512,256.")
    parser.add_argument("--graph_hidden_dim", type=int, default=256)
    parser.add_argument("--full_depths", default="3,4,5")
    parser.add_argument("--brics_layers", type=int, default=3)
    parser.add_argument("--murcko_layers", type=int, default=4)
    parser.add_argument("--fg_layers", type=int, default=2)
    parser.add_argument("--cross_attention_heads", type=int, default=8)
    add_bool_arg("use_global_view", default=True)
    add_bool_arg("use_brics_view", default=True)
    add_bool_arg("use_murcko_view", default=True)
    add_bool_arg("use_fg_view", default=True)
    parser.add_argument("--hard_reg_weight", type=float, default=1.0)
    parser.add_argument("--hard_reduction", choices=["observation", "task_balanced"], default="observation")
    add_bool_arg("use_pos_weight", default=False)
    parser.add_argument("--task_metadata", default=None)
    parser.add_argument("--task_name_column", default="task_name")
    parser.add_argument("--task_type_column", default="task_type")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.modality == "tabular":
        model, train_loader, valid_loader, test_loader, task_names = build_tabular_run(args)
    else:
        model, train_loader, valid_loader, test_loader, task_names = build_graph_run(args)
    task_types = load_task_types(
        args.task_metadata,
        task_names,
        name_column=args.task_name_column,
        type_column=args.task_type_column,
    )

    pos_weight = None
    if args.use_pos_weight:
        labels = []
        for batch in train_loader:
            labels.append(batch["y"])
        cls_mask = None
        if task_types is not None:
            from training.losses import as_task_type_tensor

            cls_mask = as_task_type_tensor(task_types, len(task_names)).cpu()
        pos_weight = compute_pos_weight_from_labels(torch.cat(labels, dim=0), cls_mask=cls_mask)

    loss_computer = MultiTaskLossComputer(
        task_types=task_types,
        task_names=task_names,
        hard_reg_weight=args.hard_reg_weight,
        hard_reduction=args.hard_reduction,
        pos_weight=pos_weight,
    )
    optimizer = make_optimizer(
        model,
        {"name": args.optimizer, "lr": args.lr, "weight_decay": args.weight_decay},
    )
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        loss_computer=loss_computer,
        task_types=task_types,
        task_names=task_names,
        optimizer=optimizer,
        device=args.device,
        output_dir=args.output_dir,
        monitor=args.monitor,
        monitor_mode=args.monitor_mode,
        patience=args.patience,
        modality=args.modality,
    )
    trainer.fit(args.epochs)


if __name__ == "__main__":
    main()
