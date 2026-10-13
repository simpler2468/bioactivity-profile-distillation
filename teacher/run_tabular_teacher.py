import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import TabularMLP
from teacher.common import (
    clip_prob_to_logit,
    indices_from_split,
    load_feature_npz,
    run_id,
    save_json,
    save_prediction_npz,
    write_metric_table,
)
from training.losses import MultiTaskLossComputer, compute_pos_weight_from_labels
from training.metrics import compute_multitask_metrics, summarize_metrics



def set_seed(seed):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_epoch_progress(epoch, epochs, train_loss, monitor, monitor_value, best_value, elapsed):
    monitor_text = f"{monitor}={monitor_value:.4f}" if np.isfinite(monitor_value) else f"{monitor}=nan"
    best_text = f"best_{monitor}={best_value:.4f}" if best_value is not None and np.isfinite(best_value) else f"best_{monitor}=nan"
    return (
        f"Epoch {epoch:03d}/{epochs:03d} | "
        f"train_loss={train_loss:.4f} | "
        f"{monitor_text} | {best_text} | elapsed={elapsed:.1f}s"
    )

def subset(data, indices):
    return data["X"][indices], data["y"][indices], data["smiles"][indices]


def predict_mlp(model, x, batch_size, device):
    model.eval()
    preds = []
    loader = DataLoader(TensorDataset(torch.tensor(x, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            preds.append(model(xb.to(device)).detach().cpu().numpy())
    return np.vstack(preds).astype(np.float32)


def train_mlp(data, train_idx, valid_idx, args, run_dir):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = TabularMLP(
        input_dim=data["X"].shape[1],
        output_dim=data["y"].shape[1],
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos_weight = compute_pos_weight_from_labels(data["y"][train_idx]) if args.use_pos_weight else None
    loss_fn = MultiTaskLossComputer(
        task_types=None,
        hard_reduction=args.hard_reduction,
        pos_weight=pos_weight,
    )

    train_x = torch.tensor(data["X"][train_idx], dtype=torch.float32)
    train_y = torch.tensor(data["y"][train_idx], dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True)

    best_metric = None
    best_state = None
    bad_epochs = 0
    history = []
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss, _ = loss_fn.compute(pred, yb)
            loss.backward()
            if args.gradient_clip_norm and args.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(xb)
            total_seen += len(xb)

        row = {"epoch": epoch, "train_loss": total_loss / max(total_seen, 1)}
        if len(valid_idx) > 0:
            valid_pred = predict_mlp(model, data["X"][valid_idx], args.eval_batch_size, device)
            valid_metrics = compute_multitask_metrics(
                valid_pred,
                data["y"][valid_idx],
                task_types=None,
                task_names=data["task_names"],
            )
            valid_summary = summarize_metrics(valid_metrics)
            row.update({f"valid_{k}": v for k, v in valid_summary.items()})
            monitor = valid_summary.get(args.monitor, np.nan)
        else:
            monitor = -row["train_loss"]

        if best_metric is None or (np.isfinite(monitor) and monitor > best_metric):
            best_metric = float(monitor)
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        history.append(row)
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


def train_ml_taskwise(data, train_idx, args, run_dir):
    x_train = data["X"][train_idx]
    y_train = data["y"][train_idx]
    models = []
    skipped = []
    for task_idx, task_name in enumerate(data["task_names"]):
        y = y_train[:, task_idx]
        valid = np.isfinite(y)
        yv = y[valid].astype(int)
        if len(yv) == 0 or len(np.unique(yv)) < 2:
            models.append(None)
            skipped.append({"task_index": task_idx, "task_name": task_name, "reason": "less_than_two_classes"})
            continue

        if args.model == "RF":
            clf = RandomForestClassifier(
                n_estimators=args.rf_n_estimators,
                max_features=args.rf_max_features,
                min_samples_leaf=args.rf_min_samples_leaf,
                n_jobs=args.n_jobs,
                random_state=args.seed,
                class_weight=args.class_weight,
            )
        elif args.model == "SVM":
            clf = SVC(
                C=args.svm_c,
                gamma=args.svm_gamma,
                kernel=args.svm_kernel,
                class_weight=args.class_weight,
                random_state=args.seed,
            )
        else:
            raise ValueError(f"Unsupported ML model: {args.model}")
        clf.fit(x_train[valid], yv)
        models.append(clf)

    if args.save_ml_models:
        import joblib

        joblib.dump(models, run_dir / "model.pkl")
    pd.DataFrame(skipped).to_csv(run_dir / "skipped_tasks.csv", index=False)
    return models


def predict_ml_taskwise(models, x, model_name):
    pred = np.full((len(x), len(models)), np.nan, dtype=np.float32)
    for task_idx, clf in enumerate(models):
        if clf is None:
            continue
        if model_name == "RF":
            classes = list(clf.classes_)
            if 1 in classes:
                pos_col = classes.index(1)
                prob = clf.predict_proba(x)[:, pos_col]
            else:
                prob = np.zeros(len(x), dtype=np.float32)
            pred[:, task_idx] = clip_prob_to_logit(prob)
        elif model_name == "SVM":
            pred[:, task_idx] = clf.decision_function(x).astype(np.float32)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    return pred


def save_all_predictions(data, run_dir, indices_by_split, raw_scores_by_split, args):
    for split_name, indices in indices_by_split.items():
        raw_score = raw_scores_by_split[split_name]
        save_prediction_npz(
            run_dir / f"{split_name}_predictions.npz",
            split_name=split_name,
            row_index=indices,
            smiles=data["smiles"][indices],
            y_true=data["y"][indices],
            raw_score=raw_score,
            task_names=data["task_names"],
        )
        metrics_df = compute_multitask_metrics(raw_score, data["y"][indices], task_types=None, task_names=data["task_names"])
        write_metric_table(
            run_dir / f"{split_name}_metrics.csv",
            metrics_df,
            {
                "dataset": args.dataset,
                "feature": args.feature,
                "model": args.model,
                "split": args.split_id,
                "fold": args.fold,
                "seed": args.seed,
                "prediction_split": split_name,
            },
        )
        save_json(run_dir / f"{split_name}_summary.json", summarize_metrics(metrics_df))


def parse_args():
    parser = argparse.ArgumentParser(description="Train one fold of a tabular teacher and save raw-score predictions.")
    parser.add_argument("--dataset", default="Tox21_Toxcast")
    parser.add_argument("--feature", required=True)
    parser.add_argument("--model", choices=["MLP", "RF", "SVM"], required=True)
    parser.add_argument("--split_id", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--feature_npz", required=True)
    parser.add_argument("--fold_csv", required=True)
    parser.add_argument("--base_csv", default=None, help="Optional full raw CSV; reserved for audit.")
    parser.add_argument("--output_root", default="teacher")
    parser.add_argument("--epochs", type=int, default=50)
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
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default=None)
    parser.add_argument("--use_pos_weight", action="store_true")
    parser.add_argument("--hard_reduction", default="observation", choices=["observation", "task", "task_balanced", "task_mean"])
    parser.add_argument("--gradient_clip_norm", type=float, default=0.0)
    parser.add_argument("--rf_n_estimators", type=int, default=500)
    parser.add_argument("--rf_max_features", default="sqrt")
    parser.add_argument("--rf_min_samples_leaf", type=int, default=1)
    parser.add_argument("--svm_c", type=float, default=1.0)
    parser.add_argument("--svm_gamma", default="scale")
    parser.add_argument("--svm_kernel", default="rbf")
    parser.add_argument("--class_weight", default="balanced")
    parser.add_argument("--save_ml_models", action="store_true")
    parser.add_argument("--n_jobs", type=int, default=-1)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    output_root = Path(args.output_root)
    run_dir = output_root / args.dataset / "runs" / run_id(args.feature, args.model, args.split_id, args.fold, args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "run_config.json", vars(args))

    data = load_feature_npz(args.feature_npz)
    fold_df = pd.read_csv(args.fold_csv)
    train_idx = indices_from_split(data["smiles"], fold_df, "training")
    oof_idx = indices_from_split(data["smiles"], fold_df, "test")
    inner_valid_idx = indices_from_split(data["smiles"], fold_df, "valid") if (fold_df["group"].astype(str) == "valid").any() else np.asarray([], dtype=np.int64)
    global_valid_idx = np.where(data["groups"] == "valid")[0].astype(np.int64)
    global_test_idx = np.where(data["groups"] == "test")[0].astype(np.int64)

    if args.model == "MLP":
        model, device = train_mlp(data, train_idx, inner_valid_idx, args, run_dir)
        raw_scores = {
            "oof": predict_mlp(model, data["X"][oof_idx], args.eval_batch_size, device),
            "valid": predict_mlp(model, data["X"][global_valid_idx], args.eval_batch_size, device) if len(global_valid_idx) else np.zeros((0, len(data["task_names"])), dtype=np.float32),
            "test": predict_mlp(model, data["X"][global_test_idx], args.eval_batch_size, device),
        }
    else:
        models = train_ml_taskwise(data, train_idx, args, run_dir)
        raw_scores = {
            "oof": predict_ml_taskwise(models, data["X"][oof_idx], args.model),
            "valid": predict_ml_taskwise(models, data["X"][global_valid_idx], args.model) if len(global_valid_idx) else np.zeros((0, len(data["task_names"])), dtype=np.float32),
            "test": predict_ml_taskwise(models, data["X"][global_test_idx], args.model),
        }

    indices = {"oof": oof_idx, "valid": global_valid_idx, "test": global_test_idx}
    save_all_predictions(data, run_dir, indices, raw_scores, args)
    save_json(
        run_dir / "run_metadata.json",
        {
            "run_dir": str(run_dir),
            "train_size": int(len(train_idx)),
            "oof_size": int(len(oof_idx)),
            "inner_valid_size": int(len(inner_valid_idx)),
            "valid_size": int(len(global_valid_idx)),
            "test_size": int(len(global_test_idx)),
            "raw_score_convention": "MLP=logits, SVM=decision_function, RF=logit(clipped predict_proba positive class)",
        },
    )
    print(f"Saved run: {run_dir}")


if __name__ == "__main__":
    main()
