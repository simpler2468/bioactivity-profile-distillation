import numpy as np
import pandas as pd
import torch

try:
    from sklearn import metrics as sk_metrics

    SKLEARN_AVAILABLE = True
except ImportError:
    sk_metrics = None
    SKLEARN_AVAILABLE = False

from .losses import as_task_type_tensor


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _safe_metric(fn, *args, default=np.nan, **kwargs):
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return default


def _binary_metrics(y_true, logits):
    row = {
        "n_valid": int(len(y_true)),
        "n_pos": int((y_true == 1).sum()),
        "n_neg": int((y_true == 0).sum()),
        "accuracy": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "specificity": np.nan,
        "f1": np.nan,
        "mcc": np.nan,
        "roc_auc": np.nan,
        "average_precision": np.nan,
    }
    if len(y_true) == 0 or not SKLEARN_AVAILABLE:
        return row

    prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    pred = (logits >= 0.0).astype(np.int64)

    row["accuracy"] = _safe_metric(sk_metrics.accuracy_score, y_true, pred)
    row["precision"] = _safe_metric(sk_metrics.precision_score, y_true, pred, zero_division=0)
    row["recall"] = _safe_metric(sk_metrics.recall_score, y_true, pred, zero_division=0)
    if row["n_neg"] > 0:
        row["specificity"] = _safe_metric(sk_metrics.recall_score, y_true, pred, pos_label=0, zero_division=0)
    row["f1"] = _safe_metric(sk_metrics.f1_score, y_true, pred, zero_division=0)
    row["mcc"] = _safe_metric(sk_metrics.matthews_corrcoef, y_true, pred)
    if row["n_pos"] > 0 and row["n_neg"] > 0:
        row["roc_auc"] = _safe_metric(sk_metrics.roc_auc_score, y_true, logits)
        row["average_precision"] = _safe_metric(sk_metrics.average_precision_score, y_true, prob)
    return row


def _regression_metrics(y_true, pred):
    row = {
        "n_valid": int(len(y_true)),
        "mae": np.nan,
        "mse": np.nan,
        "rmse": np.nan,
        "r2": np.nan,
        "pearson": np.nan,
    }
    if len(y_true) == 0:
        return row

    err = pred - y_true
    row["mae"] = float(np.mean(np.abs(err)))
    row["mse"] = float(np.mean(err ** 2))
    row["rmse"] = float(np.sqrt(row["mse"]))
    if len(y_true) >= 2 and np.var(y_true) > 0:
        if SKLEARN_AVAILABLE:
            row["r2"] = _safe_metric(sk_metrics.r2_score, y_true, pred)
        else:
            ss_res = np.sum((y_true - pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            row["r2"] = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
        row["pearson"] = float(np.corrcoef(y_true, pred)[0, 1])
    return row


def compute_multitask_metrics(pred, labels, task_types=None, task_names=None):
    """
    Compute per-task metrics for mixed classification/regression predictions.

    Classification predictions are logits. Regression predictions are raw
    values. NaN labels are ignored per task.
    """
    pred = _to_numpy(pred).astype(np.float64)
    labels = _to_numpy(labels).astype(np.float64)
    if pred.ndim != 2 or labels.ndim != 2:
        raise ValueError(f"pred and labels must be 2D, got {pred.shape} and {labels.shape}.")
    if pred.shape != labels.shape:
        raise ValueError(f"pred shape {pred.shape} does not match labels shape {labels.shape}.")

    num_tasks = pred.shape[1]
    cls_mask = as_task_type_tensor(task_types, num_tasks).detach().cpu().numpy().astype(bool)
    if task_names is None:
        task_names = [f"task_{idx}" for idx in range(num_tasks)]
    if len(task_names) != num_tasks:
        raise ValueError(f"task_names has {len(task_names)} entries, expected {num_tasks}.")

    rows = []
    for task_idx, task_name in enumerate(task_names):
        valid = np.isfinite(labels[:, task_idx]) & np.isfinite(pred[:, task_idx])
        y_true = labels[valid, task_idx]
        y_pred = pred[valid, task_idx]
        if cls_mask[task_idx]:
            metric_row = _binary_metrics(y_true.astype(np.int64), y_pred)
            task_type = "classification"
        else:
            metric_row = _regression_metrics(y_true, y_pred)
            task_type = "regression"
        metric_row.update(
            {
                "task_index": task_idx,
                "task_name": task_name,
                "task_type": task_type,
            }
        )
        rows.append(metric_row)

    columns = [
        "task_index",
        "task_name",
        "task_type",
        "n_valid",
        "n_pos",
        "n_neg",
        "roc_auc",
        "average_precision",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "mcc",
        "rmse",
        "mae",
        "mse",
        "r2",
        "pearson",
    ]
    df = pd.DataFrame(rows)
    for column in columns:
        if column not in df.columns:
            df[column] = np.nan
    return df[columns]


def summarize_metrics(metrics_df):
    metrics_df = metrics_df.copy()
    summary = {
        "num_tasks": int(len(metrics_df)),
        "num_classification_tasks": int((metrics_df["task_type"] == "classification").sum()),
        "num_regression_tasks": int((metrics_df["task_type"] == "regression").sum()),
    }

    cls = metrics_df[metrics_df["task_type"] == "classification"]
    reg = metrics_df[metrics_df["task_type"] == "regression"]

    for column in ["roc_auc", "average_precision", "accuracy", "f1", "mcc"]:
        summary[f"mean_{column}"] = float(pd.to_numeric(cls[column], errors="coerce").mean()) if len(cls) else np.nan
    for column in ["rmse", "mae", "mse", "r2", "pearson"]:
        summary[f"mean_{column}"] = float(pd.to_numeric(reg[column], errors="coerce").mean()) if len(reg) else np.nan

    summary["valid_classification_auc_tasks"] = int(pd.to_numeric(cls["roc_auc"], errors="coerce").notna().sum())
    summary["valid_regression_r2_tasks"] = int(pd.to_numeric(reg["r2"], errors="coerce").notna().sum())
    return summary


class PredictionCollector:
    def __init__(self):
        self.pred = []
        self.labels = []

    def update(self, pred, labels):
        self.pred.append(_to_numpy(pred))
        self.labels.append(_to_numpy(labels))

    def compute(self, task_types=None, task_names=None):
        if not self.pred:
            raise ValueError("No predictions collected.")
        pred = np.concatenate(self.pred, axis=0)
        labels = np.concatenate(self.labels, axis=0)
        metrics_df = compute_multitask_metrics(pred, labels, task_types=task_types, task_names=task_names)
        return metrics_df, summarize_metrics(metrics_df)
