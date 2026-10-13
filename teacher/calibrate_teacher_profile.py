import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teacher.common import save_json
from training.metrics import compute_multitask_metrics, summarize_metrics


def sigmoid(x):
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def bce_from_logits(logits, labels):
    return np.logaddexp(0.0, logits) - labels * logits


def fit_platt(raw_score, labels, max_iter=1000):
    valid = np.isfinite(raw_score) & np.isfinite(labels)
    x = raw_score[valid].astype(np.float64)
    y = labels[valid].astype(np.float64)
    row = {
        "n_valid": int(len(y)),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "a": 1.0,
        "b": 0.0,
        "success": False,
        "loss_before": np.nan,
        "loss_after": np.nan,
        "message": "",
    }
    if len(y) == 0:
        row["message"] = "no finite labels"
        return row
    row["loss_before"] = float(np.mean(bce_from_logits(x, y)))
    if row["n_pos"] == 0 or row["n_neg"] == 0:
        row["loss_after"] = row["loss_before"]
        row["message"] = "single class in train_oof"
        return row

    try:
        from scipy.optimize import minimize

        def objective(params):
            a, b = params
            z = a * x + b
            return float(np.mean(bce_from_logits(z, y)))

        def gradient(params):
            a, b = params
            z = a * x + b
            err = sigmoid(z) - y
            return np.asarray([np.mean(err * x), np.mean(err)], dtype=np.float64)

        result = minimize(
            objective,
            x0=np.asarray([1.0, 0.0], dtype=np.float64),
            jac=gradient,
            method="L-BFGS-B",
            options={"maxiter": int(max_iter)},
        )
        if result.success and np.all(np.isfinite(result.x)):
            row["a"] = float(result.x[0])
            row["b"] = float(result.x[1])
            row["success"] = True
            row["message"] = str(result.message)
        else:
            row["message"] = str(result.message)
    except Exception as exc:
        row["message"] = f"fit failed: {exc}"

    calibrated = row["a"] * x + row["b"]
    row["loss_after"] = float(np.mean(bce_from_logits(calibrated, y)))
    return row


def apply_platt(raw_score, params):
    out = np.full_like(raw_score, np.nan, dtype=np.float32)
    for task_idx, row in params.iterrows():
        score = raw_score[:, int(task_idx)].astype(np.float64)
        valid = np.isfinite(score)
        out[valid, int(task_idx)] = (float(row["a"]) * score[valid] + float(row["b"])).astype(np.float32)
    return out


def expected_calibration_error(labels, prob, n_bins=10):
    valid = np.isfinite(labels) & np.isfinite(prob)
    y = labels[valid].astype(np.float64)
    p = prob[valid].astype(np.float64)
    if len(y) == 0:
        return np.nan
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        left, right = edges[idx], edges[idx + 1]
        if idx == n_bins - 1:
            mask = (p >= left) & (p <= right)
        else:
            mask = (p >= left) & (p < right)
        if not np.any(mask):
            continue
        ece += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return ece


def calibration_metrics(logits, labels, task_names, n_bins=10):
    from sklearn import metrics as sk_metrics

    prob = sigmoid(logits)
    rows = []
    for task_idx, task_name in enumerate(task_names):
        valid = np.isfinite(labels[:, task_idx]) & np.isfinite(logits[:, task_idx])
        y = labels[valid, task_idx].astype(np.int64)
        p = prob[valid, task_idx].astype(np.float64)
        row = {
            "task_index": task_idx,
            "task_name": task_name,
            "n_valid": int(len(y)),
            "n_pos": int((y == 1).sum()),
            "n_neg": int((y == 0).sum()),
            "log_loss": np.nan,
            "brier_score": np.nan,
            "ece": np.nan,
        }
        if len(y) and row["n_pos"] > 0 and row["n_neg"] > 0:
            row["log_loss"] = float(sk_metrics.log_loss(y, np.clip(p, 1e-7, 1.0 - 1e-7), labels=[0, 1]))
            row["brier_score"] = float(sk_metrics.brier_score_loss(y, p))
            row["ece"] = expected_calibration_error(y, p, n_bins=n_bins)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_calibration(df):
    return {
        "mean_log_loss": float(pd.to_numeric(df["log_loss"], errors="coerce").mean()),
        "mean_brier_score": float(pd.to_numeric(df["brier_score"], errors="coerce").mean()),
        "mean_ece": float(pd.to_numeric(df["ece"], errors="coerce").mean()),
        "valid_log_loss_tasks": int(pd.to_numeric(df["log_loss"], errors="coerce").notna().sum()),
    }


def save_profile(output_dir, payload, metadata):
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "profile_arrays.npz", **payload)
    for key, value in payload.items():
        if key.endswith("_raw_score") or key.endswith("_prob") or key.endswith("_y") or key.endswith("_row_index"):
            np.save(output_dir / f"{key}.npy", value)
    save_json(output_dir / "profile_metadata.json", metadata)


def parse_args():
    parser = argparse.ArgumentParser(description="Fit task-wise Platt scaling on train OOF teacher scores.")
    parser.add_argument("--input_profile_dir", required=True)
    parser.add_argument("--output_profile_dir", required=True)
    parser.add_argument("--metrics_dir", required=True)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--ece_bins", type=int, default=10)
    parser.add_argument("--copy_selection", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    input_profile_dir = Path(args.input_profile_dir)
    output_profile_dir = Path(args.output_profile_dir)
    metrics_dir = Path(args.metrics_dir)
    arrays = np.load(input_profile_dir / "profile_arrays.npz", allow_pickle=True)
    task_names = arrays["task_names"].astype(str).tolist()

    param_rows = []
    for task_idx, task_name in enumerate(task_names):
        params = fit_platt(
            arrays["train_oof_raw_score"][:, task_idx],
            arrays["train_oof_y"][:, task_idx],
            max_iter=args.max_iter,
        )
        params["task_index"] = task_idx
        params["task_name"] = task_name
        param_rows.append(params)
    params_df = pd.DataFrame(param_rows).sort_values("task_index").reset_index(drop=True)

    payload = {
        "train_oof_raw_score": apply_platt(arrays["train_oof_raw_score"], params_df),
        "valid_raw_score": apply_platt(arrays["valid_raw_score"], params_df),
        "test_raw_score": apply_platt(arrays["test_raw_score"], params_df),
        "train_oof_y": arrays["train_oof_y"],
        "train_oof_row_index": arrays["train_oof_row_index"],
        "train_oof_smiles": arrays["train_oof_smiles"],
        "valid_y": arrays["valid_y"],
        "valid_row_index": arrays["valid_row_index"],
        "valid_smiles": arrays["valid_smiles"],
        "test_y": arrays["test_y"],
        "test_row_index": arrays["test_row_index"],
        "test_smiles": arrays["test_smiles"],
        "task_names": arrays["task_names"],
    }
    for split in ["train_oof", "valid", "test"]:
        payload[f"{split}_prob"] = sigmoid(payload[f"{split}_raw_score"]).astype(np.float32)

    save_profile(
        output_profile_dir,
        payload,
        {
            "source_profile_dir": str(input_profile_dir),
            "score_convention": "Platt-scaled logits in *_raw_score; probabilities in *_prob.",
            "platt_formula": "calibrated_logit = a * raw_score + b",
            "num_tasks": len(task_names),
        },
    )
    params_df.to_csv(output_profile_dir / "platt_parameters.csv", index=False)
    if args.copy_selection and (input_profile_dir / "task_teacher_selection.csv").exists():
        shutil.copy2(input_profile_dir / "task_teacher_selection.csv", output_profile_dir / "task_teacher_selection.csv")

    metrics_dir.mkdir(parents=True, exist_ok=True)
    all_metric_rows = []
    summary = {}
    for split in ["train_oof", "valid", "test"]:
        task_metrics = compute_multitask_metrics(payload[f"{split}_raw_score"], payload[f"{split}_y"], task_names=task_names)
        cal_metrics = calibration_metrics(
            payload[f"{split}_raw_score"],
            payload[f"{split}_y"],
            task_names,
            n_bins=args.ece_bins,
        )
        merged = task_metrics.merge(cal_metrics, on=["task_index", "task_name", "n_valid", "n_pos", "n_neg"], how="left")
        merged["prediction_split"] = split
        merged.to_csv(metrics_dir / f"{split}_per_task_metrics.csv", index=False)
        all_metric_rows.append(merged)
        split_summary = summarize_metrics(task_metrics)
        split_summary.update(summarize_calibration(cal_metrics))
        summary[split] = split_summary

    pd.concat(all_metric_rows, ignore_index=True).to_csv(metrics_dir / "all_per_task_metrics.csv", index=False)
    params_df.to_csv(metrics_dir / "platt_parameters.csv", index=False)
    save_json(metrics_dir / "profile_metric_summary.json", summary)
    print(f"Saved calibrated profile: {output_profile_dir}")
    print(f"Saved calibration metrics: {metrics_dir}")


if __name__ == "__main__":
    main()
