import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def parse_profile_name(name):
    items = {}
    for part in str(name).split("__"):
        if "=" in part:
            key, value = part.split("=", 1)
            items[key] = value
    return items


def summarize_profile(profile_metrics_dir):
    test_csv = profile_metrics_dir / "test_per_task_metrics.csv"
    summary_json = profile_metrics_dir / "profile_metric_summary.json"
    if not test_csv.exists():
        return None
    df = pd.read_csv(test_csv)
    auc = pd.to_numeric(df.get("roc_auc"), errors="coerce")
    tox21_mask = df["task_name"].isin(TOX21_12_TASKS)
    tox21_auc = pd.to_numeric(df.loc[tox21_mask, "roc_auc"], errors="coerce")
    row = {
        "profile": profile_metrics_dir.name,
        "test_tox21_12_mean_auc": float(tox21_auc.mean()),
        "test_tox21_12_valid_tasks": int(tox21_auc.notna().sum()),
        "test_all_93_mean_auc": float(auc.mean()),
        "test_all_93_valid_tasks": int(auc.notna().sum()),
    }
    row.update(parse_profile_name(profile_metrics_dir.name))
    if summary_json.exists():
        try:
            summary = json.loads(summary_json.read_text(encoding="utf-8"))
            if "test" in summary:
                for key, value in summary["test"].items():
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        row[f"test_summary_{key}"] = float(value)
        except Exception:
            pass
    return row


def main():
    parser = argparse.ArgumentParser(description="Summarize teacher profile test AUC values.")
    parser.add_argument("--metrics_root", default=str(PROJECT_ROOT / "teacher/Tox21_Toxcast/metrics"))
    parser.add_argument("--output_csv", default=None)
    args = parser.parse_args()

    metrics_root = Path(args.metrics_root)
    rows = []
    for child in sorted(metrics_root.iterdir() if metrics_root.exists() else []):
        if child.is_dir():
            row = summarize_profile(child)
            if row:
                rows.append(row)
    if not rows:
        raise SystemExit(f"No profile metric directories found under {metrics_root}")
    out = pd.DataFrame(rows).sort_values("test_tox21_12_mean_auc", ascending=False)
    output_csv = Path(args.output_csv) if args.output_csv else metrics_root / "teacher_profile_test_auc_summary.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(out[["profile", "test_tox21_12_mean_auc", "test_tox21_12_valid_tasks", "test_all_93_mean_auc", "test_all_93_valid_tasks"]].to_string(index=False))
    print(f"Saved: {output_csv}")


if __name__ == "__main__":
    main()
