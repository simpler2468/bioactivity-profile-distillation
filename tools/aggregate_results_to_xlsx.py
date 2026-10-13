"""Collect whatever has already been run into one inspection workbook.

The scanner is intentionally tolerant: an incomplete experiment directory is
still useful, so missing model families are reported as missing rows rather
than treated as an error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _flatten(value, prefix=""):
    rows = {}
    if isinstance(value, dict):
        for key, item in value.items():
            key = f"{prefix}.{key}" if prefix else str(key)
            rows.update(_flatten(item, key))
    elif isinstance(value, (list, tuple)):
        rows[prefix] = json.dumps(value, ensure_ascii=False)
    else:
        rows[prefix] = value
    return rows


def _relative(path, project_root):
    return str(path.resolve().relative_to(project_root.resolve()))


def collect_metric_tables(project_root):
    records = []
    search_roots = [project_root / name for name in ("teacher", "student", "hpo", "analysis")]
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*.csv"):
            name = path.name.lower()
            if "metric" not in name and "summary" not in name and name != "trials.csv":
                continue
            try:
                table = pd.read_csv(path)
            except Exception as exc:
                records.append({"source_file": _relative(path, project_root), "read_error": repr(exc)})
                continue
            if table.empty:
                records.append({"source_file": _relative(path, project_root), "rows": 0})
                continue
            table = table.copy()
            table.insert(0, "source_file", _relative(path, project_root))
            records.extend(table.to_dict(orient="records"))
    return pd.DataFrame(records)


def collect_json_summaries(project_root):
    records = []
    search_roots = [project_root / name for name in ("teacher", "student", "hpo", "analysis")]
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            name = path.name.lower()
            if not ("summary" in name or name in {"best_config.json", "run_config.json", "profile_metadata.json"}):
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                row = _flatten(payload)
                row["source_file"] = _relative(path, project_root)
                records.append(row)
            except Exception as exc:
                records.append({"source_file": _relative(path, project_root), "read_error": repr(exc)})
    return pd.DataFrame(records)


def collect_file_inventory(project_root):
    rows = []
    for root_name in ("teacher", "student", "hpo", "analysis"):
        root = project_root / root_name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file():
                rows.append(
                    {
                        "source_file": _relative(path, project_root),
                        "suffix": path.suffix,
                        "size_bytes": path.stat().st_size,
                    }
                )
    return pd.DataFrame(rows)


def write_workbook(project_root, output_path):
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metric_rows = collect_metric_tables(project_root)
    summary_rows = collect_json_summaries(project_root)
    inventory = collect_file_inventory(project_root)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        (metric_rows if not metric_rows.empty else pd.DataFrame([{"status": "No metric CSV found"}])).to_excel(
            writer, sheet_name="metric_rows", index=False
        )
        (summary_rows if not summary_rows.empty else pd.DataFrame([{"status": "No summary JSON found"}])).to_excel(
            writer, sheet_name="summaries", index=False
        )
        (inventory if not inventory.empty else pd.DataFrame([{"status": "No result files found"}])).to_excel(
            writer, sheet_name="file_inventory", index=False
        )
    return output_path, len(metric_rows), len(summary_rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate available teacher/student/HPO outputs into XLSX.")
    parser.add_argument("--project_root", default=None)
    parser.add_argument("--output", default="analysis/result_summaries/model_results_summary.xlsx")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[1]
    output, metric_count, summary_count = write_workbook(project_root, args.output)
    print(f"Saved {output}")
    print(f"Collected {metric_count} metric rows and {summary_count} JSON summaries.")


if __name__ == "__main__":
    main()
