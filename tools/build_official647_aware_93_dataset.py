"""Build a 93-endpoint dataset with an official-Tox21-aware test mask.

The source split keeps all 876 test compounds. The 647 rows matching the
processed official Tox21 test group retain their current first 12 labels. The
other 229 test rows retain auxiliary labels but receive missing values in the
first 12 Tox21 columns, so Tox21 metrics use exactly the official benchmark
rows without discarding auxiliary test compounds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(frame: pd.DataFrame) -> str:
    payload = frame.fillna("<NA>").astype(str).to_csv(
        index=False, lineterminator="\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build(project_root: Path) -> dict:
    base_path = project_root / "data/raw_data/merge_data/Tox21_Toxcast/contain_valid/data_by_ecfp4_similarity.csv"
    official_path = project_root / "data/raw_data/Tox21/official benchmark test set.csv"
    output_path = project_root / "data/raw_data/merge_data/Tox21_Toxcast/contain_valid/data_by_ecfp4_similarity_official647_aware.csv"
    audit_dir = project_root / "analysis/data_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = audit_dir / "official647_aware_test_provenance.csv"
    stats_path = audit_dir / "official647_aware_build_stats.json"

    base = pd.read_csv(base_path, dtype=object)
    official_all = pd.read_csv(official_path, dtype=object)
    first12 = list(base.columns[1:13])
    auxiliary = list(base.columns[13:-1])
    if base.columns[-1] != "group" or official_all.columns[-1] != "group":
        raise ValueError("Both files must end with a group column.")
    official = official_all.loc[official_all["group"].eq("test")].copy()
    if len(official) != 647:
        raise ValueError(f"Expected 647 official test rows, found {len(official)}.")

    official_smiles = set(official["SMILES"])
    result = base.copy()
    test_mask = base["group"].eq("test").to_numpy()
    records = []
    for index, row in base.loc[test_mask].iterrows():
        is_official = row["SMILES"] in official_smiles
        before_known = int(
            pd.to_numeric(base.loc[index, first12], errors="coerce").notna().sum()
        )
        auxiliary_known = int(
            pd.to_numeric(base.loc[index, auxiliary], errors="coerce").notna().sum()
        )
        if is_official:
            action = "preserve_current_first12_labels"
        else:
            result.loc[index, first12] = np.nan
            action = "mask_first12_keep_auxiliary_labels"
        after_known = int(
            pd.to_numeric(result.loc[index, first12], errors="coerce").notna().sum()
        )
        records.append(
            {
                "source_row_index": int(index),
                "SMILES": row["SMILES"],
                "official_test_match": bool(is_official),
                "first12_known_before": before_known,
                "first12_known_after": after_known,
                "auxiliary81_known": auxiliary_known,
                "action": action,
            }
        )

    if sum(r["official_test_match"] for r in records) != 647:
        raise AssertionError("The current test split does not contain 647 official rows.")
    if sum(not r["official_test_match"] for r in records) != 229:
        raise AssertionError("The current test split does not contain 229 additional rows.")
    if not all(r["first12_known_after"] == 0 for r in records if not r["official_test_match"]):
        raise AssertionError("Additional test rows still contain first-12 labels.")

    non_test = ~test_mask
    for column in base.columns:
        left = result.loc[non_test, column].fillna("<NA>").astype(str).reset_index(drop=True)
        right = base.loc[non_test, column].fillna("<NA>").astype(str).reset_index(drop=True)
        if not left.equals(right):
            raise AssertionError(f"Training/validation changed in column {column}.")

    result = result[base.columns]
    result.to_csv(output_path, index=False)
    pd.DataFrame(records).to_csv(provenance_path, index=False)

    extra = [r for r in records if not r["official_test_match"]]
    stats = {
        "policy": "retain all 876 test rows; preserve current labels on 647 official rows; mask first 12 Tox21 labels on 229 additional test rows while retaining auxiliary labels",
        "source_base_csv": str(base_path),
        "source_official_csv": str(official_path),
        "output_csv": str(output_path),
        "provenance_csv": str(provenance_path),
        "shape": [int(result.shape[0]), int(result.shape[1])],
        "endpoint_count": int(len(base.columns) - 2),
        "tox21_endpoint_count": 12,
        "auxiliary_endpoint_count": 81,
        "group_counts": {str(k): int(v) for k, v in result["group"].value_counts().items()},
        "test_rows": 876,
        "official_test_rows": 647,
        "additional_test_rows": 229,
        "additional_test_rows_with_first12_before_mask": int(
            sum(r["first12_known_before"] > 0 for r in extra)
        ),
        "additional_test_first12_labels_masked": 229,
        "additional_test_rows_with_auxiliary_labels_retained": int(
            sum(r["auxiliary81_known"] > 0 for r in extra)
        ),
        "training_valid_unchanged": True,
        "official_rows_first12_preserved": 647,
        "source_digest": digest(base),
        "output_digest": digest(result),
    }
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    print(json.dumps(build(args.project_root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
