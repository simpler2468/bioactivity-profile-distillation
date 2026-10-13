"""Rebuild the 93-endpoint dataset directly from raw Tox21 and ToxCast files.

Inputs:
    data/raw_data/Tox21/origin_data(before_wash)/train.csv
    data/raw_data/Tox21/origin_data(before_wash)/test.csv
    data/raw_data/Tox21_from_toxcast/data.csv

The Tox21 processor keeps the 647 original test rows, while training/test
canonical overlaps are resolved with test priority. ToxCast labels are then
joined by canonical structure. ToxCast-only compounds are assigned to test if
their ECFP4 similarity cluster contains an official Tox21 test compound;
remaining ToxCast-only compounds enter the training pool. Validation is drawn
from the training pool by whole ECFP4 clusters.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.tox21_dataset_processor import Tox21DatasetProcessor


def build(project_root: Path, train_ratio: float = 0.9, random_seed: int = 42) -> dict:
    raw_tox21_dir = project_root / "data/raw_data/Tox21/origin_data(before_wash)"
    train_path = raw_tox21_dir / "train.csv"
    test_path = raw_tox21_dir / "test.csv"
    toxcast_path = project_root / "data/raw_data/Tox21_from_toxcast/data.csv"

    output_dir = project_root / "data/raw_data/merge_data/Tox21_Toxcast/from_raw_rebuild"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "data_by_ecfp4_similarity.csv"
    tox21_output_path = output_dir / "processed_tox21.csv"
    toxcast_output_path = output_dir / "processed_toxcast.csv"
    test_provenance_path = output_dir / "test_provenance.csv"
    stats_path = output_dir / "build_stats.json"

    processor = Tox21DatasetProcessor()
    tox21 = processor.build_tox21_base_df(
        train_path=train_path,
        test_path=test_path,
        output_path=tox21_output_path,
        save=False,
    )
    processed_tox21 = tox21.copy()
    tox21_label_cols = list(tox21.columns[1:-1])

    toxcast_raw = pd.read_csv(toxcast_path, dtype=object)
    toxcast_label_cols = [
        col for col in toxcast_raw.columns if col not in {"SMILES", "group"}
    ]
    if len(tox21_label_cols) != 12 or len(toxcast_label_cols) != 81:
        raise ValueError(
            f"Expected 12 Tox21 and 81 ToxCast labels; got "
            f"{len(tox21_label_cols)} and {len(toxcast_label_cols)}."
        )
    toxcast = processor.build_processed_df(
        toxcast_raw,
        label_cols=toxcast_label_cols,
        default_group=pd.NA,
    )

    # Canonical keys are used only for joining. The output keeps the original
    # Tox21 test rows, including duplicate rows and unstandardized records.
    tox21 = tox21.copy()
    tox21["_canonical_key"] = [
        processor.standardize_smiles(smiles)[0] for smiles in tox21["SMILES"]
    ]
    tox21["_source"] = np.where(
        tox21["group"].eq("test"), "tox21_official_test", "tox21_training"
    )

    toxcast_by_key = toxcast.set_index("SMILES")
    for col in toxcast_label_cols:
        tox21[col] = tox21["_canonical_key"].map(toxcast_by_key[col])

    tox21_keys = set(tox21["_canonical_key"].dropna())
    toxcast_only = toxcast.loc[~toxcast["SMILES"].isin(tox21_keys)].copy()
    toxcast_only["_source"] = "toxcast_only"
    for col in tox21_label_cols:
        toxcast_only[col] = np.nan
    toxcast_only["group"] = pd.NA
    toxcast_only["_canonical_key"] = toxcast_only["SMILES"]

    output_cols = [
        "SMILES",
        *tox21_label_cols,
        *toxcast_label_cols,
        "group",
        "_canonical_key",
        "_source",
    ]
    tox21_rows = tox21[output_cols].copy()
    toxcast_only = toxcast_only[output_cols].copy()
    merged = pd.concat([tox21_rows, toxcast_only], ignore_index=True)

    assigned = processor.assign_group(
        merged,
        similarity_threshold=0.7,
        n_bits=1024,
        radius=2,
    )
    split = processor.valid_from_train_by_ecfp4_similarity(
        assigned,
        train_ratio=train_ratio,
        random_seed=random_seed,
        similarity_threshold=0.7,
        n_bits=1024,
        radius=2,
        output_path=output_path,
        save=False,
    )

    test_source_counts = (
        split.loc[split["group"].eq("test"), "_source"]
        .value_counts(dropna=False)
        .to_dict()
    )
    first12_test_known = (
        split.loc[split["group"].eq("test"), tox21_label_cols]
        .apply(pd.to_numeric, errors="coerce")
        .notna()
        .sum(axis=1)
    )
    if int((split["group"] == "test").sum()) != 647 + int(test_source_counts.get("toxcast_only", 0)):
        raise AssertionError("Unexpected test-source accounting.")
    if int((split.loc[split["_source"] == "toxcast_only", tox21_label_cols]
            .apply(pd.to_numeric, errors="coerce").notna().sum().sum())) != 0:
        raise AssertionError("ToxCast-only rows must not carry Tox21 labels.")

    test_provenance = split.loc[
        split["group"].eq("test"), ["SMILES", "group", "_source"]
    ].copy()
    test_provenance["first12_known"] = first12_test_known.to_numpy()
    test_provenance.to_csv(test_provenance_path, index=False)

    final = split.drop(columns=["_canonical_key", "_source"])
    final = final[["SMILES"] + tox21_label_cols + toxcast_label_cols + ["group"]]
    final.to_csv(output_path, index=False)
    processed_tox21.to_csv(tox21_output_path, index=False)
    toxcast.to_csv(toxcast_output_path, index=False)

    stats = {
        "pipeline": "raw Tox21 train/test -> processed Tox21 -> canonical ToxCast join -> ECFP4 test-cluster assignment -> ECFP4 validation split",
        "raw_tox21_train": str(train_path),
        "raw_tox21_test": str(test_path),
        "raw_toxcast": str(toxcast_path),
        "processed_tox21_output": str(tox21_output_path),
        "processed_toxcast_output": str(toxcast_output_path),
        "test_provenance_output": str(test_provenance_path),
        "output_csv": str(output_path),
        "stats_json": str(stats_path),
        "endpoint_count": 93,
        "tox21_endpoint_count": 12,
        "toxcast_auxiliary_endpoint_count": 81,
        "raw_tox21_train_rows": int(len(pd.read_csv(train_path))),
        "raw_tox21_test_rows": int(len(pd.read_csv(test_path))),
        "processed_tox21_rows": int(len(tox21)),
        "processed_tox21_group_counts": {
            str(k): int(v) for k, v in tox21["group"].value_counts().items()
        },
        "processed_toxcast_rows": int(len(toxcast)),
        "canonical_overlap_rows": int(len(toxcast) - len(toxcast_only)),
        "toxcast_only_rows": int(len(toxcast_only)),
        "merged_rows_before_split": int(len(merged)),
        "final_rows": int(len(final)),
        "final_group_counts": {
            str(k): int(v) for k, v in final["group"].value_counts().items()
        },
        "test_source_counts": {str(k): int(v) for k, v in test_source_counts.items()},
        "test_rows_with_any_first12_label": int((first12_test_known > 0).sum()),
        "test_rows_with_zero_first12_labels": int((first12_test_known == 0).sum()),
        "unstandardized_tox21_rows_retained_for_row_identity": int(
            tox21["_canonical_key"].isna().sum()
        ),
        "train_ratio": float(train_ratio),
        "valid_random_seed": int(random_seed),
        "similarity_threshold": 0.7,
        "ecfp4_n_bits": 1024,
        "ecfp4_radius": 2,
        "processor_stats": processor.stats,
    }
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(build(args.project_root, args.train_ratio, args.random_seed), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
