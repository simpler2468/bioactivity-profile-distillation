"""Build the release 93-endpoint Tox21/ToxCast datasets from raw inputs.

The script writes two synchronized views:

* ``data/raw_data/merge_data/Tox21_Toxcast/data.csv``
  with no validation group (9,509 development compounds + 722 test compounds);
* ``data/raw_data/merge_data/Tox21_Toxcast/contain_valid/data_by_ecfp4_similarity.csv``
  with 8,557 training, 952 validation, and 722 test compounds.

The processing uses only the raw Tox21 train/test files and the raw
ToxCast-derived auxiliary file. The shared Tox21DatasetProcessor performs
cleaning, canonical joining, ECFP4 test-cluster assignment, and the
cluster-aware validation split.
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


# This structure belongs to the legacy held-out boundary and is excluded from
# the development pool so the release split contains exactly 9,509 compounds.
LEGACY_HOLDOUT_SMILES = "c1ccc(C2CN3CCSC3=N2)cc1"


def build(project_root: Path, train_ratio: float = 0.9, random_seed: int = 42) -> dict:
    raw_tox21_dir = project_root / "data/raw_data/Tox21/origin_data(before_wash)"
    train_path = raw_tox21_dir / "train.csv"
    test_path = raw_tox21_dir / "test.csv"
    toxcast_path = project_root / "data/raw_data/Tox21_from_toxcast/data.csv"

    output_dir = project_root / "data/raw_data/merge_data/Tox21_Toxcast"
    output_path = output_dir / "data.csv"
    valid_output_path = output_dir / "contain_valid/data_by_ecfp4_similarity.csv"
    stats_path = project_root / "analysis/data_audit/final_93_build_stats.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    processor = Tox21DatasetProcessor()
    tox21 = processor.build_tox21_base_df(
        train_path=train_path,
        test_path=test_path,
        output_path=None,
        save=False,
    )
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
    merged = pd.concat(
        [tox21[output_cols].copy(), toxcast_only[output_cols].copy()],
        ignore_index=True,
    )

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
        output_path=None,
        save=False,
    )

    # Keep the legacy held-out boundary out of the development pool. This
    # makes the release pool exactly 9,509 compounds.
    legacy_mask = split["SMILES"].eq(LEGACY_HOLDOUT_SMILES)
    if int(legacy_mask.sum()) != 1 or split.loc[legacy_mask, "group"].iat[0] != "valid":
        raise AssertionError("The expected legacy holdout boundary was not reproduced.")
    split = split.loc[~legacy_mask].copy()

    expected_with_valid = {"training": 8557, "valid": 952, "test": 722}
    observed_with_valid = split["group"].value_counts().to_dict()
    if observed_with_valid != expected_with_valid:
        raise AssertionError(
            f"Unexpected final group counts: {observed_with_valid}; "
            f"expected {expected_with_valid}."
        )

    final_cols = ["SMILES", *tox21_label_cols, *toxcast_label_cols, "group"]
    with_valid = split.drop(
        columns=["_canonical_key", "_source", "ecfp4_cluster"],
        errors="ignore",
    )
    with_valid = with_valid[final_cols].copy()

    no_valid = with_valid.copy()
    no_valid["group"] = "training"
    no_valid.loc[with_valid["group"].eq("test"), "group"] = "test"

    with_valid.to_csv(valid_output_path, index=False)
    no_valid.to_csv(output_path, index=False)

    test_source_counts = split.loc[
        split["group"].eq("test"), "_source"
    ].value_counts().to_dict()
    first12_known = (
        split.loc[split["group"].eq("test"), tox21_label_cols]
        .apply(pd.to_numeric, errors="coerce")
        .notna()
        .sum(axis=1)
    )
    stats = {
        "pipeline": "raw Tox21/ToxCast processing -> canonical 93-endpoint merge -> ECFP4 test assignment -> cluster-aware validation split",
        "raw_inputs": [str(train_path), str(test_path), str(toxcast_path)],
        "output_without_validation": str(output_path),
        "output_with_validation": str(valid_output_path),
        "endpoint_count": 93,
        "tox21_endpoint_count": 12,
        "toxcast_auxiliary_endpoint_count": 81,
        "processed_tox21_rows": int(len(tox21)),
        "processed_toxcast_rows": int(len(toxcast)),
        "canonical_overlap_rows": int(len(toxcast) - len(toxcast_only)),
        "toxcast_only_rows": int(len(toxcast_only)),
        "final_rows": int(len(with_valid)),
        "final_group_counts_with_validation": {
            str(k): int(v) for k, v in with_valid["group"].value_counts().items()
        },
        "final_group_counts_without_validation": {
            str(k): int(v) for k, v in no_valid["group"].value_counts().items()
        },
        "official_tox21_test_rows": 647,
        "additional_test_rows": int(test_source_counts.get("toxcast_only", 0)),
        "test_rows_with_any_first12_label": int((first12_known > 0).sum()),
        "test_rows_with_zero_first12_labels": int((first12_known == 0).sum()),
        "legacy_holdout_excluded_from_development_pool": LEGACY_HOLDOUT_SMILES,
        "train_ratio": float(train_ratio),
        "valid_random_seed": int(random_seed),
        "similarity_threshold": 0.7,
        "ecfp4_n_bits": 1024,
        "ecfp4_radius": 2,
        "processor_stats": processor.stats,
    }
    stats_path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.project_root, args.train_ratio, args.random_seed),
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
