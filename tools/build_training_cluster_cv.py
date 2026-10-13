"""Generate cluster-based cross-validation files from the validated dataset.

Input:
    data/raw_data/merge_data/Tox21_Toxcast/contain_valid/
    data_by_ecfp4_similarity.csv

Output:
    data/raw_data/merge_data/Tox21_Toxcast/contain_valid/training_cluster_CV/

Only rows labelled ``training`` in the input are used. ECFP4 clusters are
kept intact, so a cluster cannot be split across the training and validation
folds. Each fold is written both with a validation group and as a no-valid
view in which validation rows are relabelled as training rows.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.tox21_dataset_processor import Tox21DatasetProcessor


def build(
    project_root: Path,
    n_folds: int = 10,
    random_seed: int = 42,
    similarity_threshold: float = 0.7,
    n_bits: int = 1024,
    radius: int = 2,
) -> dict:
    input_path = (
        project_root
        / "data/raw_data/merge_data/Tox21_Toxcast/contain_valid/data_by_ecfp4_similarity.csv"
    )
    output_dir = (
        project_root
        / "data/raw_data/merge_data/Tox21_Toxcast/contain_valid/training_cluster_CV"
    )
    stats_path = project_root / "analysis/data_audit/training_cluster_cv_stats.json"

    if not input_path.exists():
        raise FileNotFoundError(f"Validated dataset not found: {input_path}")

    processor = Tox21DatasetProcessor()
    outputs = processor.training_cluster_cv_by_ecfp4_similarity(
        input_path,
        n_folds=n_folds,
        random_seed=random_seed,
        output_dir=output_dir,
        similarity_threshold=similarity_threshold,
        n_bits=n_bits,
        radius=radius,
        include_cluster_col=False,
        save=True,
    )

    stats = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "n_folds": int(n_folds),
        "random_seed": int(random_seed),
        "similarity_threshold": float(similarity_threshold),
        "ecfp4_n_bits": int(n_bits),
        "ecfp4_radius": int(radius),
        "outputs": outputs,
        "processor_stats": processor.stats,
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--n_folds", type=int, default=10)
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.project_root, args.n_folds, args.random_seed),
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
