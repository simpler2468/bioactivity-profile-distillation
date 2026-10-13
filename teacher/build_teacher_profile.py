import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teacher.common import load_feature_npz, load_prediction_npz, profile_id, run_id, save_json


def collect_seed_profile(args, seed):
    full = load_feature_npz(args.feature_npz)
    n_rows = len(full["smiles"])
    n_tasks = len(full["task_names"])
    train_mask = full["groups"] == "training"
    valid_mask = full["groups"] == "valid"
    test_mask = full["groups"] == "test"

    train_oof = np.full((n_rows, n_tasks), np.nan, dtype=np.float32)
    train_oof_y = np.asarray(full["y"], dtype=np.float32)
    oof_seen = np.zeros(n_rows, dtype=np.int32)
    valid_preds = []
    valid_rows = None
    test_preds = []
    test_rows = None

    for fold in args.folds:
        current_run_id = run_id(args.feature, args.model, args.split_id, fold, seed)
        run_dir = Path(args.output_root) / args.dataset / "runs" / current_run_id
        oof = load_prediction_npz(run_dir / "oof_predictions.npz")
        train_oof[oof["row_index"]] = oof["raw_score"]
        oof_seen[oof["row_index"]] += 1

        valid_path = run_dir / "valid_predictions.npz"
        if valid_path.exists():
            valid = load_prediction_npz(valid_path)
            if len(valid["row_index"]):
                valid_rows = valid["row_index"] if valid_rows is None else valid_rows
                if not np.array_equal(valid_rows, valid["row_index"]):
                    raise ValueError(f"valid row_index mismatch in {run_dir}")
                valid_preds.append(valid["raw_score"])

        test = load_prediction_npz(run_dir / "test_predictions.npz")
        test_rows = test["row_index"] if test_rows is None else test_rows
        if not np.array_equal(test_rows, test["row_index"]):
            raise ValueError(f"test row_index mismatch in {run_dir}")
        test_preds.append(test["raw_score"])

    if valid_preds:
        valid_pred = np.nanmean(np.stack(valid_preds, axis=0), axis=0).astype(np.float32)
        valid_y = full["y"][valid_rows]
        valid_smiles = full["smiles"][valid_rows]
    else:
        valid_rows = np.where(valid_mask)[0].astype(np.int64)
        valid_pred = np.zeros((0, n_tasks), dtype=np.float32)
        valid_y = np.zeros((0, n_tasks), dtype=np.float32)
        valid_smiles = np.asarray([], dtype=str)

    test_pred = np.nanmean(np.stack(test_preds, axis=0), axis=0).astype(np.float32)
    test_y = full["y"][test_rows]
    test_smiles = full["smiles"][test_rows]

    train_rows = np.where(train_mask)[0].astype(np.int64)
    missing_oof = train_rows[oof_seen[train_rows] == 0]
    duplicate_oof = train_rows[oof_seen[train_rows] > 1]
    return {
        "seed": seed,
        "train_oof_raw_score": train_oof[train_rows],
        "train_oof_y": train_oof_y[train_rows],
        "train_oof_row_index": train_rows,
        "train_oof_smiles": full["smiles"][train_rows],
        "valid_raw_score": valid_pred,
        "valid_y": valid_y,
        "valid_row_index": valid_rows,
        "valid_smiles": valid_smiles,
        "test_raw_score": test_pred,
        "test_y": test_y,
        "test_row_index": test_rows,
        "test_smiles": test_smiles,
        "task_names": full["task_names"],
        "missing_oof": missing_oof,
        "duplicate_oof": duplicate_oof,
    }


def save_profile(profile_dir, payload, metadata):
    profile_dir.mkdir(parents=True, exist_ok=True)
    np.save(profile_dir / "train_oof_raw_score.npy", payload["train_oof_raw_score"])
    np.save(profile_dir / "train_oof_y.npy", payload["train_oof_y"])
    np.save(profile_dir / "train_oof_row_index.npy", payload["train_oof_row_index"])
    np.save(profile_dir / "valid_raw_score.npy", payload["valid_raw_score"])
    np.save(profile_dir / "valid_y.npy", payload["valid_y"])
    np.save(profile_dir / "valid_row_index.npy", payload["valid_row_index"])
    np.save(profile_dir / "test_raw_score.npy", payload["test_raw_score"])
    np.save(profile_dir / "test_y.npy", payload["test_y"])
    np.save(profile_dir / "test_row_index.npy", payload["test_row_index"])
    np.savez_compressed(
        profile_dir / "profile_arrays.npz",
        train_oof_raw_score=payload["train_oof_raw_score"],
        train_oof_y=payload["train_oof_y"],
        train_oof_row_index=payload["train_oof_row_index"],
        train_oof_smiles=np.asarray(payload["train_oof_smiles"], dtype=object),
        valid_raw_score=payload["valid_raw_score"],
        valid_y=payload["valid_y"],
        valid_row_index=payload["valid_row_index"],
        valid_smiles=np.asarray(payload["valid_smiles"], dtype=object),
        test_raw_score=payload["test_raw_score"],
        test_y=payload["test_y"],
        test_row_index=payload["test_row_index"],
        test_smiles=np.asarray(payload["test_smiles"], dtype=object),
        task_names=np.asarray(payload["task_names"], dtype=object),
    )
    save_json(profile_dir / "profile_metadata.json", metadata)


def parse_args():
    parser = argparse.ArgumentParser(description="Build complete OOF/valid/test teacher profiles from fold predictions.")
    parser.add_argument("--dataset", default="Tox21_Toxcast")
    parser.add_argument("--feature", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split_id", required=True)
    parser.add_argument("--feature_npz", required=True)
    parser.add_argument("--output_root", default="teacher")
    parser.add_argument("--folds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--seeds", required=True, help="Comma-separated seeds.")
    parser.add_argument("--ensemble_name", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    args.folds = [int(x) for x in str(args.folds).split(",") if x != ""]
    seeds = [int(x) for x in str(args.seeds).split(",") if x != ""]
    seed_profiles = []

    for seed in seeds:
        payload = collect_seed_profile(args, seed)
        seed_profiles.append(payload)
        seed_dir = Path(args.output_root) / args.dataset / "profiles" / profile_id(args.feature, args.model, args.split_id, seed)
        save_profile(
            seed_dir,
            payload,
            {
                "dataset": args.dataset,
                "feature": args.feature,
                "model": args.model,
                "split": args.split_id,
                "seed": seed,
                "folds": args.folds,
                "missing_oof_count": int(len(payload["missing_oof"])),
                "duplicate_oof_count": int(len(payload["duplicate_oof"])),
            },
        )
        print(f"Saved seed profile: {seed_dir}")

    if len(seed_profiles) > 1:
        base = seed_profiles[0]
        for payload in seed_profiles[1:]:
            for key in ["train_oof_row_index", "valid_row_index", "test_row_index"]:
                if not np.array_equal(base[key], payload[key]):
                    raise ValueError(f"Cannot ensemble profiles; {key} mismatch.")
        ensemble_payload = dict(base)
        for key in ["train_oof_raw_score", "valid_raw_score", "test_raw_score"]:
            ensemble_payload[key] = np.nanmean(np.stack([p[key] for p in seed_profiles], axis=0), axis=0).astype(np.float32)
        ensemble_name = args.ensemble_name or f"ensemble{len(seed_profiles)}"
        ensemble_dir = Path(args.output_root) / args.dataset / "profiles" / profile_id(args.feature, args.model, args.split_id, ensemble_name)
        save_profile(
            ensemble_dir,
            ensemble_payload,
            {
                "dataset": args.dataset,
                "feature": args.feature,
                "model": args.model,
                "split": args.split_id,
                "seed": ensemble_name,
                "member_seeds": seeds,
                "folds": args.folds,
            },
        )
        print(f"Saved ensemble profile: {ensemble_dir}")


if __name__ == "__main__":
    main()
