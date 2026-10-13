import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teacher.common import run_id


def config_to_cli(config):
    args = []
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                args.append(f"--{key}")
            continue
        args.extend([f"--{key}", str(value)])
    return args


def run_command(command, dry_run=False):
    print(" ".join(str(x) for x in command))
    if dry_run:
        return
    subprocess.run(command, check=True)


def parse_int_list(value):
    return [int(x) for x in str(value).split(",") if str(x).strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP teacher folds using a saved HPO best_config.json.")
    parser.add_argument("--dataset", default="Tox21_Toxcast")
    parser.add_argument("--feature", required=True)
    parser.add_argument("--feature_npz", required=True)
    parser.add_argument("--fold_root", required=True)
    parser.add_argument("--best_config", required=True)
    parser.add_argument("--output_root", default=str(PROJECT_ROOT / "teacher"))
    parser.add_argument("--split_id", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seeds", default="2026,2027,2028,2029,2030")
    parser.add_argument("--folds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = parse_int_list(args.seeds)
    folds = parse_int_list(args.folds)
    best_payload = json.loads(Path(args.best_config).read_text(encoding="utf-8"))
    config = dict(best_payload["config"])

    train_script = PROJECT_ROOT / "teacher/run_tabular_teacher.py"
    build_script = PROJECT_ROOT / "teacher/build_teacher_profile.py"
    eval_script = PROJECT_ROOT / "teacher/evaluate_teacher_profiles.py"
    output_root = Path(args.output_root)
    profiles_root = output_root / args.dataset / "profiles"
    metrics_root = output_root / args.dataset / "metrics"
    fold_root = Path(args.fold_root)

    for seed in seeds:
        for fold in folds:
            fold_csv = fold_root / f"training_cluster_CV_{fold}.csv"
            current_run_id = run_id(args.feature, "MLP", args.split_id, fold, seed)
            run_dir = output_root / args.dataset / "runs" / current_run_id
            done_file = run_dir / "test_predictions.npz"
            if done_file.exists() and not args.force:
                print(f"Skip existing run: {current_run_id}")
                continue

            command = [
                args.python,
                str(train_script),
                "--dataset",
                args.dataset,
                "--feature",
                args.feature,
                "--model",
                "MLP",
                "--split_id",
                args.split_id,
                "--fold",
                str(fold),
                "--seed",
                str(seed),
                "--feature_npz",
                str(args.feature_npz),
                "--fold_csv",
                str(fold_csv),
                "--output_root",
                str(output_root),
                "--epochs",
                str(args.epochs),
                "--patience",
                str(args.patience),
                "--eval_batch_size",
                str(args.eval_batch_size),
                "--monitor",
                "mean_roc_auc",
            ]
            if args.device:
                command.extend(["--device", args.device])
            command.extend(config_to_cli(config))
            run_command(command, dry_run=args.dry_run)

        build_seed = [
            args.python,
            str(build_script),
            "--dataset",
            args.dataset,
            "--feature",
            args.feature,
            "--model",
            "MLP",
            "--split_id",
            args.split_id,
            "--feature_npz",
            str(args.feature_npz),
            "--output_root",
            str(output_root),
            "--folds",
            ",".join(str(x) for x in folds),
            "--seeds",
            str(seed),
        ]
        run_command(build_seed, dry_run=args.dry_run)

    build_ensemble = [
        args.python,
        str(build_script),
        "--dataset",
        args.dataset,
        "--feature",
        args.feature,
        "--model",
        "MLP",
        "--split_id",
        args.split_id,
        "--feature_npz",
        str(args.feature_npz),
        "--output_root",
        str(output_root),
        "--folds",
        ",".join(str(x) for x in folds),
        "--seeds",
        ",".join(str(x) for x in seeds),
        "--ensemble_name",
        f"ensemble{len(seeds)}",
    ]
    run_command(build_ensemble, dry_run=args.dry_run)

    evaluate = [
        args.python,
        str(eval_script),
        "--profiles_root",
        str(profiles_root),
        "--output_root",
        str(metrics_root),
    ]
    run_command(evaluate, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
