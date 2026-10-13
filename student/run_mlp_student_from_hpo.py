import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from student.run_mlp_student import student_run_id


def config_to_cli(config):
    args = []
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                args.append(f"--{key}")
            continue
        args.extend([f"--{key}", str(value)])
    return args


def parse_int_list(value):
    return [int(x) for x in str(value).split(",") if str(x).strip()]


def run_command(command, dry_run=False):
    print(" ".join(str(x) for x in command))
    if dry_run:
        return
    subprocess.run(command, check=True)


def summarize_runs(output_root, dataset, run_ids, summary_path):
    rows = []
    for run_id in run_ids:
        run_dir = Path(output_root) / dataset / "runs" / run_id
        config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        for split in ["train", "valid", "test"]:
            metrics_path = run_dir / f"{split}_metrics.csv"
            if not metrics_path.exists():
                continue
            df = pd.read_csv(metrics_path)
            rows.append(
                {
                    "run_id": run_id,
                    "feature": config["feature"],
                    "tasks": config["tasks"],
                    "seed": config["seed"],
                    "prediction_split": split,
                    "mean_roc_auc": float(pd.to_numeric(df["roc_auc"], errors="coerce").mean()),
                    "mean_average_precision": float(pd.to_numeric(df["average_precision"], errors="coerce").mean()),
                    "valid_auc_tasks": int(pd.to_numeric(df["roc_auc"], errors="coerce").notna().sum()),
                    "epochs": len(pd.read_csv(run_dir / "epoch_metrics.csv")),
                }
            )
    summary = pd.DataFrame(rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run fixed-split student MLP seeds using an HPO best config.")
    parser.add_argument("--dataset", default="Tox21_Toxcast")
    parser.add_argument("--feature", required=True)
    parser.add_argument("--tasks", choices=["tox21_12", "all_93"], required=True)
    parser.add_argument("--feature_npz", required=True)
    parser.add_argument("--best_config", required=True)
    parser.add_argument("--output_root", default=str(PROJECT_ROOT / "student"))
    parser.add_argument("--split_id", default="contain_valid_ecfp4_similarity_hpo")
    parser.add_argument("--loss_name", default="mask_bce")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seeds", default="2026,2027,2028,2029,2030")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--teacher_profile_dir", default=None)
    parser.add_argument("--soft_profile_mode", default="full", choices=["full", "target_only", "non_target"])
    parser.add_argument("--soft_cls_weight", type=float, default=0.0)
    parser.add_argument("--soft_reg_weight", type=float, default=0.0)
    parser.add_argument("--aux_reduction", default="observation")
    parser.add_argument("--aux_target_clip", type=float, default=0.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = parse_int_list(args.seeds)
    best_payload = json.loads(Path(args.best_config).read_text(encoding="utf-8"))
    if "base_config" in best_payload:
        config = dict(best_payload["base_config"])
        loss_config = dict(best_payload.get("loss_config", {}))
        if args.soft_cls_weight == 0.0:
            args.soft_cls_weight = float(loss_config.get("soft_cls_weight", 0.0))
        if args.soft_reg_weight == 0.0:
            args.soft_reg_weight = float(loss_config.get("soft_reg_weight", 0.0))
        args.aux_reduction = str(loss_config.get("aux_reduction", args.aux_reduction))
        args.aux_target_clip = float(loss_config.get("aux_target_clip", args.aux_target_clip))
    else:
        config = dict(best_payload["config"])
    train_script = PROJECT_ROOT / "student/run_mlp_student.py"
    output_root = Path(args.output_root)
    run_ids = []

    for seed in seeds:
        run_id = student_run_id(args.feature, "MLP", args.tasks, args.loss_name, args.split_id, seed)
        run_ids.append(run_id)
        done_file = output_root / args.dataset / "runs" / run_id / "test_predictions.npz"
        if done_file.exists() and not args.force:
            print(f"Skip existing student run: {run_id}")
            continue
        command = [
            args.python,
            str(train_script),
            "--dataset",
            args.dataset,
            "--feature",
            args.feature,
            "--tasks",
            args.tasks,
            "--feature_npz",
            str(args.feature_npz),
            "--output_root",
            str(output_root),
            "--split_id",
            args.split_id,
            "--seed",
            str(seed),
            "--loss_name",
            args.loss_name,
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
        if args.teacher_profile_dir:
            command.extend(["--teacher_profile_dir", args.teacher_profile_dir])
        command.extend(["--soft_profile_mode", args.soft_profile_mode])
        command.extend(
            [
                "--soft_cls_weight",
                str(args.soft_cls_weight),
                "--soft_reg_weight",
                str(args.soft_reg_weight),
                "--aux_reduction",
                args.aux_reduction,
                "--aux_target_clip",
                str(args.aux_target_clip),
            ]
        )
        command.extend(config_to_cli(config))
        run_command(command, dry_run=args.dry_run)

    if not args.dry_run:
        summarize_runs(
            output_root,
            args.dataset,
            run_ids,
            output_root / args.dataset / "metrics" / f"feature={args.feature}__model=MLP__tasks={args.tasks}__loss={args.loss_name}__split={args.split_id}__summary.csv",
        )


if __name__ == "__main__":
    main()
