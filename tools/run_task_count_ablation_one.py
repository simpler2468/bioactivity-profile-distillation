import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment_naming import short_value, student_run_id


def parse_seeds(value):
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def compact_seed(seed):
    text = str(seed)
    if text.isdigit() and len(text) == 4 and text.startswith("20"):
        return text[-2:]
    return text


def add_config_args(cmd, config):
    for key, value in config.items():
        if value is None:
            continue
        option = f"--{key}"
        if isinstance(value, bool):
            if value:
                cmd.append(option)
        else:
            cmd.extend([option, str(value)])


def run(cmd):
    print(" ".join(str(part) for part in cmd), flush=True)
    subprocess.check_call([str(part) for part in cmd], cwd=PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Run one eGIN BCE task-count ablation group and its seed ensemble.")
    parser.add_argument("--best_config", required=True)
    parser.add_argument("--dataset", default="Tox21_Toxcast")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--task_file", required=True)
    parser.add_argument("--graph_dir", required=True)
    parser.add_argument("--output_root", default="student")
    parser.add_argument("--split_id", default="contain_valid_task_count_ablation")
    parser.add_argument("--seeds", default="2026,2027,2028,2029,2030")
    parser.add_argument("--monitor", default="tox21_12_mean_roc_auc")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    payload = json.loads(Path(args.best_config).read_text())
    config = dict(payload["config"])
    seeds = parse_seeds(args.seeds)

    for seed in seeds:
        run_id = student_run_id("graph", "GIN", args.tasks, "mask_bce", args.split_id, seed)
        run_dir = Path(args.output_root) / args.dataset / "runs" / run_id
        if (run_dir / "test_predictions.npz").exists():
            print(f"Skip existing task-count ablation run: {run_id}", flush=True)
            continue

        cmd = [
            sys.executable,
            PROJECT_ROOT / "student" / "run_graph_student.py",
            "--dataset",
            args.dataset,
            "--model",
            "GIN",
            "--feature",
            "graph",
            "--tasks",
            args.tasks,
            "--task_file",
            args.task_file,
            "--graph_dir",
            args.graph_dir,
            "--output_root",
            args.output_root,
            "--split_id",
            args.split_id,
            "--seed",
            str(seed),
            "--loss_name",
            "mask_bce",
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--eval_batch_size",
            str(args.eval_batch_size),
            "--num_workers",
            str(args.num_workers),
            "--monitor",
            args.monitor,
            "--device",
            args.device,
        ]
        add_config_args(cmd, config)
        run(cmd)

    task_alias = short_value("tasks", args.tasks)
    split_alias = short_value("split", args.split_id)
    pattern = f"f=g__m=gin__t={task_alias}__l=bce__sp={split_alias}__s="
    ensemble_cmd = [
        sys.executable,
        PROJECT_ROOT / "student" / "evaluate_seed_ensemble.py",
        "--dataset",
        args.dataset,
        "--runs_root",
        str(Path(args.output_root) / args.dataset / "runs"),
        "--output_root",
        args.output_root,
        "--seeds",
        ",".join(compact_seed(seed) for seed in seeds),
        "--pattern",
        pattern,
        "--summary_name",
        "task_count_ablation_egin_bce_ensemble_summary.csv",
    ]
    run(ensemble_cmd)


if __name__ == "__main__":
    main()
