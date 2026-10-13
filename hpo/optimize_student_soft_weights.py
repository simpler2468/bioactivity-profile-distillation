import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from student.run_mlp_student import TOX21_12_TASKS, student_run_id
from teacher.common import save_json

LOSS_WEIGHT_CHOICES = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]


def require_optuna():
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit("Optuna is not installed. Install it with: pip install optuna") from exc
    return optuna


def safe_name(value):
    return str(value).replace(" ", "-").replace("/", "-").replace("\\", "-")


def config_to_cli(config):
    args = []
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                args.append(f"--{key}")
            continue
        args.extend([f"--{key}", str(value)])
    return args


def valid_objective_auc(metrics_csv, objective_tasks):
    df = pd.read_csv(metrics_csv)
    if objective_tasks == "tox21_12":
        df = df[df["task_name"].isin(TOX21_12_TASKS)]
    elif objective_tasks != "all":
        raise ValueError(f"Unsupported objective_tasks: {objective_tasks}")
    return float(pd.to_numeric(df["roc_auc"], errors="coerce").mean())


def load_base_config(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "config" not in payload:
        raise KeyError(f"{path} does not contain a 'config' field.")
    return payload, dict(payload["config"])


def run_id_for_args(args, split_id):
    if args.student_type == "mlp":
        return student_run_id(args.feature, "MLP", args.tasks, "mask_bce_soft", split_id, args.seed)
    return student_run_id("graph", args.model, args.tasks, "mask_bce_soft", split_id, args.seed)


def build_train_command(args, base_config, loss_config, split_id, trial_output_root):
    if args.student_type == "mlp":
        train_script = PROJECT_ROOT / "student/run_mlp_student.py"
        command = [
            args.python, str(train_script),
            "--dataset", args.dataset,
            "--feature", args.feature,
            "--tasks", args.tasks,
            "--feature_npz", str(args.feature_npz),
            "--output_root", str(trial_output_root),
            "--split_id", split_id,
            "--seed", str(args.seed),
            "--loss_name", "mask_bce_soft",
            "--epochs", str(args.epochs),
            "--patience", str(args.patience),
            "--eval_batch_size", str(args.eval_batch_size),
            "--monitor", "mean_roc_auc",
        ]
    else:
        train_script = PROJECT_ROOT / "student/run_graph_student.py"
        command = [
            args.python, str(train_script),
            "--dataset", args.dataset,
            "--model", args.model,
            "--feature", "graph",
            "--tasks", args.tasks,
            "--graph_dir", str(args.graph_dir),
            "--output_root", str(trial_output_root),
            "--split_id", split_id,
            "--seed", str(args.seed),
            "--loss_name", "mask_bce_soft",
            "--epochs", str(args.epochs),
            "--patience", str(args.patience),
            "--eval_batch_size", str(args.eval_batch_size),
            "--num_workers", str(args.num_workers),
            "--monitor", "mean_roc_auc",
            "--soft_profile_mode", args.soft_profile_mode,
        ]
    if args.device:
        command.extend(["--device", args.device])
    command.extend(["--teacher_profile_dir", args.teacher_profile_dir])
    command.extend(config_to_cli(loss_config))
    command.extend(config_to_cli(base_config))
    return command


def parse_args():
    parser = argparse.ArgumentParser(description="Grid HPO for soft-label loss weight with fixed BCE-only base config.")
    parser.add_argument("--dataset", default="Tox21_Toxcast")
    parser.add_argument("--student_type", choices=["mlp", "graph"], required=True)
    parser.add_argument("--model", choices=["GIN"], default=None)
    parser.add_argument("--feature", default=None)
    parser.add_argument("--tasks", choices=["all_93"], default="all_93")
    parser.add_argument("--feature_npz", default=None)
    parser.add_argument("--graph_dir", default=None)
    parser.add_argument("--base_best_config", required=True)
    parser.add_argument("--teacher_profile_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--n_trials", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--objective_tasks", choices=["tox21_12", "all"], default="tox21_12")
    parser.add_argument("--study_name", default=None)
    parser.add_argument("--storage", default=None)
    parser.add_argument("--weight_choices", default=",".join(str(x) for x in LOSS_WEIGHT_CHOICES))
    parser.add_argument("--soft_profile_mode", default="full", choices=["full", "target_only", "non_target"])
    return parser.parse_args()


def validate_args(args):
    if args.student_type == "mlp":
        if not args.feature:
            raise ValueError("--feature is required for --student_type mlp.")
        if not args.feature_npz:
            raise ValueError("--feature_npz is required for --student_type mlp.")
    else:
        if not args.model:
            raise ValueError("--model is required for --student_type graph.")
        if not args.graph_dir:
            raise ValueError("--graph_dir is required for --student_type graph.")


def main():
    args = parse_args()
    validate_args(args)
    optuna = require_optuna()
    weight_choices = [float(item) for item in str(args.weight_choices).split(",") if item.strip()]
    n_trials = args.n_trials if args.n_trials is not None else len(weight_choices)
    base_payload, base_config = load_base_config(args.base_best_config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_output_root = output_dir / "trial_runs"
    trial_output_root.mkdir(parents=True, exist_ok=True)
    storage = args.storage or f"sqlite:///{(output_dir / 'study.db').as_posix()}"
    subject = args.feature if args.student_type == "mlp" else args.model
    study_name = args.study_name or f"student_soft_{safe_name(args.student_type)}_{safe_name(subject)}_{safe_name(args.tasks)}"

    def objective(trial):
        loss_config = {
            "soft_cls_weight": trial.suggest_categorical("soft_cls_weight", weight_choices),
            "soft_reg_weight": 0.0,
            "aux_reduction": "observation",
            "aux_target_clip": 0.0,
        }
        split_id = f"hpo_soft_{safe_name(args.student_type)}_{safe_name(subject)}_{safe_name(args.tasks)}_trial_{trial.number:04d}"
        command = build_train_command(args, base_config, loss_config, split_id, trial_output_root)
        run_dir = trial_output_root / args.dataset / "runs" / run_id_for_args(args, split_id)
        save_json(output_dir / f"trial_{trial.number:04d}_config.json", {"base_config": base_config, "loss_config": loss_config, "command": command, "run_dir": str(run_dir)})
        subprocess.run(command, check=True)
        value = valid_objective_auc(run_dir / "valid_metrics.csv", args.objective_tasks)
        trial.set_user_attr("valid_auc", value)
        trial.set_user_attr("objective_tasks", args.objective_tasks)
        trial.set_user_attr("run_dir", str(run_dir))
        trial.set_user_attr("split_id", split_id)
        return value

    sampler = optuna.samplers.GridSampler({"soft_cls_weight": weight_choices}, seed=args.seed)
    study = optuna.create_study(study_name=study_name, storage=storage, direction="maximize", sampler=sampler, load_if_exists=True)
    study.optimize(objective, n_trials=n_trials, timeout=args.timeout)
    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state", "user_attrs"))
    trials_df.to_csv(output_dir / "trials.csv", index=False)
    best_trial = study.best_trial
    best_loss_config = {
        "soft_cls_weight": float(best_trial.params.get("soft_cls_weight", 0.0)),
        "soft_reg_weight": 0.0,
        "aux_reduction": "observation",
        "aux_target_clip": 0.0,
    }
    best_payload = {
        "study_name": study_name,
        "best_trial_number": best_trial.number,
        "best_value": best_trial.value,
        "objective": f"valid_{args.objective_tasks}_mean_roc_auc",
        "dataset": args.dataset,
        "student_type": args.student_type,
        "model": args.model if args.student_type == "graph" else "MLP",
        "feature": args.feature if args.student_type == "mlp" else "graph",
        "tasks": args.tasks,
        "loss_name": "mask_bce_soft",
        "teacher_profile_dir": args.teacher_profile_dir,
        "base_best_config": str(args.base_best_config),
        "base_config": base_config,
        "base_hpo_payload": base_payload,
        "loss_config": best_loss_config,
        "searched_params": dict(best_trial.params),
        "trial_dir": best_trial.user_attrs.get("run_dir"),
        "weight_choices": weight_choices,
        "search_space": {"soft_cls_weight": weight_choices},
        "sampler": "GridSampler",
    }
    save_json(output_dir / "best_config.json", best_payload)
    best_trial_dir = best_trial.user_attrs.get("run_dir")
    if best_trial_dir and Path(best_trial_dir).exists():
        best_copy = output_dir / "best_trial"
        if best_copy.exists():
            shutil.rmtree(best_copy)
        shutil.copytree(best_trial_dir, best_copy)
    print(f"Saved soft-weight HPO results under: {output_dir}")
    print(f"Best trial: {best_trial.number}, value={best_trial.value:.6f}, params={dict(best_trial.params)}")


if __name__ == "__main__":
    main()
