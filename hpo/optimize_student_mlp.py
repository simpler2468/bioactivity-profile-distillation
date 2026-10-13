import argparse
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


def require_optuna():
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit("Optuna is not installed. Install it with: pip install optuna") from exc
    return optuna


def safe_name(value):
    return str(value).replace(" ", "-").replace("/", "-").replace("\\", "-")


def sample_config(trial):
    return {
        "num_hidden_layers": trial.suggest_categorical("num_hidden_layers", [2, 3, 4]),
        "shrink_ratio": trial.suggest_categorical("shrink_ratio", [0.5, 0.65, 0.8]),
        "min_hidden_dim": trial.suggest_categorical("min_hidden_dim", [64, 128, 186, 256]),
        "dropout": trial.suggest_categorical("dropout", [0.1, 0.2, 0.3, 0.4, 0.5]),
        "lr": trial.suggest_categorical("lr", [1e-4, 3e-4, 5e-4, 1e-3, 3e-3]),
        "weight_decay": trial.suggest_categorical("weight_decay", [1e-6, 1e-5, 1e-4, 1e-3]),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "use_pos_weight": trial.suggest_categorical("use_pos_weight", [True, False]),
        "hard_reduction": trial.suggest_categorical("hard_reduction", ["observation", "task"]),
        "activation": "leaky_relu",
        "norm_type": "layer_norm",
        "negative_slope": 0.01,
        "gradient_clip_norm": 1.0,
    }


def complete_config(params):
    config = {
        "activation": "leaky_relu",
        "norm_type": "layer_norm",
        "negative_slope": 0.01,
        "gradient_clip_norm": 1.0,
    }
    config.update(params)
    return config


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


def parse_args():
    parser = argparse.ArgumentParser(description="Optuna HPO for fixed-split student MLP.")
    parser.add_argument("--dataset", default="Tox21_Toxcast")
    parser.add_argument("--feature", required=True)
    parser.add_argument("--tasks", choices=["tox21_12", "all_93"], required=True)
    parser.add_argument("--feature_npz", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--objective_tasks", choices=["tox21_12", "all"], default=None)
    parser.add_argument("--study_name", default=None)
    parser.add_argument("--storage", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    optuna = require_optuna()
    objective_tasks = args.objective_tasks or ("tox21_12" if args.tasks == "tox21_12" else "all")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_output_root = output_dir / "trial_runs"
    trial_output_root.mkdir(parents=True, exist_ok=True)
    storage = args.storage or f"sqlite:///{(output_dir / 'study.db').as_posix()}"
    study_name = args.study_name or f"student_MLP_{safe_name(args.feature)}_{safe_name(args.tasks)}"
    train_script = PROJECT_ROOT / "student/run_mlp_student.py"

    def objective(trial):
        config = sample_config(trial)
        split_id = f"hpo_student_mlp_{safe_name(args.feature)}_{safe_name(args.tasks)}_trial_{trial.number:04d}"
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
            str(trial_output_root),
            "--split_id",
            split_id,
            "--seed",
            str(args.seed),
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

        run_dir = (
            trial_output_root
            / args.dataset
            / "runs"
            / student_run_id(args.feature, "MLP", args.tasks, "mask_bce", split_id, args.seed)
        )
        save_json(output_dir / f"trial_{trial.number:04d}_config.json", {"config": config, "command": command, "run_dir": str(run_dir)})
        subprocess.run(command, check=True)
        value = valid_objective_auc(run_dir / "valid_metrics.csv", objective_tasks)
        trial.set_user_attr("valid_auc", value)
        trial.set_user_attr("objective_tasks", objective_tasks)
        trial.set_user_attr("run_dir", str(run_dir))
        trial.set_user_attr("split_id", split_id)
        return value

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state", "user_attrs"))
    trials_df.to_csv(output_dir / "trials.csv", index=False)
    best_trial = study.best_trial
    best_payload = {
        "study_name": study_name,
        "best_trial_number": best_trial.number,
        "best_value": best_trial.value,
        "objective": f"valid_{objective_tasks}_mean_roc_auc",
        "dataset": args.dataset,
        "feature": args.feature,
        "tasks": args.tasks,
        "feature_npz": str(args.feature_npz),
        "seed": args.seed,
        "config": complete_config(dict(best_trial.params)),
        "searched_params": dict(best_trial.params),
        "trial_dir": best_trial.user_attrs.get("run_dir"),
    }
    save_json(output_dir / "best_config.json", best_payload)
    best_trial_dir = best_trial.user_attrs.get("run_dir")
    if best_trial_dir and Path(best_trial_dir).exists():
        best_copy = output_dir / "best_trial"
        if best_copy.exists():
            shutil.rmtree(best_copy)
        shutil.copytree(best_trial_dir, best_copy)
    elif best_trial_dir:
        print(f"Skip copying missing best trial directory: {best_trial_dir}")
    print(f"Saved HPO results under: {output_dir}")
    print(f"Best trial: {best_trial.number}, value={best_trial.value:.6f}")


if __name__ == "__main__":
    main()
