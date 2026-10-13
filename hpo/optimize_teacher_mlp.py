import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teacher.common import run_id, save_json


TOX21_12_TASKS = [
    "NR-AhR",
    "NR-AR",
    "NR-AR-LBD",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
]


def require_optuna():
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit(
            "Optuna is not installed in the current Python environment. "
            "Install it before running HPO, for example: pip install optuna"
        ) from exc
    return optuna


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


def safe_name(value):
    return str(value).replace(" ", "-").replace("/", "-").replace("\\", "-")


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


def tox21_12_mean_auc(metrics_csv):
    df = pd.read_csv(metrics_csv)
    sub = df[df["task_name"].isin(TOX21_12_TASKS)].copy()
    auc = pd.to_numeric(sub["roc_auc"], errors="coerce")
    return float(auc.mean())


def copy_best_trial(best_run_dir, best_trial_dir):
    if best_trial_dir.exists():
        shutil.rmtree(best_trial_dir)
    shutil.copytree(best_run_dir, best_trial_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Optuna HPO for Tox21_Toxcast teacher MLP.")
    parser.add_argument("--dataset", default="Tox21_Toxcast")
    parser.add_argument("--feature", default="descriptor")
    parser.add_argument("--feature_npz", default=str(PROJECT_ROOT / "data/fp_descriptor_data/merge_data/Tox21_Toxcast/contain_valid/data_by_ecfp4_similarity_descriptor.npz"))
    parser.add_argument("--fold_csv", default=str(PROJECT_ROOT / "data/raw_data/merge_data/Tox21_Toxcast/contain_valid/training_cluster_CV/training_cluster_CV_0.csv"))
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "hpo/Tox21_Toxcast/teacher_MLP_descriptor"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=None, help="Optuna timeout in seconds.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--study_name", default="teacher_MLP_descriptor")
    parser.add_argument("--storage", default=None, help="Optional Optuna storage URI. Defaults to sqlite under output_dir.")
    return parser.parse_args()


def main():
    args = parse_args()
    optuna = require_optuna()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_output_root = output_dir / "trial_runs"
    trial_output_root.mkdir(parents=True, exist_ok=True)
    storage = args.storage or f"sqlite:///{(output_dir / 'study.db').as_posix()}"
    train_script = PROJECT_ROOT / "teacher/run_tabular_teacher.py"

    def objective(trial):
        config = sample_config(trial)
        split_id = f"hpo_teacher_mlp_{safe_name(args.feature)}_trial_{trial.number:04d}"
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
            split_id,
            "--fold",
            str(args.fold),
            "--seed",
            str(args.seed),
            "--feature_npz",
            str(args.feature_npz),
            "--fold_csv",
            str(args.fold_csv),
            "--output_root",
            str(trial_output_root),
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

        trial_dir = (
            trial_output_root
            / args.dataset
            / "runs"
            / run_id(args.feature, "MLP", split_id, args.fold, args.seed)
        )
        save_json(
            output_dir / f"trial_{trial.number:04d}_config.json",
            {"config": config, "command": command, "trial_dir": str(trial_dir)},
        )
        subprocess.run(command, check=True)

        objective_value = tox21_12_mean_auc(trial_dir / "oof_metrics.csv")
        trial.set_user_attr("tox21_12_oof_auc", objective_value)
        trial.set_user_attr("trial_dir", str(trial_dir))
        trial.set_user_attr("split_id", split_id)
        return objective_value

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=args.study_name,
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
        "study_name": args.study_name,
        "best_trial_number": best_trial.number,
        "best_value": best_trial.value,
        "objective": "tox21_12_oof_mean_roc_auc",
        "feature": args.feature,
        "dataset": args.dataset,
        "fold_csv": str(args.fold_csv),
        "feature_npz": str(args.feature_npz),
        "seed": args.seed,
        "fold": args.fold,
        "config": complete_config(dict(best_trial.params)),
        "searched_params": dict(best_trial.params),
        "trial_dir": best_trial.user_attrs.get("trial_dir"),
    }
    save_json(output_dir / "best_config.json", best_payload)

    best_trial_dir = best_trial.user_attrs.get("trial_dir")
    if best_trial_dir:
        copy_best_trial(Path(best_trial_dir), output_dir / "best_trial")
    print(f"Saved HPO results under: {output_dir}")
    print(f"Best trial: {best_trial.number}, value={best_trial.value:.6f}")


if __name__ == "__main__":
    main()
