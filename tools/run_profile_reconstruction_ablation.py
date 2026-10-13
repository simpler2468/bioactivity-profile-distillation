"""Launch PCA-profile reconstruction variants with distinct result namespaces."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


COMPONENTS = {"pc1", "pc1_5", "pc1_10", "pc2_10"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run PCA profile-reconstruction ablations from an HPO config.")
    parser.add_argument("--runner", choices=["mlp", "graph"], required=True)
    parser.add_argument("--components", default="pc1,pc1_5,pc1_10,pc2_10")
    parser.add_argument("--profile_root", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("runner_args", nargs=argparse.REMAINDER, help="Arguments passed to the HPO student runner after --.")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    runner = project_root / "student" / f"run_{args.runner}_student_from_hpo.py"
    if not runner.exists():
        raise FileNotFoundError(runner)
    forwarded = list(args.runner_args)
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    for component_set in [item.strip() for item in args.components.split(",") if item.strip()]:
        if component_set not in COMPONENTS:
            raise ValueError(f"Unsupported component set: {component_set}")
        split_id = f"contain_valid_aux_soft_{component_set}"
        profile_dir = Path(args.profile_root) / f"feature=pca_reconstruction__model=Teacher__split=pca_soft__seed=mix__components={component_set}"
        command = [
            args.python,
            str(runner),
            "--split_id",
            split_id,
            "--teacher_profile_dir",
            str(profile_dir),
            "--soft_profile_mode",
            "full",
            *forwarded,
        ]
        print(" ".join(str(item) for item in command))
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
