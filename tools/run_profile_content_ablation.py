"""Launch the three profile-content soft-supervision variants without overlap."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run full/target-only/non-target profile-content ablations.")
    parser.add_argument("--runner", choices=["mlp", "graph"], required=True)
    parser.add_argument("--modes", default="full,target_only,non_target")
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
    for mode in [item.strip() for item in args.modes.split(",") if item.strip()]:
        if mode not in {"full", "target_only", "non_target"}:
            raise ValueError(f"Unsupported profile-content mode: {mode}")
        split_id = {
            "full": "contain_valid_aux_soft",
            "target_only": "contain_valid_aux_soft_target_only",
            "non_target": "contain_valid_aux_soft_non_target",
        }[mode]
        command = [
            args.python,
            str(runner),
            "--split_id",
            split_id,
            "--soft_profile_mode",
            mode,
            *forwarded,
        ]
        print(" ".join(str(item) for item in command))
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
