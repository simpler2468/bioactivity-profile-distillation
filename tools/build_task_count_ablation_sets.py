import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from student.run_mlp_student import TOX21_12_TASKS
from teacher.common import save_json


DEFAULT_GRAPH_DIR = (
    PROJECT_ROOT
    / "data"
    / "graph_data"
    / "merge_data"
    / "Tox21_Toxcast"
    / "contain_valid"
    / "data_by_ecfp4_similarity"
)


TASK_SET_SPECS = [
    ("tox21_12", 0),
    ("tox21_12_plus_20", 20),
    ("tox21_12_plus_40", 40),
    ("tox21_12_plus_60", 60),
    ("all_93", 81),
]


def load_task_names(graph_dir):
    path = Path(graph_dir) / "task_names.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description="Build nested task sets for Tox21/ToxCast task-count ablation.")
    parser.add_argument("--graph_dir", default=str(DEFAULT_GRAPH_DIR))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "data" / "task_sets" / "Tox21_Toxcast" / "task_count_ablation_seed2026"),
    )
    args = parser.parse_args()

    task_names = load_task_names(args.graph_dir)
    missing_tox21 = [task for task in TOX21_12_TASKS if task not in task_names]
    if missing_tox21:
        raise ValueError(f"Missing Tox21-12 tasks: {missing_tox21}")

    toxcast_tasks = [task for task in task_names if task not in TOX21_12_TASKS]
    if len(toxcast_tasks) != 81:
        raise ValueError(f"Expected 81 non-Tox21 tasks, found {len(toxcast_tasks)}.")

    rng = random.Random(args.seed)
    shuffled_toxcast = list(toxcast_tasks)
    rng.shuffle(shuffled_toxcast)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for task_set_id, extra_count in TASK_SET_SPECS:
        selected = list(TOX21_12_TASKS) + shuffled_toxcast[:extra_count]
        path = output_dir / f"{task_set_id}.txt"
        path.write_text("\n".join(selected) + "\n")
        rows.append(
            {
                "task_set": task_set_id,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "num_tox21_tasks": len(TOX21_12_TASKS),
                "num_toxcast_tasks": extra_count,
                "num_tasks": len(selected),
            }
        )

    pd.DataFrame(rows).to_csv(output_dir / "task_sets.csv", index=False)
    save_json(
        output_dir / "task_order_metadata.json",
        {
            "seed": args.seed,
            "graph_dir": str(Path(args.graph_dir)),
            "tox21_12_tasks": TOX21_12_TASKS,
            "shuffled_toxcast_tasks": shuffled_toxcast,
            "task_sets": rows,
            "note": "All task sets are nested: Tox21-12 first, then the first N shuffled ToxCast tasks.",
        },
    )
    print(f"Saved task-count ablation task sets under: {output_dir}")


if __name__ == "__main__":
    main()
