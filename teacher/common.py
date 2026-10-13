import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_naming import (
    sanitize_id,
    teacher_profile_id as compact_teacher_profile_id,
    teacher_run_id as compact_teacher_run_id,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

NON_LABEL_COLUMNS = {"SMILES", "group"}


def run_id(feature, model, split, fold, seed):
    return compact_teacher_run_id(feature, model, split, fold, seed)


def profile_id(feature, model, split, seed):
    return compact_teacher_profile_id(feature, model, split, seed)


def ensure_project_path():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def load_feature_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    return {
        "X": data["X"].astype(np.float32),
        "y": data["y"].astype(np.float32),
        "groups": data["groups"].astype(str),
        "smiles": data["smiles"].astype(str),
        "task_names": data["task_names"].astype(str).tolist(),
        "feature_names": data["feature_names"].astype(str).tolist(),
    }


def infer_task_names(df):
    return [col for col in df.columns if col not in NON_LABEL_COLUMNS]


def build_smiles_index(smiles):
    mapping = {}
    duplicates = []
    for idx, smi in enumerate(smiles):
        if smi in mapping:
            duplicates.append(smi)
        mapping[str(smi)] = int(idx)
    if duplicates:
        raise ValueError(f"SMILES must be unique; found duplicates, first: {duplicates[:5]}")
    return mapping


def indices_from_split(feature_smiles, split_df, group):
    mapping = build_smiles_index(feature_smiles)
    selected = split_df[split_df["group"].astype(str) == str(group)]
    missing = [smi for smi in selected["SMILES"].astype(str) if smi not in mapping]
    if missing:
        raise ValueError(f"{len(missing)} split SMILES are missing from feature npz, first: {missing[:5]}")
    return np.asarray([mapping[smi] for smi in selected["SMILES"].astype(str)], dtype=np.int64)


def clip_prob_to_logit(prob, eps=1e-6):
    prob = np.asarray(prob, dtype=np.float64)
    prob = np.clip(prob, eps, 1.0 - eps)
    return np.log(prob / (1.0 - prob)).astype(np.float32)


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")


def save_prediction_npz(path, split_name, row_index, smiles, y_true, raw_score, task_names, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "split_name": np.asarray(split_name, dtype=object),
        "row_index": np.asarray(row_index, dtype=np.int64),
        "smiles": np.asarray(smiles, dtype=object),
        "y_true": np.asarray(y_true, dtype=np.float32),
        "raw_score": np.asarray(raw_score, dtype=np.float32),
        "task_names": np.asarray(task_names, dtype=object),
    }
    if extra:
        payload.update(extra)
    np.savez_compressed(path, **payload)


def load_prediction_npz(path):
    data = np.load(path, allow_pickle=True)
    return {
        "split_name": str(np.asarray(data["split_name"]).item()),
        "row_index": data["row_index"].astype(np.int64),
        "smiles": data["smiles"].astype(str),
        "y_true": data["y_true"].astype(np.float32),
        "raw_score": data["raw_score"].astype(np.float32),
        "task_names": data["task_names"].astype(str).tolist(),
    }


def write_metric_table(path, metrics_df, metadata):
    df = metrics_df.copy()
    for key, value in metadata.items():
        df[key] = value
    leading = list(metadata.keys()) + [col for col in df.columns if col not in metadata]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df[leading].to_csv(path, index=False)
