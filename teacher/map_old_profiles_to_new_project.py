from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


FEATURES = ["av", "ec4", "desc", "avd", "e4d"]
MODELS = ["mlp", "rf", "svm"]
MODEL_SEEDS = {"mlp": "ensemble5", "rf": "26", "svm": "26"}
MODEL_SPLITS = {
    "mlp": ["cve", "contain_valid_ecfp4_similarity"],
    "rf": ["contain_valid_ecfp4_similarity_no_valid"],
    "svm": ["contain_valid_ecfp4_similarity_no_valid"],
}
GROUP_KEYS = {
    "training": ("train_oof_raw_score", "train_oof_y", "train_oof_smiles"),
    "valid": ("valid_raw_score", "valid_y", "valid_smiles"),
    "test": ("test_raw_score", "test_y", "test_smiles"),
}


def canonical_smiles(smiles: object) -> Optional[str]:
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def largest_fragment_canonical_smiles(smiles: object) -> Optional[str]:
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
    except Exception:
        return None
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60, 60)
    return 1.0 / (1.0 + np.exp(-x))


def masked_bce_from_logits(scores: np.ndarray, labels: np.ndarray) -> float:
    mask = np.isfinite(scores) & np.isfinite(labels)
    if mask.sum() == 0:
        return float("nan")
    z = scores[mask]
    y = labels[mask]
    return float(np.mean(np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z)))))


def fit_platt_one_task(raw_score: np.ndarray, y: np.ndarray) -> dict:
    mask = np.isfinite(raw_score) & np.isfinite(y)
    x = raw_score[mask].astype(float)
    target = y[mask].astype(float)
    n_valid = int(mask.sum())
    n_pos = int(np.sum(target == 1))
    n_neg = int(np.sum(target == 0))
    if n_valid == 0 or n_pos == 0 or n_neg == 0:
        return {
            "a": 1.0,
            "b": 0.0,
            "success": False,
            "message": "insufficient labels for Platt scaling",
            "n_valid": n_valid,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "loss_before": masked_bce_from_logits(raw_score, y),
            "loss_after": masked_bce_from_logits(raw_score, y),
        }

    def objective(params: np.ndarray) -> float:
        a, b = params
        z = a * x + b
        return float(np.mean(np.maximum(z, 0.0) - z * target + np.log1p(np.exp(-np.abs(z)))))

    before = objective(np.array([1.0, 0.0], dtype=float))
    result = minimize(objective, np.array([1.0, 0.0], dtype=float), method="L-BFGS-B")
    if result.success:
        a, b = result.x.astype(float)
    else:
        a, b = 1.0, 0.0
    after = objective(np.array([a, b], dtype=float))
    return {
        "a": float(a),
        "b": float(b),
        "success": bool(result.success),
        "message": str(result.message),
        "n_valid": n_valid,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "loss_before": float(before),
        "loss_after": float(after),
    }


def per_task_metrics(scores: np.ndarray, labels: np.ndarray, task_names: Sequence[str]) -> pd.DataFrame:
    rows = []
    probs = sigmoid(scores)
    for idx, task in enumerate(task_names):
        y = labels[:, idx]
        p = probs[:, idx]
        mask = np.isfinite(y) & np.isfinite(p)
        n = int(mask.sum())
        n_pos = int(np.nansum(y[mask] == 1)) if n else 0
        n_neg = int(np.nansum(y[mask] == 0)) if n else 0
        roc = np.nan
        pr = np.nan
        ll = np.nan
        if n and n_pos > 0 and n_neg > 0:
            roc = float(roc_auc_score(y[mask], p[mask]))
            pr = float(average_precision_score(y[mask], p[mask]))
            ll = float(log_loss(y[mask], np.clip(p[mask], 1e-7, 1 - 1e-7), labels=[0, 1]))
        rows.append({
            "task_index": idx,
            "task_name": task,
            "n_valid": n,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "roc_auc": roc,
            "average_precision": pr,
            "log_loss": ll,
        })
    return pd.DataFrame(rows)


def metric_summary(metrics: pd.DataFrame, task_names: Sequence[str]) -> dict:
    tox21_12 = set(task_names[:12])
    is_tox21 = metrics["task_name"].isin(tox21_12)
    valid_roc = metrics["roc_auc"].notna()
    return {
        "all_93_mean_roc_auc": float(metrics.loc[valid_roc, "roc_auc"].mean()),
        "all_93_valid_tasks": int(valid_roc.sum()),
        "tox21_12_mean_roc_auc": float(metrics.loc[is_tox21 & valid_roc, "roc_auc"].mean()),
        "tox21_12_valid_tasks": int((is_tox21 & valid_roc).sum()),
    }


def save_json(path: Path, data: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_task_names(csv_path: Path) -> List[str]:
    df = pd.read_csv(csv_path, nrows=1)
    return [c for c in df.columns if c not in {"SMILES", "group"}]


def find_profile_dir(old_profiles_root: Path, feature: str, model: str) -> Path:
    seed = MODEL_SEEDS[model]
    for split in MODEL_SPLITS[model]:
        candidate = old_profiles_root / f"f={feature}__m={model}__sp={split}__s={seed}"
        if (candidate / "profile_arrays.npz").exists():
            return candidate
    raise FileNotFoundError(f"No profile found for feature={feature}, model={model}")


def build_old_lookup(arrays: np.lib.npyio.NpzFile, old_task_names: Sequence[str]) -> Tuple[dict, dict, dict]:
    direct_lookup = defaultdict(list)
    fragment_lookup = defaultdict(list)
    stats = {
        "old_duplicate_canonical_entries": 0,
        "old_duplicate_fragment_entries": 0,
        "old_unreadable_smiles": 0,
        "old_unreadable_fragment_smiles": 0,
        "old_entries": 0,
    }
    for group, (score_key, y_key, smiles_key) in GROUP_KEYS.items():
        scores = arrays[score_key]
        labels = arrays[y_key]
        smiles = arrays[smiles_key]
        for row_idx, smi in enumerate(smiles):
            stats["old_entries"] += 1
            entry = {
                "group": group,
                "row_idx": row_idx,
                "smiles": str(smi),
                "score": scores[row_idx].astype(float),
                "label": labels[row_idx].astype(float),
            }
            canon = canonical_smiles(smi)
            if canon is None:
                stats["old_unreadable_smiles"] += 1
            else:
                direct_lookup[canon].append(entry)
            frag = largest_fragment_canonical_smiles(smi)
            if frag is None:
                stats["old_unreadable_fragment_smiles"] += 1
            else:
                fragment_lookup[frag].append(entry)
    stats["old_unique_canonical"] = len(direct_lookup)
    stats["old_unique_fragment_canonical"] = len(fragment_lookup)
    stats["old_duplicate_canonical_entries"] = int(sum(max(0, len(v) - 1) for v in direct_lookup.values()))
    stats["old_duplicate_fragment_entries"] = int(sum(max(0, len(v) - 1) for v in fragment_lookup.values()))
    return direct_lookup, fragment_lookup, stats

def choose_old_entry(entries: Sequence[dict], new_group: str) -> Tuple[dict, str]:
    priorities = {
        "training": ["training", "valid", "test"],
        "valid": ["valid", "training", "test"],
        "test": ["test", "valid", "training"],
    }.get(new_group, [new_group, "test", "valid", "training"])
    for group in priorities:
        for entry in entries:
            if entry["group"] == group:
                return entry, group
    return entries[0], entries[0]["group"]


def map_profile_to_new(profile_dir: Path, new_csv_path: Path, task_names: Sequence[str]) -> Tuple[dict, dict, pd.DataFrame]:
    arrays = np.load(profile_dir / "profile_arrays.npz", allow_pickle=True)
    old_task_names = [str(x) for x in arrays["task_names"].tolist()]
    if list(old_task_names) != list(task_names):
        raise ValueError(f"Task names do not match for {profile_dir}")
    direct_lookup, fragment_lookup, old_stats = build_old_lookup(arrays, old_task_names)

    new_df = pd.read_csv(new_csv_path)
    labels_all = new_df[task_names].to_numpy(dtype=float)
    group_indices = {
        "training": new_df.index[new_df["group"].astype(str) == "training"].to_numpy(dtype=int),
        "valid": new_df.index[new_df["group"].astype(str) == "valid"].to_numpy(dtype=int),
        "test": new_df.index[new_df["group"].astype(str) == "test"].to_numpy(dtype=int),
    }

    result = {}
    for group, indices in group_indices.items():
        prefix = "train_oof" if group == "training" else group
        result[f"{prefix}_raw_score"] = np.full((len(indices), len(task_names)), np.nan, dtype=float)
        result[f"{prefix}_y"] = labels_all[indices].astype(float)
        result[f"{prefix}_smiles"] = new_df.loc[indices, "SMILES"].astype(str).to_numpy(dtype=object)
        result[f"{prefix}_row_index"] = indices.astype(int)
    result["task_names"] = np.asarray(task_names, dtype=object)

    position_by_row_index = {}
    for group, indices in group_indices.items():
        for pos, row_idx in enumerate(indices):
            position_by_row_index[int(row_idx)] = (group, pos)

    missing_rows = []
    source_counter = Counter()
    duplicate_match_rows = 0
    unreadable_new_rows = 0
    mapped_counter = Counter()

    for row_idx, row in new_df.iterrows():
        group = str(row["group"])
        if group not in group_indices:
            continue
        prefix = "train_oof" if group == "training" else group
        pos = position_by_row_index[int(row_idx)][1]
        smi = row["SMILES"]
        canon = canonical_smiles(smi)
        if canon is None:
            unreadable_new_rows += 1
            missing_rows.append({"row_index": int(row_idx), "group": group, "SMILES": smi, "reason": "new_smiles_unreadable"})
            continue
        entries = direct_lookup.get(canon)
        match_mode = "direct_canonical"
        frag = None
        if not entries:
            frag = largest_fragment_canonical_smiles(smi)
            entries = fragment_lookup.get(frag) if frag is not None else None
            match_mode = "largest_fragment_canonical"
        if not entries:
            missing_rows.append({
                "row_index": int(row_idx),
                "group": group,
                "SMILES": smi,
                "canonical_smiles": canon,
                "largest_fragment_canonical_smiles": frag,
                "reason": "not_found_in_old_profiles",
            })
            continue
        if len(entries) > 1:
            duplicate_match_rows += 1
        entry, source_group = choose_old_entry(entries, group)
        result[f"{prefix}_raw_score"][pos] = entry["score"]
        source_counter[(group, source_group)] += 1
        source_counter[(group, match_mode)] += 1
        mapped_counter[group] += 1

    stats = dict(old_stats)
    stats.update({
        "profile_dir": str(profile_dir),
        "new_rows": int(len(new_df)),
        "new_training_rows": int(len(group_indices["training"])),
        "new_valid_rows": int(len(group_indices["valid"])),
        "new_test_rows": int(len(group_indices["test"])),
        "mapped_training_rows": int(mapped_counter["training"]),
        "mapped_valid_rows": int(mapped_counter["valid"]),
        "mapped_test_rows": int(mapped_counter["test"]),
        "missing_rows": int(len(missing_rows)),
        "unreadable_new_rows": int(unreadable_new_rows),
        "new_rows_with_multiple_old_matches": int(duplicate_match_rows),
        "source_group_counts": {f"new_{k[0]}__old_{k[1]}": int(v) for k, v in sorted(source_counter.items())},
    })
    return result, stats, pd.DataFrame(missing_rows)

def write_profile_arrays(output_dir: Path, arrays: Mapping[str, np.ndarray], metadata: Mapping) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "profile_arrays.npz", **arrays)
    np.save(output_dir / "train_oof_raw_score.npy", arrays["train_oof_raw_score"])
    np.save(output_dir / "train_oof_y.npy", arrays["train_oof_y"])
    np.save(output_dir / "valid_raw_score.npy", arrays["valid_raw_score"])
    np.save(output_dir / "valid_y.npy", arrays["valid_y"])
    np.save(output_dir / "test_raw_score.npy", arrays["test_raw_score"])
    np.save(output_dir / "test_y.npy", arrays["test_y"])
    save_json(output_dir / "profile_metadata.json", metadata)


def candidate_output_name(feature: str, model: str) -> str:
    return f"f={feature}__m={model}__sp=mapped_cve__s={MODEL_SEEDS[model]}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Map old Tox21_ToxCast teacher profiles to the new project by canonical SMILES.")
    parser.add_argument("--old_project", default=r"E:\z_data\tox21_final2")
    parser.add_argument("--new_project", default=r"E:\z_data\tox21_clean")
    parser.add_argument("--dataset", default="Tox21_Toxcast")
    parser.add_argument("--new_csv", default=r"data\raw_data\merge_data\Tox21_Toxcast\contain_valid\data_by_ecfp4_similarity.csv")
    parser.add_argument("--save_candidates", action="store_true", default=True)
    args = parser.parse_args()

    old_project = Path(args.old_project)
    new_project = Path(args.new_project)
    new_csv_path = new_project / args.new_csv
    old_profiles_root = old_project / "teacher" / args.dataset / "profiles"
    new_profiles_root = new_project / "teacher" / args.dataset / "profiles"
    new_metrics_root = new_project / "teacher" / args.dataset / "metrics"
    audit_dir = new_project / "teacher" / args.dataset / "mapping_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    task_names = read_task_names(new_csv_path)
    candidates = []
    missing_tables = []
    audit_rows = []

    for feature in FEATURES:
        for model in MODELS:
            profile_dir = find_profile_dir(old_profiles_root, feature, model)
            arrays, stats, missing = map_profile_to_new(profile_dir, new_csv_path, task_names)
            candidate_name = candidate_output_name(feature, model)
            out_dir = new_profiles_root / candidate_name
            metadata = {
                "dataset": args.dataset,
                "feature": feature,
                "model": model,
                "split": "mapped_from_old_canonical_smiles",
                "seed": MODEL_SEEDS[model],
                "source_project": str(old_project),
                "source_profile_dir": str(profile_dir),
                "new_csv": str(new_csv_path),
                "mapping_stats": stats,
            }
            if args.save_candidates:
                write_profile_arrays(out_dir, arrays, metadata)
            train_metrics = per_task_metrics(arrays["train_oof_raw_score"], arrays["train_oof_y"], task_names)
            valid_metrics = per_task_metrics(arrays["valid_raw_score"], arrays["valid_y"], task_names)
            test_metrics = per_task_metrics(arrays["test_raw_score"], arrays["test_y"], task_names)
            metric_dir = new_metrics_root / candidate_name
            metric_dir.mkdir(parents=True, exist_ok=True)
            train_metrics.to_csv(metric_dir / "train_oof_per_task_metrics.csv", index=False)
            valid_metrics.to_csv(metric_dir / "valid_per_task_metrics.csv", index=False)
            test_metrics.to_csv(metric_dir / "test_per_task_metrics.csv", index=False)
            save_json(metric_dir / "profile_metric_summary.json", {
                "train_oof": metric_summary(train_metrics, task_names),
                "valid": metric_summary(valid_metrics, task_names),
                "test": metric_summary(test_metrics, task_names),
                "mapping_stats": stats,
            })
            for row in train_metrics.itertuples(index=False):
                candidates.append({
                    "candidate_name": candidate_name,
                    "feature": feature,
                    "model": model,
                    "task_index": int(row.task_index),
                    "task_name": row.task_name,
                    "train_oof_roc_auc": row.roc_auc,
                })
            audit_rows.append({"candidate_name": candidate_name, **stats})
            if len(missing):
                missing = missing.copy()
                missing.insert(0, "candidate_name", candidate_name)
                missing_tables.append(missing)

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(audit_dir / "canonical_mapping_summary.csv", index=False)
    if missing_tables:
        pd.concat(missing_tables, ignore_index=True).to_csv(audit_dir / "canonical_mapping_missing_rows.csv", index=False)
    else:
        pd.DataFrame(columns=["candidate_name", "row_index", "group", "SMILES", "canonical_smiles", "largest_fragment_canonical_smiles", "reason"]).to_csv(audit_dir / "canonical_mapping_missing_rows.csv", index=False)

    candidate_metrics = pd.DataFrame(candidates)
    candidate_metrics.to_csv(new_metrics_root / "mapped_candidate_train_oof_auc_by_task.csv", index=False)
    selected_rows = []
    raw_by_candidate = {}
    for feature in FEATURES:
        for model in MODELS:
            name = candidate_output_name(feature, model)
            z = np.load(new_profiles_root / name / "profile_arrays.npz", allow_pickle=True)
            raw_by_candidate[name] = {k: z[k] for k in z.files}

    selected_train = np.full_like(next(iter(raw_by_candidate.values()))["train_oof_raw_score"], np.nan, dtype=float)
    selected_valid = np.full_like(next(iter(raw_by_candidate.values()))["valid_raw_score"], np.nan, dtype=float)
    selected_test = np.full_like(next(iter(raw_by_candidate.values()))["test_raw_score"], np.nan, dtype=float)
    train_y = next(iter(raw_by_candidate.values()))["train_oof_y"]
    valid_y = next(iter(raw_by_candidate.values()))["valid_y"]
    test_y = next(iter(raw_by_candidate.values()))["test_y"]

    for task_idx, task in enumerate(task_names):
        sub = candidate_metrics[candidate_metrics["task_index"] == task_idx].copy()
        sub = sub[np.isfinite(sub["train_oof_roc_auc"])]
        if sub.empty:
            chosen = candidate_metrics[candidate_metrics["task_index"] == task_idx].iloc[0]
        else:
            chosen = sub.sort_values("train_oof_roc_auc", ascending=False).iloc[0]
        cname = str(chosen["candidate_name"])
        selected_train[:, task_idx] = raw_by_candidate[cname]["train_oof_raw_score"][:, task_idx]
        selected_valid[:, task_idx] = raw_by_candidate[cname]["valid_raw_score"][:, task_idx]
        selected_test[:, task_idx] = raw_by_candidate[cname]["test_raw_score"][:, task_idx]
        selected_rows.append({
            "task_index": task_idx,
            "task_name": task,
            "selected_candidate": cname,
            "selected_feature": chosen["feature"],
            "selected_model": chosen["model"],
            "train_oof_roc_auc": chosen["train_oof_roc_auc"],
        })

    first = next(iter(raw_by_candidate.values()))
    selected_arrays = {
        "train_oof_raw_score": selected_train,
        "train_oof_y": train_y,
        "train_oof_smiles": first["train_oof_smiles"],
        "train_oof_row_index": first["train_oof_row_index"],
        "valid_raw_score": selected_valid,
        "valid_y": valid_y,
        "valid_smiles": first["valid_smiles"],
        "valid_row_index": first["valid_row_index"],
        "test_raw_score": selected_test,
        "test_y": test_y,
        "test_smiles": first["test_smiles"],
        "test_row_index": first["test_row_index"],
        "task_names": np.asarray(task_names, dtype=object),
    }
    selected_name = "f=taskwise_selected_mapped__m=tea__sp=oofauc__s=mixed"
    selected_dir = new_profiles_root / selected_name
    write_profile_arrays(selected_dir, selected_arrays, {
        "dataset": args.dataset,
        "feature": "taskwise_selected_mapped",
        "model": "Teacher",
        "split": "oofauc_mapped_from_old_canonical_smiles",
        "seed": "mixed",
        "source_project": str(old_project),
        "selection_metric": "train_oof_roc_auc",
        "new_csv": str(new_csv_path),
    })
    selection_df = pd.DataFrame(selected_rows)
    selection_df.to_csv(selected_dir / "task_teacher_selection.csv", index=False)

    for split_name, score_key, y_key in [
        ("train_oof", "train_oof_raw_score", "train_oof_y"),
        ("valid", "valid_raw_score", "valid_y"),
        ("test", "test_raw_score", "test_y"),
    ]:
        metrics = per_task_metrics(selected_arrays[score_key], selected_arrays[y_key], task_names)
        metric_dir = new_metrics_root / selected_name
        metric_dir.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(metric_dir / f"{split_name}_per_task_metrics.csv", index=False)
    save_json(new_metrics_root / selected_name / "profile_metric_summary.json", {
        "train_oof": metric_summary(per_task_metrics(selected_train, train_y, task_names), task_names),
        "valid": metric_summary(per_task_metrics(selected_valid, valid_y, task_names), task_names),
        "test": metric_summary(per_task_metrics(selected_test, test_y, task_names), task_names),
    })

    platt_rows = []
    platt_train = np.full_like(selected_train, np.nan, dtype=float)
    platt_valid = np.full_like(selected_valid, np.nan, dtype=float)
    platt_test = np.full_like(selected_test, np.nan, dtype=float)
    for task_idx, task in enumerate(task_names):
        fit = fit_platt_one_task(selected_train[:, task_idx], train_y[:, task_idx])
        a = fit["a"]
        b = fit["b"]
        platt_train[:, task_idx] = a * selected_train[:, task_idx] + b
        platt_valid[:, task_idx] = a * selected_valid[:, task_idx] + b
        platt_test[:, task_idx] = a * selected_test[:, task_idx] + b
        platt_rows.append({"task_index": task_idx, "task_name": task, **fit})

    platt_name = "f=taskwise_selected_platt_mapped__m=tea__sp=oofauc__s=mixed"
    platt_dir = new_profiles_root / platt_name
    platt_arrays = dict(selected_arrays)
    platt_arrays.update({
        "train_oof_raw_score": platt_train,
        "valid_raw_score": platt_valid,
        "test_raw_score": platt_test,
        "train_oof_prob": sigmoid(platt_train),
        "valid_prob": sigmoid(platt_valid),
        "test_prob": sigmoid(platt_test),
    })
    write_profile_arrays(platt_dir, platt_arrays, {
        "dataset": args.dataset,
        "feature": "taskwise_selected_platt_mapped",
        "model": "Teacher",
        "split": "oofauc_mapped_from_old_canonical_smiles",
        "seed": "mixed",
        "source_profile_dir": str(selected_dir),
        "platt_fit_data": "mapped train OOF raw score and new train labels",
        "new_csv": str(new_csv_path),
    })
    pd.DataFrame(platt_rows).to_csv(platt_dir / "platt_parameters.csv", index=False)

    platt_metric_dir = new_metrics_root / platt_name
    platt_metric_dir.mkdir(parents=True, exist_ok=True)
    platt_summaries = {}
    for split_name, scores, labels in [
        ("train_oof", platt_train, train_y),
        ("valid", platt_valid, valid_y),
        ("test", platt_test, test_y),
    ]:
        metrics = per_task_metrics(scores, labels, task_names)
        metrics.to_csv(platt_metric_dir / f"{split_name}_per_task_metrics.csv", index=False)
        platt_summaries[split_name] = metric_summary(metrics, task_names)
    pd.DataFrame(platt_rows).to_csv(platt_metric_dir / "platt_parameters.csv", index=False)
    save_json(platt_metric_dir / "profile_metric_summary.json", {
        **platt_summaries,
        "source_selected_profile": str(selected_dir),
        "mapping_audit_dir": str(audit_dir),
    })

    print("Saved mapped candidate profiles, task-wise selected profile, and Platt-scaled profile.")
    print(f"Mapping audit: {audit_dir}")
    print(f"Selected profile: {selected_dir}")
    print(f"Platt profile: {platt_dir}")
    print("Platt test summary:")
    print(json.dumps(platt_summaries["test"], ensure_ascii=False, indent=2))
    total_missing = int(pd.read_csv(audit_dir / "canonical_mapping_missing_rows.csv").shape[0])
    print(f"Total missing mapped rows across candidates: {total_missing}")


if __name__ == "__main__":
    main()
