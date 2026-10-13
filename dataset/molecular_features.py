import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.DataStructs import ConvertToNumpyArray

try:
    from rdkit.Avalon import pyAvalonTools
    AVALON_AVAILABLE = True
except ImportError:
    pyAvalonTools = None
    AVALON_AVAILABLE = False


NON_LABEL_COLUMNS = {"SMILES", "group"}


def _mirror_feature_output_dir(input_csv, output_root="data/fp_descriptor_data"):
    input_csv = Path(input_csv)
    parts = list(input_csv.parent.parts)
    if "raw_data" in parts:
        parts[parts.index("raw_data")] = Path(output_root).name
        if len(parts) > 0 and parts[0].lower() == "data":
            return Path(*parts)
        return Path(*parts)
    return Path(output_root) / input_csv.parent.name


def _infer_task_names(data):
    return [col for col in data.columns if col not in NON_LABEL_COLUMNS]


def _bitvect_to_array(bitvect, n_bits):
    arr = np.zeros((n_bits,), dtype=np.float32)
    ConvertToNumpyArray(bitvect, arr)
    return arr


def make_ecfp4(mol, n_bits=1024):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    return _bitvect_to_array(fp, n_bits)


def make_avalon(mol, n_bits=1024):
    if not AVALON_AVAILABLE:
        raise RuntimeError("RDKit Avalon fingerprint is not available in this environment.")
    fp = pyAvalonTools.GetAvalonFP(mol, nBits=n_bits)
    return _bitvect_to_array(fp, n_bits)


def descriptor_definitions():
    return list(Descriptors._descList)


def make_descriptors(mol, desc_defs=None):
    desc_defs = desc_defs or descriptor_definitions()
    values = []
    failed = []
    for name, func in desc_defs:
        try:
            value = func(mol)
        except Exception as exc:
            value = np.nan
            failed.append(f"{name}: {type(exc).__name__}")
        values.append(value)
    return np.asarray(values, dtype=np.float64), failed


def normalize_descriptors_by_training_scale(descriptor_matrix, groups, quantile=0.95, clip_value=10.0):
    descriptor_matrix = np.asarray(descriptor_matrix, dtype=np.float64)
    groups = np.asarray(groups).astype(str)
    finite = np.isfinite(descriptor_matrix)
    invalid_count = int((~finite).sum())

    train_mask = groups == "training"
    if train_mask.sum() == 0:
        print("Warning: no training rows found; descriptor scales are estimated from all rows.")
        train_mask = np.ones(len(groups), dtype=bool)

    train_values = descriptor_matrix[train_mask]
    scale = np.ones(descriptor_matrix.shape[1], dtype=np.float64)
    for col_idx in range(descriptor_matrix.shape[1]):
        col = train_values[:, col_idx]
        col = col[np.isfinite(col)]
        if len(col) == 0:
            scale[col_idx] = 1.0
            continue
        col_scale = np.nanquantile(np.abs(col), quantile)
        if not np.isfinite(col_scale) or col_scale <= 0:
            col_scale = 1.0
        scale[col_idx] = float(col_scale)

    normalized = descriptor_matrix.copy()
    normalized[~np.isfinite(normalized)] = 0.0
    normalized = normalized / scale
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    if clip_value is not None and clip_value > 0:
        normalized = np.clip(normalized, -float(clip_value), float(clip_value))
    return normalized.astype(np.float32), scale, invalid_count


def _save_npz(
    output_path,
    X,
    y,
    groups,
    smiles,
    task_names,
    feature_names,
    failed_records,
    descriptor_scale=None,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    failed_indices = np.asarray([item["index"] for item in failed_records], dtype=np.int64)
    failed_smiles = np.asarray([item["smiles"] for item in failed_records], dtype=object)
    failed_reasons = np.asarray([item["reason"] for item in failed_records], dtype=object)

    payload = {
        "X": np.asarray(X, dtype=np.float32),
        "y": np.asarray(y, dtype=np.float32),
        "groups": np.asarray(groups, dtype=object),
        "smiles": np.asarray(smiles, dtype=object),
        "task_names": np.asarray(task_names, dtype=object),
        "feature_names": np.asarray(feature_names, dtype=object),
        "failed_indices": failed_indices,
        "failed_smiles": failed_smiles,
        "failed_reasons": failed_reasons,
    }
    if descriptor_scale is not None:
        payload["descriptor_scale"] = np.asarray(descriptor_scale, dtype=np.float32)

    np.savez_compressed(output_path, **payload)
    print(f"Saved: {output_path}")
    print(f"  X shape: {payload['X'].shape}, y shape: {payload['y'].shape}")
    print(f"  failed records saved: {len(failed_records)}")


def _select_rows(records, feature_keys):
    selected_indices = []
    selected_smiles = []
    selected_groups = []
    selected_y = []
    selected_features = []
    failed_records = []

    for record in records:
        missing = [key for key in feature_keys if key not in record["features"]]
        if missing:
            failed_records.append(
                {
                    "index": record["index"],
                    "smiles": record["smiles"],
                    "reason": "; ".join(record["errors"].get(key, f"{key}_missing") for key in missing),
                }
            )
            continue
        selected_indices.append(record["index"])
        selected_smiles.append(record["smiles"])
        selected_groups.append(record["group"])
        selected_y.append(record["labels"])
        selected_features.append(np.concatenate([record["features"][key] for key in feature_keys]))

    if selected_features:
        X = np.vstack(selected_features).astype(np.float32)
        y = np.vstack(selected_y).astype(np.float32)
    else:
        feature_dim = 0
        for key in feature_keys:
            if key == "Avalon":
                feature_dim += 0
            elif key == "ECFP4":
                feature_dim += 0
            elif key == "descriptor":
                feature_dim += 0
        X = np.zeros((0, feature_dim), dtype=np.float32)
        y = np.zeros((0, 0), dtype=np.float32)

    return {
        "indices": selected_indices,
        "smiles": selected_smiles,
        "groups": selected_groups,
        "y": y,
        "X": X,
        "failed_records": failed_records,
    }


def build_and_save_molecular_features(
    input_csv,
    output_dir=None,
    fp_n_bits=1024,
    descriptor_scale_quantile=0.95,
    descriptor_clip_value=10.0,
):
    input_csv = Path(input_csv)
    data = pd.read_csv(input_csv)
    if "SMILES" not in data.columns:
        raise ValueError(f"{input_csv} must contain a SMILES column.")
    if "group" not in data.columns:
        raise ValueError(f"{input_csv} must contain a group column.")

    output_dir = Path(output_dir) if output_dir is not None else _mirror_feature_output_dir(input_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = input_csv.stem

    task_names = _infer_task_names(data)
    labels = data[task_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    groups = data["group"].astype(str).to_numpy()
    smiles_list = data["SMILES"].astype(str).to_numpy()

    print(f"Input CSV: {input_csv}")
    print(f"Output dir: {output_dir}")
    print(f"Rows: {len(data)}")
    print(f"Tasks: {len(task_names)}")
    print(f"Group counts: {data['group'].value_counts(dropna=False).to_dict()}")
    print(f"Fingerprint bits: {fp_n_bits}")
    print(f"Avalon available: {AVALON_AVAILABLE}")

    desc_defs = descriptor_definitions()
    descriptor_names = [name for name, _ in desc_defs]
    records = []
    mol_failed = []
    descriptor_function_failures = {}

    for idx, smiles in enumerate(smiles_list):
        record = {
            "index": int(idx),
            "smiles": smiles,
            "group": groups[idx],
            "labels": labels[idx],
            "features": {},
            "errors": {},
        }
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            reason = "rdkit_mol_failed"
            record["errors"]["Avalon"] = reason
            record["errors"]["ECFP4"] = reason
            record["errors"]["descriptor"] = reason
            mol_failed.append({"index": int(idx), "smiles": smiles, "reason": reason})
            records.append(record)
            continue

        try:
            record["features"]["ECFP4"] = make_ecfp4(mol, n_bits=fp_n_bits)
        except Exception as exc:
            record["errors"]["ECFP4"] = f"ecfp4_failed: {type(exc).__name__}: {exc}"

        if AVALON_AVAILABLE:
            try:
                record["features"]["Avalon"] = make_avalon(mol, n_bits=fp_n_bits)
            except Exception as exc:
                record["errors"]["Avalon"] = f"avalon_failed: {type(exc).__name__}: {exc}"
        else:
            record["errors"]["Avalon"] = "avalon_unavailable"

        try:
            desc_values, desc_failed = make_descriptors(mol, desc_defs=desc_defs)
            record["features"]["descriptor_raw"] = desc_values
            for failure in desc_failed:
                descriptor_function_failures[failure] = descriptor_function_failures.get(failure, 0) + 1
        except Exception as exc:
            record["errors"]["descriptor"] = f"descriptor_failed: {type(exc).__name__}: {exc}"

        records.append(record)

    descriptor_records = [record for record in records if "descriptor_raw" in record["features"]]
    if descriptor_records:
        descriptor_raw = np.vstack([record["features"]["descriptor_raw"] for record in descriptor_records])
        descriptor_groups = [record["group"] for record in descriptor_records]
        descriptor_norm, descriptor_scale, invalid_descriptor_count = normalize_descriptors_by_training_scale(
            descriptor_raw,
            descriptor_groups,
            quantile=descriptor_scale_quantile,
            clip_value=descriptor_clip_value,
        )
        for record, desc in zip(descriptor_records, descriptor_norm):
            record["features"]["descriptor"] = desc
        print(f"Descriptor count: {len(descriptor_names)}")
        print(f"Descriptor invalid values replaced with 0 before scaling: {invalid_descriptor_count}")
        print(f"Descriptor scale quantile from training: {descriptor_scale_quantile}")
        print(f"Descriptor normalized values clipped to +/- {descriptor_clip_value}")
    else:
        descriptor_scale = np.ones(len(descriptor_names), dtype=np.float32)
        print("Warning: no descriptor features were generated successfully.")

    print(f"RDKit molecule failures: {len(mol_failed)}")
    if mol_failed:
        print("First molecule failures:")
        for item in mol_failed[:10]:
            print(f"  row {item['index']}: {item['smiles']} -> {item['reason']}")

    if descriptor_function_failures:
        print(f"Descriptor function-level failures: {sum(descriptor_function_failures.values())}")
        for failure, count in list(descriptor_function_failures.items())[:10]:
            print(f"  {failure} ({count})")

    feature_sets = [
        ("Avalon", ["Avalon"], [f"Avalon_{i}" for i in range(fp_n_bits)], None),
        ("ECFP4", ["ECFP4"], [f"ECFP4_{i}" for i in range(fp_n_bits)], None),
        ("descriptor", ["descriptor"], descriptor_names, descriptor_scale),
        (
            "Avalon_descriptor",
            ["Avalon", "descriptor"],
            [f"Avalon_{i}" for i in range(fp_n_bits)] + descriptor_names,
            descriptor_scale,
        ),
        (
            "ECFP4_descriptor",
            ["ECFP4", "descriptor"],
            [f"ECFP4_{i}" for i in range(fp_n_bits)] + descriptor_names,
            descriptor_scale,
        ),
    ]

    output_paths = {}
    for suffix, keys, feature_names, scale in feature_sets:
        if "Avalon" in keys and not AVALON_AVAILABLE:
            print(f"Skip {suffix}: Avalon fingerprint is unavailable.")
            continue
        selected = _select_rows(records, keys)
        output_path = output_dir / f"{output_stem}_{suffix}.npz"
        print(f"Preparing {suffix}: valid rows {len(selected['smiles'])}, failed rows {len(selected['failed_records'])}")
        _save_npz(
            output_path=output_path,
            X=selected["X"],
            y=selected["y"],
            groups=selected["groups"],
            smiles=selected["smiles"],
            task_names=task_names,
            feature_names=feature_names,
            failed_records=selected["failed_records"],
            descriptor_scale=scale,
        )
        output_paths[suffix] = output_path

    return output_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Build ECFP4, Avalon, and RDKit descriptor features from a raw CSV.")
    parser.add_argument("--input_csv", default=None, help="Raw CSV containing SMILES, group, and task label columns.")
    parser.add_argument("--input_root", default=None, help="Build all CSV files under this root.")
    parser.add_argument("--pattern", default="*.csv", help="Glob pattern used with --input_root.")
    parser.add_argument("--output_dir", default=None, help="Optional output directory. Defaults to raw_data mirrored under data/fp_descriptor_data.")
    parser.add_argument("--fp_n_bits", type=int, default=1024, help="Bit length for ECFP4 and Avalon fingerprints.")
    parser.add_argument("--dry_run", action="store_true", help="List target CSV files without writing feature files.")
    parser.add_argument(
        "--descriptor_scale_quantile",
        type=float,
        default=0.95,
        help="Training-set absolute-value quantile used as descriptor divisor.",
    )
    parser.add_argument(
        "--descriptor_clip_value",
        type=float,
        default=10.0,
        help="Clip normalized descriptor values to +/- this value. Use <=0 to disable clipping.",
    )
    return parser.parse_args()


def iter_input_csvs(input_root, pattern="*.csv"):
    input_root = Path(input_root)
    yield from sorted(input_root.rglob(pattern))


if __name__ == "__main__":
    args = parse_args()
    if bool(args.input_csv) == bool(args.input_root):
        raise ValueError("Provide exactly one of --input_csv or --input_root.")

    if args.input_csv:
        if args.dry_run:
            print(Path(args.input_csv))
        else:
            build_and_save_molecular_features(
                input_csv=args.input_csv,
                output_dir=args.output_dir,
                fp_n_bits=args.fp_n_bits,
                descriptor_scale_quantile=args.descriptor_scale_quantile,
                descriptor_clip_value=args.descriptor_clip_value if args.descriptor_clip_value > 0 else None,
            )
    else:
        csv_paths = list(iter_input_csvs(args.input_root, pattern=args.pattern))
        if args.dry_run:
            for path in csv_paths:
                print(path)
            print(f"Matched CSV files: {len(csv_paths)}")
        else:
            for path in csv_paths:
                build_and_save_molecular_features(
                    input_csv=path,
                    output_dir=None,
                    fp_n_bits=args.fp_n_bits,
                    descriptor_scale_quantile=args.descriptor_scale_quantile,
                    descriptor_clip_value=args.descriptor_clip_value if args.descriptor_clip_value > 0 else None,
                )
