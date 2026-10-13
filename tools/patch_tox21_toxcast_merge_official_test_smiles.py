from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(r"E:\z_data\tox21_final2")
SOURCE_MERGE_DIR = SOURCE_ROOT / "data" / "raw_data" / "merge_data" / "Tox21_Toxcast"
TARGET_MERGE_DIR = PROJECT_ROOT / "data" / "raw_data" / "merge_data" / "Tox21_Toxcast"
OFFICIAL_TEST_CSV = PROJECT_ROOT / "data" / "raw_data" / "Tox21" / "origin_data(before_wash)" / "test.csv"
REPORT_DIR = PROJECT_ROOT / "analysis" / "data_audit"

PATCH_RELATIVE_FILES = [
    Path("data.csv"),
    Path("contain_valid") / "data_by_ecfp4_similarity.csv",
]

PUBCHEM_SMILES_REPLACEMENTS = {
    "NCGC00357062-01": "C1=CC(=CC=C1C2=C(C(=C(N2)C(F)(F)F)Br)C#N)Cl",
    "NCGC00357026-01": "C1=CC=C(C=C1)C2=NC3=CC=CC=C3N2",
}
NON_LABEL_COLUMNS = {"ID", "SMILES", "canonical_smiles"}


def canonical_smiles(smiles: object) -> str | None:
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def load_official_test() -> pd.DataFrame:
    official = pd.read_csv(OFFICIAL_TEST_CSV)
    if "smiles" in official.columns and "SMILES" not in official.columns:
        official = official.rename(columns={"smiles": "SMILES"})
    official = official.copy()
    official["SMILES"] = official.apply(
        lambda row: PUBCHEM_SMILES_REPLACEMENTS.get(str(row["ID"]), row["SMILES"]),
        axis=1,
    )
    official["canonical_smiles"] = official["SMILES"].map(canonical_smiles)
    failed = official[official["canonical_smiles"].isna()][["ID", "SMILES"]]
    if not failed.empty:
        raise ValueError(f"Official test still has RDKit failures: {failed.to_dict('records')}")
    if len(official) != 647:
        raise ValueError(f"Expected 647 official test rows, got {len(official)}")
    return official


def copy_merge_dir() -> None:
    if not SOURCE_MERGE_DIR.exists():
        raise FileNotFoundError(SOURCE_MERGE_DIR)
    if TARGET_MERGE_DIR.exists():
        shutil.rmtree(TARGET_MERGE_DIR)
    TARGET_MERGE_DIR.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SOURCE_MERGE_DIR, TARGET_MERGE_DIR)


def patch_merge_csv(path: Path, official: pd.DataFrame) -> dict:
    df = pd.read_csv(path).copy()
    if "SMILES" not in df.columns or "group" not in df.columns:
        raise ValueError(f"{path} must contain SMILES and group columns")

    df["_canonical_smiles"] = df["SMILES"].map(canonical_smiles)
    official_canon = set(official["canonical_smiles"])

    non_test = df[df["group"].astype(str) != "test"].copy()
    removed_non_test = non_test[non_test["_canonical_smiles"].isin(official_canon)].copy()
    non_test = non_test[~non_test["_canonical_smiles"].isin(official_canon)].copy()

    test = df[df["group"].astype(str) == "test"].copy()
    official_by_canon = defaultdict(list)
    for _, row in official.iterrows():
        official_by_canon[row["canonical_smiles"]].append(row)

    label_cols = [c for c in official.columns if c not in NON_LABEL_COLUMNS]
    patched_existing = 0
    for idx, row in test.iterrows():
        records = official_by_canon.get(row["_canonical_smiles"], [])
        if not records:
            continue
        rec = records[0]
        patched_existing += 1
        test.at[idx, "SMILES"] = rec["SMILES"]
        for col in label_cols:
            if col in test.columns:
                test.at[idx, col] = rec[col]

    # Append official test rows absent from the retained old merge test by SMILES frequency.
    test_counts = Counter(test["SMILES"].astype(str))
    official_counts = Counter(official["SMILES"].astype(str))
    append_rows = []
    for smi, needed in official_counts.items():
        deficit = needed - test_counts.get(smi, 0)
        if deficit <= 0:
            continue
        rec = official[official["SMILES"].astype(str).eq(smi)].iloc[0]
        for _ in range(deficit):
            row = pd.Series({col: np.nan for col in df.columns})
            row["SMILES"] = rec["SMILES"]
            row["group"] = "test"
            row["_canonical_smiles"] = rec["canonical_smiles"]
            for col in label_cols:
                if col in row.index:
                    row[col] = rec[col]
            append_rows.append(row)

    out = pd.concat([non_test, test, pd.DataFrame(append_rows)], ignore_index=True)
    out = out.drop(columns=["_canonical_smiles"], errors="ignore")
    out.to_csv(path, index=False)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = "__".join(path.relative_to(TARGET_MERGE_DIR).parts).replace(".csv", "")
    removed_non_test.drop(columns=["_canonical_smiles"], errors="ignore").to_csv(
        REPORT_DIR / f"{safe_name}_removed_non_test_official_overlap.csv",
        index=False,
    )

    check = pd.read_csv(path)
    check_canon = check["SMILES"].map(canonical_smiles)
    group = check["group"].astype(str)
    train_valid = set(check_canon[group.isin(["training", "valid"])].dropna())
    test_canon = set(check_canon[group.eq("test")].dropna())
    return {
        "file": str(path.relative_to(PROJECT_ROOT)),
        "rows": int(len(check)),
        "group_counts": {str(k): int(v) for k, v in check["group"].value_counts().to_dict().items()},
        "rdkit_failed": int(check_canon.isna().sum()),
        "patched_existing_test_rows": int(patched_existing),
        "appended_official_test_rows": int(len(append_rows)),
        "removed_non_test_official_overlap_rows": int(len(removed_non_test)),
        "train_valid_test_canonical_overlap": int(len(train_valid & test_canon)),
    }


def main() -> None:
    official = load_official_test()
    copy_merge_dir()
    summaries = [patch_merge_csv(TARGET_MERGE_DIR / rel, official) for rel in PATCH_RELATIVE_FILES]
    extra = TARGET_MERGE_DIR / "official_tox21_test_smiles.csv"
    if extra.exists():
        extra.unlink()
    summary = {
        "mode": "merge_test_876_keep_old_merge_test_and_append_missing_official_tox21_test_rows",
        "official_test_rows": int(len(official)),
        "official_test_unique_smiles": int(official["SMILES"].nunique()),
        "official_test_unique_canonical": int(official["canonical_smiles"].nunique()),
        "patched_files": summaries,
        "training_cluster_CV_copied": (TARGET_MERGE_DIR / "contain_valid" / "training_cluster_CV").exists(),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "tox21_toxcast_merge_test876_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
