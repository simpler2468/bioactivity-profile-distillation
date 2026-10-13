from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdmolops
from rdkit.Chem.Scaffolds import MurckoScaffold

try:
    from dataset.molecular_features import (
        AVALON_AVAILABLE,
        descriptor_definitions,
        make_avalon,
        make_descriptors,
        make_ecfp4,
    )
except ImportError:
    from molecular_features import (
        AVALON_AVAILABLE,
        descriptor_definitions,
        make_avalon,
        make_descriptors,
        make_ecfp4,
    )


class Tox21DatasetProcessor:
    """
    Processor for building cleaned Tox21 raw datasets.

    Current scope:
    - clean SMILES and keep labels/groups aligned
    - merge duplicate standardized SMILES and resolve label conflicts
    - assign Murcko or generic scaffold codes
    - build the base Tox21 train/test dataset
    """

    DEFAULT_TRAIN_PATH = "data/raw_data/Tox21/origin_data(before_wash)/train.csv"
    DEFAULT_TEST_PATH = "data/raw_data/Tox21/origin_data(before_wash)/test.csv"
    DEFAULT_TOX21_OUTPUT_PATH = "data/raw_data/Tox21/data.csv"
    DEFAULT_TOXCAST_PATH = "data/raw_data/Tox21_from_toxcast/data.csv"
    DEFAULT_TOX21_TOXCAST_OUTPUT_PATH = "data/raw_data/merge_data/Tox21_Toxcast/data.csv"
    DEFAULT_TOX21_TOXCAST_ECFP4_SPLIT_PATH = (
        "data/raw_data/merge_data/Tox21_Toxcast/contain_valid/data_by_ecfp4_similarity.csv"
    )
    DEFAULT_TRAINING_CLUSTER_CV_DIR = (
        "data/raw_data/merge_data/Tox21_Toxcast/contain_valid/training_cluster_CV"
    )
    DEFAULT_TOXRIC_TOX21_TOXCAST_OUTPUT_DIR = (
        "data/raw_data/merge_data/TOXRIC_Tox21_Toxcast"
    )
    DEFAULT_PUBCHEM_TOX21_TOXCAST_OUTPUT_DIR = (
        "data/raw_data/merge_data/Pubchem_Tox21_Toxcast"
    )
    DEFAULT_TOXRIC_SPLIT_TEMPLATE = (
        "data/raw_data/TOXRIC/TOXRIC_canonical_in_vivo_{split_id}.csv"
    )
    DEFAULT_PUBCHEM_SPLIT_TEMPLATE = (
        "data/raw_data/Pubchem/pubchem_canonical_in_vivo_{split_id}.csv"
    )
    DEFAULT_METADATA_COLS = ["ID", "inchikey", "sdftitle", "order", "CVfold", "set"]
    DEFAULT_GROUP_PRIORITY = ("test", "valid", "training")
    PUBCHEM_SMILES_REPLACEMENTS = {
        "NCGC00357062-01": "C1=CC(=CC=C1C2=C(C(=C(N2)C(F)(F)F)Br)C#N)Cl",
        "NCGC00357026-01": "C1=CC=C(C=C1)C2=NC3=CC=CC=C3N2",
    }

    def __init__(
        self,
        salt_get_max_fragment=True,
        metadata_cols=None,
        scaffold_method="murcko",
    ):
        self.salt_get_max_fragment = salt_get_max_fragment
        self.metadata_cols = list(metadata_cols or self.DEFAULT_METADATA_COLS)
        self.scaffold_method = scaffold_method
        self._smiles_cache = {}
        self.stats = {}

    @staticmethod
    def _read_df(df):
        if isinstance(df, (str, Path)):
            return pd.read_csv(df)
        return df.copy()

    def _infer_label_cols(self, df, label_cols=None):
        if label_cols is not None:
            return list(label_cols)

        ignored = set(self.metadata_cols + ["SMILES", "smiles", "group"])
        return [col for col in df.columns if col not in ignored]

    @staticmethod
    def _as_label_df(labels):
        if isinstance(labels, pd.DataFrame):
            return labels.reset_index(drop=True).copy()
        if isinstance(labels, pd.Series):
            return labels.reset_index(drop=True).to_frame()
        return pd.DataFrame(labels).reset_index(drop=True)

    def solve_rdkit_failed_to_read(self, test_df, id_col="ID", smiles_col="SMILES"):
        """
        Fix known official Tox21 test SMILES that RDKit cannot parse.

        These replacements were obtained by querying PubChem with the NCGC IDs.
        This method intentionally does not implement online lookup; it only
        applies the two fixed replacements required by the official test set.
        """
        data = self._read_df(test_df)
        if id_col not in data.columns:
            raise ValueError(f"Test dataframe is missing ID column: {id_col}")
        if smiles_col not in data.columns:
            raise ValueError(f"Test dataframe is missing SMILES column: {smiles_col}")

        replacement_records = []
        for compound_id, replacement_smiles in self.PUBCHEM_SMILES_REPLACEMENTS.items():
            mask = data[id_col].astype(str) == compound_id
            match_count = int(mask.sum())
            if match_count != 1:
                raise ValueError(
                    f"Expected exactly one official test row for {compound_id}, found {match_count}."
                )
            original_smiles = str(data.loc[mask, smiles_col].iloc[0])
            data.loc[mask, smiles_col] = replacement_smiles
            replacement_records.append(
                {
                    "ID": compound_id,
                    "original_smiles": original_smiles,
                    "replacement_smiles": replacement_smiles,
                    "source": "PubChem query by NCGC ID",
                }
            )

        self.stats["solve_rdkit_failed_to_read"] = {
            "num_replacements": len(replacement_records),
            "replacement_records": replacement_records,
        }
        return data

    def standardize_smiles(self, smiles):
        smiles = str(smiles)
        if smiles in self._smiles_cache:
            return self._smiles_cache[smiles]

        if smiles == "" or smiles.lower() == "nan":
            result = (None, "rdkit_failed")
            self._smiles_cache[smiles] = result
            return result

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            result = (None, "rdkit_failed")
            self._smiles_cache[smiles] = result
            return result

        if "." in smiles:
            fragments = rdmolops.GetMolFrags(mol, asMols=True)
            if len(fragments) > 1:
                if not self.salt_get_max_fragment:
                    result = (None, "mixture")
                    self._smiles_cache[smiles] = result
                    return result
                mol = max(fragments, key=lambda fragment: fragment.GetNumHeavyAtoms())

        has_carbon = any(atom.GetSymbol() == "C" for atom in mol.GetAtoms())
        if not has_carbon:
            result = (None, "inorganic")
            self._smiles_cache[smiles] = result
            return result

        result = (Chem.MolToSmiles(mol), None)
        self._smiles_cache[smiles] = result
        return result

    def clean_compounds(self, smiles_list, labels, groups, standardize_smiles=True):
        """
        Clean compounds while keeping SMILES, labels, and groups position-aligned.

        Returns
        -------
        tuple[list[str], pandas.DataFrame, list[str], dict]
            Cleaned standardized SMILES, aligned labels, aligned groups, and stats.
        """
        smiles_series = pd.Series(smiles_list).reset_index(drop=True)
        label_df = self._as_label_df(labels)
        group_series = pd.Series(groups).reset_index(drop=True)

        if not (len(smiles_series) == len(label_df) == len(group_series)):
            raise ValueError("smiles_list, labels, and groups must have the same length.")

        stats = {
            "origin_rows": int(len(smiles_series)),
            "rdkit_failed": 0,
            "mixture_removed": 0,
            "inorganic": 0,
            "kept_rows": 0,
        }

        clean_smiles = []
        clean_label_indices = []
        clean_groups = []

        for idx, smiles in smiles_series.items():
            if standardize_smiles:
                standardized_smiles, reason = self.standardize_smiles(smiles)
            else:
                if pd.isna(smiles) or str(smiles) == "" or str(smiles).lower() == "nan":
                    standardized_smiles, reason = None, "rdkit_failed"
                else:
                    standardized_smiles, reason = str(smiles), None
            if standardized_smiles is None:
                if reason == "mixture":
                    stats["mixture_removed"] += 1
                else:
                    stats[reason] += 1
                continue

            clean_smiles.append(standardized_smiles)
            clean_label_indices.append(idx)
            clean_groups.append(group_series.iloc[idx])
            stats["kept_rows"] += 1

        clean_labels = label_df.iloc[clean_label_indices].reset_index(drop=True)
        return clean_smiles, clean_labels, clean_groups, stats

    @staticmethod
    def _resolve_group(groups, group_priority):
        group_set = set(pd.Series(groups).dropna().astype(str))
        for group in group_priority:
            if group in group_set:
                return group
        if group_set:
            return sorted(group_set)[0]
        return np.nan

    def resolve_label_conflicts(
        self,
        smiles_list,
        labels,
        groups,
        group_priority=None,
    ):
        """
        Merge duplicate SMILES and resolve label conflicts.

        Label rules for each task:
        - all values empty -> NaN
        - one unique non-null value -> keep it
        - conflicting non-null values -> NaN
        """
        group_priority = tuple(group_priority or self.DEFAULT_GROUP_PRIORITY)
        smiles_series = pd.Series(smiles_list, name="SMILES").reset_index(drop=True)
        label_df = self._as_label_df(labels)
        group_series = pd.Series(groups, name="group").reset_index(drop=True)

        if not (len(smiles_series) == len(label_df) == len(group_series)):
            raise ValueError("smiles_list, labels, and groups must have the same length.")

        work_df = pd.concat([smiles_series, label_df, group_series], axis=1)
        label_cols = list(label_df.columns)
        rows = []
        conflict_counts = {col: 0 for col in label_cols}

        for smiles, group_df in work_df.groupby("SMILES", sort=False):
            row = {"SMILES": smiles}
            for col in label_cols:
                values = pd.to_numeric(group_df[col], errors="coerce").dropna()
                unique_values = values.unique()
                if len(unique_values) == 0:
                    row[col] = np.nan
                elif len(unique_values) == 1:
                    row[col] = unique_values[0]
                else:
                    row[col] = np.nan
                    conflict_counts[col] += 1
            row["group"] = self._resolve_group(group_df["group"], group_priority)
            rows.append(row)

        merged_df = pd.DataFrame(rows)
        merged_df = merged_df[["SMILES"] + label_cols + ["group"]]

        stats = {
            "origin_rows": int(len(work_df)),
            "final_rows": int(len(merged_df)),
            "duplicate_rows_merged": int(len(work_df) - len(merged_df)),
            "label_conflicts": conflict_counts,
            "group_counts": merged_df["group"].value_counts(dropna=False).to_dict(),
        }

        return (
            merged_df["SMILES"].tolist(),
            merged_df[label_cols].reset_index(drop=True),
            merged_df["group"].tolist(),
            stats,
        )

    def get_scaffold_codes(self, smiles_list, scaffold_method=None):
        """
        Return one scaffold code per SMILES.

        Supported scaffold methods:
        - murcko: RDKit Murcko scaffold
        - generic: RDKit MakeScaffoldGeneric applied to the molecule, matching
          the draft notebook experiment
        """
        scaffold_method = scaffold_method or self.scaffold_method
        scaffold_method = str(scaffold_method).lower()
        if scaffold_method not in {"murcko", "generic"}:
            raise ValueError("scaffold_method must be 'murcko' or 'generic'.")

        scaffold_codes = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                scaffold_codes.append("__NO_SCAFFOLD__")
                continue

            try:
                if scaffold_method == "murcko":
                    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                else:
                    scaffold = MurckoScaffold.MakeScaffoldGeneric(mol)
            except Exception:
                scaffold_codes.append("__NO_SCAFFOLD__")
                continue

            if scaffold is None or scaffold.GetNumAtoms() == 0:
                scaffold_codes.append("__NO_SCAFFOLD__")
            else:
                scaffold_codes.append(Chem.MolToSmiles(scaffold))
        return scaffold_codes

    @staticmethod
    def _empty_fingerprint_matrix(row_count, n_bits):
        return np.zeros((int(row_count), int(n_bits)), dtype=np.float32)

    def get_ecfp4_fingerprints(self, smiles_list, n_bits=1024, return_failed=False):
        """
        Return an ECFP4 fingerprint matrix for a SMILES list.

        Invalid molecules keep their original row position and receive an all-zero
        fingerprint. Use return_failed=True to also get failed row records.
        """
        fingerprints = self._empty_fingerprint_matrix(len(smiles_list), n_bits)
        failed_records = []

        for idx, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                failed_records.append(
                    {"index": int(idx), "smiles": smiles, "reason": "rdkit_mol_failed"}
                )
                continue
            try:
                fingerprints[idx] = make_ecfp4(mol, n_bits=n_bits)
            except Exception as exc:
                failed_records.append(
                    {
                        "index": int(idx),
                        "smiles": smiles,
                        "reason": f"ecfp4_failed: {type(exc).__name__}: {exc}",
                    }
                )

        if return_failed:
            return fingerprints, failed_records
        return fingerprints

    def get_avalon_fingerprints(self, smiles_list, n_bits=1024, return_failed=False):
        """
        Return an Avalon fingerprint matrix for a SMILES list.

        Invalid or failed molecules keep their original row position and receive
        an all-zero fingerprint. Use return_failed=True to also get failed records.
        """
        if not AVALON_AVAILABLE:
            raise RuntimeError("RDKit Avalon fingerprint is not available in this environment.")

        fingerprints = self._empty_fingerprint_matrix(len(smiles_list), n_bits)
        failed_records = []

        for idx, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                failed_records.append(
                    {"index": int(idx), "smiles": smiles, "reason": "rdkit_mol_failed"}
                )
                continue
            try:
                fingerprints[idx] = make_avalon(mol, n_bits=n_bits)
            except Exception as exc:
                failed_records.append(
                    {
                        "index": int(idx),
                        "smiles": smiles,
                        "reason": f"avalon_failed: {type(exc).__name__}: {exc}",
                    }
                )

        if return_failed:
            return fingerprints, failed_records
        return fingerprints

    def get_descriptors(self, smiles_list, return_failed=False):
        """
        Return raw RDKit descriptors for a SMILES list.

        The returned DataFrame keeps one row per input molecule and uses RDKit
        descriptor names as columns. Invalid molecule rows are filled with NaN.
        """
        desc_defs = descriptor_definitions()
        descriptor_names = [name for name, _ in desc_defs]
        descriptor_rows = []
        failed_records = []

        for idx, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                descriptor_rows.append(np.full((len(descriptor_names),), np.nan, dtype=np.float64))
                failed_records.append(
                    {"index": int(idx), "smiles": smiles, "reason": "rdkit_mol_failed"}
                )
                continue

            values, desc_failed = make_descriptors(mol, desc_defs=desc_defs)
            descriptor_rows.append(values)
            if desc_failed:
                failed_records.append(
                    {
                        "index": int(idx),
                        "smiles": smiles,
                        "reason": "; ".join(desc_failed),
                    }
                )

        descriptors = pd.DataFrame(descriptor_rows, columns=descriptor_names)
        if return_failed:
            return descriptors, failed_records
        return descriptors

    @staticmethod
    def _find_parent(parents, item):
        while parents[item] != item:
            parents[item] = parents[parents[item]]
            item = parents[item]
        return item

    @classmethod
    def _union_parent(cls, parents, ranks, left, right):
        left_root = cls._find_parent(parents, left)
        right_root = cls._find_parent(parents, right)
        if left_root == right_root:
            return
        if ranks[left_root] < ranks[right_root]:
            parents[left_root] = right_root
        elif ranks[left_root] > ranks[right_root]:
            parents[right_root] = left_root
        else:
            parents[right_root] = left_root
            ranks[left_root] += 1

    def get_ecfp4_similarity_clusters(
        self,
        smiles_list,
        similarity_threshold=0.7,
        n_bits=1024,
        radius=2,
        return_stats=False,
    ):
        """
        Cluster compounds by ECFP4 Tanimoto single-linkage.

        Two compounds are connected when their ECFP4 Tanimoto similarity is at
        least similarity_threshold. Cluster ids are connected components of this
        graph, so A can share a cluster with C through B even if A-C similarity
        is below the threshold.
        """
        if not 0 <= similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1.")

        smiles_series = pd.Series(smiles_list).reset_index(drop=True)
        fingerprints = []
        failed_records = []
        for idx, smiles in smiles_series.items():
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                fingerprints.append(None)
                failed_records.append(
                    {"index": int(idx), "smiles": smiles, "reason": "rdkit_mol_failed"}
                )
                continue
            try:
                fingerprints.append(
                    AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
                )
            except Exception as exc:
                fingerprints.append(None)
                failed_records.append(
                    {
                        "index": int(idx),
                        "smiles": smiles,
                        "reason": f"ecfp4_failed: {type(exc).__name__}: {exc}",
                    }
                )

        parents = list(range(len(fingerprints)))
        ranks = [0] * len(fingerprints)
        valid_indices = [idx for idx, fp in enumerate(fingerprints) if fp is not None]

        edge_count = 0
        for pos, left_idx in enumerate(valid_indices):
            right_indices = valid_indices[pos + 1 :]
            if not right_indices:
                continue
            similarities = DataStructs.BulkTanimotoSimilarity(
                fingerprints[left_idx],
                [fingerprints[right_idx] for right_idx in right_indices],
            )
            for right_idx, similarity in zip(right_indices, similarities):
                if similarity >= similarity_threshold:
                    self._union_parent(parents, ranks, left_idx, right_idx)
                    edge_count += 1

        root_to_cluster_id = {}
        cluster_ids = []
        for idx in range(len(fingerprints)):
            root = self._find_parent(parents, idx)
            if root not in root_to_cluster_id:
                root_to_cluster_id[root] = f"ecfp4_tanimoto_{similarity_threshold:g}_{len(root_to_cluster_id):06d}"
            cluster_ids.append(root_to_cluster_id[root])

        if return_stats:
            cluster_counts = pd.Series(cluster_ids).value_counts()
            stats = {
                "molecules": int(len(cluster_ids)),
                "clusters": int(cluster_counts.shape[0]),
                "repeated_clusters": int((cluster_counts > 1).sum()),
                "singleton_clusters": int((cluster_counts == 1).sum()),
                "molecules_in_repeated_clusters": int(cluster_counts[cluster_counts > 1].sum()),
                "molecules_in_singleton_clusters": int(cluster_counts[cluster_counts == 1].sum()),
                "largest_cluster_size": int(cluster_counts.max()) if len(cluster_counts) else 0,
                "edge_count": int(edge_count),
                "failed_records": failed_records,
                "similarity_threshold": float(similarity_threshold),
                "n_bits": int(n_bits),
                "radius": int(radius),
            }
            return cluster_ids, stats
        return cluster_ids

    @staticmethod
    def _label_columns(df):
        return [col for col in df.columns if col not in ["SMILES", "group"]]

    @staticmethod
    def _group_counts_for_stats(groups):
        group_series = pd.Series(groups, dtype=object)
        group_series = group_series.where(group_series.notna(), "NA")
        return group_series.value_counts(dropna=False).to_dict()

    def clear_group(self, df, group_col="group"):
        """
        Return a dataframe with all group labels cleared to NA.
        """
        data = self._read_df(df)
        data[group_col] = pd.NA
        self.stats["clear_group"] = {
            "rows": int(len(data)),
            "group_col": group_col,
            "group_counts": self._group_counts_for_stats(data[group_col]),
        }
        return data

    def assign_na_group_by_random_split(
        self,
        df,
        target_groups=("training", "test"),
        group_probs=None,
        random_seed=42,
        group_col="group",
    ):
        """
        Randomly assign NA group rows using existing group proportions by default.

        Existing non-NA group labels are preserved. When group_probs is None,
        probabilities are estimated from non-NA rows whose group is in
        target_groups. group_probs can be either a dict or a sequence aligned
        with target_groups.
        """
        data = self._read_df(df)
        if group_col not in data.columns:
            raise ValueError(f"Input dataframe is missing group column: {group_col}")

        target_groups = tuple(target_groups)
        if len(target_groups) < 2:
            raise ValueError("target_groups must contain at least two groups.")

        na_mask = data[group_col].isna()
        if not na_mask.any():
            self.stats["assign_na_group_by_random_split"] = {
                "rows": int(len(data)),
                "assigned_rows": 0,
                "target_groups": list(target_groups),
                "probabilities": {},
                "random_seed": int(random_seed),
                "final_group_counts": self._group_counts_for_stats(data[group_col]),
            }
            return data

        if group_probs is None:
            reference_counts = data.loc[
                data[group_col].isin(target_groups),
                group_col,
            ].value_counts()
            if reference_counts.sum() == 0:
                raise ValueError(
                    "Cannot infer group probabilities because no non-NA rows match target_groups."
                )
            probabilities = np.asarray(
                [reference_counts.get(group, 0) for group in target_groups],
                dtype=np.float64,
            )
            probabilities = probabilities / probabilities.sum()
        elif isinstance(group_probs, dict):
            probabilities = np.asarray(
                [group_probs.get(group, 0.0) for group in target_groups],
                dtype=np.float64,
            )
            if probabilities.sum() <= 0:
                raise ValueError("group_probs must contain positive probability mass.")
            probabilities = probabilities / probabilities.sum()
        else:
            probabilities = np.asarray(group_probs, dtype=np.float64)
            if len(probabilities) != len(target_groups):
                raise ValueError("group_probs must have the same length as target_groups.")
            if probabilities.sum() <= 0:
                raise ValueError("group_probs must contain positive probability mass.")
            probabilities = probabilities / probabilities.sum()

        rng = np.random.default_rng(random_seed)
        assigned_values = rng.choice(
            np.asarray(target_groups, dtype=object),
            size=int(na_mask.sum()),
            p=probabilities,
        )
        data = data.copy()
        data.loc[na_mask, group_col] = assigned_values

        self.stats["assign_na_group_by_random_split"] = {
            "rows": int(len(data)),
            "assigned_rows": int(na_mask.sum()),
            "target_groups": list(target_groups),
            "probabilities": {
                group: float(prob)
                for group, prob in zip(target_groups, probabilities)
            },
            "assigned_group_counts": self._group_counts_for_stats(assigned_values),
            "random_seed": int(random_seed),
            "final_group_counts": self._group_counts_for_stats(data[group_col]),
        }
        return data

    def valid_from_train_by_random_split(
        self,
        df,
        valid_ratio_from_training=0.1,
        random_seed=42,
        output_path=None,
        save=True,
    ):
        """
        Randomly split valid rows from training while preserving test rows.
        """
        if not 0 < valid_ratio_from_training < 1:
            raise ValueError("valid_ratio_from_training must be between 0 and 1.")

        data = self._read_df(df)
        missing = [col for col in ["SMILES", "group"] if col not in data.columns]
        if missing:
            raise ValueError(f"Input dataframe is missing required columns: {missing}")
        if data["group"].isna().any():
            raise ValueError("Input dataframe group column must not contain empty values.")

        original_group = data["group"].copy()
        data = data.copy()
        training_indices = data.index[data["group"] == "training"].to_numpy()
        valid_size = int(round(len(training_indices) * valid_ratio_from_training))
        rng = np.random.default_rng(random_seed)
        valid_indices = rng.choice(training_indices, size=valid_size, replace=False)
        data.loc[valid_indices, "group"] = "valid"

        if save:
            if output_path is None:
                raise ValueError("output_path is required when save=True.")
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(output_path, index=False)

        self.stats["valid_from_train_by_random_split"] = {
            "input_rows": int(len(data)),
            "original_group_counts": self._group_counts_for_stats(original_group),
            "final_group_counts": self._group_counts_for_stats(data["group"]),
            "valid_ratio_from_training": float(valid_ratio_from_training),
            "target_valid_size": int(valid_size),
            "actual_valid_size": int((data["group"] == "valid").sum()),
            "random_seed": int(random_seed),
            "output_path": str(output_path) if output_path is not None else None,
        }
        return data

    def merge_dataset(self, df1, df2):
        """
        Merge two already-processed datasets by SMILES with df1 priority.

        Both inputs are expected to have clean canonical SMILES and resolved
        internal duplicate labels. df1 must have a usable group column. df2 must
        have a group column, but empty df2 groups are preserved as empty when
        the compound only exists in df2. For overlapping compounds, group
        follows df1. Label columns are unioned; if an
        unexpected duplicate label column has non-null values in both datasets,
        df1 is used first.
        """
        left = self._read_df(df1)
        right = self._read_df(df2)

        for name, data in [("df1", left), ("df2", right)]:
            missing = [col for col in ["SMILES", "group"] if col not in data.columns]
            if missing:
                raise ValueError(f"{name} is missing required columns: {missing}")
            duplicated = data["SMILES"].duplicated()
            if duplicated.any():
                examples = data.loc[duplicated, "SMILES"].head(5).tolist()
                raise ValueError(
                    f"{name} contains duplicate SMILES. Resolve internal duplicates first. "
                    f"Examples: {examples}"
                )
            if name == "df1" and data["group"].isna().any():
                raise ValueError("df1 group values must not be empty because df1 has merge priority.")

        left_label_cols = self._label_columns(left)
        right_label_cols = self._label_columns(right)
        label_cols = list(dict.fromkeys(left_label_cols + right_label_cols))

        left_indexed = left.set_index("SMILES", drop=False)
        right_indexed = right.set_index("SMILES", drop=False)
        all_smiles = list(dict.fromkeys(left_indexed.index.tolist() + right_indexed.index.tolist()))

        left_aligned = left_indexed.reindex(all_smiles)
        right_aligned = right_indexed.reindex(all_smiles)
        left_present = pd.Series(all_smiles, index=all_smiles).isin(left_indexed.index).to_numpy()

        merged_columns = {"SMILES": np.asarray(all_smiles, dtype=object)}
        for col in label_cols:
            left_values = (
                left_aligned[col]
                if col in left_aligned.columns
                else pd.Series(np.nan, index=all_smiles)
            )
            right_values = (
                right_aligned[col]
                if col in right_aligned.columns
                else pd.Series(np.nan, index=all_smiles)
            )
            merged_columns[col] = left_values.combine_first(right_values).to_numpy()

        merged_columns["group"] = np.where(
            left_present,
            left_aligned["group"].to_numpy(dtype=object),
            right_aligned["group"].to_numpy(dtype=object),
        )
        merged = pd.DataFrame(merged_columns)
        merged = merged[["SMILES"] + label_cols + ["group"]]
        merged = merged.sort_values(["group", "SMILES"]).reset_index(drop=True)

        self.stats["merge_dataset"] = {
            "df1_rows": int(len(left_indexed)),
            "df2_rows": int(len(right_indexed)),
            "merged_rows": int(len(merged)),
            "overlap_smiles": int(len(set(left_indexed.index) & set(right_indexed.index))),
            "df1_label_cols": left_label_cols,
            "df2_label_cols": right_label_cols,
            "merged_label_cols": label_cols,
            "group_counts": self._group_counts_for_stats(merged["group"]),
        }
        return merged

    def assign_group(
        self,
        df,
        similarity_threshold=0.7,
        n_bits=1024,
        radius=2,
        cluster_col="ecfp4_cluster",
    ):
        """
        Assign empty groups by ECFP4 cluster using the conservative rule.

        Original test rows stay test. Original training rows stay training. Only
        empty group rows are assigned:
        - if their cluster contains at least one original test row -> test
        - otherwise -> training

        The returned dataframe includes the ECFP4 cluster id column.
        """
        data = self._read_df(df)
        missing = [col for col in ["SMILES", "group"] if col not in data.columns]
        if missing:
            raise ValueError(f"Input dataframe is missing required columns: {missing}")

        cluster_ids, cluster_stats = self.get_ecfp4_similarity_clusters(
            data["SMILES"],
            similarity_threshold=similarity_threshold,
            n_bits=n_bits,
            radius=radius,
            return_stats=True,
        )
        data = data.copy()
        data[cluster_col] = cluster_ids

        original_group = data["group"].copy()
        na_mask = original_group.isna()
        test_clusters = set(data.loc[original_group == "test", cluster_col])

        data.loc[na_mask & data[cluster_col].isin(test_clusters), "group"] = "test"
        data.loc[na_mask & ~data[cluster_col].isin(test_clusters), "group"] = "training"

        assigned_from_na = data.loc[na_mask, "group"].value_counts(dropna=False).to_dict()
        cluster_group_counts = (
            data.assign(group_label=data["group"].where(data["group"].notna(), "NA"))
            .groupby([cluster_col, "group_label"])
            .size()
            .unstack(fill_value=0)
        )
        for col in ["training", "valid", "test", "NA"]:
            if col not in cluster_group_counts.columns:
                cluster_group_counts[col] = 0

        self.stats["assign_group"] = {
            "input_rows": int(len(data)),
            "original_group_counts": self._group_counts_for_stats(original_group),
            "assigned_from_na": assigned_from_na,
            "final_group_counts": self._group_counts_for_stats(data["group"]),
            "cluster_col": cluster_col,
            "cluster_stats": cluster_stats,
            "clusters_with_original_test": int(len(test_clusters)),
            "training_test_cluster_overlap_after_assignment": int(
                (
                    (cluster_group_counts["training"] > 0)
                    & (cluster_group_counts["test"] > 0)
                ).sum()
            ),
            "remaining_na_rows": int(data["group"].isna().sum()),
            "similarity_threshold": float(similarity_threshold),
            "n_bits": int(n_bits),
            "radius": int(radius),
        }
        return data

    def build_processed_df(
        self,
        df,
        label_cols=None,
        group_col="group",
        default_group=pd.NA,
        standardize_smiles=True,
    ):
        """
        Process a generic SMILES-label dataframe into the common dataset format.

        This helper only wraps the public processing steps:
        clean_compounds -> resolve_label_conflicts -> assemble dataframe.

        Parameters
        ----------
        df : str | Path | pandas.DataFrame
            Input dataframe or CSV path containing SMILES and label columns.
        label_cols : list[str] | None
            Label columns. When None, all non-SMILES/non-group columns are used.
        group_col : str
            Existing group column to use when present.
        default_group : object
            Group value used when group_col is absent. Defaults to pd.NA.
        """
        data = self._read_df(df)
        if "smiles" in data.columns and "SMILES" not in data.columns:
            data = data.rename(columns={"smiles": "SMILES"})
        if "SMILES" not in data.columns:
            raise ValueError("Input dataframe must contain a SMILES or smiles column.")

        if label_cols is None:
            ignored = {"SMILES", "smiles", group_col}
            label_cols = [col for col in data.columns if col not in ignored]
        else:
            label_cols = list(label_cols)

        missing_labels = [col for col in label_cols if col not in data.columns]
        if missing_labels:
            raise ValueError(f"Input dataframe is missing label columns: {missing_labels}")

        if group_col in data.columns:
            groups = data[group_col]
        else:
            groups = pd.Series([default_group] * len(data), name="group")

        clean_smiles, clean_labels, clean_groups, clean_stats = self.clean_compounds(
            data["SMILES"],
            data[label_cols],
            groups,
            standardize_smiles=standardize_smiles,
        )
        clean_smiles_series = pd.Series(clean_smiles)
        if not clean_smiles_series.duplicated().any():
            merged_smiles = clean_smiles
            merged_labels = clean_labels.apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
            merged_groups = list(clean_groups)
            dedup_stats = {
                "origin_rows": int(len(clean_smiles)),
                "final_rows": int(len(clean_smiles)),
                "duplicate_rows_merged": 0,
                "label_conflicts": {col: 0 for col in label_cols},
                "group_counts": self._group_counts_for_stats(merged_groups),
            }
        else:
            merged_smiles, merged_labels, merged_groups, dedup_stats = self.resolve_label_conflicts(
                clean_smiles,
                clean_labels,
                clean_groups,
            )

        result = pd.concat(
            [
                pd.Series(merged_smiles, name="SMILES"),
                merged_labels.reset_index(drop=True),
                pd.Series(merged_groups, name="group"),
            ],
            axis=1,
        )
        result = result[["SMILES"] + label_cols + ["group"]]

        self.stats["build_processed_df"] = {
            "input_rows": int(len(data)),
            "label_cols": label_cols,
            "clean": clean_stats,
            "deduplicate": dedup_stats,
            "final_rows": int(len(result)),
            "group_counts": self._group_counts_for_stats(result["group"]),
            "standardize_smiles": bool(standardize_smiles),
        }
        return result

    def build_tox21_base_df(
        self,
        train_path=DEFAULT_TRAIN_PATH,
        test_path=DEFAULT_TEST_PATH,
        output_path=DEFAULT_TOX21_OUTPUT_PATH,
        label_cols=None,
        save=True,
    ):
        """
        Build cleaned Tox21 train/test dataset and save data/raw_data/Tox21/data.csv.
        """
        train_df = self._read_df(train_path)
        test_df = self._read_df(test_path)

        if "smiles" in train_df.columns and "SMILES" not in train_df.columns:
            train_df = train_df.rename(columns={"smiles": "SMILES"})
        if "smiles" in test_df.columns and "SMILES" not in test_df.columns:
            test_df = test_df.rename(columns={"smiles": "SMILES"})
        if "SMILES" not in train_df.columns or "SMILES" not in test_df.columns:
            raise ValueError("Tox21 train/test files must contain a SMILES or smiles column.")

        label_cols = self._infer_label_cols(train_df, label_cols)
        for col in label_cols:
            if col not in test_df.columns:
                test_df[col] = np.nan

        official_test_df = self.solve_rdkit_failed_to_read(test_df, id_col="ID", smiles_col="SMILES")
        official_test_output = official_test_df[["SMILES"] + label_cols].copy()
        official_test_output["group"] = "test"

        data_for_training_resolution = pd.concat(
            [
                train_df[["SMILES"] + label_cols].assign(group="training"),
                official_test_df[["SMILES"] + label_cols].assign(group="test"),
            ],
            ignore_index=True,
        )

        clean_smiles, clean_labels, clean_groups, clean_stats = self.clean_compounds(
            data_for_training_resolution["SMILES"],
            data_for_training_resolution[label_cols],
            data_for_training_resolution["group"],
        )
        merged_smiles, merged_labels, merged_groups, conflict_stats = self.resolve_label_conflicts(
            clean_smiles,
            clean_labels,
            clean_groups,
        )

        result = pd.concat(
            [
                pd.Series(merged_smiles, name="SMILES"),
                merged_labels.reset_index(drop=True),
                pd.Series(merged_groups, name="group"),
            ],
            axis=1,
        )
        training_result = result[result["group"] == "training"].copy()
        output_cols = ["SMILES"] + label_cols + ["group"]
        result = pd.concat(
            [training_result[output_cols], official_test_output[output_cols]],
            ignore_index=True,
        )
        result = result.sort_values(["group", "SMILES"]).reset_index(drop=True)

        self.stats["build_tox21_base_df"] = {
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "official_test_rows_retained": int(len(official_test_output)),
            "clean": clean_stats,
            "deduplicate": conflict_stats,
            "training_rows_after_removing_test_canonical_duplicates": int(len(training_result)),
            "final_rows": int(len(result)),
            "group_counts": result["group"].value_counts().to_dict(),
            "output_path": str(output_path),
            "test_smiles_replacements": self.stats.get("solve_rdkit_failed_to_read"),
        }

        if save:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(output_path, index=False)

        return result

    def get_tox21_toxcast_merge_data(
        self,
        tox21_train_path=DEFAULT_TRAIN_PATH,
        tox21_test_path=DEFAULT_TEST_PATH,
        toxcast_path=DEFAULT_TOXCAST_PATH,
        output_path=DEFAULT_TOX21_TOXCAST_OUTPUT_PATH,
        similarity_threshold=0.7,
        n_bits=1024,
        radius=2,
        include_cluster_col=False,
        save=True,
    ):
        """
        Build and optionally save the processed Tox21 + ToxCast merged dataset.

        The method deliberately reuses the public processing steps:
        build_tox21_base_df -> build_processed_df -> merge_dataset -> assign_group.
        Empty ToxCast-only groups are assigned by ECFP4 cluster using the
        conservative rule implemented in assign_group.
        """
        tox21_df = self.build_tox21_base_df(
            train_path=tox21_train_path,
            test_path=tox21_test_path,
            save=False,
        )

        toxcast_raw = self._read_df(toxcast_path)
        if "smiles" in toxcast_raw.columns and "SMILES" not in toxcast_raw.columns:
            toxcast_raw = toxcast_raw.rename(columns={"smiles": "SMILES"})
        if "SMILES" not in toxcast_raw.columns:
            raise ValueError("ToxCast dataframe must contain a SMILES or smiles column.")

        toxcast_label_cols = [col for col in toxcast_raw.columns if col not in ["SMILES", "group"]]
        toxcast_df = self.build_processed_df(
            toxcast_raw,
            label_cols=toxcast_label_cols,
            default_group=pd.NA,
        )

        merged_df = self.merge_dataset(tox21_df, toxcast_df)
        assigned_df = self.assign_group(
            merged_df,
            similarity_threshold=similarity_threshold,
            n_bits=n_bits,
            radius=radius,
            cluster_col="ecfp4_cluster",
        )

        output_df = assigned_df.copy()
        if not include_cluster_col and "ecfp4_cluster" in output_df.columns:
            output_df = output_df.drop(columns=["ecfp4_cluster"])

        if save:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_df.to_csv(output_path, index=False)

        self.stats["get_tox21_toxcast_merge_data"] = {
            "tox21_rows": int(len(tox21_df)),
            "toxcast_raw_rows": int(len(toxcast_raw)),
            "toxcast_processed_rows": int(len(toxcast_df)),
            "merged_rows_before_assignment": int(len(merged_df)),
            "final_rows": int(len(output_df)),
            "overlap_smiles": int(len(set(tox21_df["SMILES"]) & set(toxcast_df["SMILES"]))),
            "final_group_counts": self._group_counts_for_stats(output_df["group"]),
            "label_cols": [
                col for col in output_df.columns if col not in ["SMILES", "group", "ecfp4_cluster"]
            ],
            "include_cluster_col": bool(include_cluster_col),
            "similarity_threshold": float(similarity_threshold),
            "n_bits": int(n_bits),
            "radius": int(radius),
            "output_path": str(output_path),
            "tox21_stats": self.stats.get("build_tox21_base_df"),
            "toxcast_stats": self.stats.get("build_processed_df"),
            "merge_stats": self.stats.get("merge_dataset"),
            "assign_group_stats": self.stats.get("assign_group"),
        }
        return output_df

    def _build_tox21_toxcast_auxiliary_df(
        self,
        tox21_train_path=DEFAULT_TRAIN_PATH,
        tox21_test_path=DEFAULT_TEST_PATH,
        toxcast_path=DEFAULT_TOXCAST_PATH,
    ):
        tox21_df = self.build_tox21_base_df(
            train_path=tox21_train_path,
            test_path=tox21_test_path,
            save=False,
        )

        toxcast_raw = self._read_df(toxcast_path)
        if "smiles" in toxcast_raw.columns and "SMILES" not in toxcast_raw.columns:
            toxcast_raw = toxcast_raw.rename(columns={"smiles": "SMILES"})
        if "SMILES" not in toxcast_raw.columns:
            raise ValueError("ToxCast dataframe must contain a SMILES or smiles column.")
        toxcast_label_cols = [
            col for col in toxcast_raw.columns if col not in ["SMILES", "group"]
        ]
        toxcast_df = self.build_processed_df(
            toxcast_raw,
            label_cols=toxcast_label_cols,
            default_group=pd.NA,
        )
        tox21_toxcast_df = self.merge_dataset(tox21_df, toxcast_df)
        return self.clear_group(tox21_toxcast_df)

    def _get_in_vivo_tox21_toxcast_merge_data(
        self,
        dataset_name,
        in_vivo_path_template,
        output_dir,
        split_ids=(0, 1, 2, 3, 4),
        tox21_train_path=DEFAULT_TRAIN_PATH,
        tox21_test_path=DEFAULT_TEST_PATH,
        toxcast_path=DEFAULT_TOXCAST_PATH,
        random_seed=42,
        valid_ratio_from_training=0.1,
        save_valid_for_split0=True,
        standardize_in_vivo_smiles=False,
        save=True,
    ):
        auxiliary_df = self._build_tox21_toxcast_auxiliary_df(
            tox21_train_path=tox21_train_path,
            tox21_test_path=tox21_test_path,
            toxcast_path=toxcast_path,
        )

        output_dir = Path(output_dir)
        output_paths = {}
        valid_output_path = None
        split_stats = {}

        for split_id in split_ids:
            in_vivo_path = in_vivo_path_template.format(split_id=split_id)
            in_vivo_raw = self._read_df(in_vivo_path)
            if "smiles" in in_vivo_raw.columns and "SMILES" not in in_vivo_raw.columns:
                in_vivo_raw = in_vivo_raw.rename(columns={"smiles": "SMILES"})
            if "SMILES" not in in_vivo_raw.columns:
                raise ValueError(f"{in_vivo_path} must contain a SMILES or smiles column.")
            if "group" not in in_vivo_raw.columns:
                raise ValueError(f"{in_vivo_path} must contain a group column.")

            in_vivo_label_cols = [
                col for col in in_vivo_raw.columns if col not in ["SMILES", "group"]
            ]
            in_vivo_df = self.build_processed_df(
                in_vivo_raw,
                label_cols=in_vivo_label_cols,
                group_col="group",
                standardize_smiles=standardize_in_vivo_smiles,
            )
            merged_df = self.merge_dataset(in_vivo_df, auxiliary_df)
            assigned_df = self.assign_na_group_by_random_split(
                merged_df,
                target_groups=("training", "test"),
                random_seed=random_seed + int(split_id),
            )

            output_path = output_dir / f"data_{split_id}.csv"
            if save:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                assigned_df.to_csv(output_path, index=False)
            output_paths[int(split_id)] = str(output_path) if save else assigned_df

            valid_stats = None
            if save_valid_for_split0 and int(split_id) == 0:
                valid_output_path = output_dir / "contain_valid" / "data_0.csv"
                valid_df = self.valid_from_train_by_random_split(
                    assigned_df,
                    valid_ratio_from_training=valid_ratio_from_training,
                    random_seed=random_seed,
                    output_path=valid_output_path,
                    save=save,
                )
                valid_stats = {
                    "output_path": str(valid_output_path),
                    "group_counts": self._group_counts_for_stats(valid_df["group"]),
                }

            split_stats[int(split_id)] = {
                "input_path": str(in_vivo_path),
                "in_vivo_rows": int(len(in_vivo_df)),
                "merged_rows": int(len(assigned_df)),
                "overlap_smiles_with_auxiliary": int(
                    len(set(in_vivo_df["SMILES"]) & set(auxiliary_df["SMILES"]))
                ),
                "group_counts": self._group_counts_for_stats(assigned_df["group"]),
                "output_path": str(output_path),
                "random_seed_for_na_assignment": int(random_seed + int(split_id)),
                "valid_split0_stats": valid_stats,
            }

        self.stats[f"get_{dataset_name.lower()}_tox21_toxcast_merge_data"] = {
            "dataset_name": dataset_name,
            "split_ids": [int(split_id) for split_id in split_ids],
            "output_dir": str(output_dir),
            "output_paths": output_paths,
            "valid_output_path": str(valid_output_path) if valid_output_path else None,
            "auxiliary_rows": int(len(auxiliary_df)),
            "auxiliary_group_counts": self._group_counts_for_stats(auxiliary_df["group"]),
            "random_seed": int(random_seed),
            "valid_ratio_from_training": float(valid_ratio_from_training),
            "standardize_in_vivo_smiles": bool(standardize_in_vivo_smiles),
            "split_stats": split_stats,
        }
        return output_paths

    def get_toxric_tox21_toxcast_merge_data(
        self,
        split_ids=(0, 1, 2, 3, 4),
        output_dir=DEFAULT_TOXRIC_TOX21_TOXCAST_OUTPUT_DIR,
        in_vivo_path_template=DEFAULT_TOXRIC_SPLIT_TEMPLATE,
        random_seed=42,
        valid_ratio_from_training=0.1,
        save_valid_for_split0=True,
        standardize_in_vivo_smiles=False,
        save=True,
    ):
        return self._get_in_vivo_tox21_toxcast_merge_data(
            dataset_name="TOXRIC",
            in_vivo_path_template=in_vivo_path_template,
            output_dir=output_dir,
            split_ids=split_ids,
            random_seed=random_seed,
            valid_ratio_from_training=valid_ratio_from_training,
            save_valid_for_split0=save_valid_for_split0,
            standardize_in_vivo_smiles=standardize_in_vivo_smiles,
            save=save,
        )

    def get_pubchem_tox21_toxcast_merge_data(
        self,
        split_ids=(0, 1, 2, 3, 4),
        output_dir=DEFAULT_PUBCHEM_TOX21_TOXCAST_OUTPUT_DIR,
        in_vivo_path_template=DEFAULT_PUBCHEM_SPLIT_TEMPLATE,
        random_seed=42,
        valid_ratio_from_training=0.1,
        save_valid_for_split0=True,
        standardize_in_vivo_smiles=False,
        save=True,
    ):
        return self._get_in_vivo_tox21_toxcast_merge_data(
            dataset_name="Pubchem",
            in_vivo_path_template=in_vivo_path_template,
            output_dir=output_dir,
            split_ids=split_ids,
            random_seed=random_seed,
            valid_ratio_from_training=valid_ratio_from_training,
            save_valid_for_split0=save_valid_for_split0,
            standardize_in_vivo_smiles=standardize_in_vivo_smiles,
            save=save,
        )

    def valid_from_train_by_ecfp4_similarity(
        self,
        df,
        train_ratio=0.9,
        random_seed=42,
        output_path=DEFAULT_TOX21_TOXCAST_ECFP4_SPLIT_PATH,
        similarity_threshold=0.7,
        n_bits=1024,
        radius=2,
        include_cluster_col=False,
        save=True,
    ):
        """
        Split valid rows from training by ECFP4 cluster.

        Input df must contain training and test groups. Test rows are preserved.
        Training rows are moved to valid by whole ECFP4 cluster, so the same
        cluster cannot appear in both final training and valid groups.
        """
        if not 0 < train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1.")

        data = self._read_df(df)
        missing = [col for col in ["SMILES", "group"] if col not in data.columns]
        if missing:
            raise ValueError(f"Input dataframe is missing required columns: {missing}")

        allowed_groups = {"training", "test"}
        actual_groups = set(data["group"].dropna().astype(str))
        unexpected_groups = actual_groups - allowed_groups
        if unexpected_groups:
            raise ValueError(
                "Input dataframe group column must contain only 'training' and 'test'. "
                f"Unexpected groups: {sorted(unexpected_groups)}"
            )
        if data["group"].isna().any():
            raise ValueError("Input dataframe group column must not contain empty values.")

        original_group = data["group"].copy()
        cluster_ids, cluster_stats = self.get_ecfp4_similarity_clusters(
            data["SMILES"],
            similarity_threshold=similarity_threshold,
            n_bits=n_bits,
            radius=radius,
            return_stats=True,
        )
        data = data.copy()
        data["ecfp4_cluster"] = cluster_ids

        training_mask = data["group"] == "training"
        training_cluster_sizes = data.loc[training_mask, "ecfp4_cluster"].value_counts()
        target_valid_size = int(round(int(training_mask.sum()) * (1.0 - train_ratio)))

        rng = np.random.default_rng(random_seed)
        cluster_order = training_cluster_sizes.index.to_numpy(copy=True)
        rng.shuffle(cluster_order)

        valid_clusters = []
        valid_size = 0
        for cluster_id in cluster_order:
            cluster_size = int(training_cluster_sizes.loc[cluster_id])
            if valid_size >= target_valid_size:
                break
            if valid_size == 0:
                valid_clusters.append(cluster_id)
                valid_size += cluster_size
                continue

            current_gap = abs(target_valid_size - valid_size)
            new_gap = abs(target_valid_size - (valid_size + cluster_size))
            if valid_size < target_valid_size or new_gap < current_gap:
                valid_clusters.append(cluster_id)
                valid_size += cluster_size

        valid_cluster_set = set(valid_clusters)
        data.loc[
            training_mask & data["ecfp4_cluster"].isin(valid_cluster_set),
            "group",
        ] = "valid"

        final_training_clusters = set(data.loc[data["group"] == "training", "ecfp4_cluster"])
        final_valid_clusters = set(data.loc[data["group"] == "valid", "ecfp4_cluster"])
        training_valid_overlap = final_training_clusters & final_valid_clusters
        if training_valid_overlap:
            raise RuntimeError(
                "ECFP4 cluster split failed: clusters appear in both training and valid."
            )

        output_df = data.copy()
        if not include_cluster_col:
            output_df = output_df.drop(columns=["ecfp4_cluster"])

        if save:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_df.to_csv(output_path, index=False)

        self.stats["valid_from_train_by_ecfp4_similarity"] = {
            "input_rows": int(len(data)),
            "original_group_counts": self._group_counts_for_stats(original_group),
            "final_group_counts": self._group_counts_for_stats(data["group"]),
            "train_ratio": float(train_ratio),
            "target_valid_size": int(target_valid_size),
            "actual_valid_size": int((data["group"] == "valid").sum()),
            "random_seed": int(random_seed),
            "valid_cluster_count": int(len(valid_cluster_set)),
            "training_valid_cluster_overlap": int(len(training_valid_overlap)),
            "cluster_stats": cluster_stats,
            "similarity_threshold": float(similarity_threshold),
            "n_bits": int(n_bits),
            "radius": int(radius),
            "include_cluster_col": bool(include_cluster_col),
            "output_path": str(output_path),
        }
        return output_df

    def training_cluster_cv_by_ecfp4_similarity(
        self,
        df,
        n_folds=10,
        random_seed=42,
        output_dir=DEFAULT_TRAINING_CLUSTER_CV_DIR,
        similarity_threshold=0.7,
        n_bits=1024,
        radius=2,
        include_cluster_col=False,
        save=True,
    ):
        """
        Generate N-fold CV files from training rows only by ECFP4 cluster.

        Only rows with group == "training" are kept. ECFP4 clusters are assigned
        to folds, and every cluster belongs to exactly one fold. For split i,
        fold i is test and fold (i + 1) % n_folds is valid. A no-valid version is
        also saved for each split, where valid rows are converted back to
        training while test rows stay test.
        """
        if int(n_folds) < 2:
            raise ValueError("n_folds must be at least 2.")
        n_folds = int(n_folds)

        data = self._read_df(df)
        missing = [col for col in ["SMILES", "group"] if col not in data.columns]
        if missing:
            raise ValueError(f"Input dataframe is missing required columns: {missing}")

        training_data = data[data["group"] == "training"].copy().reset_index(drop=True)
        if training_data.empty:
            raise ValueError("Input dataframe has no rows with group == 'training'.")

        cluster_ids, cluster_stats = self.get_ecfp4_similarity_clusters(
            training_data["SMILES"],
            similarity_threshold=similarity_threshold,
            n_bits=n_bits,
            radius=radius,
            return_stats=True,
        )
        training_data["ecfp4_cluster"] = cluster_ids

        cluster_sizes = training_data["ecfp4_cluster"].value_counts()
        if len(cluster_sizes) < n_folds:
            raise ValueError(
                f"n_folds={n_folds} is larger than the number of ECFP4 clusters "
                f"({len(cluster_sizes)})."
            )

        rng = np.random.default_rng(random_seed)
        cluster_table = pd.DataFrame(
            {
                "cluster": cluster_sizes.index.to_numpy(),
                "size": cluster_sizes.to_numpy(dtype=int),
                "tie_breaker": rng.random(len(cluster_sizes)),
            }
        ).sort_values(["size", "tie_breaker"], ascending=[False, True])

        fold_sizes = np.zeros(n_folds, dtype=int)
        cluster_to_fold = {}
        for _, row in cluster_table.iterrows():
            fold_id = int(np.argmin(fold_sizes))
            cluster_to_fold[row["cluster"]] = fold_id
            fold_sizes[fold_id] += int(row["size"])

        training_data["cv_fold"] = training_data["ecfp4_cluster"].map(cluster_to_fold).astype(int)

        output_dir = Path(output_dir)
        if save:
            output_dir.mkdir(parents=True, exist_ok=True)

        with_valid_outputs = {}
        no_valid_outputs = {}
        split_stats = {}
        base_output_cols = list(training_data.columns)
        if not include_cluster_col:
            base_output_cols = [
                col for col in base_output_cols if col not in ["ecfp4_cluster", "cv_fold"]
            ]

        for split_idx in range(n_folds):
            valid_fold = (split_idx + 1) % n_folds

            split_df = training_data.copy()
            split_df["group"] = "training"
            split_df.loc[split_df["cv_fold"] == split_idx, "group"] = "test"
            split_df.loc[split_df["cv_fold"] == valid_fold, "group"] = "valid"

            no_valid_df = split_df.copy()
            no_valid_df.loc[no_valid_df["group"] == "valid", "group"] = "training"

            for candidate_df, candidate_name in [
                (split_df, "with_valid"),
                (no_valid_df, "no_valid"),
            ]:
                cluster_group_counts = (
                    candidate_df.groupby(["ecfp4_cluster", "group"])
                    .size()
                    .unstack(fill_value=0)
                )
                nonzero_group_counts = (cluster_group_counts > 0).sum(axis=1)
                if int((nonzero_group_counts > 1).sum()) != 0:
                    raise RuntimeError(
                        f"{candidate_name} split {split_idx} has clusters crossing groups."
                    )

            split_output_df = split_df[base_output_cols].copy()
            no_valid_output_df = no_valid_df[base_output_cols].copy()

            with_valid_path = output_dir / f"training_cluster_CV_{split_idx}.csv"
            no_valid_path = output_dir / f"training_cluster_CV_no_valid{split_idx}.csv"

            if save:
                split_output_df.to_csv(with_valid_path, index=False)
                no_valid_output_df.to_csv(no_valid_path, index=False)
                with_valid_outputs[split_idx] = str(with_valid_path)
                no_valid_outputs[split_idx] = str(no_valid_path)
            else:
                with_valid_outputs[split_idx] = split_output_df
                no_valid_outputs[split_idx] = no_valid_output_df

            split_stats[split_idx] = {
                "test_fold": int(split_idx),
                "valid_fold": int(valid_fold),
                "with_valid_group_counts": self._group_counts_for_stats(split_df["group"]),
                "no_valid_group_counts": self._group_counts_for_stats(no_valid_df["group"]),
            }

        self.stats["training_cluster_cv_by_ecfp4_similarity"] = {
            "input_rows": int(len(data)),
            "training_rows": int(len(training_data)),
            "n_folds": int(n_folds),
            "random_seed": int(random_seed),
            "fold_sizes": {int(idx): int(size) for idx, size in enumerate(fold_sizes)},
            "cluster_count": int(len(cluster_sizes)),
            "largest_cluster_size": int(cluster_sizes.max()),
            "cluster_stats": cluster_stats,
            "similarity_threshold": float(similarity_threshold),
            "n_bits": int(n_bits),
            "radius": int(radius),
            "include_cluster_col": bool(include_cluster_col),
            "output_dir": str(output_dir),
            "with_valid_outputs": with_valid_outputs,
            "no_valid_outputs": no_valid_outputs,
            "split_stats": split_stats,
        }

        return {
            "with_valid": with_valid_outputs,
            "no_valid": no_valid_outputs,
        }


# Compatibility alias for the old class-name style.
Tox21Datasetprocessor = Tox21DatasetProcessor
