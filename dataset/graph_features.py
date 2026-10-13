import argparse
import json
from pathlib import Path

import dgl
import numpy as np
import pandas as pd
import torch
from dgl.data.graph_serialize import save_graphs
from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold


NON_LABEL_COLUMNS = {"SMILES", "group"}

ATOM_DIM = 8
BOND_DIM = 2

ALLOWABLE_FEATURES = {
    "possible_chirality_list": [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ],
    "possible_hybridization_list": [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED,
    ],
    "possible_bonds": [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    "possible_bond_dirs": [
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT,
    ],
}


def _mirror_graph_output_dir(input_csv, output_root="data/graph_data"):
    input_csv = Path(input_csv)
    parts = list(input_csv.parent.parts)
    if "raw_data" in parts:
        parts[parts.index("raw_data")] = Path(output_root).name
        return Path(*parts)
    return Path(output_root) / input_csv.parent.name


def infer_task_names(data):
    return [col for col in data.columns if col not in NON_LABEL_COLUMNS]


def _safe_index(values, item, default_index=0):
    try:
        return values.index(item)
    except ValueError:
        return default_index


def atom_features(atom):
    atomic_num = atom.GetAtomicNum()
    degree = atom.GetDegree()
    chiral_idx = _safe_index(ALLOWABLE_FEATURES["possible_chirality_list"], atom.GetChiralTag())
    formal_charge = atom.GetFormalCharge()
    hyb_idx = _safe_index(ALLOWABLE_FEATURES["possible_hybridization_list"], atom.GetHybridization())
    num_h = atom.GetTotalNumHs()
    implicit_valence = atom.GetImplicitValence()
    covalent_radius = Chem.GetPeriodicTable().GetRcovalent(atomic_num)

    return np.asarray(
        [
            atomic_num / 118.0,
            min(degree, 10) / 10.0,
            chiral_idx / 3.0,
            (np.clip(formal_charge, -5, 5) + 5) / 10.0,
            hyb_idx / 6.0,
            min(num_h, 8) / 8.0,
            min(max(implicit_valence, 0), 6) / 6.0,
            covalent_radius / 2.5,
        ],
        dtype=np.float32,
    )


def bond_features(bond):
    bond_type_idx = _safe_index(ALLOWABLE_FEATURES["possible_bonds"], bond.GetBondType())
    bond_dir_idx = _safe_index(ALLOWABLE_FEATURES["possible_bond_dirs"], bond.GetBondDir())
    return np.asarray([bond_type_idx / 3.0, bond_dir_idx / 2.0], dtype=np.float32)


def mol_to_dgl_graph(mol, atom_mask_indices=None):
    atom_mask_indices = set(atom_mask_indices or [])
    num_atoms = mol.GetNumAtoms()
    graph = dgl.graph(([], []), num_nodes=num_atoms)

    atom_array = np.vstack([atom_features(atom) for atom in mol.GetAtoms()]).astype(np.float32)
    atom_mask = np.asarray([0.0 if idx in atom_mask_indices else 1.0 for idx in range(num_atoms)], dtype=np.float32)

    src = []
    dst = []
    edge_array = []
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        feat = bond_features(bond)
        src.extend([u, v])
        dst.extend([v, u])
        edge_array.extend([feat, feat])

    if src:
        graph.add_edges(src, dst)
        edge_tensor = torch.tensor(np.vstack(edge_array), dtype=torch.float32)
    else:
        edge_tensor = torch.zeros((0, BOND_DIM), dtype=torch.float32)

    atom_tensor = torch.tensor(atom_array, dtype=torch.float32)
    mask_tensor = torch.tensor(atom_mask, dtype=torch.float32)

    graph.ndata["atom_feat"] = atom_tensor
    graph.ndata["atom_mask"] = mask_tensor
    graph.edata["bond_feat"] = edge_tensor

    # Legacy aliases let archived MaskGNN-style code and the new models read the same graph.
    graph.ndata["node"] = atom_tensor
    graph.ndata["smask"] = mask_tensor
    graph.edata["edge"] = edge_tensor
    return graph


def get_brics_fragments(mol):
    break_bonds = set()
    for (a1, a2), _ in BRICS.FindBRICSBonds(mol):
        break_bonds.add(tuple(sorted((int(a1), int(a2)))))

    adjacency = [[] for _ in range(mol.GetNumAtoms())]
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        if tuple(sorted((u, v))) not in break_bonds:
            adjacency[u].append(v)
            adjacency[v].append(u)

    visited = [False] * mol.GetNumAtoms()
    fragments = []
    for start in range(mol.GetNumAtoms()):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        component = []
        while stack:
            atom_idx = stack.pop()
            component.append(atom_idx)
            for neighbor in adjacency[atom_idx]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        fragments.append(sorted(component))
    return fragments


def get_murcko_atom_indices(mol):
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return []
    match = mol.GetSubstructMatch(scaffold)
    if match:
        return sorted(match)
    matches = mol.GetSubstructMatches(scaffold)
    return sorted(matches[0]) if matches else []


FG_WITHOUT_CA_SMARTS = [
    "[N;D2]-[C;D3](=O)-[C;D1;H3]",
    "C(=O)[O;D1]",
    "C(=O)[O;D2]-[C;D1;H3]",
    "C(=O)-[H]",
    "C(=O)-[N;D1]",
    "C(=O)-[C;D1;H3]",
    "[N;D2]=[C;D2]=[O;D1]",
    "[N;D2]=[C;D2]=[S;D1]",
    "[N;D3](=[O;D1])[O;D1]",
    "[N;R0]=[O;D1]",
    "[N;R0]-[O;D1]",
    "[N;R0]-[C;D1;H3]",
    "[N;R0]=[C;D1;H2]",
    "[N;D2]=[N;D2]-[C;D1;H3]",
    "[N;D2]=[N;D1]",
    "[N;D2]#[N;D1]",
    "[C;D2]#[N;D1]",
    "[S;D4](=[O;D1])(=[O;D1])-[N;D1]",
    "[N;D2]-[S;D4](=[O;D1])(=[O;D1])-[C;D1;H3]",
    "[S;D4](=O)(=O)-[O;D1]",
    "[S;D4](=O)(=O)-[O;D2]-[C;D1;H3]",
    "[S;D4](=O)(=O)-[C;D1;H3]",
    "[S;D4](=O)(=O)-[Cl]",
    "[S;D3](=O)-[C;D1]",
    "[S;D2]-[C;D1;H3]",
    "[S;D1]",
    "[S;D1]",
    "[#9,#17,#35,#53]",
    "[C;D4]([C;D1])([C;D1])-[C;D1]",
    "[C;D4](F)(F)F",
    "[C;D2]#[C;D1;H]",
    "[C;D3]1-[C;D2]-[C;D2]1",
    "[O;D2]-[C;D2]-[C;D1;H3]",
    "[O;D2]-[C;D1;H3]",
    "[O;D1]",
    "[O;D1]",
    "[N;D1]",
    "[N;D1]",
    "[N;D1]",
]


def _load_functional_group_catalog():
    names = [f"fg_{i}" for i in range(len(FG_WITHOUT_CA_SMARTS))]
    patterns = [Chem.MolFromSmarts(smarts) for smarts in FG_WITHOUT_CA_SMARTS]
    return names, patterns


FG_NAMES, FG_WITHOUT_CA_PATTERNS = _load_functional_group_catalog()


def _deduplicate_atom_sets(atom_sets):
    normalized = []
    seen = set()
    for atom_set in sorted((tuple(sorted(set(item))) for item in atom_sets), key=lambda item: (-len(item), item)):
        if not atom_set or atom_set in seen:
            continue
        if any(set(atom_set).issubset(existing) for existing in seen):
            continue
        seen.add(atom_set)
        normalized.append(list(atom_set))
    return normalized


def get_functional_group_matches(mol):
    atom_sets = []
    names = []
    for name, pattern in zip(FG_NAMES, FG_WITHOUT_CA_PATTERNS):
        if pattern is None:
            continue
        try:
            matches = mol.GetSubstructMatches(pattern)
        except Exception:
            continue
        for match in matches:
            atom_sets.append(list(match))
            names.append(name)

    deduped = _deduplicate_atom_sets(atom_sets)
    deduped_names = []
    for atoms in deduped:
        atom_tuple = tuple(atoms)
        name = "functional_group"
        for raw_atoms, raw_name in zip(atom_sets, names):
            if tuple(sorted(raw_atoms)) == atom_tuple:
                name = raw_name
                break
        deduped_names.append(name)
    return deduped, deduped_names


def _subgraphs_from_atom_sets(graph, atom_sets):
    subgraphs = []
    for atom_set in atom_sets:
        if not atom_set:
            continue
        subgraphs.append(graph.subgraph(atom_set))
    return subgraphs


def _save_graph_bin(path, graphs, labels):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not graphs:
        marker = path.with_suffix(path.suffix + ".empty.json")
        marker.write_text(
            json.dumps({"graph_file": path.name, "num_graphs": 0, "reason": "no graphs for this view"}, indent=2),
            encoding="utf-8",
        )
        return False
    save_graphs(str(path), graphs, labels)
    return True


def _to_object_array(items):
    arr = np.empty(len(items), dtype=object)
    arr[:] = items
    return arr


def build_and_save_graph_features(input_csv, output_dir=None, include_subgraphs=False):
    input_csv = Path(input_csv)
    data = pd.read_csv(input_csv)
    if "SMILES" not in data.columns:
        raise ValueError(f"{input_csv} must contain a SMILES column.")
    if "group" not in data.columns:
        raise ValueError(f"{input_csv} must contain a group column.")

    output_dir = Path(output_dir) if output_dir is not None else _mirror_graph_output_dir(input_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_names = infer_task_names(data)
    labels = data[task_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    groups = data["group"].astype(str).to_numpy()
    smiles_list = data["SMILES"].astype(str).to_numpy()

    include_subgraphs = bool(include_subgraphs)
    full_graphs = []
    brics_graphs = []
    murcko_graphs = []
    fg_graphs = []
    metadata_rows = []
    failed_records = []
    substructure_masks = {
        "brics_atom_indices": [],
        "murcko_atom_indices": [],
        "fg_atom_indices": [],
        "fg_names": [],
    }
    graph_labels = []
    graph_row_indices = []

    brics_parent_graph_index = []
    brics_parent_row_index = []
    brics_local_index = []
    murcko_parent_graph_index = []
    murcko_parent_row_index = []
    murcko_local_index = []
    fg_parent_graph_index = []
    fg_parent_row_index = []
    fg_local_index = []

    for row_index, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            failed_records.append({"row_index": row_index, "SMILES": smiles, "reason": "rdkit_mol_failed"})
            continue

        try:
            graph = mol_to_dgl_graph(mol)
            graph_index = len(full_graphs)
            full_graphs.append(graph)
            graph_labels.append(labels[row_index])
            graph_row_indices.append(row_index)

            brics_atom_sets = []
            murcko_atom_set = []
            fg_atom_sets = []
            fg_names = []
            brics_subgraphs = []
            murcko_subgraphs = []
            fg_subgraphs = []

            if include_subgraphs:
                brics_atom_sets = get_brics_fragments(mol)
                murcko_atom_set = get_murcko_atom_indices(mol)
                fg_atom_sets, fg_names = get_functional_group_matches(mol)

                brics_subgraphs = _subgraphs_from_atom_sets(graph, brics_atom_sets)
                murcko_subgraphs = _subgraphs_from_atom_sets(graph, [murcko_atom_set] if murcko_atom_set else [])
                fg_subgraphs = _subgraphs_from_atom_sets(graph, fg_atom_sets)

                for local_index, subgraph in enumerate(brics_subgraphs):
                    brics_graphs.append(subgraph)
                    brics_parent_graph_index.append(graph_index)
                    brics_parent_row_index.append(row_index)
                    brics_local_index.append(local_index)
                for local_index, subgraph in enumerate(murcko_subgraphs):
                    murcko_graphs.append(subgraph)
                    murcko_parent_graph_index.append(graph_index)
                    murcko_parent_row_index.append(row_index)
                    murcko_local_index.append(local_index)
                for local_index, subgraph in enumerate(fg_subgraphs):
                    fg_graphs.append(subgraph)
                    fg_parent_graph_index.append(graph_index)
                    fg_parent_row_index.append(row_index)
                    fg_local_index.append(local_index)

                substructure_masks["brics_atom_indices"].append(brics_atom_sets)
                substructure_masks["murcko_atom_indices"].append(murcko_atom_set)
                substructure_masks["fg_atom_indices"].append(fg_atom_sets)
                substructure_masks["fg_names"].append(fg_names)

            metadata_rows.append(
                {
                    "row_index": row_index,
                    "graph_index": graph_index,
                    "SMILES": smiles,
                    "group": groups[row_index],
                    "num_atoms": int(mol.GetNumAtoms()),
                    "num_bonds": int(mol.GetNumBonds()),
                    "brics_count": len(brics_subgraphs),
                    "murcko_count": len(murcko_subgraphs),
                    "fg_count": len(fg_subgraphs),
                }
            )
        except Exception as exc:
            failed_records.append(
                {
                    "row_index": row_index,
                    "SMILES": smiles,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )

    if graph_labels:
        y = torch.tensor(np.vstack(graph_labels), dtype=torch.float32)
    else:
        y = torch.zeros((0, len(task_names)), dtype=torch.float32)

    metadata = pd.DataFrame(metadata_rows)
    failed = pd.DataFrame(failed_records, columns=["row_index", "SMILES", "reason"])

    saved_full = _save_graph_bin(
        output_dir / "graphs.bin",
        full_graphs,
        {
            "y": y,
            "row_index": torch.tensor(graph_row_indices, dtype=torch.long),
        },
    )
    saved_brics = False
    saved_murcko = False
    saved_fg = False
    if include_subgraphs:
        saved_brics = _save_graph_bin(
            output_dir / "brics_graphs.bin",
            brics_graphs,
            {
                "parent_graph_index": torch.tensor(brics_parent_graph_index, dtype=torch.long),
                "parent_row_index": torch.tensor(brics_parent_row_index, dtype=torch.long),
                "local_index": torch.tensor(brics_local_index, dtype=torch.long),
            },
        )
        saved_murcko = _save_graph_bin(
            output_dir / "murcko_graphs.bin",
            murcko_graphs,
            {
                "parent_graph_index": torch.tensor(murcko_parent_graph_index, dtype=torch.long),
                "parent_row_index": torch.tensor(murcko_parent_row_index, dtype=torch.long),
                "local_index": torch.tensor(murcko_local_index, dtype=torch.long),
            },
        )
        saved_fg = _save_graph_bin(
            output_dir / "fg_graphs.bin",
            fg_graphs,
            {
                "parent_graph_index": torch.tensor(fg_parent_graph_index, dtype=torch.long),
                "parent_row_index": torch.tensor(fg_parent_row_index, dtype=torch.long),
                "local_index": torch.tensor(fg_local_index, dtype=torch.long),
            },
        )

    metadata.to_csv(output_dir / "metadata.csv", index=False)
    failed.to_csv(output_dir / "failed_records.csv", index=False)
    (output_dir / "task_names.json").write_text(json.dumps(task_names, indent=2), encoding="utf-8")
    if include_subgraphs:
        torch.save(substructure_masks, output_dir / "substructure_masks.pt")

    schema = {
        "source_csv": str(input_csv),
        "num_input_rows": int(len(data)),
        "num_graphs": int(len(full_graphs)),
        "num_failed": int(len(failed_records)),
        "atom_dim": ATOM_DIM,
        "bond_dim": BOND_DIM,
        "node_feature_keys": ["atom_feat", "node"],
        "edge_feature_keys": ["bond_feat", "edge"],
        "atom_mask_keys": ["atom_mask", "smask"],
        "label_key": "y",
        "task_names_file": "task_names.json",
        "metadata_file": "metadata.csv",
        "failed_records_file": "failed_records.csv",
        "full_graph_file": "graphs.bin" if saved_full else None,
        "subgraph_files": {
            "brics": "brics_graphs.bin" if saved_brics else None,
            "murcko": "murcko_graphs.bin" if saved_murcko else None,
            "functional_group": "fg_graphs.bin" if saved_fg else None,
        },
        "substructure_masks_file": "substructure_masks.pt" if include_subgraphs else None,
        "subgraph_alignment": "Subgraph bins store parent_graph_index, parent_row_index, and local_index labels." if include_subgraphs else None,
    }
    (output_dir / "graph_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    if include_subgraphs:
        sub_schema = {
            "brics_atom_indices": "List aligned to metadata.csv graph rows; each item is a list of atom-index lists.",
            "murcko_atom_indices": "List aligned to metadata.csv graph rows; each item is one atom-index list or empty.",
            "fg_atom_indices": "List aligned to metadata.csv graph rows; each item is a list of atom-index lists.",
            "fg_names": "List aligned to metadata.csv graph rows; names for functional-group atom sets.",
        }
        (output_dir / "substructure_schema.json").write_text(json.dumps(sub_schema, indent=2), encoding="utf-8")

    print(f"Input CSV: {input_csv}")
    print(f"Output dir: {output_dir}")
    print(f"Saved full graphs: {len(full_graphs)}")
    print(f"Saved BRICS subgraphs: {len(brics_graphs)}")
    print(f"Saved Murcko subgraphs: {len(murcko_graphs)}")
    print(f"Saved functional-group subgraphs: {len(fg_graphs)}")
    print(f"Failed rows: {len(failed_records)}")
    return {
        "output_dir": output_dir,
        "num_graphs": len(full_graphs),
        "num_failed": len(failed_records),
        "num_brics_subgraphs": len(brics_graphs),
        "num_murcko_subgraphs": len(murcko_graphs),
        "num_fg_subgraphs": len(fg_graphs),
    }


def iter_input_csvs(input_root, pattern="*.csv"):
    input_root = Path(input_root)
    yield from sorted(input_root.rglob(pattern))


def parse_args():
    parser = argparse.ArgumentParser(description="Build DGL graph features from raw CSV files.")
    parser.add_argument("--input_csv", default=None, help="Raw CSV containing SMILES, group, and task label columns.")
    parser.add_argument("--input_root", default=None, help="Build all CSV files under this root.")
    parser.add_argument("--pattern", default="*.csv", help="Glob pattern used with --input_root.")
    parser.add_argument("--output_dir", default=None, help="Optional output directory for a single --input_csv run.")
    parser.add_argument("--dry_run", action="store_true", help="List target CSV files without writing graph features.")
    parser.add_argument("--include_subgraphs", action="store_true", help="Also build BRICS, Murcko, and functional-group subgraphs. Disabled by default for eGIN.")
    return parser.parse_args()


def main():
    args = parse_args()
    if bool(args.input_csv) == bool(args.input_root):
        raise ValueError("Provide exactly one of --input_csv or --input_root.")

    if args.input_csv:
        if args.dry_run:
            print(Path(args.input_csv))
            return
        build_and_save_graph_features(args.input_csv, output_dir=args.output_dir, include_subgraphs=args.include_subgraphs)
        return

    csv_paths = list(iter_input_csvs(args.input_root, pattern=args.pattern))
    if args.dry_run:
        for path in csv_paths:
            print(path)
        print(f"Matched CSV files: {len(csv_paths)}")
        return
    for path in csv_paths:
        build_and_save_graph_features(path, include_subgraphs=args.include_subgraphs)


if __name__ == "__main__":
    main()
