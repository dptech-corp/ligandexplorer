"""
Graph construction from PDB files for GNN training.

- Ignores hydrogen atoms
- Node feature: atomic number (int, for nn.Embedding)
- Edge: kNN ∩ radius hybrid (each atom connects to at most k nearest
  neighbours within a distance cutoff, giving uniform node degree while
  respecting a physical interaction range)
- Edge attr: interatomic distance
- Residue boundary: edge feature (0=same residue, 1=cross-residue)
"""
import os
import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data

ELEMENT_TO_Z = {
    'H': 1, 'HE': 2, 'LI': 3, 'BE': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
    'F': 9, 'NE': 10, 'NA': 11, 'MG': 12, 'AL': 13, 'SI': 14, 'P': 15,
    'S': 16, 'CL': 17, 'AR': 18, 'K': 19, 'CA': 20, 'SC': 21, 'TI': 22,
    'V': 23, 'CR': 24, 'MN': 25, 'FE': 26, 'CO': 27, 'NI': 28, 'CU': 29,
    'ZN': 30, 'GA': 31, 'GE': 32, 'AS': 33, 'SE': 34, 'BR': 35, 'KR': 36,
    'RB': 37, 'SR': 38, 'Y': 39, 'ZR': 40, 'NB': 41, 'MO': 42, 'TC': 43,
    'RU': 44, 'RH': 45, 'PD': 46, 'AG': 47, 'CD': 48, 'IN': 49, 'SN': 50,
    'SB': 51, 'TE': 52, 'I': 53, 'XE': 54, 'CS': 55, 'BA': 56, 'LA': 57,
    'CE': 58, 'PR': 59, 'ND': 60, 'PM': 61, 'SM': 62, 'EU': 63, 'GD': 64,
    'TB': 65, 'DY': 66, 'HO': 67, 'ER': 68, 'TM': 69, 'YB': 70, 'LU': 71,
    'HF': 72, 'TA': 73, 'W': 74, 'RE': 75, 'OS': 76, 'IR': 77, 'PT': 78,
    'AU': 79, 'HG': 80, 'TL': 81, 'PB': 82, 'BI': 83, 'PO': 84, 'AT': 85,
    'RN': 86, 'FR': 87, 'RA': 88, 'AC': 89, 'TH': 90, 'PA': 91, 'U': 92,
}


def parse_pdb(pdb_path):
    """Parse PDB, return list of (atomic_number, x, y, z, residue_idx). Skip H."""
    atoms = []
    resid_map = {}
    resid_counter = 0

    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue

            elem = line[76:78].strip().upper() if len(line) >= 78 else ""
            if not elem:
                raw = line[12:16].strip()
                elem = raw.lstrip("0123456789").rstrip("0123456789'\"").upper()
                if not elem:
                    continue

            if elem == "H" or elem.startswith("H") and len(elem) <= 2 and elem not in ELEMENT_TO_Z:
                continue
            if elem == "H":
                continue

            z = ELEMENT_TO_Z.get(elem, 0)
            if z == 0:
                continue

            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z_coord = float(line[46:54])
            except (ValueError, IndexError):
                continue

            chain = line[21] if len(line) > 21 else " "
            resname = line[17:20].strip() if len(line) > 20 else ""
            resseq = line[22:26].strip() if len(line) > 26 else "0"
            icode = line[26] if len(line) > 26 else " "
            res_key = (chain, resname, resseq, icode)

            if res_key not in resid_map:
                resid_map[res_key] = resid_counter
                resid_counter += 1

            atoms.append((z, x, y, z_coord, resid_map[res_key]))

    return atoms


def _build_knn_radius_edges(pos_np, res_np, cutoff, max_neighbors):
    """Build symmetric edges using kNN ∩ radius-graph hybrid.

    For each atom, connect to at most *max_neighbors* nearest neighbours
    whose distance is < *cutoff*.  Edges are made bidirectional and
    deduplicated.

    Returns (edge_index, edge_attr, res_boundary) as numpy arrays,
    or (None, None, None) when no valid edges exist.
    """
    n = len(pos_np)
    if n < 2:
        return None, None, None

    tree = cKDTree(pos_np)
    k_query = min(max_neighbors + 1, n)  # +1 because query includes self
    dists_knn, idx_knn = tree.query(pos_np, k=k_query,
                                     distance_upper_bound=cutoff)

    # Vectorised flatten — skip column-0 (self)
    src = np.repeat(np.arange(n), k_query - 1)
    dst = idx_knn[:, 1:].ravel()
    d = dists_knn[:, 1:].ravel()

    valid = (dst < n) & np.isfinite(d)
    src, dst, d = src[valid], dst[valid], d[valid].astype(np.float32)

    if len(src) == 0:
        return None, None, None

    # Make symmetric (union of i→j and j→i), then deduplicate
    all_src = np.concatenate([src, dst])
    all_dst = np.concatenate([dst, src])
    all_d = np.concatenate([d, d])

    edge_id = all_src.astype(np.int64) * n + all_dst.astype(np.int64)
    _, uniq = np.unique(edge_id, return_index=True)

    row, col = all_src[uniq], all_dst[uniq]
    dists = all_d[uniq]

    edge_index = np.stack([row, col]).astype(np.int64)
    boundary = (res_np[row] != res_np[col]).astype(np.int64)

    return edge_index, dists, boundary


def structure_to_graph(pdb_path, cutoff=5.0, max_neighbors=32):
    """Convert a PDB file to a PyG Data object.

    Uses kNN ∩ radius-graph: each atom connects to at most
    *max_neighbors* nearest neighbours within *cutoff* Å.

    Returns Data with:
        z: (N,) int - atomic numbers
        pos: (N, 3) float - coordinates
        edge_index: (2, E) long
        edge_attr: (E,) float - distances
        residue_boundary: (E,) long - 0=same residue, 1=cross-residue
        residue_idx: (N,) long - residue index per atom
    """
    atoms = parse_pdb(pdb_path)
    if len(atoms) == 0:
        return None

    z_list, coords, res_idx = [], [], []
    for atomic_num, x, y, zc, ri in atoms:
        z_list.append(atomic_num)
        coords.append([x, y, zc])
        res_idx.append(ri)

    z_tensor = torch.tensor(z_list, dtype=torch.long)
    pos = torch.tensor(coords, dtype=torch.float)
    res_tensor = torch.tensor(res_idx, dtype=torch.long)

    if not torch.isfinite(pos).all():
        return None

    n = len(atoms)
    if n == 1:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, dtype=torch.float)
        res_boundary = torch.zeros(0, dtype=torch.long)
    else:
        ei, ea, rb = _build_knn_radius_edges(
            pos.numpy(), res_tensor.numpy(), cutoff, max_neighbors,
        )
        if ei is None:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_attr = torch.zeros(0, dtype=torch.float)
            res_boundary = torch.zeros(0, dtype=torch.long)
        else:
            edge_index = torch.from_numpy(ei)
            edge_attr = torch.from_numpy(ea)
            res_boundary = torch.from_numpy(rb)

    data = Data(
        z=z_tensor,
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        residue_boundary=res_boundary,
        residue_idx=res_tensor,
    )
    return data


def complex_to_graph(protein_path, ligand_path, pocket_cutoff=6.0,
                     edge_cutoff=5.0, max_pocket_atoms=800,
                     max_neighbors=32):
    """Build a protein-ligand complex graph for Task B.

    Extracts pocket atoms (within pocket_cutoff of any ligand atom),
    then builds a joint graph using kNN ∩ radius-graph edges.

    Returns Data with additional:
        node_type: (N,) long - 0=protein, 1=ligand
        ligand_mask: (N,) bool
    """
    prot_atoms = parse_pdb(protein_path)
    lig_atoms = parse_pdb(ligand_path)
    if not prot_atoms or not lig_atoms:
        return None

    lig_coords = np.array([[a[1], a[2], a[3]] for a in lig_atoms])
    prot_coords = np.array([[a[1], a[2], a[3]] for a in prot_atoms])

    lig_tree = cKDTree(lig_coords)
    dists_to_lig, _ = lig_tree.query(prot_coords, k=1)
    pocket_mask = dists_to_lig < pocket_cutoff
    pocket_indices = np.where(pocket_mask)[0]
    pocket_dists = dists_to_lig[pocket_indices]

    if len(pocket_indices) > max_pocket_atoms:
        closest = np.argsort(pocket_dists)[:max_pocket_atoms]
        pocket_indices = pocket_indices[closest]

    pocket_atoms = [prot_atoms[i] for i in pocket_indices]

    max_prot_res = max(a[4] for a in pocket_atoms) + 1 if pocket_atoms else 0

    all_atoms = []
    node_types = []
    for atomic_num, x, y, zc, ri in pocket_atoms:
        all_atoms.append((atomic_num, x, y, zc, ri))
        node_types.append(0)
    for atomic_num, x, y, zc, ri in lig_atoms:
        all_atoms.append((atomic_num, x, y, zc, ri + max_prot_res))
        node_types.append(1)

    if not all_atoms:
        return None

    z_list = [a[0] for a in all_atoms]
    coords = [[a[1], a[2], a[3]] for a in all_atoms]
    res_idx = [a[4] for a in all_atoms]

    z_tensor = torch.tensor(z_list, dtype=torch.long)
    pos = torch.tensor(coords, dtype=torch.float)
    res_tensor = torch.tensor(res_idx, dtype=torch.long)
    node_type = torch.tensor(node_types, dtype=torch.long)
    ligand_mask = node_type == 1

    ei, ea, rb = _build_knn_radius_edges(
        pos.numpy(), res_tensor.numpy(), edge_cutoff, max_neighbors,
    )
    if ei is None:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, dtype=torch.float)
        res_boundary = torch.zeros(0, dtype=torch.long)
    else:
        edge_index = torch.from_numpy(ei)
        edge_attr = torch.from_numpy(ea)
        res_boundary = torch.from_numpy(rb)

    data = Data(
        z=z_tensor,
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        residue_boundary=res_boundary,
        residue_idx=res_tensor,
        node_type=node_type,
        ligand_mask=ligand_mask,
    )
    return data
