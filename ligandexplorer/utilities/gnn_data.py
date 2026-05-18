"""
Graph construction from PDB files for GNN classification (v2).

Features:
- Node: atomic number z, covalent degree, ring membership count
- Edge: interatomic distance, edge type (0=spatial, 1=covalent)
- Graph-level: amide bond count, element composition ratios (C/N/O/S/P)

Short-range spatial graph (cutoff=3.0 A) + inferred covalent bonds.
"""
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

COVALENT_RADII = {
    1: 0.31, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
    11: 1.66, 12: 1.41, 14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02,
    19: 2.03, 20: 1.76, 26: 1.24, 29: 1.32, 30: 1.22, 35: 1.20,
    53: 1.39,
}

SHORT_CUTOFF = 3.0

# Max values for embedding lookups (clamp to these)
MAX_DEGREE = 6
MAX_RING_COUNT = 8


def parse_pdb(pdb_path):
    """Parse PDB, return list of (atomic_number, x, y, z, residue_idx). Skip H."""
    atoms = []
    resid_map = {}
    resid_counter = 0
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            altloc = line[16] if len(line) > 16 else " "
            if altloc not in (" ", "", "A"):
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


def parse_pdb_with_conect(pdb_path):
    """Parse PDB atoms and CONECT records."""
    atoms = []
    conect = set()
    kept_serials = set()
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("CONECT"):
                try:
                    src = int(line[6:11])
                except ValueError:
                    continue
                for start in range(11, len(line), 5):
                    field = line[start:start + 5].strip()
                    if not field:
                        continue
                    try:
                        dst = int(field)
                    except ValueError:
                        continue
                    if src != dst:
                        conect.add(tuple(sorted((src, dst))))
                continue
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            altloc = line[16] if len(line) > 16 else " "
            if altloc not in (" ", "", "A"):
                continue
            elem = line[76:78].strip().upper() if len(line) >= 78 else ""
            if not elem:
                raw = line[12:16].strip()
                elem = raw.lstrip("0123456789").rstrip("0123456789'\"").upper()
                if not elem:
                    continue
            if elem == "H" or elem.startswith("H") and len(elem) <= 2 and elem not in ELEMENT_TO_Z:
                continue
            z = ELEMENT_TO_Z.get(elem, 0)
            if z == 0:
                continue
            try:
                serial = int(line[6:11])
                x = float(line[30:38])
                y = float(line[38:46])
                z_coord = float(line[46:54])
            except (ValueError, IndexError):
                continue
            kept_serials.add(serial)
            atoms.append((z, x, y, z_coord, serial))
    conect = {edge for edge in conect if edge[0] in kept_serials and edge[1] in kept_serials}
    return atoms, conect


def _build_spatial_edges(pos_np, cutoff, max_neighbors):
    """Build symmetric spatial edges within cutoff."""
    n = len(pos_np)
    if n < 2:
        return set()
    tree = cKDTree(pos_np)
    k_query = min(max_neighbors + 1, n)
    dists_knn, idx_knn = tree.query(pos_np, k=k_query, distance_upper_bound=cutoff)
    src = np.repeat(np.arange(n), k_query - 1)
    dst = idx_knn[:, 1:].ravel()
    d = dists_knn[:, 1:].ravel()
    valid = (dst < n) & np.isfinite(d)
    src, dst = src[valid], dst[valid]
    edges = set()
    for s, t in zip(src.tolist(), dst.tolist()):
        edges.add((s, t))
        edges.add((t, s))
    return edges


def _infer_covalent_pairs(z_np, pos_np, tolerance=0.45):
    """Infer covalent bonds from atomic distances and covalent radii."""
    pairs = set()
    n = len(z_np)
    if n < 2:
        return pairs
    tree = cKDTree(pos_np)
    candidates = tree.query_pairs(r=2.4)
    for i, j in candidates:
        ri = COVALENT_RADII.get(int(z_np[i]), 0.77)
        rj = COVALENT_RADII.get(int(z_np[j]), 0.77)
        max_dist = ri + rj + tolerance
        d = float(np.linalg.norm(pos_np[i] - pos_np[j]))
        if 0.35 <= d <= max_dist:
            pairs.add((i, j))
    return pairs


def _compute_covalent_degree(n_atoms, cov_pairs):
    """Compute covalent degree for each atom."""
    degree = np.zeros(n_atoms, dtype=np.int64)
    for i, j in cov_pairs:
        degree[i] += 1
        degree[j] += 1
    return np.clip(degree, 0, MAX_DEGREE)


def _compute_ring_count(n_atoms, cov_pairs):
    """Compute number of SSSR rings each atom belongs to.

    Uses a BFS-based approach to find all rings up to size 8.
    """
    ring_count = np.zeros(n_atoms, dtype=np.int64)
    if not cov_pairs:
        return np.clip(ring_count, 0, MAX_RING_COUNT)

    adj = [[] for _ in range(n_atoms)]
    for i, j in cov_pairs:
        adj[i].append(j)
        adj[j].append(i)

    visited_rings = set()
    for start in range(n_atoms):
        if not adj[start]:
            continue
        # BFS to find shortest cycles through start
        # Use DFS with depth limit for efficiency
        stack = [(start, -1, [start])]
        while stack:
            node, parent, path = stack.pop()
            if len(path) > 8:
                continue
            for nb in adj[node]:
                if nb == parent:
                    continue
                if nb == start and len(path) >= 3:
                    ring = tuple(sorted(path))
                    if ring not in visited_rings:
                        visited_rings.add(ring)
                        for atom_idx in path:
                            ring_count[atom_idx] += 1
                elif nb not in path and len(path) < 8:
                    stack.append((nb, node, path + [nb]))

    return np.clip(ring_count, 0, MAX_RING_COUNT)


def _count_amide_bonds(z_np, cov_pairs):
    """Count amide bond patterns: N bonded to C which is bonded to O (C=O-N).

    Detects the N-C(=O) motif characteristic of peptide bonds.
    Returns integer count.
    """
    if not cov_pairs:
        return 0

    n = len(z_np)
    adj = [set() for _ in range(n)]
    for i, j in cov_pairs:
        adj[i].add(j)
        adj[j].add(i)

    amide_count = 0
    # Look for C atoms (z=6) bonded to both N (z=7) and O (z=8)
    for c_idx in range(n):
        if z_np[c_idx] != 6:
            continue
        has_n = False
        has_terminal_o = False
        for nb in adj[c_idx]:
            if z_np[nb] == 7:
                has_n = True
            elif z_np[nb] == 8:
                # Terminal O (degree 1) suggests C=O rather than C-O-C
                if len(adj[nb]) == 1:
                    has_terminal_o = True
        if has_n and has_terminal_o:
            amide_count += 1

    return amide_count


def _compute_element_ratios(z_np):
    """Compute element composition ratios: [frac_C, frac_N, frac_O, frac_S, frac_P].

    Returns 5-dim float array normalized by total atom count.
    """
    n = len(z_np)
    if n == 0:
        return np.zeros(5, dtype=np.float32)
    counts = np.zeros(5, dtype=np.float32)
    for z in z_np:
        if z == 6:
            counts[0] += 1
        elif z == 7:
            counts[1] += 1
        elif z == 8:
            counts[2] += 1
        elif z == 16:
            counts[3] += 1
        elif z == 15:
            counts[4] += 1
    return counts / n


def _count_functional_groups(z_np, pos_np, cov_pairs):
    """Count functional groups using bond-length-based bond order inference.

    Bond order thresholds (Angstrom):
        C=O: < 1.30      C-O: >= 1.30      C=C: < 1.40 (non-ring)
        C-C: >= 1.40      C=N: < 1.33       aromatic: avg < 1.43

    Returns 32-dim float32 array:
      --- Bond-order derived ---
        [ 0] carbonyl_count       C=O (O degree=1, dist<1.30)
        [ 1] hydroxyl_count       C-OH (O degree=1, dist>=1.30)
        [ 2] ester_count          C(=O)-O-C
        [ 3] ether_count          C-O-C (no adjacent C=O)
        [ 4] phosphoester_count   P-O bonds
        [ 5] aromatic_ring_count  5-6 C/N rings (avg bond<1.43)
        [ 6] sp2_carbon_frac      C with 3 heavy neighbors / total C
        [ 7] longest_carbon_chain diameter of C-C subgraph
        [ 8] terminal_amine       N degree=1
        [ 9] terminal_carboxyl    C with 2 short-O (COO-)
        [10] guanidinium_count    C bonded to 3 N
        [11] hydroxyl_on_sp3C     C-OH where C is sp3 (sugar indicator)
        [12] disulfide_count      S-S bonds
        [13] rotatable_bonds      non-ring single bonds
      --- Sugar / Nucleic acid specific ---
        [14] pyranose_ring_count  6-membered ring with 1 O + 5 C
        [15] furanose_ring_count  5-membered ring with 1 O + 4 C
        [16] glycosidic_bond      ring-C - O - ring-C (inter-sugar link)
      --- Heterocycle / Organic specific ---
        [17] N_heterocycle_count  5-6 membered ring containing N
        [18] halogen_count        F + Cl + Br + I atoms
        [19] thioether_count      C-S-C (S degree=2)
        [20] sulfonyl_count       S bonded to >=2 terminal O (S=O)
        [21] CC_double_nonaro     non-aromatic-ring C=C (dist<1.40)
      --- Lipid / size features ---
        [22] n_aliphatic_tails    carbon chain segments > 8C (non-ring sp3)
        [23] ring_atom_fraction   atoms in any ring / total atoms
      --- Normalized ratios ---
        [24] heavy_atom_count     total atom count (log-scaled, log10(n))
        [25] amide_per_atom       amide_count / n (from caller)
      --- Backbone topology (from chain tracing) ---
        [26] backbone_amide_ratio backbone_amides / total_amides
        [27] sidechain_amide_cnt  amide Cs NOT on backbone chain
        [28] has_n_terminus       1.0 if backbone N-terminus detected
        [29] has_c_terminus       1.0 if backbone C-terminus detected
        [30] has_both_termini     1.0 if both N and C termini (linear peptide)
        [31] backbone_length_norm backbone_chain_length / total_atoms
    """
    FUNC_DIM = 32
    n = len(z_np)
    result = np.zeros(FUNC_DIM, dtype=np.float32)
    if not cov_pairs or n < 2:
        result[24] = np.log10(max(n, 1))
        return result

    adj = [set() for _ in range(n)]
    bond_dist = {}
    for i, j in cov_pairs:
        adj[i].add(j)
        adj[j].add(i)
        d = float(np.linalg.norm(pos_np[i] - pos_np[j]))
        bond_dist[(i, j)] = d
        bond_dist[(j, i)] = d

    # ---- [0] Carbonyl: O degree=1, bonded to C, dist < 1.30 ----
    carbonyl_oxygens = set()
    for o_idx in range(n):
        if z_np[o_idx] != 8 or len(adj[o_idx]) != 1:
            continue
        nb = next(iter(adj[o_idx]))
        if z_np[nb] == 6 and bond_dist.get((o_idx, nb), 2.0) < 1.30:
            result[0] += 1
            carbonyl_oxygens.add(o_idx)

    # ---- [1] Hydroxyl: O degree=1, bonded to C, dist >= 1.30 ----
    hydroxyl_oxygens = set()
    for o_idx in range(n):
        if z_np[o_idx] != 8 or len(adj[o_idx]) != 1 or o_idx in carbonyl_oxygens:
            continue
        nb = next(iter(adj[o_idx]))
        if z_np[nb] == 6 and bond_dist.get((o_idx, nb), 2.0) >= 1.30:
            result[1] += 1
            hydroxyl_oxygens.add(o_idx)

    # ---- [2] Ester / [3] Ether: bridging O between two C ----
    for o_idx in range(n):
        if z_np[o_idx] != 8 or len(adj[o_idx]) != 2:
            continue
        nbs = list(adj[o_idx])
        if z_np[nbs[0]] != 6 or z_np[nbs[1]] != 6:
            continue
        has_adj_carbonyl = any(
            c_nb in carbonyl_oxygens
            for c in nbs for c_nb in adj[c] if c_nb != o_idx
        )
        if has_adj_carbonyl:
            result[2] += 1
        else:
            result[3] += 1

    # ---- [4] Phosphoester: P-O bonds ----
    for i, j in cov_pairs:
        if (z_np[i] == 15 and z_np[j] == 8) or (z_np[i] == 8 and z_np[j] == 15):
            result[4] += 1

    # ---- Ring detection (shared infrastructure) ----
    # Find ALL 3-8 membered rings via DFS
    all_rings = set()   # frozenset of atom indices
    ring_list = []      # list of tuples for later use
    for start in range(n):
        if len(adj[start]) < 2:
            continue
        stack = [(start, -1, [start])]
        while stack:
            node, parent, path = stack.pop()
            if len(path) > 8:
                continue
            for nb in adj[node]:
                if nb == parent:
                    continue
                if nb == start and 3 <= len(path) <= 8:
                    ring_key = frozenset(path)
                    if ring_key not in all_rings:
                        all_rings.add(ring_key)
                        ring_list.append(tuple(path))
                elif nb not in path and len(path) < 8:
                    stack.append((nb, node, path + [nb]))

    ring_atom_set = set()
    for ring in ring_list:
        ring_atom_set.update(ring)

    # Classify rings
    aromatic_rings = set()
    pyranose_count = 0
    furanose_count = 0
    n_heterocycle_count = 0
    ring_c_atoms = set()  # C atoms that are in any ring

    for ring in ring_list:
        rsize = len(ring)
        elems = [z_np[a] for a in ring]
        n_O_in_ring = elems.count(8)
        n_C_in_ring = elems.count(6)
        n_N_in_ring = elems.count(7)

        for a in ring:
            if z_np[a] == 6:
                ring_c_atoms.add(a)

        # [5] Aromatic: 5-6 C/N ring with avg bond < 1.43
        if rsize in (5, 6) and all(e in (6, 7) for e in elems):
            dists_in_ring = []
            for k in range(rsize):
                a, b = ring[k], ring[(k+1) % rsize]
                dists_in_ring.append(bond_dist.get((a, b), 1.5))
            if sum(dists_in_ring) / len(dists_in_ring) < 1.43:
                aromatic_rings.add(frozenset(ring))

        # [14] Pyranose: 6-membered with 1 O + 5 C
        if rsize == 6 and n_O_in_ring == 1 and n_C_in_ring == 5:
            pyranose_count += 1

        # [15] Furanose: 5-membered with 1 O + 4 C
        if rsize == 5 and n_O_in_ring == 1 and n_C_in_ring == 4:
            furanose_count += 1

        # [17] N-heterocycle: 5-6 membered ring containing at least 1 N
        if rsize in (5, 6) and n_N_in_ring >= 1:
            n_heterocycle_count += 1

    result[5] = len(aromatic_rings)
    result[14] = pyranose_count
    result[15] = furanose_count
    result[17] = n_heterocycle_count

    # ---- [6] sp2 carbon fraction ----
    n_carbon = sum(1 for i in range(n) if z_np[i] == 6)
    n_sp2 = sum(1 for i in range(n) if z_np[i] == 6 and len(adj[i]) == 3)
    result[6] = n_sp2 / max(n_carbon, 1)

    # ---- [7] Longest carbon chain (BFS diameter of C-C subgraph) ----
    carbon_adj = [set() for _ in range(n)]
    for i, j in cov_pairs:
        if z_np[i] == 6 and z_np[j] == 6:
            carbon_adj[i].add(j)
            carbon_adj[j].add(i)
    max_chain = 0
    visited_chain = [False] * n
    for start in range(n):
        if z_np[start] != 6 or visited_chain[start] or not carbon_adj[start]:
            continue
        dist = {start: 0}
        queue = [start]
        farthest = start
        while queue:
            node = queue.pop(0)
            visited_chain[node] = True
            for nb in carbon_adj[node]:
                if nb not in dist:
                    dist[nb] = dist[node] + 1
                    queue.append(nb)
                    if dist[nb] > dist[farthest]:
                        farthest = nb
        dist2 = {farthest: 0}
        queue2 = [farthest]
        while queue2:
            node = queue2.pop(0)
            for nb in carbon_adj[node]:
                if nb not in dist2:
                    dist2[nb] = dist2[node] + 1
                    queue2.append(nb)
        max_chain = max(max_chain, max(dist2.values()) + 1)
    result[7] = max_chain

    # ---- [8] Terminal amine ----
    for i in range(n):
        if z_np[i] == 7 and len(adj[i]) == 1:
            result[8] += 1

    # ---- [9] Terminal carboxyl: C bonded to 2 short-bond terminal O ----
    for c_idx in range(n):
        if z_np[c_idx] != 6:
            continue
        short_o = sum(1 for nb in adj[c_idx]
                      if z_np[nb] == 8 and len(adj[nb]) == 1
                      and bond_dist.get((c_idx, nb), 2.0) < 1.30)
        if short_o >= 2:
            result[9] += 1

    # ---- [10] Guanidinium: C bonded to >= 3 N ----
    for c_idx in range(n):
        if z_np[c_idx] != 6:
            continue
        if sum(1 for nb in adj[c_idx] if z_np[nb] == 7) >= 3:
            result[10] += 1

    # ---- [11] Hydroxyl on sp3 carbon (sugar/polyol indicator) ----
    for o_idx in hydroxyl_oxygens:
        nb = next(iter(adj[o_idx]))
        if len(adj[nb]) >= 4:
            result[11] += 1

    # ---- [12] Disulfide ----
    for i, j in cov_pairs:
        if z_np[i] == 16 and z_np[j] == 16:
            result[12] += 1

    # ---- [13] Rotatable bonds: non-ring single bonds ----
    ring_bond_set = set()
    for ring in ring_list:
        for k in range(len(ring)):
            a, b = ring[k], ring[(k+1) % len(ring)]
            ring_bond_set.add((min(a, b), max(a, b)))
    n_rotatable = 0
    for i, j in cov_pairs:
        key = (min(i, j), max(i, j))
        if key in ring_bond_set:
            continue
        if bond_dist.get((i, j), 1.5) > 1.40:
            n_rotatable += 1
    result[13] = n_rotatable

    # ---- [16] Glycosidic bond: ring-C - O(degree=2) - ring-C ----
    for o_idx in range(n):
        if z_np[o_idx] != 8 or len(adj[o_idx]) != 2:
            continue
        nbs = list(adj[o_idx])
        if (z_np[nbs[0]] == 6 and nbs[0] in ring_c_atoms
                and z_np[nbs[1]] == 6 and nbs[1] in ring_c_atoms):
            result[16] += 1

    # ---- [18] Halogen count: F(9), Cl(17), Br(35), I(53) ----
    halogens = {9, 17, 35, 53}
    result[18] = sum(1 for i in range(n) if z_np[i] in halogens)

    # ---- [19] Thioether: C-S-C where S degree=2 ----
    for s_idx in range(n):
        if z_np[s_idx] != 16 or len(adj[s_idx]) != 2:
            continue
        nbs = list(adj[s_idx])
        if z_np[nbs[0]] == 6 and z_np[nbs[1]] == 6:
            result[19] += 1

    # ---- [20] Sulfonyl: S bonded to >= 2 terminal O ----
    for s_idx in range(n):
        if z_np[s_idx] != 16:
            continue
        term_o = sum(1 for nb in adj[s_idx]
                     if z_np[nb] == 8 and len(adj[nb]) == 1)
        if term_o >= 2:
            result[20] += 1

    # ---- [21] Non-aromatic C=C double bonds (dist < 1.40, not in aromatic ring) ----
    aromatic_atoms = set()
    for ring in aromatic_rings:
        aromatic_atoms.update(ring)
    for i, j in cov_pairs:
        if z_np[i] != 6 or z_np[j] != 6:
            continue
        if i in aromatic_atoms and j in aromatic_atoms:
            continue
        if bond_dist.get((i, j), 1.5) < 1.40:
            result[21] += 1

    # ---- [22] Number of long aliphatic tail segments (>8 sp3 C in non-ring chain) ----
    sp3_carbon_nonring = set()
    for i in range(n):
        if z_np[i] == 6 and len(adj[i]) <= 4 and i not in ring_c_atoms:
            sp3_carbon_nonring.add(i)
    sp3_adj = [set() for _ in range(n)]
    for i, j in cov_pairs:
        if i in sp3_carbon_nonring and j in sp3_carbon_nonring:
            sp3_adj[i].add(j)
            sp3_adj[j].add(i)
    visited_sp3 = [False] * n
    n_long_tails = 0
    for start in sp3_carbon_nonring:
        if visited_sp3[start] or not sp3_adj[start]:
            continue
        component = []
        queue = [start]
        while queue:
            node = queue.pop(0)
            if visited_sp3[node]:
                continue
            visited_sp3[node] = True
            component.append(node)
            for nb in sp3_adj[node]:
                if not visited_sp3[nb]:
                    queue.append(nb)
        if len(component) > 8:
            n_long_tails += 1
    result[22] = n_long_tails

    # ---- [23] Ring atom fraction ----
    result[23] = len(ring_atom_set) / n

    # ---- [24] Heavy atom count (log10 scaled) ----
    result[24] = np.log10(max(n, 1))

    # ---- [25] amide_per_atom: filled by caller (needs amide_count) ----

    # ---- [26-31] Backbone topology features ----
    amide_carbons = set()
    for c_idx in range(n):
        if z_np[c_idx] != 6:
            continue
        has_n = any(z_np[nb] == 7 for nb in adj[c_idx])
        has_to = any(z_np[nb] == 8 and len(adj[nb]) == 1 for nb in adj[c_idx])
        if has_n and has_to:
            amide_carbons.add(c_idx)

    if amide_carbons:
        bb = _trace_peptide_backbone(z_np, adj, amide_carbons)
        total_amides = len(amide_carbons)
        result[26] = bb['backbone_amides'] / total_amides
        result[27] = bb['sidechain_amides']
        result[28] = 1.0 if bb['has_n_terminus'] else 0.0
        result[29] = 1.0 if bb['has_c_terminus'] else 0.0
        result[30] = 1.0 if (bb['has_n_terminus'] and bb['has_c_terminus']) else 0.0
        result[31] = bb['chain_length'] / n

    return result


def _trace_peptide_backbone(z_np, adj, amide_carbons):
    """Trace the peptide backbone chain by walking amide_C -> N -> Ca -> next_amide_C.

    Returns dict with:
        is_cyclic: bool
        backbone_amides: int  (amide Cs on the traced chain)
        sidechain_amides: int (amide Cs NOT on the chain)
        has_n_terminus: bool
        has_c_terminus: bool
        chain_length: int
    """
    result = dict(is_cyclic=False, backbone_amides=0, sidechain_amides=len(amide_carbons),
                  has_n_terminus=False, has_c_terminus=False, chain_length=0)
    if len(amide_carbons) < 1:
        return result

    backbone_adj = {c: set() for c in amide_carbons}
    for c_idx in amide_carbons:
        for n_atom in adj[c_idx]:
            if z_np[n_atom] != 7:
                continue
            for ca in adj[n_atom]:
                if z_np[ca] != 6 or ca == c_idx:
                    continue
                for next_c in adj[ca]:
                    if next_c in amide_carbons and next_c != c_idx:
                        backbone_adj[c_idx].add(next_c)
                        backbone_adj[next_c].add(c_idx)

    terminals = [c for c in amide_carbons if len(backbone_adj[c]) <= 1]
    start = terminals[0] if terminals else next(iter(amide_carbons))

    chain = [start]
    visited = {start}
    current = start
    is_cyclic = False

    while True:
        found_next = False
        for neighbor in backbone_adj[current]:
            if neighbor == start and len(chain) >= 3:
                is_cyclic = True
                break
            if neighbor not in visited:
                chain.append(neighbor)
                visited.add(neighbor)
                current = neighbor
                found_next = True
                break
        if not found_next or is_cyclic:
            break

    on_backbone = set(chain)
    off_backbone = amide_carbons - on_backbone

    has_n_term = False
    has_c_term = False

    if not is_cyclic and len(chain) >= 2:
        for end_c in (chain[0], chain[-1]):
            # N-terminus: H2N-Ca-amide_C(=O)-...
            for ca_cand in adj[end_c]:
                if z_np[ca_cand] != 6 or ca_cand in amide_carbons:
                    continue
                for nb in adj[ca_cand]:
                    if z_np[nb] == 7:
                        n_heavy = sum(1 for x in adj[nb] if z_np[x] != 1)
                        if n_heavy <= 1:
                            has_n_term = True

            # C-terminus: ...-amide_C(=O)-NH-Ca-COO⁻ (or -C=O with missing OXT)
            for n_atom in adj[end_c]:
                if z_np[n_atom] != 7:
                    continue
                for ca_cand in adj[n_atom]:
                    if z_np[ca_cand] != 6 or ca_cand == end_c or ca_cand in amide_carbons:
                        continue
                    for carboxyl_c in adj[ca_cand]:
                        if z_np[carboxyl_c] != 6 or carboxyl_c in amide_carbons:
                            continue
                        term_o = sum(1 for x in adj[carboxyl_c]
                                     if z_np[x] == 8 and len(adj[x]) == 1)
                        has_n_on_c = any(z_np[x] == 7 for x in adj[carboxyl_c])
                        if term_o >= 2 or (term_o >= 1 and not has_n_on_c):
                            has_c_term = True

    result['is_cyclic'] = is_cyclic
    result['backbone_amides'] = len(chain)
    result['sidechain_amides'] = len(off_backbone)
    result['has_n_terminus'] = has_n_term
    result['has_c_terminus'] = has_c_term
    result['chain_length'] = len(chain)
    return result


def _detect_amide_macrocycle(z_np, cov_pairs):
    """Detect whether amide bonds form a macrocyclic backbone.

    Uses explicit backbone chain tracing (amide_C -> N -> Ca -> next_amide_C)
    for robust cyclic peptide detection.

    Returns:
        n_amide_bonds: int - number of amide bonds detected
        is_backbone_cyclic: int - 1 if backbone forms a cycle, 0 otherwise
    """
    if not cov_pairs:
        return 0, 0

    n = len(z_np)
    adj = [set() for _ in range(n)]
    for i, j in cov_pairs:
        adj[i].add(j)
        adj[j].add(i)

    amide_carbons = set()
    for c_idx in range(n):
        if z_np[c_idx] != 6:
            continue
        has_n = any(z_np[nb] == 7 for nb in adj[c_idx])
        has_terminal_o = any(z_np[nb] == 8 and len(adj[nb]) == 1 for nb in adj[c_idx])
        if has_n and has_terminal_o:
            amide_carbons.add(c_idx)

    n_amide = len(amide_carbons)
    if n_amide < 3:
        return n_amide, 0

    stats = _trace_peptide_backbone(z_np, adj, amide_carbons)
    return n_amide, int(stats['is_cyclic'])


def _build_union_edges(pos_np, z_np, cutoff, max_neighbors):
    """Build union of short-range spatial edges and covalent edges.

    Covalent bonds are ALWAYS inferred from atomic coordinates and covalent radii.
    Returns: edge_index (2,E), edge_attr (E,), edge_type (E,), cov_pairs set
    """
    spatial = _build_spatial_edges(pos_np, cutoff, max_neighbors)
    covalent_undirected = _infer_covalent_pairs(z_np, pos_np)

    covalent = set()
    for i, j in covalent_undirected:
        covalent.add((i, j))
        covalent.add((j, i))

    edges = sorted(spatial | covalent)
    if not edges:
        empty_i = np.zeros((2, 0), dtype=np.int64)
        empty_f = np.zeros(0, dtype=np.float32)
        empty_l = np.zeros(0, dtype=np.int64)
        return empty_i, empty_f, empty_l, covalent_undirected

    row = np.array([e[0] for e in edges], dtype=np.int64)
    col = np.array([e[1] for e in edges], dtype=np.int64)
    dists = np.linalg.norm(pos_np[row] - pos_np[col], axis=1).astype(np.float32)

    edge_type = np.zeros(len(edges), dtype=np.int64)
    for idx, edge in enumerate(edges):
        if edge in covalent:
            edge_type[idx] = 1

    return np.stack([row, col]), dists, edge_type, covalent_undirected


def structure_to_graph(pdb_path, cutoff=SHORT_CUTOFF, max_neighbors=32):
    """Convert PDB to PyG Data with enriched features.

    Covalent bonds are inferred purely from atomic coordinates and covalent radii.
    Returns None only if file cannot be parsed.
    """
    atoms = parse_pdb(pdb_path)
    if len(atoms) == 0:
        return None

    z_list, coords = [], []
    for atomic_num, x, y, zc, _resid in atoms:
        z_list.append(atomic_num)
        coords.append([x, y, zc])

    z_np = np.array(z_list, dtype=np.int64)
    pos_np = np.array(coords, dtype=np.float32)

    if not np.isfinite(pos_np).all():
        return None

    n = len(atoms)
    if n == 1:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, dtype=torch.float)
        edge_type_arr = torch.zeros(0, dtype=torch.long)
        cov_pairs = set()
    else:
        ei, ea, et, cov_pairs = _build_union_edges(
            pos_np, z_np, cutoff, max_neighbors)
        edge_index = torch.from_numpy(ei)
        edge_attr = torch.from_numpy(ea)
        edge_type_arr = torch.from_numpy(et)

    # Node-level features
    degree = _compute_covalent_degree(n, cov_pairs)
    ring_count = _compute_ring_count(n, cov_pairs)

    # Graph-level features
    amide_count = _count_amide_bonds(z_np, cov_pairs)
    elem_ratios = _compute_element_ratios(z_np)
    _, is_backbone_cyclic = _detect_amide_macrocycle(z_np, cov_pairs)
    func_groups = _count_functional_groups(z_np, pos_np, cov_pairs)
    func_groups[25] = amide_count / max(n, 1)  # amide_per_atom

    return Data(
        z=torch.tensor(z_list, dtype=torch.long),
        pos=torch.tensor(coords, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_type=edge_type_arr,
        degree=torch.from_numpy(degree),
        ring_count=torch.from_numpy(ring_count),
        amide_count=torch.tensor([amide_count], dtype=torch.long),
        elem_ratios=torch.from_numpy(elem_ratios),
        is_backbone_cyclic=torch.tensor([is_backbone_cyclic], dtype=torch.long),
        func_groups=torch.from_numpy(func_groups),
    )


def complex_to_graph(protein_path, ligand_path, pocket_cutoff=6.0,
                     edge_cutoff=5.0, max_pocket_atoms=800, max_neighbors=32):
    """Build protein-ligand complex graph for Task B (hybrid: spatial + covalent)."""
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

    all_atoms, node_types = [], []
    for atomic_num, x, y, zc, ri in pocket_atoms:
        all_atoms.append((atomic_num, x, y, zc))
        node_types.append(0)
    for atomic_num, x, y, zc, ri in lig_atoms:
        all_atoms.append((atomic_num, x, y, zc))
        node_types.append(1)
    if not all_atoms:
        return None

    z_list = [a[0] for a in all_atoms]
    coords = [[a[1], a[2], a[3]] for a in all_atoms]
    z_np = np.array(z_list, dtype=np.int64)
    pos_np = np.array(coords, dtype=np.float32)

    spatial = _build_spatial_edges(pos_np, edge_cutoff, max_neighbors)
    cov_pairs = _infer_covalent_pairs(z_np, pos_np)
    covalent = set()
    for i, j in cov_pairs:
        covalent.add((i, j))
        covalent.add((j, i))

    all_edges = sorted(spatial | covalent)
    if all_edges:
        row = np.array([e[0] for e in all_edges], dtype=np.int64)
        col = np.array([e[1] for e in all_edges], dtype=np.int64)
        dists = np.linalg.norm(pos_np[row] - pos_np[col], axis=1).astype(np.float32)
        edge_index = torch.from_numpy(np.stack([row, col]))
        edge_attr = torch.from_numpy(dists)
        et = np.zeros(len(all_edges), dtype=np.int64)
        for idx, edge in enumerate(all_edges):
            if edge in covalent:
                et[idx] = 1
        edge_type = torch.from_numpy(et)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, dtype=torch.float)
        edge_type = torch.zeros(0, dtype=torch.long)

    z_tensor = torch.tensor(z_list, dtype=torch.long)
    pos = torch.tensor(coords, dtype=torch.float)
    node_type = torch.tensor(node_types, dtype=torch.long)
    ligand_mask = node_type == 1

    return Data(z=z_tensor, pos=pos, edge_index=edge_index, edge_attr=edge_attr,
                edge_type=edge_type, node_type=node_type, ligand_mask=ligand_mask)
