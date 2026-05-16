"""Molecule-level descriptors that don't depend on bond order.

Three families of features are produced here:

1. Element / charge counts (size 16).
2. Physicochemical "equivalents" computed by geometry and rule
   logic instead of RDKit's bond-order-dependent descriptors. Size 12.
3. Bond-length histogram (6 buckets x [count, mean]) -> size 12.
4. Topological / ring features (size 6).
5. 3D shape features (size 8).

The matching column names live in :mod:`lgbm_featurizer.schema`.
"""
from __future__ import annotations

from typing import Dict, List

import networkx as nx
import numpy as np
from rdkit import Chem

from .chemistry import (
    ALKALI_ZS,
    ALKALINE_EARTH_ZS,
    ATOMIC_MASS,
    HETEROATOM_ZS,
    METAL_ZS,
    TRANSITION_ZS,
    donor_acceptor_flags,
    is_ring_aromatic_geom,
)
from .loader import LigandHandle
from .schema import BOND_LENGTH_BUCKETS

# Rotatable-bond SMARTS uses wildcard ~ to be compatible with our
# single-bond RWMol. Definition: any non-ring single-like bond between
# two heavy atoms each with degree >= 2 (excluding terminal atoms).
_ROTATABLE_BOND_SMARTS = Chem.MolFromSmarts("[!D1]~;!@[!D1]")


# ---------------------------------------------------------------------------
# Element / charge counts (block: element_counts, dim 16)
# ---------------------------------------------------------------------------

def element_counts(handle: LigandHandle) -> List[float]:
    z = handle.z_array
    n = handle.n_atoms
    counts = {
        "count_C": int(np.sum(z == 6)),
        "count_N": int(np.sum(z == 7)),
        "count_O": int(np.sum(z == 8)),
        "count_P": int(np.sum(z == 15)),
        "count_S": int(np.sum(z == 16)),
        "count_F": int(np.sum(z == 9)),
        "count_Cl": int(np.sum(z == 17)),
        "count_Br": int(np.sum(z == 35)),
        "count_I": int(np.sum(z == 53)),
    }
    z_set = set(z.tolist())
    counts["count_metal_alkali"] = sum(
        int(np.sum(z == zi)) for zi in ALKALI_ZS if zi in z_set
    )
    counts["count_metal_alkaline_earth"] = sum(
        int(np.sum(z == zi)) for zi in ALKALINE_EARTH_ZS if zi in z_set
    )
    counts["count_metal_transition"] = sum(
        int(np.sum(z == zi)) for zi in TRANSITION_ZS if zi in z_set
    )
    other_metals = METAL_ZS - ALKALI_ZS - ALKALINE_EARTH_ZS - TRANSITION_ZS
    counts["count_metal_other"] = sum(
        int(np.sum(z == zi)) for zi in other_metals if zi in z_set
    )
    n_hetero = sum(int(np.sum(z == zi)) for zi in HETEROATOM_ZS)
    counts["heteroatom_ratio"] = float(n_hetero) / max(n, 1)

    fc = handle.formal_charges
    counts["formal_charge_pos_count"] = int(np.sum(fc > 0))
    counts["formal_charge_neg_count"] = int(np.sum(fc < 0))

    # Schema order:
    return [
        counts["count_C"], counts["count_N"], counts["count_O"],
        counts["count_P"], counts["count_S"], counts["count_F"],
        counts["count_Cl"], counts["count_Br"], counts["count_I"],
        counts["count_metal_alkali"], counts["count_metal_alkaline_earth"],
        counts["count_metal_transition"], counts["count_metal_other"],
        counts["heteroatom_ratio"],
        counts["formal_charge_pos_count"], counts["formal_charge_neg_count"],
    ]


# ---------------------------------------------------------------------------
# Physicochemical geometry equivalents (block: physchem_geom, dim 12)
# ---------------------------------------------------------------------------

def _molecular_weight(handle: LigandHandle) -> float:
    """Sum of standard atomic masses; ignores implicit H since they don't
    affect bond-order-independent feature design materially.
    """
    return float(
        sum(ATOMIC_MASS.get(int(z), 0.0) for z in handle.z_array)
    )


def _hbd_hba_counts(handle: LigandHandle) -> tuple:
    """Rule-based HBD / HBA counts (no implicit-H storage needed).

    Aromatic N / O / S atoms use the sp2 path of
    :func:`lgbm_featurizer.chemistry.donor_acceptor_flags` so pyridine
    N is correctly classified as 0-H acceptor (rather than the
    sp3-valence default which would assign it an implicit H).
    """
    donors = 0
    acceptors = 0
    aromatic_mask = handle.aromatic_mask
    for i in range(handle.n_atoms):
        z = int(handle.z_array[i])
        deg = int(handle.heavy_degree[i])
        fc = int(handle.formal_charges[i])
        is_arom = bool(aromatic_mask[i])
        d_flag, a_flag = donor_acceptor_flags(
            z, deg, fc, is_aromatic=is_arom
        )
        if d_flag:
            donors += 1
        if a_flag:
            acceptors += 1
    return donors, acceptors


def _num_rotatable_bonds(handle: LigandHandle) -> int:
    """Count non-ring bonds where each endpoint has degree >= 2.

    Uses the SMARTS `[!D1]~;!@[!D1]` on our single-bond RWMol so that
    the result is independent of bond perception.
    """
    if _ROTATABLE_BOND_SMARTS is None or handle.rdmol.GetNumAtoms() < 2:
        return 0
    matches = handle.rdmol.GetSubstructMatches(_ROTATABLE_BOND_SMARTS,
                                               uniquify=True)
    # SMARTS returns ordered tuples; canonicalize to set of frozensets.
    unique = {frozenset(m) for m in matches}
    return len(unique)


def _aromatic_rings_geom(handle: LigandHandle) -> tuple:
    """Return ``(num_aromatic, num_aliphatic, num_saturated)`` ring counts
    using only geometry / connectivity (no perception).

    - Aromatic: ring of size 5 or 6 passing the planarity + angle test.
    - Aliphatic: any ring of size 3-7 that is not aromatic.
    - Saturated: aliphatic ring whose atoms each have degree consistent
      with sp3 (i.e. no atom with degree < 2 in the ring; we accept
      degree 2/3/4 since terminal-degree-1 means broken ring anyway).
    """
    n_aromatic = 0
    n_aliphatic = 0
    n_saturated = 0
    for ring in handle.rings:
        ring_size = len(ring)
        if ring_size < 3 or ring_size > 12:
            continue
        is_aromatic = is_ring_aromatic_geom(ring, handle.coords)
        if is_aromatic:
            n_aromatic += 1
            continue
        n_aliphatic += 1
        # A ring is "saturated" in our loose sense if every atom has the
        # expected single-bond degree (>= 2 in-ring + maybe substituent).
        # This is a structural check rather than chemical sp3 detection.
        all_ok = True
        for idx in ring:
            if handle.heavy_degree[idx] < 2:
                all_ok = False
                break
        if all_ok:
            n_saturated += 1
    return n_aromatic, n_aliphatic, n_saturated


def _spiro_and_bridgehead_atoms(handle: LigandHandle) -> tuple:
    """Identify spiro and bridgehead atoms from ring membership.

    Definitions (used in many chemoinformatics libraries):

    - Spiro atom: belongs to exactly two rings that share exactly one
      atom (that atom).
    - Bridgehead atom: belongs to exactly two rings that share two or
      more atoms (and the atom is one of them).
    """
    ring_membership: Dict[int, List[int]] = {}
    for ring_idx, ring in enumerate(handle.rings):
        for atom_idx in ring:
            ring_membership.setdefault(atom_idx, []).append(ring_idx)
    spiro = 0
    bridgehead = 0
    # Pre-compute pairwise ring intersections.
    n_rings = len(handle.rings)
    ring_sets = [set(r) for r in handle.rings]
    shared_atom_count: Dict[tuple, int] = {}
    for i in range(n_rings):
        for j in range(i + 1, n_rings):
            shared_atom_count[(i, j)] = len(ring_sets[i] & ring_sets[j])
    for atom_idx, members in ring_membership.items():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = sorted((members[i], members[j]))
                shared = shared_atom_count.get((a, b), 0)
                if shared == 1:
                    spiro += 1
                elif shared >= 2:
                    bridgehead += 1
                break
            else:
                continue
            break
    return spiro, bridgehead


def _fraction_csp3_geom(handle: LigandHandle) -> float:
    """Fraction of carbon atoms whose 4-neighbour geometry is sp3-like.

    For each carbon with heavy degree 4 we compute the average bond
    angle around the atom; if it's within 15 degrees of 109.5 we call
    it sp3. Returns 0.0 when there are no carbons.
    """
    coords = handle.coords
    n_c = 0
    n_sp3 = 0
    for i in range(handle.n_atoms):
        if int(handle.z_array[i]) != 6:
            continue
        n_c += 1
        nbrs = handle.neighbors[i]
        # 4-coordinate carbon: average of all C(C-X-Y) angles around i.
        if len(nbrs) != 4:
            continue
        angles = []
        for a_idx in range(len(nbrs)):
            for b_idx in range(a_idx + 1, len(nbrs)):
                v1 = coords[nbrs[a_idx]] - coords[i]
                v2 = coords[nbrs[b_idx]] - coords[i]
                denom = float(
                    np.linalg.norm(v1) * np.linalg.norm(v2)
                ) + 1e-9
                cos_a = float(np.dot(v1, v2) / denom)
                cos_a = max(-1.0, min(1.0, cos_a))
                angles.append(np.degrees(np.arccos(cos_a)))
        mean_angle = float(np.mean(angles))
        if abs(mean_angle - 109.5) < 15.0:
            n_sp3 += 1
    return float(n_sp3) / n_c if n_c > 0 else 0.0


def physchem_geom(handle: LigandHandle) -> List[float]:
    mw = _molecular_weight(handle)
    hac = handle.n_atoms
    nhet = sum(1 for z in handle.z_array if int(z) in HETEROATOM_ZS)
    hbd, hba = _hbd_hba_counts(handle)
    n_rot = _num_rotatable_bonds(handle)
    f_csp3 = _fraction_csp3_geom(handle)
    n_arom, n_aliph, n_sat = _aromatic_rings_geom(handle)
    n_spiro, n_bridge = _spiro_and_bridgehead_atoms(handle)
    return [
        mw, hac, nhet, hbd, hba, n_rot, f_csp3,
        n_arom, n_aliph, n_sat, n_spiro, n_bridge,
    ]


# ---------------------------------------------------------------------------
# Bond-length histogram (block: bond_length_hist, dim 12)
# ---------------------------------------------------------------------------

def bond_length_histogram(handle: LigandHandle) -> List[float]:
    if not handle.bonds:
        return [0.0] * (2 * len(BOND_LENGTH_BUCKETS))

    lengths = []
    for i, j in handle.bonds:
        d = float(np.linalg.norm(handle.coords[i] - handle.coords[j]))
        lengths.append(d)

    out: List[float] = []
    for lo, hi in BOND_LENGTH_BUCKETS:
        in_bucket = [d for d in lengths if lo <= d < hi]
        out.append(float(len(in_bucket)))
        out.append(float(np.mean(in_bucket)) if in_bucket else 0.0)
    return out


# ---------------------------------------------------------------------------
# Topology (block: topology, dim 6)
# ---------------------------------------------------------------------------

def _longest_chain(graph: nx.Graph) -> int:
    """Approximate longest shortest-path (graph diameter) of the
    largest connected component.

    Implementation: 2-BFS double sweep. BFS from an arbitrary start
    finds the furthest node ``u``; a second BFS from ``u`` returns
    its eccentricity, which equals the true diameter for trees and
    is a tight lower bound for general graphs. Cost is O(V + E),
    avoiding the O(N x (V + E)) BFS-from-every-node approach that
    dominated DNA / large-peptide feature time in the legacy
    implementation.
    """
    n = graph.number_of_nodes()
    if n <= 1:
        return 0
    # Largest connected component only.
    components = list(nx.connected_components(graph))
    if not components:
        return 0
    biggest = max(components, key=len)
    if len(biggest) <= 1:
        return 0
    subg = graph.subgraph(biggest)
    start = next(iter(biggest))
    farthest_dist = nx.single_source_shortest_path_length(subg, start)
    u = max(farthest_dist, key=farthest_dist.get)
    second = nx.single_source_shortest_path_length(subg, u)
    return int(max(second.values()))


def topology(handle: LigandHandle) -> List[float]:
    n_rings = len(handle.rings)
    ring_atoms = set()
    for r in handle.rings:
        ring_atoms.update(r)
    ring_atom_fraction = float(len(ring_atoms)) / max(handle.n_atoms, 1)
    longest_chain = _longest_chain(handle.graph)
    # Fused rings: ring pairs sharing >=2 atoms.
    n_fused = 0
    ring_sets = [set(r) for r in handle.rings]
    for i in range(len(ring_sets)):
        for j in range(i + 1, len(ring_sets)):
            if len(ring_sets[i] & ring_sets[j]) >= 2:
                n_fused += 1
    # Aromatic heterocycles: aromatic rings containing N/O/S.
    n_arom_het = 0
    for ring in handle.rings:
        if not is_ring_aromatic_geom(ring, handle.coords):
            continue
        ring_z = {int(handle.z_array[i]) for i in ring}
        if ring_z & {7, 8, 16}:
            n_arom_het += 1
    # Mode ring size (most common ring size; 0 if no rings).
    if handle.rings:
        from collections import Counter
        sizes = [len(r) for r in handle.rings]
        ring_size_mode = Counter(sizes).most_common(1)[0][0]
    else:
        ring_size_mode = 0
    return [
        float(n_rings),
        float(ring_atom_fraction),
        float(longest_chain),
        float(n_fused),
        float(n_arom_het),
        float(ring_size_mode),
    ]


# ---------------------------------------------------------------------------
# 3D shape (block: shape_3d, dim 8)
# ---------------------------------------------------------------------------

def shape_3d(handle: LigandHandle) -> List[float]:
    coords = handle.coords
    n = coords.shape[0]
    if n == 0:
        return [0.0] * 8
    if n == 1:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    centroid = coords.mean(axis=0)
    rel = coords - centroid
    rg = float(np.sqrt(np.mean(np.sum(rel ** 2, axis=1))))
    # Inertia eigenvalues (PCA on coordinates).
    cov = rel.T @ rel / n
    eig = np.linalg.eigvalsh(cov)  # ascending
    eig = np.sort(eig)[::-1]
    lam1, lam2, lam3 = float(eig[0]), float(eig[1]), float(eig[2])
    # Asphericity = lam1 - (lam2 + lam3) / 2
    asphericity = lam1 - (lam2 + lam3) / 2.0
    acylindricity = lam2 - lam3
    # Sphericity proxy: ratio of smallest to largest eigenvalue.
    sphericity = lam3 / lam1 if lam1 > 1e-9 else 0.0
    # Molecular diameter: max pairwise atom distance.
    diffs = coords[:, None, :] - coords[None, :, :]
    diameter = float(np.linalg.norm(diffs, axis=2).max())
    return [
        rg, lam1, lam2, lam3, asphericity, acylindricity,
        sphericity, diameter,
    ]
