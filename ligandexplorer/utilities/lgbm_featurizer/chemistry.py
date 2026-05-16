"""Rule-based / geometry-based chemistry primitives.

This module deliberately avoids RDKit's bond-order perception, since the
input is always PDB and bond orders cannot be reliably recovered. All
chemistry-flavored concepts (implicit H count, formal charge, donor /
acceptor flags, aromaticity) are implemented as pure geometric or
graph-degree rules.

The public API is consumed by :mod:`lgbm_featurizer.descriptors` and
:mod:`lgbm_featurizer.templates`.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Element / atomic-number tables
# ---------------------------------------------------------------------------

ELEMENT_TO_Z: Dict[str, int] = {
    "H": 1, "HE": 2, "LI": 3, "BE": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "NE": 10, "NA": 11, "MG": 12, "AL": 13, "SI": 14, "P": 15,
    "S": 16, "CL": 17, "AR": 18, "K": 19, "CA": 20, "SC": 21, "TI": 22,
    "V": 23, "CR": 24, "MN": 25, "FE": 26, "CO": 27, "NI": 28, "CU": 29,
    "ZN": 30, "GA": 31, "GE": 32, "AS": 33, "SE": 34, "BR": 35, "KR": 36,
    "RB": 37, "SR": 38, "Y": 39, "ZR": 40, "NB": 41, "MO": 42, "TC": 43,
    "RU": 44, "RH": 45, "PD": 46, "AG": 47, "CD": 48, "IN": 49, "SN": 50,
    "SB": 51, "TE": 52, "I": 53, "XE": 54, "CS": 55, "BA": 56, "LA": 57,
    "CE": 58, "PR": 59, "ND": 60, "PM": 61, "SM": 62, "EU": 63, "GD": 64,
    "TB": 65, "DY": 66, "HO": 67, "ER": 68, "TM": 69, "YB": 70, "LU": 71,
    "HF": 72, "TA": 73, "W": 74, "RE": 75, "OS": 76, "IR": 77, "PT": 78,
    "AU": 79, "HG": 80, "TL": 81, "PB": 82, "BI": 83, "PO": 84, "AT": 85,
    "RN": 86, "FR": 87, "RA": 88, "AC": 89, "TH": 90, "PA": 91, "U": 92,
}

Z_TO_ELEMENT: Dict[int, str] = {z: el for el, z in ELEMENT_TO_Z.items()}

# Standard atomic masses (most abundant isotope or weighted average) in
# atomic mass units. Used for MW. Keys are atomic numbers.
ATOMIC_MASS: Dict[int, float] = {
    1: 1.008, 5: 10.81, 6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998,
    11: 22.990, 12: 24.305, 14: 28.085, 15: 30.974, 16: 32.06, 17: 35.45,
    19: 39.098, 20: 40.078, 25: 54.938, 26: 55.845, 27: 58.933,
    28: 58.693, 29: 63.546, 30: 65.38, 33: 74.922, 34: 78.971,
    35: 79.904, 53: 126.904,
}

# Covalent radii in Angstrom (Cordero 2008 / Pyykko subset). Used for
# bond perception via radius-sum + tolerance.
COVALENT_RADII: Dict[int, float] = {
    1: 0.31, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
    11: 1.66, 12: 1.41, 13: 1.21, 14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02,
    19: 2.03, 20: 1.76, 25: 1.39, 26: 1.32, 27: 1.26, 28: 1.24,
    29: 1.32, 30: 1.22, 33: 1.19, 34: 1.20, 35: 1.20, 53: 1.39,
}
DEFAULT_COVALENT_RADIUS = 0.77

BOND_TOLERANCE = 0.45

# Atomic numbers we treat as metals / mono-atomic ions. These are excluded
# from covalent-bond inference (they only form coordination bonds in
# typical PDB structures, which we deliberately do not capture in v2.0).
METAL_ZS: FrozenSet[int] = frozenset({
    # Alkali / alkaline earth
    3, 11, 19, 37, 55, 4, 12, 20, 38, 56,
    # Transition metals 1st-3rd row
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    72, 73, 74, 75, 76, 77, 78, 79, 80,
    # Post-transition metals / metalloids commonly seen in PDB
    13, 31, 49, 50, 81, 82, 83,
    # Lanthanides
    57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
})

ALKALI_ZS: FrozenSet[int] = frozenset({3, 11, 19, 37, 55})
ALKALINE_EARTH_ZS: FrozenSet[int] = frozenset({4, 12, 20, 38, 56})
TRANSITION_ZS: FrozenSet[int] = frozenset({
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    72, 73, 74, 75, 76, 77, 78, 79, 80,
})

HALOGEN_ZS: FrozenSet[int] = frozenset({9, 17, 35, 53})
HETEROATOM_ZS: FrozenSet[int] = frozenset({
    7, 8, 9, 15, 16, 17, 33, 34, 35, 53,
})

# Typical maximum-valence (in covalent bonds) used to estimate implicit
# H counts for organic elements.  Keys are atomic numbers. For atoms not
# in this table the implicit-H estimate is 0.
TYPICAL_VALENCE: Dict[int, int] = {
    5: 3,   # B
    6: 4,   # C
    7: 3,   # N (neutral); +1 when 4-coordinate
    8: 2,   # O
    9: 1,   # F
    14: 4,  # Si
    15: 3,  # P (also commonly 5)
    16: 2,  # S (also commonly 4 or 6)
    17: 1,  # Cl
    34: 2,  # Se
    35: 1,  # Br
    53: 1,  # I
}


def normalize_element(symbol: Optional[str]) -> str:
    """Return the upper-cased / stripped element symbol, or ``''`` for
    empty input. PDB files sometimes store ``'Cl'`` / ``' C'``; this
    keeps comparisons consistent throughout the package.
    """
    if symbol is None:
        return ""
    return symbol.strip().upper()


def to_atomic_number(symbol: Optional[str]) -> int:
    """Return atomic number for an element symbol, or ``0`` if unknown."""
    return ELEMENT_TO_Z.get(normalize_element(symbol), 0)


def is_metal(z: int) -> bool:
    return z in METAL_ZS


def is_halogen(z: int) -> bool:
    return z in HALOGEN_ZS


def is_heteroatom(z: int) -> bool:
    return z in HETEROATOM_ZS


def element_class(z: int) -> str:
    """Coarse element class used in feature blocks.

    Returns one of: ``'C'``, ``'N'``, ``'O'``, ``'P'``, ``'S'``,
    ``'halogen'``, ``'metal_alkali'``, ``'metal_alkaline_earth'``,
    ``'metal_transition'``, ``'metal_other'``, ``'other'``.
    """
    if z == 6:
        return "C"
    if z == 7:
        return "N"
    if z == 8:
        return "O"
    if z == 15:
        return "P"
    if z == 16:
        return "S"
    if z in HALOGEN_ZS:
        return "halogen"
    if z in ALKALI_ZS:
        return "metal_alkali"
    if z in ALKALINE_EARTH_ZS:
        return "metal_alkaline_earth"
    if z in TRANSITION_ZS:
        return "metal_transition"
    if z in METAL_ZS:
        return "metal_other"
    return "other"


# ---------------------------------------------------------------------------
# Bond perception (covalent radius sum + tolerance, KD-tree filtered)
# ---------------------------------------------------------------------------

def infer_covalent_bonds(
    z_array: Sequence[int],
    coords: np.ndarray,
    tolerance: float = BOND_TOLERANCE,
    min_distance: float = 0.35,
    exclude_metals: bool = True,
) -> List[Tuple[int, int]]:
    """Infer the heavy-atom covalent connectivity from coordinates.

    Implements the same algorithm used by the GNN backend
    (:func:`ligandexplorer.utilities.gnn_data._infer_covalent_pairs`)
    but as a stand-alone function so the sandbox does not depend on the
    main repository.

    Parameters
    ----------
    z_array : sequence of int
        Atomic numbers, one per atom.
    coords : (N, 3) ndarray
        Cartesian coordinates in Angstrom.
    tolerance : float
        Extra slack added to the sum of covalent radii. ``0.45`` matches
        the GNN-side default and is permissive enough to keep
        C-Br (1.94 A), C-I (2.14 A), S-S (2.05 A).
    min_distance : float
        Pairs closer than this distance are rejected (assumed identical
        atoms / parse errors).
    exclude_metals : bool
        If True, no covalent bond is created between a metal atom and
        anything else. Metal-ligand contacts are captured separately
        via the interactions channel (not part of mol-only schema).

    Returns
    -------
    list of (i, j)
        Sorted pairs with ``i < j``.
    """
    n = len(z_array)
    if n < 2:
        return []

    coords = np.asarray(coords, dtype=float)
    if coords.shape != (n, 3):
        raise ValueError(
            f"coords shape {coords.shape} does not match {n} atoms"
        )

    tree = cKDTree(coords)
    # 2.5 A covers the longest typical covalent bond in our table
    # (S-S 2.35, C-I 2.14) plus tolerance headroom.
    candidate_pairs = tree.query_pairs(r=2.5 + tolerance)
    if not candidate_pairs:
        return []

    pairs: List[Tuple[int, int]] = []
    for i, j in candidate_pairs:
        zi = int(z_array[i])
        zj = int(z_array[j])
        if exclude_metals and (is_metal(zi) or is_metal(zj)):
            continue
        ri = COVALENT_RADII.get(zi, DEFAULT_COVALENT_RADIUS)
        rj = COVALENT_RADII.get(zj, DEFAULT_COVALENT_RADIUS)
        max_dist = ri + rj + tolerance
        d = float(np.linalg.norm(coords[i] - coords[j]))
        if min_distance <= d <= max_dist:
            pairs.append((i, j) if i < j else (j, i))
    pairs.sort()
    return pairs


# ---------------------------------------------------------------------------
# Implicit hydrogen estimation
# ---------------------------------------------------------------------------

def estimate_implicit_h(
    z: int, heavy_degree: int, formal_charge: int = 0
) -> int:
    """Estimate implicit hydrogen count from heavy-atom valence rules.

    For organic atoms (B, C, N, O, F, Si, P, S, Se, halogens) we assume
    the typical valence and subtract the heavy-atom degree and the
    formal charge:

    - Carbon valence 4 -> ``4 - heavy_degree``.
    - Neutral nitrogen valence 3 -> ``3 - heavy_degree``.
    - Positive nitrogen (formal_charge=+1) valence 4 -> ``4 - heavy_degree``.
    - Neutral oxygen valence 2 -> ``2 - heavy_degree``.
    - Negative oxygen (formal_charge=-1, carboxylate / phenolate)
      valence 1 -> ``1 - heavy_degree``.

    Returns ``0`` for elements not in :data:`TYPICAL_VALENCE` (e.g.
    metals, noble gases). Never returns a negative value.
    """
    base = TYPICAL_VALENCE.get(z, 0)
    if base == 0:
        return 0
    expected = base + formal_charge
    return max(0, expected - heavy_degree)


# ---------------------------------------------------------------------------
# Donor / acceptor labelling
# ---------------------------------------------------------------------------

def donor_acceptor_flags(
    z: int,
    heavy_degree: int,
    formal_charge: int = 0,
    is_aromatic: bool = False,
) -> Tuple[bool, bool]:
    """Return ``(is_donor, is_acceptor)`` for the heavy atom.

    Donor rule: the atom has at least one implicit H AND it is one of
    ``{N, O, S}``.
    Acceptor rule: the atom is ``{N, O, S}`` and has at least one
    "spare valence" after H placement.

    When ``is_aromatic`` is True we treat the atom as sp2 with no
    spare valence for H:
    - aromatic 2-coordinate N (pyridine-like) -> 0 H, accepts.
    - aromatic 3-coordinate N (pyrrole-like)  -> 0 H, no accept.
    - aromatic 2-coordinate O / S (furan / thiophene-like)
      -> 0 H, accepts.
    """
    el = element_class(z)
    if el not in {"N", "O", "S"}:
        return (False, False)

    if is_aromatic:
        if el == "N":
            is_donor = heavy_degree <= 2  # pyrrole NH has degree 2 *and* H
            # Heuristic: degree-2 aromatic N is pyridine-like (acceptor, 0H);
            # degree-3 aromatic N is pyrrole-like (donor, 1H, no acceptor).
            if heavy_degree == 3:
                return (True, False)
            return (False, True if formal_charge <= 0 else False)
        if el == "O":
            return (False, formal_charge <= 0)
        return (False, heavy_degree <= 2 and formal_charge <= 0)

    h_count = estimate_implicit_h(z, heavy_degree, formal_charge)
    is_donor = h_count > 0
    if el == "O":
        is_acceptor = formal_charge <= 0
    elif el == "N":
        # 4-coordinate ammonium is not an acceptor in the usual sense.
        is_acceptor = heavy_degree + h_count <= 3 and formal_charge <= 0
    else:  # S
        is_acceptor = heavy_degree <= 2 and formal_charge <= 0
    return (is_donor, is_acceptor)


# ---------------------------------------------------------------------------
# Geometric aromaticity detection
# ---------------------------------------------------------------------------

def is_ring_aromatic_geom(
    ring_indices: Sequence[int],
    coords: np.ndarray,
    plane_rmsd_threshold: float = 0.15,
    angle_tolerance_deg: float = 12.0,
) -> bool:
    """Return True if the given ring is plausibly aromatic by geometry.

    Two checks are applied:

    1. The ring atoms are co-planar within
       ``plane_rmsd_threshold`` Angstrom RMSD to their best-fit plane.
    2. The internal bond angles are close to the regular polygon
       interior angle (108 for 5-ring, 120 for 6-ring) within
       ``angle_tolerance_deg`` degrees.

    Rings of size 3, 4 or >7 always return False (no benzene/pyridine
    /furan/imidazole motif).
    """
    n = len(ring_indices)
    if n < 5 or n > 6:
        return False

    pts = np.asarray([coords[i] for i in ring_indices], dtype=float)
    centroid = pts.mean(axis=0)
    rel = pts - centroid
    # Best fit plane via SVD; smallest singular value = plane RMSD scale.
    _, sigma, _ = np.linalg.svd(rel, full_matrices=False)
    plane_rmsd = float(sigma[-1] / np.sqrt(n))
    if plane_rmsd > plane_rmsd_threshold:
        return False

    # Interior angles between consecutive ring edges, projected onto the
    # plane (after centering). For a regular n-gon the interior angle is
    # 180 * (n-2) / n.
    target_angle = 180.0 * (n - 2) / n
    angles: List[float] = []
    for i in range(n):
        a = pts[(i - 1) % n]
        b = pts[i]
        c = pts[(i + 1) % n]
        v1 = a - b
        v2 = c - b
        cos_angle = float(
            np.dot(v1, v2)
            / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        )
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angles.append(np.degrees(np.arccos(cos_angle)))

    mean_dev = float(np.mean([abs(a - target_angle) for a in angles]))
    return mean_dev <= angle_tolerance_deg


# ---------------------------------------------------------------------------
# Formal-charge rule annotations
# ---------------------------------------------------------------------------

def annotate_formal_charges(
    z_array: Sequence[int],
    heavy_degree: Sequence[int],
    neighbors: Sequence[Sequence[int]],
) -> np.ndarray:
    """Return integer formal charges per atom from simple rules.

    Recognised motifs (heavy-atom only, no H needed):

    - Carboxylate / phosphate / sulfonate terminal O attached to a
      heavy atom that itself carries 2+ other terminal-O neighbours:
      ``charge = -1`` on each terminal O.
    - Ammonium / amine quaternary nitrogen: ``N`` with heavy degree 4
      -> ``charge = +1``.
    - Guanidinium central C with 3 N neighbours where at least 2 of
      those N have heavy degree <= 1: the terminal N's each get +1/2,
      rounded by attribution to one terminal N marked ``+1``.

    The output ndarray has length ``len(z_array)`` and dtype int8.
    """
    n = len(z_array)
    charges = np.zeros(n, dtype=np.int8)
    for i in range(n):
        z = int(z_array[i])
        if z == 7 and heavy_degree[i] == 4:
            charges[i] = 1
            continue
        # Carboxylate / phosphate / sulfonate detection from terminal O:
        if z == 8 and heavy_degree[i] == 1:
            for nb in neighbors[i]:
                if int(z_array[nb]) in {6, 15, 16}:  # C/P/S center
                    # count other terminal-O neighbours of the center
                    other_term_o = 0
                    for nb2 in neighbors[nb]:
                        if (
                            nb2 != i
                            and int(z_array[nb2]) == 8
                            and heavy_degree[nb2] == 1
                        ):
                            other_term_o += 1
                    if other_term_o >= 1:
                        charges[i] = -1
                        break
    # Guanidinium: central C with three N neighbours, at least two with
    # heavy degree 1. Mark one of the N as +1 (canonical resonance).
    for i in range(n):
        if int(z_array[i]) != 6:
            continue
        nb_ns = [nb for nb in neighbors[i] if int(z_array[nb]) == 7]
        if len(nb_ns) < 3:
            continue
        terminal_ns = [nb for nb in nb_ns if heavy_degree[nb] == 1]
        if len(terminal_ns) >= 2:
            charges[terminal_ns[0]] = 1
    return charges
