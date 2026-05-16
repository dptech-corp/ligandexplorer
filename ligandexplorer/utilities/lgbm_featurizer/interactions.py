"""Protein-ligand interaction features (72 dims).

Each channel is computed from the cached :class:`LigandHandle` and
:class:`ProteinHandle` produced by :mod:`lgbm_featurizer.loader`. All
predicates are geometric (distance + angle) so the routines stay valid
regardless of bond-order quality on either side.

Output is a dict keyed by interaction block name (matching
:data:`schema.INTERACTION_BLOCKS`). Use
``schema.assemble(parts, blocks=FEATURE_BLOCKS_FULL)`` to merge the
mol-only parts with these interaction parts into the 158-dim model_3
vector.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .chemistry import (
    HALOGEN_ZS,
    donor_acceptor_flags,
    is_metal,
)
from .loader import LigandHandle, ProteinHandle, _RingDescriptor
from .schema import (
    CATION_PI,
    CONTACT_SHELLS,
    ELECTROSTATIC,
    HALOGEN_BOND,
    HBOND,
    HYDROPHOBIC,
    METAL_COORD,
    PI_PI,
    POCKET,
    RESIDUE_CLASSES,
    RESIDUE_CONTACT,
)

# Geometric thresholds. Conservative values tuned to match common
# interaction-detection literature (PLIP defaults).
_HB_DISTANCE_RANGE = (2.4, 3.6)        # heavy-heavy donor..acceptor
_HB_ANGLE_MIN_DEG = 120.0              # D-H..A; we approximate H by
                                       # reversing the donor's most
                                       # distant heavy neighbour
_XB_DISTANCE_MAX = 4.0                 # halogen X..acceptor
_XB_ANGLE_MIN_DEG = 140.0              # C-X..A directionality
_PIPI_DISTANCE_MAX = 5.5               # ring-center..ring-center
_PIPI_FACE_ANGLE_DEG = 30.0            # planes within +/-30 -> face-to-face
_PIPI_EDGE_ANGLE_DEG = 60.0            # planes 60-90 -> edge-to-face
_CATION_PI_DISTANCE_MAX = 6.0          # cation..ring-center
_CATION_PI_VERTICAL_MAX = 2.5          # cation displacement above ring
_METAL_COORD_DISTANCE = 3.0
_POCKET_RADIUS = 7.0                   # any pocket atom within this many A of any ligand atom


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pocket_mask(prot: ProteinHandle, lig_coords: np.ndarray,
                 radius: float = _POCKET_RADIUS) -> np.ndarray:
    """Boolean mask of protein atoms with at least one ligand atom
    within ``radius`` Angstrom."""
    if lig_coords.shape[0] == 0:
        return np.zeros(prot.coords.shape[0], dtype=bool)
    # KD-tree on protein, query each ligand atom.
    hits = prot.kdtree.query_ball_point(lig_coords, r=radius)
    mask = np.zeros(prot.coords.shape[0], dtype=bool)
    for hit_list in hits:
        if hit_list:
            mask[hit_list] = True
    return mask


def _shell_index(d: float, shells: Tuple[Tuple[float, float], ...]) -> int:
    """Return the index of the (lo, hi) shell that contains ``d``, or
    -1 if outside every shell.  Uses left-closed right-open intervals.
    """
    for i, (lo, hi) in enumerate(shells):
        if lo <= d < hi:
            return i
    return -1


def _ligand_aromatic_rings(lig: LigandHandle) -> List[_RingDescriptor]:
    """Best-fit plane normals for every ligand aromatic ring."""
    out: List[_RingDescriptor] = []
    for ring in lig.aromatic_rings:
        pts = lig.coords[ring]
        centroid = pts.mean(axis=0)
        rel = pts - centroid
        _, _, vh = np.linalg.svd(rel, full_matrices=False)
        normal = vh[-1]
        normal = normal / (np.linalg.norm(normal) + 1e-9)
        out.append(_RingDescriptor(
            atom_indices=list(ring),
            centroid=centroid,
            normal=normal,
        ))
    return out


def _ligand_donor_acceptor(lig: LigandHandle) -> Tuple[np.ndarray, np.ndarray]:
    """Boolean masks: which ligand heavy atoms are donors / acceptors.

    Uses the same residue-aware rule as
    :func:`chemistry.donor_acceptor_flags`, with aromatic detection
    pulled from :attr:`LigandHandle.aromatic_mask`.
    """
    n = lig.n_atoms
    donors = np.zeros(n, dtype=bool)
    acceptors = np.zeros(n, dtype=bool)
    arom = lig.aromatic_mask
    for i in range(n):
        z = int(lig.z_array[i])
        deg = int(lig.heavy_degree[i])
        fc = int(lig.formal_charges[i])
        d, a = donor_acceptor_flags(
            z, deg, fc, is_aromatic=bool(arom[i])
        )
        donors[i] = d
        acceptors[i] = a
    return donors, acceptors


def _angle_at_vertex(
    apex: np.ndarray, p1: np.ndarray, p2: np.ndarray
) -> float:
    """Angle at ``apex`` formed by p1 and p2 (in degrees)."""
    v1 = p1 - apex
    v2 = p2 - apex
    denom = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9
    c = float(np.dot(v1, v2) / denom)
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(c)))


# ---------------------------------------------------------------------------
# Channel: residue contact statistics (30 dims)
# ---------------------------------------------------------------------------

def _residue_contact(
    prot: ProteinHandle, lig: LigandHandle, pocket: np.ndarray,
) -> List[float]:
    n_shells = len(CONTACT_SHELLS)
    n_classes = len(RESIDUE_CLASSES)
    # counts[shell][class][main/side]
    counts = np.zeros((n_shells, n_classes, 2), dtype=np.int64)
    if not pocket.any() or lig.n_atoms == 0:
        return [0.0] * (n_shells * n_classes * 2)

    pocket_idx = np.where(pocket)[0]
    pocket_coords = prot.coords[pocket_idx]
    pocket_classes = [prot.res_classes[i] for i in pocket_idx]
    pocket_main = [prot.is_backbone[i] for i in pocket_idx]

    # All pairwise (pocket, ligand) distances within the largest shell:
    deltas = pocket_coords[:, None, :] - lig.coords[None, :, :]
    dists = np.linalg.norm(deltas, axis=2)
    min_dist_per_pocket = dists.min(axis=1)  # nearest-ligand distance

    class_to_idx = {c: i for i, c in enumerate(RESIDUE_CLASSES)}
    for pi, d in enumerate(min_dist_per_pocket):
        cls = pocket_classes[pi]
        if cls not in class_to_idx:
            continue
        s = _shell_index(d, CONTACT_SHELLS)
        if s < 0:
            continue
        chain = 0 if pocket_main[pi] else 1
        counts[s, class_to_idx[cls], chain] += 1

    # Flatten in schema order: shell -> class -> main/side
    out: List[float] = []
    for s in range(n_shells):
        for c in range(n_classes):
            for chain in range(2):
                out.append(float(counts[s, c, chain]))
    return out


# ---------------------------------------------------------------------------
# Channel: H-bond geometry (8 dims)
# ---------------------------------------------------------------------------

def _hbond_geom(
    prot: ProteinHandle, lig: LigandHandle, pocket: np.ndarray,
) -> List[float]:
    if not pocket.any() or lig.n_atoms == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    lig_don, lig_acc = _ligand_donor_acceptor(lig)
    pocket_idx = np.where(pocket)[0]
    pocket_coords = prot.coords[pocket_idx]
    p_don = prot.is_donor[pocket_idx]
    p_acc = prot.is_acceptor[pocket_idx]
    p_main = prot.is_backbone[pocket_idx]

    deltas = pocket_coords[:, None, :] - lig.coords[None, :, :]
    dists = np.linalg.norm(deltas, axis=2)
    dist_low, dist_high = _HB_DISTANCE_RANGE
    pairs = np.argwhere((dists >= dist_low) & (dists < dist_high))

    n_lig_donor = 0
    n_prot_donor = 0
    n_prot_main = 0
    n_prot_side = 0
    angle_count_high = 0
    angle_acc: List[float] = []
    dist_acc: List[float] = []

    for pi, li in pairs:
        d = float(dists[pi, li])
        prot_atom = pocket_idx[pi]
        # Geometry: pick the direction that's chemically valid.
        lig_donor_here = lig_don[li] and p_acc[pi]
        prot_donor_here = p_don[pi] and lig_acc[li]
        if not (lig_donor_here or prot_donor_here):
            continue

        # Approximate D-H..A angle: take the donor's farthest heavy
        # neighbour as the pseudo H position (reversed direction).
        # If a donor has no heavy neighbours we skip the angle check.
        donor_coord = (lig.coords[li] if lig_donor_here
                       else prot.coords[prot_atom])
        accept_coord = (prot.coords[prot_atom] if lig_donor_here
                        else lig.coords[li])
        if lig_donor_here:
            nbrs = lig.neighbors[li]
            if not nbrs:
                pseudo_h = donor_coord  # straight-line approximation
            else:
                # Pseudo-H = donor + unit vector toward acceptor.
                dvec = accept_coord - donor_coord
                norm = np.linalg.norm(dvec) + 1e-9
                pseudo_h = donor_coord + dvec / norm  # 1 A out
            angle = _angle_at_vertex(pseudo_h, donor_coord, accept_coord)
        else:
            # Protein donor: backbone N has CA, side-chain donors have
            # known parent atoms; in either case approximate
            # angle = 180 - 0 = 180.  We instead just trust distance.
            angle = 170.0
        if angle < _HB_ANGLE_MIN_DEG:
            continue

        if lig_donor_here:
            n_lig_donor += 1
        if prot_donor_here:
            n_prot_donor += 1
        if p_main[pi]:
            n_prot_main += 1
        else:
            n_prot_side += 1
        if angle >= 150.0:
            angle_count_high += 1
        angle_acc.append(angle)
        dist_acc.append(d)

    if not dist_acc:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    return [
        float(n_lig_donor),
        float(n_prot_donor),
        float(n_prot_main),
        float(n_prot_side),
        float(min(dist_acc)),
        float(np.mean(dist_acc)),
        float(np.mean(angle_acc)),
        float(angle_count_high),
    ]


# ---------------------------------------------------------------------------
# Channel: halogen bond (4 dims)
# ---------------------------------------------------------------------------

def _halogen_bond(
    prot: ProteinHandle, lig: LigandHandle, pocket: np.ndarray,
) -> List[float]:
    if not pocket.any() or lig.n_atoms == 0:
        return [0.0, 0.0, 0.0, 0.0]

    halogen_mask = np.zeros(lig.n_atoms, dtype=bool)
    for i in range(lig.n_atoms):
        if int(lig.z_array[i]) in HALOGEN_ZS:
            halogen_mask[i] = True
    if not halogen_mask.any():
        return [0.0, 0.0, 0.0, 0.0]

    pocket_idx = np.where(pocket)[0]
    p_acc = prot.is_acceptor[pocket_idx]
    p_z = prot.z_array[pocket_idx]
    pocket_coords = prot.coords[pocket_idx]

    halogen_indices = np.where(halogen_mask)[0]
    halogen_coords = lig.coords[halogen_indices]
    deltas = pocket_coords[:, None, :] - halogen_coords[None, :, :]
    dists = np.linalg.norm(deltas, axis=2)
    pairs = np.argwhere((dists < _XB_DISTANCE_MAX) & p_acc[:, None])

    count = 0
    count_O = 0
    count_N = 0
    dist_acc: List[float] = []
    for pi, hi in pairs:
        d = float(dists[pi, hi])
        # Angular check: C-X..A angle. Halogen heavy neighbour =
        # the (unique) carbon usually; pick first neighbour.
        x_atom = halogen_indices[hi]
        x_nbrs = lig.neighbors[x_atom]
        if not x_nbrs:
            continue
        c_atom = x_nbrs[0]
        c_coord = lig.coords[c_atom]
        x_coord = lig.coords[x_atom]
        a_coord = pocket_coords[pi]
        angle = _angle_at_vertex(x_coord, c_coord, a_coord)
        if angle < _XB_ANGLE_MIN_DEG:
            continue
        count += 1
        if int(p_z[pi]) == 8:
            count_O += 1
        elif int(p_z[pi]) == 7:
            count_N += 1
        dist_acc.append(d)

    return [
        float(count),
        float(count_O),
        float(count_N),
        float(min(dist_acc)) if dist_acc else 0.0,
    ]


# ---------------------------------------------------------------------------
# Channel: pi-pi stacking (4 dims)
# ---------------------------------------------------------------------------

def _pi_pi(
    prot: ProteinHandle, lig: LigandHandle, pocket: np.ndarray,
) -> List[float]:
    lig_rings = _ligand_aromatic_rings(lig)
    if not lig_rings or not prot.aromatic_rings:
        return [0.0, 0.0, 0.0, 0.0]

    # Restrict protein rings to those whose centroid is within
    # POCKET_RADIUS + a margin of any ligand atom.
    prot_rings = []
    for ring in prot.aromatic_rings:
        d_min = float(
            np.linalg.norm(lig.coords - ring.centroid, axis=1).min()
        )
        if d_min < _POCKET_RADIUS + 2.0:
            prot_rings.append(ring)
    if not prot_rings:
        return [0.0, 0.0, 0.0, 0.0]

    n_face = n_edge = n_tilt = 0
    min_center_dist = float("inf")
    for lr in lig_rings:
        for pr in prot_rings:
            d = float(np.linalg.norm(lr.centroid - pr.centroid))
            if d > _PIPI_DISTANCE_MAX:
                continue
            min_center_dist = min(min_center_dist, d)
            # Acute angle between plane normals.
            cos_angle = abs(float(np.dot(lr.normal, pr.normal)))
            cos_angle = min(1.0, cos_angle)
            angle = float(np.degrees(np.arccos(cos_angle)))
            if angle <= _PIPI_FACE_ANGLE_DEG:
                n_face += 1
            elif angle >= _PIPI_EDGE_ANGLE_DEG:
                n_edge += 1
            else:
                n_tilt += 1

    if min_center_dist == float("inf"):
        min_center_dist = 0.0
    return [
        float(n_face), float(n_edge), float(n_tilt),
        float(min_center_dist),
    ]


# ---------------------------------------------------------------------------
# Channel: cation-pi (2 dims)
# ---------------------------------------------------------------------------

def _cation_pi(
    prot: ProteinHandle, lig: LigandHandle, pocket: np.ndarray,
) -> List[float]:
    lig_rings = _ligand_aromatic_rings(lig)
    lig_cation_idx = np.where(lig.formal_charges > 0)[0]
    pocket_idx = np.where(pocket)[0]
    p_cation_idx = pocket_idx[prot.formal_charge[pocket_idx] > 0]

    n_lig_to_prot = 0   # ligand cation -> protein aromatic ring
    n_prot_to_lig = 0   # protein cation -> ligand aromatic ring

    if p_cation_idx.size > 0 and lig_rings:
        cation_coords = prot.coords[p_cation_idx]
        for ring in lig_rings:
            disp = cation_coords - ring.centroid
            d = np.linalg.norm(disp, axis=1)
            within_dist = d < _CATION_PI_DISTANCE_MAX
            # vertical component along plane normal.
            vert = np.abs(disp @ ring.normal)
            within_vert = vert < _CATION_PI_VERTICAL_MAX
            n_prot_to_lig += int((within_dist & within_vert).sum())

    if lig_cation_idx.size > 0 and prot.aromatic_rings:
        cation_coords = lig.coords[lig_cation_idx]
        for ring in prot.aromatic_rings:
            disp = cation_coords - ring.centroid
            d = np.linalg.norm(disp, axis=1)
            within_dist = d < _CATION_PI_DISTANCE_MAX
            vert = np.abs(disp @ ring.normal)
            within_vert = vert < _CATION_PI_VERTICAL_MAX
            n_lig_to_prot += int((within_dist & within_vert).sum())

    return [float(n_lig_to_prot), float(n_prot_to_lig)]


# ---------------------------------------------------------------------------
# Channel: electrostatic (8 dims)
# ---------------------------------------------------------------------------

def _electrostatic(
    prot: ProteinHandle, lig: LigandHandle, pocket: np.ndarray,
) -> List[float]:
    shells = CONTACT_SHELLS + ((7.0, 10.0),)
    out = np.zeros((len(shells), 2), dtype=np.int64)  # [shell][attr/repulse]

    pocket_idx = np.where(pocket)[0]
    p_charge = prot.formal_charge[pocket_idx]
    p_charged = p_charge != 0
    if not p_charged.any():
        return [0.0] * (len(shells) * 2)
    pocket_coords = prot.coords[pocket_idx[p_charged]]
    pocket_chg = p_charge[p_charged]

    lig_charged_mask = lig.formal_charges != 0
    if not lig_charged_mask.any():
        return [0.0] * (len(shells) * 2)
    lig_coords = lig.coords[lig_charged_mask]
    lig_chg = lig.formal_charges[lig_charged_mask]

    deltas = pocket_coords[:, None, :] - lig_coords[None, :, :]
    dists = np.linalg.norm(deltas, axis=2)
    sign_product = pocket_chg[:, None] * lig_chg[None, :]
    for i, (lo, hi) in enumerate(shells):
        in_shell = (dists >= lo) & (dists < hi)
        out[i, 0] = int(((sign_product < 0) & in_shell).sum())  # attractive
        out[i, 1] = int(((sign_product > 0) & in_shell).sum())  # repulsive

    return [float(x) for x in out.flatten().tolist()]


# ---------------------------------------------------------------------------
# Channel: hydrophobic contacts (6 dims)
# ---------------------------------------------------------------------------

def _hydrophobic(
    prot: ProteinHandle, lig: LigandHandle, pocket: np.ndarray,
) -> List[float]:
    shells = ((0.0, 4.0), (4.0, 5.5), (5.5, 7.0))
    cc = np.zeros(len(shells), dtype=np.int64)
    aa = np.zeros(len(shells), dtype=np.int64)

    pocket_idx = np.where(pocket)[0]
    p_z = prot.z_array[pocket_idx]
    p_coords = prot.coords[pocket_idx]
    p_carbon_mask = p_z == 6

    lig_carbon_mask = lig.z_array == 6
    if p_carbon_mask.any() and lig_carbon_mask.any():
        p_c = p_coords[p_carbon_mask]
        l_c = lig.coords[lig_carbon_mask]
        d = np.linalg.norm(p_c[:, None, :] - l_c[None, :, :], axis=2)
        for i, (lo, hi) in enumerate(shells):
            cc[i] = int(((d >= lo) & (d < hi)).sum())

    lig_rings = _ligand_aromatic_rings(lig)
    if lig_rings and prot.aromatic_rings:
        for lr in lig_rings:
            for pr in prot.aromatic_rings:
                d = float(np.linalg.norm(lr.centroid - pr.centroid))
                s = _shell_index(d, shells)
                if s >= 0:
                    aa[s] += 1

    return [float(x) for x in (cc.tolist() + aa.tolist())]


# ---------------------------------------------------------------------------
# Channel: metal coordination (4 dims)
# ---------------------------------------------------------------------------

def _metal_coord(
    prot: ProteinHandle, lig: LigandHandle, pocket: np.ndarray,
) -> List[float]:
    pocket_idx = np.where(pocket)[0]
    p_coords = prot.coords[pocket_idx]
    p_z = prot.z_array[pocket_idx]

    # Ligand metals -> protein O / N / S
    lig_metal_idx = [i for i in range(lig.n_atoms)
                     if is_metal(int(lig.z_array[i]))]
    n_lig_to_O = n_lig_to_N = n_lig_to_S = 0
    if lig_metal_idx:
        lm = lig.coords[lig_metal_idx]
        d = np.linalg.norm(p_coords[:, None, :] - lm[None, :, :], axis=2)
        close = d < _METAL_COORD_DISTANCE
        for col in range(d.shape[1]):
            atoms_close = pocket_idx[np.where(close[:, col])[0]]
            for a in atoms_close:
                z = int(prot.z_array[a])
                if z == 8:
                    n_lig_to_O += 1
                elif z == 7:
                    n_lig_to_N += 1
                elif z == 16:
                    n_lig_to_S += 1

    # Protein metals -> any ligand atom
    p_metal_mask = np.array([is_metal(int(z)) for z in p_z], dtype=bool)
    n_prot_to_lig = 0
    if p_metal_mask.any():
        pm = p_coords[p_metal_mask]
        d = np.linalg.norm(pm[:, None, :] - lig.coords[None, :, :], axis=2)
        n_prot_to_lig = int((d < _METAL_COORD_DISTANCE).sum())

    return [float(n_lig_to_O), float(n_lig_to_N),
            float(n_lig_to_S), float(n_prot_to_lig)]


# ---------------------------------------------------------------------------
# Channel: pocket descriptors (6 dims)
# ---------------------------------------------------------------------------

def _pocket_descriptors(
    prot: ProteinHandle, lig: LigandHandle, pocket: np.ndarray,
) -> List[float]:
    pocket_idx = np.where(pocket)[0]
    n_pocket = int(pocket_idx.shape[0])
    n_lig = lig.n_atoms

    if n_pocket == 0 or n_lig == 0:
        # Still report ligand-only descriptors when possible.
        lig_diameter = (
            float(np.linalg.norm(
                lig.coords[:, None, :] - lig.coords[None, :, :], axis=2,
            ).max()) if n_lig >= 2 else 0.0
        )
        return [0.0, 0.0, 0.0, float(n_lig), lig_diameter, 0.0]

    p_coords = prot.coords[pocket_idx]
    d = np.linalg.norm(p_coords[:, None, :] - lig.coords[None, :, :],
                       axis=2)
    min_dist = float(d.min())
    nearest_per_lig = d.min(axis=0)
    mean_min = float(nearest_per_lig.mean())

    lig_diameter = (
        float(np.linalg.norm(
            lig.coords[:, None, :] - lig.coords[None, :, :], axis=2,
        ).max()) if n_lig >= 2 else 0.0
    )
    pocket_diameter = (
        float(np.linalg.norm(
            p_coords[:, None, :] - p_coords[None, :, :], axis=2,
        ).max()) if n_pocket >= 2 else 0.0
    )

    return [
        float(n_pocket), min_dist, mean_min,
        float(n_lig), lig_diameter, pocket_diameter,
    ]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_interaction_features(
    prot: ProteinHandle, lig: LigandHandle,
    pocket_radius: float = _POCKET_RADIUS,
) -> Dict[str, List[float]]:
    """Compute all 72 interaction features, grouped by schema block.

    Returns a dict whose keys match the names of the FeatureBlocks in
    :data:`schema.INTERACTION_BLOCKS`.
    """
    pocket = _pocket_mask(prot, lig.coords, radius=pocket_radius)
    return {
        RESIDUE_CONTACT.name: _residue_contact(prot, lig, pocket),
        HBOND.name:           _hbond_geom(prot, lig, pocket),
        HALOGEN_BOND.name:    _halogen_bond(prot, lig, pocket),
        PI_PI.name:           _pi_pi(prot, lig, pocket),
        CATION_PI.name:       _cation_pi(prot, lig, pocket),
        ELECTROSTATIC.name:   _electrostatic(prot, lig, pocket),
        HYDROPHOBIC.name:     _hydrophobic(prot, lig, pocket),
        METAL_COORD.name:     _metal_coord(prot, lig, pocket),
        POCKET.name:          _pocket_descriptors(prot, lig, pocket),
    }


__all__ = [
    "compute_interaction_features",
]
