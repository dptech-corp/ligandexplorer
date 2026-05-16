"""PDB loading and ``LigandHandle`` caching.

Loads a ligand PDB / CIF into:

- a cleaned BioPython structure (only first model, only main altloc, no
  hydrogens),
- a numpy coordinate array (N, 3),
- a NetworkX heavy-atom connectivity graph (single bonds inferred via
  :func:`lgbm_featurizer.chemistry.infer_covalent_bonds`),
- a ring list from ``networkx.cycle_basis``,
- a single-bond RDKit ``RWMol`` for SMARTS pattern matching,
- formal charges from rule annotation.

All of these are cached on a single :class:`LigandHandle` dataclass so
the downstream feature modules can pull them without re-computing.

This module is the only place in the package that touches BioPython
and RDKit construction. Everything else operates on the cached arrays
/ graph / mol.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from .chemistry import (
    annotate_formal_charges,
    infer_covalent_bonds,
    is_ring_aromatic_geom,
    normalize_element,
    to_atomic_number,
)

# Suppress noisy BioPython warnings (PDBConstructionWarning,
# DiscontinuousChainWarning, etc.).
try:
    from Bio import BiopythonWarning
    warnings.simplefilter("ignore", BiopythonWarning)
except ImportError:  # pragma: no cover - biopython always available here
    pass

from Bio.PDB import MMCIFParser, PDBParser  # noqa: E402

from rdkit import Chem  # noqa: E402
from rdkit import RDLogger  # noqa: E402

# Silence RDKit's own warning stream; we deliberately bypass perception.
RDLogger.DisableLog("rdApp.*")


def _select_parser(path: str):
    lower = path.lower()
    if lower.endswith(".cif") or lower.endswith(".mmcif"):
        return MMCIFParser(QUIET=True)
    return PDBParser(QUIET=True)


# ---------------------------------------------------------------------------
# Structure cleanup
# ---------------------------------------------------------------------------

def _iter_clean_atoms(structure):
    """Yield ``(element_z, x, y, z, resname)`` from the first model only,
    skipping hydrogens and non-primary altloc atoms.

    Primary altloc is the empty/space altloc, or 'A' if the empty
    altloc has been displaced.
    """
    first_model = next(iter(structure), None)
    if first_model is None:
        return

    for chain in first_model:
        for residue in chain:
            resname = residue.get_resname().strip().upper()
            # Track which atom names we've already emitted at this
            # residue, so secondary altlocs of the same atom name are
            # discarded.
            seen_names: set = set()
            for atom in residue:
                element = normalize_element(getattr(atom, "element", None))
                if element == "H":
                    continue
                altloc = atom.get_altloc()
                if altloc not in ("", " ", "A"):
                    continue
                name = atom.get_name().strip()
                key = (chain.id, residue.id, name)
                if key in seen_names:
                    continue
                seen_names.add(key)

                z = to_atomic_number(element)
                if z == 0:
                    # Fall back to inferring from the atom name's
                    # leading alpha characters (handles cases where the
                    # element column is blank).
                    raw = atom.get_name().strip()
                    stripped = raw.lstrip("0123456789").rstrip(
                        "0123456789'\""
                    )
                    z = to_atomic_number(stripped[:2]) or to_atomic_number(
                        stripped[:1]
                    )
                if z == 0:
                    continue

                coord = atom.get_coord()
                yield z, float(coord[0]), float(coord[1]), float(
                    coord[2]
                ), resname


# ---------------------------------------------------------------------------
# LigandHandle
# ---------------------------------------------------------------------------

@dataclass
class LigandHandle:
    """All cached views of a single ligand needed for feature extraction."""

    path: str
    z_array: np.ndarray              # (N,) int32 atomic numbers
    coords: np.ndarray               # (N, 3) float64
    resnames: List[str]              # length N, residue 3-letter codes
    bonds: List[Tuple[int, int]]     # sorted, i < j
    graph: nx.Graph                  # heavy-atom connectivity
    rings: List[List[int]]           # cycle_basis output
    formal_charges: np.ndarray       # (N,) int8
    rdmol: Chem.Mol                  # single-bond RWMol, sanitize=False

    # Lazy caches.
    _heavy_degree: Optional[np.ndarray] = field(default=None, repr=False)
    _neighbors: Optional[List[List[int]]] = field(default=None, repr=False)
    _aromatic_mask: Optional[np.ndarray] = field(default=None, repr=False)
    _aromatic_rings: Optional[List[List[int]]] = field(default=None,
                                                       repr=False)

    @property
    def n_atoms(self) -> int:
        return int(self.z_array.shape[0])

    @property
    def heavy_degree(self) -> np.ndarray:
        if self._heavy_degree is None:
            arr = np.zeros(self.n_atoms, dtype=np.int32)
            for i in range(self.n_atoms):
                arr[i] = self.graph.degree(i) if i in self.graph else 0
            self._heavy_degree = arr
        return self._heavy_degree

    @property
    def neighbors(self) -> List[List[int]]:
        if self._neighbors is None:
            out: List[List[int]] = []
            for i in range(self.n_atoms):
                if i in self.graph:
                    out.append(sorted(self.graph.neighbors(i)))
                else:
                    out.append([])
            self._neighbors = out
        return self._neighbors

    @property
    def aromatic_rings(self) -> List[List[int]]:
        """List of rings that pass the geometric aromaticity test.

        Computed lazily from :data:`rings` and :data:`coords`.
        """
        if self._aromatic_rings is None:
            out: List[List[int]] = []
            for ring in self.rings:
                if is_ring_aromatic_geom(ring, self.coords):
                    out.append(list(ring))
            self._aromatic_rings = out
        return self._aromatic_rings

    @property
    def aromatic_mask(self) -> np.ndarray:
        """Boolean mask of atoms that participate in any aromatic ring."""
        if self._aromatic_mask is None:
            mask = np.zeros(self.n_atoms, dtype=bool)
            for ring in self.aromatic_rings:
                for idx in ring:
                    mask[idx] = True
            self._aromatic_mask = mask
        return self._aromatic_mask


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def load_ligand(path: str) -> LigandHandle:
    """Parse a ligand PDB / CIF and return a populated :class:`LigandHandle`.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist (delegated to BioPython).
    ValueError
        If the file contains zero heavy atoms.
    """
    parser = _select_parser(path)
    structure = parser.get_structure("lig", path)

    z_list: List[int] = []
    coord_list: List[Tuple[float, float, float]] = []
    resnames: List[str] = []
    for z, x, y, zc, resname in _iter_clean_atoms(structure):
        z_list.append(z)
        coord_list.append((x, y, zc))
        resnames.append(resname)

    if not z_list:
        raise ValueError(f"no heavy atoms parsed from {path!r}")

    z_array = np.asarray(z_list, dtype=np.int32)
    coords = np.asarray(coord_list, dtype=np.float64)
    bonds = infer_covalent_bonds(z_array.tolist(), coords)

    graph = nx.Graph()
    for i in range(len(z_list)):
        graph.add_node(i, z=int(z_array[i]))
    for i, j in bonds:
        graph.add_edge(i, j)

    rings = [list(cycle) for cycle in nx.cycle_basis(graph)]

    heavy_degree = [graph.degree(i) for i in range(len(z_list))]
    neighbor_lists = [sorted(graph.neighbors(i)) for i in range(len(z_list))]
    formal_charges = annotate_formal_charges(
        z_array.tolist(), heavy_degree, neighbor_lists
    )

    rdmol = _build_single_bond_rdmol(z_array, coords, bonds)

    return LigandHandle(
        path=path,
        z_array=z_array,
        coords=coords,
        resnames=resnames,
        bonds=bonds,
        graph=graph,
        rings=rings,
        formal_charges=formal_charges,
        rdmol=rdmol,
    )


# ---------------------------------------------------------------------------
# Single-bond RDKit Mol construction (no perception, no sanitize)
# ---------------------------------------------------------------------------

def _build_single_bond_rdmol(
    z_array: np.ndarray,
    coords: np.ndarray,
    bonds: Sequence[Tuple[int, int]],
) -> Chem.Mol:
    """Build an RDKit ``Mol`` from atomic numbers + coords + bonds.

    Each bond is set to ``BondType.SINGLE`` regardless of actual bond
    order (we intentionally refuse to guess). ``SetNoImplicit(True)``
    on every atom disables RDKit's implicit-H bookkeeping so SMARTS
    matching doesn't accidentally consider phantom H's. The Mol is not
    sanitized; aromaticity / kekulization / valence checks are skipped.

    The 3D conformer is attached so descriptors that need coordinates
    (3D shape, bond-length histogram) can read them directly.
    """
    mol = Chem.RWMol()
    atom_indices: List[int] = []
    for z in z_array.tolist():
        atom = Chem.Atom(int(z))
        atom.SetNoImplicit(True)
        atom_indices.append(mol.AddAtom(atom))

    for i, j in bonds:
        mol.AddBond(int(atom_indices[i]), int(atom_indices[j]),
                    Chem.BondType.SINGLE)

    conf = Chem.Conformer(len(atom_indices))
    for idx, (x, y, z) in enumerate(coords.tolist()):
        conf.SetAtomPosition(idx, (float(x), float(y), float(z)))
    mol.AddConformer(conf, assignId=True)

    final = mol.GetMol()
    # Populate ring info without running full sanitize / aromaticity
    # perception. SMARTS operators like ``!@`` (ring-membership) need
    # RingInfo to be initialized, otherwise GetSubstructMatches raises
    # "RingInfo not initialized". Symmetric SSSR is purely topological
    # and does not touch bond orders / aromaticity.
    Chem.GetSymmSSSR(final)
    return final


# ---------------------------------------------------------------------------
# Mini in-memory PDB writer for tests / fixtures
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Protein handle (for protein-ligand interaction features)
# ---------------------------------------------------------------------------

# Backbone atom names (standard amino acid main-chain).
_BACKBONE_NAMES: FrozenSet[str] = frozenset({"N", "CA", "C", "O", "OXT"})

# Per-residue donor / acceptor maps. These are the well-known
# heavy-atom positions on standard amino-acid side chains that act as
# H-bond donors or acceptors. (Backbone N is always donor; backbone O is
# always acceptor; we add those separately below.)
_SIDE_CHAIN_DONORS: Dict[str, FrozenSet[str]] = {
    "ARG": frozenset({"NE", "NH1", "NH2"}),
    "ASN": frozenset({"ND2"}),
    "GLN": frozenset({"NE2"}),
    "HIS": frozenset({"ND1", "NE2"}),     # either tautomer
    "LYS": frozenset({"NZ"}),
    "SER": frozenset({"OG"}),
    "THR": frozenset({"OG1"}),
    "TRP": frozenset({"NE1"}),
    "TYR": frozenset({"OH"}),
    "CYS": frozenset({"SG"}),
}
_SIDE_CHAIN_ACCEPTORS: Dict[str, FrozenSet[str]] = {
    "ASN": frozenset({"OD1"}),
    "ASP": frozenset({"OD1", "OD2"}),
    "GLN": frozenset({"OE1"}),
    "GLU": frozenset({"OE1", "OE2"}),
    "HIS": frozenset({"ND1", "NE2"}),
    "SER": frozenset({"OG"}),
    "THR": frozenset({"OG1"}),
    "TYR": frozenset({"OH"}),
    "MET": frozenset({"SD"}),
}

# Residues that carry a formal charge on their side chain.
_POSITIVE_RES_ATOMS: Dict[str, FrozenSet[str]] = {
    "LYS": frozenset({"NZ"}),
    "ARG": frozenset({"NH1", "NH2", "NE", "CZ"}),
    "HIS": frozenset({"ND1", "NE2"}),  # treated as half-positive
}
_NEGATIVE_RES_ATOMS: Dict[str, FrozenSet[str]] = {
    "ASP": frozenset({"OD1", "OD2"}),
    "GLU": frozenset({"OE1", "OE2"}),
}

# Aromatic ring atoms per residue (used for pi-pi and cation-pi).
_AROMATIC_RING_ATOMS: Dict[str, Tuple[Tuple[str, ...], ...]] = {
    "PHE": (("CG", "CD1", "CE1", "CZ", "CE2", "CD2"),),
    "TYR": (("CG", "CD1", "CE1", "CZ", "CE2", "CD2"),),
    "TRP": (
        ("CG", "CD1", "NE1", "CE2", "CD2"),                 # pyrrole
        ("CD2", "CE2", "CE3", "CZ2", "CH2", "CZ3"),         # benzo
    ),
    "HIS": (("CG", "ND1", "CE1", "NE2", "CD2"),),
}

# 5-class residue grouping for contact statistics.
_RESCLASS = {
    # hydrophobic includes GLY/PRO/CYS so the model still sees them.
    "ALA": "hydrophobic", "VAL": "hydrophobic", "LEU": "hydrophobic",
    "ILE": "hydrophobic", "MET": "hydrophobic", "PRO": "hydrophobic",
    "GLY": "hydrophobic", "CYS": "hydrophobic",
    "SER": "polar", "THR": "polar", "ASN": "polar", "GLN": "polar",
    "LYS": "positive", "ARG": "positive", "HIS": "positive",
    "ASP": "negative", "GLU": "negative",
    "PHE": "aromatic", "TYR": "aromatic", "TRP": "aromatic",
}


@dataclass
class _RingDescriptor:
    """A protein aromatic ring identified for pi-pi / cation-pi."""
    atom_indices: List[int]
    centroid: np.ndarray     # (3,)
    normal: np.ndarray       # (3,) unit vector


@dataclass
class ProteinHandle:
    """Pre-processed protein structure ready for interaction queries.

    Holds only heavy atoms, with per-atom flags pre-computed so the
    interaction module can do its work as vectorised KD-tree queries.
    The structure is **not pocket-restricted**; the interaction layer
    queries a KD-tree to find the pocket region given a specific ligand.
    """

    path: str
    z_array: np.ndarray              # (N,) int32 atomic numbers
    coords: np.ndarray               # (N, 3) float64
    resnames: List[str]              # length N
    atom_names: List[str]            # length N (stripped, upper-case)
    res_classes: List[str]           # length N, one of RESIDUE_CLASSES or ""
    is_backbone: np.ndarray          # (N,) bool
    is_donor: np.ndarray             # (N,) bool, residue-aware
    is_acceptor: np.ndarray          # (N,) bool, residue-aware
    formal_charge: np.ndarray        # (N,) int8 from residue rule
    kdtree: cKDTree
    aromatic_rings: List[_RingDescriptor]


def _aromatic_ring_for_residue(
    resname: str, atom_idx_map: Dict[Tuple[str, str], int], coords: np.ndarray,
) -> List[_RingDescriptor]:
    """Build one or two :class:`_RingDescriptor` instances if the residue
    is aromatic (PHE/TYR/TRP/HIS) and all ring-atom names are present
    in the parsed structure.
    """
    out: List[_RingDescriptor] = []
    if resname not in _AROMATIC_RING_ATOMS:
        return out
    for atom_names in _AROMATIC_RING_ATOMS[resname]:
        try:
            indices = [atom_idx_map[(resname, name)] for name in atom_names]
        except KeyError:
            continue
        pts = coords[indices]
        centroid = pts.mean(axis=0)
        # Best-fit plane normal via SVD.
        rel = pts - centroid
        _, _, vh = np.linalg.svd(rel, full_matrices=False)
        normal = vh[-1]
        normal = normal / (np.linalg.norm(normal) + 1e-9)
        out.append(_RingDescriptor(
            atom_indices=list(indices),
            centroid=centroid,
            normal=normal,
        ))
    return out


def load_protein(path: str) -> ProteinHandle:
    """Parse a protein structure and pre-compute interaction-friendly
    annotations.

    Notes
    -----
    - Only heavy atoms are kept.
    - Atoms with alternate-location indicators other than '' / ' ' / 'A'
      are dropped.
    - Non-standard residues (anything not in
      :data:`_RESCLASS`) still contribute coordinates but their
      donor / acceptor / charge / ring annotations are conservative
      (all False / 0).
    """
    parser = _select_parser(path)
    structure = parser.get_structure("prot", path)

    z_list: List[int] = []
    coords_list: List[Tuple[float, float, float]] = []
    res_list: List[str] = []
    aname_list: List[str] = []
    is_bb: List[bool] = []
    is_don: List[bool] = []
    is_acc: List[bool] = []
    fc: List[int] = []
    resclass: List[str] = []

    # Track per-residue atom-name maps for aromatic ring extraction.
    per_residue_atom_idx: Dict[Tuple, Dict[str, int]] = {}

    first_model = next(iter(structure), None)
    if first_model is None:
        raise ValueError(f"no models in protein {path!r}")

    for chain in first_model:
        chain_id = chain.id
        for residue in chain:
            resname = residue.get_resname().strip().upper()
            res_key = (chain_id, residue.id, resname)
            seen_names: set = set()
            for atom in residue:
                element = normalize_element(getattr(atom, "element", None))
                if element == "H":
                    continue
                altloc = atom.get_altloc()
                if altloc not in ("", " ", "A"):
                    continue
                name = atom.get_name().strip().upper()
                if name in seen_names:
                    continue
                seen_names.add(name)
                z = to_atomic_number(element)
                if z == 0:
                    continue
                idx = len(z_list)
                z_list.append(z)
                c = atom.get_coord()
                coords_list.append((float(c[0]), float(c[1]), float(c[2])))
                res_list.append(resname)
                aname_list.append(name)
                cls = _RESCLASS.get(resname, "")
                resclass.append(cls)
                backbone = name in _BACKBONE_NAMES
                is_bb.append(backbone)
                # Donors: backbone N (always); side-chain N/O per map.
                donor = False
                if backbone and name == "N" and resname != "PRO":
                    donor = True
                elif name in _SIDE_CHAIN_DONORS.get(resname, frozenset()):
                    donor = True
                is_don.append(donor)
                # Acceptors: backbone O; side-chain N/O/S per map.
                acceptor = False
                if backbone and name in ("O", "OXT"):
                    acceptor = True
                elif name in _SIDE_CHAIN_ACCEPTORS.get(resname, frozenset()):
                    acceptor = True
                is_acc.append(acceptor)
                # Side-chain formal charge.
                fch = 0
                if name in _POSITIVE_RES_ATOMS.get(resname, frozenset()):
                    fch = 1 if resname in ("LYS", "ARG") else 0  # HIS is partial
                elif name in _NEGATIVE_RES_ATOMS.get(resname, frozenset()):
                    fch = -1
                fc.append(fch)

                per_residue_atom_idx.setdefault(res_key, {})[name] = idx

    if not z_list:
        raise ValueError(f"no heavy atoms parsed from protein {path!r}")

    z_arr = np.asarray(z_list, dtype=np.int32)
    coords = np.asarray(coords_list, dtype=np.float64)
    tree = cKDTree(coords)

    # Aromatic ring extraction (one or two rings per qualifying residue).
    aromatic_rings: List[_RingDescriptor] = []
    for res_key, name_map in per_residue_atom_idx.items():
        _, _, resname = res_key
        if resname not in _AROMATIC_RING_ATOMS:
            continue
        for atom_names in _AROMATIC_RING_ATOMS[resname]:
            try:
                indices = [name_map[a] for a in atom_names]
            except KeyError:
                continue
            pts = coords[indices]
            centroid = pts.mean(axis=0)
            rel = pts - centroid
            _, _, vh = np.linalg.svd(rel, full_matrices=False)
            normal = vh[-1]
            normal = normal / (np.linalg.norm(normal) + 1e-9)
            aromatic_rings.append(_RingDescriptor(
                atom_indices=list(indices),
                centroid=centroid,
                normal=normal,
            ))

    return ProteinHandle(
        path=path,
        z_array=z_arr,
        coords=coords,
        resnames=res_list,
        atom_names=aname_list,
        res_classes=resclass,
        is_backbone=np.asarray(is_bb, dtype=bool),
        is_donor=np.asarray(is_don, dtype=bool),
        is_acceptor=np.asarray(is_acc, dtype=bool),
        formal_charge=np.asarray(fc, dtype=np.int8),
        kdtree=tree,
        aromatic_rings=aromatic_rings,
    )


# ---------------------------------------------------------------------------
# Mini in-memory PDB writer for tests / fixtures
# ---------------------------------------------------------------------------

def write_minimal_pdb(
    path: str,
    atoms: Sequence[Tuple[str, Tuple[float, float, float]]],
    resname: str = "LIG",
    chain: str = "A",
) -> None:
    """Write a deliberately minimal PDB file from a list of
    ``(element, (x, y, z))`` tuples.

    Used only by the test fixtures to keep the repo light. Not part of
    the public API but exported for tests.
    """
    lines = []
    for i, (element, (x, y, z)) in enumerate(atoms, start=1):
        atom_name = f"{element.upper()}{i}"[:4]
        lines.append(
            f"HETATM{i:>5} {atom_name:<4} {resname:<3} {chain}"
            f"{1:>4}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}"
            f"  1.00  0.00          "
            f"{element.upper():>2}\n"
        )
    lines.append("END\n")
    with open(path, "w") as fp:
        fp.writelines(lines)
