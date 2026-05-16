"""High-level entry point that assembles the full feature vector.

``compute_ligand_features(pdb_path)`` is the single function the
training and inference pipelines should call.

Failure handling
----------------
Any exception during parsing or featurization is logged via ``print``
(stderr) and the function returns a zero-filled vector of length
:data:`lgbm_featurizer.schema.FEATURE_DIM`. This keeps CSV row layout
intact when a PDB is malformed.
"""
from __future__ import annotations

import sys
import traceback
from typing import List

from .descriptors import (
    bond_length_histogram,
    element_counts,
    physchem_geom,
    shape_3d,
    topology,
)
from .interactions import compute_interaction_features
from .loader import LigandHandle, load_ligand, load_protein
from .residues import residue_counts
from .schema import (
    FEATURE_BLOCKS_FULL,
    FEATURE_DIM,
    FEATURE_DIM_FULL,
    assemble,
    zero_vector,
)
from .templates import smarts_group_counts


def _mol_parts(handle: LigandHandle):
    return {
        "element_counts": element_counts(handle),
        "physchem_geom": physchem_geom(handle),
        "bond_length_hist": bond_length_histogram(handle),
        "topology": topology(handle),
        "shape_3d": shape_3d(handle),
        "smarts_groups": smarts_group_counts(handle),
        "residue_counts": residue_counts(handle.resnames),
    }


def compute_ligand_features(pdb_path: str) -> List[float]:
    """Compute the v2.0-zbo-mol feature vector for a single PDB ligand.

    Returns
    -------
    list of float
        Length :data:`FEATURE_DIM`. On any exception, returns a zero
        vector of the same length so callers can keep CSV rows aligned.
    """
    try:
        handle = load_ligand(pdb_path)
    except Exception as exc:
        print(
            f"[lgbm_featurizer] parse error for {pdb_path!r}: {exc}",
            file=sys.stderr,
        )
        return zero_vector()

    try:
        return assemble(_mol_parts(handle))
    except Exception:
        print(
            f"[lgbm_featurizer] featurize error for {pdb_path!r}\n"
            + traceback.format_exc(),
            file=sys.stderr,
        )
        return zero_vector()


def compute_complex_features(
    protein_pdb: str,
    ligand_pdb: str,
) -> List[float]:
    """Compute the v2.0-zbo-full feature vector for a protein-ligand
    pair (used by model_3, the binary "is real ligand?" classifier).

    The 158-dim layout is the mol-only block (86) followed by the
    interaction block (72); see :data:`schema.FEATURE_COLUMNS_FULL` for
    the column order.

    Returns
    -------
    list of float
        Length :data:`FEATURE_DIM_FULL`. On any exception, returns a
        zero vector of the same length so callers can keep CSV rows
        aligned.
    """
    try:
        lig = load_ligand(ligand_pdb)
    except Exception as exc:
        print(
            f"[lgbm_featurizer] ligand parse error for {ligand_pdb!r}: {exc}",
            file=sys.stderr,
        )
        return zero_vector(FEATURE_BLOCKS_FULL)
    try:
        prot = load_protein(protein_pdb)
    except Exception as exc:
        print(
            f"[lgbm_featurizer] protein parse error for "
            f"{protein_pdb!r}: {exc}",
            file=sys.stderr,
        )
        return zero_vector(FEATURE_BLOCKS_FULL)

    try:
        parts = _mol_parts(lig)
        parts.update(compute_interaction_features(prot, lig))
        return assemble(parts, blocks=FEATURE_BLOCKS_FULL)
    except Exception:
        print(
            f"[lgbm_featurizer] complex featurize error for "
            f"protein={protein_pdb!r} ligand={ligand_pdb!r}\n"
            + traceback.format_exc(),
            file=sys.stderr,
        )
        return zero_vector(FEATURE_BLOCKS_FULL)


def feature_dim() -> int:
    """Convenience accessor for :data:`FEATURE_DIM`."""
    return FEATURE_DIM


def feature_dim_full() -> int:
    """Convenience accessor for :data:`FEATURE_DIM_FULL`."""
    return FEATURE_DIM_FULL
