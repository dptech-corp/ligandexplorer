"""LGBM featurizer v2 (zero-bond-order, mol-only).

Sandbox package for re-training LGBM models on the legacy ligand
classification pipeline. Lives alongside the original
``generate_feature.py`` and ``*_feature.py`` scripts and does not touch
the main ligandexplorer repository.

Design principles
-----------------
1. PDB files do not carry bond-order information. We refuse to perceive
   it via ``Chem.MolFromPDBFile`` / ``DetermineBonds`` etc. because those
   routines fail silently on non-standard ligands.
2. All RDKit ``RWMol`` objects in this package are built with
   ``BondType.SINGLE`` only, and never sanitized. RDKit is used only as
   a SMARTS engine and 3D toolbox.
3. Substructure matching uses ``~`` (any-bond) SMARTS.
4. Bond-order-sensitive concepts (aromaticity, HBD/HBA, sp3 fraction,
   formal charge) are computed from geometry or rule-based logic, not
   from RDKit's perception routines.

Public surface
--------------
- ``compute_ligand_features(pdb_path)`` -> ``list[float]`` of length
  :data:`FEATURE_DIM` (85). See :mod:`lgbm_featurizer.schema` for the
  exact column layout.
- ``FEATURE_SCHEMA_VERSION``: schema identifier stamped into CSV headers.
- ``FEATURE_COLUMNS``: ordered list of column names.
- ``FEATURE_DIM``: total dimension (must equal ``len(FEATURE_COLUMNS)``).
"""
from .schema import (
    FEATURE_COLUMNS,
    FEATURE_COLUMNS_FULL,
    FEATURE_DIM,
    FEATURE_DIM_FULL,
    FEATURE_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION_FULL,
)

__all__ = [
    "FEATURE_COLUMNS",
    "FEATURE_COLUMNS_FULL",
    "FEATURE_DIM",
    "FEATURE_DIM_FULL",
    "FEATURE_SCHEMA_VERSION",
    "FEATURE_SCHEMA_VERSION_FULL",
    "compute_ligand_features",
    "compute_complex_features",
]


def compute_ligand_features(pdb_path):
    """Compute the mol-only feature vector for a single ligand PDB.

    Imported lazily so that consumers that only need the schema constants
    do not pay the cost of importing RDKit / BioPython.
    """
    from .compute import compute_ligand_features as _impl
    return _impl(pdb_path)


def compute_complex_features(protein_pdb, ligand_pdb):
    """Compute the full (mol + interaction) feature vector for a
    protein-ligand pair. Used by model_3 (binary 'is real ligand?').
    """
    from .compute import compute_complex_features as _impl
    return _impl(protein_pdb, ligand_pdb)
