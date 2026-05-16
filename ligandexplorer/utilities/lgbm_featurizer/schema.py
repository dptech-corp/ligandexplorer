"""Feature schema (column names, ordering, version stamps).

The schema is the single source of truth for which numbers go where in
the output CSV. Any change here must be accompanied by a version bump
and a retraining of the LGBM models.

Two ordered schemas live here:

- ``v2.0-zbo-mol``: 86 mol-only dims, used by ``model_1`` (3-class) and
  ``model_2`` (5-class). Composed of :data:`FEATURE_BLOCKS`.
- ``v2.0-zbo-full``: 86 mol + 72 interaction = 158 dims, used by
  ``model_3`` (binary "real ligand?"). Composed of
  :data:`FEATURE_BLOCKS_FULL` which is
  :data:`FEATURE_BLOCKS` + :data:`INTERACTION_BLOCKS`.

``compute.py`` assembles the final vector by calling each section's
producer function and concatenating in schema order.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

FEATURE_SCHEMA_VERSION = "v2.0-zbo-mol"
FEATURE_SCHEMA_VERSION_FULL = "v2.0-zbo-full"


@dataclass(frozen=True)
class FeatureBlock:
    """Specification for one named block in the feature vector."""
    name: str
    columns: Tuple[str, ...]

    @property
    def dim(self) -> int:
        return len(self.columns)


ELEMENT_COUNTS = FeatureBlock(
    name="element_counts",
    columns=(
        "count_C", "count_N", "count_O", "count_P", "count_S",
        "count_F", "count_Cl", "count_Br", "count_I",
        "count_metal_alkali", "count_metal_alkaline_earth",
        "count_metal_transition", "count_metal_other",
        "heteroatom_ratio",
        "formal_charge_pos_count", "formal_charge_neg_count",
    ),
)

PHYSCHEM_GEOM = FeatureBlock(
    name="physchem_geom",
    columns=(
        "mw",
        "heavy_atom_count",
        "num_heteroatoms",
        "hbd_rule",
        "hba_rule",
        "num_rotatable_bonds_smarts",
        "fraction_csp3_geom",
        "num_aromatic_rings_geom",
        "num_aliphatic_rings",
        "num_saturated_rings",
        "num_spiro_atoms",
        "num_bridgehead_atoms",
    ),
)

# 6 length buckets, each producing (count, mean_length) -> 12 dims.
BOND_LENGTH_BUCKETS = (
    (1.10, 1.30),
    (1.30, 1.42),
    (1.42, 1.55),
    (1.55, 1.75),
    (1.75, 2.00),
    (2.00, 2.50),
)


def _bond_length_columns() -> Tuple[str, ...]:
    cols: List[str] = []
    for lo, hi in BOND_LENGTH_BUCKETS:
        tag = f"{int(lo*100):03d}_{int(hi*100):03d}"
        cols.append(f"bond_len_count_{tag}")
        cols.append(f"bond_len_mean_{tag}")
    return tuple(cols)


BOND_LENGTH_HIST = FeatureBlock(
    name="bond_length_hist",
    columns=_bond_length_columns(),
)

TOPOLOGY = FeatureBlock(
    name="topology",
    columns=(
        "num_rings_basis",
        "ring_atom_fraction",
        "longest_chain",
        "num_fused_rings",
        "num_aromatic_heterocycles_geom",
        "ring_size_mode",
    ),
)

SHAPE_3D = FeatureBlock(
    name="shape_3d",
    columns=(
        "radius_of_gyration",
        "pca_lambda_1",
        "pca_lambda_2",
        "pca_lambda_3",
        "asphericity",
        "acylindricity",
        "sphericity",
        "molecular_diameter",
    ),
)

# 20 SMARTS-based functional groups (all using ~ wildcard bonds).
SMARTS_GROUPS = FeatureBlock(
    name="smarts_groups",
    columns=(
        "smarts_carboxyl",
        "smarts_ester_like",
        "smarts_amide_like",
        "smarts_urea_like",
        "smarts_guanidinium",
        "smarts_amine_primary",
        "smarts_amine_secondary",
        "smarts_amine_tertiary",
        "smarts_amine_quaternary",
        "smarts_hydroxyl",
        "smarts_ether",
        "smarts_ketone_like",
        "smarts_nitro",
        "smarts_nitrile",
        "smarts_sulfonate",
        "smarts_sulfonamide",
        "smarts_phosphate",
        "smarts_phosphonate",
        "smarts_thiol",
        "smarts_disulfide",
    ),
)

# 12 resname-derived residue features.
RESIDUE_COUNTS = FeatureBlock(
    name="residue_counts",
    columns=(
        "res_aa_hydrophobic",
        "res_aa_polar",
        "res_aa_positive",
        "res_aa_negative",
        "res_aa_aromatic",
        "res_aa_glycine",
        "res_aa_proline",
        "res_aa_cysteine",
        "res_nuc_purine",
        "res_nuc_pyrimidine",
        "res_sugar_total",
        "res_cofactor_or_other",
    ),
)

# Master ordering for the mol-only schema. Add new blocks here when
# extending the mol schema and bump FEATURE_SCHEMA_VERSION.
FEATURE_BLOCKS: Tuple[FeatureBlock, ...] = (
    ELEMENT_COUNTS,
    PHYSCHEM_GEOM,
    BOND_LENGTH_HIST,
    TOPOLOGY,
    SHAPE_3D,
    SMARTS_GROUPS,
    RESIDUE_COUNTS,
)

FEATURE_COLUMNS: Tuple[str, ...] = tuple(
    col for block in FEATURE_BLOCKS for col in block.columns
)

FEATURE_DIM: int = len(FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# Interaction blocks (model_3 / "is real ligand" binary classifier)
# ---------------------------------------------------------------------------
# Geometric protein-ligand interaction features. Total: 72 columns.

# Distance shells used by both residue-contact and electrostatic blocks.
# Left-closed right-open so a contact at exactly 3.5 A lands in the
# second shell.
CONTACT_SHELLS: Tuple[Tuple[float, float], ...] = (
    (0.0, 3.5),
    (3.5, 5.0),
    (5.0, 7.0),
)

# Coarse residue classes used for contact statistics. Aligns with the
# residue classifier in :mod:`lgbm_featurizer.residues`.
RESIDUE_CLASSES: Tuple[str, ...] = (
    "hydrophobic", "polar", "positive", "negative", "aromatic",
)


def _residue_contact_columns() -> Tuple[str, ...]:
    cols: List[str] = []
    for lo, hi in CONTACT_SHELLS:
        tag = f"{int(lo*10):02d}_{int(hi*10):02d}"
        for cls in RESIDUE_CLASSES:
            for chain in ("main", "side"):
                cols.append(f"res_contact_{tag}_{cls}_{chain}")
    return tuple(cols)


RESIDUE_CONTACT = FeatureBlock(
    name="residue_contact",
    columns=_residue_contact_columns(),
)
assert RESIDUE_CONTACT.dim == 30, RESIDUE_CONTACT.dim

HBOND = FeatureBlock(
    name="hbond",
    columns=(
        "hb_count_lig_donor_to_prot",
        "hb_count_prot_donor_to_lig",
        "hb_count_prot_main",
        "hb_count_prot_side",
        "hb_min_dist",
        "hb_mean_dist",
        "hb_mean_angle_deg",
        "hb_count_angle_ge_150",
    ),
)
assert HBOND.dim == 8, HBOND.dim

HALOGEN_BOND = FeatureBlock(
    name="halogen_bond",
    columns=(
        "xb_count",
        "xb_count_acceptor_O",
        "xb_count_acceptor_N",
        "xb_min_dist",
    ),
)
assert HALOGEN_BOND.dim == 4, HALOGEN_BOND.dim

PI_PI = FeatureBlock(
    name="pi_pi",
    columns=(
        "pi_face_to_face",
        "pi_edge_to_face",
        "pi_tilted",
        "pi_min_center_dist",
    ),
)
assert PI_PI.dim == 4, PI_PI.dim

CATION_PI = FeatureBlock(
    name="cation_pi",
    columns=(
        "catpi_lig_to_prot",
        "catpi_prot_to_lig",
    ),
)
assert CATION_PI.dim == 2, CATION_PI.dim


def _electrostatic_columns() -> Tuple[str, ...]:
    shells = CONTACT_SHELLS + ((7.0, 10.0),)
    cols: List[str] = []
    for lo, hi in shells:
        tag = f"{int(lo*10):02d}_{int(hi*10):02d}"
        cols.append(f"elec_attract_{tag}")
        cols.append(f"elec_repulse_{tag}")
    return tuple(cols)


ELECTROSTATIC = FeatureBlock(
    name="electrostatic",
    columns=_electrostatic_columns(),
)
assert ELECTROSTATIC.dim == 8, ELECTROSTATIC.dim


HYDROPHOBIC = FeatureBlock(
    name="hydrophobic_contact",
    columns=(
        # 3 shells x (C-C contact, aromatic-aromatic ring center contact)
        "hphob_CC_00_40",
        "hphob_CC_40_55",
        "hphob_CC_55_70",
        "hphob_arom_00_40",
        "hphob_arom_40_55",
        "hphob_arom_55_70",
    ),
)
assert HYDROPHOBIC.dim == 6, HYDROPHOBIC.dim

METAL_COORD = FeatureBlock(
    name="metal_coord",
    columns=(
        "metal_lig_to_prot_O",
        "metal_lig_to_prot_N",
        "metal_lig_to_prot_S",
        "metal_prot_to_lig_any",
    ),
)
assert METAL_COORD.dim == 4, METAL_COORD.dim

POCKET = FeatureBlock(
    name="pocket",
    columns=(
        "pocket_n_heavy_atoms",
        "pocket_min_lig_dist",
        "pocket_mean_lig_min_dist",
        "lig_n_heavy_atoms",
        "lig_diameter",
        "pocket_diameter",
    ),
)
assert POCKET.dim == 6, POCKET.dim


INTERACTION_BLOCKS: Tuple[FeatureBlock, ...] = (
    RESIDUE_CONTACT,
    HBOND,
    HALOGEN_BOND,
    PI_PI,
    CATION_PI,
    ELECTROSTATIC,
    HYDROPHOBIC,
    METAL_COORD,
    POCKET,
)

INTERACTION_COLUMNS: Tuple[str, ...] = tuple(
    col for block in INTERACTION_BLOCKS for col in block.columns
)
INTERACTION_DIM: int = len(INTERACTION_COLUMNS)

# Combined mol + interaction layout (used by model_3).
FEATURE_BLOCKS_FULL: Tuple[FeatureBlock, ...] = (
    FEATURE_BLOCKS + INTERACTION_BLOCKS
)
FEATURE_COLUMNS_FULL: Tuple[str, ...] = tuple(
    col for block in FEATURE_BLOCKS_FULL for col in block.columns
)
FEATURE_DIM_FULL: int = len(FEATURE_COLUMNS_FULL)


def assemble(parts, blocks: Tuple[FeatureBlock, ...] = FEATURE_BLOCKS) -> List[float]:
    """Flatten a ``{block.name: sequence_of_floats}`` mapping in schema
    order and validate the result.

    By default uses the mol-only :data:`FEATURE_BLOCKS`. Pass
    :data:`FEATURE_BLOCKS_FULL` to assemble the 158-dim mol+interaction
    vector for model_3.

    Raises
    ------
    ValueError
        If a block is missing, has the wrong length, or the final
        dimension does not match the expected total.
    """
    out: List[float] = []
    for block in blocks:
        if block.name not in parts:
            raise ValueError(
                f"missing feature block {block.name!r} during assemble"
            )
        values = parts[block.name]
        if len(values) != block.dim:
            raise ValueError(
                f"block {block.name!r} expected {block.dim} values, "
                f"got {len(values)}"
            )
        out.extend(float(v) for v in values)
    expected = sum(b.dim for b in blocks)
    if len(out) != expected:
        raise ValueError(
            f"assembled vector has length {len(out)}, expected {expected}"
        )
    return out


def zero_vector(blocks: Tuple[FeatureBlock, ...] = FEATURE_BLOCKS) -> List[float]:
    """Return a zero-filled vector of the expected length. Used by
    :mod:`lgbm_featurizer.compute` when an individual PDB fails to
    parse, so the CSV row layout stays intact.
    """
    return [0.0] * sum(b.dim for b in blocks)
