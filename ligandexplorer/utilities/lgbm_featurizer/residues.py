"""Residue-name based feature aggregation.

The legacy implementation tried to identify amino acids by subgraph
isomorphism against handwritten backbone+sidechain templates. That
fails for N/C terminal residues, side-chain modifications (pSer/pThr/
pTyr), unusual rotamers, and so on. PDB structures already carry
authoritative three-letter residue names; using them directly is far
more accurate.

The output is a length-12 vector mapped to the
:data:`lgbm_featurizer.schema.RESIDUE_COUNTS` columns.
"""
from __future__ import annotations

from collections import Counter
from typing import FrozenSet, List, Sequence


# ---------------------------------------------------------------------------
# Residue-name dictionaries
# ---------------------------------------------------------------------------

# 20 standard amino acids grouped by side-chain physicochemistry.
_AA_HYDROPHOBIC: FrozenSet[str] = frozenset(
    {"ALA", "VAL", "LEU", "ILE", "MET"}
)
_AA_POLAR: FrozenSet[str] = frozenset(
    {"SER", "THR", "ASN", "GLN"}
)
_AA_POSITIVE: FrozenSet[str] = frozenset({"LYS", "ARG", "HIS"})
_AA_NEGATIVE: FrozenSet[str] = frozenset({"ASP", "GLU"})
_AA_AROMATIC: FrozenSet[str] = frozenset({"PHE", "TYR", "TRP"})

# Special-case residues that get their own column.
_AA_GLYCINE: FrozenSet[str] = frozenset({"GLY"})
_AA_PROLINE: FrozenSet[str] = frozenset({"PRO"})
_AA_CYSTEINE: FrozenSet[str] = frozenset({"CYS"})

# Modified amino acids commonly seen in PDB ligands. Folded back into
# the corresponding canonical class so the feature stays compact.
_MODIFIED_AA_MAP = {
    "SEP": "polar",        # phospho-serine
    "TPO": "polar",        # phospho-threonine
    "PTR": "aromatic",     # phospho-tyrosine
    "MSE": "hydrophobic",  # selenomethionine
    "HYP": "polar",        # hydroxyproline -> still treated as polar
    "CSO": "polar",        # S-hydroxycysteine
}

# Nucleic acid bases (DNA + RNA). PDB convention uses single-letter
# resnames for residue-level entries (DA/DG/DC/DT for DNA, A/G/C/U
# for RNA). Some structures also use long-form (ADE/GUA/...).
_NUC_PURINES: FrozenSet[str] = frozenset({
    "A", "G", "DA", "DG",
    "ADE", "GUA", "DADE", "DGUA",
    "AMP", "ADP", "ATP", "GMP", "GDP", "GTP",
})
_NUC_PYRIMIDINES: FrozenSet[str] = frozenset({
    "C", "U", "T", "DC", "DT", "DU",
    "CYT", "URA", "THY",
    "CMP", "CDP", "CTP", "UMP", "UDP", "UTP",
    "TMP", "TDP", "TTP",
})

# Common sugars (monosaccharides + named PDB CCD codes for sugars).
_SUGAR_NAMES: FrozenSet[str] = frozenset({
    "GLC", "BGC", "ALA",  # ALA collision is intentional? no -> remove
    "GAL", "BGA", "MAN", "BMA", "FUC", "FUL",
    "NAG", "NDG", "NGA",  # N-acetyl sugars
    "XYS", "XYL", "RIB", "ARA", "LYX",
    "BMA", "MAN", "GLA",
    "SIA", "NEU", "NAN",  # sialic acids
    "MAL", "LAT", "TRE",  # disaccharides
})
# Drop accidental collision with the amino acid 'ALA'.
_SUGAR_NAMES = frozenset(_SUGAR_NAMES - {"ALA"})

# Frequently encountered cofactors / bound non-ligand ligands.
_COFACTORS_AND_OTHERS: FrozenSet[str] = frozenset({
    "HEM", "HEC", "HEA",   # heme variants
    "NAD", "NAP", "NDP",
    "FAD", "FMN",
    "COA", "ACO",
    "ATP", "ADP", "AMP", "GTP", "GDP", "GMP",
    "SAM", "SAH",
    "PLP", "PMP",
    "BTN",                 # biotin
    "MGD",                 # molybdopterin
    "F3S", "SF4",          # iron-sulphur clusters
    "BCL", "BCB",          # bacteriochlorophyll
})


def _classify_residue(resname: str) -> str:
    """Return the coarse residue class label for a 3-letter resname.

    Returns one of:
    ``aa_hydrophobic / aa_polar / aa_positive / aa_negative /
    aa_aromatic / aa_glycine / aa_proline / aa_cysteine /
    nuc_purine / nuc_pyrimidine / sugar / cofactor_or_other``.
    """
    name = resname.strip().upper()
    if not name:
        return "cofactor_or_other"

    if name in _AA_HYDROPHOBIC:
        return "aa_hydrophobic"
    if name in _AA_POLAR:
        return "aa_polar"
    if name in _AA_POSITIVE:
        return "aa_positive"
    if name in _AA_NEGATIVE:
        return "aa_negative"
    if name in _AA_AROMATIC:
        return "aa_aromatic"
    if name in _AA_GLYCINE:
        return "aa_glycine"
    if name in _AA_PROLINE:
        return "aa_proline"
    if name in _AA_CYSTEINE:
        return "aa_cysteine"
    if name in _MODIFIED_AA_MAP:
        category = _MODIFIED_AA_MAP[name]
        if category == "hydrophobic":
            return "aa_hydrophobic"
        if category == "polar":
            return "aa_polar"
        if category == "aromatic":
            return "aa_aromatic"
    if name in _NUC_PURINES:
        # The Adenosine-monophosphate family is both a nucleotide AND a
        # cofactor; we count them as nucleotides here because the
        # informative bit is the base, not the phosphate.
        return "nuc_purine"
    if name in _NUC_PYRIMIDINES:
        return "nuc_pyrimidine"
    if name in _SUGAR_NAMES:
        return "sugar"
    return "cofactor_or_other"


def residue_feature_columns() -> tuple:
    return (
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
    )


_LABEL_ORDER = (
    "aa_hydrophobic",
    "aa_polar",
    "aa_positive",
    "aa_negative",
    "aa_aromatic",
    "aa_glycine",
    "aa_proline",
    "aa_cysteine",
    "nuc_purine",
    "nuc_pyrimidine",
    "sugar",
    "cofactor_or_other",
)


def residue_counts(resnames: Sequence[str]) -> List[float]:
    """Aggregate per-atom resnames into the 12 columns.

    Each *residue* (not atom!) contributes 1 to its bucket. A residue
    is identified by its 3-letter resname; we therefore count *unique
    residues per atom-group*. To avoid having to thread the (chain,
    resseq) tuple through the loader, the implementation accepts a
    per-atom resname sequence and counts each contiguous run of
    identical resnames as one residue. This is a fine approximation
    for typical ligand PDBs where residues are listed sequentially.
    """
    counts = Counter()
    last = None
    for name in resnames:
        cls = _classify_residue(name)
        # Count one residue per contiguous block of identical resnames.
        # ``last`` is reset between blocks.
        if name != last:
            counts[cls] += 1
            last = name
    return [float(counts.get(label, 0)) for label in _LABEL_ORDER]


__all__ = [
    "_classify_residue",
    "residue_counts",
    "residue_feature_columns",
]
