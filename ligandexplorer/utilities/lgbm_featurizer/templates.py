"""SMARTS-based functional-group counting.

All SMARTS use the ``~`` (any) bond operator so matches are valid on
the single-bond RWMol produced by :mod:`lgbm_featurizer.loader`. Each
SMARTS targets a heavy-atom topological pattern; bond orders are
deliberately not specified.

When the same physical substructure can be matched in multiple
permutations (e.g. carboxyl matched once for each terminal O), we
deduplicate by the *unordered* set of matched atom indices.

Patterns intentionally avoid aromaticity flags (``a`` / ``A``) because
those depend on perception. For aromaticity-sensitive features (e.g.
distinguishing aromatic ether vs aliphatic ether), use the
geometric-aromatic detection from
:mod:`lgbm_featurizer.chemistry` plus a SMARTS structural filter.
"""
from __future__ import annotations

from typing import List, Set, Tuple

from rdkit import Chem

from .loader import LigandHandle


# ---------------------------------------------------------------------------
# SMARTS catalogue
# ---------------------------------------------------------------------------
# Each entry: (column-name, SMARTS string).  Column names must match
# :data:`lgbm_featurizer.schema.SMARTS_GROUPS.columns` exactly and in
# the same order.

_SMARTS_SPEC: Tuple[Tuple[str, str], ...] = (
    # Carboxyl-like: C attached to two terminal O. We use the strict
    # carboxylic form (#6 center) here; phosphate / sulfonate / nitro
    # are distinguished by their dedicated patterns below.
    ("smarts_carboxyl",
        "[#6](~[#8;D1])~[#8;D1]"),
    # Ester-like: non-ring C with one terminal O and one bridging O
    # (the bridging O leads to another C). Restricted to !R to avoid
    # matching anomeric ring carbons that have an exocyclic hydroxyl
    # in sugars.
    ("smarts_ester_like",
        "[#6;D3;!R](~[#8;D1])~[#8;D2]~[#6]"),
    # Amide-like: non-ring C with one terminal O and one N.
    ("smarts_amide_like",
        "[#6;D3;!R](~[#8;D1])~[#7]"),
    # Urea-like: non-ring N-C(=O)-N.
    ("smarts_urea_like",
        "[#7]~[#6;D3;!R](~[#8;D1])~[#7]"),
    # Guanidinium: C bonded to three N (>= 2 of which terminal).
    ("smarts_guanidinium",
        "[#6](~[#7;D1])(~[#7;D1])~[#7]"),
    # Primary amine: terminal N attached to C.
    ("smarts_amine_primary",
        "[#7;D1]~[#6]"),
    # Secondary amine: non-ring N with degree 2 between two C atoms,
    # and not adjacent to a carbonyl-like C (which would make it an
    # amide N, already counted by smarts_amide_like).
    ("smarts_amine_secondary",
        "[#7;D2;!R;!$([#7]~[#6;D3](~[#8;D1])~*)](~[#6])~[#6]"),
    # Tertiary amine: non-ring N with degree 3, three C neighbours,
    # not adjacent to a carbonyl-like C.
    ("smarts_amine_tertiary",
        "[#7;D3;!R;!$([#7]~[#6;D3](~[#8;D1])~*)](~[#6])(~[#6])~[#6]"),
    # Quaternary ammonium: N with degree 4.
    ("smarts_amine_quaternary",
        "[#7;D4]"),
    # Hydroxyl: terminal O on a C/Si center that does NOT also carry a
    # second terminal O (which would make it a carboxylate / similar).
    # The recursive ``!$(...)`` clause excludes such centers.
    ("smarts_hydroxyl",
        "[#8;D1]~[#6,#14;!$([#6,#14](~[#8;D1])~[#8;D1])]"),
    # Ether: bridging O between two heavy atoms.
    ("smarts_ether",
        "[#8;D2](~*)~*"),
    # Ketone-like: non-ring 3-coord C with one terminal O, two C
    # neighbours. We use ``!R`` to exclude phenol-like ring carbons and
    # sugar anomeric carbons that have an exocyclic OH.
    ("smarts_ketone_like",
        "[#6;D3;!R](~[#8;D1])(~[#6])~[#6]"),
    # Nitro: N with two terminal O attached.
    ("smarts_nitro",
        "[#7](~[#8;D1])~[#8;D1]"),
    # Nitrile: terminal C-N where both have degree 1.
    ("smarts_nitrile",
        "[#6;D1]~[#7;D1]"),
    # Sulfonate: S with three terminal O.
    ("smarts_sulfonate",
        "[#16](~[#8;D1])(~[#8;D1])~[#8;D1]"),
    # Sulfonamide: S with two terminal O and one N.
    ("smarts_sulfonamide",
        "[#16](~[#8;D1])(~[#8;D1])~[#7]"),
    # Phosphate: P with three or four terminal O.
    ("smarts_phosphate",
        "[#15](~[#8;D1])(~[#8;D1])(~[#8;D1])~[#8]"),
    # Phosphonate: P with two terminal O and one C.
    ("smarts_phosphonate",
        "[#15](~[#8;D1])(~[#8;D1])~[#6]"),
    # Thiol: terminal S attached to C.
    ("smarts_thiol",
        "[#6]~[#16;D1]"),
    # Disulfide: two adjacent S each with degree 2.
    ("smarts_disulfide",
        "[#16;D2]~[#16;D2]"),
)


# Pre-compile.  We rely on RWMol's RingInfo being populated; SMARTS
# without ``@`` operators still work without RingInfo, but RDKit
# sometimes calls into it internally, so the loader-side
# ``GetSymmSSSR`` call covers us.
_COMPILED_SMARTS: List[Tuple[str, Chem.Mol]] = []
for name, smarts in _SMARTS_SPEC:
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        raise RuntimeError(
            f"SMARTS template {name!r} failed to compile: {smarts!r}"
        )
    _COMPILED_SMARTS.append((name, patt))


def smarts_group_columns() -> Tuple[str, ...]:
    return tuple(name for name, _ in _SMARTS_SPEC)


def _unique_match_count(mol: Chem.Mol, patt: Chem.Mol) -> int:
    """Return the number of substructure matches deduplicated by the
    unordered set of involved atom indices.

    RDKit's ``GetSubstructMatches`` already passes ``uniquify=True`` by
    default which canonicalises permutations of equivalent atoms; we
    additionally collapse any remaining symmetric mappings (e.g. the
    two terminal O's of a carboxylate, which RDKit may still emit
    twice for the carboxyl pattern with both orderings).
    """
    matches = mol.GetSubstructMatches(patt, uniquify=True,
                                       useChirality=False)
    if not matches:
        return 0
    seen: Set[frozenset] = set()
    for m in matches:
        seen.add(frozenset(m))
    return len(seen)


def smarts_group_counts(handle: LigandHandle) -> List[float]:
    """Return SMARTS group counts in :data:`schema.SMARTS_GROUPS.columns`
    order.

    Returns all zeros for handles with fewer than 2 heavy atoms.
    """
    if handle.rdmol.GetNumAtoms() < 2:
        return [0.0] * len(_COMPILED_SMARTS)
    out: List[float] = []
    for _name, patt in _COMPILED_SMARTS:
        out.append(float(_unique_match_count(handle.rdmol, patt)))
    return out


__all__ = [
    "smarts_group_columns",
    "smarts_group_counts",
]
