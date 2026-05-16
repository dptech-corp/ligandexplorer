"""LGBM backend shim (v2.0-zbo).

This module replaces the legacy ~1600-line LGBM workflow with a thin
adapter that routes through :mod:`ligandexplorer.utilities.lgbm_featurizer`
(zero-bond-order featurizer, 86 mol dims + 72 interaction dims).

Public contract -- kept byte-identical to the legacy interface so the
upstream :mod:`ligandexplorer.utilities.ligand_discriminate` dispatcher
does not need to know whether the GNN or LGBM backend is active:

- ``load_model_and_pred(input_pdb, LGBM_Model_package=None) -> str``
    Returns the mol-class label. **LGBM only emits 7 of the 9 classes**
    that the GNN backend can return: ``ions``, ``mem``, ``dna``, ``gly``,
    ``organic``, ``peptide``, ``rna``. Two GNN-only categories
    (``peptide_like``, ``cyclic_peptide``) are deliberately absent --
    the v2 LGBM training set has no examples for them, so the
    ligand_discriminate ``elif`` branches for those classes simply
    never fire in the LGBM path. That is by design.

- ``load_model_and_pred_ligand(protein_pdb, ligand_pdb,
    LGBM_Model_package=None) -> int``
    Returns ``0`` (non-ligand) or ``1`` (real ligand) using model_3 on
    the 158-dim mol+interaction vector.

The actual feature computation lives in
:mod:`ligandexplorer.utilities.lgbm_featurizer`. The models are loaded
once per worker process via
:func:`ligandexplorer.workflow._load_lgbm_models` into
:class:`ligandexplorer.workflow.ModelContainer`.

Errors during featurisation or prediction fall through to safe
defaults (``'organic'`` for the mol classifier, ``0`` for the
binary classifier) so the upstream pipeline never crashes on a single
malformed PDB.
"""
from __future__ import annotations

import sys
from typing import Optional

import numpy as np
import pandas as pd

from ligandexplorer.utilities.lgbm_featurizer import (
    FEATURE_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION_FULL,
    compute_complex_features,
    compute_ligand_features,
)

__all__ = [
    "load_model_and_pred",
    "load_model_and_pred_ligand",
    "FEATURE_SCHEMA_VERSION",
    "FEATURE_SCHEMA_VERSION_FULL",
]


# ---------------------------------------------------------------------------
# Class mappings (must match the training script in
# .../old_data/feature/feature/lgbm_featurizer/train.py)
# ---------------------------------------------------------------------------

# model_1 (3-class coarse): 0=ions, 1=mem, 2=other -> handed to model_2.
_MODEL1_CLASSES = ("ions", "mem", "other")

# model_2 (5-class fine, only when model_1 == 2 'other'):
# 0=dna, 1=gly, 2=organic, 3=peptide, 4=rna.
_MODEL2_CLASSES = ("dna", "gly", "organic", "peptide", "rna")

# model_3 is binary -- the integer 0/1 it returns is forwarded directly to
# ligand_discriminate; no string label table is needed here.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_models_loaded():
    """Return the ``ModelContainer`` class with all six LGBM artefacts
    populated. Imports lazily to avoid a circular import at module load.
    """
    from ligandexplorer.workflow import ModelContainer
    missing = [
        attr for attr in (
            "model_1", "model_2", "model_3",
            "scaler_1", "scaler_2", "scaler_3",
        ) if getattr(ModelContainer, attr, None) is None
    ]
    if missing:
        raise RuntimeError(
            "LGBM artefacts not loaded; missing: " + ", ".join(missing)
            + " -- ensure backend='lgbm' and _load_lgbm_models() ran."
        )
    return ModelContainer


def _predict_one(model, scaler, vec, n_dim_expected):
    """Scale a single feature vector and run the LGBM ``predict``.

    The scaler and LGBM model were both fit on pandas DataFrames at
    training time, so they carry ``feature_names_in_`` / ``feature_name_``
    attributes. We rehydrate the matching column names on the input so
    sklearn does not emit
    ``UserWarning: X does not have valid feature names``
    on every call. Falls back to a plain ndarray if the trained artefact
    happens to lack the names (older pickles).
    """
    if len(vec) != n_dim_expected:
        raise ValueError(
            f"feature vector length {len(vec)} != expected "
            f"{n_dim_expected}"
        )
    X = np.asarray(vec, dtype=np.float32).reshape(1, -1)

    scaler_names = getattr(scaler, 'feature_names_in_', None)
    if scaler_names is not None:
        X = pd.DataFrame(X, columns=scaler_names)
    Xs = scaler.transform(X)

    lgbm_names = getattr(model, 'feature_name_', None)
    if lgbm_names is not None:
        Xs = pd.DataFrame(np.asarray(Xs), columns=lgbm_names)
    return int(model.predict(Xs)[0])


# ---------------------------------------------------------------------------
# Public API -- molecule-class cascade (model_1 -> model_2)
# ---------------------------------------------------------------------------

def load_model_and_pred(input_pdb: str,
                        LGBM_Model_package: Optional[object] = None) -> str:
    """Predict the molecule-class label for a ligand PDB.

    Parameters
    ----------
    input_pdb : str
        Path to the candidate ligand PDB / CIF file.
    LGBM_Model_package : Any
        Kept for backwards compatibility with the legacy signature; the
        models are now loaded into :class:`ModelContainer` at worker
        start-up so this argument is ignored.

    Returns
    -------
    str
        One of ``'ions'``, ``'mem'``, ``'dna'``, ``'gly'``, ``'organic'``,
        ``'peptide'``, ``'rna'``. Returns ``'organic'`` on any internal
        error so the upstream loop keeps moving.
    """
    try:
        mc = _check_models_loaded()
    except RuntimeError as exc:
        print(f"LGBM mol prediction error: {exc}", file=sys.stderr)
        return "organic"

    try:
        feats = compute_ligand_features(input_pdb)
        p1 = _predict_one(mc.model_1, mc.scaler_1, feats, n_dim_expected=86)
        coarse = _MODEL1_CLASSES[p1]
        if coarse != "other":
            return coarse
        p2 = _predict_one(mc.model_2, mc.scaler_2, feats, n_dim_expected=86)
        return _MODEL2_CLASSES[p2]
    except Exception as exc:  # noqa: BLE001
        print(
            f"LGBM mol prediction error on {input_pdb!r}: {exc}",
            file=sys.stderr,
        )
        return "organic"


# ---------------------------------------------------------------------------
# Public API -- binary "is real ligand?" (model_3)
# ---------------------------------------------------------------------------

def load_model_and_pred_ligand(
    protein_pdb: str,
    ligand_pdb: str,
    LGBM_Model_package: Optional[object] = None,
) -> int:
    """Decide whether ``ligand_pdb`` is a real ligand in the binding
    pocket of ``protein_pdb``.

    Returns
    -------
    int
        ``1`` if the model thinks this is a real ligand, ``0`` otherwise
        (or on any internal error).
    """
    try:
        mc = _check_models_loaded()
    except RuntimeError as exc:
        print(f"LGBM ligand prediction error: {exc}", file=sys.stderr)
        return 0

    try:
        feats = compute_complex_features(protein_pdb, ligand_pdb)
        return _predict_one(
            mc.model_3, mc.scaler_3, feats, n_dim_expected=158,
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"LGBM ligand prediction error on "
            f"protein={protein_pdb!r} ligand={ligand_pdb!r}: {exc}",
            file=sys.stderr,
        )
        return 0
