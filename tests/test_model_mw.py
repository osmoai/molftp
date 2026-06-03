"""End-to-end model test: predict a molecular-weight-threshold class from structure alone.

Molecular weight is a deterministic function of structure, so a working molFTP
inference→predict pipeline should classify an MW-median-threshold target well above chance on
a set of diverse molecules. This is a learning/sanity test for the whole stack
(feature generation + downstream estimator), not just shape-checking.
"""
import numpy as np
import pytest

pytest.importorskip("_molftp")
pytest.importorskip("molftp")
Chem = pytest.importorskip("rdkit.Chem")
from rdkit.Chem import Descriptors
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from molftp.predict import MolFTPClassifier


def _diverse_molecules(n=200):
    """n distinct, valid molecules spanning ring systems, heteroatoms, halogens and groups,
    each paired with its RDKit molecular weight."""
    cores = ['c1ccccc1', 'C1CCCCC1', 'c1ccncc1', 'c1ccsc1', 'C1CCNCC1', 'c1cccnc1', 'C1CCOCC1', 'c1ccoc1']
    links = ['', 'C', 'CC', 'CCC', 'O', 'N', 'CO', 'CN', 'S', 'CCO']
    tails = ['C', 'O', 'N', 'F', 'Cl', 'Br', 'C(F)(F)F', 'C#N', 'C(=O)O', 'CO']
    out, seen = [], set()
    for c in cores:
        for l in links:
            for t in tails:
                s = c + l + t
                if s in seen:
                    continue
                m = Chem.MolFromSmiles(s)
                if m is None:
                    continue
                seen.add(s)
                out.append((s, Descriptors.MolWt(m)))
                if len(out) >= n:
                    return out
    return out


def test_mw_threshold_classification():
    data = _diverse_molecules(200)
    assert len(data) == 200, f"only generated {len(data)} valid molecules"
    smiles = [s for s, _ in data]
    mw = np.array([w for _, w in data])

    # Binary target: above/below the median MW -> balanced classes.
    y = (mw > np.median(mw)).astype(int)
    assert 0.4 < y.mean() < 0.6, f"target not balanced: {y.mean():.2f}"

    Xtr, Xte, ytr, yte = train_test_split(smiles, y, test_size=0.25, random_state=0, stratify=y)

    clf = MolFTPClassifier(radius=3).fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, proba)
    acc = (clf.predict(Xte) == yte).mean()

    # Observed ~0.87 AUC / ~0.80 ACC; assert comfortably above chance with margin for stability.
    assert auc > 0.70, f"MW-threshold AUC too low: {auc:.3f} (chance=0.5)"
    assert acc > 0.65, f"MW-threshold accuracy too low: {acc:.3f}"
