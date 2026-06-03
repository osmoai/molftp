"""MolFTP — Molecular Fragment-Target Prevalence.

High-performance molecular feature generation based on fragment-target prevalence statistics,
with a C++ core. The Python API is split into two clearly separated layers:

    inference  (molftp.features)   SMILES ──fit/transform──▶ feature vectors
    predict    (molftp.predict)    SMILES ──features──estimator──▶ labels / probabilities

Inference / feature generation
------------------------------
    from molftp.features import MultiTaskPrevalenceGenerator   # or: from molftp import ...
    gen = MultiTaskPrevalenceGenerator(radius=6, method="key_loo", k_threshold=2)
    gen.fit(train_smiles, y_train)
    X = gen.transform(test_smiles)            # feature matrix, no labels

Prediction (end-to-end SMILES → label)
--------------------------------------
    from molftp.predict import MolFTPClassifier
    clf = MolFTPClassifier(radius=6).fit(train_smiles, y_train)
    proba = clf.predict_proba(test_smiles)[:, 1]

Methods: 'key_loo' (k-filtering + Key-LOO rescaling) and 'dummy_masking' (per-fold masking).
See BUILD.md for building the C++ extension and docs/ for the full API and method notes.
"""

# --- inference / feature-generation layer (SMILES -> features) ---
from .prevalence import PrevalenceGenerator, MultiTaskPrevalenceGenerator
from . import features

# --- prediction layer (SMILES -> labels) ---
from .predict import MolFTPClassifier
from . import predict

__version__ = "1.6.0"

__all__ = [
    # inference
    "PrevalenceGenerator",
    "MultiTaskPrevalenceGenerator",
    "features",
    # predict
    "MolFTPClassifier",
    "predict",
]
