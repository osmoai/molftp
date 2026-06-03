"""Tests for the predict layer (molftp.predict.MolFTPClassifier).

Verifies the clean inference (transform) vs predict (predict/predict_proba) separation and
the scikit-learn-style estimator behaviour.
"""
import numpy as np
import pytest

pytest.importorskip("_molftp")
molftp = pytest.importorskip("molftp")
from molftp.predict import MolFTPClassifier


@pytest.fixture(scope="module")
def data():
    smiles = ['CCO', 'CCN', 'CCC', 'c1ccccc1', 'c1ccccc1O', 'CC(=O)O', 'CCCl', 'CCBr', 'CCF',
              'CCCCO', 'CCCCN', 'c1ccncc1', 'CCCO', 'CCCN', 'c1ccccc1C', 'CCCCCO']
    y = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0])
    return smiles, y


def test_layers_exposed():
    assert hasattr(molftp, "features") and hasattr(molftp, "predict")
    from molftp.features import MultiTaskPrevalenceGenerator, PrevalenceGenerator  # noqa: F401


def test_fit_predict_proba_shapes(data):
    smiles, y = data
    clf = MolFTPClassifier(radius=3).fit(smiles, y)
    pred = clf.predict(smiles)
    assert pred.shape == (len(smiles),)
    proba = clf.predict_proba(smiles)
    assert proba.shape == (len(smiles), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_transform_is_inference_only(data):
    smiles, y = data
    clf = MolFTPClassifier(radius=3).fit(smiles, y)
    X = clf.transform(smiles[:4])
    assert X.ndim == 2 and X.shape[0] == 4


def test_transform_before_fit_raises(data):
    smiles, _ = data
    with pytest.raises(RuntimeError):
        MolFTPClassifier(radius=3).transform(smiles)


def test_custom_estimator(data):
    smiles, y = data
    from sklearn.ensemble import RandomForestClassifier
    clf = MolFTPClassifier(
        radius=3, estimator=RandomForestClassifier(n_estimators=8, random_state=0)
    ).fit(smiles, y)
    assert clf.predict(smiles).shape == (len(smiles),)


def test_single_str_is_rejected(data):
    smiles, y = data
    clf = MolFTPClassifier(radius=3).fit(smiles, y)
    with pytest.raises(TypeError):
        clf.predict("CCO")  # a bare string is a common mistake; must be a sequence


def test_length_mismatch_raises(data):
    smiles, y = data
    with pytest.raises(ValueError):
        MolFTPClassifier(radius=3).fit(smiles, y[:-1])


def test_predict_proba_requires_support(data):
    smiles, y = data

    class _NoProba:
        def fit(self, X, y):
            self.classes_ = np.unique(y)
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    clf = MolFTPClassifier(radius=3, estimator=_NoProba()).fit(smiles, y)
    with pytest.raises(AttributeError):
        clf.predict_proba(smiles)
