"""molftp.predict — the PREDICT layer (features → labels).

molFTP is split into two clearly separated halves:

    inference  (molftp.features / molftp.prevalence)   SMILES ──fit/transform──▶ feature vectors
    predict    (molftp.predict, this module)           SMILES ──features──estimator──▶ labels / proba

This module never re-implements feature generation; it *composes* a molFTP feature generator
(the inference half) with any scikit-learn-compatible estimator (the predict half) and exposes
the familiar ``fit`` / ``predict`` / ``predict_proba`` / ``transform`` API. Keeping the two
concerns in separate modules means you can:

  * use the inference layer alone (``transform``) to get features for your own model, or
  * use this predict layer for an end-to-end ``SMILES → label`` estimator.

Example
-------
>>> from molftp.predict import MolFTPClassifier
>>> clf = MolFTPClassifier(radius=4).fit(train_smiles, y_train)
>>> proba = clf.predict_proba(test_smiles)[:, 1]
>>> labels = clf.predict(test_smiles)
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .prevalence import MultiTaskPrevalenceGenerator

__all__ = ["MolFTPClassifier"]


def _check_smiles(smiles: Sequence[str]) -> list:
    if isinstance(smiles, str):
        raise TypeError("smiles must be a sequence of SMILES strings, not a single str")
    smiles = list(smiles)
    if len(smiles) == 0:
        raise ValueError("smiles is empty")
    if not all(isinstance(s, str) for s in smiles):
        raise TypeError("every element of smiles must be a str")
    return smiles


class MolFTPClassifier:
    """scikit-learn-style classifier mapping molecule SMILES → class labels.

    The estimator is a two-stage pipeline:

      1. **inference** — a :class:`molftp.prevalence.MultiTaskPrevalenceGenerator` turns SMILES
         into fragment-target-prevalence feature vectors (``transform``);
      2. **predict** — a downstream scikit-learn estimator maps those features to labels.

    Parameters
    ----------
    radius : int, default=6
        Morgan radius for the molFTP feature generator.
    method : {'key_loo', 'dummy_masking'}, default='key_loo'
        Feature-generation method (see :class:`MultiTaskPrevalenceGenerator`).
    k_threshold : int, default=2
        Key-LOO rare-key filter (keep keys whose molecule- and total-count ≥ k_threshold).
    estimator : sklearn estimator, optional
        Downstream classifier. Defaults to ``LogisticRegression(max_iter=1000)``. Must implement
        ``fit``/``predict`` (and ``predict_proba`` if you call :meth:`predict_proba`).
    generator : MultiTaskPrevalenceGenerator, optional
        Supply a pre-configured generator instead of constructing one from the keyword args above.
    **generator_kwargs
        Extra keyword args forwarded to the generator constructor (e.g. ``nBits``, ``sim_thresh``,
        ``stat_1d``, ``num_threads``).

    Attributes
    ----------
    generator : MultiTaskPrevalenceGenerator
        The fitted feature generator (inference half).
    estimator : object
        The fitted downstream estimator (predict half).
    classes_ : np.ndarray or None
        Class labels seen during :meth:`fit` (from the downstream estimator).
    """

    def __init__(
        self,
        *,
        radius: int = 6,
        method: str = "key_loo",
        k_threshold: int = 2,
        estimator=None,
        generator: Optional[MultiTaskPrevalenceGenerator] = None,
        **generator_kwargs,
    ):
        if generator is not None:
            self.generator = generator
        else:
            self.generator = MultiTaskPrevalenceGenerator(
                radius=radius, method=method, k_threshold=k_threshold, **generator_kwargs
            )
        if estimator is None:
            from sklearn.linear_model import LogisticRegression
            estimator = LogisticRegression(max_iter=1000)
        self.estimator = estimator
        self.classes_ = None
        self._fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, smiles: Sequence[str], y) -> "MolFTPClassifier":
        """Fit the feature generator and the downstream estimator.

        Parameters
        ----------
        smiles : sequence of str
            Training molecule SMILES.
        y : array-like of shape (n_samples,)
            Class labels.
        """
        smiles = _check_smiles(smiles)
        y = np.asarray(y)
        if y.ndim > 1:
            y = y.ravel()
        if len(y) != len(smiles):
            raise ValueError(f"len(y)={len(y)} != len(smiles)={len(smiles)}")

        self.generator.fit(smiles, y)
        X = np.asarray(self.generator.transform(smiles))
        self.estimator.fit(X, y)
        self.classes_ = getattr(self.estimator, "classes_", None)
        self._fitted = True
        return self

    # ------------------------------------------------------- inference half
    def transform(self, smiles: Sequence[str]) -> np.ndarray:
        """INFERENCE only: SMILES → molFTP feature matrix (no labels, no prediction)."""
        if not self._fitted:
            raise RuntimeError("call fit() before transform()")
        smiles = _check_smiles(smiles)
        return np.asarray(self.generator.transform(smiles))

    # --------------------------------------------------------- predict half
    def predict(self, smiles: Sequence[str]) -> np.ndarray:
        """SMILES → predicted class labels."""
        return self.estimator.predict(self.transform(smiles))

    def predict_proba(self, smiles: Sequence[str]) -> np.ndarray:
        """SMILES → class probabilities (requires an estimator with predict_proba)."""
        if not hasattr(self.estimator, "predict_proba"):
            raise AttributeError(
                f"{type(self.estimator).__name__} has no predict_proba; "
                "pass an estimator that supports it."
            )
        return self.estimator.predict_proba(self.transform(smiles))

    def fit_predict(self, smiles: Sequence[str], y) -> np.ndarray:
        """Convenience: ``fit(smiles, y)`` then ``predict(smiles)``."""
        return self.fit(smiles, y).predict(smiles)
