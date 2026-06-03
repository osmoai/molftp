"""molftp.features — the INFERENCE layer (SMILES → feature vectors).

This is the feature-generation half of molFTP: it learns fragment-target prevalence on
training data (``fit``) and emits fixed-length feature vectors for any molecules
(``transform``). It does NOT predict labels — that is the job of :mod:`molftp.predict`.

    inference  (this module)   SMILES ──fit/transform──▶ feature vectors
    predict    (molftp.predict)  SMILES ──features──estimator──▶ labels

The generators themselves live in :mod:`molftp.prevalence`; this module simply re-exports them
under an intention-revealing name so the two layers have distinct import paths:

    from molftp.features import MultiTaskPrevalenceGenerator   # multi-task (NaN-sparse labels)
    from molftp.features import PrevalenceGenerator            # single-task
"""

from .prevalence import PrevalenceGenerator, MultiTaskPrevalenceGenerator

__all__ = ["PrevalenceGenerator", "MultiTaskPrevalenceGenerator"]
