"""Regression + edge-case tests that pin the behaviour fixed in the hardening pass.

Each test names the issue it guards against so a future change that reintroduces the bug fails
loudly here.
"""
import pickle

import numpy as np
import pytest

_molftp = pytest.importorskip("_molftp")
from molftp.prevalence import MultiTaskPrevalenceGenerator

SMI = ['CCO', 'CCN', 'CCC', 'c1ccccc1', 'c1ccccc1O', 'CC(=O)O', 'CCCl', 'CCBr', 'CCF',
       'CCCCO', 'CCCCN', 'c1ccncc1', 'CCCO', 'CCCN']
Y = np.array([[i % 2] for i in range(len(SMI))], dtype=float)


def _fit(k_threshold=2, method='key_loo'):
    g = MultiTaskPrevalenceGenerator(radius=2, k_threshold=k_threshold, method=method)
    g.fit(SMI, Y, task_names=['t'])
    return g


# --- k_threshold (was silently ignored in 815f951) ------------------------------------------
def test_k_threshold_accepted_by_cpp_constructor():
    g = _molftp.MultiTaskPrevalenceGenerator(radius=2, k_threshold=7, use_key_loo=True, verbose=False)
    assert g is not None


def test_k_threshold_changes_features():
    X1 = np.asarray(_fit(k_threshold=1).transform(SMI))
    X5 = np.asarray(_fit(k_threshold=5).transform(SMI))
    assert not np.allclose(X1, X5), "k_threshold has no effect on features — regression of 815f951"


# --- chi2 vs chisq aliasing (sequential path fell through to Fisher) -------------------------
def test_chi2_threaded_equals_sequential():
    vg = _molftp.VectorizedFTPGenerator(nBits=2048, sim_thresh=0.5, max_pairs=1000, max_triplets=1000)
    labels = [int(v[0]) for v in Y]
    seq = vg.build_1d_ftp_stats(SMI, labels, 2, "chi2", 0.5)
    thr = vg.build_1d_ftp_stats_threaded(SMI, labels, 2, "chi2", 0.5, num_threads=2)
    assert set(seq) == set(thr)
    for k in seq:
        assert abs(seq[k] - thr[k]) < 1e-9


# --- pickle / save-load (separate __getstate__/__setstate__ didn't restore state) -----------
def test_pickle_roundtrip_preserves_features():
    g = _fit()
    g2 = pickle.loads(pickle.dumps(g))
    assert g2.is_fitted_  # Python wrapper exposes is_fitted_ (the C++ object uses is_fitted())
    np.testing.assert_allclose(np.asarray(g.transform(SMI)), np.asarray(g2.transform(SMI)), atol=1e-10)


# --- out-of-sample inference collapse (dummy_masking train_indices misuse) -------------------
def test_out_of_sample_inference_no_collapse():
    g = MultiTaskPrevalenceGenerator(radius=2, method='dummy_masking')
    g.fit(SMI[:10], Y[:10], task_names=['t'])
    Xte = np.asarray(g.transform(SMI[10:]))  # smaller, unseen batch — must not collapse/overflow
    assert Xte.shape[0] == len(SMI[10:])
    assert np.isfinite(Xte).all()


# --- determinism --------------------------------------------------------------------------
def test_transform_is_deterministic():
    g = _fit()
    np.testing.assert_array_equal(np.asarray(g.transform(SMI)), np.asarray(g.transform(SMI)))


# --- quiet by default (cout was ungated in transform/fit) -----------------------------------
def test_quiet_by_default(capfd):
    g = _fit()
    g.transform(SMI)
    out, _ = capfd.readouterr()
    for banner in ("TRANSFORMING TO MULTI-TASK", "Multi-task features created", "ALL TASK PREVALENCE BUILT"):
        assert banner not in out, f"unexpected stdout banner: {banner!r}"


# --- edge cases ---------------------------------------------------------------------------
def test_single_molecule():
    g = _fit()
    X = np.asarray(g.transform(['CCO']))
    assert X.shape[0] == 1


def test_invalid_smiles_do_not_crash():
    g = MultiTaskPrevalenceGenerator(radius=2, method='dummy_masking')
    smi = ['CCO', 'not_a_smiles', 'c1ccccc1', '', 'CCCl']
    y = np.array([[0], [1], [0], [1], [0]], dtype=float)
    g.fit(smi, y, task_names=['t'])
    X = np.asarray(g.transform(smi))
    assert X.shape[0] == len(smi) and np.isfinite(X).all()


def test_inference_independent_of_batch():
    # A molecule embedded alone must match its embedding within a larger batch.
    g = _fit()
    i = 3
    x_single = np.asarray(g.transform([SMI[i]]))[0]
    x_batch = np.asarray(g.transform(SMI))[i]
    np.testing.assert_allclose(x_single, x_batch, atol=1e-10)
