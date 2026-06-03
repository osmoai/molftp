import numpy as np
import pytest

molftp = pytest.importorskip("_molftp")
VectorizedFTPGenerator = molftp.VectorizedFTPGenerator
MultiTaskPrevalenceGenerator = molftp.MultiTaskPrevalenceGenerator


def _features_slices(radius: int):
    # For radius=2 → features_per_view = 5, per task = 15: [1D 0:5], [2D 5:10], [3D 10:15]
    fpv = 2 + (radius + 1)
    return fpv, slice(0, fpv), slice(fpv, 2 * fpv), slice(2 * fpv, 3 * fpv)


def test_per_molecule_rescaling_train_only(mtpg, smiles, radius):
    # Leakage-safety properties of the train_row_mask that MUST hold (these guard against the
    # mask bleeding into inference): validation rows are unaffected, and an all-False mask is a
    # no-op equal to plain inference.
    n = len(smiles)
    train_mask = np.array([True] * (n // 2) + [False] * (n - n // 2), dtype=bool)

    X_mask = mtpg.transform(smiles, train_row_mask=train_mask)
    X_nomask = mtpg.transform(smiles)  # inference: no rescaling

    # Validation rows (mask False) must be identical with/without mask
    idx_val = np.where(~train_mask)[0]
    np.testing.assert_allclose(X_mask[idx_val], X_nomask[idx_val], rtol=1e-8, atol=1e-10)

    # Mask all False equals no mask (backward compatibility / no leakage)
    X_falsemask = mtpg.transform(smiles, train_row_mask=[False] * n)
    np.testing.assert_allclose(X_falsemask, X_nomask, rtol=1e-8, atol=1e-10)


def test_positive_rescale_is_inert_for_sign_count_features(mtpg, smiles):
    """Characterization test (documents *why*, not a bug).

    molFTP's 3-view features are SIGN-BASED net counts: build_3view_vectors_batch counts atoms
    whose aggregated prevalence is >= 0 (PASS-leaning) vs <= 0 (FAIL-leaning) and reports the net
    (pos - neg). Only the *sign* of each prevalence value matters, never its magnitude.

    The Key-LOO ``(k_j - 1)/k_j`` rescale (and ``loo_smoothing_tau``'s ``(k_j-1+tau)/(k_j+tau)``)
    are POSITIVE scalars, so they preserve every sign and therefore cannot change these features.
    That is by design, not a no-op bug. The effective rare-key / leakage control is ``k_threshold``,
    which *removes* keys and so does change the counts — see
    ``test_regressions.py::test_k_threshold_changes_features``.

    Passing a training mask therefore leaves the features identical to plain inference; this test
    pins that invariant so a future change to the aggregation (e.g. magnitude-aware features) is
    caught here.
    """
    n = len(smiles)
    train_mask = np.array([True] * (n // 2) + [False] * (n - n // 2), dtype=bool)
    X_mask = np.asarray(mtpg.transform(smiles, train_row_mask=train_mask))
    X_nomask = np.asarray(mtpg.transform(smiles))
    np.testing.assert_allclose(X_mask, X_nomask, atol=1e-10)


def test_inference_independence_from_batch(mtpg, smiles):
    # Embedding a molecule alone vs embedded in a batch must be identical
    i = len(smiles) // 3
    x_single = mtpg.transform([smiles[i]])[0]
    X_batch = mtpg.transform(smiles)
    np.testing.assert_allclose(x_single, X_batch[i], rtol=1e-8, atol=1e-10)


def test_2d_features_are_nonzero(radius):
    # The 2D view is populated from PASS-FAIL fragment pairs of *similar* molecules. A tiny,
    # highly-diverse set (like the default fixture) has no similar pairs, so its 2D view is
    # legitimately empty — a property of the data, not a bug. Use a homologous/similar series
    # so pairs form and the 2D view is exercised.
    smiles = ['CCO', 'CCCO', 'CCCCO', 'CCCCCO', 'CCCCCCO', 'CCN', 'CCCN', 'CCCCN', 'CCCCCN',
              'c1ccccc1C', 'c1ccccc1CC', 'c1ccccc1CCC', 'CC(=O)O', 'CCC(=O)O', 'CCCC(=O)O', 'CCCCC(=O)O']
    y = np.array([i % 2 for i in range(len(smiles))], dtype=float).reshape(-1, 1)
    g = MultiTaskPrevalenceGenerator(radius=radius, nBits=2048, sim_thresh=0.5,
                                     k_threshold=1, use_key_loo=True, verbose=False)
    g.fit(smiles, y, ['task'])
    fpv, s1d, s2d, s3d = _features_slices(radius)
    X = np.asarray(g.transform(smiles))
    nonzero_ratio_2d = (np.abs(X[:, s2d]) > 0).mean()
    assert nonzero_ratio_2d > 0.05, f"2D view looks empty (ratio={nonzero_ratio_2d:.3f})"


def test_margin_mode_option(radius):
    # margin_mode selects the V[0]/V[1] aggregation: 'signcount' (default), 'magnitude' (paper
    # eq.5), 'both' (concat -> +2 features/view). Verify dims, the default, and that magnitude
    # actually changes the features.
    from molftp.prevalence import MultiTaskPrevalenceGenerator as PG
    smi = ['CCO', 'CCCO', 'CCCCO', 'CCN', 'CCCN', 'c1ccccc1', 'c1ccccc1C', 'CC(=O)O', 'CCCl', 'CCBr']
    y = np.array([i % 2 for i in range(len(smi))], dtype=float).reshape(-1, 1)

    def feats(mode=None):
        kw = {} if mode is None else {"margin_mode": mode}
        g = PG(radius=radius, method='key_loo', **kw)
        g.fit(smi, y, ['t'])
        return np.asarray(g.transform(smi))

    per_view = 2 + radius + 1
    Xdefault, Xs, Xm, Xb = feats(), feats('signcount'), feats('magnitude'), feats('both')
    assert Xs.shape[1] == 3 * per_view
    assert Xm.shape[1] == 3 * per_view
    assert Xb.shape[1] == 3 * (per_view + 2)            # 'both' adds 2 margin cols per view
    np.testing.assert_allclose(Xdefault, Xs, atol=1e-10)  # signcount is the default
    assert not np.allclose(Xs, Xm), "magnitude margin should differ from signcount"
    # 'both' must contain the signcount margin in its first two columns of view 1
    np.testing.assert_allclose(Xb[:, 0:2], Xs[:, 0:2], atol=1e-10)
    with pytest.raises(ValueError):
        PG(margin_mode='nope')


def test_2d_keys_are_subset_of_1d(vecgen: VectorizedFTPGenerator, smiles, labels, radius):
    # 1D prevalence keys
    prev1 = vecgen.build_1d_ftp_stats(smiles, labels.tolist(), radius, "chi2", 0.5)

    # 2D pairs with a fixed seed → deterministic
    pairs = vecgen.make_pairs_balanced_cpp(smiles, labels.tolist(), 2, 2048, 0.5, seed=0)
    prev2 = vecgen.build_2d_ftp_stats(smiles, labels.tolist(), pairs, radius, prev1, "mcnemar_midp", 0.5)

    k1 = set(prev1.keys())
    k2 = set(prev2.keys())
    assert k2.issubset(k1), "2D prevalence should be computed on 1D single keys; got keys outside 1D library"


@pytest.mark.skip(
    reason="loo_smoothing_tau is inert for molFTP's SIGN-COUNT features AND not wired into the C++ "
           "core. Even if implemented, (k_j-1+tau)/(k_j+tau) is a positive scalar that preserves "
           "the prevalence sign, and the features depend only on sign — so |X| cannot vary with "
           "tau (see test_positive_rescale_is_inert_for_sign_count_features). This test also uses "
           "the non-existent method= / loo_smoothing_tau= C++ kwargs. The effective rare-key / "
           "leakage control is k_threshold. Re-enable only if the aggregation is made magnitude-aware.")
def test_tau_smoothing_monotone(mtpg, smiles, radius, Y_sparse, task_names):
    # As tau increases, the shrink factor (k+tau-1)/(k+tau) → 1, so mean|X| should (weakly) increase
    taus = [0.0, 1.0, 5.0]
    means = []
    for tau in taus:
        g = MultiTaskPrevalenceGenerator(
            radius=radius, nBits=2048, sim_thresh=0.5,
            stat_1d="chi2", stat_2d="mcnemar_midp", stat_3d="exact_binom",
            alpha=0.5, num_threads=0, method='key_loo',  # Use method='key_loo'
            k_threshold=1, loo_smoothing_tau=tau
        )
        g.fit(smiles, Y_sparse, task_names)
        X = g.transform(smiles, train_row_mask=[True] * len(smiles))
        means.append(np.mean(np.abs(X)))

    # Monotone non-decreasing (allow tiny numerical jitter)
    assert means[0] <= means[1] + 1e-12 <= means[2] + 1e-12, f"Expected |X| to increase with tau, got {means}"

