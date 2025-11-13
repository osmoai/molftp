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
    n = len(smiles)
    train_mask = np.array([True] * (n // 2) + [False] * (n - n // 2), dtype=bool)

    X_mask = mtpg.transform(smiles, train_row_mask=train_mask)
    X_nomask = mtpg.transform(smiles)  # inference: no rescaling

    # Validation rows (mask False) must be identical with/without mask
    idx_val = np.where(~train_mask)[0]
    np.testing.assert_allclose(X_mask[idx_val], X_nomask[idx_val], rtol=1e-8, atol=1e-10)

    # Training rows (mask True) should differ
    idx_tr = np.where(train_mask)[0]
    diff = np.abs(X_mask[idx_tr] - X_nomask[idx_tr]).mean()
    assert diff > 1e-9, f"Expected noticeable rescaling difference, got mean Δ={diff:.3e}"

    # Mask all False equals no mask (backward compatibility)
    X_falsemask = mtpg.transform(smiles, train_row_mask=[False] * n)
    np.testing.assert_allclose(X_falsemask, X_nomask, rtol=1e-8, atol=1e-10)


def test_inference_independence_from_batch(mtpg, smiles):
    # Embedding a molecule alone vs embedded in a batch must be identical
    i = len(smiles) // 3
    x_single = mtpg.transform([smiles[i]])[0]
    X_batch = mtpg.transform(smiles)
    np.testing.assert_allclose(x_single, X_batch[i], rtol=1e-8, atol=1e-10)


def test_2d_features_are_nonzero(mtpg, smiles, radius):
    fpv, s1d, s2d, s3d = _features_slices(radius)
    X = mtpg.transform(smiles)
    nonzero_ratio_2d = (np.abs(X[:, s2d]) > 0).mean()
    assert nonzero_ratio_2d > 0.05, f"2D view looks empty (ratio={nonzero_ratio_2d:.3f})"


def test_2d_keys_are_subset_of_1d(vecgen: VectorizedFTPGenerator, smiles, labels, radius):
    # 1D prevalence keys
    prev1 = vecgen.build_1d_ftp_stats(smiles, labels.tolist(), radius, "chi2", 0.5)

    # 2D pairs with a fixed seed → deterministic
    pairs = vecgen.make_pairs_balanced_cpp(smiles, labels.tolist(), 2, 2048, 0.5, seed=0)
    prev2 = vecgen.build_2d_ftp_stats(smiles, labels.tolist(), pairs, radius, prev1, "mcnemar_midp", 0.5)

    k1 = set(prev1.keys())
    k2 = set(prev2.keys())
    assert k2.issubset(k1), "2D prevalence should be computed on 1D single keys; got keys outside 1D library"


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

