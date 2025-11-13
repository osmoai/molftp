import pickle
import numpy as np
import pytest

molftp = pytest.importorskip("_molftp")
VectorizedFTPGenerator = molftp.VectorizedFTPGenerator
MultiTaskPrevalenceGenerator = molftp.MultiTaskPrevalenceGenerator


def test_pickle_round_trip(mtpg, smiles):
    # State round-trip through __getstate__/__setstate__ preserves transform
    state = mtpg.__getstate__()
    mtpg2 = MultiTaskPrevalenceGenerator()  # defaults; __setstate__ will overwrite
    mtpg2.__setstate__(state)

    X1 = mtpg.transform(smiles)
    X2 = mtpg2.transform(smiles)
    np.testing.assert_allclose(X1, X2, rtol=1e-8, atol=1e-10)

    # Also check with a training mask
    mask = [True] * (len(smiles) // 2) + [False] * (len(smiles) - len(smiles) // 2)
    X1m = mtpg.transform(smiles, train_row_mask=mask)
    X2m = mtpg2.transform(smiles, train_row_mask=mask)
    np.testing.assert_allclose(X1m, X2m, rtol=1e-8, atol=1e-10)


def test_threaded_vs_sequential_1d(vecgen: VectorizedFTPGenerator, smiles, labels):
    # Threaded 1D prevalence equals sequential (within fp rounding)
    prev_seq = vecgen.build_1d_ftp_stats(smiles, labels.tolist(), 2, "chi2", 0.5)
    prev_thr = vecgen.build_1d_ftp_stats_threaded(smiles, labels.tolist(), 2, "chi2", 0.5, num_threads=2)

    # Same key sets
    assert set(prev_seq.keys()) == set(prev_thr.keys())
    # Values very close
    for k in prev_seq.keys():
        assert abs(prev_seq[k] - prev_thr[k]) < 1e-9, f"Mismatch for key {k}: {prev_seq[k]} vs {prev_thr[k]}"


def test_invalid_smiles_do_not_crash(vecgen: VectorizedFTPGenerator, radius):
    smiles = ["invalid!!", "CCO", "c1ccccc1", "CCCl", "BrCCBr"]
    labels = [0, 1, 0, 1, 1]
    # Should not raise
    prev = vecgen.build_1d_ftp_stats(smiles, labels, radius, "chi2", 0.5)
    assert isinstance(prev, dict)
    assert len(prev) > 0  # some valid keys must be present

