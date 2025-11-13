"""
Test indexed miners produce identical results to legacy O(N²) scans.

This test verifies that the new indexed exact Tanimoto search produces
identical feature matrices to the legacy implementation, ensuring correctness
of the performance optimization.
"""

import os
import random
import numpy as np
import pytest

# Try to import molftp
try:
    import molftp
    from molftp.prevalence import MultiTaskPrevalenceGenerator
    MOLFTP_AVAILABLE = True
except ImportError:
    MOLFTP_AVAILABLE = False
    pytest.skip("molftp not available", allow_module_level=True)

def make_synthetic(n=200, pos_ratio=0.3, seed=0):
    """Create synthetic SMILES dataset with deterministic labels."""
    # Simple, valid chains: "CCC...", deterministic
    smiles = ["C" * k for k in range(3, 3 + n)]
    labels = np.array([1 if (i / n) < pos_ratio else 0 for i in range(n)], dtype=int)
    
    # Shuffle deterministically so PASS/FAIL are mixed
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    smiles = [smiles[i] for i in order]
    labels = labels[order]
    
    return smiles, labels

def run_fit_transform(force_legacy=False, seed=42):
    """Run fit/transform with specified legacy flag."""
    if force_legacy:
        os.environ["MOLFTP_FORCE_LEGACY_SCAN"] = "1"
    else:
        os.environ.pop("MOLFTP_FORCE_LEGACY_SCAN", None)
    
    smiles, y = make_synthetic(n=200, seed=seed)
    
    # Split 80/20
    n = len(smiles)
    ntr = int(0.8 * n)
    Xtr, Xva = smiles[:ntr], smiles[ntr:]
    ytr, yva = y[:ntr], y[ntr:]
    
    # Use deterministic settings
    gen = MultiTaskPrevalenceGenerator(
        radius=6,
        nBits=2048,
        sim_thresh=0.7,
        num_threads=-1,
        method='dummy_masking'
    )
    
    # Fit on training data
    gen.fit(Xtr, ytr.reshape(-1, 1), task_names=['task1'])
    
    # Transform both train and validation
    Ftr = gen.transform(Xtr)
    Fva = gen.transform(Xva)
    
    # Return both matrices concatenated to compare end-to-end
    return np.vstack([Ftr, Fva])

@pytest.mark.fast
def test_indexed_vs_legacy_features_identical():
    """Test that indexed and legacy miners produce identical features."""
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Run with legacy scan
    F_legacy = run_fit_transform(force_legacy=True, seed=42)
    
    # Run with indexed scan (default)
    F_index = run_fit_transform(force_legacy=False, seed=42)
    
    # Check shapes match
    assert F_index.shape == F_legacy.shape, \
        f"Shape mismatch: indexed={F_index.shape}, legacy={F_legacy.shape}"
    
    # Check exact equality (within floating point tolerance)
    np.testing.assert_allclose(
        F_index, F_legacy, 
        rtol=0, atol=1e-10,
        err_msg="Indexed and legacy miners produced different features"
    )
    
    print(f"✅ Test passed: {F_index.shape[0]} samples, {F_index.shape[1]} features")
    print(f"   Max absolute difference: {np.max(np.abs(F_index - F_legacy)):.2e}")

@pytest.mark.fast
def test_indexed_miners_produce_features():
    """Sanity check: indexed miners produce non-zero features."""
    random.seed(42)
    np.random.seed(42)
    
    F = run_fit_transform(force_legacy=False, seed=42)
    
    # Check we have features
    assert F.shape[0] > 0, "No samples in feature matrix"
    assert F.shape[1] > 0, "No features in feature matrix"
    
    # Check at least some non-zero features
    assert np.any(F != 0), "All features are zero"
    
    print(f"✅ Sanity check passed: {F.shape[0]} samples, {F.shape[1]} features")
    print(f"   Non-zero features: {np.count_nonzero(F)} / {F.size}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

