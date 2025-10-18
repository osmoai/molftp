"""
Basic tests for MolFTP functionality
"""

import pytest
import numpy as np
from molftp import MultiTaskPrevalenceGenerator


def test_import():
    """Test that the package imports correctly."""
    from molftp import MultiTaskPrevalenceGenerator
    assert MultiTaskPrevalenceGenerator is not None


def test_single_task_keyloo():
    """Test single-task Key-LOO feature generation."""
    smiles = ["CC", "CCC", "CCCC"]
    labels = np.array([0, 1, 0])
    
    gen = MultiTaskPrevalenceGenerator(radius=2, method='key_loo')
    gen.fit(smiles, labels.reshape(-1, 1), task_names=['test'])
    features = gen.transform(smiles)
    
    assert features.shape[0] == len(smiles)
    assert features.shape[1] > 0
    assert not np.any(np.isnan(features))


def test_multitask_keyloo():
    """Test multi-task Key-LOO feature generation."""
    smiles = ["CC", "CCC", "CCCC"]
    labels = np.array([
        [0, 1],
        [1, 0],
        [0, 1]
    ], dtype=float)
    
    gen = MultiTaskPrevalenceGenerator(radius=2, method='key_loo')
    gen.fit(smiles, labels, task_names=['task1', 'task2'])
    features = gen.transform(smiles)
    
    assert features.shape[0] == len(smiles)
    assert features.shape[1] > 0
    assert not np.any(np.isnan(features))


def test_single_task_dummymask():
    """Test single-task Dummy-Masking feature generation."""
    smiles = ["CC", "CCC", "CCCC"]
    labels = np.array([0, 1, 0])
    
    gen = MultiTaskPrevalenceGenerator(radius=2, method='dummy_masking')
    gen.fit(smiles, labels.reshape(-1, 1), task_names=['test'])
    
    train_indices = [0, 1]  # First 2 molecules
    features = gen.transform(smiles, train_indices_per_task=[train_indices])
    
    assert features.shape[0] == len(smiles)
    assert features.shape[1] > 0
    assert not np.any(np.isnan(features))


def test_multitask_sparse_labels():
    """Test multi-task with sparse labels (NaN)."""
    smiles = ["CC", "CCC", "CCCC"]
    labels = np.array([
        [0, np.nan],
        [1, 0],
        [0, 1]
    ], dtype=float)
    
    gen = MultiTaskPrevalenceGenerator(radius=2, method='key_loo')
    gen.fit(smiles, labels, task_names=['task1', 'task2'])
    features = gen.transform(smiles)
    
    assert features.shape[0] == len(smiles)
    assert features.shape[1] > 0
    assert not np.any(np.isnan(features))


def test_feature_consistency():
    """Test that features are consistent across multiple calls."""
    smiles = ["CC", "CCC"]
    labels = np.array([0, 1])
    
    gen = MultiTaskPrevalenceGenerator(radius=2, method='key_loo')
    gen.fit(smiles, labels.reshape(-1, 1), task_names=['test'])
    
    features1 = gen.transform(smiles)
    features2 = gen.transform(smiles)
    
    np.testing.assert_array_equal(features1, features2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

