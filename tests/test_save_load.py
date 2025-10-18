"""
Test save_features() and load_features() functionality
"""

import pytest
import numpy as np
import tempfile
import os
from molftp import MultiTaskPrevalenceGenerator


def test_save_load_keyloo():
    """Test save/load for Key-LOO features"""
    smiles = ["CC", "CCC", "CCCC"]
    labels = np.array([[0], [1], [0]])
    
    # Fit and save
    gen1 = MultiTaskPrevalenceGenerator(radius=2, method='key_loo')
    gen1.fit(smiles, labels)
    
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        filepath = f.name
    
    try:
        gen1.save_features(filepath)
        
        # Load
        gen2 = MultiTaskPrevalenceGenerator.load_features(filepath)
        
        # Check attributes
        assert gen2.method == 'key_loo'
        assert gen2.radius == 2
        assert gen2.is_fitted_ == True
        assert gen2.n_tasks_ == 1
        
        # Transform with both
        X1 = gen1.transform(smiles)
        X2 = gen2.transform(smiles)
        
        # Should be identical
        np.testing.assert_allclose(X1, X2, rtol=1e-10)
        print("✅ Key-LOO save/load test passed")
    finally:
        os.unlink(filepath)


def test_save_load_dummymask():
    """Test save/load for Dummy-Masking features"""
    smiles = ["CC", "CCC", "CCCC"]
    labels = np.array([[0], [1], [0]])
    train_indices = [0, 1]
    
    # Fit and save
    gen1 = MultiTaskPrevalenceGenerator(radius=2, method='dummy_masking')
    gen1.fit(smiles, labels)
    
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        filepath = f.name
    
    try:
        gen1.save_features(filepath)
        
        # Load
        gen2 = MultiTaskPrevalenceGenerator.load_features(filepath)
        
        # Check attributes
        assert gen2.method == 'dummy_masking'
        assert gen2.radius == 2
        assert gen2.is_fitted_ == True
        assert gen2.n_tasks_ == 1
        
        # Transform with both
        X1 = gen1.transform(smiles, train_indices_per_task=[train_indices])
        X2 = gen2.transform(smiles, train_indices_per_task=[train_indices])
        
        # Should be identical
        np.testing.assert_allclose(X1, X2, rtol=1e-10)
        print("✅ Dummy-Masking save/load test passed")
    finally:
        os.unlink(filepath)


def test_save_load_multitask():
    """Test save/load for multi-task features"""
    smiles = ["CC", "CCC", "CCCC", "CCCCC"]
    labels = np.array([[0, 1], [1, 0], [0, np.nan], [1, 1]])
    
    # Fit and save
    gen1 = MultiTaskPrevalenceGenerator(radius=2, method='key_loo')
    gen1.fit(smiles, labels, task_names=['task1', 'task2'])
    
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        filepath = f.name
    
    try:
        gen1.save_features(filepath)
        
        # Load
        gen2 = MultiTaskPrevalenceGenerator.load_features(filepath)
        
        # Check attributes
        assert gen2.n_tasks_ == 2
        assert gen2.task_names_ == ['task1', 'task2']
        assert gen2.is_fitted_ == True
        
        # Transform with both
        X1 = gen1.transform(smiles)
        X2 = gen2.transform(smiles)
        
        # Should be identical
        np.testing.assert_allclose(X1, X2, rtol=1e-10)
        print("✅ Multi-task save/load test passed")
    finally:
        os.unlink(filepath)


def test_save_before_fit_raises_error():
    """Test that save_features() raises error if not fitted"""
    gen = MultiTaskPrevalenceGenerator(radius=2, method='key_loo')
    
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        filepath = f.name
    
    try:
        with pytest.raises(ValueError, match="Must call fit"):
            gen.save_features(filepath)
        print("✅ Save before fit error test passed")
    finally:
        if os.path.exists(filepath):
            os.unlink(filepath)


def test_load_with_new_molecules():
    """Test that loaded generator works with different molecules"""
    train_smiles = ["CC", "CCC", "CCCC"]
    train_labels = np.array([[0], [1], [0]])
    test_smiles = ["CCCCC", "CCCCCC"]  # Different molecules
    
    # Fit and save
    gen1 = MultiTaskPrevalenceGenerator(radius=2, method='key_loo')
    gen1.fit(train_smiles, train_labels)
    
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        filepath = f.name
    
    try:
        gen1.save_features(filepath)
        
        # Load
        gen2 = MultiTaskPrevalenceGenerator.load_features(filepath)
        
        # Transform new molecules with both generators
        X1 = gen1.transform(test_smiles)
        X2 = gen2.transform(test_smiles)
        
        # Should be identical
        np.testing.assert_allclose(X1, X2, rtol=1e-10)
        print("✅ New molecules test passed")
    finally:
        os.unlink(filepath)


if __name__ == '__main__':
    # Run tests manually
    test_save_load_keyloo()
    test_save_load_dummymask()
    test_save_load_multitask()
    test_save_before_fit_raises_error()
    test_load_with_new_molecules()
    print("\n✅ All save/load tests passed!")
