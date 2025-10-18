"""
Example: Single-Task Key-LOO Feature Generation

This example demonstrates how to use MolFTP with the Key-LOO (Key Leave-One-Out)
method for single-task molecular property prediction.

Key-LOO applies statistical filtering and rescaling to improve extrapolation
to molecules with novel fragments.
"""

from molftp import MultiTaskPrevalenceGenerator
import numpy as np

# Sample SMILES strings and binary labels
smiles = [
    "CC",           # Ethane (inactive)
    "CCC",          # Propane (active)
    "CCCC",         # Butane (inactive)
    "CCCCC",        # Pentane (active)
    "CCCCCC",       # Hexane (inactive)
    "C1CCCCC1",     # Cyclohexane (active)
    "c1ccccc1",     # Benzene (inactive)
    "CCO",          # Ethanol (active)
]
labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])

print("MolFTP Single-Task Key-LOO Example")
print("=" * 50)
print(f"Number of molecules: {len(smiles)}")
print(f"Labels: {labels}")
print()

# Generate features with Key-LOO method
# radius=6: Use Morgan fingerprints with radius 6
# method='key_loo': Apply Key-LOO filtering and rescaling
gen = MultiTaskPrevalenceGenerator(radius=6, method='key_loo')

# Fit the generator on the data
gen.fit(smiles, labels.reshape(-1, 1), task_names=['activity'])

# Transform molecules to feature vectors
features = gen.transform(smiles)

print(f"Features shape: {features.shape}")
print(f"  - {features.shape[0]} molecules")
print(f"  - {features.shape[1]} features per molecule")
print()
print(f"Example feature vector (first molecule):")
print(f"  Min: {np.min(features[0]):.4f}")
print(f"  Max: {np.max(features[0]):.4f}")
print(f"  Mean: {np.mean(features[0]):.4f}")
print(f"  Non-zero features: {np.count_nonzero(features[0])}/{features.shape[1]}")
print()

# Features can now be used with any ML model
print("✓ Features ready for use with scikit-learn, XGBoost, LightGBM, etc.")
print()
print("Next steps:")
print("  1. Use these features with your favorite ML model")
print("  2. Perform cross-validation for robust evaluation")
print("  3. Compare with other featurization methods (e.g., Morgan fingerprints)")

