"""
Example: Multi-Task Key-LOO Feature Generation

This example demonstrates how to use MolFTP for multi-task learning with the
Key-LOO method. Multiple related tasks are handled simultaneously, potentially
improving performance through shared representations.

Ideal for scenarios where molecules have labels for multiple related properties
(e.g., multiple bioactivity endpoints, multiple physicochemical properties).
"""

from molftp import MultiTaskPrevalenceGenerator
import numpy as np

# Sample SMILES strings
smiles = [
    "CC", "CCC", "CCCC", "CCCCC", "CCCCCC",
    "C1CCCCC1", "c1ccccc1", "CCO", "CCCO", "CCCCO"
]

# Multi-task labels (3 tasks: activity, solubility, toxicity)
# Shape: (n_molecules, n_tasks)
labels = np.array([
    [0, 1, 0],  # CC
    [1, 1, 0],  # CCC
    [0, 0, 1],  # CCCC
    [1, 0, 1],  # CCCCC
    [0, 1, 0],  # CCCCCC
    [1, 1, 1],  # Cyclohexane
    [0, 0, 1],  # Benzene
    [1, 1, 0],  # Ethanol
    [1, 1, 0],  # Propanol
    [0, 1, 1],  # Butanol
], dtype=float)

task_names = ['activity', 'solubility', 'toxicity']

print("MolFTP Multi-Task Key-LOO Example")
print("=" * 60)
print(f"Number of molecules: {len(smiles)}")
print(f"Number of tasks: {labels.shape[1]}")
print(f"Task names: {task_names}")
print()

# Generate multi-task features with Key-LOO
gen = MultiTaskPrevalenceGenerator(radius=6, method='key_loo')

# Fit on all tasks simultaneously
gen.fit(smiles, labels, task_names=task_names)

# Transform to multi-task feature vectors
features = gen.transform(smiles)

print(f"Multi-task features shape: {features.shape}")
print(f"  - {features.shape[0]} molecules")
print(f"  - {features.shape[1]} total features")
print(f"  - {features.shape[1] // len(task_names)} features per task")
print()

# Features are concatenated: [task1_features | task2_features | task3_features]
n_features_per_task = features.shape[1] // len(task_names)

print("Feature breakdown by task:")
for i, task_name in enumerate(task_names):
    start_idx = i * n_features_per_task
    end_idx = (i + 1) * n_features_per_task
    task_features = features[:, start_idx:end_idx]
    
    print(f"  {task_name:12s}: columns {start_idx:3d}-{end_idx:3d}, "
          f"mean={np.mean(task_features):.4f}, "
          f"non-zero={np.count_nonzero(task_features[0])}/{n_features_per_task}")
print()

print("✓ Multi-task features ready for use with multi-task models")
print()
print("Usage with XGBoost multi-task:")
print("  - Train a single model on all tasks simultaneously")
print("  - Handle sparse labels (NaN) with custom objective functions")
print("  - Benefit from shared representations across related tasks")

