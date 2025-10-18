"""
Example: Multi-Task Dummy-Masking with Cross-Validation

This example demonstrates multi-task learning with Dummy-Masking in a
cross-validation setting. This is the most advanced usage of MolFTP.

Dummy-Masking builds features on all data but masks test-only keys per fold
and per task, handling sparse labels (NaN) correctly.
"""

from molftp import MultiTaskPrevalenceGenerator
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Sample SMILES
smiles = [
    "CC", "CCC", "CCCC", "CCCCC", "CCCCCC",
    "C1CCCCC1", "c1ccccc1", "CCO", "CCCO", "CCCCO",
    "CC(C)C", "CC(C)CC", "c1ccc(C)cc1", "c1ccc(O)cc1", "CCN"
]

# Multi-task labels with sparse data (NaN = not measured)
labels = np.array([
    [0, 1, np.nan],      # Task 3 not measured
    [1, 1, 0],
    [0, np.nan, 1],      # Task 2 not measured
    [1, 0, 1],
    [0, 1, np.nan],
    [1, 1, 1],
    [0, 0, 1],
    [1, np.nan, 0],
    [1, 1, 0],
    [0, 1, 1],
    [1, 0, np.nan],
    [0, 1, 0],
    [1, 0, 1],
    [0, np.nan, 1],
    [1, 1, 0],
], dtype=float)

task_names = ['activity', 'solubility', 'toxicity']

print("MolFTP Multi-Task Dummy-Masking with CV Example")
print("=" * 60)
print(f"Number of molecules: {len(smiles)}")
print(f"Number of tasks: {labels.shape[1]}")
print(f"Task names: {task_names}")
print()

# Check sparsity per task
for i, task_name in enumerate(task_names):
    n_measured = np.sum(~np.isnan(labels[:, i]))
    print(f"  {task_name:12s}: {n_measured}/{len(labels)} measured ({100*n_measured/len(labels):.1f}%)")
print()

# 3-fold CV (using only Task 1 for stratification)
n_splits = 3
task1_labels_for_split = labels[:, 0].copy()
# For stratification, temporarily fill NaN with 0
task1_labels_for_split[np.isnan(task1_labels_for_split)] = 0

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

print(f"Running {n_splits}-fold cross-validation...")
print()

fold_results = {task: [] for task in task_names}

for fold, (train_idx, test_idx) in enumerate(skf.split(smiles, task1_labels_for_split), 1):
    print(f"Fold {fold}/{n_splits}:")
    
    # Prepare data
    X_train = [smiles[i] for i in train_idx]
    X_test = [smiles[i] for i in test_idx]
    X_all = X_train + X_test
    Y_train = labels[train_idx]
    Y_test = labels[test_idx]
    Y_all = np.vstack([Y_train, Y_test])
    
    # Build multi-task prevalence
    gen = MultiTaskPrevalenceGenerator(radius=6, method='dummy_masking')
    gen.fit(X_all, Y_all, task_names=task_names)
    
    # Compute train indices per task (excluding NaN labels)
    train_indices_per_task = [
        [i for i in range(len(X_train)) if not np.isnan(Y_train[i, task_idx])]
        for task_idx in range(len(task_names))
    ]
    
    # Transform with per-task masking
    features = gen.transform(X_all, train_indices_per_task=train_indices_per_task)
    
    X_train_feat = features[:len(X_train)]
    X_test_feat = features[len(X_train):]
    
    # Evaluate each task separately
    for task_idx, task_name in enumerate(task_names):
        # Get valid test samples for this task
        valid_test_mask = ~np.isnan(Y_test[:, task_idx])
        if valid_test_mask.sum() < 2:
            continue
        
        y_test_task = Y_test[valid_test_mask, task_idx]
        X_test_task = X_test_feat[valid_test_mask]
        
        # Get valid train samples for this task
        valid_train_mask = ~np.isnan(Y_train[:, task_idx])
        y_train_task = Y_train[valid_train_mask, task_idx]
        X_train_task = X_train_feat[valid_train_mask]
        
        # Train simple model
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_task, y_train_task)
        
        # Evaluate
        y_pred_proba = model.predict_proba(X_test_task)[:, 1]
        score = roc_auc_score(y_test_task, y_pred_proba)
        fold_results[task_name].append(score)
        
        print(f"  {task_name:12s}: AUROC = {score:.4f}")
    print()

# Print summary
print("Summary across folds:")
for task_name in task_names:
    scores = fold_results[task_name]
    if scores:
        print(f"  {task_name:12s}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
print()
print("✓ Multi-task Dummy-Masking handles sparse labels correctly")

