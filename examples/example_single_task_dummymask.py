"""
Example: Single-Task Dummy-Masking with Cross-Validation

This example demonstrates how to use MolFTP with the Dummy-Masking method
for single-task molecular property prediction within a cross-validation loop.

Dummy-Masking builds features on all data but masks test-only keys per fold,
providing a fair evaluation while maximizing statistical power.
"""

from molftp import MultiTaskPrevalenceGenerator
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Sample SMILES and labels
smiles = [
    "CC", "CCC", "CCCC", "CCCCC", "CCCCCC",
    "C1CCCCC1", "c1ccccc1", "CCO", "CCCO", "CCCCO",
    "CC(C)C", "CC(C)CC", "c1ccc(C)cc1", "c1ccc(O)cc1", "CCN",
    "CCCN", "CC(=O)C", "CCC(=O)C", "CCOC", "CCCOC"
]
labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

print("MolFTP Single-Task Dummy-Masking with CV Example")
print("=" * 60)
print(f"Number of molecules: {len(smiles)}")
print(f"Positive samples: {np.sum(labels)}/{len(labels)}")
print()

# 5-fold cross-validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

print(f"Running {n_splits}-fold cross-validation...")
print()

fold_scores = []

for fold, (train_idx, test_idx) in enumerate(skf.split(smiles, labels), 1):
    # Prepare fold data
    X_train = [smiles[i] for i in train_idx]
    X_test = [smiles[i] for i in test_idx]
    X_all = X_train + X_test
    y_train = labels[train_idx]
    y_test = labels[test_idx]
    y_all = np.concatenate([y_train, y_test])
    
    # Build prevalence on ALL data (train + test)
    gen = MultiTaskPrevalenceGenerator(radius=6, method='dummy_masking')
    gen.fit(X_all, y_all.reshape(-1, 1), task_names=['activity'])
    
    # Transform with train indices (masks test-only keys)
    train_indices = list(range(len(X_train)))
    features = gen.transform(X_all, train_indices_per_task=[train_indices])
    
    # Split features
    X_train_feat = features[:len(X_train)]
    X_test_feat = features[len(X_train):]
    
    # Simple logistic regression for demonstration
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_feat, y_train)
    
    # Predict and evaluate
    y_pred_proba = model.predict_proba(X_test_feat)[:, 1]
    score = roc_auc_score(y_test, y_pred_proba)
    fold_scores.append(score)
    
    print(f"Fold {fold}/{n_splits}: AUROC = {score:.4f}")

print()
print(f"Mean AUROC: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
print()
print("✓ Dummy-Masking ensures fair CV by masking test-only keys per fold")

