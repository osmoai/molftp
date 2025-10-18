"""
Complete ML workflow: MolFTP features + XGBoost

This example demonstrates a full machine learning pipeline with XGBoost:
1. Build MolFTP features using Dummy-Masking
2. Save features for reuse
3. Train XGBoost classifier
4. Evaluate on test set
5. Apply to completely new molecules
"""

from molftp import MultiTaskPrevalenceGenerator
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

print("=" * 70)
print("MolFTP + XGBoost: Complete ML Workflow")
print("=" * 70)

# Sample molecular data
smiles = [
    "CC",           # Ethane
    "CCC",          # Propane
    "CCCC",         # Butane
    "CCCCC",        # Pentane
    "CCCCCC",       # Hexane
    "C1CCCCC1",     # Cyclohexane
    "c1ccccc1",     # Benzene
    "CCO",          # Ethanol
    "CCCO",         # Propanol
    "CCCCO",        # Butanol
]
labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

print(f"\nDataset: {len(smiles)} molecules")
print(f"Labels: {labels.sum()} positive, {len(labels) - labels.sum()} negative")

# Split data into train/test
train_smiles, test_smiles, y_train, y_test = train_test_split(
    smiles, labels, test_size=0.3, random_state=42, stratify=labels
)

print(f"\nTrain set: {len(train_smiles)} molecules")
print(f"Test set: {len(test_smiles)} molecules")

# Step 1: Build MolFTP features using Dummy-Masking method
print("\n" + "=" * 70)
print("Step 1: Building MolFTP Features (Dummy-Masking)")
print("=" * 70)
gen = MultiTaskPrevalenceGenerator(radius=6, method='dummy_masking')
gen.fit(train_smiles, y_train.reshape(-1, 1))

# Step 2: Save features for later reuse
print("\n" + "=" * 70)
print("Step 2: Saving Features")
print("=" * 70)
gen.save_features('dummymask_features.pkl')

# Step 3: Transform training data (Dummy-Masking requires train indices)
print("\n" + "=" * 70)
print("Step 3: Transforming Training Data")
print("=" * 70)
train_indices = list(range(len(train_smiles)))
X_train = gen.transform(train_smiles, train_indices_per_task=[train_indices])
print(f"Training features shape: {X_train.shape}")

# Step 4: Train XGBoost model
print("\n" + "=" * 70)
print("Step 4: Training XGBoost")
print("=" * 70)
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)
model.fit(X_train, y_train)
print("✅ XGBoost model trained successfully")

# Step 5: Transform test data and evaluate
print("\n" + "=" * 70)
print("Step 5: Evaluating on Test Set")
print("=" * 70)
X_test = gen.transform(test_smiles, train_indices_per_task=[train_indices])
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"Test set performance:")
print(f"  Accuracy:  {accuracy:.3f}")
print(f"  ROC-AUC:   {roc_auc:.3f}")
print(f"  Predictions: {y_pred}")
print(f"  True labels: {y_test}")

# Step 6: Load features and apply to completely new molecules
print("\n" + "=" * 70)
print("Step 6: Applying to New Molecules (Load from disk)")
print("=" * 70)
gen_loaded = MultiTaskPrevalenceGenerator.load_features('dummymask_features.pkl')

new_smiles = ["CCCCCCCC", "c1ccc(O)cc1"]  # Octane, Phenol
new_smiles_names = ["Octane", "Phenol"]

X_new = gen_loaded.transform(new_smiles, train_indices_per_task=[train_indices])
y_new_pred = model.predict(X_new)
y_new_proba = model.predict_proba(X_new)[:, 1]

print(f"\nNew molecule predictions:")
for i, (name, smi, pred, prob) in enumerate(zip(new_smiles_names, new_smiles, y_new_pred, y_new_proba)):
    print(f"  {name:12s} ({smi:15s}): class={pred}, probability={prob:.3f}")

print("\n" + "=" * 70)
print("✅ Complete ML workflow finished successfully!")
print("=" * 70)
print("\nKey takeaways:")
print("  • MolFTP features work seamlessly with XGBoost")
print("  • Dummy-Masking requires train_indices for proper masking")
print("  • save_features() / load_features() enable model reuse")
print("  • XGBoost with n_jobs=-1 uses all CPU cores for speed")
print("  • Can handle both small and large-scale datasets")

