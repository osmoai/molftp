"""
Complete ML workflow: MolFTP features + Logistic Regression

This example demonstrates a full machine learning pipeline:
1. Build MolFTP features using Key-LOO
2. Save features for reuse
3. Train Logistic Regression classifier
4. Evaluate on test set
5. Apply to completely new molecules
"""

from molftp import MultiTaskPrevalenceGenerator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

print("=" * 70)
print("MolFTP + Logistic Regression: Complete ML Workflow")
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

# Step 1: Build MolFTP features using Key-LOO method
print("\n" + "=" * 70)
print("Step 1: Building MolFTP Features (Key-LOO)")
print("=" * 70)
gen = MultiTaskPrevalenceGenerator(radius=6, method='key_loo')
gen.fit(train_smiles, y_train.reshape(-1, 1))

# Step 2: Save features for later reuse
print("\n" + "=" * 70)
print("Step 2: Saving Features")
print("=" * 70)
gen.save_features('keyloo_features.pkl')

# Step 3: Transform training data
print("\n" + "=" * 70)
print("Step 3: Transforming Training Data")
print("=" * 70)
X_train = gen.transform(train_smiles)
print(f"Training features shape: {X_train.shape}")

# Step 4: Train Logistic Regression model
print("\n" + "=" * 70)
print("Step 4: Training Logistic Regression")
print("=" * 70)
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train, y_train)
print("✅ Model trained successfully")

# Step 5: Transform test data and evaluate
print("\n" + "=" * 70)
print("Step 5: Evaluating on Test Set")
print("=" * 70)
X_test = gen.transform(test_smiles)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

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
gen_loaded = MultiTaskPrevalenceGenerator.load_features('keyloo_features.pkl')

new_smiles = ["CCCCCCCC", "c1ccc(O)cc1"]  # Octane, Phenol
new_smiles_names = ["Octane", "Phenol"]

X_new = gen_loaded.transform(new_smiles)
y_new_pred = clf.predict(X_new)
y_new_proba = clf.predict_proba(X_new)[:, 1]

print(f"\nNew molecule predictions:")
for i, (name, smi, pred, prob) in enumerate(zip(new_smiles_names, new_smiles, y_new_pred, y_new_proba)):
    print(f"  {name:12s} ({smi:15s}): class={pred}, probability={prob:.3f}")

print("\n" + "=" * 70)
print("✅ Complete ML workflow finished successfully!")
print("=" * 70)
print("\nKey takeaways:")
print("  • MolFTP features capture fragment-target prevalence")
print("  • save_features() / load_features() enable model reuse")
print("  • Key-LOO method provides good generalization")
print("  • Can be combined with any sklearn classifier")

