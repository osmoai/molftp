#!/usr/bin/env python3
"""
Test MolFTP performance on biodegradation dataset with speed and metrics.
Tests both Dummy-Masking and Key-LOO methods.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add parent directory to path to find biodegradation data
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "osmomain" / "src" / "sandbox" / "guillaume" / "biodegradation"))

try:
    from biodeg2025.load_data import load_biodegradation_data
except ImportError:
    # Fallback: try to load data directly
    def load_biodegradation_data():
        """Load biodegradation dataset."""
        data_path = Path.home() / "Downloads" / "biodegradation_data.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"Biodegradation data not found at {data_path}")
        import pandas as pd
        df = pd.read_csv(data_path)
        smiles = df['SMILES'].tolist()
        labels = df['Label'].values
        return smiles, labels

import molftp
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix
)

def test_molftp_performance(method='dummy_masking', k_threshold=2):
    """Test MolFTP performance on biodegradation dataset."""
    print("="*80)
    print(f"MOLFTP PERFORMANCE TEST - {method.upper()}")
    print("="*80)
    
    # Load data
    print("\n[Loading biodegradation dataset...]")
    smiles, labels = load_biodegradation_data()
    print(f"  ✓ Loaded {len(smiles)} molecules")
    print(f"  ✓ Positive: {np.sum(labels == 1)} ({np.sum(labels == 1)/len(labels)*100:.1f}%)")
    print(f"  ✓ Negative: {np.sum(labels == 0)} ({np.sum(labels == 0)/len(labels)*100:.1f}%)")
    
    # Split data (80/20)
    train_smiles, valid_smiles, train_labels, valid_labels = train_test_split(
        smiles, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"\n[Data split]")
    print(f"  Train: {len(train_smiles)} molecules")
    print(f"  Valid: {len(valid_smiles)} molecules")
    
    # Initialize MolFTP
    print(f"\n[Initializing MolFTP ({method})...]")
    if method == 'dummy_masking':
        gen = molftp.MultiTaskPrevalenceGenerator(
            radius=6,
            method='dummy_masking',
            num_threads=-1
        )
        # Fit on all data
        all_smiles = train_smiles + valid_smiles
        all_labels = np.concatenate([train_labels, valid_labels])
        gen.fit(all_smiles, all_labels.reshape(-1, 1), task_names=['biodegradation'])
        
        # Transform with train indices
        train_indices = list(range(len(train_smiles)))
        train_features = gen.transform(all_smiles, train_indices_per_task=[train_indices])
        train_features_final = train_features[:len(train_smiles)]
        valid_features_final = train_features[len(train_smiles):]
    else:  # key_loo
        gen = molftp.MultiTaskPrevalenceGenerator(
            radius=6,
            method='key_loo',
            key_loo_k=k_threshold,
            rescale_key_loo=True,
            num_threads=-1
        )
        gen.fit(train_smiles, train_labels.reshape(-1, 1), task_names=['biodegradation'])
        train_features_final = gen.transform(train_smiles)
        valid_features_final = gen.transform(valid_smiles)
    
    print(f"  ✓ Features shape: {train_features_final.shape[1]} features per molecule")
    
    # Train classifier
    print(f"\n[Training Random Forest classifier...]")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(train_features_final, train_labels)
    
    # Predictions
    print(f"\n[Making predictions...]")
    train_proba = clf.predict_proba(train_features_final)[:, 1]
    valid_proba = clf.predict_proba(valid_features_final)[:, 1]
    train_pred = clf.predict(train_features_final)
    valid_pred = clf.predict(valid_features_final)
    
    # Metrics
    print(f"\n[Computing metrics...]")
    
    # Training metrics
    train_roc = roc_auc_score(train_labels, train_proba)
    train_pr = average_precision_score(train_labels, train_proba)
    train_bacc = balanced_accuracy_score(train_labels, train_pred)
    train_prec = precision_score(train_labels, train_pred)
    train_rec = recall_score(train_labels, train_pred)
    train_f1 = f1_score(train_labels, train_pred)
    train_cm = confusion_matrix(train_labels, train_pred)
    
    # Validation metrics
    valid_roc = roc_auc_score(valid_labels, valid_proba)
    valid_pr = average_precision_score(valid_labels, valid_proba)
    valid_bacc = balanced_accuracy_score(valid_labels, valid_pred)
    valid_prec = precision_score(valid_labels, valid_pred)
    valid_rec = recall_score(valid_labels, valid_pred)
    valid_f1 = f1_score(valid_labels, valid_pred)
    valid_cm = confusion_matrix(valid_labels, valid_pred)
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    print("\n📊 Training Set Metrics:")
    print(f"  PR-AUC:        {train_pr:.4f}")
    print(f"  ROC-AUC:       {train_roc:.4f}")
    print(f"  Balanced Acc:  {train_bacc:.4f}")
    print(f"  Precision:     {train_prec:.4f}")
    print(f"  Recall:        {train_rec:.4f}")
    print(f"  F1-Score:      {train_f1:.4f}")
    print(f"  Confusion Matrix: TN={train_cm[0,0]}, FP={train_cm[0,1]}, FN={train_cm[1,0]}, TP={train_cm[1,1]}")
    
    print("\n📊 Validation Set Metrics:")
    print(f"  PR-AUC:        {valid_pr:.4f}")
    print(f"  ROC-AUC:       {valid_roc:.4f}")
    print(f"  Balanced Acc:  {valid_bacc:.4f}")
    print(f"  Precision:     {valid_prec:.4f}")
    print(f"  Recall:        {valid_rec:.4f}")
    print(f"  F1-Score:      {valid_f1:.4f}")
    print(f"  Confusion Matrix: TN={valid_cm[0,0]}, FP={valid_cm[0,1]}, FN={valid_cm[1,0]}, TP={valid_cm[1,1]}")
    
    return {
        'method': method,
        'train_metrics': {
            'pr_auc': train_pr,
            'roc_auc': train_roc,
            'balanced_acc': train_bacc,
            'precision': train_prec,
            'recall': train_rec,
            'f1': train_f1,
            'confusion_matrix': train_cm.tolist()
        },
        'valid_metrics': {
            'pr_auc': valid_pr,
            'roc_auc': valid_roc,
            'balanced_acc': valid_bacc,
            'precision': valid_prec,
            'recall': valid_rec,
            'f1': valid_f1,
            'confusion_matrix': valid_cm.tolist()
        }
    }

def test_speed(method='dummy_masking', k_threshold=2):
    """Test MolFTP speed on biodegradation dataset."""
    print("="*80)
    print(f"MOLFTP SPEED TEST - {method.upper()}")
    print("="*80)
    
    # Load data
    print("\n[Loading biodegradation dataset...]")
    smiles, labels = load_biodegradation_data()
    print(f"  ✓ Loaded {len(smiles)} molecules")
    
    # Split data
    train_smiles, valid_smiles, train_labels, valid_labels = train_test_split(
        smiles, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Initialize MolFTP
    if method == 'dummy_masking':
        gen = molftp.MultiTaskPrevalenceGenerator(
            radius=6,
            method='dummy_masking',
            num_threads=-1
        )
        all_smiles = train_smiles + valid_smiles
        all_labels = np.concatenate([train_labels, valid_labels])
        
        # Time fit
        print("\n[Timing fit()...]")
        start = time.time()
        gen.fit(all_smiles, all_labels.reshape(-1, 1), task_names=['biodegradation'])
        fit_time = time.time() - start
        
        # Time transform
        print("[Timing transform()...]")
        train_indices = list(range(len(train_smiles)))
        start = time.time()
        features = gen.transform(all_smiles, train_indices_per_task=[train_indices])
        transform_time = time.time() - start
        
        train_features = features[:len(train_smiles)]
        valid_features = features[len(train_smiles):]
    else:  # key_loo
        gen = molftp.MultiTaskPrevalenceGenerator(
            radius=6,
            method='key_loo',
            key_loo_k=k_threshold,
            rescale_key_loo=True,
            num_threads=-1
        )
        
        # Time fit
        print("\n[Timing fit()...]")
        start = time.time()
        gen.fit(train_smiles, train_labels.reshape(-1, 1), task_names=['biodegradation'])
        fit_time = time.time() - start
        
        # Time transform
        print("[Timing transform()...]")
        start = time.time()
        train_features = gen.transform(train_smiles)
        train_transform_time = time.time() - start
        
        start = time.time()
        valid_features = gen.transform(valid_smiles)
        valid_transform_time = time.time() - start
        transform_time = train_transform_time + valid_transform_time
    
    print("\n" + "="*80)
    print("TIMING RESULTS")
    print("="*80)
    print(f"\n⏱️  Fit time:        {fit_time:.4f}s ({len(smiles)} molecules)")
    print(f"⏱️  Transform time:  {transform_time:.4f}s")
    print(f"⏱️  Total time:      {fit_time + transform_time:.4f}s")
    print(f"⚡ Throughput:      {len(smiles) / (fit_time + transform_time):.1f} molecules/s")
    
    return {
        'method': method,
        'n_molecules': len(smiles),
        'fit_time': fit_time,
        'transform_time': transform_time,
        'total_time': fit_time + transform_time,
        'throughput': len(smiles) / (fit_time + transform_time)
    }

def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("MOLFTP BIODEGRADATION DATASET TEST")
    print("="*80)
    print(f"MolFTP Version: {molftp.__version__}")
    print("="*80)
    
    results = {}
    
    # Test Dummy-Masking
    print("\n\n")
    speed_dummy = test_speed('dummy_masking')
    print("\n\n")
    metrics_dummy = test_molftp_performance('dummy_masking')
    results['dummy_masking'] = {**speed_dummy, **metrics_dummy}
    
    # Test Key-LOO
    print("\n\n")
    speed_keyloo = test_speed('key_loo', k_threshold=2)
    print("\n\n")
    metrics_keyloo = test_molftp_performance('key_loo', k_threshold=2)
    results['key_loo'] = {**speed_keyloo, **metrics_keyloo}
    
    # Summary
    print("\n\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("\n📊 Dummy-Masking:")
    print(f"  Fit time:        {speed_dummy['fit_time']:.4f}s")
    print(f"  Valid PR-AUC:    {metrics_dummy['valid_metrics']['pr_auc']:.4f}")
    print(f"  Valid ROC-AUC:   {metrics_dummy['valid_metrics']['roc_auc']:.4f}")
    print(f"  Valid BAcc:      {metrics_dummy['valid_metrics']['balanced_acc']:.4f}")
    
    print("\n📊 Key-LOO (k=2):")
    print(f"  Fit time:        {speed_keyloo['fit_time']:.4f}s")
    print(f"  Valid PR-AUC:    {metrics_keyloo['valid_metrics']['pr_auc']:.4f}")
    print(f"  Valid ROC-AUC:   {metrics_keyloo['valid_metrics']['roc_auc']:.4f}")
    print(f"  Valid BAcc:      {metrics_keyloo['valid_metrics']['balanced_acc']:.4f}")
    
    print("\n✅ All tests completed!")

if __name__ == '__main__':
    main()

