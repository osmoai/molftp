#!/usr/bin/env python3
"""
Benchmark both Key-LOO and Dummy-Masking methods on biodegradation dataset.
Compare performance metrics and check for data leakage.
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List
import sys
import json
import argparse
sys.path.insert(0, '.')

from molftp import MultiTaskPrevalenceGenerator
import _molftp as ftp
PROXAMP_OFF = ftp.PROXAMP_OFF
PROXAMP_TRAIN_SHARE = ftp.PROXAMP_TRAIN_SHARE
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def murcko_scaffold(smi):
    """Extract Murcko scaffold from SMILES."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return smi
        scaffold = rdMolDescriptors.MurckoScaffoldSmiles(mol=mol)
        return scaffold if scaffold else smi
    except:
        return smi

def create_scaffold_split(smiles: List[str], labels: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Create scaffold-based train/validation split."""
    print("\nCreating scaffold-based train/validation split...")
    
    scaffolds = np.array([murcko_scaffold(s) for s in smiles])
    
    def robust_hash(s):
        """Robust hash that handles empty strings and None."""
        if not s or s == "":
            return hash(str(s))
        return hash(s)
    
    # Use hash-based split (80/20)
    mask = np.array([robust_hash(scaf) % 5 for scaf in scaffolds])
    train_idx = np.where(mask != 0)[0]
    test_idx = np.where(mask == 0)[0]
    
    # Fallback to random split if scaffold split fails
    if len(train_idx) == 0 or len(test_idx) == 0:
        print("  ⚠ Scaffold split resulted in empty set, using random split instead")
        train_idx, test_idx = train_test_split(
            np.arange(len(smiles)), test_size=test_size, random_state=random_state, stratify=labels
        )
    
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    
    print(f"  ✓ Training set: {len(train_idx)} molecules ({100*len(train_idx)/len(smiles):.1f}%)")
    print(f"  ✓ Test set: {len(test_idx)} molecules ({100*len(test_idx)/len(smiles):.1f}%)")
    print(f"  ✓ Training positives: {labels[train_idx].sum()} ({100*labels[train_idx].mean():.2f}%)")
    print(f"  ✓ Test positives: {labels[test_idx].sum()} ({100*labels[test_idx].mean():.2f}%)")
    
    # Verify scaffold separation
    train_scaffolds = set(scaffolds[train_idx])
    test_scaffolds = set(scaffolds[test_idx])
    scaffold_overlap = train_scaffolds.intersection(test_scaffolds)
    if scaffold_overlap:
        print(f"  ⚠ WARNING: {len(scaffold_overlap)} scaffolds appear in both train and test!")
    else:
        print(f"  ✅ No scaffold overlap between train and test sets")
    
    return train_idx, test_idx

def load_biodegradation_data() -> Tuple[list, np.ndarray]:
    """Load biodegradation dataset."""
    possible_paths = [
        Path(__file__).parent.parent.parent / "biodeg2025" / "data" / "biodegradation_combined.csv",
        Path(__file__).parent / "data" / "biodegradation_combined.csv",
        Path("/Users/guillaume-osmo/Github/osmomain/src/sandbox/guillaume/biodegradation/biodeg2025/data/biodegradation_combined.csv"),
    ]
    
    csv_file = None
    for p in possible_paths:
        if p.exists():
            csv_file = p
            break
    
    if csv_file is None:
        raise FileNotFoundError(f"Data file not found. Tried: {possible_paths}")
    
    print(f"Loading data from: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Extract SMILES
    if 'canonical_smiles' in df.columns:
        smiles_col = 'canonical_smiles'
    elif 'smiles' in df.columns:
        smiles_col = 'smiles'
    elif 'SMILES' in df.columns:
        smiles_col = 'SMILES'
    else:
        raise ValueError(f"No SMILES column found. Available: {df.columns.tolist()}")
    
    smiles_list = df[smiles_col].tolist()
    
    # Extract labels
    if 'label' in df.columns:
        labels_raw = df['label'].values
    elif 'Label' in df.columns:
        labels_raw = df['Label'].values
    elif 'Biodegradable' in df.columns:
        labels_raw = df['Biodegradable'].values
    else:
        raise ValueError(f"No label column found. Available: {df.columns.tolist()}")
    
    # Filter valid SMILES and matching labels, removing duplicates
    # CRITICAL FIX: Ensure 1:1 correspondence between SMILES and labels
    valid_smiles = []
    valid_labels = []
    seen_smiles = set()
    
    for i, smi in enumerate(smiles_list):
        # Check if we have a valid label for this index
        if i >= len(labels_raw):
            continue
        lab = labels_raw[i]
        if pd.isna(lab):
            continue
        try:
            lab_int = int(float(lab))
        except (ValueError, TypeError):
            continue
        
        # Validate SMILES
        mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            smi_str = str(smi)
            # Skip duplicates - keep first occurrence
            if smi_str not in seen_smiles:
                valid_smiles.append(smi_str)
                valid_labels.append(lab_int)
                seen_smiles.add(smi_str)
    
    labels = np.array(valid_labels, dtype=int)
    
    print(f"Loaded {len(valid_smiles)} valid molecules (duplicates removed)")
    if len(labels) > 0:
        print(f"  Positive: {np.sum(labels == 1)} ({np.sum(labels == 1)/len(labels)*100:.1f}%)")
        print(f"  Negative: {np.sum(labels == 0)} ({np.sum(labels == 0)/len(labels)*100:.1f}%)")
    
    return valid_smiles, labels

def test_method(method_name: str, smiles_train, labels_train, smiles_test, labels_test, 
                smiles_all, labels_all, ncm_config=None):
    """Test a single method (key_loo, dummy_masking, or ncm variants)."""
    print(f"\n{'='*80}")
    print(f"TESTING METHOD: {method_name.upper()}")
    print(f"{'='*80}")
    
    # Create generator - use key_loo as base for NCM methods
    base_method = 'key_loo' if method_name.startswith('ncm') else method_name
    gen = MultiTaskPrevalenceGenerator(
        radius=6,
        nBits=2048,
        sim_thresh=0.7,
        stat_1d='chi2',
        stat_2d='mcnemar_midp',
        stat_3d='exact_binom',
        alpha=0.5,
        num_threads=-1,
        method=base_method
    )
    
    # Configure NCM if needed
    if method_name.startswith('ncm') and ncm_config:
        gen.set_proximity_mode(ncm_config.get('mode', 'hier_backoff'))
        gen.set_notclose_masking(
            gap=ncm_config.get('dmax', 1),
            min_parent_depth=0,
            require_all_components=True,
            debug=False
        )
        # Configure amplitude if specified
        amp_source = ncm_config.get('amp_source', PROXAMP_OFF)
        if amp_source != PROXAMP_OFF:
            gen.set_proximity_amplitude(
                source=amp_source,
                prior_alpha=ncm_config.get('amp_alpha', 1.0),
                gamma=ncm_config.get('amp_gamma', 1.0),
                cap_min=ncm_config.get('amp_cap_min', 0.25),
                cap_max=ncm_config.get('amp_cap_max', 1.0),
                apply_to_train_rows=False
            )
            gen.set_proximity_amp_components_policy(
                ncm_config.get('first_component_only', False)
            )
            gen.set_proximity_amp_distance_beta(
                ncm_config.get('dist_beta', 0.0)
            )
    
    # Fit on ALL data (train+test) - this is correct for both methods
    print(f"\nFitting on {len(smiles_all)} molecules (train+test)...")
    all_labels_2d = labels_all.reshape(-1, 1)
    
    t0 = time.time()
    gen.fit(smiles_all, all_labels_2d, task_names=['Biodegradable'])
    fit_time = time.time() - t0
    
    print(f"✅ Fit completed in {fit_time:.3f}s")
    print(f"   Features: {gen.get_n_features()}")
    
    # Transform based on method
    print(f"\nTransforming...")
    t0 = time.time()
    
    if method_name == 'dummy_masking':
        # Dummy-Masking: train_indices_per_task must be indices into the smiles parameter
        # Since we fit on smiles_all, we need to transform smiles_all and split the results
        train_indices_full = [list(range(len(smiles_train)))]  # Indices into smiles_all
        
        # Transform full dataset (required for dummy masking to work correctly)
        X_all = gen.transform(smiles_all, train_indices_per_task=train_indices_full)
        
        # Split results
        X_train = X_all[:len(smiles_train)]
        X_test = X_all[len(smiles_train):]
    else:  # key_loo or ncm variants
        # Key-LOO/NCM: Use train_row_mask for training data rescaling
        # For NCM, we need to transform all data together with train_row_mask
        train_row_mask = [True] * len(smiles_train) + [False] * len(smiles_test)
        X_all = gen.transform(smiles_all, train_row_mask=train_row_mask)
        
        # Split results
        X_train = X_all[:len(smiles_train)]
        X_test = X_all[len(smiles_train):]
    
    transform_time = time.time() - t0
    
    # CRITICAL VALIDATION: Verify shapes match expectations
    assert X_train.shape[0] == len(smiles_train), \
        f"Train shape mismatch: X_train.shape[0]={X_train.shape[0]}, len(smiles_train)={len(smiles_train)}"
    assert X_test.shape[0] == len(smiles_test), \
        f"Test shape mismatch: X_test.shape[0]={X_test.shape[0]}, len(smiles_test)={len(smiles_test)}"
    assert X_train.shape[1] == X_test.shape[1], \
        f"Feature dimension mismatch: X_train.shape[1]={X_train.shape[1]}, X_test.shape[1]={X_test.shape[1]}"
    
    print(f"✅ Transform completed in {transform_time:.3f}s")
    print(f"   Train shape: {X_train.shape}")
    print(f"   Test shape: {X_test.shape}")
    
    # Check for data leakage: Are train and test features identical?
    if np.allclose(X_train[:min(10, len(X_train))], X_test[:min(10, len(X_test))], atol=1e-6):
        print("⚠️  WARNING: First 10 samples of train and test features are identical!")
    
    # Check feature statistics
    print(f"\nFeature statistics:")
    print(f"   Train - Mean: {X_train.mean():.6f}, Std: {X_train.std():.6f}, Min: {X_train.min():.6f}, Max: {X_train.max():.6f}")
    print(f"   Test  - Mean: {X_test.mean():.6f}, Std: {X_test.std():.6f}, Min: {X_test.min():.6f}, Max: {X_test.max():.6f}")
    print(f"   Train-Test diff - Mean: {np.abs(X_train.mean() - X_test.mean()):.6f}")
    
    # Train XGBoost
    try:
        import xgboost as xgb
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        # Handle class imbalance
        pos_weight = np.sum(labels_train == 0) / max(np.sum(labels_train == 1), 1)
        model.set_params(scale_pos_weight=pos_weight)
        
        t0 = time.time()
        model.fit(X_train, labels_train)
        train_time = time.time() - t0
        
        # Predictions
        y_pred_proba_train = model.predict_proba(X_train)[:, 1]
        y_pred_proba_test = model.predict_proba(X_test)[:, 1]
        
        y_pred_train = (y_pred_proba_train > 0.5).astype(int)
        y_pred_test = (y_pred_proba_test > 0.5).astype(int)
        
        # Metrics
        auc_train = roc_auc_score(labels_train, y_pred_proba_train)
        auc_test = roc_auc_score(labels_test, y_pred_proba_test)
        
        bacc_train = balanced_accuracy_score(labels_train, y_pred_train)
        bacc_test = balanced_accuracy_score(labels_test, y_pred_test)
        
        f1_train = f1_score(labels_train, y_pred_train)
        f1_test = f1_score(labels_test, y_pred_test)
        
        # CRITICAL VALIDATION: Verify metrics are in valid ranges
        assert 0 <= auc_train <= 1, f"Invalid train AUC: {auc_train}"
        assert 0 <= auc_test <= 1, f"Invalid test AUC: {auc_test}"
        assert 0 <= bacc_train <= 1, f"Invalid train BAcc: {bacc_train}"
        assert 0 <= bacc_test <= 1, f"Invalid test BAcc: {bacc_test}"
        assert 0 <= f1_train <= 1, f"Invalid train F1: {f1_train}"
        assert 0 <= f1_test <= 1, f"Invalid test F1: {f1_test}"
        
        # Confusion matrices
        cm_train = confusion_matrix(labels_train, y_pred_train)
        cm_test = confusion_matrix(labels_test, y_pred_test)
        
        return {
            'method': method_name,
            'fit_time': fit_time,
            'transform_time': transform_time,
            'train_time': train_time,
            'auc_train': auc_train,
            'auc_test': auc_test,
            'bacc_train': bacc_train,
            'bacc_test': bacc_test,
            'f1_train': f1_train,
            'f1_test': f1_test,
            'cm_train': cm_train,
            'cm_test': cm_test,
            'X_train_shape': X_train.shape,
            'X_test_shape': X_test.shape,
            'train_feat_mean': X_train.mean(),
            'test_feat_mean': X_test.mean(),
        }
    except ImportError:
        print("⚠️  XGBoost not available")
        return None

def main():
    """Test both methods and compare."""
    parser = argparse.ArgumentParser(description='MolFTP Benchmark')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    parser.add_argument('--split', choices=['random', 'scaffold'], default='random', 
                       help='Split method: random or scaffold (default: random). Run separately and compare manually.')
    args = parser.parse_args()
    
    print("="*80)
    print("MOLFTP BENCHMARK: KEY-LOO vs DUMMY-MASKING")
    print("="*80)
    
    # Load data
    smiles, labels = load_biodegradation_data()
    
    # Run only ONE split at a time to avoid state leakage
    # To compare: run separately with --split random and --split scaffold, then compare outputs
    split_name = args.split
    
    all_results = {}
    
    # Single split execution
    if True:  # Single split
        print("\n" + "="*80)
        print(f"SPLIT METHOD: {split_name.upper()}")
        print("="*80)
        
        if split_name == 'random':
            # Random split (80/20) - IMPORTANT: Use stratify to maintain class balance
            smiles_train, smiles_test, labels_train, labels_test = train_test_split(
                smiles, labels, test_size=0.2, random_state=42, stratify=labels
            )
        else:  # scaffold
            # Scaffold split
            train_idx, test_idx = create_scaffold_split(smiles, labels, test_size=0.2, random_state=42)
            smiles_train = [smiles[i] for i in train_idx]
            smiles_test = [smiles[i] for i in test_idx]
            labels_train = labels[train_idx]
            labels_test = labels[test_idx]
        
        print(f"\nDataset split ({split_name}):")
        print(f"  Train: {len(smiles_train)} molecules")
        print(f"    Positive: {np.sum(labels_train == 1)} ({np.sum(labels_train == 1)/len(labels_train)*100:.1f}%)")
        print(f"    Negative: {np.sum(labels_train == 0)} ({np.sum(labels_train == 0)/len(labels_train)*100:.1f}%)")
        print(f"  Test: {len(smiles_test)} molecules")
        print(f"    Positive: {np.sum(labels_test == 1)} ({np.sum(labels_test == 1)/len(labels_test)*100:.1f}%)")
        print(f"    Negative: {np.sum(labels_test == 0)} ({np.sum(labels_test == 0)/len(labels_test)*100:.1f}%)")
        
        # Verify no overlap
        train_set = set(smiles_train)
        test_set = set(smiles_test)
        overlap = train_set.intersection(test_set)
        if overlap:
            print(f"\n⚠️  WARNING: {len(overlap)} molecules appear in both train and test!")
        else:
            print(f"\n✅ No overlap between train and test sets")
        
        # All data for fitting (both methods fit on all data)
        smiles_all = smiles_train + smiles_test
        labels_all = np.concatenate([labels_train, labels_test])
        
        # Test all methods
        results_keyloo = test_method('key_loo', smiles_train, labels_train, smiles_test, labels_test,
                                     smiles_all, labels_all)
        
        results_dummy = test_method('dummy_masking', smiles_train, labels_train, smiles_test, labels_test,
                                    smiles_all, labels_all)
        
        # Test NCM methods
        results_ncm_backoff = None
        results_ncm_backoff_amp = None
        try:
            results_ncm_backoff = test_method('ncm_backoff', smiles_train, labels_train, smiles_test, labels_test,
                                              smiles_all, labels_all, 
                                              ncm_config={'mode': 'hier_backoff', 'dmax': 1, 'amp_source': PROXAMP_OFF})
        except Exception as e:
            print(f"⚠️  NCM Backoff failed: {e}")
            import traceback
            traceback.print_exc()
        
        try:
            results_ncm_backoff_amp = test_method('ncm_backoff_amp', smiles_train, labels_train, smiles_test, labels_test,
                                                  smiles_all, labels_all,
                                                  ncm_config={'mode': 'hier_backoff', 'dmax': 1, 
                                                             'amp_source': PROXAMP_TRAIN_SHARE, 'amp_alpha': 1.0,
                                                             'amp_gamma': 1.0, 'amp_cap_min': 0.25, 'amp_cap_max': 1.0,
                                                             'first_component_only': False, 'dist_beta': 0.0})
        except Exception as e:
            print(f"⚠️  NCM Backoff+Amp failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Store results for this split
        all_results[split_name] = {
            'key_loo': results_keyloo,
            'dummy_masking': results_dummy,
            'ncm_backoff': results_ncm_backoff,
            'ncm_backoff_amp': results_ncm_backoff_amp,
            'dataset': {
                'n_train': len(smiles_train),
                'n_test': len(smiles_test),
                'n_total': len(smiles),
                'train_pos_pct': float(np.sum(labels_train == 1) / len(labels_train) * 100),
                'test_pos_pct': float(np.sum(labels_test == 1) / len(labels_test) * 100),
            }
        }
        
        # Compare results for this split
        print("\n" + "="*80)
        print(f"COMPARISON: ALL METHODS ({split_name.upper()} SPLIT)")
        print("="*80)
    
        if results_keyloo and results_dummy:
            print(f"\n{'Metric':<25} {'Key-LOO':<15} {'Dummy':<15} {'NCM-Backoff':<15} {'NCM-Backoff+Amp':<15}")
            print("-"*85)
            ncm_auc = f"{results_ncm_backoff['auc_test']:.4f}" if results_ncm_backoff else "N/A"
            ncm_amp_auc = f"{results_ncm_backoff_amp['auc_test']:.4f}" if results_ncm_backoff_amp else "N/A"
            print(f"{'Test AUC':<25} {results_keyloo['auc_test']:<15.4f} {results_dummy['auc_test']:<15.4f} {ncm_auc:<15} {ncm_amp_auc:<15}")
            
            ncm_bacc = f"{results_ncm_backoff['bacc_test']:.4f}" if results_ncm_backoff else "N/A"
            ncm_amp_bacc = f"{results_ncm_backoff_amp['bacc_test']:.4f}" if results_ncm_backoff_amp else "N/A"
            print(f"{'Test BAcc':<25} {results_keyloo['bacc_test']:<15.4f} {results_dummy['bacc_test']:<15.4f} {ncm_bacc:<15} {ncm_amp_bacc:<15}")
            
            ncm_f1 = f"{results_ncm_backoff['f1_test']:.4f}" if results_ncm_backoff else "N/A"
            ncm_amp_f1 = f"{results_ncm_backoff_amp['f1_test']:.4f}" if results_ncm_backoff_amp else "N/A"
            print(f"{'Test F1':<25} {results_keyloo['f1_test']:<15.4f} {results_dummy['f1_test']:<15.4f} {ncm_f1:<15} {ncm_amp_f1:<15}")
            
            ncm_fit = f"{results_ncm_backoff['fit_time']:.3f}" if results_ncm_backoff else "N/A"
            ncm_amp_fit = f"{results_ncm_backoff_amp['fit_time']:.3f}" if results_ncm_backoff_amp else "N/A"
            print(f"{'Fit Time (s)':<25} {results_keyloo['fit_time']:<15.3f} {results_dummy['fit_time']:<15.3f} {ncm_fit:<15} {ncm_amp_fit:<15}")
            
            ncm_trans = f"{results_ncm_backoff['transform_time']:.3f}" if results_ncm_backoff else "N/A"
            ncm_amp_trans = f"{results_ncm_backoff_amp['transform_time']:.3f}" if results_ncm_backoff_amp else "N/A"
            print(f"{'Transform Time (s)':<25} {results_keyloo['transform_time']:<15.3f} {results_dummy['transform_time']:<15.3f} {ncm_trans:<15} {ncm_amp_trans:<15}")
            
            print(f"\n{'Confusion Matrix (Test)':<25} {'Key-LOO':<30} {'Dummy-Masking':<30}")
            print("-"*80)
            print(f"{'TN, FP':<25} {results_keyloo['cm_test'][0,0]}, {results_keyloo['cm_test'][0,1]:<25} {results_dummy['cm_test'][0,0]}, {results_dummy['cm_test'][0,1]}")
            print(f"{'FN, TP':<25} {results_keyloo['cm_test'][1,0]}, {results_keyloo['cm_test'][1,1]:<25} {results_dummy['cm_test'][1,0]}, {results_dummy['cm_test'][1,1]}")
            
            # Check for suspicious perfect scores
            print(f"\n{'='*80}")
            print("DATA LEAKAGE CHECK")
            print(f"{'='*80}")
            method_results = [results_keyloo, results_dummy]
            if results_ncm_backoff:
                method_results.append(results_ncm_backoff)
            if results_ncm_backoff_amp:
                method_results.append(results_ncm_backoff_amp)
            
            high_auc = [r for r in method_results if r['auc_test'] > 0.99]
            if high_auc:
                print("⚠️  WARNING: Very high test AUC (>0.99) - possible data leakage!")
                for r in high_auc:
                    print(f"   {r['method']}: AUC={r['auc_test']:.4f}")
            else:
                print("✅ Test AUC seems reasonable for all methods")
    
    # NOTE: To compare Random vs Scaffold splits, run separately:
    #   python test_both_methods_benchmark.py --split random > random_results.txt
    #   python test_both_methods_benchmark.py --split scaffold > scaffold_results.txt
    # Then compare the outputs manually to avoid any state leakage between runs.
    
    # Output JSON if requested
    if args.json:
        # Output format: always single split (random or scaffold)
        if split_name == 'random':
            # Old format (backward compatible with benchmark history)
            split_results = all_results[split_name]
            json_result = {}
            if split_results['key_loo']:
                json_result['key_loo'] = {
                    'auc_test': float(split_results['key_loo']['auc_test']),
                    'auc_train': float(split_results['key_loo']['auc_train']),
                    'bacc_test': float(split_results['key_loo']['bacc_test']),
                    'bacc_train': float(split_results['key_loo']['bacc_train']),
                    'f1_test': float(split_results['key_loo']['f1_test']),
                    'f1_train': float(split_results['key_loo']['f1_train']),
                    'fit_time': float(split_results['key_loo']['fit_time']),
                    'transform_time': float(split_results['key_loo']['transform_time']),
                    'train_time': float(split_results['key_loo']['train_time']),
                    'confusion_matrix_test': split_results['key_loo']['cm_test'].tolist(),
                }
            if split_results['dummy_masking']:
                json_result['dummy_masking'] = {
                    'auc_test': float(split_results['dummy_masking']['auc_test']),
                    'auc_train': float(split_results['dummy_masking']['auc_train']),
                    'bacc_test': float(split_results['dummy_masking']['bacc_test']),
                    'bacc_train': float(split_results['dummy_masking']['bacc_train']),
                    'f1_test': float(split_results['dummy_masking']['f1_test']),
                    'f1_train': float(split_results['dummy_masking']['f1_train']),
                    'fit_time': float(split_results['dummy_masking']['fit_time']),
                    'transform_time': float(split_results['dummy_masking']['transform_time']),
                    'train_time': float(split_results['dummy_masking']['train_time']),
                    'confusion_matrix_test': split_results['dummy_masking']['cm_test'].tolist(),
                }
            if split_results['ncm_backoff']:
                json_result['ncm_backoff'] = {
                    'auc_test': float(split_results['ncm_backoff']['auc_test']),
                    'auc_train': float(split_results['ncm_backoff']['auc_train']),
                    'bacc_test': float(split_results['ncm_backoff']['bacc_test']),
                    'bacc_train': float(split_results['ncm_backoff']['bacc_train']),
                    'f1_test': float(split_results['ncm_backoff']['f1_test']),
                    'f1_train': float(split_results['ncm_backoff']['f1_train']),
                    'fit_time': float(split_results['ncm_backoff']['fit_time']),
                    'transform_time': float(split_results['ncm_backoff']['transform_time']),
                    'train_time': float(split_results['ncm_backoff']['train_time']),
                    'confusion_matrix_test': split_results['ncm_backoff']['cm_test'].tolist(),
                }
            if split_results['ncm_backoff_amp']:
                json_result['ncm_backoff_amp'] = {
                    'auc_test': float(split_results['ncm_backoff_amp']['auc_test']),
                    'auc_train': float(split_results['ncm_backoff_amp']['auc_train']),
                    'bacc_test': float(split_results['ncm_backoff_amp']['bacc_test']),
                    'bacc_train': float(split_results['ncm_backoff_amp']['bacc_train']),
                    'f1_test': float(split_results['ncm_backoff_amp']['f1_test']),
                    'f1_train': float(split_results['ncm_backoff_amp']['f1_train']),
                    'fit_time': float(split_results['ncm_backoff_amp']['fit_time']),
                    'transform_time': float(split_results['ncm_backoff_amp']['transform_time']),
                    'train_time': float(split_results['ncm_backoff_amp']['train_time']),
                    'confusion_matrix_test': split_results['ncm_backoff_amp']['cm_test'].tolist(),
                }
            json_result['dataset'] = split_results['dataset']
            
            print("\nBENCHMARK_JSON_START")
            print(json.dumps(json_result, indent=2))
            print("BENCHMARK_JSON_END")
        else:
            # Scaffold split format (same structure, different split name)
            split_results = all_results[split_name]
            json_result = {}
            if split_results['key_loo']:
                json_result['key_loo'] = {
                    'auc_test': float(split_results['key_loo']['auc_test']),
                    'auc_train': float(split_results['key_loo']['auc_train']),
                    'bacc_test': float(split_results['key_loo']['bacc_test']),
                    'bacc_train': float(split_results['key_loo']['bacc_train']),
                    'f1_test': float(split_results['key_loo']['f1_test']),
                    'f1_train': float(split_results['key_loo']['f1_train']),
                    'fit_time': float(split_results['key_loo']['fit_time']),
                    'transform_time': float(split_results['key_loo']['transform_time']),
                    'train_time': float(split_results['key_loo']['train_time']),
                    'confusion_matrix_test': split_results['key_loo']['cm_test'].tolist(),
                }
            if split_results['dummy_masking']:
                json_result['dummy_masking'] = {
                    'auc_test': float(split_results['dummy_masking']['auc_test']),
                    'auc_train': float(split_results['dummy_masking']['auc_train']),
                    'bacc_test': float(split_results['dummy_masking']['bacc_test']),
                    'bacc_train': float(split_results['dummy_masking']['bacc_train']),
                    'f1_test': float(split_results['dummy_masking']['f1_test']),
                    'f1_train': float(split_results['dummy_masking']['f1_train']),
                    'fit_time': float(split_results['dummy_masking']['fit_time']),
                    'transform_time': float(split_results['dummy_masking']['transform_time']),
                    'train_time': float(split_results['dummy_masking']['train_time']),
                    'confusion_matrix_test': split_results['dummy_masking']['cm_test'].tolist(),
                }
            if split_results['ncm_backoff']:
                json_result['ncm_backoff'] = {
                    'auc_test': float(split_results['ncm_backoff']['auc_test']),
                    'auc_train': float(split_results['ncm_backoff']['auc_train']),
                    'bacc_test': float(split_results['ncm_backoff']['bacc_test']),
                    'bacc_train': float(split_results['ncm_backoff']['bacc_train']),
                    'f1_test': float(split_results['ncm_backoff']['f1_test']),
                    'f1_train': float(split_results['ncm_backoff']['f1_train']),
                    'fit_time': float(split_results['ncm_backoff']['fit_time']),
                    'transform_time': float(split_results['ncm_backoff']['transform_time']),
                    'train_time': float(split_results['ncm_backoff']['train_time']),
                    'confusion_matrix_test': split_results['ncm_backoff']['cm_test'].tolist(),
                }
            if split_results['ncm_backoff_amp']:
                json_result['ncm_backoff_amp'] = {
                    'auc_test': float(split_results['ncm_backoff_amp']['auc_test']),
                    'auc_train': float(split_results['ncm_backoff_amp']['auc_train']),
                    'bacc_test': float(split_results['ncm_backoff_amp']['bacc_test']),
                    'bacc_train': float(split_results['ncm_backoff_amp']['bacc_train']),
                    'f1_test': float(split_results['ncm_backoff_amp']['f1_test']),
                    'f1_train': float(split_results['ncm_backoff_amp']['f1_train']),
                    'fit_time': float(split_results['ncm_backoff_amp']['fit_time']),
                    'transform_time': float(split_results['ncm_backoff_amp']['transform_time']),
                    'train_time': float(split_results['ncm_backoff_amp']['train_time']),
                    'confusion_matrix_test': split_results['ncm_backoff_amp']['cm_test'].tolist(),
                }
            json_result['dataset'] = split_results['dataset']
            
            print("\nBENCHMARK_JSON_START")
            print(json.dumps(json_result, indent=2))
            print("BENCHMARK_JSON_END")

if __name__ == "__main__":
    main()

