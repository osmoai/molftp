# Not-Close Masking (NCM) Method - Performance Report

## Overview

Not-Close Masking (NCM) is a novel method for handling unseen keys in molecular fragment-target prevalence (MolFTP) feature generation. NCM addresses data leakage issues present in existing methods (key_loo, dummy_masking) by using only training data for key extraction and applying hierarchical proximity-based backoff for unseen keys.

## Method Variants

### 1. NCM Backoff (`ncm_backoff`)
- **Mode**: Hierarchical backoff
- **Description**: Replaces unseen keys with the nearest training ancestor using hierarchical proximity
- **Parameters**: `dmax=1`, hierarchical distance-based backoff

### 2. NCM Backoff with Amplitude (`ncm_backoff_amp`)
- **Mode**: Hierarchical backoff with target-aware amplitude
- **Description**: Same as `ncm_backoff` but applies target-aware amplitude scaling based on training data statistics
- **Parameters**: 
  - `dmax=1`
  - `amp_source=train_share` (target-aware amplitude)
  - `amp_alpha=1.0`, `amp_gamma=1.0`
  - `amp_cap_min=0.25`, `amp_cap_max=1.0`
  - `first_component_only=False` (uses MIN over all components)
  - `dist_beta=0.0` (no distance decay)

## Key Advantages

1. **No Data Leakage**: Uses ONLY training data for key extraction (unlike key_loo which uses full dataset)
2. **Better Calibration**: Higher percentage of valid models (63-67% vs 23-60% for other methods)
3. **Production Ready**: More realistic performance estimates for deployment
4. **Robust Handling**: Hierarchical backoff handles unseen keys gracefully

## Performance Results

### Best Configurations by Split Method

#### Random Split

**Best by AUC:**
- **NCM (ncm_backoff_amp)**: R=4, sim=0.15, AUC=0.9360, BAcc=0.8611, BAcc(opt)=0.8762 ✅ VALID

**Best by BAcc:**
- **NCM (ncm_backoff_amp)**: R=7, sim=0.05, AUC=0.9336, BAcc=0.8670, BAcc(opt)=0.8693 ✅ VALID

**Best by BAcc(opt):**
- **NCM (ncm_backoff_amp)**: R=4, sim=0.15, AUC=0.9360, BAcc=0.8611, BAcc(opt)=0.8762 ✅ VALID

#### Scaffold Split

**Best by AUC:**
- **NCM (ncm_backoff)**: R=4, sim=0.15, AUC=0.9252, BAcc=0.8637, BAcc(opt)=0.8656 ✅ VALID

**Best by BAcc:**
- **NCM (ncm_backoff)**: R=4, sim=0.15, AUC=0.9252, BAcc=0.8637, BAcc(opt)=0.8656 ✅ VALID

**Best by BAcc(opt):**
- **NCM (ncm_backoff)**: R=4, sim=0.15, AUC=0.9252, BAcc=0.8637, BAcc(opt)=0.8656 ✅ VALID

#### CV5 Split (5-Fold Cross-Validation)

**Best by AUC:**
- **NCM (ncm_backoff)**: R=6, sim=0.05, AUC=0.9322±0.0121, BAcc=0.8531±0.0175, BAcc(opt)=0.8531±0.0175 ✅ VALID

**Best by BAcc:**
- **NCM (ncm_backoff_amp)**: R=4, sim=0.05, AUC=0.9319±0.0134, BAcc=0.8570±0.0187, BAcc(opt)=0.8643±0.0173 ✅ VALID

**Best by BAcc(opt):**
- **NCM (ncm_backoff_amp)**: R=9, sim=0.05, AUC=0.9313±0.0128, BAcc=0.8568±0.0165, BAcc(opt)=0.8647±0.0152 ✅ VALID

## Detailed CV5 Results with Standard Deviations

### NCM Backoff (`ncm_backoff`)

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| **AUC** | 0.9161 | 0.0183 | 0.8845 | 0.9322 |
| **BAcc** | 0.8402 | 0.0214 | 0.8001 | 0.8531 |
| **BAcc(opt)** | 0.8445 | 0.0212 | 0.8034 | 0.8531 |
| **Threshold Deviation** | 0.0885 | 0.0421 | 0.0020 | 0.3000 |
| **Valid Models** | 227/360 (63.1%) | - | - | - |

**Best Configuration:**
- R=6, sim=0.05
- AUC: 0.9322±0.0121
- BAcc: 0.8531±0.0175
- BAcc(opt): 0.8531±0.0175
- Status: ✅ VALID (threshold deviation: 0.0600)

### NCM Backoff with Amplitude (`ncm_backoff_amp`)

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| **AUC** | 0.9162 | 0.0182 | 0.8845 | 0.9333 |
| **BAcc** | 0.8403 | 0.0213 | 0.8001 | 0.8570 |
| **BAcc(opt)** | 0.8451 | 0.0211 | 0.8034 | 0.8647 |
| **Threshold Deviation** | 0.0843 | 0.0412 | 0.0020 | 0.2800 |
| **Valid Models** | 241/360 (66.9%) | - | - | - |

**Best Configuration:**
- R=4, sim=0.05 (by BAcc)
- AUC: 0.9319±0.0134
- BAcc: 0.8570±0.0187
- BAcc(opt): 0.8643±0.0173
- Status: ✅ VALID (threshold deviation: 0.1100)

**Best Configuration (by BAcc opt):**
- R=9, sim=0.05
- AUC: 0.9313±0.0128
- BAcc: 0.8568±0.0165
- BAcc(opt): 0.8647±0.0152
- Status: ✅ VALID (threshold deviation: 0.1100)

## Comparison with Other Methods

### Method Validity Rates

| Method | Valid Models | Invalid Models | Validity % | Mean Threshold Deviation |
|--------|--------------|----------------|------------|--------------------------|
| **ncm_backoff_amp** | 241/360 | 119/360 | **66.9%** | 0.0843 |
| **ncm_backoff** | 227/360 | 133/360 | **63.1%** | 0.0885 |
| **key_loo** | 218/360 | 142/360 | 60.6% | 0.0900 |
| **dummy_masking** | 84/360 | 276/360 | 23.3% | 0.2039 |

### Performance Comparison (Best Valid Models)

| Split | Method | AUC | BAcc | BAcc(opt) | Status |
|-------|--------|-----|------|-----------|--------|
| **Random** | ncm_backoff_amp | 0.9360 | 0.8611 | 0.8762 | ✅ VALID |
| **Random** | key_loo | 0.9398 | 0.8624 | 0.8728 | ✅ VALID |
| **Random** | dummy_masking | 0.9204 | 0.8398 | 0.8454 | ✅ VALID |
| **Scaffold** | ncm_backoff | 0.9252 | 0.8637 | 0.8656 | ✅ VALID |
| **Scaffold** | key_loo | 0.9304 | 0.8448 | 0.8582 | ✅ VALID |
| **Scaffold** | dummy_masking | 0.9032 | 0.8366 | 0.8423 | ✅ VALID |
| **CV5** | ncm_backoff | 0.9322±0.0121 | 0.8531±0.0175 | 0.8531±0.0175 | ✅ VALID |
| **CV5** | key_loo | 0.9365±0.0121 | 0.8631±0.0128 | 0.8675±0.0155 | ✅ VALID |
| **CV5** | dummy_masking | 0.9151±0.0201 | 0.8428±0.0216 | 0.8491±0.0222 | ✅ VALID |

## Key Findings

1. **NCM methods achieve competitive performance** with key_loo while avoiding data leakage
2. **NCM has better calibration** than dummy_masking (66.9% vs 23.3% valid models)
3. **NCM CV5 results show low variance** (std ~0.012-0.018 for AUC, ~0.016-0.019 for BAcc)
4. **NCM is production-ready** with realistic performance estimates

## Recommended Configuration

For **production use**, we recommend:

**NCM Backoff with Amplitude (`ncm_backoff_amp`)**:
- **Radius**: 4-7
- **Similarity Threshold**: 0.05-0.15
- **Parameters**:
  - `dmax=1`
  - `amp_source=train_share`
  - `amp_alpha=1.0`, `amp_gamma=1.0`
  - `amp_cap_min=0.25`, `amp_cap_max=1.0`
  - `first_component_only=False`
  - `dist_beta=0.0`

**Expected Performance**:
- AUC: 0.931-0.936 (CV5: 0.9319±0.0134)
- BAcc: 0.857-0.867 (CV5: 0.8570±0.0187)
- BAcc(opt): 0.864-0.876 (CV5: 0.8643±0.0173)
- Validity: ~67% (well-calibrated models)

## Implementation Details

### Core C++ Implementation
- File: `src/molftp_core.cpp`
- Key classes: `ProximityMode`, `NCMAmplitudeParams`, `NCMCounts`
- Methods: `set_proximity_mode()`, `set_proximity_params()`, `set_proximity_amplitude()`

### Python API
- File: `molftp/prevalence.py`
- Methods:
  - `set_proximity_mode(mode)`
  - `set_proximity_params(dmax, lambda_val, train_only)`
  - `set_proximity_amplitude(source, prior_alpha, gamma, cap_min, cap_max, apply_to_train_rows)`
  - `set_proximity_amp_components_policy(first_component_only)`
  - `set_proximity_amp_distance_beta(dist_beta)`

## Conclusion

NCM methods provide a robust, production-ready solution for molecular fragment-target prevalence feature generation. They achieve competitive performance with existing methods while avoiding data leakage and providing better model calibration. The CV5 results demonstrate consistent performance with low variance, making NCM suitable for real-world deployment.

