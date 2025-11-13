# MolFTP Test Suite

## Overview

This test suite validates the Key-LOO fixes in v1.3.0, including:
- Per-molecule Key-LOO rescaling (train-only)
- Inference invariants (batch independence)
- 2D filtering fix (uses 1D counts)
- Smoothing parameter behavior (tau)
- Pickle round-trip compatibility
- Threading parity

## Running Tests

### Prerequisites

1. Build and install MolFTP:
   ```bash
   python setup.py bdist_wheel
   pip install dist/molftp-*.whl
   ```

2. Install pytest:
   ```bash
   pip install pytest pytest-cov
   ```

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_kloo_core.py -v

# With coverage
pytest --cov=_molftp --cov-report=term-missing tests/
```

## Test Coverage

### `test_kloo_core.py`
- `test_per_molecule_rescaling_train_only`: Validates that rescaling is applied only to training molecules
- `test_inference_independence_from_batch`: Ensures batch size doesn't affect feature values
- `test_2d_features_are_nonzero`: Verifies 2D features are populated (not all zero)
- `test_2d_keys_are_subset_of_1d`: Validates 2D filtering uses 1D counts
- `test_tau_smoothing_monotone`: Checks that smoothing parameter behaves correctly

### `test_pickle_and_threaded.py`
- `test_pickle_round_trip`: Validates pickle serialization/deserialization
- `test_threaded_vs_sequential_1d`: Ensures threaded and sequential results match
- `test_invalid_smiles_do_not_crash`: Robustness test for invalid SMILES

## Expected Results

All tests should pass. The suite validates:
- ✅ Per-molecule rescaling works correctly
- ✅ Inference features are independent of batch size
- ✅ 2D features are non-zero (fix verified)
- ✅ 2D keys are subset of 1D keys (correct filtering)
- ✅ Smoothing parameter behaves as expected
- ✅ Pickle round-trip preserves state
- ✅ Threading produces identical results

## Notes

- Tests use a small radius (2) for fast execution
- Tests use synthetic data with halogens for binary classification
- Some tests may be skipped if `_molftp` extension is not available (handled gracefully)

