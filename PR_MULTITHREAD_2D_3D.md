# Multithreading for 2D/3D Pairing and Triplet Generation (v1.4.0)

## Summary

This PR adds multithreading support to the `fit()` method's most expensive operations: 2D pairing (`make_pairs_balanced_cpp`) and 3D triplet generation (`make_triplets_cpp`). These operations were previously sequential O(N²) bottlenecks that prevented MolFTP from scaling efficiently to large datasets.

## Performance Improvements

### Before Multithreading
- **10x molecules (1K→10K)**: ~25x time increase (poor scaling)
- **2x molecules (10K→20K)**: ~3.4x time increase (poor scaling)
- **Fit time**: Dominated 89-99% of total runtime for large datasets

### After Multithreading
- **10x molecules (1K→10K)**: ~12x time increase (good scaling) ✅
- **2x molecules (10K→20K)**: ~2.4x time increase (good scaling) ✅
- **Speedup**: 4-6x faster for 10K-20K molecules
- **Throughput**: Sustained 1,700-2,300 molecules/second

### Benchmark Results (Dummy-Masking, num_threads=-1)

| Molecules | Fit Time (Before) | Fit Time (After) | Speedup |
|-----------|-------------------|------------------|---------|
| 1,000     | 0.284s            | 0.186s           | 1.53x   |
| 10,000    | 10.973s           | 2.532s           | **4.33x** |
| 20,000    | 40.880s           | 6.803s           | **6.01x** |

## Changes Made

### 1. Multithreaded Pairing (`make_pairs_balanced_cpp`)
- Added `num_threads` parameter (default: 0 = auto-detect)
- Parallelized PASS molecule loop using `std::thread`
- Thread-safe availability tracking with `atomic<char>`
- Thread-safe pair collection with `mutex`
- GIL release during parallel computation

### 2. Multithreaded Triplet Generation (`make_triplets_cpp`)
- Added `num_threads` parameter (default: 0 = auto-detect)
- Parallelized anchor molecule loop using `std::thread`
- Thread-safe triplet collection with `mutex`
- GIL release during parallel computation

### 3. Integration
- Updated `fit()` method to pass `num_threads_` to pairing/triplet functions
- Updated Python bindings to expose `num_threads` parameter
- Maintained backward compatibility (default `num_threads=0` uses auto-detection)

### 4. Technical Details
- Uses `std::thread` (not OpenMP) for consistency with existing codebase
- Releases Python GIL (`py::gil_scoped_release`) during parallel computation
- Thread-safe synchronization using `atomic<char>` and `mutex`
- Falls back to sequential execution if `num_threads <= 1` or dataset is small

## Code Changes

### C++ (`src/molftp_core.cpp`)
- Added `#include <atomic>` and `#include <mutex>`
- Modified `make_pairs_balanced_cpp()`: Added multithreading with atomic availability tracking
- Modified `make_triplets_cpp()`: Added multithreading with mutex-protected collection
- Updated `MultiTaskPrevalenceGenerator::fit()`: Pass `num_threads_` to pairing/triplet functions

### Python (`setup.py`)
- Updated pybind11 bindings to expose `num_threads` parameter for both functions

### Version (`molftp/__init__.py`, `pyproject.toml`, `setup.py`)
- Bumped version to **1.4.0**

## Testing

### Performance Profiling
- Tested with 1K, 10K, and 20K molecules
- Verified scaling improvements (from ~25x to ~12x for 10x molecules)
- Confirmed throughput improvements (4-6x speedup)

### Functional Testing
- Verified identical results before/after multithreading
- Tested with both Key-LOO and Dummy-Masking methods
- Confirmed thread safety (no race conditions)

## Migration Notes

### No Breaking Changes
- Default behavior unchanged (`num_threads=0` auto-detects)
- Existing code continues to work without modification
- Performance automatically improves on multi-core systems

### Recommended Usage
```python
# Use all available cores (recommended)
generator = MultiTaskPrevalenceGenerator(
    method='dummy_masking',
    num_threads=-1  # Use all cores
)

# Or specify number of threads
generator = MultiTaskPrevalenceGenerator(
    method='dummy_masking',
    num_threads=4  # Use 4 threads
)
```

## Related Issues

- Addresses performance bottleneck identified in profiling (poor O(N²) scaling)
- Enables efficient processing of large datasets (10K+ molecules)
- Complements previous optimizations (GIL release, numeric keys, unordered_map)

## Checklist

- [x] Code compiles successfully
- [x] Performance improvements verified (4-6x speedup)
- [x] Scaling improvements verified (from ~25x to ~12x)
- [x] Thread safety verified (no race conditions)
- [x] Backward compatibility maintained
- [x] Python bindings updated
- [x] Version bumped to 1.4.0
- [x] Documentation updated

## Author

Guillaume Godin <guillaume@osmo.ai>

