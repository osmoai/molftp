# Multithreading Implementation Plan for Feature Computation

## Current Status
- ✅ Git author configured: Guillaume Godin <guillaume@osmo.ai>
- ❌ Feature computation is **sequential** (no threading)
- ✅ `num_threads_` is already stored in `MultiTaskPrevalenceGenerator`
- ✅ `get_all_motif_keys_batch_threaded` already supports threading

## Implementation Plan

### 1. Add Threading to `build_3view_vectors_batch`
**Location**: `src/molftp_core.cpp` line ~2275

**Current Code** (sequential):
```cpp
for (int i = 0; i < n_molecules; ++i) {
    // Process molecule i
}
```

**Proposed Solution**: Use OpenMP pragma for parallelization
```cpp
#pragma omp parallel for num_threads(num_threads) if(num_threads > 0)
for (int i = 0; i < n_molecules; ++i) {
    // Process molecule i (thread-safe)
}
```

**Requirements**:
- Add `num_threads` parameter to function signature
- Ensure thread-safety (each molecule processed independently)
- Handle both "max" aggregation path (lines 2306-2420) and non-max path (lines 2291-2303)

### 2. Update Function Signatures
**Files to modify**:
- `build_3view_vectors_batch`: Add `int num_threads = 0` parameter
- `build_vectors_with_key_loo_fixed`: Add `int num_threads = 0` parameter and pass to `build_3view_vectors_batch`
- `MultiTaskPrevalenceGenerator::transform`: Pass `num_threads_` to `build_vectors_with_key_loo_fixed`

### 3. Thread Safety Considerations
- ✅ Each molecule is processed independently (no shared state)
- ✅ Prevalence maps are read-only (const references)
- ✅ Output vectors are pre-allocated (each thread writes to different indices)
- ⚠️ Need to ensure RDKit molecule parsing is thread-safe (should be fine)

### 4. Testing
- Test with `num_threads=0` (auto-detect)
- Test with `num_threads=1` (sequential)
- Test with `num_threads=4` (parallel)
- Verify results are identical across all thread counts
- Measure performance improvement on large datasets (1000+ molecules)

## Implementation Order
1. Add `num_threads` parameter to `build_3view_vectors_batch`
2. Add OpenMP pragma to molecule processing loop
3. Update `build_vectors_with_key_loo_fixed` to accept and pass `num_threads`
4. Update `MultiTaskPrevalenceGenerator::transform` to pass `num_threads_`
5. Test and verify correctness
6. Measure performance improvement

## Notes
- OpenMP is likely already available (used in RDKit/Osmordred)
- If OpenMP not available, can use `std::thread` with manual chunking
- Default `num_threads=0` should mean "use all available cores" or "auto-detect"

