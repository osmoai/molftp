# PR Structure for v1.6.0

## PR Branch: `feat/indexed-miners-speedup-v1.6.0`

## Commits (in order)

### Commit 1: Phase 1 - Indexed Neighbor Search
**Commit**: `b6c7fef`  
**Title**: `feat: 10-30× faster fit() via indexed exact Tanimoto search (v1.5.0)`

**Changes**:
- Indexed neighbor search (bit-postings index)
- Exact Tanimoto from counts
- Lower bound pruning
- Packed keys for 1D prevalence
- Lock-free threading
- Version updated to 1.5.0

### Commit 2: Phase 2 & 3 - Fingerprint Caching + Micro-optimizations
**Commit**: `0cc80b9`  
**Title**: `feat: Phase 2 & 3 optimizations - fingerprint caching and micro-optimizations`

**Changes**:
- Global fingerprint cache (`fp_global_`)
- Cache-aware postings builder
- Rare-first bit ordering
- Pre-reservations and tuned capacity
- Updated pair/triplet miners to use cache

### Commit 3: Version & Date Updates
**Commit**: `288bad0` (docs) + `[new commit]` (version)  
**Title**: `chore: Update version to 1.6.0 and fix dates to 2025`

**Changes**:
- Version updated from 1.5.0 → 1.6.0
- All dates updated from 2024 → 2025
- Documentation updated

---

## PR Summary

**Title**: `feat: 15-60× faster fit() via indexed exact Tanimoto search + caching (v1.6.0)`

**Description**: See `V1.5.0_READY_FOR_PR.md` (updated with Phase 2 & 3)

**Key Points**:
- Phase 1: Indexed neighbor search (10-30× speedup)
- Phase 2: Fingerprint caching (1.3-2.0× additional)
- Phase 3: Micro-optimizations (1.1-1.3× additional)
- Combined: 15-60× total speedup expected on 69k molecules
- Version: 1.6.0
- Date: 2025-01-13

---

## Files Changed

### Core Implementation
- `src/molftp_core.cpp`: All three phases

### Version Files
- `molftp/__init__.py`: v1.6.0
- `pyproject.toml`: v1.6.0
- `setup.py`: v1.6.0

### Tests
- `tests/test_indexed_miners_equivalence.py`

### CI/CD
- `.github/workflows/ci.yml`

### Documentation
- `V1.5.0_READY_FOR_PR.md` (updated)
- `PHASE2_PHASE3_SUMMARY.md`
- `PHASE2_PHASE3_COMPLETE.md`
- `COMMIT_INSTRUCTIONS.md`
- `PR_STRUCTURE.md` (this file)

---

**Status**: ✅ Ready for PR creation  
**Author**: Guillaume Godin <guillaume@osmo.ai>  
**Date**: 2025-11-13 (November 13, 2025)

