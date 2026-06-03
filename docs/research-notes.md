# Research notes & open questions

## 1. molFTP margin features: magnitude (paper) vs sign-count (code)

### TL;DR
The Key-LOO `(k_j−1)/k_j` prevalence rescale and `loo_smoothing_tau` are **inert** in the current
implementation, and that inertness is **confirmed by the paper's own Figure 6**. The reason is a
discrepancy between the paper and the code in how the *margin* features are built. Deciding what to
do about it (options a/b/c below) **requires re-benchmarking** — we cannot conclude from the paper
alone, because the paper's headline numbers were produced by the current (sign-count) code.

### What the code does (`build_3view_vectors_batch`, default `atom_aggregation="max"`)
Per view the feature vector is `[V0, V1, V2..V(2+R)]` with `R = radius`:
```
p = #atoms with aggregated prevalence ≥ 0 ;  n = #atoms with prevalence ≤ 0
V0   = p − n                       # overall NET SIGN-COUNT
V1   = (p − n) / n_atoms           # normalized net sign-count
V2+d = (pos_d − neg_d) / n_atoms   # per-depth net sign-count (proportion)
```
Only the **sign** of each atom-localized prevalence enters; magnitude is discarded. Hence any
**positive** rescale of the prevalence values — `(k_j−1)/k_j`, `(k_j−1+τ)/(k_j+τ)`, or the
dummy-mask factor — preserves every sign and **cannot change the vector**.

### What the paper specifies (arXiv:2510.06029, Godin 2025)
- **Eq. (5):** `V^(1D) = [ margin, margin_rel, net_0, …, net_R ]`.
- **"MolFTP vector" section (verbatim):** *"the margin is defined as the **maximum positive
  contribution minus the minimum negative contribution** across fragment-level (atom-localized)
  scores … the relative margin is its normalized counterpart."* → **margin is MAGNITUDE-based**
  (a max/min pool of the signed scores), **not** a sign count.
- **Key-LOO:** *"remove the influence of keys observed in only one molecule (singletons)"* →
  this is exactly `k_threshold ≥ 2` filtering. ✓ matches the code's real leakage lever.
- **Dummy-masking factor correction:** `n_train_with_key / n_total_with_key` (a per-key positive
  scalar). **Figure 6 (top row): μ=0.000, σ=0.000** change in the 1D/2D/3D proportion vectors —
  the paper's own ablation empirically shows the rescale is inert on these features, while key-LOO
  (singleton removal) is what actually moves them (middle/bottom rows, σ>0).

### The discrepancy, precisely
| feature | code (`"max"` path) | paper |
|---|---|---|
| `net_0..net_R` (proportions) | net sign-count ✓ | net / proportion ✓ (consistent) |
| `margin` (`V0`), `margin_rel` (`V1`) | net sign-count ✗ | **max(+) − min(−)** magnitude ✗ |

So the **proportion** features match the paper; the **margin / relative-margin** features do not —
the code computes a sign-count where the paper specifies a magnitude margin. If the margin were
magnitude-based as in the paper, the LOO rescale would no longer be inert (it scales the very
magnitudes the margin reads). Note: the alternative `generate_ftp_vector` path (non-`max`
aggregation) may differ; the deviation above is specifically in the default `"max"` fast path.

### Options for the inert LOO params
- **(a) leave as-is** — rescale stays inert; consistent with the *reported results*, but the
  margin/relative-margin features don't match the paper text.
- **(b) delete** the `(k_j−1)/k_j` rescale + `loo_smoothing_tau` as dead code — empirically inert
  (Figure 6 confirms); leakage control remains in `k_threshold` / key-LOO. Premature if (c) is wanted.
- **(c) magnitude-aware margin** — implement the paper's `margin = max(+) − min(−)` for `V0/V1`.
  This makes the LOO rescale meaningful again and matches the paper.

### Can we conclude now, or must we re-test? → **Re-test before choosing (c).**
Already concluded (no new tests needed):
- the rescale's inertness is **real and confirmed by the paper's Figure 6** (σ=0.000);
- `k_threshold` / key-LOO (singleton removal) is the genuine leakage lever (paper-confirmed);
- the **margin features deviate from the paper** (sign-count vs magnitude) — the core open item.

Cannot conclude without re-benchmarking: the paper's headline numbers (Table 2 — XGBoost key-LOO:
**AUROC 0.9053 ± 0.0109, AUPRC 0.9490 ± 0.0058**, BBBP4094) were produced by the **current
sign-count code**. Switching `V0/V1` to the magnitude margin changes the features, so it must be
validated against those numbers.

### Proposed experiment
1. On a branch, implement the paper-faithful magnitude margin for `V0`/`V1`
   (`max(prevalence > 0) − min(prevalence < 0)`, plus its normalized relative margin); keep
   `net_0..net_R` as proportions.
2. Re-run BBBP 10-fold CV (LogReg, RandomForest, XGBoost) at `R=6`, `sim_thresh=0.5`, both key-LOO
   and dummy-masking; compare to Table 2 (target ≈ AUROC 0.905 / AUPRC 0.949) and to the current
   sign-count code on the same splits.
3. Decide:
   - magnitude-margin **matches or beats** sign-count → adopt **(c)**, keep the LOO rescale (now
     meaningful), and the paper and code agree;
   - magnitude-margin **underperforms** → the sign-count code is the de-facto method: either **(a)**
     keep and correct the paper text to describe sign-counts, or **(b)** delete the inert rescale.

Until then: the code is documented as sign-count (`docs/api.md`), and the rescale/`tau` are marked
inert with a passing characterization test
(`tests/test_kloo_core.py::test_positive_rescale_is_inert_for_sign_count_features`).
