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

### Experiment — DONE (the margin is now a configurable option)

An isolated magnitude-margin build (`max(+) − min(−)` for `V0/V1`, `net_d` unchanged) was compared
head-to-head against the sign-count build on identical seeded folds: 5-fold CV, `R=6`,
`sim_thresh=0.5`, key-LOO, `k_threshold=2`. Three feature modes (**ratio** = sign-count,
**magnitude**, **concat** = `[ratio | magnitude]`) × three models (LogReg, RandomForest, and the
**mlxTM** Tsetlin Machine on thermometer-binarized features). AUROC:

| dataset | model | ratio | magnitude | concat |
|---|---|---|---|---|
| BBBP | LR | 0.8994 | **0.9059** | 0.9025 |
| BBBP | RF | 0.9019 | **0.9078** | 0.9070 |
| BBBP | mlxTM | 0.8942 | 0.8926 | **0.8946** |
| MDR1 | LR | 0.9576 | **0.9680** | 0.9657 |
| MDR1 | RF | 0.9589 | 0.9626 | **0.9661** |
| MDR1 | mlxTM | 0.9630 | **0.9669** | 0.9666 |
| MOR | LR | 0.9149 | **0.9170** | 0.9157 |
| MOR | RF | 0.9360 | **0.9366** | 0.9352 |

The sign-count numbers reproduce the paper's Table 2 (BBBP RF 0.902 vs paper key-LOO 0.8995 /
XGB 0.9053), validating the harness.

**Findings.** `magnitude ≥ concat ≥ ratio` for LR/RF in almost every cell — magnitude is best or
tied-best in 6/8 LR+RF configs; concat sits between (it rarely beats magnitude alone, occasionally
edges it on RF/AUPRC). The mlxTM is essentially mode-agnostic (thermometer binarization discards the
margin magnitude). Gains are small (~+0.005–0.01 AUROC) and within fold-std, but the **direction is
consistent** and matches the paper's robustness claim (Fig. 3).

**Decision — implemented as an option, not a breaking change.** `MultiTaskPrevalenceGenerator` and
`MolFTPClassifier` now accept `margin_mode`:
- `'signcount'` (**default**, backward-compatible, reproduces the published numbers);
- `'magnitude'` (paper eq. 5; the small, consistent best for LR/RF — recommended when matching the
  paper or squeezing accuracy);
- `'both'` (concatenate ratio + magnitude, `+2` features/view).

This makes the paper-faithful margin available (so the LOO rescale is meaningful under `'magnitude'`)
while keeping the de-facto sign-count behaviour as the default. The `(k_j−1)/k_j` rescale / `tau`
remain inert under the default `'signcount'`; under `'magnitude'` they would scale the margin, so a
follow-up could wire the LOO rescale to bite there. Coverage: `tests/test_kloo_core.py::test_margin_mode_option`.
