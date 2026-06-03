# MolFTP API

MolFTP turns molecules (SMILES) into interpretable, statistically-grounded feature vectors
based on **fragment-target prevalence**, and (optionally) predicts labels from them. The Python
API is split into two clearly separated layers:

| layer | module | does | entry points |
|---|---|---|---|
| **inference** | `molftp.features` (a.k.a. `molftp.prevalence`) | SMILES → feature vectors | `MultiTaskPrevalenceGenerator`, `PrevalenceGenerator` |
| **predict** | `molftp.predict` | SMILES → labels / probabilities | `MolFTPClassifier` |

```
inference:  SMILES ──fit/transform──▶ feature vectors
predict:    SMILES ──features──estimator──▶ labels / probabilities
```

The predict layer never re-implements feature generation — it *composes* a feature generator
with a scikit-learn estimator. Use the inference layer alone when you want features for your own
model; use the predict layer for an end-to-end `SMILES → label` estimator.

---

## Install / build

See [BUILD.md](../BUILD.md). In short:

```bash
conda env create -f environment.yml && conda activate molftp
pip install -e .
```

---

## Inference layer — feature generation

### `MultiTaskPrevalenceGenerator` (recommended)

Multi-task generator with NaN-sparse label support, backed by the C++ core.

```python
import numpy as np
from molftp.features import MultiTaskPrevalenceGenerator

gen = MultiTaskPrevalenceGenerator(radius=6, method="key_loo", k_threshold=2)
gen.fit(train_smiles, y_train)            # y_train: (n,) or (n, n_tasks) with NaN for missing
X_test = gen.transform(test_smiles)       # -> np.ndarray, shape (n_test, n_tasks * 3*(2+radius+1))
```

Feature width per task is `3 * (2 + radius + 1)` — three views (1D single fragments, 2D pairs,
3D triplets), each contributing `2 + radius + 1` aggregated statistics. For `radius=6` that is
27 features per task.

Persistence (uses the C++ `py::pickle` protocol under the hood):

```python
gen.save_features("model.pkl")
gen2 = MultiTaskPrevalenceGenerator.load_features("model.pkl")
# or plain pickle — the generator round-trips correctly:
import pickle; gen2 = pickle.loads(pickle.dumps(gen))
```

### `PrevalenceGenerator`

Single-task generator (`fit` / `transform` / `fit_transform`). Returns the three per-view
matrices; concatenate for a flat feature matrix.

---

## Predict layer — `MolFTPClassifier`

A scikit-learn-style classifier: `fit` / `predict` / `predict_proba` / `transform`.

```python
from molftp.predict import MolFTPClassifier

clf = MolFTPClassifier(radius=6, method="key_loo", k_threshold=2).fit(train_smiles, y_train)
labels = clf.predict(test_smiles)
proba  = clf.predict_proba(test_smiles)[:, 1]
X      = clf.transform(test_smiles)        # inference only (features), no prediction
```

- `estimator=` — supply any scikit-learn estimator (default `LogisticRegression(max_iter=1000)`).
  `predict_proba` requires an estimator that implements it.
- `generator=` — supply a pre-configured `MultiTaskPrevalenceGenerator` instead of the keyword args.
- For leakage-safe cross-validation, fit a fresh classifier **per fold** on that fold's training
  molecules only.

---

## Methods: `key_loo` vs `dummy_masking`

- **`key_loo`** — Key Leave-One-Out. Counts key occurrences, **filters rare keys**
  (`k_threshold`, see below), and applies a Key-LOO rescaling pass on training rows. Use when you
  fit on the full (train+valid) set and want rare-fragment filtering.
- **`dummy_masking`** — builds full prevalence without rare-key filtering; at inference,
  out-of-sample molecules use the **frozen** fitted prevalence (keys unseen in training contribute
  0). Call `transform(smiles)` with **no** `train_indices_per_task` for out-of-sample inference.

---

## Parameter reference

| parameter | default | meaning |
|---|---|---|
| `radius` | 6 | Morgan radius for fragment enumeration. |
| `method` | `key_loo` | `key_loo` or `dummy_masking` (see above). |
| `k_threshold` | 2 | **Key-LOO rare-key filter.** A key is kept only if its per-molecule count *and* its total count are `>= k_threshold`. `1` keeps everything; `2` drops keys seen in a single molecule; `3` drops keys seen in ≤2. Higher = more aggressive filtering of rare fragments. |
| `nBits` | 2048 | Fingerprint width for the similarity/pairing step. |
| `sim_thresh` | 0.5 | Tanimoto threshold for forming 2D/3D fragment pairs/triplets. |
| `stat_1d` / `stat_2d` / `stat_3d` | `chi2` / `mcnemar_midp` / `exact_binom` | Significance test per view. |
| `alpha` | 0.5 | Additive smoothing on contingency cells. |
| `num_threads` | -1 | `-1` = all cores, `0` = auto, `>0` = fixed. |

### Notes on `k_threshold`

`k_threshold` is passed all the way through to the C++ core and genuinely changes which keys
survive (and therefore the features). Earlier releases stored it in Python but did **not** pass it
to C++ (it was hardcoded to 2); that is fixed — `test_regressions.py::test_k_threshold_changes_features`
guards against a regression.

### Sign-count features: why the LOO magnitude rescale and `loo_smoothing_tau` are inert

molFTP's three-view features are **sign-based net counts**: for each view, `transform` counts the
atoms whose aggregated fragment prevalence is ≥ 0 (PASS-leaning) vs ≤ 0 (FAIL-leaning) and reports
the net `(pos − neg)` (overall and per depth). Only the **sign** of each prevalence value matters,
never its magnitude. Two consequences worth knowing:

- The Key-LOO `(k_j−1)/k_j` prevalence rescale and `loo_smoothing_tau`'s `(k_j−1+τ)/(k_j+τ)` are
  **positive scalars** — they preserve every sign, so they **cannot change the features**. This is a
  property of the feature design, not a no-op bug. Passing a training mask therefore yields the same
  features as plain inference (pinned by
  `test_kloo_core.py::test_positive_rescale_is_inert_for_sign_count_features`). `loo_smoothing_tau`
  is additionally not wired into the C++ core; setting it ≠ 1.0 emits a `RuntimeWarning`.
- The **effective** rare-key / leakage control is **`k_threshold`**, which *removes* keys (singletons
  at `k_threshold ≥ 2`) and so does change the counts — see
  `test_regressions.py::test_k_threshold_changes_features`. A label-permutation test confirms the
  pipeline is leakage-safe in practice.

If you want the LOO magnitude correction to actually influence the model, the aggregation in
`build_3view_vectors_batch` would need to be made magnitude-aware (e.g. summing/maxing *signed*
prevalence instead of counting signs) — a deliberate change that would shift all downstream results.
