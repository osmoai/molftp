# MolFTP: Molecular Fragment-Target Prevalence

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python 3.8+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![C++20](https://img.shields.io/badge/C++-20-blue.svg)](https://isocpp.org/)
[![RDKit 2026.03](https://img.shields.io/badge/RDKit-2026.03-green.svg)](https://www.rdkit.org/)

High-performance molecular feature generation based on fragment-target prevalence statistics. MolFTP generates interpretable, statistically-grounded features for molecular property prediction with state-of-the-art performance.

**📄 Research Paper**: [Fast Leave-One-Out Approximation from Fragment-Target Prevalence Vectors (molFTP)](https://arxiv.org/abs/2510.06029) (arXiv:2510.06029)

## Features

✨ **Key-LOO Method**: Statistical filtering with leave-one-out rescaling for improved extrapolation to novel fragments

🎯 **Dummy-Masking Method**: Per-fold feature masking for fair cross-validation while maximizing statistical power

🚀 **Multi-Task Learning**: Native support for multiple related prediction tasks with sparse labels (NaN handling)

⚡ **High Performance**: Optimized C++ implementation with Python bindings (10-100x faster than pure Python)

📊 **Interpretable**: Features based on fragment prevalence statistics (chi-squared, McNemar, Fisher's exact tests)

🔬 **Production-Ready**: Extensively validated, mathematically proven correct, publication-quality code

## Installation

### Requirements

- Python >= 3.9
- RDKit (latest tested: **2026.03**; 2022.03+ expected to work)
- A **C++20** compiler (clang on macOS, gcc/clang on Linux) — RDKit 2026.03 headers use C++20
- NumPy, pandas, scikit-learn

MolFTP has a C++ core that links against RDKit's **C++ headers and libraries**. The plain
`pip install rdkit` wheel is runtime-only and cannot build it — you need the conda-forge dev
packages (`librdkit-dev` + `libboost-devel`). `environment.yml` sets all of this up in one step.

### Install from source (recommended)

```bash
git clone https://github.com/osmoai/molftp.git
cd molftp

# One command — RDKit 2026.03 + C++ headers (librdkit-dev) + Boost (libboost-devel) + toolchain
conda env create -f environment.yml      # or: mamba env create -f environment.yml
conda activate molftp

# Build + install (editable). setup.py auto-detects RDKit from the active env.
pip install -e .

# Verify
python -c "import molftp; print('molftp', molftp.__version__, 'OK')"
```

See **[BUILD.md](BUILD.md)** for build internals, a custom-RDKit (`RDKIT_PREFIX`) path, and
troubleshooting.

## Quick Start

### Single-Task Key-LOO

```python
from molftp import MultiTaskPrevalenceGenerator
import numpy as np

# Your molecular data
smiles = ["CC", "CCC", "CCCC", "CCCCC", "CCCCCC"]
labels = np.array([0, 1, 0, 1, 0])

# Generate features with Key-LOO
gen = MultiTaskPrevalenceGenerator(radius=6, method='key_loo')
gen.fit(smiles, labels.reshape(-1, 1), task_names=['activity'])
features = gen.transform(smiles)

print(f"Features shape: {features.shape}")
# Features shape: (5, 27)  # 27 features per molecule
```

### Multi-Task with Sparse Labels

```python
# Multi-task labels (NaN = not measured)
labels_multitask = np.array([
    [0, 1, np.nan],
    [1, 1, 0],
    [0, np.nan, 1],
], dtype=float)

# Generate multi-task features
gen = MultiTaskPrevalenceGenerator(radius=6, method='key_loo')
gen.fit(smiles, labels_multitask, task_names=['task1', 'task2', 'task3'])
features = gen.transform(smiles)

print(f"Multi-task features shape: {features.shape}")
# Features shape: (3, 81)  # 27 features per task × 3 tasks
```

### End-to-end prediction (`SMILES → label`)

The API is split into an **inference** layer (features) and a **predict** layer (labels):

```python
from molftp.predict import MolFTPClassifier

clf = MolFTPClassifier(radius=6, method='key_loo', k_threshold=2).fit(train_smiles, y_train)
labels = clf.predict(test_smiles)
proba  = clf.predict_proba(test_smiles)[:, 1]
X      = clf.transform(test_smiles)     # inference only: features, no prediction
```

`MolFTPClassifier` composes a molFTP feature generator with any scikit-learn estimator
(`estimator=`, default `LogisticRegression`). See **[docs/api.md](docs/api.md)** for the full API,
the inference/predict separation, parameter semantics (incl. `k_threshold`), and method notes.

## Examples

See the `examples/` directory for comprehensive examples:

- **`example_single_task_keyloo.py`**: Basic single-task feature generation
- **`example_single_task_dummymask.py`**: Cross-validation with Dummy-Masking
- **`example_multitask_keyloo.py`**: Multi-task feature generation
- **`example_multitask_dummymask.py`**: Multi-task CV with sparse labels

## Methods

### Key-LOO (Key Leave-One-Out)

- Filters rare keys: keeps a key only if its per-molecule **and** total counts are `>= k_threshold` (default 2)
- `k_threshold` is passed through to the C++ core and genuinely changes the features (see [docs/api.md](docs/api.md))
- Best for: Final model training, prediction on new molecules
- Features are **task-independent** (can be pre-computed once)

### Dummy-Masking

- Builds prevalence on all available data
- Masks test-only keys per fold (set to 0)
- Renormalizes training keys by `N_train / N_full`
- Best for: Fair cross-validation, hyperparameter tuning
- Features are **fold-dependent** (computed per CV fold)

## API Reference

### MultiTaskPrevalenceGenerator

```python
MultiTaskPrevalenceGenerator(
    radius=6,                    # Morgan fingerprint radius
    method='key_loo',           # 'key_loo' or 'dummy_masking'
    key_loo_k=2,               # Min molecules per key (Key-LOO only, default 2)
    rescale_key_loo=True,      # Apply rescaling (Key-LOO only)
    num_threads=-1,            # Number of threads (-1 = all cores)
    counting_method='total'    # 'total', 'unique', or 'binary'
)
```

**Methods**:

- **`fit(smiles, labels, task_names)`**: Build prevalence statistics
  - `smiles`: List of SMILES strings
  - `labels`: np.array of shape `(n_molecules, n_tasks)`
  - `task_names`: List of task names
  
- **`transform(smiles, train_indices_per_task=None)`**: Generate features
  - `smiles`: List of SMILES strings
  - `train_indices_per_task`: For Dummy-Masking only, list of train indices per task
  - Returns: np.array of shape `(n_molecules, n_features)`

## Performance

On the BBBP dataset (Blood-Brain Barrier Penetration, 2039 molecules):

| Method | Single-Task AUROC | Multi-Task AUROC | Speedup vs Python |
|--------|-------------------|------------------|-------------------|
| Key-LOO | 0.9369 ± 0.0115 | **0.9513 ± 0.0085** | 50-100x |
| Dummy-Masking | 0.9205 ± 0.0149 | 0.9110 ± 0.0165 | 50-100x |

Multi-task Key-LOO achieves **state-of-the-art performance** on BBB prediction tasks (new paper in preparation).

## Citation

If you use MolFTP in your research, please cite:

```bibtex
@article{godin2025molftp,
  title={Fast Leave-One-Out Approximation from Fragment-Target Prevalence Vectors (molFTP): From Dummy Masking to Key-LOO for Leakage-Free Feature Construction},
  author={Godin, Guillaume},
  journal={arXiv preprint arXiv:2510.06029},
  year={2025},
  url={https://arxiv.org/abs/2510.06029}
}
```

**Paper**: [Fast Leave-One-Out Approximation from Fragment-Target Prevalence Vectors (molFTP)](https://arxiv.org/abs/2510.06029)  
**Code**: [https://github.com/osmoai/molftp](https://github.com/osmoai/molftp)

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025, Guillaume GODIN Osmo labs pbc. All rights reserved.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- **Author**: Guillaume GODIN (Osmo labs pbc)
- Built on RDKit for molecular structure handling
- Uses pybind11 for Python-C++ interoperability
- Inspired by statistical methods in cheminformatics and bioinformatics

## Support

- **Issues**: [GitHub Issues](https://github.com/osmoai/molftp/issues)
- **Documentation**: See `examples/` directory and this README
- **Contact**: Open an issue for questions or bug reports

---

**MolFTP** - High-performance, interpretable molecular features for the modern ML era.

Developed by Guillaume GODIN @ Osmo labs pbc.

