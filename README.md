# MolFTP: Molecular Fragment-Target Prevalence

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![C++17](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)

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

- Python >= 3.8
- RDKit >= 2022.3.0
- NumPy >= 1.19.0
- C++17 compatible compiler

### Install from source

```bash
# Clone the repository
git clone https://github.com/osmoai/molftp.git
cd molftp

# Create and activate conda environment with build tools
mamba create -n rdkit_dev cmake librdkit-dev eigen libboost-devel compilers
conda activate rdkit_dev

# Install Python dependencies
conda install -c conda-forge numpy pandas scikit-learn
conda install -c conda-forge rdkit

# Build and install
python setup.py install
```

**Note**: Use `mamba` for faster dependency resolution, or replace with `conda` if mamba is not installed.

### Quick install with pip (coming soon)

```bash
pip install molftp
```

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

## Examples

See the `examples/` directory for comprehensive examples:

- **`example_single_task_keyloo.py`**: Basic single-task feature generation
- **`example_single_task_dummymask.py`**: Cross-validation with Dummy-Masking
- **`example_multitask_keyloo.py`**: Multi-task feature generation
- **`example_multitask_dummymask.py`**: Multi-task CV with sparse labels

## Methods

### Key-LOO (Key Leave-One-Out)

- Filters keys appearing in <= k molecules (default k=2)
- Applies rescaling factor: `(n - k) / n` for better extrapolation
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

