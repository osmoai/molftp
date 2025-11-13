"""
MolFTP - Molecular Fragment-Target Prevalence

High-performance molecular feature generation based on fragment-target
prevalence statistics with C++ implementation.

Key-LOO: Build features from full dataset (k-filtering + rescaling)
Dummy-Masking: Build features with per-fold masking (requires train indices)
"""

from .prevalence import MultiTaskPrevalenceGenerator

__version__ = "1.5.0"

__all__ = ["MultiTaskPrevalenceGenerator"]
