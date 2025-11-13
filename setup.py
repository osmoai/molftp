#!/usr/bin/env python3
"""
Setup script for MolFTP (Molecular Fragment-Target Prevalence)
High-performance C++ implementation with Python bindings
"""

from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11
import os
import sys

# Try to detect RDKit installation
def find_rdkit_paths():
    """Attempt to find RDKit installation paths."""
    import subprocess
    import sysconfig
    
    # Try conda environment
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    if conda_prefix:
        include = os.path.join(conda_prefix, 'include')
        lib = os.path.join(conda_prefix, 'lib')
        if os.path.exists(os.path.join(include, 'rdkit')):
            return include, lib
    
    # Try system Python site-packages
    site_packages = sysconfig.get_paths()["purelib"]
    rdkit_include = os.path.join(site_packages, 'rdkit', 'include')
    if os.path.exists(rdkit_include):
        return rdkit_include, os.path.join(site_packages, 'rdkit', 'lib')
    
    # Fallback to common locations
    common_paths = [
        ('/usr/local/include', '/usr/local/lib'),
        ('/opt/homebrew/include', '/opt/homebrew/lib'),
        ('/usr/include', '/usr/lib'),
    ]
    
    for include_path, lib_path in common_paths:
        if os.path.exists(os.path.join(include_path, 'rdkit')):
            return include_path, lib_path
    
    # If not found, return empty and hope compiler finds it
    print("Warning: Could not auto-detect RDKit paths. Using system defaults.")
    return '', ''

rdkit_include, rdkit_lib = find_rdkit_paths()

include_dirs = [pybind11.get_include()]
library_dirs = []

if rdkit_include:
    include_dirs.extend([rdkit_include, os.path.join(rdkit_include, 'rdkit')])
if rdkit_lib:
    library_dirs.append(rdkit_lib)

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "_molftp",
        ["src/molftp_core.cpp"],
        include_dirs=include_dirs,
        libraries=[
            "RDKitSmilesParse",
            "RDKitDescriptors",
            "RDKitFingerprints",
            "RDKitSubstructMatch",
            "RDKitDataStructs",
            "RDKitGraphMol",
            "RDKitRDGeneral"
        ],
        library_dirs=library_dirs,
        language='c++',
        cxx_std=17,
        define_macros=[('PYBIND11_SIMPLE_GIL_MANAGEMENT', None)],
        extra_compile_args=['-O3', '-march=native'] if sys.platform != 'win32' else ['/O2'],
    ),
]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="molftp",
    version="1.6.0",
    author="Guillaume GODIN",
    author_email="",
    description="Molecular Fragment-Target Prevalence: High-performance feature generation for molecular property prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/osmoai/molftp",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "rdkit>=2022.3.0",
        "pybind11>=2.10.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
        ],
        "ml": [
            "xgboost>=1.5.0",
            "lightgbm>=3.2.0",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="molecular-features cheminformatics machine-learning molecular-property-prediction fragment-prevalence",
)

