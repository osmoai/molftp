#!/usr/bin/env python3
"""
Setup script for MolFTP - Molecular Fragment-Target Prevalence
"""

import os
import sys
import subprocess
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11

def find_rdkit_paths():
    """Find RDKit installation paths."""
    import subprocess
    import sysconfig
    
    # First, check environment variables (set by parent build process)
    rdkit_include_env = os.environ.get('RDKIT_INCLUDE', '')
    rdkit_lib_env = os.environ.get('RDKIT_LIB', '')
    if rdkit_include_env and os.path.exists(rdkit_include_env):
        print(f"✅ Using RDKit from environment: {rdkit_include_env}")
        return rdkit_include_env, rdkit_lib_env
    
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
    
    print("Warning: Could not auto-detect RDKit paths. Using system defaults.")
    return '', ''

# Find RDKit paths
rdkit_include, rdkit_lib = find_rdkit_paths()

# Get pybind11 include path
pybind11_include = pybind11.get_include()

# Define the extension
ext_modules = [
    Pybind11Extension(
        "_molftp",
        sources=["src/molftp_core.cpp"],
        include_dirs=[
            # RDKit headers (need both the main include and the rdkit subdirectory)
            rdkit_include,
            os.path.join(rdkit_include, "rdkit"),
            # pybind11 headers
            pybind11_include,
            # Python headers
            "/Users/guillaume-osmo/miniconda3/envs/rdkit_build_py311/include/python3.11",
            # Boost headers (from Conan)
            "/Users/guillaume-osmo/Github/rdkit-pypi/conan/direct_deploy/boost/include",
        ],
        libraries=["RDKitGraphMol", "RDKitSmilesParse", "RDKitRDGeneral", "RDKitFingerprints", "RDKitDataStructs", "boost_system", "boost_thread", "boost_filesystem"],
        library_dirs=[
            rdkit_lib,
            "/Users/guillaume-osmo/Github/rdkit-pypi/conan/direct_deploy/boost/lib",
        ],
        language='c++',
        cxx_std=17,
    ),
]

setup(
    name="molftp",
    version="1.0.0",
    author="MolFTP Contributors",
    author_email="",
    description="Molecular Fragment-Target Prevalence: High-performance feature generation for molecular property prediction",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.11",
    install_requires=[
        "rdkit>=2025.3",
        "numpy>=1.23,<2",
        "pandas>=2.0",
        "scikit-learn>=1.3",
    ],
    packages=["molftp"],
    package_dir={"molftp": "molftp"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
)
