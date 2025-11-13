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

# Conan Boost paths (hardcoded for direct use)
conan_boost_include = '/Users/guillaume-osmo/Github/rdkit-pypi/conan/direct_deploy/boost/include'
conan_boost_lib = '/Users/guillaume-osmo/Github/rdkit-pypi/conan/direct_deploy/boost/lib'

# Find RDKit paths (respects RDKIT_INCLUDE and RDKIT_LIB environment variables)
rdkit_include, rdkit_lib = find_rdkit_paths()

# Fallback to conda environment if not found via find_rdkit_paths
if not rdkit_include:
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    rdkit_include = os.path.join(conda_prefix, 'include')
    rdkit_lib = os.path.join(conda_prefix, 'lib', 'python3.11', 'site-packages', 'rdkit', '.dylibs')

# Build include directories - prioritize Conan Boost and RDKit includes
include_dirs = [
    conan_boost_include,
    rdkit_include,  # Use RDKit headers
    os.path.join(rdkit_include, 'rdkit'),  # Add rdkit subdirectory for RDGeneral/export.h
    pybind11.get_include(),
    '/opt/homebrew/include', # Homebrew include
]

# Build library directories - prioritize Conan Boost and RDKit libs
library_dirs = [
    conan_boost_lib,
    rdkit_lib,  # Use RDKit libs
    '/opt/homebrew/lib', # Homebrew lib
    '/opt/homebrew/Cellar/libomp/21.1.5/lib',  # OpenMP library (libomp)
]

# Check if RDKit headers are available
rdkit_headers_available = False
if rdkit_include:
    rdkit_headers_available = os.path.exists(os.path.join(rdkit_include, 'rdkit', 'GraphMol', 'ROMol.h'))

# Define the extension only if RDKit headers are available
ext_modules = []
if rdkit_headers_available:
    print("RDKit headers found. Building C++ extension.")
    ext_modules = [
        Pybind11Extension(
            "_molftp",
            sources=["src/molftp_core.cpp"],
            include_dirs=include_dirs,
            libraries=["RDKitGraphMol", "RDKitSmilesParse", "RDKitRDGeneral", "RDKitFingerprints", "RDKitDataStructs", "boost_system", "boost_thread", "boost_filesystem"],
            library_dirs=library_dirs,
            language='c++',
            cxx_std=17,
            extra_compile_args=[
                '-fvisibility=hidden',
                '-g0',
                '-std=c++17',
                '-mmacosx-version-min=10.14',
                '-O3',
                '-ffast-math',
                '-march=native',
                '-Xpreprocessor', '-fopenmp'  # OpenMP support (libomp installed)
            ],
            extra_link_args=[
                '-lomp'  # Link OpenMP library
            ],
        ),
    ]
else:
    print("Warning: RDKit headers not found. Building Python-only package.")

setup(
    name="molftp",
    version="1.3.0",
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
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
)
