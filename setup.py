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
    """Attempt to find RDKit installation paths.
    
    According to BUILD_SUCCESS_ALL_WHEELS.md:
    1. Check RDKIT_INCLUDE environment variable first (for build-time headers)
    2. RDKit wheel includes headers in site-packages/rdkit/include/rdkit/
    3. Conda-forge RDKit has headers in CONDA_PREFIX/include/rdkit/
    """
    import sysconfig
    
    # First: Check environment variables (for build-time headers from RDKit build)
    rdkit_include_env = os.environ.get('RDKIT_INCLUDE', '')
    rdkit_lib_env = os.environ.get('RDKIT_LIB', '')
    if rdkit_include_env and os.path.exists(rdkit_include_env):
        print(f"✅ Using RDKit from environment: {rdkit_include_env}")
        return rdkit_include_env, rdkit_lib_env
    
    # Second: Try RDKit site-packages include directory (rdkit-pypi wheel)
    # Wheel structure: rdkit/include/rdkit/RDGeneral/export.h
    # So we need rdkit/include/rdkit in include path for <RDGeneral/export.h>
    # Check site-packages directly (don't require RDKit import to work)
    site_packages = sysconfig.get_paths()["purelib"]
    rdkit_include_wheel = os.path.join(site_packages, 'rdkit', 'include', 'rdkit')
    if os.path.exists(rdkit_include_wheel) and os.path.exists(os.path.join(rdkit_include_wheel, 'RDGeneral')):
        # Return rdkit/include/rdkit so <RDGeneral/export.h> resolves correctly
        rdkit_path = os.path.join(site_packages, 'rdkit')
        rdkit_lib_wheel = os.path.join(rdkit_path, '.dylibs')
        if not os.path.exists(rdkit_lib_wheel):
            rdkit_lib_wheel = os.path.join(rdkit_path, 'lib') if os.path.exists(os.path.join(rdkit_path, 'lib')) else site_packages
        print(f"✅ Found RDKit wheel headers: {rdkit_include_wheel}")
        return rdkit_include_wheel, rdkit_lib_wheel
    
    # Also try importing RDKit (if it works)
    try:
        import rdkit
        rdkit_path = os.path.dirname(rdkit.__file__)
        # Check for include/rdkit directory in rdkit package (wheel structure)
        rdkit_include_wheel = os.path.join(rdkit_path, 'include', 'rdkit')
        if os.path.exists(rdkit_include_wheel) and os.path.exists(os.path.join(rdkit_include_wheel, 'RDGeneral')):
            # Return rdkit/include/rdkit so <RDGeneral/export.h> resolves correctly
            rdkit_lib_wheel = os.path.join(rdkit_path, '.dylibs')
            if not os.path.exists(rdkit_lib_wheel):
                rdkit_lib_wheel = os.path.join(rdkit_path, 'lib') if os.path.exists(os.path.join(rdkit_path, 'lib')) else os.path.dirname(rdkit_path)
            return rdkit_include_wheel, rdkit_lib_wheel
    except ImportError:
        pass
    
    # Third: Try conda environment include directory (conda-forge RDKit)
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    if conda_prefix:
        include = os.path.join(conda_prefix, 'include')
        lib = os.path.join(conda_prefix, 'lib')
        # Check for rdkit subdirectory (conda-forge installation)
        rdkit_include = os.path.join(include, 'rdkit')
        if os.path.exists(rdkit_include) and os.path.exists(os.path.join(rdkit_include, 'RDGeneral')):
            return include, lib  # Return parent include dir so <RDGeneral/export.h> works
        # Check if RDKit headers are directly in include (some installations)
        if os.path.exists(os.path.join(include, 'RDGeneral')):
            return include, lib
    
    # Fallback: Try system Python site-packages
    site_packages = sysconfig.get_paths()["purelib"]
    rdkit_include = os.path.join(site_packages, 'rdkit', 'include')
    if os.path.exists(rdkit_include):
        return rdkit_include, os.path.join(site_packages, 'rdkit', 'lib')
    
    # Last resort: common locations
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
    print("  Hint: Install RDKit from conda-forge: conda install -c conda-forge rdkit")
    return '', ''

rdkit_include, rdkit_lib = find_rdkit_paths()

# Conan Boost paths (for consistency with RDKit build)
conan_boost_include = '/Users/guillaume-osmo/Github/rdkit-pypi/conan/direct_deploy/boost/include'
conan_boost_lib = '/Users/guillaume-osmo/Github/rdkit-pypi/conan/direct_deploy/boost/lib'

include_dirs = [
    pybind11.get_include(),
    conan_boost_include,  # Boost headers (required by RDKit)
]
library_dirs = [
    conan_boost_lib,  # Boost libraries
]

if rdkit_include:
    # RDKit headers structure depends on installation type:
    # - Wheel: rdkit/include/rdkit/RDGeneral/export.h -> need rdkit/include/rdkit in path
    # - Conda: include/rdkit/RDGeneral/export.h -> need include in path
    # Check if this is the wheel structure (ends with /rdkit)
    if rdkit_include.endswith('/rdkit') or rdkit_include.endswith('\\rdkit'):
        # Wheel structure: already pointing to rdkit/include/rdkit
        include_dirs.append(rdkit_include)
    else:
        # Conda structure: rdkit_include is parent, need to add rdkit subdirectory
        include_dirs.append(rdkit_include)
        rdkit_subdir = os.path.join(rdkit_include, 'rdkit')
        if os.path.exists(rdkit_subdir):
            include_dirs.append(rdkit_subdir)
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
        extra_link_args=['-Wl,-rpath,@loader_path/rdkit/.dylibs'] if sys.platform == 'darwin' else [],
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

