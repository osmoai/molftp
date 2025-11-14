#!/usr/bin/env python3
"""
Setup script for MolFTP (Molecular Fragment-Target Prevalence)
High-performance C++ implementation with Python bindings
"""

from setuptools import setup, find_packages, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11
import os
import sys
import glob
import sysconfig
from pathlib import Path

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
# Only add Boost include when building Boost.Python (to avoid conflicts with pybind11)
conan_boost_include = '/Users/guillaume-osmo/Github/rdkit-pypi/conan/direct_deploy/boost/include'
conan_boost_lib = '/Users/guillaume-osmo/Github/rdkit-pypi/conan/direct_deploy/boost/lib'

# Choose binding backend early to conditionally add Boost includes
binding_backend = os.environ.get('MOLFTP_BINDING', 'pybind11').lower()
if binding_backend not in ['pybind11', 'boost']:
    print(f"Warning: Unknown MOLFTP_BINDING='{binding_backend}', defaulting to 'pybind11'")
    binding_backend = 'pybind11'

include_dirs = [
    pybind11.get_include(),
    "cpp",  # For fast_prevalence headers
    "cpp/wrappers",  # For rdkit_python_h.h wrapper
]
library_dirs = []

# Always add conan boost include path (RDKit headers need boost/config.hpp)
# This is from the RDKit build and is required for RDKit headers to compile
if os.path.exists(conan_boost_include):
    include_dirs.append(conan_boost_include)  # Boost headers (required for RDKit)
    print(f"✅ Added Conan Boost include: {conan_boost_include}")
if os.path.exists(conan_boost_lib):
    library_dirs.append(conan_boost_lib)  # Boost libraries
    print(f"✅ Added Conan Boost lib: {conan_boost_lib}")

# Only add Boost.Python-specific paths when building Boost.Python bindings
# For pybind11 builds, we still need boost/config.hpp but not boost/python.hpp

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

# binding_backend already set above
print(f"🔧 Building MolFTP with {binding_backend} bindings")

# Common RDKit libraries
rdkit_libraries = [
    "RDKitSmilesParse",
    "RDKitDescriptors",
    "RDKitFingerprints",
    "RDKitSubstructMatch",
    "RDKitDataStructs",
    "RDKitGraphMol",
    "RDKitRDGeneral"
]

if binding_backend == 'boost':
    # Boost.Python bindings
    import numpy as np
    
    def detect_boost_python_libs():
        """Detect Boost.Python and Boost.NumPy libraries.
        
        Checks:
        1. Conan boost directory (from RDKit build) - /Users/guillaume-osmo/Github/rdkit-pypi/conan/direct_deploy/boost/lib/
        2. Conda environment
        3. System paths
        """
        libs, libdirs = [], []
        candidates = []
        
        # First: Check conan boost directory (from RDKit build)
        conan_boost_lib = '/Users/guillaume-osmo/Github/rdkit-pypi/conan/direct_deploy/boost/lib'
        if os.path.exists(conan_boost_lib):
            candidates.insert(0, conan_boost_lib)  # Prioritize conan boost
            print(f"✅ Checking Conan Boost lib: {conan_boost_lib}")
        
        # Second: Conda environment
        if os.environ.get("CONDA_PREFIX"):
            candidates.append(os.path.join(os.environ["CONDA_PREFIX"], "lib"))
        
        # Third: System paths
        candidates.extend(sysconfig.get_config_vars().get("LIBDIR", "").split(os.pathsep))
        candidates.extend(["/usr/local/lib", "/usr/lib", "/opt/homebrew/lib"])
        candidates = [p for p in candidates if p and os.path.exists(p)]
        
        # Python tag (e.g., 311, 312)
        tag = f"{sys.version_info.major}{sys.version_info.minor}"
        
        # Try several names
        names = [f"boost_python{tag}", "boost_python3", "boost_python"]
        names_np = [f"boost_numpy{tag}", "boost_numpy3", "boost_numpy"]
        
        found = set()
        found_libs = {}  # Map libname -> (libdir, full_path)
        for d in candidates:
            for pat in names + names_np:
                for f in glob.glob(os.path.join(d, f"lib{pat}*")):
                    libdirs.append(d)
                    base = Path(f).name
                    libname = base[3:] if base.startswith("lib") else base
                    libname = libname.split(".")[0]
                    found.add(libname)
                    if libname not in found_libs:
                        found_libs[libname] = (d, f)
        
        py = next((x for x in names if x in found), None)
        npy = next((x for x in names_np if x in found), None)
        
        if not py:
            raise RuntimeError(
                f"Could not find Boost.Python library. "
                f"Searched in: {candidates}. "
                f"Found: {sorted(found)}. "
                f"Python tag: {tag}. "
                f"Install: conda install -c conda-forge libboost-python"
            )
        
        if not npy:
            print(f"⚠️  Warning: Boost.NumPy not found, but continuing with Boost.Python only")
            print(f"   Found libraries: {sorted(found)}")
            # Boost.NumPy might not be required for basic functionality
            return [py], sorted(set(libdirs))
        
        print(f"✅ Found Boost.Python: {py} in {found_libs.get(py, ('unknown', ''))[0]}")
        print(f"✅ Found Boost.NumPy: {npy} in {found_libs.get(npy, ('unknown', ''))[0]}")
        
        return [py, npy], sorted(set(libdirs))
    
    boost_libs, boost_libdirs = detect_boost_python_libs()
    
    import platform
    extra_compile_args = ['-O3', '-march=native', '-std=c++17']
    if platform.machine().lower() in ("x86_64", "amd64"):
        extra_compile_args.append('-mpopcnt')
    
    if sys.platform == 'win32':
        extra_compile_args = ['/O2', '/std:c++17']  # MSVC already has __popcnt64
    
    ext_modules = [
        Extension(
            "_molftp_boost",  # Different module name to avoid conflicts
            ["cpp/bindings/boost/molftp_core_boost.cpp", "cpp/molftp/fast_prevalence.cpp"],
            include_dirs=include_dirs + [np.get_include(), "cpp"],
            library_dirs=library_dirs + boost_libdirs,
            libraries=rdkit_libraries + boost_libs,
            extra_link_args=['-Wl,-rpath,@loader_path/rdkit/.dylibs', '-Wl,-rpath,' + os.path.join(os.environ.get('CONDA_PREFIX', ''), 'lib/python3.11/site-packages/rdkit/.dylibs')] if sys.platform == 'darwin' else [],
            language='c++',
            extra_compile_args=extra_compile_args,
        ),
    ]
    print("✅ Using Boost.Python bindings")
else:
    # pybind11 bindings (default)
    import platform
    extra_compile_args_pybind = ['-O3', '-march=native']
    if platform.machine().lower() in ("x86_64", "amd64"):
        extra_compile_args_pybind.append('-mpopcnt')
    
    if sys.platform == 'win32':
        extra_compile_args_pybind = ['/O2']  # MSVC already has __popcnt64
    
    ext_modules = [
        Pybind11Extension(
            "_molftp",
            ["src/molftp_core.cpp", "cpp/molftp/fast_prevalence.cpp"],
            include_dirs=include_dirs + ["cpp"],
            libraries=rdkit_libraries,
            library_dirs=library_dirs,
            extra_link_args=['-Wl,-rpath,@loader_path/rdkit/.dylibs', '-Wl,-rpath,' + os.path.join(os.environ.get('CONDA_PREFIX', ''), 'lib/python3.11/site-packages/rdkit/.dylibs')] if sys.platform == 'darwin' else [],
            language='c++',
            cxx_std=17,
            define_macros=[('PYBIND11_SIMPLE_GIL_MANAGEMENT', None)],
            extra_compile_args=extra_compile_args_pybind,
        ),
    ]
    print("✅ Using pybind11 bindings (default)")

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="molftp",
    version="1.8.0",
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
        "pybind11>=2.10.0",  # Required for default pybind11 bindings
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

