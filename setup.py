#!/usr/bin/env python3
"""Build script for MolFTP — C++ core (src/molftp_core.cpp) + pybind11 bindings.

The extension links against RDKit's C++ headers and libraries. The supported, tested
path is a conda-forge environment (see environment.yml and BUILD.md): conda-forge's
`rdkit` ships the C++ headers ($PREFIX/include/rdkit) and libraries ($PREFIX/lib), and
pulls in Boost. The plain `pip install rdkit` wheel is RUNTIME-ONLY (no C++ headers) and
cannot be used to build this extension.

Project metadata lives in pyproject.toml; this file only declares the C++ extension.
"""
import os
import sys

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11


def find_rdkit():
    """Return (prefix, include_dirs, lib_dir) for an RDKit install with C++ headers.

    Resolution order:
      1. RDKIT_PREFIX env var  (explicit override)
      2. CONDA_PREFIX          (the active conda env)
      3. sys.prefix            (the running interpreter's prefix — correct for a conda
                                env python even when the env isn't `conda activate`-d)
    Handles both header layouts: include/rdkit/GraphMol/... (conda-forge) and
    include/GraphMol/... (some installs).
    """
    candidates = [p for p in (os.environ.get("RDKIT_PREFIX"),
                              os.environ.get("CONDA_PREFIX"),
                              sys.prefix) if p]
    for prefix in candidates:
        inc = os.path.join(prefix, "include")
        if os.path.exists(os.path.join(inc, "rdkit", "GraphMol", "RDKitBase.h")):
            return prefix, [inc, os.path.join(inc, "rdkit")], os.path.join(prefix, "lib")
        if os.path.exists(os.path.join(inc, "GraphMol", "RDKitBase.h")):
            return prefix, [inc], os.path.join(prefix, "lib")

    sys.exit(
        "\nERROR: RDKit C++ headers not found.\n"
        "MolFTP's C++ core links against RDKit's headers + libraries, which the plain\n"
        "`pip install rdkit` wheel does NOT ship. Use a conda-forge environment:\n\n"
        "    mamba env create -f environment.yml   # or: conda env create -f environment.yml\n"
        "    conda activate molftp\n"
        "    pip install -e .\n\n"
        "Or set RDKIT_PREFIX to an RDKit install containing include/rdkit/ and lib/.\n"
    )


prefix, rdkit_includes, lib_dir = find_rdkit()
print(f"[molftp] building against RDKit in: {prefix}")

RDKIT_LIBS = [
    "RDKitSmilesParse", "RDKitFingerprints", "RDKitSubstructMatch",
    "RDKitDescriptors", "RDKitDataStructs", "RDKitGraphMol", "RDKitRDGeneral",
]

# Runtime search path so the loader finds the RDKit dylibs without DYLD_/LD_LIBRARY_PATH.
rpath = [f"-Wl,-rpath,{lib_dir}"] if sys.platform in ("darwin",) or sys.platform.startswith("linux") else []

ext_modules = [
    Pybind11Extension(
        "_molftp",
        ["src/molftp_core.cpp"],
        include_dirs=[pybind11.get_include(), *rdkit_includes],
        libraries=RDKIT_LIBS,
        library_dirs=[lib_dir],
        extra_link_args=rpath,
        language="c++",
        cxx_std=20,  # RDKit 2026.03 headers use C++20 (constexpr virtual, constexpr dtors)
        define_macros=[("PYBIND11_SIMPLE_GIL_MANAGEMENT", None)],
        extra_compile_args=["-O3"] if sys.platform != "win32" else ["/O2"],
    ),
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext})
