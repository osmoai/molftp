# Building MolFTP

MolFTP has a C++ core (`src/molftp_core.cpp`) with pybind11 bindings that links against
**RDKit's C++ headers and libraries**. The standard `pip install rdkit` wheel is
*runtime-only* — it does **not** ship the C++ headers — so it cannot build this extension.
conda-forge's `rdkit` does ship them (and pulls in Boost), so that is the supported path.

## Quick start (recommended)

```bash
# 1. Create the environment — RDKit 2026.03 + build deps, one command
conda env create -f environment.yml      # or: mamba env create -f environment.yml
conda activate molftp

# 2. Build + install in editable mode
pip install -e .

# 3. Verify
python -c "import molftp; print('molftp', molftp.__version__, 'OK')"
pytest -q                                 # optional: run the test suite
```

`setup.py` auto-detects RDKit from the active conda env via `$CONDA_PREFIX` — no paths to
edit, no environment variables to set.

## How detection works

`setup.py` resolves the RDKit prefix in this order:

1. **`RDKIT_PREFIX`** — explicit override (set this to use a custom RDKit build).
2. **`CONDA_PREFIX`** — the active conda env (the recommended path above).

It then adds the right include directories (handling both the conda-forge
`include/rdkit/GraphMol/...` layout and the plain `include/GraphMol/...` layout) and links
the seven RDKit libraries the core needs. An `-rpath` to the env's `lib/` is baked in, so
the RDKit dylibs are found at import time **without** any `DYLD_LIBRARY_PATH` /
`LD_LIBRARY_PATH` juggling.

## Custom RDKit location

If you built RDKit yourself (headers under `<prefix>/include/rdkit/` and libs under
`<prefix>/lib/`):

```bash
export RDKIT_PREFIX=/path/to/your/rdkit/prefix
pip install -e .
```

## Requirements

- A **C++20** compiler (clang on macOS, gcc or clang on Linux). RDKit 2026.03's headers use
  C++20 features (`constexpr virtual`, `constexpr` destructors), so C++17 no longer compiles.
  `setup.py` sets `cxx_std=20`.
- RDKit **2026.03** is what we build against; 2022.03+ is expected to work.
- Tested on macOS (Apple Silicon) and Linux x86-64.

## Why conda-forge (the dev-package split)

conda-forge splits RDKit into separate packages, and this trips up most build attempts:

| package | ships | needed to… |
|---|---|---|
| `rdkit` | Python module + runtime libs | *run* molftp |
| `librdkit-dev` | **C++ headers** (`include/rdkit/...`) + dev symlinks | *build* molftp |
| `libboost-devel` | **Boost headers** (RDKit headers `#include <boost/...>`) | *build* molftp |

`environment.yml` lists all three. Installing only `rdkit` is the #1 cause of
"RDKit C++ headers not found" and "`boost/...` file not found".

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ERROR: RDKit C++ headers not found` at build | Missing `librdkit-dev` (or not in the conda env / used the pip `rdkit` wheel). Re-create from `environment.yml`, `conda activate molftp`, or set `RDKIT_PREFIX`. |
| `fatal error: 'boost/...' file not found` | Missing `libboost-devel`. Re-create the env from `environment.yml`. |
| `error: constexpr ... virtual function cannot be constexpr` (in `Geometry/point.h`) | Compiling RDKit 2026.03 headers as C++17. Ensure `cxx_std=20` (already set) and a C++20-capable compiler (`cxx-compiler` from conda-forge). |
| `ImportError: library not loaded ... libRDKit*.dylib` at import | RDKit dylibs not on the loader path. Import from the same conda env you built in; the baked-in `-rpath` handles this automatically. |
| Linker can't find `-lRDKit*` | RDKit libs missing from the env. Re-create from `environment.yml`. |
