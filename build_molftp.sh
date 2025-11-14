#!/bin/bash
# Build script for MolFTP v1.8.0
# This script compiles MolFTP and sets up the runtime environment

set -e  # Exit on error

echo "=========================================="
echo "MolFTP v1.8.0 Build Script"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "setup.py" ]; then
    echo "❌ Error: setup.py not found. Please run this script from the molftp directory."
    exit 1
fi

# Check if conda environment is activated
if [ -z "$CONDA_PREFIX" ]; then
    echo "❌ Error: Conda environment not activated. Please run: conda activate build-rdkit-pypi"
    exit 1
fi

echo "✅ Conda environment: $CONDA_PREFIX"

# Verify RDKit is installed
if ! python -c "import rdkit" 2>/dev/null; then
    echo "❌ Error: RDKit not found. Please install RDKit in the conda environment."
    exit 1
fi

RDKIT_DYLIBS="$CONDA_PREFIX/lib/python3.11/site-packages/rdkit/.dylibs"
if [ ! -d "$RDKIT_DYLIBS" ]; then
    echo "❌ Error: RDKit libraries not found at: $RDKIT_DYLIBS"
    exit 1
fi

echo "✅ RDKit libraries found at: $RDKIT_DYLIBS"

# Step 1: Create symlinks
echo ""
echo "Step 1: Creating RDKit library symlinks..."
for lib in SmilesParse Descriptors Fingerprints SubstructMatch DataStructs GraphMol RDGeneral; do
    TARGET="$RDKIT_DYLIBS/libRDKit${lib}.dylib"
    LINK="libRDKit${lib}.dylib"
    
    if [ -f "$TARGET" ]; then
        ln -sf "$TARGET" "$LINK"
        echo "  ✓ Created symlink: $LINK → $TARGET"
    else
        echo "  ⚠️  Warning: Library not found: $TARGET"
    fi
done

# Step 2: Compile
echo ""
echo "Step 2: Compiling MolFTP..."
python setup.py build_ext --inplace

# Step 3: Verify compilation
echo ""
echo "Step 3: Verifying compilation..."
if python -c "import sys; sys.path.insert(0, '.'); from molftp import MultiTaskPrevalenceGenerator; print('✅ Module imports successfully!')" 2>&1; then
    echo "✅ Compilation successful!"
else
    echo "❌ Error: Module import failed. Check errors above."
    exit 1
fi

# Step 4: Set up environment variables
echo ""
echo "Step 4: Setting up runtime environment..."
export DYLD_LIBRARY_PATH="$RDKIT_DYLIBS:$DYLD_LIBRARY_PATH"
echo "✅ DYLD_LIBRARY_PATH set to: $DYLD_LIBRARY_PATH"

# Step 5: Test import with environment
echo ""
echo "Step 5: Testing module import with runtime environment..."
if python -c "import sys; sys.path.insert(0, '.'); from molftp import MultiTaskPrevalenceGenerator; gen = MultiTaskPrevalenceGenerator(radius=6); print(f'✅ Generator created: {gen.get_n_features()} features')" 2>&1; then
    echo "✅ Runtime test successful!"
else
    echo "❌ Error: Runtime test failed. Check DYLD_LIBRARY_PATH."
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ Build Complete!"
echo "=========================================="
echo ""
echo "To use MolFTP, set DYLD_LIBRARY_PATH:"
echo "  export DYLD_LIBRARY_PATH=$RDKIT_DYLIBS:\$DYLD_LIBRARY_PATH"
echo ""
echo "Or run tests with:"
echo "  DYLD_LIBRARY_PATH=$RDKIT_DYLIBS:\$DYLD_LIBRARY_PATH python test_biodegradation_speed_metrics.py"
echo ""

