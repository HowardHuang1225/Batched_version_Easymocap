#!/bin/bash
set -e  

echo ">>> Installing main EasyMocap package (editable mode)..."
pip install -e . || { echo "Failed to install EasyMocap"; exit 1; }

echo ">>> Installing pymatch subpackage (editable mode)..."
cd library/pymatch || { echo "Cannot find directory: library/pymatch"; exit 1; }
pip install -e . || { echo "Failed to install pymatch"; exit 1; }

echo "All installations completed successfully!"
