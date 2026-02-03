#!/bin/bash
set -e

ENV_NAME="mini_scientist"

echo "=== Bootstrapping Mini AI-Scientist Environment using Local Conda ==="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda could not be found."
    exit 1
fi

# Create environment if it doesn't exist
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' already exists. Updating..."
else
    echo "Creating environment '$ENV_NAME'..."
    conda create -n $ENV_NAME python=3.10 -y
fi

# Activate environment
# Note: 'conda activate' inside script requires eval of conda shell hook
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

echo "Installing dependencies..."
# Install base requirements
pip install -r requirements.txt

# Install project specific requirements
pip install pysr sympy networkx pandas

# Install Julia for PySR
echo "Checking PySR Julia installation..."
python -c "import pysr; pysr.install()"

echo "=== Bootstrap Complete ==="
echo "To activate the environment, run: conda activate $ENV_NAME"
