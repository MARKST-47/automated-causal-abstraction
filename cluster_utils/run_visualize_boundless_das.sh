#!/bin/bash
set -e

# Force absolute project context
cd /home/causalml26_team002/Causal_Project

echo "=== 1. Setting Up Absolute Environment Isolation ==="
export PYTHONNOUSERSITE=1

# HIDING THE TOKEN: Dynamically read from an external uncommitted file named hf_token.txt
if [ -f "hf_token.txt" ]; then
    export HF_TOKEN=$(cat hf_token.txt)
else
    echo "ERROR: hf_token.txt not found. Please create it at the project root."
    exit 1
fi

# Bypass internal container OS username lookup bugs
export USER="causalml26_team002"
export LOGNAME="causalml26_team002"
export HOME="/home/causalml26_team002"

# Force all runtime compilation caches to temporary writable spaces
export TORCHINDUCTOR_CACHE_DIR="/tmp/causal_torch_cache"
export TORCH_TRITON_CACHE_DIR="/tmp/causal_triton_cache"
export HF_HOME="/tmp/causal_hf_cache"
export XDG_CACHE_HOME="/tmp/causal_xdg_cache"

echo "=== 2. Creating Environment Linked to Factory GPU Drivers ==="
# --system-site-packages exposes the container's native, working PyTorch setup
/opt/conda/bin/python -m venv --system-site-packages container_venv
source container_venv/bin/activate

echo "=== 3. Installing Supporting Packages Safely ==="
# Install safe utility libraries with exact tokenizers range for transformers 4.48
pip install --no-cache-dir -q "tokenizers>=0.21.0,<0.22.0" "huggingface-hub>=0.24.0" safetensors datasets evaluate networkx scikit-learn matplotlib seaborn tqdm filelock regex requests pyyaml psutil

# ABSOLUTE LOCKOUT - Pin transformers to 4.48.0 to completely avoid the v5.0 NameError bug,
# using --no-deps to ensure pip never touches or corrupts the container's working GPU drivers.
pip install --no-cache-dir --no-deps -q "transformers==4.48.0" pyvene nnsight accelerate

# Expose local repository structure for 'my_pyvene'
export PYTHONPATH="/home/causalml26_team002/Causal_Project:${PYTHONPATH}"

# Force standard command-line progress bars and main-thread data loading across source files
sed -i 's/tqdm.notebook/tqdm/g' src/*.py
sed -i 's/num_workers=[1-9]/num_workers=0/g' src/*.py

python src/visualize_boundless_das.py --base /home/causalml26_team002/Causal_Project 
