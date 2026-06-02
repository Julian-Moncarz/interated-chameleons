#!/bin/bash
# Neural Chameleons - Pod Setup Script

set -e

echo "=== Installing uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

echo "=== Installing neovim ==="
curl -LO https://github.com/neovim/neovim/releases/latest/download/nvim-linux-x86_64.tar.gz
rm -rf /opt/nvim-linux-x86_64
tar -C /opt -xzf nvim-linux-x86_64.tar.gz
rm nvim-linux-x86_64.tar.gz
export PATH="$PATH:/opt/nvim-linux-x86_64/bin"

echo "=== Setting up cache directories ==="
echo "XDG_CACHE_HOME, HF_HOME, HF_HUB_CACHE, UV_CACHE_DIR,"
echo "UV_PYTHON_INSTALL_DIR, UV_PYTHON_CACHE_DIR,"
echo "TORCHINDUCTOR_CACHE_DIR, TRITON_CACHE_DIR"
export V=/workspace
mkdir -p $V/{.cache,tmp,uv-cache,uv-python,hf}
export TMPDIR=$V/tmp
export XDG_CACHE_HOME=$V/.cache
export HF_HOME=$V/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export UV_CACHE_DIR=$V/uv-cache
export UV_PYTHON_INSTALL_DIR=$V/uv-python
export UV_PYTHON_CACHE_DIR=$V/uv-python-archives
export TORCHINDUCTOR_CACHE_DIR=$V/.cache/torchinductor
export TRITON_CACHE_DIR=$V/.cache/triton

echo "=== Creating venv and installing requirements ==="
uv venv --python 3.11
uv pip install -r requirements.txt
# flash-attn disabled - prebuilt wheels have ABI mismatches with runpod torch builds
# Using PyTorch SDPA instead (native, no compatibility issues)
# uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

echo "=== Pre-downloading model ==="
MODEL_CACHE="$HF_HUB_CACHE/models--IlyaGusev--gemma-2-9b-it-abliterated"
if [ -d "$MODEL_CACHE/snapshots" ] && [ "$(ls -A $MODEL_CACHE/snapshots 2>/dev/null)" ]; then
    echo "Model already cached at $MODEL_CACHE, skipping download"
else
    uv run python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
print('Downloading Gemma-2-9b-it-abliterated...')
AutoTokenizer.from_pretrained('IlyaGusev/gemma-2-9b-it-abliterated')
AutoModelForCausalLM.from_pretrained('IlyaGusev/gemma-2-9b-it-abliterated', torch_dtype='auto', device_map='auto')
print('Model cached!')
"
fi

echo "=== Setup Complete ==="
