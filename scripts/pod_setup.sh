#!/bin/bash
# Run this first on the pod to set up environment

set -e

echo "=== Pod Setup ==="

# Install requirements
pip install -r requirements.txt

# Pre-download models (do this while setting up to save time later)
echo "Pre-downloading models..."
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
print('Downloading Gemma-2-9b-it-abliterated...')
AutoTokenizer.from_pretrained('IlyaGusev/gemma-2-9b-it-abliterated')
AutoModelForCausalLM.from_pretrained(
    'IlyaGusev/gemma-2-9b-it-abliterated',
    torch_dtype='auto',
    device_map='auto'
)
print('Model cached!')
"

echo "=== Setup Complete ==="
