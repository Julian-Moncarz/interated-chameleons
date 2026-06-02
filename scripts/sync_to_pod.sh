#!/bin/bash
# Sync local code to runpod (code only, preserves outputs on pod)
# Usage: ./sync_to_pod.sh

echo "=== Packaging code for runpod ==="

# Create archive of CODE ONLY (not data/outputs)
tar -czf /tmp/neuralcham_code.tar.gz \
    --exclude='checkpoints' \
    --exclude='outputs' \
    --exclude='data' \
    --exclude='hf' \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.venv' \
    --exclude='uv-cache' \
    --exclude='*.log' \
    src scripts main.py setup.sh pyproject.toml requirements.txt uv.lock

echo "Archive created: /tmp/neuralcham_code.tar.gz"

# Send to pod
runpodctl send /tmp/neuralcham_code.tar.gz

echo ""
echo "=== On the pod, run: ==="
echo "  runpodctl receive"
echo "  # Clean old code (keeps checkpoints/outputs/data)"
echo "  rm -rf src scripts main.py setup.sh pyproject.toml requirements.txt uv.lock"
echo "  tar -xzf neuralcham_code.tar.gz"
echo "  rm neuralcham_code.tar.gz"
