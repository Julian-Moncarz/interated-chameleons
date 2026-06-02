#!/bin/bash
# Download results from runpod
# Run this on your local machine after the pod completes

echo "=== Downloading results from pod ==="
echo "On the pod, run first:"
echo "  tar -czf results.tar.gz checkpoints/ outputs/"
echo "  runpodctl send results.tar.gz"
echo ""
echo "Then run on local:"
echo "  runpodctl receive"
echo "  tar -xzf results.tar.gz"
