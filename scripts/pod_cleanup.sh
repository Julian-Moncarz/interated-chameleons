#!/bin/bash
# Cleanup script for runpod - removes source code, keeps outputs/data
# Run this before syncing new code

set -e

cd /workspace

echo "Cleaning up source code directories..."

# Remove code directories (will be replaced with new code)
rm -rf src scripts

# Remove code files
rm -f main.py setup.sh pyproject.toml requirements.txt uv.lock

# Remove temp/staging
rm -rf tempgit tmp

# Remove old tarball
rm -f project.tgz

# Keep these (outputs/data/caches):
# - checkpoints/
# - outputs/
# - data/
# - hf/
# - uv-cache/
# - uv-python/
# - *.md files (optional docs)
# - *.log files

echo "Cleaned. Preserved directories:"
ls -la

echo ""
echo "Ready for new code sync!"
