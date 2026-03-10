#!/usr/bin/env bash
# Install flash-attn (requires conda's PyTorch at build time; slow to build).
# Run with the vg-spv env activated, or pass conda env path.
# Usage: bash scripts/install_flash_attn.sh
#    or: bash scripts/install_flash_attn.sh ~/scratch/envs/vg-spv

set -e
if [ -n "$1" ]; then
  CONDA_PREFIX="$1"
fi
if [ -z "$CONDA_PREFIX" ]; then
  echo "Activate your vg-spv env first, or pass it: bash $0 /path/to/env"
  exit 1
fi
echo "Installing flash-attn (this may take a while)..."
"$CONDA_PREFIX/bin/pip" install --no-build-isolation flash-attn
