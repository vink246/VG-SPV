#!/usr/bin/env bash
# Install flash-attn (requires conda's PyTorch at build time; may use prebuilt wheel).
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
# Keep temp and pip cache on the same filesystem as the env to avoid cross-device link
export TMPDIR="${CONDA_PREFIX}/.tmp"
export PIP_CACHE_DIR="${CONDA_PREFIX}/.cache/pip"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
echo "Installing flash-attn (may use prebuilt wheel; TMPDIR and pip cache under env)..."
"$CONDA_PREFIX/bin/pip" install --no-build-isolation flash-attn
