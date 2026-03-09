#!/usr/bin/env bash
# Install GroundingDINO after the conda env is created.
# Run with the vg-spv env activated, or pass conda env path.
# Usage: bash scripts/install_grounding_dino.sh
#    or: bash scripts/install_grounding_dino.sh ~/scratch/envs/vg-spv

set -e
if [ -n "$1" ]; then
  CONDA_PREFIX="$1"
fi
if [ -z "$CONDA_PREFIX" ]; then
  echo "Activate your vg-spv env first, or pass it: bash $0 /path/to/env"
  exit 1
fi
"$CONDA_PREFIX/bin/pip" install --no-build-isolation 'git+https://github.com/IDEA-Research/GroundingDINO.git'
