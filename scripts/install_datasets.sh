#!/usr/bin/env bash

# Download and process HADES and MM-SafetyBench datasets.
# Usage: bash scripts/install_datasets.sh
#    or: bash scripts/install_datasets.sh /path/to/conda/env/prefix
#    or: bash scripts/install_datasets.sh vg-spv   # env name (if not a directory, uses -n)
#
# Conda resolution (first match wins):
#   1) First argument: conda env prefix directory -> conda run -p "..."
#      If it is not a directory, treated as env name -> conda run -n "..."
#   2) Else if CONDA_DEFAULT_ENV is set: conda run -n "$CONDA_DEFAULT_ENV"
#   3) Else: conda run -n vg-spv

set -e

if [ -n "$1" ]; then
  if [ -d "$1" ]; then
    CONDA_TARGET=(-p "$1")
  else
    CONDA_TARGET=(-n "$1")
  fi
elif [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
  CONDA_TARGET=(-n "$CONDA_DEFAULT_ENV")
else
  CONDA_TARGET=(-n vg-spv)
fi

echo "Using conda run ${CONDA_TARGET[*]}"

echo "Downloading Monosail/HADES dataset..."
conda run --no-capture-output "${CONDA_TARGET[@]}" hf download --repo-type dataset Monosail/HADES --local-dir data/Hades

echo "Downloading PKU-Alignment/MM-SafetyBench dataset..."
conda run --no-capture-output "${CONDA_TARGET[@]}" hf download --repo-type dataset PKU-Alignment/MM-SafetyBench --local-dir data/mm-safebench

echo "Processing HADES dataset..."
conda run --no-capture-output "${CONDA_TARGET[@]}" python scripts/process_hf_data.py --data_folder data/Hades --test_size 0.2 --seed 42

echo "Processing MM-SafetyBench dataset..."
conda run --no-capture-output "${CONDA_TARGET[@]}" python scripts/process_hf_data.py --data_folder data/mm-safebench --test_size 0.2 --seed 42

echo "Data download and processing complete!"
