#!/usr/bin/env bash

# Download and process HADES and MM-SafetyBench datasets.
# Usage: bash scripts/download_and_process_data.sh
#    or: bash scripts/download_and_process_data.sh ~/scratch/envs/vg-spv

set -e

# Determine the conda target (either by path if provided, or fallback to the env name)
if [ -n "$1" ]; then
  CONDA_TARGET="-p $1"
elif [ -n "$CONDA_DEFAULT_ENV" ]; then
  CONDA_TARGET="-n $CONDA_DEFAULT_ENV"
else
  # Fallback to the explicit environment name if neither is available
  CONDA_TARGET="-n vg-spv"
fi

echo "Using Conda target: $CONDA_TARGET"

echo "Downloading Monosail/HADES dataset..."
conda run --no-capture-output $CONDA_TARGET hf download --repo-type dataset Monosail/HADES --local-dir data/Hades

echo "Downloading PKU-Alignment/MM-SafetyBench dataset..."
conda run --no-capture-output $CONDA_TARGET hf download --repo-type dataset PKU-Alignment/MM-SafetyBench --local-dir data/mm-safebench

echo "Processing HADES dataset..."
conda run --no-capture-output $CONDA_TARGET python scripts/process_hf_data.py --data_folder data/Hades --test_size 0.2 --seed 42

echo "Processing MM-SafetyBench dataset..."
conda run --no-capture-output $CONDA_TARGET python scripts/process_hf_data.py --data_folder data/mm-safebench --test_size 0.2 --seed 42

echo "Data download and processing complete!"