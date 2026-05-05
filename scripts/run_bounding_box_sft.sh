#!/usr/bin/env bash
# Bounding-box SFT (LoRA) on MM-SafetyBench CSV traces. Run from repo root:
#   MODEL_NAME=meta-llama/Llama-3.2-11B-Vision-Instruct bash scripts/run_bounding_box_sft.sh
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python train/run_bounding_box_sft.py \
  --model_name "${MODEL_NAME:?Set MODEL_NAME}" \
  --output_dir "${OUTPUT_DIR:-outputs/bounding_box_sft}" \
  --bf16 \
  --gradient_checkpointing \
  "$@"
