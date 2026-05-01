#!/usr/bin/env bash
# Bounding-box SFT (LoRA). Run from repo root: bash scripts/run_bounding_box_sft.sh
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python train/run_bounding_box_sft.py \
  --model_name "${MODEL_NAME:-llava-hf/llava-1.6-mistral-7b-hf}" \
  --dataset_id "${DATASET_ID:-PaDT-MLLM/RefCOCO}" \
  --split "${SPLIT:-train}" \
  --output_dir "${OUTPUT_DIR:-outputs/bounding_box_sft}" \
  --bf16 \
  --gradient_checkpointing \
  "$@"
