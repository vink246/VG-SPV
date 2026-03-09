#!/usr/bin/env bash
# Launch DPO training for VG-SPV (Qwen3-VL-2B, VGSPVTrainer).
# Run from repository root: bash scripts/run_dpo_train.sh

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Optional: set GPU(s)
# export CUDA_VISIBLE_DEVICES=0

python train/run_dpo.py \
  --output_dir "${OUTPUT_DIR:-outputs/dpo}" \
  --num_train_epochs "${NUM_EPOCHS:-1}" \
  --per_device_train_batch_size "${BATCH_SIZE:-2}" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-4}" \
  --bf16 \
  "$@"
