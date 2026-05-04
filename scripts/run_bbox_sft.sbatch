#!/bin/bash
#SBATCH --job-name=bbox-sft
#SBATCH --output=logs/bbox-sft-%j.out
#SBATCH --error=logs/bbox-sft-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=psomu3@gatech.edu
#
# ============================================================================
# Bounding-box SFT (LoRA) SLURM dispatcher.
#
# Submit from the repo root:
#   sbatch scripts/run_bbox_sft.sbatch
#
# Notes:
# - This script writes periodic HF step checkpoints under OUTPUT_DIR, but the
#   training code is configured to keep only the *latest* checkpoint folder
#   (save_total_limit=1). The PEFT crash-recovery bundle is also written to
#   OUTPUT_DIR/adapter_latest/ (overwritten each save).
# ============================================================================

set -euo pipefail

mkdir -p logs

# If your cluster uses modules, keep this; otherwise it is harmless to delete.
module load cuda/12.1.1 || true

# ----------------------------------------------------------------------------
# Editable knobs
# ----------------------------------------------------------------------------

MODEL_NAME="meta-llama/Llama-3.2-11B-Vision-Instruct"

# If resuming, set to an existing PEFT adapter directory (e.g. checkpoint-200/ or adapter_latest/).
# Leave empty to start a fresh LoRA run.
RESUME_ADAPTER_PATH=""

OUTPUT_DIR="outputs/bbox_sft_run"

# Checkpoint cadence (0 disables). Only the latest folder is kept.
SAVE_EVERY_STEPS=200

# Training hyperparams (adjust as needed)
EPOCHS=1
PER_DEVICE_TRAIN_BS=1
GRAD_ACCUM=8
LR="2e-4"
WARMUP_RATIO="0.03"

# Dtype/device flags
BF16="--bf16"
GRAD_CKPT="--gradient_checkpointing"

# ----------------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------------
echo "==== job env ===="
echo "JOB ID       : ${SLURM_JOB_ID:-N/A}"
echo "NODE         : $(hostname)"
echo "PWD          : $(pwd)"
echo "PYTHON       : $(which python) ($(python --version 2>&1))"
echo "MODEL_NAME   : ${MODEL_NAME}"
echo "OUTPUT_DIR   : ${OUTPUT_DIR}"
echo "RESUME       : ${RESUME_ADAPTER_PATH:-<none>}"
echo "SAVE_STEPS   : ${SAVE_EVERY_STEPS}"
echo "================="
nvidia-smi -L 2>/dev/null || echo "  (nvidia-smi not available)"
echo "================="

# ----------------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------------

RESUME_ARGS=()
if [[ -n "${RESUME_ADAPTER_PATH}" ]]; then
  RESUME_ARGS=( --resume_adapter_path "${RESUME_ADAPTER_PATH}" )
fi

conda run -n vg-spv --no-capture-output python -u train/run_bounding_box_sft.py \
  --model_name "${MODEL_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BS}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --learning_rate "${LR}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --save_every_steps "${SAVE_EVERY_STEPS}" \
  ${BF16} \
  ${GRAD_CKPT} \
  "${RESUME_ARGS[@]}"

