#!/bin/bash
#SBATCH --job-name=bbox-sft
#SBATCH --output=logs/bbox-sft-%j.out
#SBATCH --error=logs/bbox-sft-%j.err
#SBATCH --time=15:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=A40|L40S|A100-40GB|A100-80GB|H100|H200
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=%u@gatech.edu
#
# ============================================================================
# Bounding-box SFT (LoRA) on MM-SafetyBench Method-2 CSV traces.
#
# Submit from the repo root:
#   sbatch scripts/run_bbox_sft.sbatch
#
# Each epoch: HF checkpoint (adapter weights; optimizer optional) + eval_loss on
# VGSPV_EVAL_CSV. Default save_total_limit=0 in the Python script keeps every epoch checkpoint.
# adapter_latest/ is refreshed on each save (crash recovery).
# ============================================================================

set -euo pipefail

mkdir -p logs

module load cuda/12.1.1 || true

# ----------------------------------------------------------------------------
# Editable knobs
# ----------------------------------------------------------------------------

MODEL_NAME="meta-llama/Llama-3.2-11B-Vision-Instruct"

RESUME_ADAPTER_PATH=""

OUTPUT_DIR="outputs/bbox_sft_llama"

VGSPV_CSV="data/mm-safebench_1/extracted_data/traces/train_method2.csv"
VGSPV_EVAL_CSV="data/mm-safebench_1/extracted_data/traces/test_method2.csv"

EVAL_EARLY_STOPPING_PATIENCE=3

SAVE_EVERY_STEPS=0

EPOCHS=1
PER_DEVICE_TRAIN_BS=1
GRAD_ACCUM=8
LR="2e-4"
WARMUP_RATIO="0.03"

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
echo "VGSPV_CSV    : ${VGSPV_CSV}"
echo "VGSPV_EVAL   : ${VGSPV_EVAL_CSV}"
echo "EVAL_ES_PAT  : ${EVAL_EARLY_STOPPING_PATIENCE}"
echo "SAVE_STEPS   : ${SAVE_EVERY_STEPS} (0 = per-epoch checkpoint + val loss)"
echo "RESUME       : ${RESUME_ADAPTER_PATH:-<none>}"
echo "================="
nvidia-smi -L 2>/dev/null || echo "  (nvidia-smi not available)"
echo "================="

RESUME_ARGS=()
if [[ -n "${RESUME_ADAPTER_PATH}" ]]; then
  RESUME_ARGS=( --resume_adapter_path "${RESUME_ADAPTER_PATH}" )
fi

conda run -p ~/scratch/envs/vg-spv --no-capture-output python -u train/run_bounding_box_sft.py \
  --model_name "${MODEL_NAME}" \
  --vgspv_csv "${VGSPV_CSV}" \
  --vgspv_eval_csv "${VGSPV_EVAL_CSV}" \
  --eval_early_stopping_patience "${EVAL_EARLY_STOPPING_PATIENCE}" \
  --output_dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BS}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --learning_rate "${LR}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  $( [[ "${SAVE_EVERY_STEPS:-0}" -gt 0 ]] && echo --save_every_steps "${SAVE_EVERY_STEPS}" ) \
  ${BF16} \
  ${GRAD_CKPT} \
  "${RESUME_ARGS[@]}"
