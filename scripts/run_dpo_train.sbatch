#!/bin/bash
#SBATCH --job-name=vgspv-dpo
#SBATCH --output=logs/vgspv-dpo-%j.out
#SBATCH --error=logs/vgspv-dpo-%j.err
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:h200
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=%u@gatech.edu
#
# ============================================================================
# VG-fDPO training (VGSPVTrainer / train/run_dpo.py) under LoRA.
#
# Train / eval splits are fixed (no SPLIT knob):
#   - Training CSV:  ${DATASET_DIR}/traces/train_${METHOD}_dpo.csv
#   - Eval/test CSV: ${DATASET_DIR}/traces/test_${METHOD}_dpo.csv
#
# Style matches scripts/run_full_pipeline.sbatch: hyperparameters live in the
# variables below and are passed explicitly to ``train/run_dpo.py``.
#
# Submit from the repo root:
#       sbatch scripts/run_dpo_train.sbatch
#
# Pre-reqs:
#   - ``vg-spv`` conda env (README §1–§5).
#   - HF access for MODEL_NAME (login / token if gated).
#   - DPO CSVs from ``run_rejected.sbatch`` (or equivalent) for BOTH splits.
#
# LoRA details:
#   - Policy **always** gets fresh trainable DPO LoRA on top of the base VLM
#     when LORA_ADAPTER_PATH is empty (see train/run_dpo.py ``attach_lora``).
#   - LORA_ADAPTER_PATH is **optional**: point at a bbox-SFT (or other) PEFT
#     checkpoint **before** stacking another LoRA for DPO on top.
#   - MERGE_LORA_ADAPTER only applies when LORA_ADAPTER_PATH is set: merge that
#     adapter into dense weights first, then attach new DPO LoRA (see README).
# ============================================================================

set -euo pipefail
module load cuda/12.1.1

# ----------------------------------------------------------------------------
# 2. Pipeline-level knobs (EDIT FREELY)
# ----------------------------------------------------------------------------

DATASET_DIR="data/mm-safebench_1/extracted_data"

# Which rejected-trace family: "method1" | "method2" (must match run_rejected outputs).
METHOD="method2"

# Optional alternate YAML base (empty = configs/dpo.yaml merged with CLI below).
DPO_CONFIG=""

MODEL_NAME="meta-llama/Llama-3.2-11B-Vision-Instruct"
OUTPUT_DIR="outputs/dpo_llama32_11b_vgfdpo"

# Optional bbox-SFT PEFT dir loaded **before** DPO LoRA. Empty = DPO LoRA on base weights only.
LORA_ADAPTER_PATH=""
# Only if LORA_ADAPTER_PATH is set: use "--merge-lora-adapter true" to merge that adapter into dense weights first.
MERGE_LORA_ADAPTER=""

PROMPT_INSTRUCTION=""

# ----------------------------------------------------------------------------
# 3. Hyperparameters (defaults aligned with configs/dpo.yaml; edit here)
# ----------------------------------------------------------------------------

NUM_TRAIN_EPOCHS=10
# Per-device batch 4 is a reasonable default; increase if you have VRAM headroom.
PER_DEVICE_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=8
LEARNING_RATE="5.0e-5"
# Full-sequence token cap (TRL DPOConfig.max_length); long rejected traces / VL need headroom.
MAX_LENGTH=8192
BETA="0.1"
BF16="true"
REF_8BIT="false"
LOGGING_STEPS=10
SAVE_STEPS=200
SAVE_TOTAL_LIMIT=3

USE_VGFDPO="true"
USE_VDPO="false"
ALPHA_VDPO="0.1"
VDPO_MARGIN_M="0.0"

ALPHA_FORMAT="8.0"
GROUNDING_MODE="spatial"    # semantic | spatial
ALPHA_SEM="12.0"
ALPHA_IOU="12.0"
S_FALLBACK_VALUE="1.0"
STRICT_SCALER_INPUTS="false"

LORA_R=64
LORA_ALPHA=128
LORA_DROPOUT="0.05"

# ----------------------------------------------------------------------------
# 4. Job env diagnostics
# ----------------------------------------------------------------------------
echo "==== job env ===="
echo "JOB ID       : ${SLURM_JOB_ID:-N/A}"
echo "NODE         : $(hostname)"
echo "PWD          : $(pwd)"
echo "PYTHON       : $(which python) ($(python --version 2>&1))"
echo "MODEL_NAME   : ${MODEL_NAME}"
echo "OUTPUT_DIR   : ${OUTPUT_DIR}"
echo "GPUs visible :"
nvidia-smi -L 2>/dev/null || echo "  (nvidia-smi not available)"
echo "================="

TRAIN_DATA_PATH="${DATASET_DIR}/traces/train_${METHOD}_dpo.csv"
EVAL_DATA_PATH="${DATASET_DIR}/traces/test_${METHOD}_dpo.csv"

for p in "${TRAIN_DATA_PATH}" "${EVAL_DATA_PATH}"; do
  if [[ ! -f "${p}" ]]; then
    echo "ERROR: DPO CSV not found: ${p}" >&2
    echo "Expected train + test DPO CSVs from run_rejected.sbatch (METHOD=${METHOD})." >&2
    exit 2
  fi
done

echo "TRAIN_DATA_PATH : ${TRAIN_DATA_PATH}"
echo "EVAL_DATA_PATH  : ${EVAL_DATA_PATH}"

# ----------------------------------------------------------------------------
# 5. Optional CLI fragments
# ----------------------------------------------------------------------------
CONFIG_ARGS=()
if [[ -n "${DPO_CONFIG}" ]]; then
  CONFIG_ARGS+=(--config "${DPO_CONFIG}")
fi

LORA_PATH_ARGS=()
if [[ -n "${LORA_ADAPTER_PATH}" ]]; then
  LORA_PATH_ARGS+=(--lora-adapter-path "${LORA_ADAPTER_PATH}")
fi

PROMPT_ARGS=()
if [[ -n "${PROMPT_INSTRUCTION}" ]]; then
  PROMPT_ARGS+=(--prompt-instruction "${PROMPT_INSTRUCTION}")
fi

echo
echo "==== DPO train (train split + test eval each epoch; VG-fDPO; LoRA; V-DPO off) ===="

conda run -n vg-spv --no-capture-output python -u train/run_dpo.py \
    "${CONFIG_ARGS[@]}" \
    --model-name               "${MODEL_NAME}" \
    --data-path                "${TRAIN_DATA_PATH}" \
    --eval-data-path           "${EVAL_DATA_PATH}" \
    --output-dir               "${OUTPUT_DIR}" \
    "${PROMPT_ARGS[@]}" \
    --num-train-epochs          "${NUM_TRAIN_EPOCHS}" \
    --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --learning-rate             "${LEARNING_RATE}" \
    --max-length                "${MAX_LENGTH}" \
    --beta                      "${BETA}" \
    --bf16                      "${BF16}" \
    --ref-8bit                  "${REF_8BIT}" \
    --logging-steps             "${LOGGING_STEPS}" \
    --save-steps                "${SAVE_STEPS}" \
    --save-total-limit          "${SAVE_TOTAL_LIMIT}" \
    --use-vgfdpo                "${USE_VGFDPO}" \
    --use-vdpo                  "${USE_VDPO}" \
    --alpha-vdpo                "${ALPHA_VDPO}" \
    --vdpo-margin-m             "${VDPO_MARGIN_M}" \
    --alpha-format              "${ALPHA_FORMAT}" \
    --grounding-mode            "${GROUNDING_MODE}" \
    --alpha-sem                 "${ALPHA_SEM}" \
    --alpha-iou                 "${ALPHA_IOU}" \
    --s-fallback-value          "${S_FALLBACK_VALUE}" \
    --strict-scaler-inputs      "${STRICT_SCALER_INPUTS}" \
    --lora-r                    "${LORA_R}" \
    --lora-alpha                "${LORA_ALPHA}" \
    --lora-dropout              "${LORA_DROPOUT}" \
    "${LORA_PATH_ARGS[@]}" \
    ${MERGE_LORA_ADAPTER}

echo
echo "==== DONE ===="
echo "Checkpoints : ${OUTPUT_DIR}"
