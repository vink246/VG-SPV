#!/bin/bash
#SBATCH --job-name=vgspv-traces
#SBATCH --output=logs/vgspv-traces-%j.out
#SBATCH --error=logs/vgspv-traces-%j.err
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:h200
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=psomu3@gatech.edu
#
# ============================================================================
# Full VG-SPV preferred-response trace generation pipeline.
#
#   Method 1   GPT-5.4-mini via OpenAI Batch /v1/responses
#                prepare -> submit -> wait -> collect
#   Method 2   Grounding DINO + per-phrase NMS
#                run
#
# Submit from the repo root:
#       sbatch scripts/run_full_pipeline.sbatch
#
# Pre-reqs:
#   - `vg-spv` conda env created (see README §1 - §5).
#   - Grounding DINO checkpoint at weights/groundingdino_swint_ogc.pth.
#   - MM-SafetyBench data extracted to data/mm-safebench_1/extracted_data/.
#   - OPENAI_API_KEY exported in the submitting shell, OR placed (one line,
#     no quotes) in ${HOME}/.config/openai/api_key.
#
# Cost note: this single job holds the GPU allocation during the OpenAI
# `wait` phase, which is idle polling. On a pay-per-GPU-hour cluster, prefer
# splitting into two SLURM jobs:
#   1. CPU-only job for Method 1 (prepare/submit/wait/collect).
#   2. GPU job for Method 2 (run), submitted with --dependency=afterok:<job1>.
# ============================================================================

set -euo pipefail

# ----------------------------------------------------------------------------
# 1. Cluster-specific setup (EDIT IF NEEDED)
# ----------------------------------------------------------------------------


# export OPENAI_API_KEY=""
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY is not set." >&2
    exit 2
fi

# ----------------------------------------------------------------------------
# 2. Pipeline-level knobs (EDIT FREELY)
# ----------------------------------------------------------------------------

# MM-SafetyBench split to process: "train" or "test".
SPLIT="train"

# MM-SafetyBench extracted_data root. Both Method 1 and Method 2 use this.
DATASET_DIR="data/mm-safebench_1/extracted_data"

# Grounding DINO checkpoint for Method 2.
DINO_CHECKPOINT="weights/groundingdino_swint_ogc.pth"

# ----------------------------------------------------------------------------
# 3. Project-level constants (NOT CLI flags)
# ----------------------------------------------------------------------------
# The following are hard-coded in the source files because we never want them
# to drift between runs. Listed here for visibility -- to change, edit the
# Python source (NOT this script) and re-run.
#
#   scripts/generate_method1_traces.py:
#       DEFAULT_BATCH_MODEL          = "gpt-5.4-mini"
#       BATCH_ENDPOINT               = "/v1/responses"
#       DEFAULT_REASONING_EFFORT     = "medium"     # "minimal"|"low"|"medium"|"high"
#       DEFAULT_REASONING_SUMMARY    = None         # None | "auto"|"concise"|"detailed"
#       DEFAULT_TEXT_VERBOSITY       = "low"        # "low"|"medium"|"high"
#       DEFAULT_MAX_OUTPUT_TOKENS    = 4000
#       MMSB_COHORT_CATEGORIES       = {"Physical_Harm", "Illegal_Activitiy"}
#       MMSB_COHORT_SUBSET_SUBSTRING = "SD"
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# 4. Job env diagnostics (logged, easy to grep on failure)
# ----------------------------------------------------------------------------
echo "==== job env ===="
echo "JOB ID       : ${SLURM_JOB_ID:-N/A}"
echo "NODE         : $(hostname)"
echo "PWD          : $(pwd)"
echo "PYTHON       : $(which python) ($(python --version 2>&1))"
echo "SPLIT        : ${SPLIT}"
echo "DATASET_DIR  : ${DATASET_DIR}"
echo "DINO_CKPT    : ${DINO_CHECKPOINT}"
echo "GPUs visible :"
nvidia-smi -L 2>/dev/null || echo "  (nvidia-smi not available)"
echo "================="
pwd
echo "================="

# ============================================================================
# Method 1 -- GPT-5.4-mini via OpenAI Batch /v1/responses
# ============================================================================

# ---- Method 1 prepare -------------------------------------------------------
# Build the OpenAI batch input JSONL chunks from the cohort-filtered metadata.
# All defaults shown inline; --limit is opt-in (set M1_LIMIT to enable).
M1_LIMIT=""    # set to e.g. M1_LIMIT="--limit 100" for smoke tests

echo
echo "==== Method 1: prepare ===="
conda run -n vg-spv --no-capture-output python -u -m scripts.generate_method1_traces prepare \
    --split        "${SPLIT}" \
    --dataset-dir  "${DATASET_DIR}" \
    --data-root    "${DATASET_DIR}" \
    --metadata     "${DATASET_DIR}/${SPLIT}_metadata.jsonl" \
    --max-lines    45000 \
    --max-bytes    188743680 \
    ${M1_LIMIT}

# ---- Method 1 submit --------------------------------------------------------
# Upload each prepared chunk and create one batch job per chunk against
# /v1/responses. State is persisted to traces/_batches/${SPLIT}/state.json.
M1_RESUBMIT=""    # set to "--resubmit" to resubmit chunks already in state.json

echo
echo "==== Method 1: submit ===="
conda run -n vg-spv --no-capture-output python -u -m scripts.generate_method1_traces submit \
    --split        "${SPLIT}" \
    --dataset-dir  "${DATASET_DIR}" \
    --data-root    "${DATASET_DIR}" \
    --metadata     "${DATASET_DIR}/${SPLIT}_metadata.jsonl" \
    ${M1_RESUBMIT}

# ---- Method 1 wait ----------------------------------------------------------
# Poll batch status until terminal (completed/failed/expired/cancelled) and
# download outputs/errors for each chunk. May take minutes to hours depending
# on the OpenAI batch queue depth at submission time.
echo
echo "==== Method 1: wait ===="
conda run -n vg-spv --no-capture-output python -u -m scripts.generate_method1_traces wait \
    --split         "${SPLIT}" \
    --dataset-dir   "${DATASET_DIR}" \
    --data-root     "${DATASET_DIR}" \
    --metadata      "${DATASET_DIR}/${SPLIT}_metadata.jsonl" \
    --poll-interval 60

# ---- Method 1 collect -------------------------------------------------------
# Parse all output_*.jsonl files, drop out-of-cohort rows, write the
# trainer-ready CSV: traces/${SPLIT}_method1.csv
echo
echo "==== Method 1: collect ===="
conda run -n vg-spv --no-capture-output python -u -m scripts.generate_method1_traces collect \
    --split        "${SPLIT}" \
    --dataset-dir  "${DATASET_DIR}" \
    --data-root    "${DATASET_DIR}" \
    --metadata     "${DATASET_DIR}/${SPLIT}_metadata.jsonl"

# ============================================================================
# Method 2 -- Grounding DINO + per-phrase NMS
# ============================================================================

# Re-detect for each row in the Method 1 CSV, run NMS, and write
# traces/${SPLIT}_method2.csv.
#
# Notes:
#   --config       defaults to None (auto-inferred from checkpoint stem).
#                  Override only if you have a custom config .py file.
#   --device       defaults to None (auto-detects cuda if available).
#                  Hard-coded to "cuda" below since this is a GPU sbatch job.
#   --overwrite    by default rows already in {split}_method2.csv are skipped
#                  (cheap resume after partial runs). Set M2_OVERWRITE to
#                  "--overwrite" to redo every row from scratch.
M2_CONFIG=""           # set to '--config path/to/Custom.py' to override
M2_OVERWRITE=""        # set to "--overwrite" to re-process all rows

echo
echo "==== Method 2: run ===="
conda run -n vg-spv --no-capture-output python -u -m scripts.generate_method2_traces \
    --split           "${SPLIT}" \
    --checkpoint      "${DINO_CHECKPOINT}" \
    --dataset-dir     "${DATASET_DIR}" \
    --device          "cuda" \
    --box-threshold   0.3 \
    --text-threshold  0.25 \
    --nms-iou         0.5 \
    --max-per-phrase  5 \
    ${M2_CONFIG} \
    ${M2_OVERWRITE}

# ============================================================================
# Done
# ============================================================================
echo
echo "==== DONE ===="
echo "Method 1 CSV : ${DATASET_DIR}/traces/${SPLIT}_method1.csv"
echo "Method 2 CSV : ${DATASET_DIR}/traces/${SPLIT}_method2.csv"
echo "Batch artifacts (input/output/state) : ${DATASET_DIR}/traces/_batches/${SPLIT}/"
