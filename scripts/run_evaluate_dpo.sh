#!/bin/bash
#SBATCH --job-name=dpo_metrics
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/dpo_metrics_%j.out
#SBATCH --error=logs/dpo_metrics_%j.err

export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=0

echo "Starting DPO-Intrinsic Metrics Evaluation at $(date)"
echo "Running on node: $SLURM_NODELIST"

conda run -n vg-spv --no-capture-output python -u -m scripts.evaluate_dpo_metrics

echo "Evaluation finished at $(date)"