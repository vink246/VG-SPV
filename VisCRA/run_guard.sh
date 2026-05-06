#!/bin/bash
#SBATCH --job-name=viscra
#SBATCH -c 10
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:h100
#SBATCH --mem=64G
#SBATCH --output=viscra_%j.out

echo "running guard"
module load anaconda3/2023.03
cd evaluation
conda run -n MRA --no-capture-output python llama_guard.py \
  --input_jsonl ./outputs_baseline/mm-safebench/Llama-3.2-11b-V.jsonl \
  --output_path ./outputs_baseline/mm-safebench/Llama-3.2-11b-V_guarded.jsonl