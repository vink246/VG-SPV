#!/bin/bash
#SBATCH --job-name=rej-traces
#SBATCH --output=logs/rej-traces-%j.out
#SBATCH --error=logs/rej-traces-%j.err
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:h200
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=%u@gatech.edu


set -euo pipefail

conda run -n vg-spv --no-capture-output python -u -m scripts.generate_rejected_traces --input "data/mm-safebench_1/extracted_data/traces/test_method1.csv" --output "data/mm-safebench_1/extracted_data/traces/test_method1_dpo.csv" --split test --method method1 --rejection-modes abliterated risk_perturb format_break --dataset-dir "data/mm-safebench_1/extracted_data" --device cuda --dtype bfloat16
conda run -n vg-spv --no-capture-output python -u -m scripts.generate_rejected_traces --input "data/mm-safebench_1/extracted_data/traces/train_method1.csv" --output "data/mm-safebench_1/extracted_data/traces/train_method1_dpo.csv" --split train --method method1 --rejection-modes abliterated risk_perturb format_break --dataset-dir "data/mm-safebench_1/extracted_data" --device cuda --dtype bfloat16
conda run -n vg-spv --no-capture-output python -u -m scripts.generate_rejected_traces --input "data/mm-safebench_1/extracted_data/traces/test_method2.csv" --output "data/mm-safebench_1/extracted_data/traces/test_method2_dpo.csv" --split test --method method2 --rejection-modes abliterated bbox_perturb risk_perturb format_break --dataset-dir "data/mm-safebench_1/extracted_data" --device cuda --dtype bfloat16
conda run -n vg-spv --no-capture-output python -u -m scripts.generate_rejected_traces --input "data/mm-safebench_1/extracted_data/traces/train_method2.csv" --output "data/mm-safebench_1/extracted_data/traces/train_method2_dpo.csv" --split train --method method2 --rejection-modes abliterated bbox_perturb risk_perturb format_break --dataset-dir "data/mm-safebench_1/extracted_data" --device cuda --dtype bfloat16
