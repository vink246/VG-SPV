#!/bin/bash
#SBATCH --job-name=viscra
#SBATCH -c 10
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:h100
#SBATCH --mem=64G
#SBATCH --output=viscra_%j.out

echo "hades"
module load anaconda3/2023.03
cd evaluation
echo "Running baseline attack using inference_vllm.py"
conda run -n MRA --no-capture-output python inference_vllm.py --attack_type baseline --model_name Llama-3.2-11b-V --dataset HADES --input_dir ../datasets/Hades --output_dir ./outputs_baseline

cd ..
echo "Running QwenMask attack using qwenmask.py"
conda run -n MRA --no-capture-output python qwenmask.py --model_name Llama-3.2-11b-V --dataset HADES  --input_dir datasets/Hades

cd evaluation
echo "Running attention-based attack using inference_vllm.py"
conda run -n MRA --no-capture-output python inference_vllm.py --attack_type attention  --model_name Llama-3.2-11b-V   --dataset HADES   --input_dir ../datasets/Hades   --output_dir ./outputs_attention

