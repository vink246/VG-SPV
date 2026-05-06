#!/bin/bash
#SBATCH --job-name=viscra
#SBATCH -c 8
#SBATCH --time=13:00:00
#SBATCH --gres=gpu:h200
#SBATCH --mem=32G
#SBATCH --output=logs/final_viscra_%j.out
#SBATCH --error=logs/final_viscra_%j.err

module load anaconda3/2023.03
cd evaluation

MODELS=(
    "Llama-3.2-11b-V_SFT"
    "Llama-3.2-11b-V_SFT_Spatial_VGFDPO_Method2"
    "Llama-3.2-11b-V_SFT_DPO_Method2"
    "Llama-3.2-11b-V_SFT_Semantic_VGFDPO_Method2"
    "Llama-3.2-11b-V_Base_Semantic_VGFDPO_Method1"

    # "Llama-3.2-11b-V_Base_Spatial_VGFDPO_Method2"
    # "Llama-3.2-11b-V_SFT_DPO_Method2_Cont"
    # "Llama-3.2-11b-V_SFT_Spatial_VGFDPO_Method2_Cont"
    # "Llama-3.2-11b-V_Base_Semantic_VGFDPO_Method1_Cont"
    # "Llama-3.2-11b-V"
)

for MODEL_NAME in "${MODELS[@]}"; do
    echo "========================================================"
    echo "Evaluating Model: ${MODEL_NAME}"
    echo "========================================================"

    # echo "Running baseline attack using inference_vllm.py"
    # conda run -n MRA --no-capture-output python inference_vllm.py \
    #     --attack_type baseline \
    #     --model_name "${MODEL_NAME}" \
    #     --dataset mm-safebench \
    #     --input_dir ../datasets/mm-safebench \
    #     --output_dir "./outputs_baseline/${MODEL_NAME}"

    # echo "MM SafetyBench: Running attention-based attack"
    # conda run -n MRA --no-capture-output python inference_vllm.py \
    #     --attack_type attention \
    #     --model_name "${MODEL_NAME}" \
    #     --dataset mm-safebench \
    #     --input_dir ../datasets/mm-safebench \
    #     --output_dir "./outputs_attention/${MODEL_NAME}"

    echo "MMSafe: Running attention-based attack"
    conda run -n vg-spv4 --no-capture-output python inference_vllm.py \
        --attack_type attention \
        --model_name "${MODEL_NAME}" \
        --dataset mm-safebench \
        --input_dir ../datasets/mm-safebench \
        --output_dir "./outputs_attention/${MODEL_NAME}"

done