#!/bin/bash
#SBATCH --job-name=viscra
#SBATCH -c 3
#SBATCH --time=3:00:00
#SBATCH --mem=10G
#SBATCH --output=logs/oai_%j.out
#SBATCH --error=logs/oai_%j.err

module load anaconda3/2023.03
cd evaluation

MODELS=(
    "Llama-3.2-11b-V_SFT"
    "Llama-3.2-11b-V_SFT_Spatial_VGFDPO_Method2"
    # "Llama-3.2-11b-V_SFT_DPO_Method2"
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

    echo "HADES: Running attention-based evaluation"
    conda run -n vg-spv4 --no-capture-output python evaluate_openai_sync.py \
        --input_file "./outputs_attention/${MODEL_NAME}/HADES/${MODEL_NAME}.jsonl" \
        --output_dir "./outputs_attention/${MODEL_NAME}/HADES"
    echo "HADES: Running baseline-based evaluation"
    conda run -n vg-spv4 --no-capture-output python evaluate_openai_sync.py \
        --input_file "./outputs_baseline/${MODEL_NAME}/HADES/${MODEL_NAME}.jsonl" \
        --output_dir "./outputs_baseline/${MODEL_NAME}/HADES"
done

echo "Evaluating Model: Llama-3.2-11b-V"
echo "HADES: Running attention-based evaluation"
conda run -n vg-spv4 --no-capture-output python evaluate_openai_sync.py \
    --input_file "/home/hice1/psomu3/scratch/VG-SPV/VisCRA/evaluation/outputs_baseline/HADES/Llama-3.2-11b-V.jsonl" \
    --output_dir "/home/hice1/psomu3/scratch/VG-SPV/VisCRA/evaluation/outputs_baseline/HADES/Llama-3.2-11b-V_openai"
echo "HADES: Running baseline-based evaluation"
conda run -n vg-spv4 --no-capture-output python evaluate_openai_sync.py \
    --input_file "/home/hice1/psomu3/scratch/VG-SPV/VisCRA/evaluation/outputs/HADES/Llama-3.2-11b-V.jsonl" \
    --output_dir "/home/hice1/psomu3/scratch/VG-SPV/VisCRA/evaluation/outputs/HADES/Llama-3.2-11b-V_openai"


    
