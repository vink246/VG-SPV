import torch
from transformers import AutoProcessor, MllamaForConditionalGeneration
from peft import PeftModel

# 1. Define your paths
base_model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
sft_lora_dir = "/home/hice1/psomu3/scratch/VG-SPV/outputs/sft_llama32_11b_vgfdpo_method2"
output_dir = "models/Llama-3.2-11B-SFT-Merged"

print("Loading base model...")
base_model = MllamaForConditionalGeneration.from_pretrained(
    base_model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

print("Applying SFT LoRA adapter...")
peft_model = PeftModel.from_pretrained(base_model, sft_lora_dir)

print("Merging weights (this takes a minute)...")
fused_model = peft_model.merge_and_unload()

print(f"Saving new fused model to {output_dir}...")
fused_model.save_pretrained(output_dir)

# Save the tokenizer/processor too!
processor = AutoProcessor.from_pretrained(base_model_id)
processor.save_pretrained(output_dir)

print("Done! You can now use this as your base model for DPO.")