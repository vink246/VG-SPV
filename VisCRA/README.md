# VisCRA: A Visual Chain Reasoning Attack for Jailbreaking Multimodal Large Language Models

## Paper link
The paper titled [VisCRA: A Visual Chain Reasoning Attack for Jailbreaking Multimodal Large Language Models](https://arxiv.org/abs/2505.19684) has been accepted for **Oral Presentation** at **EMNLP 2025**.


## Environment Setup
```bash
conda env create -f environment.yml
```


## Directory & Key Scripts
- Mask Generation: `qwenmask.py`  
- Inference & Evaluation Entrypoint: `evaluation/inference_vllm.py`  
- Llama Guard Evaluation: `evaluation/llama_guard.py`  

## Dataset Preparation
- **HADES**: Default input directory is `../datasets/Hades`  
- **mm-safebench**: Provide the path to your local copy, e.g., `../datasets/mm-safebench`  

## Model Preparation
Place the required pre-trained models into the `../pre_train_models/` directory:  
- Qwen2.5-VL-7B: `../pre_train_models/qwen2_5-vl-7b`  

For other models, please refer to the `model_path` definitions in `evaluation/inference_vllm.py`.  

---

## Usage

### 1) Running the Baseline (without VisCRA attack)
Navigate to the `evaluation` directory and run the inference script with `--attack_type baseline`:
```bash
cd evaluation
python inference_vllm.py \
  --attack_type baseline \
  --model_name qwen2_5_vl \
  --dataset HADES \
  --input_dir ../datasets/Hades \
  --output_dir ./outputs
```

**Notes**:  
- The value for `--model_name` can be found in the `choices` list within the script; it maps to the actual model weights path.  
- The output file will be saved to `./outputs/<DATASET>/<MODEL_NAME>.jsonl`.  

---

### 2) Running with the VisCRA Attack
VisCRA is a two-step process: first, generate masked data, and then perform inference on the masked images.  

#### Step A: Generate Attention Masks
```bash
python qwenmask.py \
  --model_name qwen2_5-vl \
  --dataset HADES \
  --input_dir ../datasets/Hades
```

**Outputs**:  
- Masked images are saved to: `./attention_mask_12(green)/<DATASET>/.../mask_block{0,1,2}.png`  
- The current mask block size is defined by the constant `ATTENTION_BLOCK_SIZE=12` (this can be adjusted at the top of `qwenmask.py`).  

#### Step B: Perform Inference with Masked Images
```bash
cd evaluation
python inference_vllm.py \
  --attack_type attention \
  --model_name qwen2_5_vl \
  --dataset HADES \
  --input_dir ../datasets/Hades \
  --output_dir ./outputs
```

**Notes**:  
- The script automatically locates the corresponding mask directory based on the original image’s path (e.g., `./attention_mask_12(green)/...`) and randomly selects one of the masks (`mask_block0-2.png`) for inference.  

---

## Evaluation (with Llama Guard)
Pass the JSONL output from the previous inference step to `llama_guard.py`:
```bash
cd evaluation
python llama_guard.py \
  --input_jsonl ./outputs/<DATASET>/<MODEL_NAME>.jsonl \
  --output_path ./outputs/<DATASET>/<MODEL_NAME>_guarded.jsonl
```

**Note**: Please refer to the `llama_guard.py` script for the exact argument names.  
