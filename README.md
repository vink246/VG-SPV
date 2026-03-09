# Visually-Grounded Safety Process Verification (VG-SPV)

As Multimodal Large Language Models (MLLMs) become more capable, they remain vulnerable to sophisticated multimodal jailbreaks (e.g. [VisCRA](https://arxiv.org/abs/2410.12963)) that exploit the "modality gap" between textual safety priors and visual evidence.

**VG-SPV** is a defense framework that forces MLLMs to explicitly ground their safety critiques in verifiable visual evidence. By combining Introspective Reasoning (Chain-of-Thought) with automated **Spatial Process Rewards** via [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO), VG-SPV ensures that if a model claims a safety threat exists, it must accurately provide bounding box coordinates for it. The model is optimized using **VG-fDPO (Visually-Grounded Fine-Grained DPO)** to effectively neutralize visual context attacks without increasing benign over-refusals.

Built for a research project in CS 8803 (LLMs) at Georgia Tech.

---

## Repository structure

```
VG-SPV/
├── data/                  # Scripts to download/process COCO, VLGuard, VisCRA; sample CSV
│   ├── generate_traces.py # GPT-4o API script for synthesizing data
│   └── sample_dpo_data.csv # Example CSV (image, perturbed image, chosen/rejected reasoning trace)
├── eval/                  # Scripts for running VisCRA ASR and RefCOCO benchmarks
├── inference/             # Scripts to query the model (no training)
│   ├── run_inference.py   # VL image + text inference (use --model to choose model)
│   └── utils.py           # Inference-only helpers (e.g. run_vl_inference)
├── models/                # VG-PRM reward logic and TinyLLaVA model wrappers
│   └── reward_dino.py     # Grounding DINO IoU-based reward computation
├── scripts/               # Bash scripts for launching training/eval runs
│   ├── run_dpo_train.sh   # Launch DPO training
│   └── install_grounding_dino.sh # Install Grounding DINO after env create
├── train/                 # DPO pipeline (TRL DPOTrainer) and VG-fDPO loss stub
│   ├── dpo_trainer.py     # VGSPVTrainer (override compute_loss for VG-fDPO)
│   ├── dataset_adapter.py # DPO dataset contract and CSV loader (image, perturbed_image, chosen/rejected reasoning traces)
│   └── run_dpo.py         # DPO training entrypoint
├── utils.py               # Shared, model-agnostic helpers (VL registry, load_vl_model_and_processor, build_messages, parse_dtype)
├── environment.yml        # Dependencies (torch, transformers, groundingdino, etc.)
├── README.md              # This file
└── LICENSE                # Apache 2.0
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/<your-org>/VG-SPV.git
cd VG-SPV
```

### 2. Create conda environment

```bash
conda env create -f environment.yml
conda activate vg-spv
```

If you use a custom prefix (e.g. on a cluster):

```bash
conda env create --prefix ~/scratch/envs/vg-spv -f environment.yml
conda activate ~/scratch/envs/vg-spv
```

### 3. Install Grounding DINO (required for VG-PRM reward)

Grounding DINO must be installed **after** the env is created so its build can see the conda-installed PyTorch (pip’s build isolation would otherwise fail):

```bash
pip install --no-build-isolation 'git+https://github.com/IDEA-Research/GroundingDINO.git'
```

Or use the helper script (with env activated, or pass the env path):

```bash
bash scripts/install_grounding_dino.sh
# or, for a prefix env:
bash scripts/install_grounding_dino.sh ~/scratch/envs/vg-spv
```

### 4. Dependencies

The `environment.yml` includes:

- **Python 3.10**, PyTorch (CUDA 12.1), torchvision, torchaudio  
- **Transformers**, Accelerate, PEFT, TRL, Datasets  
- **VL models**: Inference and training are model-agnostic. Supported families include **Qwen3-VL** (e.g. 2B, 4B, 8B) and **LLaVA**; add more in `utils.py` via the VL family registry.  
- **Grounding DINO** (from source) for spatial reward computation  
- **Flash Attention** (optional, for Qwen-VL memory efficiency)  

Ensure you have a CUDA-capable GPU and enough VRAM for training/eval.

### 5. (Optional) API keys and data

- For `data/generate_traces.py`: set your **OpenAI API key** (GPT-4o) if synthesizing traces.  
- Download or configure paths for **COCO**, **VLGuard**, and **VisCRA** datasets as required by the data scripts.

---

## Dataset format

Training data is stored as **CSV** files with the following columns (headers may use spaces, e.g. `perturbed image`):

| Column | Description |
|--------|-------------|
| **image** | Path to the input image |
| **perturbed image** | Path to the perturbed/jailbreak image (for VG-fDPO / visual grounding; optional) |
| **chosen reasoning trace** | Preferred reasoning trace (DPO "chosen" response) |
| **rejected reasoning trace** | Dispreferred reasoning trace (DPO "rejected" response) |

- **Standard DPO** uses `image`, `chosen reasoning trace`, and `rejected reasoning trace` (and an optional text prompt).
- **VG-SPV / VG-fDPO** (visual grounding) additionally uses `perturbed image` for the VG-fDPO loss.
- Images are always stored as **paths** in the CSV; image files live alongside the CSV or at absolute paths.
- Load a CSV with `--data_path path/to/data.csv`. See `train/dataset_adapter.py` for the exact contract and optional `--prompt_instruction`. Example: `data/sample_dpo_data.csv`.

---

## Usage (high level)

1. **Inference**: Run `inference/run_inference.py` with `--model` to query any supported VL model (e.g. Qwen3-VL-2B/4B/8B, LLaVA). Example: `python inference/run_inference.py --model Qwen/Qwen3-VL-2B-Instruct --image path/to/img.png --prompt "Describe this image"`.
2. **Data**: Run `data/generate_traces.py` to download/process datasets and synthesize traces. Output should match the [dataset format](#dataset-format) (CSV: image, perturbed image, chosen reasoning trace, rejected reasoning trace).
3. **Training**: Use scripts in `scripts/` to launch VG-fDPO training (e.g. `bash scripts/run_dpo_train.sh`). Use `--model_name` to pick any supported VL model. The `train/` directory contains the DPO pipeline (TRL DPOTrainer) with a custom trainer stub for VG-fDPO loss.
4. **Reward**: `models/reward_dino.py` provides the Grounding DINO IoU-based reward for VG-PRM.
5. **Eval**: Run VisCRA ASR and RefCOCO benchmarks from `eval/`.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
