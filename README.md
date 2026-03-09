# Visually-Grounded Safety Process Verification (VG-SPV)

As Multimodal Large Language Models (MLLMs) become more capable, they remain vulnerable to sophisticated multimodal jailbreaks (e.g. [VisCRA](https://arxiv.org/abs/2410.12963)) that exploit the "modality gap" between textual safety priors and visual evidence.

**VG-SPV** is a defense framework that forces MLLMs to explicitly ground their safety critiques in verifiable visual evidence. By combining Introspective Reasoning (Chain-of-Thought) with automated **Spatial Process Rewards** via [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO), VG-SPV ensures that if a model claims a safety threat exists, it must accurately provide bounding box coordinates for it. The model is optimized using **fine-grained Direct Preference Optimization (fDPO)** and **Vision-Guided Loss (V-DPO)** to effectively neutralize visual context attacks without increasing benign over-refusals.

Built for a research project in CS 8803 (LLMs) at Georgia Tech.

---

## Repository structure

```
VG-SPV/
├── data/                  # Scripts to download/process COCO, VLGuard, VisCRA
│   └── generate_traces.py # GPT-4o API script for synthesizing data
├── eval/                  # Scripts for running VisCRA ASR and RefCOCO benchmarks
├── models/                # VG-PRM reward logic and TinyLLaVA model wrappers
│   └── reward_dino.py     # Grounding DINO IoU-based reward computation
├── scripts/               # Bash scripts for launching training/eval runs
├── train/                 # Modified fDPO and V-DPO training loops
├── environment.yml       # Dependencies (torch, transformers, groundingdino, etc.)
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

### 3. Dependencies

The `environment.yml` includes:

- **Python 3.10**, PyTorch (CUDA 12.1), torchvision, torchaudio  
- **Transformers**, Accelerate, PEFT, TRL, Datasets  
- **Grounding DINO** (from source) for spatial reward computation  
- **Flash Attention** (optional, for Qwen-VL memory efficiency)  

Ensure you have a CUDA-capable GPU and enough VRAM for training/eval.

### 4. (Optional) API keys and data

- For `data/generate_traces.py`: set your **OpenAI API key** (GPT-4o) if synthesizing traces.  
- Download or configure paths for **COCO**, **VLGuard**, and **VisCRA** datasets as required by the data scripts.

---

## Usage (high level)

1. **Data**: Run `data/generate_traces.py` to download/process datasets and synthesize traces.  
2. **Training**: Use scripts in `scripts/` to launch fDPO/V-DPO training (implemented in `train/`).  
3. **Reward**: `models/reward_dino.py` provides the Grounding DINO IoU-based reward for VG-PRM.  
4. **Eval**: Run VisCRA ASR and RefCOCO benchmarks from `eval/`.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
