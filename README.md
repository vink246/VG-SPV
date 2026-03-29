# Visually-Grounded Safety Process Verification (VG-SPV)

As Multimodal Large Language Models (MLLMs) become more capable, they remain vulnerable to sophisticated multimodal jailbreaks (e.g. [VisCRA](https://arxiv.org/abs/2410.12963)) that exploit the "modality gap" between textual safety priors and visual evidence.

**VG-SPV** is a defense framework that forces MLLMs to explicitly ground their safety critiques in verifiable visual evidence. By combining Introspective Reasoning (Chain-of-Thought) with automated **Spatial Process Rewards** via [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO), VG-SPV ensures that if a model claims a safety threat exists, it must accurately provide bounding box coordinates for it. The model is optimized using **VG-fDPO (Visually-Grounded Fine-Grained DPO)** to effectively neutralize visual context attacks without increasing benign over-refusals.

Built for a research project in CS 8803 (LLMs) at Georgia Tech.

---

## Repository structure

```
VG-SPV/
├── data/                          # Evaluation and training datasets (e.g. HADES, MM Safety Bench, COCO, VLGuard, VisCRA); download/process scripts; sample CSV
│   ├── generate_traces.py         # GPT-4o API script for synthesizing data
│   └── sample_dpo_data.csv        # Example CSV (image, perturbed image, chosen/rejected reasoning trace)
├── weights/                       # Pretrained Grounding DINO checkpoints (see step 5 under Setup)
├── outputs/                       # Experiment outputs, visualizations, metrics, and other saved results
├── eval/                          # Scripts for running VisCRA ASR and RefCOCO benchmarks
├── inference/                     # Scripts to query the model (no training)
│   ├── run_inference.py           # VL image + text inference (use --model to choose model)
│   ├── run_grounding_dino.py      # Grounding DINO inference for box supervision / GT generation
│   └── utils.py                   # Legacy inference helpers (prefer `vlm.run_vl_inference` + `LoadedVLM`)
├── vlm/                           # VLM backends (Qwen-VL, LLaVA, TinyLLaVA): load_vlm, run_vl_inference, LoadedVLM
├── models/                        # VG-PRM reward logic and TinyLLaVA model wrappers
│   └── reward_dino.py             # Grounding DINO IoU-based reward computation
├── scripts/                       # Bash scripts for launching training/eval runs
│   ├── run_dpo_train.sh           # Launch DPO training
│   ├── install_grounding_dino.sh  # Install Grounding DINO after env create
│   └── install_flash_attn.sh      # Install flash-attn after env create (optional)
├── train/                         # DPO pipeline (TRL DPOTrainer) and VG-fDPO loss stub
│   ├── dpo_trainer.py             # VGSPVTrainer (override compute_loss for VG-fDPO)
│   ├── dataset_adapter.py         # DPO dataset contract and CSV loader (image, perturbed_image, chosen/rejected reasoning traces)
│   └── run_dpo.py                 # DPO training entrypoint
├── utils.py                       # Re-exports vlm + build_messages (chat JSON format)
├── environment.yml                # Dependencies (torch, transformers, groundingdino, etc.)
├── README.md                      # This file
└── LICENSE                        # Apache 2.0
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/vink246/VG-SPV.git
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

### 3. Install packages that need the env’s PyTorch at build time

Grounding DINO (and optionally flash-attn) must be installed **after** the env is created so their builds see the conda-installed PyTorch (pip’s build isolation would otherwise fail).

**Grounding DINO** (required for VG-PRM reward):

```bash
pip install --no-build-isolation 'git+https://github.com/IDEA-Research/GroundingDINO.git'
# or:
bash scripts/install_grounding_dino.sh
bash scripts/install_grounding_dino.sh ~/scratch/envs/vg-spv   # prefix env
```

After installation, the `groundingdino` package ships with model configs. When you run
`inference/run_grounding_dino.py`, the script **automatically selects the matching config**
from the checkpoint filename (e.g. `groundingdino_swint_ogc.pth` → `GroundingDINO_SwinT_OGC.py`).
You can override with `--config` if needed.

**Flash Attention** (optional, for Qwen-VL memory efficiency; slow to build):

```bash
pip install --no-build-isolation flash-attn
# or:
bash scripts/install_flash_attn.sh
bash scripts/install_flash_attn.sh ~/scratch/envs/vg-spv       # prefix env
```

### 4. Dependencies

The `environment.yml` includes:

- **Python 3.10**, PyTorch (CUDA 12.1), torchvision, torchaudio  
- **Transformers**, Accelerate, PEFT, TRL, Datasets  
- **VL models**: Inference and training are model-agnostic. Supported families include **Qwen3-VL** (e.g. 2B, 4B, 8B), **LLaVA**, and **TinyLLaVA**; add more in [`vlm/backends/`](vlm/backends/) and [`vlm/registry.py`](vlm/registry.py) (pattern → family → backend).  
- **Grounding DINO** (from source) for spatial reward computation — install after env create: `bash scripts/install_grounding_dino.sh`.  
- **Flash Attention** (optional, for Qwen-VL memory efficiency) — install after env create: `bash scripts/install_flash_attn.sh`.  

Ensure you have a CUDA-capable GPU and enough VRAM for training/eval.

### 5. Grounding DINO weights and configs

Once `groundingdino` is installed (see above), you only need a **pretrained checkpoint** to run
`inference/run_grounding_dino.py`. The script **automatically picks the right config** from the
installed package based on the checkpoint filename (e.g. `groundingdino_swint_ogc.pth` →
`GroundingDINO_SwinT_OGC.py`). Use `--config` only if you need to override.

- **Download pretrained weights**  
  GroundingDINO publishes weights on their [GitHub](https://github.com/IDEA-Research/GroundingDINO) (and mirrors). Typical checkpoints:

  - `groundingdino_swint_ogc.pth` (Swin-T backbone, OGC)
  - `groundingdino_swinb_cogcoor.pth` (Swin-B backbone, CogCoor)

  Store checkpoints under the repo’s `weights/` directory (create it if needed). For the default Swin-T OGC weights:

  ```bash
  mkdir weights
  cd weights
  wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
  cd ..
  ```

  On Windows without `wget`, download the same URL in a browser or run from `weights/`: `curl -L -o groundingdino_swint_ogc.pth https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth`.

- **Run Grounding DINO inference for GT generation**:

  ```bash
  python inference/run_grounding_dino.py \
    --checkpoint weights/groundingdino_swint_ogc.pth \
    --image path/to/image.jpg \
    --text-prompt "dog . person ." \
    --output-json outputs/dino_boxes.json \
    --output-viz outputs/dino_boxes_viz.jpg
  ```

  The script generates JSON box annotations (and an optional visualization) for use with
  `models/reward_dino.py` or other data-processing pipelines.

  **Transformers compatibility:** Upstream GroundingDINO targets an older `transformers` API.
  `inference/run_grounding_dino.py` applies small patches before loading: it restores
  `BertModel.get_head_mask` if missing, and fixes `get_extended_attention_mask` when the third
  argument is a `torch.device` (GroundingDINO) but newer `transformers` expect a `dtype` there.
  If you still see BERT-related errors, try pinning `transformers` to a version known to work with
  GroundingDINO (e.g. `4.41.x`).

### 6. (Optional) API keys and data

- For `data/generate_traces.py`: set your **OpenAI API key** (GPT-4o) if synthesizing traces.  
- Download or configure paths for datasets under `data/` as required by the scripts (e.g. **HADES**, **MM Safety Bench**, **COCO**, **VLGuard**, **VisCRA**).

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

1. **Inference**: Run `inference/run_inference.py` with `--model` to query any supported VL model (e.g. TinyLLaVA, LLaVA). Example: `python inference/run_inference.py --model tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B --image path/to/img.png --prompt "Describe this image"`. If you see **"Disk quota exceeded"**, the Hugging Face model cache is on a full filesystem: set the cache to a directory with space (e.g. scratch) with `export HF_HOME=~/scratch/.cache/huggingface` before running, or use `--cache_dir ~/scratch/.cache/huggingface/hub`. The script also auto-uses `$SCRATCH` for the cache when set.
2. **Data**: Run `data/generate_traces.py` to download/process datasets and synthesize traces. Output should match the [dataset format](#dataset-format) (CSV: image, perturbed image, chosen reasoning trace, rejected reasoning trace).
3. **Training**: Use scripts in `scripts/` to launch VG-fDPO training (e.g. `bash scripts/run_dpo_train.sh`). Use `--model_name` to pick any supported VL model. The `train/` directory contains the DPO pipeline (TRL DPOTrainer) with a custom trainer stub for VG-fDPO loss.
4. **Reward**: `models/reward_dino.py` provides the Grounding DINO IoU-based reward for VG-PRM. For ground-truth box generation, run `inference/run_grounding_dino.py` with `--checkpoint` (config is auto-selected from the checkpoint name).
5. **Eval**: Run VisCRA ASR and RefCOCO benchmarks from `eval/`.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
