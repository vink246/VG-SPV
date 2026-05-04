"""
Load VG-SPV DPO hyperparameters from YAML (``configs/dpo.yaml``).

All tunables live in the YAML file; ``train/run_dpo.py`` merges CLI overrides on top.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DPOTrainConfig:
    """Fields mirror ``configs/dpo.yaml`` (single flat mapping)."""

    model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf"
    data_path: str | None = None
    # Optional CSV / HF id for evaluation (e.g. test split). When set, Trainer runs eval each epoch.
    eval_data_path: str | None = None
    output_dir: str = "outputs/dpo"
    prompt_instruction: str | None = None

    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-5
    # Cap full sequence length for DPO (chosen/rejected completions); raise if long XML traces truncate.
    max_length: int = 8192
    max_prompt_length: int = 2048
    beta: float = 0.1
    bf16: bool = True
    ref_8bit: bool = False
    logging_steps: int = 10
    save_steps: int = 200
    save_total_limit: int = 3

    use_vgfdpo: bool = True
    use_vdpo: bool = False
    alpha_vdpo: float = 0.1
    vdpo_margin_m: float = 0.0
    alpha_format: float = 13.0
    grounding_mode: str = "semantic"
    alpha_sem: float = 1.0
    alpha_iou: float = 1.0
    s_fallback_value: float = 1.0
    strict_scaler_inputs: bool = False

    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_adapter_path: str | None = None
    merge_lora_adapter: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DPOTrainConfig:
        known = {f.name for f in fields(cls)}
        extra = set(d) - known
        if extra:
            raise ValueError(f"Unknown keys in DPO YAML: {sorted(extra)}")
        missing = known - set(d)
        if missing:
            raise ValueError(f"Missing keys in DPO YAML: {sorted(missing)}")
        return cls(**{k: d[k] for k in known})


def default_dpo_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "configs" / "dpo.yaml"


def load_dpo_train_config(path: str | Path) -> DPOTrainConfig:
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"DPO config must be a YAML mapping, got {type(raw)}")
    return DPOTrainConfig.from_dict(raw)


def merge_dpo_train_config(base: DPOTrainConfig, overrides: dict[str, Any]) -> DPOTrainConfig:
    d = asdict(base)
    for k, v in overrides.items():
        if v is None:
            continue
        if k not in d:
            raise KeyError(f"Unknown override key: {k}")
        d[k] = v
    return DPOTrainConfig.from_dict(d)


def dump_dpo_train_config_yaml(cfg: DPOTrainConfig) -> str:
    return yaml.safe_dump(asdict(cfg), default_flow_style=False, sort_keys=False)
