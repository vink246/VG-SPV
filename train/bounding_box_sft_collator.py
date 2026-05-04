"""
Batch multimodal conversations for bounding-box SFT: mask loss on user/prompt tokens only.

Supports HF grounding rows and optional VG-SPV CSV rows (image + instruction + chosen trace).
"""

from __future__ import annotations

from typing import Any

from train.bounding_box_sft_dataset import hf_row_to_bbox_sft_sample, vgspv_csv_row_to_bbox_sft_sample


def _split_user_assistant(messages: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    if len(messages) < 2:
        raise ValueError("Expected user + assistant messages")
    return [messages[0]], messages


class BoundingBoxSFTCollator:
    def __init__(
        self,
        processor: Any,
        hf_train: Any,
        family: str,
        *,
        hf_eval: Any | None = None,
        csv_rows: list[dict[str, Any]] | None = None,
        csv_eval_rows: list[dict[str, Any]] | None = None,
        vgspv_prompt_instruction: str = "",
    ):
        self.processor = processor
        self.hf_train = hf_train
        self.hf_eval = hf_eval
        self.family = family
        self.csv_rows = csv_rows
        self.csv_eval_rows = csv_eval_rows
        self.vgspv_prompt_instruction = vgspv_prompt_instruction
        tok = getattr(processor, "tokenizer", processor)
        tok.padding_side = "right"

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        prompt_lens: list[int] = []
        full_texts: list[str] = []
        images: list[Any] = []

        for b in batch:
            pool = b.get("pool", "train")
            src = b.get("source", "hf")
            idx = int(b["idx"])
            if pool == "eval":
                hf_src = self.hf_eval
                csv_src = self.csv_eval_rows
            else:
                hf_src = self.hf_train
                csv_src = self.csv_rows
            if src == "vgspv_csv":
                if not csv_src:
                    raise ValueError("Collator received vgspv_csv batch but CSV rows for this pool are empty")
                sample = vgspv_csv_row_to_bbox_sft_sample(
                    csv_src[idx],
                    self.vgspv_prompt_instruction,
                )
            else:
                if hf_src is None:
                    raise ValueError("Collator received hf batch but hf dataset for this pool is None")
                sample = hf_row_to_bbox_sft_sample(hf_src[idx])
            messages = sample.messages
            user_only, full_msgs = _split_user_assistant(messages)
            text_prompt = self.processor.apply_chat_template(
                user_only,
                tokenize=False,
                add_generation_prompt=True,
            )
            text_full = self.processor.apply_chat_template(
                full_msgs,
                tokenize=False,
                add_generation_prompt=False,
            )
            img = None
            for part in messages[0]["content"]:
                if isinstance(part, dict) and part.get("type") == "image":
                    img = part.get("image")
                    break
            if img is None:
                raise ValueError("Missing image in user message")
            images.append(img)
            full_texts.append(text_full)

            enc_p = self.processor(
                images=[img],
                text=[text_prompt],
                padding=False,
                return_tensors="pt",
            )
            enc_f = self.processor(
                images=[img],
                text=[text_full],
                padding=False,
                return_tensors="pt",
            )
            fp = enc_p["input_ids"][0].tolist()
            ff = enc_f["input_ids"][0].tolist()
            pl = 0
            for tok_a, tok_b in zip(fp, ff):
                if tok_a != tok_b:
                    break
                pl += 1
            prompt_lens.append(pl)

        enc = self.processor(
            images=images,
            text=full_texts,
            padding=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]
        labels = input_ids.clone()
        for i, pl in enumerate(prompt_lens):
            pl = min(pl, labels.shape[1])
            labels[i, :pl] = -100
        labels[input_ids == self.processor.tokenizer.pad_token_id] = -100
        out = dict(enc)
        out["labels"] = labels
        return out
