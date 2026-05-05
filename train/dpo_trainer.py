"""
VG-SPV DPO Trainer implementing:
  - VG-fDPO masked loss (M_G / M_L / M_total) + format-fail fallback when ``use_vgfdpo=True`` (default)
  - Standard full-sequence DPO when ``use_vgfdpo=False``
  - V-DPO contrastive term when ``use_vdpo=True`` (opt-in; off by default; paper Eq. 7).
"""

from __future__ import annotations

import math
import re
import contextlib
from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

try:
    from trl import DPOTrainer
except ImportError: 
    try:
        from trl.trainer import DPOTrainer
    except ImportError:
        from trl.trainer.dpo_trainer import DPOTrainer

from train.tag_parsing import iou_xyxy_norm, parse_first_norm_box

_RISK_TAGS = ("risk_factors", "risk_factors_with_boxes")
_TRL_PREFERENCE_NON_MODEL_KEYS = frozenset({"completion_mask", "ref_chosen_logps", "ref_rejected_logps"})
_TAG_RE = re.compile(r"<[^>]+>")


def _mllama_cross_attn_pad_suffix(cam: torch.Tensor, pad_len: int) -> torch.Tensor:
    if pad_len <= 0:
        return cam
    b, ell = cam.shape[0], cam.shape[1]
    trail = tuple(range(2, cam.dim()))
    if trail:
        active = cam.sum(dim=trail) > 0
    else:
        active = cam > 0
    rev = active.flip(dims=[1])
    idx_from_end = rev.long().argmax(dim=1)
    last_active = ell - 1 - idx_from_end
    pick = torch.where(active.any(dim=1), last_active, torch.zeros(b, dtype=torch.long, device=cam.device))
    batch_ar = torch.arange(b, device=cam.device)
    template = cam[batch_ar, pick]
    return template.unsqueeze(1).expand(b, pad_len, *cam.shape[2:]).clone()


def _sync_seq_len_tensors_with_input_ids(kwargs: dict[str, Any], seq_len: int) -> None:
    for key in ("cross_attention_mask", "mm_token_type_ids"):
        t = kwargs.get(key)
        if t is None or not isinstance(t, torch.Tensor) or t.dim() < 2:
            continue
        cur = t.shape[1]
        if cur == seq_len:
            continue
        if cur > seq_len:
            kwargs[key] = t[:, :seq_len, ...].contiguous()
            continue
        pad_len = seq_len - cur
        if key == "cross_attention_mask":
            suffix = _mllama_cross_attn_pad_suffix(t, pad_len)
            kwargs[key] = torch.cat([t, suffix], dim=1).contiguous()
        else:
            kwargs[key] = F.pad(t, (0, pad_len))


def _counter_cosine(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0.0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(dot / (na * nb))


def _tokenize_terms(text: str) -> Counter[str]:
    plain = _TAG_RE.sub(" ", text.lower())
    parts = re.split(r"[^a-z0-9_]+", plain)
    tokens = [p for p in parts if p]
    return Counter(tokens)


@dataclass
class _SampleSpans:
    valid: torch.Tensor
    m_total: torch.Tensor
    m_g: torch.Tensor
    m_l: torch.Tensor
    parsed_ok: bool
    risk_text: str


class VGSPVTrainer(DPOTrainer):
    def __init__(
        self,
        *args: Any,
        use_vgfdpo: bool = True,
        use_vdpo: bool = False,
        alpha_vdpo: float = 0.1,
        vdpo_margin_m: float = 0.0,
        alpha_format: float = 13.0,
        grounding_mode: str = "semantic",
        alpha_sem: float = 1.0,
        alpha_iou: float = 1.0,
        s_fallback_value: float = 1.0,
        strict_scaler_inputs: bool = False,
        **kwargs: Any,
    ) -> None:
        self.use_vgfdpo = bool(use_vgfdpo)
        self.use_vdpo = bool(use_vdpo)
        self.alpha_vdpo = float(alpha_vdpo)
        self.vdpo_margin_m = float(vdpo_margin_m)
        self.alpha_format = float(alpha_format)
        self.grounding_mode = str(grounding_mode).strip().lower()
        self.alpha_sem = float(alpha_sem)
        self.alpha_iou = float(alpha_iou)
        self.s_fallback_value = float(s_fallback_value)
        self.strict_scaler_inputs = bool(strict_scaler_inputs)
        super().__init__(*args, **kwargs)
        if self.grounding_mode not in {"semantic", "spatial"}:
            raise ValueError(f"Unsupported grounding_mode: {self.grounding_mode}")

    def _get_tok(self):
        """Safely fetch the tokenizer depending on the TRL version / processing_class."""
        tok = getattr(self, "tokenizer", None)
        if tok is None:
            pc = getattr(self, "processing_class", None)
            tok = getattr(pc, "tokenizer", pc)
        return tok

    def _extract_model_kwargs(self, inputs: dict[str, Any], prefix: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        needle = f"{prefix}_"
        for k, v in inputs.items():
            if not k.startswith(needle):
                continue
            nk = k[len(needle) :]
            out[nk] = v
        return out

    def _shifted_token_logps(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        log_probs = F.log_softmax(shift_logits, dim=-1)
        gathered = torch.gather(log_probs, dim=-1, index=shift_labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        return torch.where(shift_labels == -100, torch.zeros_like(gathered), gathered)

    def _extract_block_mask(self, tokens: list[int], open_tag: str, close_tag: str) -> torch.Tensor:
        mask = torch.zeros(len(tokens), dtype=torch.bool)
        if not tokens:
            return mask
        
        tok = self._get_tok()
        full_text = tok.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        
        spans = []
        search_start = 0
        while True:
            s = full_text.find(open_tag, search_start)
            if s == -1: break
            e = full_text.find(close_tag, s)
            if e == -1: break
            e += len(close_tag)
            spans.append((s, e))
            search_start = e
            
        if not spans:
            return mask
            
        prefix_lens = [0] * (len(tokens) + 1)
        for i in range(1, len(tokens) + 1):
            prefix_lens[i] = len(tok.decode(tokens[:i], skip_special_tokens=False, clean_up_tokenization_spaces=False))
            
        for (s_char, e_char) in spans:
            left_t = 0
            right_t = len(tokens)
            for i in range(1, len(tokens) + 1):
                if prefix_lens[i] > s_char and left_t == 0:
                    left_t = i - 1
                if prefix_lens[i] >= e_char:
                    right_t = i
                    break
            mask[left_t:right_t] = True
            
        return mask

    def _extract_risk_mask(self, tokens: list[int]) -> torch.Tensor:
        best = torch.zeros(len(tokens), dtype=torch.bool)
        for tag in _RISK_TAGS:
            m = self._extract_block_mask(tokens, f"<{tag}>", f"</{tag}>")
            if int(m.sum().item()) > int(best.sum().item()):
                best = m
        return best

    def _decode_from_mask(self, tokens: list[int], mask: torch.Tensor) -> str:
        if len(tokens) == 0 or int(mask.sum().item()) == 0:
            return ""
        ids = [t for t, keep in zip(tokens, mask.tolist()) if keep]
        return self._get_tok().decode(ids, skip_special_tokens=False)

    def _sample_spans(self, labels_row: torch.Tensor) -> _SampleSpans:
        valid = labels_row[1:] != -100
        tokens = labels_row[1:][valid].tolist()
        m_total = torch.ones(len(tokens), dtype=torch.bool, device=labels_row.device)
        
        m_g = self._extract_risk_mask(tokens).to(labels_row.device)
        m_logic = self._extract_block_mask(tokens, "<logic>", "</logic>").to(labels_row.device)
        m_resp = self._extract_block_mask(tokens, "<response>", "</response>").to(labels_row.device)
        m_l = m_logic | m_resp
        
        parsed_ok = bool(int(m_g.sum().item()) > 0 and int(m_logic.sum().item()) > 0 and int(m_resp.sum().item()) > 0)
        risk_text = self._decode_from_mask(tokens, m_g)
        
        return _SampleSpans(valid=valid, m_total=m_total, m_g=m_g, m_l=m_l, parsed_ok=parsed_ok, risk_text=risk_text)

    def _sum_segment_logp(self, token_logp_row: torch.Tensor, spans: _SampleSpans, mask_name: str) -> torch.Tensor:
        masked = token_logp_row[spans.valid]
        seg_mask = getattr(spans, mask_name)
        if masked.numel() == 0 or seg_mask.numel() == 0:
            return token_logp_row.new_tensor(0.0)
        return masked[seg_mask].sum()

    def _semantic_s(self, risk_chosen: str, risk_rejected: str) -> float:
        sim = _counter_cosine(_tokenize_terms(risk_chosen), _tokenize_terms(risk_rejected))
        return self.alpha_sem * (1.0 - sim)

    def _spatial_s(self, risk_chosen: str, risk_rejected: str) -> float | None:
        bc = parse_first_norm_box(risk_chosen)
        br = parse_first_norm_box(risk_rejected)
        if bc is None or br is None:
            return None
        return self.alpha_iou * (1.0 - iou_xyxy_norm(bc, br))

    def _compute_s(self, risk_chosen: str, risk_rejected: str) -> tuple[float, bool]:
        if self.grounding_mode == "semantic":
            return self._semantic_s(risk_chosen, risk_rejected), False
        sval = self._spatial_s(risk_chosen, risk_rejected)
        if sval is not None:
            return sval, False
        if self.strict_scaler_inputs:
            raise ValueError("Missing/invalid box data for spatial scaler.")
        return self.s_fallback_value, True

    def _forward_preference_pair_logps_trl(
        self, model, inputs: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = inputs["input_ids"]
        completion_mask = inputs["completion_mask"]
        b2 = input_ids.size(0)
        if b2 % 2 != 0:
            raise ValueError(f"Expected even batch size for chosen+rejected stacking, got {b2}")
        half = b2 // 2

        kwargs = {k: v for k, v in inputs.items() if k not in _TRL_PREFERENCE_NON_MODEL_KEYS}
        attn = kwargs.get("attention_mask")

        seq_len = kwargs["input_ids"].shape[1]
        _sync_seq_len_tensors_with_input_ids(kwargs, seq_len)

        labels = input_ids.clone()
        labels[completion_mask == 0] = -100
        if attn is not None:
            labels[attn == 0] = -100

        outputs = model(**kwargs, use_cache=False)
        logits = outputs.logits

        lc, lr = logits[:half], logits[half:]
        lbl_c, lbl_r = labels[:half], labels[half:]
        return (
            self._shifted_token_logps(lc, lbl_c),
            lbl_c,
            self._shifted_token_logps(lr, lbl_r),
            lbl_r,
        )

    def _forward_token_logps(self, model, inputs: dict[str, Any], prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
        kwargs = self._extract_model_kwargs(inputs, prefix)
        labels = kwargs.get("labels", None)
        if labels is None:
            raise KeyError(f"Missing `{prefix}_labels` in batch; cannot compute VG-fDPO segments.")
        outputs = model(**{k: v for k, v in kwargs.items() if k != "labels"})
        token_logps = self._shifted_token_logps(outputs.logits, labels)
        return token_logps, labels

    def _get_ref_model(self, model):
        return self.ref_model if getattr(self, "ref_model", None) is not None else model

    def _vdpo_term(self, model, inputs: dict[str, Any], chosen_policy_logp_total: torch.Tensor) -> tuple[torch.Tensor, int]:
        missing = 0
        try:
            pert_logps, pert_labels = self._forward_token_logps(model, inputs, "chosen_perturbed")
        except Exception:
            return chosen_policy_logp_total.new_zeros(chosen_policy_logp_total.shape), chosen_policy_logp_total.numel()
        vdpo = []
        for i in range(chosen_policy_logp_total.size(0)):
            valid = pert_labels[i, 1:] != -100
            lp_ib = pert_logps[i][valid].sum()
            lp_ia = chosen_policy_logp_total[i]
            vdpo.append(torch.relu(lp_ib - lp_ia + self.vdpo_margin_m))
        if not vdpo:
            return chosen_policy_logp_total.new_zeros(chosen_policy_logp_total.shape), missing
        return torch.stack(vdpo), missing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        concat_batch = "completion_mask" in inputs and "input_ids" in inputs
        try:
            if concat_batch:
                chosen_pol_logps, chosen_labels, rejected_pol_logps, rejected_labels = self._forward_preference_pair_logps_trl(
                    model, inputs
                )
            else:
                chosen_pol_logps, chosen_labels = self._forward_token_logps(model, inputs, "chosen")
                rejected_pol_logps, rejected_labels = self._forward_token_logps(model, inputs, "rejected")
        except KeyError:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        except Exception:
            if concat_batch:
                raise
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        ref_model = self._get_ref_model(model)
        # If we didn't pass a separate ref model, we must disable the LoRA adapter on the policy model
        adapter_toggle = model.disable_adapter() if (self.ref_model is None and hasattr(model, "disable_adapter")) else contextlib.nullcontext()
        with torch.no_grad(), adapter_toggle:
            if concat_batch:
                chosen_ref_logps, _, rejected_ref_logps, _ = self._forward_preference_pair_logps_trl(ref_model, inputs)
            else:
                chosen_ref_logps, _ = self._forward_token_logps(ref_model, inputs, "chosen")
                rejected_ref_logps, _ = self._forward_token_logps(ref_model, inputs, "rejected")

        bs = chosen_labels.size(0)
        losses = []
        vg_losses = []
        fmt_losses = []
        s_values = []
        invalid_s_count = 0
        parse_ok_count = 0

        chosen_pol_total = []
        for i in range(bs):
            chosen_spans = self._sample_spans(chosen_labels[i])
            rejected_spans = self._sample_spans(rejected_labels[i])

            c_total_pol = self._sum_segment_logp(chosen_pol_logps[i], chosen_spans, "m_total")
            l_total_pol = self._sum_segment_logp(rejected_pol_logps[i], rejected_spans, "m_total")
            c_total_ref = self._sum_segment_logp(chosen_ref_logps[i], chosen_spans, "m_total")
            l_total_ref = self._sum_segment_logp(rejected_ref_logps[i], rejected_spans, "m_total")
            chosen_pol_total.append(c_total_pol)

            delta_total = self.beta * ((c_total_pol - c_total_ref) - (l_total_pol - l_total_ref))

            if not self.use_vgfdpo:
                losses.append(-F.logsigmoid(delta_total))
                continue

            if chosen_spans.parsed_ok and rejected_spans.parsed_ok:
                parse_ok_count += 1
                c_l_pol = self._sum_segment_logp(chosen_pol_logps[i], chosen_spans, "m_l")
                l_l_pol = self._sum_segment_logp(rejected_pol_logps[i], rejected_spans, "m_l")
                c_l_ref = self._sum_segment_logp(chosen_ref_logps[i], chosen_spans, "m_l")
                l_l_ref = self._sum_segment_logp(rejected_ref_logps[i], rejected_spans, "m_l")
                delta_l = self.beta * ((c_l_pol - c_l_ref) - (l_l_pol - l_l_ref))

                c_g_pol = self._sum_segment_logp(chosen_pol_logps[i], chosen_spans, "m_g")
                l_g_pol = self._sum_segment_logp(rejected_pol_logps[i], rejected_spans, "m_g")
                c_g_ref = self._sum_segment_logp(chosen_ref_logps[i], chosen_spans, "m_g")
                l_g_ref = self._sum_segment_logp(rejected_ref_logps[i], rejected_spans, "m_g")
                delta_g = self.beta * ((c_g_pol - c_g_ref) - (l_g_pol - l_g_ref))

                s_val, invalid_s = self._compute_s(chosen_spans.risk_text, rejected_spans.risk_text)
                invalid_s_count += int(invalid_s)
                s_values.append(float(s_val))
                loss_vg = -F.logsigmoid(delta_l + (delta_g * s_val))
                losses.append(loss_vg)
                vg_losses.append(loss_vg.detach())
            else:
                loss_fmt = -F.logsigmoid(self.alpha_format * delta_total)
                losses.append(loss_fmt)
                fmt_losses.append(loss_fmt.detach())

        if not losses:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        base_loss = torch.stack(losses).mean()
        chosen_pol_total_t = torch.stack(chosen_pol_total) if chosen_pol_total else base_loss.new_zeros((bs,))
        vdpo_loss = base_loss.new_tensor(0.0)
        vdpo_missing = 0
        if self.use_vdpo:
            vdpo_vec, vdpo_missing = self._vdpo_term(model, inputs, chosen_pol_total_t)
            vdpo_loss = vdpo_vec.mean() if vdpo_vec.numel() > 0 else base_loss.new_tensor(0.0)
        total_loss = base_loss + self.alpha_vdpo * vdpo_loss

        metrics = {
            "loss_dpo": base_loss.item(),
            "loss_vgfdpo": torch.stack(vg_losses).mean().item() if vg_losses else 0.0,
            "loss_format": torch.stack(fmt_losses).mean().item() if fmt_losses else 0.0,
            "loss_vdpo": vdpo_loss.item(),
            "loss_total": total_loss.item(),
            "parse_success_rate": (float(parse_ok_count) / float(bs)) if self.use_vgfdpo else 1.0,
            "s_value_mean": float(sum(s_values) / len(s_values)) if s_values else 0.0,
            "s_sem_mean": float(sum(s_values) / len(s_values)) if s_values and self.grounding_mode == "semantic" else 0.0,
            "s_sp_mean": float(sum(s_values) / len(s_values)) if s_values and self.grounding_mode == "spatial" else 0.0,
            "invalid_s_fallback_count": float(invalid_s_count),
            "vdpo_missing_count": float(vdpo_missing),
        }
        if hasattr(self, "log"):
            self.log(metrics)

        if return_outputs:
            return total_loss, {"metrics": metrics}
        return total_loss