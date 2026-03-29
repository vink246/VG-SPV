"""
Compare MLLM cross-attention–based localization (Llama 3.2 Vision / Mllama) with Grounding DINO.

See README for usage (GPU, eager attention, optional DINO checkpoint).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

if "HUGGINGFACE_HUB_CACHE" not in os.environ and "HF_HOME" not in os.environ:
    scratch = os.environ.get("SCRATCH") or os.environ.get("TMPDIR") or os.environ.get("TEMP")
    if scratch:
        hf_cache = Path(scratch) / ".cache" / "huggingface" / "hub"
        hf_cache.mkdir(parents=True, exist_ok=True)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_cache)

from utils import build_messages, parse_dtype
from vlm.inference import _prepare_inputs_mllama


def default_wrapped_user_prompt(user_prompt: str) -> str:
    return (
        "You must answer by separating your output into two segments using these exact XML-style tags:\n\n"
        "<grounding>\n"
        "Describe what you perceive in the image (objects, scene, and anything relevant to safety).\n"
        "</grounding>\n\n"
        "<logic>\n"
        "Evaluate safety: whether the user request is harmful, policy-violating, or benign, "
        "and how it relates to the image.\n"
        "</logic>\n\n"
        f"User request:\n{user_prompt}"
    )


def box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """IoU for two boxes [x1,y1,x2,y2] in pixel coordinates."""
    ax1, ay1, ax2, ay2 = a.astype(np.float64)
    bx1, by1, bx2, by2 = b.astype(np.float64)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def best_factor_grid(n: int, img_h: int, img_w: int) -> tuple[int, int]:
    """Factors n into nh x nw with aspect ratio close to img_h:img_w."""
    if n <= 0:
        return 1, 1
    target = img_h / max(img_w, 1)
    best: tuple[float, int, int] | None = None
    for nh in range(1, int(math.sqrt(n)) + 2):
        if n % nh != 0:
            continue
        nw = n // nh
        ar = nh / max(nw, 1)
        score = abs(math.log(ar + 1e-8) - math.log(target + 1e-8))
        if best is None or score < best[0]:
            best = (score, nh, nw)
    if best is None:
        s = int(math.sqrt(n))
        return s, max(1, n // max(s, 1))
    return best[1], best[2]


def attention_to_2d(
    attn_1d: torch.Tensor,
    vision_config: Any,
    img_h: int,
    img_w: int,
) -> np.ndarray:
    """
    Map length-V vector over vision tokens to a 2D heatmap.
    For Mllama: V = num_tiles * ((image_size//patch_size)**2 + 1); drop one CLS-like token per tile.
    """
    v = int(attn_1d.shape[0])
    x = attn_1d.float().detach().cpu().numpy()
    patch = getattr(vision_config, "image_size", 560) // max(getattr(vision_config, "patch_size", 14), 1)
    npt = patch * patch + 1
    if v % npt == 0 and npt > 1:
        n_tiles = v // npt
        tiles = x.reshape(n_tiles, npt)[:, 1:].reshape(n_tiles, patch, patch)
        if n_tiles == 1:
            return tiles[0]
        nh, nw = best_factor_grid(n_tiles, img_h, img_w)
        canvas_h, canvas_w = nh * patch, nw * patch
        canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        for t in range(n_tiles):
            tr, tc = t // nw, t % nw
            canvas[tr * patch : (tr + 1) * patch, tc * patch : (tc + 1) * patch] = tiles[t]
        return canvas
    nh, nw = best_factor_grid(v, img_h, img_w)
    if nh * nw != v:
        nh, nw = best_factor_grid(v, img_h, img_w)
    return x.reshape(nh, nw)


def sliding_window_argmax_box(
    heat2d: np.ndarray,
    window: int,
    stride: int,
    img_h: int,
    img_w: int,
    top_k: int = 3,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float, float]:
    """
    VisCRA-style: score = sum of heat in each BxB window; pick randomly among top-k patches (arXiv:2505.19684).
    Returns xyxy in pixel coordinates relative to full image.
    """
    h, w = heat2d.shape
    if h < window or w < window:
        window = min(window, h, w)
        stride = max(1, min(stride, window))
    best_scores: list[tuple[float, int, int]] = []
    for y in range(0, h - window + 1, stride):
        for x in range(0, w - window + 1, stride):
            s = float(heat2d[y : y + window, x : x + window].sum())
            best_scores.append((s, y, x))
    best_scores.sort(key=lambda t: t[0], reverse=True)
    if not best_scores:
        return 0.0, 0.0, float(img_w), float(img_h)
    k = min(top_k, len(best_scores))
    pick = 0
    if rng is not None and k > 1:
        pick = int(rng.integers(0, k))
    _, y0, x0 = best_scores[pick]
    # Map patch grid coords to image pixels
    px1 = x0 / max(w, 1) * img_w
    py1 = y0 / max(h, 1) * img_h
    px2 = (x0 + window) / max(w, 1) * img_w
    py2 = (y0 + window) / max(h, 1) * img_h
    return px1, py1, px2, py2


def find_grounding_token_steps(
    new_token_ids: list[int],
    tokenizer: Any,
    open_tag: str = "<grounding>",
    close_tag: str = "</grounding>",
) -> tuple[int, int]:
    """Inclusive step indices (into new_token_ids) whose decode prefix lies in the grounding segment."""
    if not new_token_ids:
        return 0, -1
    open_l = open_tag.lower()
    close_l = close_tag.lower()
    start: int | None = None
    end: int | None = None
    for i in range(len(new_token_ids)):
        text = tokenizer.decode(new_token_ids[: i + 1], skip_special_tokens=False).lower()
        if start is None and open_l in text:
            start = i
        if close_l in text:
            end = i
            break
    if start is None:
        start = 0
    if end is None:
        end = len(new_token_ids) - 1
    if end < start:
        end = start
    return start, end


def collect_mllama_cross_attention_modules(model: Any) -> list[Any]:
    modules: list[Any] = []
    for _name, mod in model.named_modules():
        cls = mod.__class__.__name__
        if cls == "MllamaTextCrossAttention":
            modules.append(mod)
    return modules


def run_mllama_with_grounding_attention(
    model_name: str,
    messages: list[dict[str, Any]],
    *,
    dtype: torch.dtype,
    cache_dir: str | None,
    cross_attn_layer: int,
    max_new_tokens: int,
    do_sample: bool,
    window: int,
    stride: int,
    top_k_patches: int,
    seed: int,
) -> tuple[str, np.ndarray, dict[str, Any]]:
    from PIL import Image
    from transformers import AutoProcessor, MllamaForConditionalGeneration

    if cache_dir is not None:
        os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir

    processor = AutoProcessor.from_pretrained(model_name)
    model = MllamaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()

    cross_mods = collect_mllama_cross_attention_modules(model)
    if not cross_mods:
        raise RuntimeError(
            "No MllamaTextCrossAttention modules found. Is this a Mllama / Llama 3.2 Vision checkpoint?"
        )
    idx = cross_attn_layer if cross_attn_layer >= 0 else len(cross_mods) + cross_attn_layer
    idx = max(0, min(idx, len(cross_mods) - 1))
    target_cross = cross_mods[idx]

    captures: list[torch.Tensor] = []

    def hook(_module: Any, _inp: Any, output: Any) -> None:
        if not isinstance(output, tuple) or len(output) < 2:
            return
        attn_weights = output[1]
        if attn_weights is None:
            return
        # [B, H, Q, V] — keep decode steps only (single query position)
        if attn_weights.dim() != 4 or attn_weights.shape[-2] != 1:
            return
        captures.append(attn_weights.detach())

    handle = target_cross.register_forward_hook(hook)

    inputs = _prepare_inputs_mllama(model, processor, messages)
    input_len = inputs["input_ids"].shape[1]

    # Image size for bbox projection
    content = messages[0]["content"] if messages else []
    img_path = None
    for item in content:
        if isinstance(item, dict) and item.get("type") == "image":
            img_path = item.get("image")
            break
    if isinstance(img_path, str) and not img_path.startswith(("http://", "https://")):
        pil = Image.open(Path(img_path)).convert("RGB")
        img_w, img_h = pil.size
    else:
        img_w, img_h = 1024, 1024

    try:
        with torch.inference_mode():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
            )
    finally:
        handle.remove()

    new_ids = out_ids[0, input_len:].tolist()
    tok = getattr(processor, "tokenizer", processor)
    full_response = tok.batch_decode(
        out_ids[:, input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    n_cap = len(captures)
    n_new = len(new_ids)
    if n_cap != n_new:
        m = min(n_cap, n_new)
        if m == 0:
            meta = {
                "warning": "No cross-attention captures (need eager attn and decode steps with Q=1).",
                "cross_attn_layer_index": idx,
                "num_cross_layers": len(cross_mods),
            }
            return full_response, np.array([0.0, 0.0, float(img_w), float(img_h)]), meta
        captures = captures[:m]
        new_ids = new_ids[:m]

    g0, g1 = find_grounding_token_steps(new_ids, tok)
    stack = torch.stack([c[0].mean(dim=0)[0] for c in captures[g0 : g1 + 1]], dim=0)
    attn_mean = stack.mean(dim=0)
    if attn_mean.numel() == 0:
        attn_mean = torch.ones(1, device=stack.device)

    vision_cfg = model.config.vision_config
    heat2d = attention_to_2d(attn_mean, vision_cfg, img_h, img_w)
    heat2d = heat2d - heat2d.min()
    if heat2d.max() > 0:
        heat2d = heat2d / heat2d.max()

    rng = np.random.default_rng(seed)
    x1, y1, x2, y2 = sliding_window_argmax_box(
        heat2d, window, stride, img_h, img_w, top_k=top_k_patches, rng=rng
    )
    box = np.array([x1, y1, x2, y2], dtype=np.float32)

    meta = {
        "cross_attn_layer_index": idx,
        "num_cross_layers": len(cross_mods),
        "grounding_token_steps": [g0, g1],
        "num_decode_steps_captured": len(captures),
        "vision_tokens": int(attn_mean.shape[0]),
        "heatmap_shape": list(heat2d.shape),
    }
    return full_response, box, meta


def draw_overlay(
    image_path: str,
    mllm_box: np.ndarray | None,
    dino_box: np.ndarray | None,
    dino_label: str,
) -> np.ndarray:
    import cv2

    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    if mllm_box is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in mllm_box]
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(
            bgr,
            "MLLM attn",
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    if dino_box is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in dino_box]
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            bgr,
            f"DINO {dino_label[:40]}",
            (x1, min(y2 + 22, bgr.shape[0] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return bgr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MLLM grounding via cross-attention vs Grounding DINO (VisCRA-style relevance)."
    )
    p.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-11B-Vision-Instruct",
        help="Hugging Face Mllama model id",
    )
    p.add_argument("--image", type=str, required=True, help="Path to input image")
    p.add_argument("--prompt", type=str, required=True, help="User text (may be safety probe)")
    p.add_argument(
        "--no-wrap",
        action="store_true",
        help="Do not wrap --prompt in the <grounding>/<logic> instruction template",
    )
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16"])
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument(
        "--cross-attn-layer",
        type=int,
        default=-1,
        help="Which cross-attention block to read (0..L-1, or -1 for last). VisCRA uses a mid/deep layer on another architecture.",
    )
    p.add_argument("--window", type=int, default=12, help="Sliding window size on heatmap grid (VisCRA default 12)")
    p.add_argument("--stride", type=int, default=4, help="Sliding window stride (VisCRA default 4)")
    p.add_argument(
        "--top-k-patches",
        type=int,
        default=3,
        help="Randomly pick among top-k scoring windows (VisCRA uses top 3)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sample", action="store_true", help="Use sampling for generation")
    p.add_argument("--dino-checkpoint", type=str, default=None, help="Grounding DINO .pth (optional)")
    p.add_argument("--dino-config", type=str, default=None, help="Override DINO config .py path")
    p.add_argument(
        "--dino-text-prompt",
        type=str,
        default="object . person . weapon .",
        help='DINO class prompt, e.g. "knife . blood ."',
    )
    p.add_argument("--dino-box-threshold", type=float, default=0.25)
    p.add_argument("--dino-text-threshold", type=float, default=0.25)
    p.add_argument("--dino-device", type=str, default=None)
    p.add_argument("--no-dino", action="store_true", help="Skip Grounding DINO and IoU")
    p.add_argument(
        "--output-viz",
        type=str,
        required=True,
        help="Where to save visualization (MLLM red, DINO green)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    img_path = Path(args.image)
    if not img_path.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")

    user_text = args.prompt if args.no_wrap else default_wrapped_user_prompt(args.prompt)
    messages = build_messages([str(img_path)], user_text)
    dtype = parse_dtype(args.dtype)

    print("Loading Mllama with attn_implementation=eager (required for cross-attention weights)...")
    response, mllm_box, meta = run_mllama_with_grounding_attention(
        args.model,
        messages,
        dtype=dtype,
        cache_dir=args.cache_dir,
        cross_attn_layer=args.cross_attn_layer,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.sample,
        window=args.window,
        stride=args.stride,
        top_k_patches=args.top_k_patches,
        seed=args.seed,
    )

    print("--- Model response ---")
    print(response)
    print("--- MLLM bounding box (xyxy pixels) ---")
    print(mllm_box.tolist())
    print("--- Attention / grounding meta ---")
    for k, v in meta.items():
        print(f"  {k}: {v}")

    dino_box: np.ndarray | None = None
    dino_label = ""
    iou: float | None = None

    if not args.no_dino and args.dino_checkpoint:
        import importlib.util

        _dino_path = Path(__file__).resolve().parent / "run_grounding_dino.py"
        _spec = importlib.util.spec_from_file_location("_vg_spv_grounding_dino", _dino_path)
        if _spec is None or _spec.loader is None:
            raise RuntimeError(f"Could not load Grounding DINO helper: {_dino_path}")
        _dino = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_dino)
        load_grounding_dino_model = _dino.load_grounding_dino_model
        resolve_config_from_checkpoint = _dino.resolve_config_from_checkpoint
        run_grounding_dino = _dino.run_grounding_dino

        ckpt = args.dino_checkpoint
        if not Path(ckpt).is_file():
            raise FileNotFoundError(f"DINO checkpoint not found: {ckpt}")
        config_path = args.dino_config or resolve_config_from_checkpoint(ckpt)
        dino_dev = args.dino_device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Grounding DINO ({config_path}) on {dino_dev}...")
        dino_model = load_grounding_dino_model(config_path, ckpt, dino_dev)
        boxes_xyxy, scores, labels, _logits = run_grounding_dino(
            dino_model,
            str(img_path),
            args.dino_text_prompt,
            args.dino_box_threshold,
            args.dino_text_threshold,
        )
        if boxes_xyxy.shape[0] == 0:
            print("Grounding DINO: no boxes above threshold.")
        else:
            j = int(np.argmax(scores))
            dino_box = boxes_xyxy[j].astype(np.float32)
            dino_label = labels[j] if j < len(labels) else ""
            iou = box_iou_xyxy(mllm_box, dino_box)
            print("--- DINO best box (xyxy pixels) ---")
            print(dino_box.tolist())
            print(f"--- IoU (MLLM attn box vs best DINO box) ---\n{iou:.4f}")
    elif not args.no_dino:
        print("Skipping DINO: pass --dino-checkpoint or use --no-dino to silence this.")

    import cv2

    out_path = Path(args.output_viz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    viz = draw_overlay(str(img_path), mllm_box, dino_box, dino_label)
    cv2.imwrite(str(out_path), viz)
    print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    main()
