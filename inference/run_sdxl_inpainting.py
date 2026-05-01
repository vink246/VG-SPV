"""
Run SDXL inpainting to create VisCRA-style perturbed images.

This script is meant to prepare contrastive image pairs for V-DPO:
- Image A: original image
- Image B: inpainted image where risky object regions are replaced with benign content

Usage examples:
    # Use a hand-drawn / precomputed mask image directly
    python inference/run_sdxl_inpainting.py \
        --image data/input.jpg \
        --mask-image data/mask.png \
        --prompt "replace the weapon with a brightly colored toy water gun" \
        --output-image outputs/perturbed.png

    # Build mask from GroundingDINO JSON (expects box_xyxy keys)
    python inference/run_sdxl_inpainting.py \
        --image data/input.jpg \
        --boxes-json outputs/dino_boxes.json \
        --prompt "replace the knife with a harmless plastic spoon toy" \
        --output-image outputs/perturbed.png \
        --output-metadata outputs/perturbed.meta.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageChops, ImageDraw, ImageFilter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run SDXL inpainting on a single image.")
    p.add_argument(
        "--model-id",
        type=str,
        default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        help="Hugging Face model id or local path for an SDXL inpainting model.",
    )
    p.add_argument("--image", type=str, required=True, help="Path to source image.")
    p.add_argument(
        "--mask-image",
        type=str,
        default=None,
        help="Optional path to binary/grayscale mask image (white=inpaint, black=preserve).",
    )
    p.add_argument(
        "--boxes-json",
        type=str,
        default=None,
        help="Optional path to detections JSON (e.g., from run_grounding_dino.py).",
    )
    p.add_argument(
        "--box",
        type=str,
        action="append",
        default=[],
        help='Manual box in "x1,y1,x2,y2" format. Can be repeated.',
    )
    p.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Inpainting instruction prompt.",
    )
    p.add_argument("--negative-prompt", type=str, default=None, help="Optional negative prompt.")
    p.add_argument("--output-image", type=str, required=True, help="Path for generated image.")
    p.add_argument(
        "--output-mask",
        type=str,
        default=None,
        help="Optional path to save the final effective mask.",
    )
    p.add_argument(
        "--output-metadata",
        type=str,
        default=None,
        help="Optional path to save run metadata JSON.",
    )
    p.add_argument("--num-inference-steps", type=int, default=30, help="Diffusion denoising steps.")
    p.add_argument("--guidance-scale", type=float, default=8.0, help="Classifier-free guidance scale.")
    p.add_argument(
        "--strength",
        type=float,
        default=0.99,
        help="Inpainting strength in [0, 1). Values close to 1 edit masked area strongly.",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype for pipeline weights.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help='Torch device, e.g. "cuda", "cuda:0", or "cpu".',
    )
    p.add_argument("--seed", type=int, default=None, help="Optional seed for deterministic generation.")
    p.add_argument(
        "--dilate-px",
        type=int,
        default=0,
        help="Optional dilation (pixels) applied to generated box mask.",
    )
    p.add_argument(
        "--feather-px",
        type=int,
        default=0,
        help="Optional gaussian blur radius for smoother mask edges.",
    )
    p.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional HF cache directory override.",
    )
    return p.parse_args()


def parse_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def parse_manual_box(box_str: str) -> Tuple[float, float, float, float]:
    parts = [x.strip() for x in box_str.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Invalid --box value: {box_str!r}. Expected x1,y1,x2,y2.")
    vals = [float(x) for x in parts]
    x1, y1, x2, y2 = vals
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid box with non-positive area: {box_str!r}.")
    return x1, y1, x2, y2


def _extract_box(item: Dict[str, Any]) -> Tuple[float, float, float, float] | None:
    if "box_xyxy" in item and isinstance(item["box_xyxy"], Sequence) and len(item["box_xyxy"]) == 4:
        x1, y1, x2, y2 = [float(v) for v in item["box_xyxy"]]
        return x1, y1, x2, y2
    if "bbox" in item and isinstance(item["bbox"], Sequence) and len(item["bbox"]) == 4:
        x1, y1, x2, y2 = [float(v) for v in item["bbox"]]
        return x1, y1, x2, y2
    return None


def load_boxes_from_json(path: str) -> List[Tuple[float, float, float, float]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in boxes JSON: {path}")
    boxes: List[Tuple[float, float, float, float]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        box = _extract_box(item)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def clip_box(
    box: Tuple[float, float, float, float],
    width: int,
    height: int,
) -> Tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = box
    xi1 = max(0, min(width - 1, int(round(x1))))
    yi1 = max(0, min(height - 1, int(round(y1))))
    xi2 = max(0, min(width, int(round(x2))))
    yi2 = max(0, min(height, int(round(y2))))
    if xi2 <= xi1 or yi2 <= yi1:
        return None
    return xi1, yi1, xi2, yi2


def build_mask_from_boxes(
    width: int,
    height: int,
    boxes: Sequence[Tuple[float, float, float, float]],
    dilate_px: int = 0,
    feather_px: int = 0,
) -> Image.Image:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for box in boxes:
        clipped = clip_box(box, width=width, height=height)
        if clipped is None:
            continue
        x1, y1, x2, y2 = clipped
        draw.rectangle([x1, y1, x2, y2], fill=255)

    if dilate_px > 0:
        # MaxFilter expands white regions; kernel size should be odd.
        kernel = max(3, 2 * dilate_px + 1)
        mask = mask.filter(ImageFilter.MaxFilter(size=kernel))
    if feather_px > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_px))
    return mask


def combine_masks(mask_a: Image.Image | None, mask_b: Image.Image | None) -> Image.Image | None:
    if mask_a is None and mask_b is None:
        return None
    if mask_a is None:
        return mask_b
    if mask_b is None:
        return mask_a
    return ImageChops.lighter(mask_a, mask_b)


def main() -> None:
    args = parse_args()

    if not Path(args.image).is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if args.mask_image is None and args.boxes_json is None and len(args.box) == 0:
        raise ValueError("Provide at least one mask source: --mask-image, --boxes-json, or --box.")
    if not (0.0 <= args.strength < 1.0):
        raise ValueError("--strength must be in [0.0, 1.0).")
    if args.cache_dir is not None:
        os.environ["HUGGINGFACE_HUB_CACHE"] = args.cache_dir

    source_image = Image.open(args.image).convert("RGB")
    width, height = source_image.size

    file_mask: Image.Image | None = None
    if args.mask_image is not None:
        if not Path(args.mask_image).is_file():
            raise FileNotFoundError(f"Mask image not found: {args.mask_image}")
        file_mask = Image.open(args.mask_image).convert("L").resize((width, height), Image.NEAREST)

    boxes: List[Tuple[float, float, float, float]] = []
    if args.boxes_json is not None:
        if not Path(args.boxes_json).is_file():
            raise FileNotFoundError(f"Boxes JSON not found: {args.boxes_json}")
        boxes.extend(load_boxes_from_json(args.boxes_json))
    if args.box:
        boxes.extend(parse_manual_box(b) for b in args.box)

    box_mask: Image.Image | None = None
    if boxes:
        box_mask = build_mask_from_boxes(
            width=width,
            height=height,
            boxes=boxes,
            dilate_px=args.dilate_px,
            feather_px=args.feather_px,
        )

    mask = combine_masks(file_mask, box_mask)
    if mask is None:
        raise ValueError("Mask construction failed; no valid mask source produced pixels.")

    from diffusers import AutoPipelineForInpainting

    torch_dtype = parse_dtype(args.dtype)
    print(f"Loading inpainting model: {args.model_id}")
    pipe = AutoPipelineForInpainting.from_pretrained(args.model_id, torch_dtype=torch_dtype)
    pipe = pipe.to(args.device)

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=args.device).manual_seed(args.seed)

    print("Running SDXL inpainting...")
    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=source_image,
        mask_image=mask,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        strength=args.strength,
        generator=generator,
    ).images[0]

    out_img = Path(args.output_image)
    out_img.parent.mkdir(parents=True, exist_ok=True)
    result.save(out_img)
    print(f"Saved inpainted image to {out_img}")

    if args.output_mask:
        out_mask = Path(args.output_mask)
        out_mask.parent.mkdir(parents=True, exist_ok=True)
        mask.save(out_mask)
        print(f"Saved effective mask to {out_mask}")

    if args.output_metadata:
        out_meta = Path(args.output_metadata)
        out_meta.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "model_id": args.model_id,
            "image": str(Path(args.image).resolve()),
            "mask_image": str(Path(args.mask_image).resolve()) if args.mask_image else None,
            "boxes_json": str(Path(args.boxes_json).resolve()) if args.boxes_json else None,
            "manual_boxes_xyxy": [list(b) for b in (parse_manual_box(v) for v in args.box)],
            "num_boxes_total": len(boxes),
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "strength": args.strength,
            "dtype": args.dtype,
            "device": args.device,
            "seed": args.seed,
            "dilate_px": args.dilate_px,
            "feather_px": args.feather_px,
            "output_image": str(out_img.resolve()),
            "output_mask": str(Path(args.output_mask).resolve()) if args.output_mask else None,
        }
        out_meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Saved metadata to {out_meta}")


if __name__ == "__main__":
    main()
